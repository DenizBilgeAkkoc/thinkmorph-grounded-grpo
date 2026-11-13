"""Instrumentation around the unchanged native interleaving loop.

Only generate_image's integration steps are replaced. Native conditional
_forward_flow and native text sampling remain in use. CFG is disabled for RL.
"""
from copy import deepcopy
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from types import MethodType
import inspect
import json
import time
from unittest.mock import patch

import torch
from .sde import schedule, sde_step, ode_step
from .guidance import require_no_cfg


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temp.replace(path)


class NativeFlowBridge:
    def __init__(self, model, windows, noise_level=1.0, window_size=3):
        self.model, self.windows, self.noise_level = model, windows, noise_level
        self.window_size = window_size
        self.original = model.generate_image
        self.signature = inspect.signature(self.original)
        self.flow_keys = set(inspect.signature(model._forward_flow).parameters)
        self.records = []

    def __enter__(self):
        bridge = self
        def generate_image(model, *args, **kwargs):
            return bridge.generate(*args, **kwargs)
        self.model.generate_image = MethodType(generate_image, self.model)
        return self

    def __exit__(self, *exc):
        self.model.generate_image = self.original

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        p = bound.arguments
        require_no_cfg(p, "RL rollout")
        if p["enable_taylorseer"]:
            raise ValueError("TaylorSeer is incompatible with exact transition replay")
        self.model.language_model.model.enable_taylorseer = False
        x = p["packed_init_noises"]
        ts = schedule(p["num_timesteps"], p["timestep_shift"], x.device)
        image_index = len(self.records)
        begin, width = self.windows[image_index], self.window_size
        if begin < 0 or width < 1 or begin + width > len(ts) - 1:
            raise ValueError("Invalid stochastic transition window")
        record = {"window_start": begin, "window_size": width,
                  "guidance": {"cfg_text_scale": p["cfg_text_scale"], "cfg_img_scale": p["cfg_img_scale"]},
                  "schedule": ts.cpu(), "noise_level": self.noise_level,
                  "initial_noise": x.detach().cpu().clone(), "transitions": []}
        flow = {k: v for k, v in p.items() if k in self.flow_keys}
        # Native inferencer still constructs its CFG contexts, but the RL velocity
        # uses only the full conditional branch; do not pass unused CFG caches.
        for key in list(flow):
            if (key.startswith("cfg_text_") or key.startswith("cfg_img_")) and not key.endswith("scale"):
                del flow[key]
        for i, t in enumerate(ts[:-1]):
            active_cfg = p["cfg_interval"][0] < t <= p["cfg_interval"][1]
            flow["cfg_text_scale"] = p["cfg_text_scale"] if active_cfg else 1.0
            flow["cfg_img_scale"] = p["cfg_img_scale"] if active_cfg else 1.0
            velocity = self.model._forward_flow(x_t=x,
                timestep=t.expand(x.shape[0]), **flow).to(x.device)
            if begin <= i < begin + width:
                before = x.detach().cpu().clone()
                x, logp, _, _ = sde_step(velocity, t, ts[i + 1], x,
                    sigma_max=ts[1], noise_level=self.noise_level)
                record["transitions"].append({"index": i, "x": before,
                    "next_x": x.detach().cpu().clone(), "old_logp": logp.detach().cpu(),
                    "velocity": velocity.detach().float().cpu()})
            else:
                x = ode_step(velocity, t, ts[i + 1], x)
        self.records.append(record)
        return x.split((p["packed_seqlens"] - 2).tolist())


def recorded_inferencer(base):
    class RecordedInferencer(base):
        def begin_recording(self, folder, bridge):
            self.folder = Path(folder)
            self.folder.mkdir(parents=True, exist_ok=True)
            self.events, self.bridge = [], bridge

        def init_gen_context(self):
            ctx = super().init_gen_context()
            ctx["events"] = []
            return ctx

        def save_image(self, image):
            key = sha256(image.mode.encode() + str(image.size).encode() + image.tobytes()).hexdigest()[:20]
            name = f"asset_{key}.png"
            if not (self.folder / name).exists():
                image.save(self.folder / name)
            return name

        def update_context_text(self, text, gen_context):
            ctx = super().update_context_text(text, gen_context)
            ctx["events"].append({"kind": "text", "text": text})
            return ctx

        def update_context_image(self, image, gen_context, vae=True, vit=True):
            # Native VAE encoding also samples a posterior. Save that exact latent,
            # so training never silently changes the conditioning during replay.
            latents = []
            encode = self.vae_model.encode
            def record_encode(*a, **kw):
                value = encode(*a, **kw)
                latents.append(value.detach().cpu().clone())
                return value
            with patch.object(self.vae_model, "encode", record_encode):
                ctx = super().update_context_image(image, gen_context, vae, vit)
            event = {"kind": "image", "file": self.save_image(image), "vae": vae, "vit": vit}
            if vae:
                if len(latents) != 1:
                    raise RuntimeError("Expected one native VAE encoding per image update")
                name = f"conditioning_{len(list(self.folder.glob('conditioning_*.pt'))):03d}.pt"
                torch.save(latents[0], self.folder / name)
                event["vae_latent_file"] = name
            ctx["events"].append(event)
            return ctx

        def gen_text(self, gen_context, **kwargs):
            # Observe native torch.multinomial without changing RNG consumption or
            # token generation. Each process runs one rollout at a time.
            samples, sampling_logps = [], []
            original_multinomial = torch.multinomial
            def record_sample(*args, **kw):
                out = original_multinomial(*args, **kw)
                samples.append(out.detach().cpu().flatten().tolist())
                probabilities = args[0] if args else kw["input"]
                probabilities = probabilities.detach().double()
                probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
                sampling_logps.append(float(probabilities.gather(-1, out).log().sum().cpu()))
                return out
            started = time.time()
            with patch.object(torch, "multinomial", record_sample):
                text = super().gen_text(gen_context, **kwargs)
            if not kwargs.get("do_sample", True) or not samples:
                raise ValueError("This GRPO recorder requires stochastic native text sampling")
            tokens = [s[0] for s in samples]
            eos = self.new_token_ids["eos_token_id"]
            max_length = inspect.signature(super().gen_text).bind(gen_context, **kwargs)
            max_length.apply_defaults()
            capped = len(tokens) >= max_length.arguments["max_length"]
            ended = tokens[-1] == eos and not capped
            # At max length native generate_text discards its last sampled token.
            # Marginalize that unused draw; include EOS when it caused termination.
            targets = tokens if ended else tokens[:-1]
            inputs = [self.new_token_ids["bos_token_id"]] + targets[:-1] if targets else []
            event = {"kind": "text", "text": text, "prefix": deepcopy(gen_context["events"]),
                "input_ids": inputs, "target_ids": targets, "temperature": kwargs.get("temperature", 1.0),
                "stop": "eos" if ended else "token_limit", "sampled_tokens": len(tokens),
                "sampling_token_logps": sampling_logps if ended else sampling_logps[:-1],
                "sampling_logp": sum(sampling_logps if ended else sampling_logps[:-1]),
                "seconds": time.time() - started}
            self.events.append(event)
            atomic_json(self.folder / "trace.json", self.events)
            return text

        def gen_image(self, image_shape, gen_context, **kwargs):
            started = time.time()
            contexts = {"conditional": deepcopy(gen_context["events"]),
                "cfg_text": deepcopy(kwargs["cfg_text_precontext"]["events"]),
                "cfg_image": deepcopy(kwargs["cfg_img_precontext"]["events"])}
            image = super().gen_image(image_shape, gen_context, **kwargs)
            index = len(self.bridge.records) - 1
            state_file = f"flow_{index:02d}.pt"
            torch.save(self.bridge.records[-1], self.folder / state_file)
            settings = {k: v for k, v in kwargs.items() if not k.endswith("precontext")}
            self.events.append({"kind": "image", "file": self.save_image(image),
                "flow_file": state_file, "contexts": contexts, "image_shape": list(image_shape),
                "settings": settings, "seconds": time.time() - started})
            atomic_json(self.folder / "trace.json", self.events)
            return image
    return RecordedInferencer
