"""Minimal end-to-end native rollout -> Qwen judge -> custom GRPO update CLI.

Example: python -m thinkmorph_grpo.train --preview-config ... --recipe ...
 --manifest ... --output ... --judge-url http://HOST:PORT/v1 --max-groups 1
No optimizer update happens unless this training command is explicitly run.
"""
import argparse
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import math
import os
import time
import traceback
import torch
from PIL import Image
from .runtime import load_native
from .bridge import NativeFlowBridge, recorded_inferencer, atomic_json
from .sde import shared_windows
from .replay import attach_lora, TEXT_LIKELIHOOD_BACKEND
from .judge_v2 import GroundedQwenJudge, DirectObservationQwenJudge, JudgeAssessmentError
from .randomness import seed_everything, capture_rng_state, restore_rng_state
from .objective import GRPOConfig
from .trainer import NativeGRPOTrainer, judge_group, adapter_digest
from .guidance import require_no_cfg


PARALLEL_TEXT_LIKELIHOOD_BACKEND = "parallel_full_sequence_consistent_old_new"


def source_hashes():
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))}


@contextmanager
def record_run_failure(root, summary):
    """Leave a truthful terminal status if a rollout, judgment, or update fails."""
    try:
        yield
    except BaseException as error:
        summary.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                       finished_at_unix=time.time(),
                       error={"type": type(error).__name__, "message": str(error)})
        atomic_json(Path(root) / "summary.json", summary)
        raise


def validate_judge_recipe(recipe, judge, *, text_likelihood_backend=TEXT_LIKELIHOOD_BACKEND):
    if recipe.get("judge_version") != judge.version or recipe.get("terminal_policy_version") != judge.terminal_version:
        raise ValueError("Recipe and active judge versions differ")
    protocol = recipe.get("judge_protocol_revision")
    if protocol is not None and protocol != getattr(judge, "protocol_revision", None):
        raise ValueError("Recipe and active judge protocol revisions differ")
    backend = recipe.get("text_likelihood_backend")
    if text_likelihood_backend not in (TEXT_LIKELIHOOD_BACKEND, PARALLEL_TEXT_LIKELIHOOD_BACKEND):
        raise ValueError("Unsupported active text likelihood backend")
    if backend is not None and backend != text_likelihood_backend:
        raise ValueError("Recipe and active text likelihood backends differ")
    model = getattr(judge, "model_id", "Qwen/Qwen3-VL-32B-Instruct")
    if recipe.get("judge_model") is not None and recipe["judge_model"] != model:
        raise ValueError("Recipe and active judge models differ; use an explicit provider recipe")
    provider = getattr(judge, "provider", "local_qwen")
    if recipe.get("judge_provider") is not None and recipe["judge_provider"] != provider:
        raise ValueError("Recipe and active judge providers differ")


def make_judge(args, *, audit_directory=None):
    """Provider selection is explicit; credential values never enter config."""
    if args.judge_provider == "gemini":
        if args.judge_observation_mode != "direct":
            raise ValueError("Gemini training currently requires direct observation mode")
        from .gemini_judge import GeminiJudge
        return GeminiJudge(api_key_file=args.gemini_api_key_file,
            model=args.gemini_model, thinking=args.gemini_thinking,
            timeout=args.judge_timeout, audit_directory=audit_directory)
    if not args.judge_url:
        raise ValueError("--judge-url is required for the local Qwen provider")
    judge_class = DirectObservationQwenJudge if args.judge_observation_mode == "direct" else GroundedQwenJudge
    return judge_class(args.judge_url, os.environ.get("THINKMORPH_JUDGE_API_KEY", ""),
                       timeout=args.judge_timeout, audit_directory=audit_directory)


def judge_identity(judge):
    if hasattr(judge, "identity"):
        return judge.identity()
    return {"provider": "local_qwen", "model": "Qwen/Qwen3-VL-32B-Instruct",
            "version": getattr(judge, "version", "legacy_v0"),
            "protocol_revision": getattr(judge, "protocol_revision", None),
            "terminal_version": getattr(judge, "terminal_version", "legacy_v0"),
            "base_url": getattr(judge, "base_url", None)}


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return value


def _state_digest(value):
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(f"tensor:{tuple(tensor.shape)}:{tensor.dtype}:".encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict:")
            for key in sorted(item, key=lambda key: (str(type(key)), str(key))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(f"{type(item).__name__}:{len(item)}:".encode())
            for element in item:
                visit(element)
        else:
            digest.update(f"{type(item).__name__}:{item!r}:".encode())
    visit(value)
    return digest.hexdigest()


def save_checkpoint(path, model, optimizer, *, group, optimizer_updates, configuration):
    """Write one completed group atomically, including optimizer and random streams."""
    path = Path(path)
    state = {"schema_version": 1, "group": group, "optimizer_updates": optimizer_updates,
        "adapters": {name: value.detach().cpu().clone() for name, value in model.named_parameters()
                     if "lora_" in name},
        "policy_digest": adapter_digest(model), "optimizer": _cpu_copy(optimizer.state_dict()),
        "configuration": configuration, "rng": capture_rng_state()}
    state["optimizer_digest"] = _state_digest(state["optimizer"])
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    return state["policy_digest"]


def restore_checkpoint(path, model, optimizer, *, configuration):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema")
    if state["configuration"] != configuration:
        raise ValueError("Checkpoint configuration differs; explicit migration is required")
    if _state_digest(state["optimizer"]) != state["optimizer_digest"]:
        raise ValueError("Checkpoint optimizer digest mismatch")
    adapters = {name: value for name, value in model.named_parameters() if "lora_" in name}
    if set(adapters) != set(state["adapters"]):
        raise ValueError("Checkpoint adapter names differ")
    for name, value in adapters.items():
        saved = state["adapters"][name]
        if value.shape != saved.shape or value.dtype != saved.dtype or not torch.isfinite(saved).all():
            raise ValueError("Checkpoint adapter shape, dtype, or finiteness mismatch: " + name)
    with torch.no_grad():
        for name, value in adapters.items():
            value.copy_(state["adapters"][name].to(value.device))
    if adapter_digest(model) != state["policy_digest"]:
        raise ValueError("Checkpoint adapter digest mismatch")
    optimizer.load_state_dict(state["optimizer"])
    if _state_digest(optimizer.state_dict()) != state["optimizer_digest"]:
        raise ValueError("Restored optimizer state differs from checkpoint")
    restore_rng_state(state["rng"])
    return state


def positive_timeout(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("Judge timeout must be finite and positive")
    return value


def make_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--preview-config", required=True)
    p.add_argument("--recipe", required=True)
    p.add_argument("--manifest", required=True, help="JSON array with question, answer, task, problem_image.path")
    p.add_argument("--output", required=True)
    p.add_argument("--judge-url", help="Local Qwen OpenAI-compatible endpoint")
    p.add_argument("--judge-provider", choices=("qwen", "gemini"), default="qwen")
    p.add_argument("--gemini-model", default="gemini-3.8-flash")
    p.add_argument("--gemini-thinking", choices=("low", "medium", "high"), default="low")
    p.add_argument("--gemini-api-key-file", help="Private raw-key file; otherwise use GEMINI_API_KEY")
    p.add_argument("--judge-timeout", type=positive_timeout, default=180.0,
                   help="Finite positive timeout in seconds for each judge request (default: 180)")
    p.add_argument("--judge-workers", type=int, default=1,
                   help="Concurrent independent rollout judgments per policy worker")
    p.add_argument("--judge-observation-mode", choices=("separate", "direct"), default="separate",
                   help="Separate V9 claim extraction or one direct V10 observation assessment")
    p.add_argument("--audit-judge-rpc", action="store_true",
                   help="Record every judge RPC, response, error and token usage under output/judge_rpc")
    p.add_argument("--skip-unscorable-groups", action="store_true",
                   help="Skip a whole N=8 group after an exhausted judge assessment; never invent rewards")
    p.add_argument("--max-groups", type=int, default=1)
    p.add_argument("--max-updates", type=int, default=None, help="Total updates, including resumed updates")
    p.add_argument("--resume", help="Completed-group checkpoint; output must be a new directory")
    p.add_argument("--verify-checkpoint-reload", action="store_true",
                   help="Reload every saved checkpoint before sampling the next group")
    return p


def parse_args(argv=None):
    return make_parser().parse_args(argv)


def main():
    args = parse_args()
    cfg, recipe = json.loads(Path(args.preview_config).read_text()), json.loads(Path(args.recipe).read_text())
    examples = json.loads(Path(args.manifest).read_text())
    if not 0 < args.max_groups <= len(examples):
        raise ValueError("max-groups must select a nonempty prefix of the manifest")
    max_updates = args.max_groups if args.max_updates is None else args.max_updates
    if max_updates < 1:
        raise ValueError("max-updates must be positive")
    if args.judge_workers < 1:
        raise ValueError("judge-workers must be positive")
    require_no_cfg(cfg["native_settings"], "Training configuration")
    if cfg["group_size"] != 8 or cfg["sde_window_size"] != 3:
        raise ValueError("This first integration supports the agreed N=8/window=3 configuration")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    configuration = {"rollout": cfg, "recipe": recipe, "examples": examples,
                     "source_hashes": source_hashes(), "judge_workers": args.judge_workers,
                     "judge_timeout": args.judge_timeout,
                     "judge_observation_mode": args.judge_observation_mode,
                     "judge_provider": args.judge_provider,
                     "skip_unscorable_groups": args.skip_unscorable_groups}
    atomic_json(root / "resolved_config.json", {**configuration, "invocation": vars(args)})
    start_group, optimizer_updates = 0, 0
    summary = {"started_at_unix": time.time(), "groups": [], "optimizer_updates": optimizer_updates,
               "resume": args.resume, "status": "running", "phase": "initializing",
               "target_optimizer_updates": max_updates, "max_groups": args.max_groups,
               "groups_attempted_this_run": 0, "unscorable_groups_skipped": 0,
               "skip_unscorable_groups": args.skip_unscorable_groups, "target_reached": False}
    atomic_json(root / "summary.json", summary)
    with record_run_failure(root, summary):
        seed_everything(cfg["seed"])
        torch.set_num_threads(6)
        summary["phase"] = "verifying_judge"
        atomic_json(root / "summary.json", summary)
        judge = make_judge(args, audit_directory=root / "judge_rpc" if args.audit_judge_rpc else None)
        validate_judge_recipe(recipe, judge)
        judge.verify_model()
        configuration["judge_identity"] = judge_identity(judge)
        atomic_json(root / "resolved_config.json", {**configuration, "invocation": vars(args)})
        summary["phase"] = "loading_model"
        atomic_json(root / "summary.json", summary)
        loaded = load_native(cfg["source"], cfg["checkpoint"])
        summary["phase"] = "attaching_adapters"
        atomic_json(root / "summary.json", summary)
        groups = attach_lora(loaded[0], **recipe["lora"])
        summary["phase"] = "initializing_optimizer"
        atomic_json(root / "summary.json", summary)
        opt = recipe["optimizer"]
        optimizer = torch.optim.AdamW([
            {"params":groups["image"],"lr":opt["image_learning_rate"]},
            {"params":groups["text"],"lr":opt["text_learning_rate"]}],weight_decay=opt["weight_decay"])
        recorder = recorded_inferencer(loaded[-1])(*loaded[:-1])
        replay = loaded[-1](*loaded[:-1])
        trainer = NativeGRPOTrainer(replay, optimizer, GRPOConfig(**recipe["grpo"]))
        if args.resume:
            summary["phase"] = "restoring_checkpoint"
            atomic_json(root / "summary.json", summary)
            state = restore_checkpoint(args.resume, loaded[0], optimizer, configuration=configuration)
            start_group, optimizer_updates = state["group"] + 1, state["optimizer_updates"]
            summary["optimizer_updates"] = optimizer_updates
            atomic_json(root / "summary.json", summary)
            del state
        for gi in range(start_group, args.max_groups):
            if optimizer_updates >= max_updates:
                break
            example = examples[gi]
            summary.update(active_group=gi, phase="sampling")
            summary["groups_attempted_this_run"] += 1
            atomic_json(root / "summary.json", summary)
            group_folder = root / f"group_{gi:06d}"
            group_folder.mkdir(parents=True, exist_ok=False)
            policy_digest = adapter_digest(loaded[0])
            started = time.monotonic()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            atomic_json(group_folder / "group.json", {"example_id": example.get("example_id", example["task"]),
                "group": gi, "policy_digest": policy_digest, "prior_optimizer_updates": optimizer_updates})
            windows = shared_windows(cfg["seed"] + gi, cfg["native_settings"]["max_rounds"],
                cfg["native_settings"]["num_timesteps"], cfg["sde_window_size"])
            folders = []
            for n in range(8):
                seed = cfg["seed"] + gi * 1000 + n
                seed_everything(seed)
                folder = group_folder / f"rollout_{n:02d}"
                with NativeFlowBridge(loaded[0], windows, cfg["noise_level"], cfg["sde_window_size"]) as bridge:
                    recorder.begin_recording(folder, bridge)
                    recorder.interleave_inference([Image.open(example["problem_image"]["path"]).convert("RGB"),
                        example["question"]], **cfg["native_settings"])
                    atomic_json(folder / "sampling.json", {"seed":seed,"windows":windows,"group":gi,
                        "policy_digest": policy_digest, "prior_optimizer_updates": optimizer_updates,
                        "base_checkpoint": cfg["checkpoint"], "native_revision": cfg["native_revision"],
                        "rollout_configuration_digest": _state_digest(cfg),
                        "example_id": example.get("example_id", example["task"])})
                folders.append(folder)
                print(json.dumps({"event": "rollout_complete", "group": gi, "rollout": n,
                                  "seed": seed, "seconds": time.monotonic()-started}), flush=True)
            recorder.events=[]
            sampling_seconds = time.monotonic()-started
            summary["phase"] = "judging"
            atomic_json(root / "summary.json", summary)
            try:
                judgments = judge_group(example, folders, judge, workers=args.judge_workers)
            except JudgeAssessmentError as error:
                if not args.skip_unscorable_groups:
                    raise
                # Keep the original N=8 sampling/normalization contract. A missing
                # judgment is not a zero reward, and seven survivors are not N=8.
                skipped = {"group": gi, "example_id": example.get("example_id", example["task"]),
                    "optimized": False, "skip_reason": "unscorable_judge_group",
                    "optimizer_updates": optimizer_updates, "policy_digest": policy_digest,
                    "group_size": len(folders), "sampling_seconds": sampling_seconds,
                    "judge_seconds": time.monotonic()-started-sampling_seconds,
                    "error": {"type": type(error).__name__, "message": str(error)},
                    "traceback": traceback.format_exc(),
                    "completed_judgment_folders": [str(folder) for folder in folders
                                                   if (folder / "judge.json").is_file()]}
                atomic_json(group_folder / "group_skip.json", skipped)
                summary["unscorable_groups_skipped"] += 1
                summary["groups"].append({key: skipped[key] for key in
                    ("group", "example_id", "optimized", "skip_reason", "error")})
                summary["phase"] = "group_skipped"
                atomic_json(root / "summary.json", summary)
                print(json.dumps({"event": "group_skipped", "group": gi,
                    "example_id": skipped["example_id"], "reason": skipped["skip_reason"],
                    "error": skipped["error"], "optimizer_updates": optimizer_updates,
                    "unscorable_groups_skipped": summary["unscorable_groups_skipped"]}), flush=True)
                continue
            judge_seconds = time.monotonic()-started-sampling_seconds
            summary["phase"] = "updating"
            atomic_json(root / "summary.json", summary)
            metrics = trainer.update(folders, judgments)
            optimizer_updates += int(metrics["optimized"])
            summary.update(optimizer_updates=optimizer_updates, phase="checkpointing")
            atomic_json(root / "summary.json", summary)
            metrics.update(sampling_seconds=sampling_seconds, judge_seconds=judge_seconds,
                optimizer_updates=optimizer_updates,
                peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_available() else 0,
                peak_cuda_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if torch.cuda.is_available() else 0)
            atomic_json(group_folder / "update.json", metrics)
            checkpoint = group_folder / "checkpoint.pt"
            digest = save_checkpoint(checkpoint, loaded[0], optimizer, group=gi,
                optimizer_updates=optimizer_updates, configuration=configuration)
            if args.verify_checkpoint_reload:
                state = restore_checkpoint(checkpoint, loaded[0], optimizer, configuration=configuration)
                assert state["policy_digest"] == digest
                del state
                atomic_json(group_folder / "checkpoint_validation.json", {"reload_passed": True,
                    "policy_digest": digest, "optimizer_state_exact": True,
                    "next_group_samples_from_reloaded_policy": (
                        gi+1 < args.max_groups and optimizer_updates < max_updates)})
            summary["groups"].append({"group": gi, "example_id": example.get("example_id", example["task"]),
                "optimized": metrics["optimized"], "skip_reason": metrics.get("skip_reason"),
                "policy_digest_before": policy_digest, "policy_digest_after": digest})
            summary["optimizer_updates"] = optimizer_updates
            atomic_json(root / "summary.json", summary)
            print(json.dumps({"event": "group_complete", "group":gi,"optimized":metrics["optimized"],
                "gradient_norm":metrics["gradient_norm_before_clip"], "optimizer_updates": optimizer_updates}),flush=True)
    summary.update(status="completed", phase="finished", finished_at_unix=time.time(),
                   target_reached=optimizer_updates >= max_updates,
                   stop_reason="target_updates_reached" if optimizer_updates >= max_updates
                               else "max_groups_exhausted")
    atomic_json(root / "summary.json", summary)


if __name__ == "__main__":
    main()
