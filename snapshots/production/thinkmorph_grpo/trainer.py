"""Single-GPU, serial replay baseline for groups of eight native rollouts.

This deliberately exposes a small callable update rather than routing native traces
through simulation-specific task logic. Distributed optimization can be added later.
"""
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import re
import hashlib
import math
import time
import torch
from .judge import StepEvidence, reward_components
from .judge_v2 import JudgeAssessmentError
from .objective import group_advantages, group_returns, ppo_loss, velocity_regularizer
from .replay import (rebuild_context, text_logprob, flow_inputs, flow_logprob, reference_adapters,
                     reference_flow_inputs, replay_context_cache, TEXT_LIKELIHOOD_BACKEND, TEXT_REPLAY_MEMORY_STRATEGY)
from .bridge import atomic_json

SAMPLING_REPLAY_TOLERANCE = 1e-5


def saved_text_logprob(event, *, required):
    """Validate behavior-policy likelihood provenance before any backward pass."""
    if "sampling_logp" not in event:
        if required:
            raise ValueError("Optimization requires recorded text sampling_logp")
        return None
    value = float(event["sampling_logp"])
    if not math.isfinite(value):
        raise ValueError("Nonfinite recorded text sampling_logp")
    tokens = event.get("sampling_token_logps")
    if tokens is None:
        if required:
            raise ValueError("Optimization requires recorded per-token sampling logprobs")
    elif (len(tokens) != len(event["target_ids"]) or
          any(not math.isfinite(float(token)) for token in tokens) or
          not math.isclose(sum(tokens), value, rel_tol=0., abs_tol=1e-8)):
        raise ValueError("Inconsistent recorded text sampling logprobs")
    return torch.tensor(value, dtype=torch.float64)


def saved_logprob(value, *, required, kind):
    """Validate a saved scalar denominator without any model work."""
    if value is None:
        if required:
            raise ValueError(f"Optimization requires recorded {kind} sampling logprob")
        return None
    saved = torch.as_tensor(value).detach().cpu()
    if saved.numel() != 1 or not torch.isfinite(saved).all():
        raise ValueError(f"Invalid recorded {kind} sampling logprob")
    return saved.reshape(())


def behavior_logprob(replayed, saved, *, required, kind):
    """Use saved behavior likelihood, with a fail-closed on-policy fidelity gate."""
    replayed = replayed.detach().cpu()
    saved = saved_logprob(saved, required=required, kind=kind)
    if saved is None:
        return replayed, None, "replay_diagnostic_only"
    delta = abs(float(replayed) - float(saved))
    if not math.isfinite(delta) or (required and delta > SAMPLING_REPLAY_TOLERANCE):
        raise ValueError(f"{kind} sampling/replay likelihood mismatch: {delta}")
    return saved, delta, "recorded_sampling"


def adapter_digest(model):
    """Identity of the actual policy adapters, independent of file timestamps."""
    digest = hashlib.sha256()
    count = 0
    for name, value in model.named_parameters():
        if "lora_" in name:
            tensor = value.detach().cpu().contiguous()
            digest.update(f"{name}:{tuple(tensor.shape)}:{tensor.dtype}".encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
            count += 1
    if not count:
        raise ValueError("No policy adapters found")
    return digest.hexdigest()


def _expert(name):
    return "image" if "_moe_gen" in name else "text"


def expert_gradient_stats(model):
    result = {key: {"l2_norm": 0., "max_abs": 0., "parameters_with_grad": 0,
                    "nonzero_gradient_elements": 0} for key in ("image", "text")}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        stats = result[_expert(name)]
        stats["parameters_with_grad"] += 1
        stats["nonzero_gradient_elements"] += int(torch.count_nonzero(grad))
        stats["l2_norm"] += float(grad.double().square().sum())
        stats["max_abs"] = max(stats["max_abs"], float(grad.abs().max()))
    for stats in result.values():
        stats["l2_norm"] = math.sqrt(stats["l2_norm"])
    return result


def _parameter_updates(model, before):
    result = {key: {"l2_delta": 0., "max_abs_delta": 0., "changed_elements": 0}
              for key in ("image", "text")}
    for name, parameter in model.named_parameters():
        if name not in before:
            continue
        delta = parameter.detach().cpu() - before[name]
        stats = result[_expert(name)]
        stats["l2_delta"] += float(delta.double().square().sum())
        stats["max_abs_delta"] = max(stats["max_abs_delta"], float(delta.abs().max()))
        stats["changed_elements"] += int(torch.count_nonzero(delta))
    for stats in result.values():
        stats["l2_delta"] = math.sqrt(stats["l2_delta"])
    return result


def trace_steps(trace):
    steps = []
    for event in trace:
        if event["kind"] == "text":
            steps.append({"text": event, "image": None})
        elif steps and steps[-1]["image"] is None:
            steps[-1]["image"] = event
        else:
            raise ValueError("Malformed native text/image event order")
    return steps


def final_answer(trace):
    text = "\n".join(e["text"] for e in trace if e["kind"] == "text")
    answers = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    return answers[-1].strip() if answers else ""


def judge_group(example, folders, judge, *, write=True, workers=1):
    """Require complete judgments; preserve known components if one fails.

    The caller may skip an unscorable N=8 group, never fabricate missing rewards.
    Terminal correctness is assessed independently before process judgments.
    """
    if type(workers) is not int or workers < 1:
        raise ValueError("Judge workers must be a positive integer")
    folders = list(map(Path, folders))
    if workers > 1 and len(folders) > 1:
        # Requests within one rollout remain causally ordered; independent
        # rollouts can feed a server microbatch. map preserves rollout ordering.
        def assess(folder):
            return judge_group(example, [folder], judge, write=write, workers=1)[0]
        with ThreadPoolExecutor(max_workers=min(workers, len(folders))) as pool:
            return list(pool.map(assess, folders))
    results = []
    for folder in folders:
        trace = json.loads((folder / "trace.json").read_text())
        steps, judgments = trace_steps(trace), []
        terminal_args = (example["question"], example["problem_image"]["path"], final_answer(trace), example["answer"])
        try:
            terminal = (judge.terminal_with_constraints(*terminal_args, example.get("original_constraints"))
                if hasattr(judge, "terminal_with_constraints") else judge.terminal(*terminal_args))
        except JudgeAssessmentError as error:
            raise JudgeAssessmentError(f"{folder}: terminal assessment: {error}") from error
        if write:
            atomic_json(folder / "terminal.json", terminal)
        text_credits = [[] for _ in steps]
        before, history = example["problem_image"]["path"], []
        for i, step in enumerate(steps):
            if step["image"] is None:
                # Its observations can get credit from the preceding image judge;
                # no fabricated image reward for a nonexistent manipulation.
                judgments.append({"text": 0.0, "image": 0.0, "process": None})
                continue
            after = str(folder / step["image"]["file"])
            subsequent = steps[i+1]["text"]["text"] if i+1 < len(steps) else ""
            evidence = StepEvidence(example["question"], example["problem_image"]["path"],
                before, after, step["text"]["text"], subsequent, tuple(history),
                task=example.get("task", ""), original_constraints=example.get("original_constraints"))
            try:
                verdict = judge.process(evidence)
            except JudgeAssessmentError as error:
                if write:
                    atomic_json(folder / "judge_partial.json", {
                        "status": "unscorable", "terminal": terminal, "failed_process_step": i,
                        "completed_process_components": [
                            {"step": index, "components": reward_components(item["process"]),
                             "process": item["process"]}
                            for index, item in enumerate(judgments) if item["process"] is not None],
                        "missing_components_imputed": False,
                        "error": {"type": type(error).__name__, "message": str(error)},
                        "judge_protocol_revision": getattr(judge, "protocol_revision", None)})
                raise JudgeAssessmentError(f"{folder}: process step {i}: {error}") from error
            components = reward_components(verdict)
            text_credits[i].append(components["text"])
            if i + 1 < len(steps):
                text_credits[i + 1].append(components["post_text"])
            judgments.append({**components, "process": verdict})
            history.append(step["text"]["text"])
            before = after
        for i, credits in enumerate(text_credits):
            judgments[i]["text"] = sum(credits) / len(credits) if credits else 0.0
            judgments[i]["text_credit_components"] = credits
        value = {"steps": judgments, "terminal": terminal,
            "judge_model": getattr(judge, "model_id", "Qwen/Qwen3-VL-32B-Instruct"),
            "judge_provider": getattr(judge, "provider", "local_qwen"),
            "judge_version": getattr(judge, "version", "legacy_v0"),
            "judge_protocol_revision": getattr(judge, "protocol_revision", None),
            "terminal_policy_version": getattr(judge, "terminal_version", "legacy_v0"),
            "completed": bool(trace and trace[-1]["kind"] == "text" and final_answer(trace))}
        if write:
            atomic_json(folder / "judge.json", value)
        results.append(value)
    return results


class NativeGRPOTrainer:
    def __init__(self, inferencer, optimizer, config):
        self.infer, self.model = inferencer, inferencer.model
        self.optimizer, self.config = optimizer, config
        self.progress_callback = None

    def _progress(self, **event):
        if self.progress_callback is not None:
            self.progress_callback(event)

    def update(self, folders, judgments, *, optimize=True, defer_step=False):
        """Validate the actual gradient forwards before a single optimizer step.

        A later replay failure may follow earlier backward passes, but it cannot
        update parameters or optimizer state. Discard all partial gradients on
        failure, including interruptions. Replay prefixes live for this update
        only and are never reused across parameter updates.
        """
        if defer_step and not optimize:
            raise ValueError("Deferred gradients require full on-policy validation")
        self.optimizer.zero_grad(set_to_none=True)
        try:
            with replay_context_cache(self.infer) as cache:
                result = self._update(folders, judgments, optimize=optimize, defer_step=defer_step)
                result["replay_context_cache"] = cache.stats()
                return result
        except BaseException:
            self.optimizer.zero_grad(set_to_none=True)
            raise

    def _update(self, folders, judgments, *, optimize, defer_step=False):
        """One on-policy group; diagnostics may omit archived sampling metadata."""
        if len(folders) != 8 or len(judgments) != 8:
            raise ValueError("This integration requires a complete N=8 group")
        started = time.monotonic()
        folders = list(map(Path, folders))
        policy_digest = adapter_digest(self.model) if optimize else None
        if optimize:
            # Recomputing 'old' scores cannot make an archived group on-policy.
            for folder in folders:
                stamp = folder / "sampling.json"
                if not stamp.exists() or json.loads(stamp.read_text()).get("policy_digest") != policy_digest:
                    raise ValueError("Rollout policy does not match current adapters; generate a fresh group")
        traces = [json.loads((f / "trace.json").read_text()) for f in folders]
        groups = [trace_steps(trace) for trace in traces]
        if not all(groups):
            raise ValueError("Empty rollout in training group")
        S = max(map(len, groups))
        image_reward, text_reward = torch.zeros(8, S), torch.zeros(8, S)
        valid = torch.zeros(8, S, dtype=torch.bool)
        image_valid = torch.zeros_like(valid)
        completed = torch.tensor([bool(t and t[-1]["kind"] == "text" and final_answer(t)) for t in traces])
        terminal = torch.tensor([float(j["terminal"]["correct"]) for j in judgments])
        for n, (steps, judgment) in enumerate(zip(groups, judgments)):
            if len(steps) != len(judgment["steps"]):
                raise ValueError("Judgment/trace step mismatch")
            for s, reward in enumerate(judgment["steps"]):
                valid[n, s] = True
                image_valid[n, s] = steps[s]["image"] is not None
                image_reward[n, s], text_reward[n, s] = reward["image"], reward["text"]
        if any(not torch.isfinite(value).all() for value in (image_reward, text_reward, terminal)):
            raise ValueError("Nonfinite judge reward")
        advantages = group_advantages(image_reward, text_reward, valid, terminal, self.config,
            image_valid=image_valid, completed=completed)
        returns = group_returns(image_reward, text_reward, valid, terminal, self.config,
            image_valid=image_valid, completed=completed)
        self.optimizer.zero_grad(set_to_none=True)
        metrics = {"text_losses": [], "image_losses": [], "ratios": [], "velocity_mse": [],
            "advantages": advantages.tolist(), "returns": returns.tolist(), "config": asdict(self.config),
            "image_rewards": image_reward.tolist(), "text_rewards": text_reward.tolist(),
            "terminal_rewards": terminal.tolist(), "valid": valid.tolist(), "image_valid": image_valid.tolist(),
            "completed": completed.tolist(), "active_advantage_count": int(torch.count_nonzero(advantages)),
            "policy_digest_before": policy_digest,
            "text_likelihood_backend": TEXT_LIKELIHOOD_BACKEND,
            "text_replay_memory_strategy": TEXT_REPLAY_MEMORY_STRATEGY,
            "sampling_replay_tolerance": SAMPLING_REPLAY_TOLERANCE,
            "sampling_replay_validation": "gradient_forward_before_optimizer_step",
            # These action timings include detached context construction and
            # backward. Image time also includes frozen-reference regularization.
            "replay_timing_seconds": {"text_replay_and_backward": 0., "image_replay_and_backward": 0.}}
        if not torch.count_nonzero(advantages):
            metrics.update(optimized=False, gradients_ready=False, skip_reason="flat_advantages", gradient_norm_before_clip=0.,
                elapsed_seconds=time.monotonic()-started)
            return metrics
        # Check all text provenance without replaying the model. Actual fidelity
        # is checked using the gradient-enabled forward immediately before its
        # backward pass. The optimizer runs only after the entire group passes.
        recorded_text = {(n, s): saved_text_logprob(step["text"], required=optimize)
                         for n, steps in enumerate(groups) for s, step in enumerate(steps)}
        sampling_replay_deltas = []
        text_sampling_replay_deltas = []
        replay_details = []
        text_count = int(valid.sum())
        image_count = sum(bool(s["image"]) for steps in groups for s in steps)
        for n, (folder, steps) in enumerate(zip(folders, groups)):
            for s, step in enumerate(steps):
                action_started = time.monotonic()
                prior = rebuild_context(self.infer, step["text"]["prefix"], folder)
                lp = text_logprob(self.model, prior, step["text"])
                old_text, delta, denominator = behavior_logprob(
                    lp, recorded_text[n, s], required=optimize,
                    kind="gradient-enabled text" if optimize else "text")
                if delta is not None:
                    text_sampling_replay_deltas.append(delta)
                replay_details.append({"rollout": n, "step": s, "kind": "text", "delta": delta,
                    "per_token_delta": delta / len(step["text"]["target_ids"]) if delta is not None and step["text"]["target_ids"] else None,
                    "target_tokens": len(step["text"]["target_ids"]), "denominator_source": denominator})
                loss, ratio = ppo_loss(lp, old_text.to(lp.device), advantages[n, s], self.config)
                (loss * self.config.text_weight / text_count).backward()
                metrics["text_losses"].append(float(loss.detach()))
                metrics["ratios"].append(float(ratio))
                del prior, lp, loss
                metrics["replay_timing_seconds"]["text_replay_and_backward"] += time.monotonic() - action_started
                self._progress(rollout=n, step=s, kind="text", action_seconds=time.monotonic()-action_started,
                    completed_text_actions=len(metrics["text_losses"]), total_text_actions=text_count,
                    sampling_replay_delta=delta)
                if step["image"]:
                    action_started = time.monotonic()
                    event = step["image"]
                    state = torch.load(folder / event["flow_file"], weights_only=True)
                    # Validate every saved transition denominator before spending
                    # model work on this action. The state is loaded only once.
                    old_images = [saved_logprob(transition.get("old_logp"), required=optimize, kind="image")
                                  for transition in state["transitions"]]
                    inputs = flow_inputs(self.infer, event, folder)
                    reference_inputs = reference_flow_inputs(self.infer, event, folder) if self.config.velocity_beta else None
                    for k, transition in enumerate(state["transitions"]):
                        lp, velocity = flow_logprob(self.model, inputs, event, state, transition)
                        old_image, delta, denominator = behavior_logprob(
                            lp, old_images[k], required=optimize,
                            kind="gradient-enabled image" if optimize else "image")
                        if delta is not None:
                            sampling_replay_deltas.append(delta)
                        replay_details.append({"rollout": n, "step": s, "kind": "image",
                            "transition": transition.get("index", k), "delta": delta,
                            "per_latent_delta": delta / transition["x"].numel() if delta is not None and "x" in transition else None,
                            "denominator_source": denominator})
                        policy_loss, ratio = ppo_loss(lp, old_image.to(lp.device), advantages[n, s], self.config)
                        if reference_inputs is not None:
                            with torch.no_grad(), reference_adapters(self.model):
                                _, reference = flow_logprob(self.model, reference_inputs, event, state, transition)
                            penalty = velocity_regularizer(velocity, reference)
                            del reference
                        else:
                            penalty = velocity.new_zeros(())
                        loss = policy_loss + self.config.velocity_beta * penalty
                        (loss / (image_count * len(state["transitions"]))).backward()
                        metrics["image_losses"].append(float(policy_loss.detach()))
                        metrics["ratios"].append(float(ratio))
                        metrics["velocity_mse"].append(float(penalty.detach()))
                        del lp, velocity, loss, policy_loss, penalty
                    del inputs, reference_inputs, state
                    metrics["replay_timing_seconds"]["image_replay_and_backward"] += time.monotonic() - action_started
                    self._progress(rollout=n, step=s, kind="image", action_seconds=time.monotonic()-action_started,
                        completed_image_transitions=len(metrics["image_losses"]),
                        sampling_replay_delta=max(sampling_replay_deltas,default=0.))
        metrics.update({"sampling_replay_details": replay_details,
            "max_text_sampling_replay_logp_delta": max(text_sampling_replay_deltas, default=None),
            "max_sampling_replay_logp_delta": max(sampling_replay_deltas, default=0.0),
            "gradients_ready": True})
        if defer_step:
            # Every rank must finish on-policy validation before communication.
            # Global averaging must precede clipping, including zero/flat ranks.
            metrics.update(optimized=False, gradient_norm_before_clip=None,
                elapsed_seconds=time.monotonic()-started)
            return metrics
        metrics.update(self.finish_update(optimize=optimize))
        metrics["elapsed_seconds"] = time.monotonic()-started
        return metrics

    def finish_update(self, *, optimize=True):
        """Commit already validated gradients; distributed callers average first."""
        started = time.monotonic()
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        metrics = {"expert_gradients": expert_gradient_stats(self.model)}
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        if optimize:
            before = {name: parameter.detach().cpu().clone() for name, parameter in self.model.named_parameters()
                      if parameter.requires_grad}
            self.optimizer.step()
            metrics["parameter_updates"] = _parameter_updates(self.model, before)
            metrics["policy_digest_after"] = adapter_digest(self.model)
            del before
        finite = all(bool(torch.isfinite(parameter).all()) for parameter in trainable)
        if not finite:
            raise FloatingPointError("Optimizer produced nonfinite adapter parameters")
        metrics.update({"optimized": optimize, "gradient_norm_before_clip": float(norm),
            "trainable_parameters_finite": finite, "optimizer_step_seconds": time.monotonic()-started})
        return metrics
