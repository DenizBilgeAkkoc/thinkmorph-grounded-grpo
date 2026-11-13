"""Experimental N8 GRPO update with paired text and singleton image replay.

This module is outside production. Pairing changes execution layout only: all
returns, advantages and loss denominators still come from the full N8 question.
Use only with the validated experimental paired trace loader and native GPU
sampling/replay tests. No checkpoint/source migration is implemented here.
"""
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from . import trainer as native
from .paired_replay import paired_token_logprobs, verify_behavior_logprobs


class PairedGRPOTrainer(native.NativeGRPOTrainer):
    """Reuse native transaction cleanup, clipping, Adam and deferred stepping."""

    def _update(self, folders, judgments, *, optimize, defer_step=False):
        from .paired_rollout import load_text_batches

        if len(folders) != 8 or len(judgments) != 8:
            raise ValueError("Paired replay still requires a complete N=8 question")
        started = time.monotonic()
        folders = list(map(Path, folders))
        policy_digest = native.adapter_digest(self.model) if optimize else None
        for n, folder in enumerate(folders):
            stamp = json.loads((folder / "sampling.json").read_text())
            if stamp.get("rollout_index") != n:
                raise ValueError("Folders must follow logical rollout order 0 through 7")
            if optimize and stamp.get("policy_digest") != policy_digest:
                raise ValueError("Rollout policy differs from current adapters")
        traces = [json.loads((folder / "trace.json").read_text()) for folder in folders]
        groups = [native.trace_steps(trace) for trace in traces]
        if not all(groups):
            raise ValueError("Empty rollout in training group")
        # Validate complete trace/artifact membership even for a flat question.
        batches = load_text_batches(folders, traces)
        size = max(map(len, groups))
        image_reward, text_reward = torch.zeros(8, size), torch.zeros(8, size)
        valid = torch.zeros(8, size, dtype=torch.bool)
        image_valid = torch.zeros_like(valid)
        completed = torch.tensor([bool(t and t[-1]["kind"] == "text" and native.final_answer(t)) for t in traces])
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
        advantages = native.group_advantages(image_reward, text_reward, valid, terminal, self.config,
            image_valid=image_valid, completed=completed)
        returns = native.group_returns(image_reward, text_reward, valid, terminal, self.config,
            image_valid=image_valid, completed=completed)
        self.optimizer.zero_grad(set_to_none=True)
        metrics = dict(text_losses=[], image_losses=[], ratios=[], velocity_mse=[],
            advantages=advantages.tolist(), returns=returns.tolist(), config=asdict(self.config),
            image_rewards=image_reward.tolist(), text_rewards=text_reward.tolist(),
            terminal_rewards=terminal.tolist(), valid=valid.tolist(), image_valid=image_valid.tolist(),
            completed=completed.tolist(), active_advantage_count=int(torch.count_nonzero(advantages)),
            policy_digest_before=policy_digest, text_likelihood_backend="native_incremental",
            text_replay_layout="packed_active_text_v1",
            text_replay_memory_strategy="layerwise_packed_checkpoint",
            sampling_replay_tolerance=native.SAMPLING_REPLAY_TOLERANCE,
            sampling_replay_validation="gradient_forward_before_optimizer_step",
            replay_timing_seconds=dict(text_replay_and_backward=0., image_replay_and_backward=0.),
            text_batch_sizes=[len(batch["members"]) for batch in batches],
            loss_normalization="original_complete_N8_question")
        if not torch.count_nonzero(advantages):
            metrics.update(optimized=False, gradients_ready=False, skip_reason="flat_advantages",
                gradient_norm_before_clip=0., elapsed_seconds=time.monotonic() - started)
            return metrics

        recorded = {(n, s): native.saved_text_logprob(step["text"], required=optimize)
                    for n, steps in enumerate(groups) for s, step in enumerate(steps)}
        text_count = int(valid.sum())
        image_count = sum(bool(step["image"]) for steps in groups for step in steps)
        text_deltas, image_deltas, details = [], [], []

        # Both members use one matched forward/backward. The original text-count
        # denominator is applied once to the sum; no pair-level normalization.
        for batch in batches:
            action_started = time.monotonic()
            members = batch["members"]
            events = [member["event"] for member in members]
            priors = [native.rebuild_context(self.infer, member["event"]["prefix"],
                folders[member["rollout_index"]]) for member in members]
            for member, prior in zip(members, priors):
                if (prior["kv_lens"] != member["prefix_kv_lens"] or
                        prior["ropes"] != member["prefix_ropes"]):
                    raise ValueError("Reconstructed prefix layout differs from recorded sampling")
            scores = paired_token_logprobs(self.model, priors, events,
                active_lane_schedule=batch["active_lane_schedule"], bos_id=batch["bos_id"],
                layout_revision=batch["layout_revision"])
            if len(scores) != len(members):
                raise ValueError("Paired replay omitted a member")
            token_checks = (verify_behavior_logprobs(scores, events, tolerance=native.SAMPLING_REPLAY_TOLERANCE)
                            if optimize else [None] * len(members))
            if any(check is not None and check["max_token_logp_delta"] > native.SAMPLING_REPLAY_TOLERANCE
                   for check in token_checks):
                raise ValueError("Paired text per-token sampling/replay likelihood mismatch")
            losses, batch_metrics = [], []
            for member, token_scores, token_check in zip(members, scores, token_checks):
                n, s = member["rollout_index"], member["text_action_index"]
                event = member["event"]
                if event != groups[n][s]["text"]:
                    raise ValueError("Validated batch does not reference the current trace action")
                if token_scores.numel() != len(event["target_ids"]):
                    raise ValueError("Paired replay omitted retained action tokens")
                lp = token_scores.sum()
                old, delta, denominator = native.behavior_logprob(lp, recorded[n, s],
                    required=optimize, kind="gradient-enabled paired text" if optimize else "paired text")
                loss, ratio = native.ppo_loss(lp, old.to(lp.device), advantages[n, s], self.config)
                losses.append(loss)
                batch_metrics.append((float(loss.detach()), float(ratio)))
                if delta is not None:
                    text_deltas.append(delta)
                details.append(dict(rollout=n, step=s, kind="text", delta=delta,
                    per_token_delta=delta / len(event["target_ids"]) if delta is not None and event["target_ids"] else None,
                    target_tokens=len(event["target_ids"]), denominator_source=denominator,
                    max_token_logp_delta=token_check["max_token_logp_delta"] if token_check is not None else None,
                    batch_id=batch["batch_id"], replay_layout=batch["layout_revision"]))
            # Validate every member before backpropagating this batch.
            (torch.stack(losses).sum() * self.config.text_weight / text_count).backward()
            for loss_value, ratio_value in batch_metrics:
                metrics["text_losses"].append(loss_value)
                metrics["ratios"].append(ratio_value)
            del priors, scores, losses, token_scores, lp, loss
            elapsed = time.monotonic() - action_started
            metrics["replay_timing_seconds"]["text_replay_and_backward"] += elapsed
            self._progress(rollout=members[0]["rollout_index"], step=members[0]["text_action_index"],
                kind="text", batch_id=batch["batch_id"], batch_size=len(members),
                action_seconds=elapsed, completed_text_actions=len(metrics["text_losses"]),
                total_text_actions=text_count, sampling_replay_delta=max(text_deltas, default=0.))

        # Image integration and likelihoods remain the established singleton
        # path, including the original frozen-reference velocity regularizer.
        for n, (folder, steps) in enumerate(zip(folders, groups)):
            for s, step in enumerate(steps):
                if step["image"] is None:
                    continue
                action_started = time.monotonic()
                event = step["image"]
                state = torch.load(folder / event["flow_file"], weights_only=True)
                old_images = [native.saved_logprob(t.get("old_logp"), required=optimize, kind="image")
                              for t in state["transitions"]]
                inputs = native.flow_inputs(self.infer, event, folder)
                reference_inputs = native.reference_flow_inputs(self.infer, event, folder) if self.config.velocity_beta else None
                for k, transition in enumerate(state["transitions"]):
                    lp, velocity = native.flow_logprob(self.model, inputs, event, state, transition)
                    old, delta, denominator = native.behavior_logprob(lp, old_images[k], required=optimize,
                        kind="gradient-enabled image" if optimize else "image")
                    if delta is not None:
                        image_deltas.append(delta)
                    details.append(dict(rollout=n, step=s, kind="image", transition=transition.get("index", k),
                        delta=delta, per_latent_delta=delta / transition["x"].numel()
                            if delta is not None and "x" in transition else None,
                        denominator_source=denominator))
                    policy_loss, ratio = native.ppo_loss(lp, old.to(lp.device), advantages[n, s], self.config)
                    if reference_inputs is not None:
                        with torch.no_grad(), native.reference_adapters(self.model):
                            _, reference = native.flow_logprob(self.model, reference_inputs, event, state, transition)
                        penalty = native.velocity_regularizer(velocity, reference)
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
                elapsed = time.monotonic() - action_started
                metrics["replay_timing_seconds"]["image_replay_and_backward"] += elapsed
                self._progress(rollout=n, step=s, kind="image", action_seconds=elapsed,
                    completed_image_transitions=len(metrics["image_losses"]),
                    sampling_replay_delta=max(image_deltas, default=0.))

        metrics.update(sampling_replay_details=details, max_text_sampling_replay_logp_delta=max(text_deltas, default=None),
            max_sampling_replay_logp_delta=max(image_deltas, default=0.), gradients_ready=True)
        if defer_step:
            metrics.update(optimized=False, gradient_norm_before_clip=None, elapsed_seconds=time.monotonic() - started)
            return metrics
        metrics.update(self.finish_update(optimize=optimize))
        metrics["elapsed_seconds"] = time.monotonic() - started
        return metrics
