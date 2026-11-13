"""Isolated N8 trial using consistent parallel old/new text probabilities.

Text denominators are frozen full-causal scores from the current archived
policy, not the recorded native sampler probabilities. Their discrepancy is
persisted as a diagnostic. Image fidelity, N8 normalization and optimization
remain the established native paths. This module is not a production migration.
"""
from dataclasses import asdict
import json
import os
import math
from pathlib import Path
import time

import torch

from . import trainer as native
from .parallel_replay import (PARALLEL_REPLAY_REVISION, parallel_token_logprobs,
                              snapshot_parallel_token_logprobs)


TEXT_LIKELIHOOD_BACKEND = "parallel_full_sequence_consistent_old_new"
DENOMINATOR_SOURCE = "parallel_old_policy_snapshot"


def _parameter_versions(model):
    return tuple((name, id(value), value._version, value.dtype, tuple(value.shape))
                 for name, value in model.named_parameters())


def _prefix_versions(priors):
    return tuple((tuple(prior["kv_lens"]), tuple(prior["ropes"]),
        tuple((field, layer, None if value is None else
               (id(value), value.data_ptr(), value._version, value.dtype, tuple(value.shape)))
              for field in ("key_cache", "value_cache")
              for layer, value in sorted(getattr(prior["past_key_values"], field).items())))
        for prior in priors)


def _verify_parallel_old_new(scores, old_scores, *, tolerance):
    """Guard unchanged-policy identity independently of recorded sampling."""
    if len(scores) != len(old_scores):
        raise ValueError("Parallel old/new lane counts differ")
    checks = []
    for lane, (current, old) in enumerate(zip(scores, old_scores)):
        if (current.shape != old.shape or old.requires_grad or
                current.dtype != torch.float64 or old.dtype != torch.float64 or
                not torch.isfinite(current).all() or not torch.isfinite(old).all()):
            raise ValueError("Invalid parallel old/new likelihood snapshots")
        difference = current.detach() - old
        token_delta = float(difference.abs().max()) if current.numel() else 0.
        summed_delta = abs(float(current.detach().sum() - old.sum()))
        if token_delta > tolerance or summed_delta > tolerance:
            raise ValueError(f"Parallel lane {lane} old/new unchanged-policy mismatch: {token_delta}, {summed_delta}")
        checks.append(dict(lane=lane, max_token_logp_delta=token_delta,
            summed_logp_delta=summed_delta, unchanged_policy_ratio=math.exp(float(difference.sum())),
            tokens=current.numel(), tolerance=tolerance))
    return checks


class ParallelGRPOTrainer(native.NativeGRPOTrainer):
    """Reuse native transaction cleanup, clipping, Adam and deferred stepping."""

    def _update(self, folders, judgments, *, optimize, defer_step=False):
        from .paired_rollout import load_text_batches

        if len(folders) != 8 or len(judgments) != 8:
            raise ValueError("Paired replay still requires a complete N=8 question")
        started = time.monotonic()
        folders = list(map(Path, folders))
        policy_digest = native.adapter_digest(self.model)
        parameter_versions = _parameter_versions(self.model)
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
        # Validate the complete generation artifacts before choosing a separate
        # deterministic likelihood layout. The recorded sampling layout is intact.
        from .execution_tuning import replay_batches
        batches = replay_batches(batches, width=2)
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
            policy_digest_before=policy_digest, text_likelihood_backend=TEXT_LIKELIHOOD_BACKEND,
            text_replay_layout="packed_active_text_v1",
            text_replay_memory_strategy="full_causal_parallel_with_existing_layer_and_head_checkpoints",
            text_parallel_replay_revision=PARALLEL_REPLAY_REVISION,
            text_denominator_source=DENOMINATOR_SOURCE,
            sampling_replay_tolerance=native.SAMPLING_REPLAY_TOLERANCE,
            sampling_replay_validation="text_parallel_old_new_identity_and_unchanged_native_image_fidelity",
            text_sampling_discrepancy_is_diagnostic_only=True,
            old_snapshot_seconds=0., text_old_policy_snapshots=[],
            replay_timing_seconds=dict(text_replay_and_backward=0., image_replay_and_backward=0.,
                                      text_old_policy_snapshot=0.),
            text_batch_sizes=[len(batch["members"]) for batch in batches],
            generation_text_batch_size=int(os.environ.get('THINKMORPH_TEXT_BATCH_SIZE','2')),
            replay_layer_checkpointing=os.environ.get('THINKMORPH_REPLAY_CHECKPOINT','on'),
            loss_normalization="original_complete_N8_question")
        if not torch.count_nonzero(advantages):
            metrics.update(optimized=False, gradients_ready=False, skip_reason="flat_advantages",
                gradient_norm_before_clip=0., elapsed_seconds=time.monotonic() - started)
            return metrics

        recorded = {(n, s): native.saved_text_logprob(step["text"], required=optimize)
                    for n, steps in enumerate(groups) for s, step in enumerate(steps)}
        text_count = int(valid.sum())
        image_count = sum(bool(step["image"]) for steps in groups for step in steps)
        text_deltas, image_deltas, details, old_new_deltas, old_new_token_deltas = [], [], [], [], []

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
            prefix_versions = _prefix_versions(priors)
            if _parameter_versions(self.model) != parameter_versions:
                raise ValueError("Policy changed before parallel old-policy snapshot")
            snapshot_started = time.monotonic()
            old_scores = snapshot_parallel_token_logprobs(self.model, priors, events,
                active_lane_schedule=batch["active_lane_schedule"], bos_id=batch["bos_id"],
                layout_revision=batch["layout_revision"])
            # CPU values persist exact float64 old scores and synchronize snapshot
            # work before measuring its cost. This time stays inside text cost.
            old_score_values = [value.cpu().tolist() for value in old_scores]
            snapshot_seconds = time.monotonic() - snapshot_started
            metrics["old_snapshot_seconds"] += snapshot_seconds
            metrics["replay_timing_seconds"]["text_old_policy_snapshot"] += snapshot_seconds
            scores = parallel_token_logprobs(self.model, priors, events,
                active_lane_schedule=batch["active_lane_schedule"], bos_id=batch["bos_id"],
                layout_revision=batch["layout_revision"])
            if len(scores) != len(members):
                raise ValueError("Paired replay omitted a member")
            token_checks = _verify_parallel_old_new(scores, old_scores,
                tolerance=native.SAMPLING_REPLAY_TOLERANCE)
            if (_parameter_versions(self.model) != parameter_versions or
                    _prefix_versions(priors) != prefix_versions):
                raise ValueError("Policy or prefix changed across parallel old/new scoring")
            metrics["text_old_policy_snapshots"].append(dict(batch_id=batch["batch_id"],
                policy_digest=policy_digest, old_policy_digest=policy_digest, new_policy_digest=policy_digest,
                policy_parameter_versions_unchanged=True, prefix_versions_unchanged=True,
                denominator_source=DENOMINATOR_SOURCE, scorer_revision=PARALLEL_REPLAY_REVISION,
                checkpoint_chunk_size=128, replay_layout=batch["layout_revision"],
                active_lane_schedule=batch["active_lane_schedule"],
                prefix_kv_lens=[prior["kv_lens"] for prior in priors],
                prefix_ropes=[prior["ropes"] for prior in priors],
                old_snapshot_seconds=snapshot_seconds,
                members=[dict(rollout=member["rollout_index"], step=member["text_action_index"],
                              input_ids=event["input_ids"], target_ids=event["target_ids"],
                              temperature=event["temperature"], old_token_logps=values,
                              old_summed_logp=sum(values), old_new_check=check)
                         for member,event,values,check in zip(members,events,old_score_values,token_checks)]))
            losses, batch_metrics = [], []
            for member, token_scores, old_tokens, token_check in zip(members, scores, old_scores, token_checks):
                n, s = member["rollout_index"], member["text_action_index"]
                event = member["event"]
                if event != groups[n][s]["text"]:
                    raise ValueError("Validated batch does not reference the current trace action")
                if token_scores.numel() != len(event["target_ids"]):
                    raise ValueError("Paired replay omitted retained action tokens")
                lp = token_scores.sum()
                old = old_tokens.sum()
                saved = recorded[n, s]
                signed_sampling_delta = float(lp.detach().cpu()) - float(saved) if saved is not None else None
                delta = abs(signed_sampling_delta) if signed_sampling_delta is not None else None
                sampled_tokens = event.get("sampling_token_logps")
                sampling_token_delta = (float((token_scores.detach() - token_scores.new_tensor(sampled_tokens)).abs().max())
                    if sampled_tokens is not None and token_scores.numel() else 0. if sampled_tokens is not None else None)
                old_new_deltas.append(token_check["summed_logp_delta"])
                old_new_token_deltas.append(token_check["max_token_logp_delta"])
                loss, ratio = native.ppo_loss(lp, old.to(lp.device), advantages[n, s], self.config)
                losses.append(loss)
                batch_metrics.append((float(loss.detach()), float(ratio)))
                if delta is not None:
                    text_deltas.append(delta)
                details.append(dict(rollout=n, step=s, kind="text", delta=delta,
                    per_token_delta=delta / len(event["target_ids"]) if delta is not None and event["target_ids"] else None,
                    target_tokens=len(event["target_ids"]), denominator_source=DENOMINATOR_SOURCE,
                    sampling_comparison_is_diagnostic=True, signed_sampling_logp_delta=signed_sampling_delta,
                    max_token_sampling_logp_delta=sampling_token_delta,
                    old_snapshot_vs_recorded_logp_delta=float(old.cpu())-float(saved) if saved is not None else None,
                    old_new_summed_logp_delta=token_check["summed_logp_delta"],
                    max_old_new_token_logp_delta=token_check["max_token_logp_delta"],
                    old_policy_digest=policy_digest, new_policy_digest=policy_digest,
                    batch_id=batch["batch_id"], replay_layout=batch["layout_revision"]))
            # Validate every member before backpropagating this batch.
            (torch.stack(losses).sum() * self.config.text_weight / text_count).backward()
            for loss_value, ratio_value in batch_metrics:
                metrics["text_losses"].append(loss_value)
                metrics["ratios"].append(ratio_value)
            if _parameter_versions(self.model) != parameter_versions or _prefix_versions(priors) != prefix_versions:
                raise ValueError("Policy or prefix changed during parallel text backward")
            del priors, scores, old_scores, old_score_values, losses, token_scores, old_tokens, old, lp, loss
            elapsed = time.monotonic() - action_started
            metrics["replay_timing_seconds"]["text_replay_and_backward"] += elapsed
            self._progress(rollout=members[0]["rollout_index"], step=members[0]["text_action_index"],
                kind="text", batch_id=batch["batch_id"], batch_size=len(members),
                action_seconds=elapsed, completed_text_actions=len(metrics["text_losses"]),
                total_text_actions=text_count, sampling_replay_delta=max(text_deltas, default=0.))

        if any('batched_image_replay' in e for t in traces for e in t if e['kind']=='image'):
            from .image_batching import load_image_batches, backward_images
            image_batches = load_image_batches(folders,traces)
            metrics['image_batch_sizes'] = [len(b['members']) for b in image_batches]
            metrics['sampling_replay_validation'] = 'text_parallel_identity_and_matched_batched_image_fidelity'
            backward_images(self,image_batches,advantages,image_count,metrics,details,image_deltas,optimize)
        else:
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

        if _parameter_versions(self.model) != parameter_versions or native.adapter_digest(self.model) != policy_digest:
            raise ValueError("Policy changed before the single optimizer step")
        metrics.update(sampling_replay_details=details, max_text_sampling_replay_logp_delta=max(text_deltas, default=None),
            max_old_new_logp_delta=max(old_new_deltas, default=0.),
            max_old_new_token_logp_delta=max(old_new_token_deltas, default=0.),
            old_new_policy_digest_verified=policy_digest, old_new_policy_identity_verified=True,
            max_sampling_replay_logp_delta=max(image_deltas, default=0.), gradients_ready=True)
        if defer_step:
            metrics.update(optimized=False, gradient_norm_before_clip=None, elapsed_seconds=time.monotonic() - started)
            return metrics
        metrics.update(self.finish_update(optimize=optimize))
        metrics["elapsed_seconds"] = time.monotonic() - started
        return metrics
