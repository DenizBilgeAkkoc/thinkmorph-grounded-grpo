"""CPU-only scalar mappings from persisted ThinkMorph training measurements."""
import math
import statistics


def numeric(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def stats(output, key, values):
    values = [float(v) for v in values if numeric(v)]
    if values:
        output.update({key + "_mean": statistics.fmean(values), key + "_min": min(values),
                       key + "_max": max(values), key + "_count": len(values),
                       key + "_std": statistics.pstdev(values)})


def masked(values, mask):
    return [v for row, valid in zip(values, mask) for v, keep in zip(row, valid) if keep]


def step_rows(metrics):
    """Preserve the trainer's ordered action/ratio and per-branch loss lists."""
    text_index = image_index = 0
    details = metrics.get("sampling_replay_details", [])
    ratios = metrics.get("ratios", [])
    if details and len(details) != len(ratios):
        raise ValueError("Persisted action and ratio counts differ")
    for detail, ratio in zip(details, ratios):
        kind = detail["kind"]
        n, s = detail["rollout"], detail["step"]
        advantage = metrics["advantages"][n][s]
        clip = metrics["config"]["clip_range"]
        row = dict(detail, ratio=ratio, advantage=advantage,
            ratio_outside_clip=float(ratio < 1-clip or ratio > 1+clip),
            surrogate_clipped=float((advantage > 0 and ratio > 1+clip) or
                                    (advantage < 0 and ratio < 1-clip)))
        if kind == "text":
            row["loss"] = metrics["text_losses"][text_index]; text_index += 1
        else:
            row["loss"] = metrics["image_losses"][image_index]
            row["velocity_mse"] = metrics["velocity_mse"][image_index]; image_index += 1
        yield row


def aggregate_wave(wave, configuration, examples):
    workers = [w for w in wave["workers"] if w.get("assigned")]
    measured = [w["metrics"] for w in workers if w.get("accepted") and "metrics" in w]
    output = {"train/global_step": wave["optimizer_updates"], "train/global_wave": wave["wave"],
        "train/questions_committed": wave["last_group"] + 1,
        "train/rollouts_in_committed_questions": (wave["last_group"] + 1)*8,
        "train/optimized": int(wave["optimized"]),
        "train/accepted_questions": wave["accepted_question_count"],
        "train/gradient_questions": wave["gradient_question_count"],
        "train/skipped_questions": sum(not w.get("accepted") for w in workers),
        "train/flat_questions": sum(w.get("accepted", False) and not w.get("gradients_ready", False) for w in workers),
        "timing/wave_wall_seconds": wave["wall_seconds"],
        "timing/checkpoint_seconds": wave.get("checkpoint_seconds", 0),
        "integrity/rank_policy_and_adam_identical": int(wave.get("adapters_and_optimizer_identical_across_ranks", False)),
        "integrity/checkpoint_reload_verified": int(wave.get("checkpoint_reload_verified", False))}
    recipe = configuration.get("recipe", {})
    for key in ("provider", "model", "model_id", "version", "protocol_revision", "thinking"):
        value = configuration.get("judge", {}).get(key)
        if isinstance(value, (str, int, float, bool)):
            output["judge/"+key] = value
    for branch in ("text", "image"):
        value = recipe.get("optimizer", {}).get(branch + "_learning_rate")
        if numeric(value):
            output["optimizer/" + branch + "_lr"] = value
    for name in ("text_losses", "image_losses", "velocity_mse", "ratios"):
        stats(output, "diagnostic/raw_action_" + name, [v for m in measured for v in m.get(name, [])])
    objectives = []
    for metrics in measured:
        config = metrics.get("config", {})
        text_losses = metrics.get("text_losses", [])
        text_objective = config.get("text_weight", 0.) * statistics.fmean(text_losses) if text_losses else 0.
        images = {}
        for row in step_rows(metrics):
            if row["kind"] == "image":
                images.setdefault((row["rollout"], row["step"]), []).append(row)
        image_objective = statistics.fmean(statistics.fmean(r["loss"] for r in rows) for rows in images.values()) if images else 0.
        velocity_objective = (config.get("velocity_beta", 0.) * statistics.fmean(
            statistics.fmean(r["velocity_mse"] for r in rows) for rows in images.values())) if images else 0.
        objectives.append((text_objective, image_objective, velocity_objective))
    if objectives:
        for i, name in enumerate(("text_objective", "image_policy_objective", "velocity_penalty_objective")):
            output["train/" + name] = statistics.fmean(o[i] for o in objectives)
        output["train/loss"] = statistics.fmean(sum(o) for o in objectives)
        output["train/objective_question_count"] = len(objectives)
    for name, mask in (("image_rewards", "image_valid"), ("text_rewards", "valid"),
                       ("advantages", "valid"), ("returns", "valid")):
        stats(output, "reward/" + name, [v for m in measured for v in masked(m.get(name, []), m.get(mask, []))])
    stats(output, "train/accuracy", [v for m in measured for v in m.get("terminal_rewards", [])])
    stats(output, "train/completion", [v for m in measured for v in m.get("completed", [])])
    for name in ("sampling_seconds", "judge_seconds", "judge_tail_seconds", "sampling_and_judging_seconds",
                 "judge_worker_seconds", "judge_span_seconds", "replay_backward_seconds", "local_work_seconds",
                 "readiness_wait_seconds", "gradient_sync_seconds", "optimizer_step_seconds"):
        stats(output, "timing/" + name, [w.get(name) for w in workers])
    for name in ("peak_cuda_allocated_gib", "peak_cuda_reserved_gib"):
        stats(output, "memory/" + name, [w.get(name) for w in workers])
    for name in ("text_replay_and_backward", "image_replay_and_backward", "text_old_policy_snapshot"):
        stats(output, "timing/" + name, [m.get("replay_timing_seconds", {}).get(name) for m in measured])
    for name in ("max_old_new_logp_delta", "max_old_new_token_logp_delta", "max_sampling_replay_logp_delta"):
        stats(output, "fidelity/" + name, [m.get(name) for m in measured])
    # A large native-sampling gap is explicitly a diagnostic for this backend,
    # rather than being mislabeled as the old/new identity check.
    stats(output, "diagnostic/native_vs_parallel_text_logp_delta",
          [m.get("max_text_sampling_replay_logp_delta") for m in measured])
    optimizer = next((w["optimizer_metrics"] for w in workers if w.get("optimizer_metrics")), {})
    if numeric(optimizer.get("gradient_norm_before_clip")):
        output["train/grad_norm"] = optimizer["gradient_norm_before_clip"]
    for category in ("expert_gradients", "parameter_updates"):
        for branch, values in optimizer.get(category, {}).items():
            for name, value in values.items():
                if numeric(value):
                    output[f"optimizer/{branch}/{category}/{name}"] = value
    for name, value in wave.get("sample_history_counts", {}).items():
        if numeric(value):
            output["train/history/" + name] = value
    all_rows = [row for metrics in measured for row in step_rows(metrics)]
    for step in sorted({r["step"] for r in all_rows}):
        rows = [r for r in all_rows if r["step"] == step]
        prefix = f"train/step_{step}/"
        for kind in ("text", "image"):
            branch = [r for r in rows if r["kind"] == kind]
            for source, target in (("loss", kind+"_loss"), ("ratio", kind+"_ratio"),
                                   ("ratio_outside_clip", kind+"_ratio_outside_clip"),
                                   ("surrogate_clipped", kind+"_clipfrac")):
                if branch:
                    output[prefix+target] = statistics.fmean(r[source] for r in branch)
        images = [r for r in rows if r["kind"] == "image"]
        if images:
            output[prefix+"policy_loss"] = statistics.fmean(r["loss"] for r in images)
            output[prefix+"vel_mse"] = statistics.fmean(r["velocity_mse"] for r in images)
        for name, mask_name, target in (("image_rewards", "image_valid", "image_reward"),
                                       ("text_rewards", "valid", "text_reward"),
                                       ("advantages", "valid", "advantage"), ("returns", "valid", "return")):
            values = [values[step] for m in measured for values, valid in zip(m.get(name, []), m.get(mask_name, []))
                      if step < len(valid) and valid[step]]
            if values:
                output[prefix+target] = statistics.fmean(values)
                output[prefix+target+"_std"] = statistics.pstdev(values)
        for kind, field, name in (("text", "old_new_summed_logp_delta", "text_old_new_abs_delta"),
                                   ("text", "delta", "text_native_sampling_abs_delta"),
                                   ("image", "delta", "image_sampling_replay_abs_delta")):
            values = [r.get(field) for r in rows if r["kind"] == kind and numeric(r.get(field))]
            if values:
                output[prefix+name] = statistics.fmean(values)
    if all_rows:
        output["train/clipfrac"] = statistics.fmean(r["surrogate_clipped"] for r in all_rows)
        output["train/ratio_outside_clip"] = statistics.fmean(r["ratio_outside_clip"] for r in all_rows)
    for worker in workers:
        rank = worker["rank"]
        for name in ("sampling_seconds", "judge_tail_seconds", "replay_backward_seconds", "local_work_seconds",
                     "readiness_wait_seconds", "peak_cuda_allocated_gib", "peak_cuda_reserved_gib"):
            if numeric(worker.get(name)):
                output[f"rank/{rank}/{name}"] = worker[name]
    by_task = {}
    for worker in workers:
        example = examples.get(worker["group"], {})
        by_task.setdefault(str(example.get("task", "unknown")).replace("/", "_"), []).append(worker)
    for task, selected in by_task.items():
        output[f"task/{task}/questions"] = len(selected)
        output[f"task/{task}/accepted_questions"] = sum(w.get("accepted", False) for w in selected)
        stats(output, f"task/{task}/accuracy", [v for w in selected for v in w.get("metrics", {}).get("terminal_rewards", [])])
    return output
