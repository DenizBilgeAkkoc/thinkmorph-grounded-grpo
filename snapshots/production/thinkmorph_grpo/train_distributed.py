"""Synchronous native GRPO: one complete N8 question per GPU and shared AdamW.

Launch with torchrun --standalone --nproc-per-node=N -m thinkmorph_grpo.train_distributed.
Each rank retains the original within-question normalization. Gradients are
averaged between questions before a single shared clipping/optimizer step.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import socket
import time
import traceback
import uuid


def pin_visible_gpu():
    """Narrow torchrun's visibility before importing CUDA-using project code."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        selected = str(local_rank)
    else:
        devices = [item.strip() for item in visible.split(",") if item.strip()]
        if len(devices) != local_world:
            raise RuntimeError("Launch torchrun with exactly one visible GPU per local worker")
        selected = devices[local_rank]
    os.environ["CUDA_VISIBLE_DEVICES"] = selected
    return selected


class DistributedPhaseError(RuntimeError):
    pass


class UnscorableJudgeGroup(RuntimeError):
    """Only a judge worker exhausted semantic validation, never native sampling."""


class Coordinator:
    """Small CPU status collectives; a failed rank never silently loses its vote."""
    def __init__(self, root, rank, world, control, dist, atomic_json):
        self.root, self.rank, self.world = root, rank, world
        self.control, self.dist, self.atomic_json = control, dist, atomic_json
        self.ready = False
        self.summary = None

    def gather(self, value):
        values = [None] * self.world
        self.dist.all_gather_object(values, value, group=self.control)
        return values

    def status(self, phase, **details):
        if not self.ready:
            return
        directory = self.root / "ranks" / f"rank_{self.rank:05d}"
        directory.mkdir(parents=True, exist_ok=True)
        self.atomic_json(directory / "status.json",
            dict(rank=self.rank, phase=phase, updated_at_unix=time.time(), **details))
        if self.rank == 0 and self.summary is not None:
            self.summary.update(phase=phase, updated_at_unix=time.time())
            self.atomic_json(self.root / "summary.json", self.summary)

    def phase(self, label, function, *, details=None):
        result, error = None, None
        try:
            self.status(label, **(details or {}))
            result = function()
        except BaseException as exception:
            error = {"rank": self.rank, "type": type(exception).__name__,
                     "message": str(exception), "traceback": traceback.format_exc()}
        statuses = self.gather(dict(rank=self.rank, error=error))
        errors = [item["error"] for item in statuses if item["error"] is not None]
        if errors:
            if self.ready:
                directory = self.root / "ranks" / f"rank_{self.rank:05d}"
                directory.mkdir(parents=True, exist_ok=True)
                self.atomic_json(directory / "failure.json",
                                 dict(phase=label, errors=errors))
            raise DistributedPhaseError(label + ": " + "; ".join(
                f"rank {item['rank']} {item['type']}: {item['message']}" for item in errors))
        return result

    def same(self, label, value):
        values = self.gather(value)
        if any(item != values[0] for item in values[1:]):
            raise DistributedPhaseError(label + " differs across ranks: " + repr(values))
        return values


def _atomic_torch_save(torch, path, value):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _checkpoint_paths(path):
    return path.with_name(path.name + ".rng.pt"), path.with_name(path.name + ".complete.json")


def _restore(torch, base, checkpoint, model, optimizer, configuration, rank, world,
             *, include_sample_history=False):
    from . import sample_history as history
    checkpoint = Path(checkpoint)
    rng_path, complete_path = _checkpoint_paths(checkpoint)
    marker = json.loads(complete_path.read_text())
    rng = torch.load(rng_path, map_location="cpu", weights_only=True)
    if marker.get("schema_version") != 1 or rng.get("schema_version") != 1:
        raise ValueError("Unsupported distributed checkpoint metadata")
    if rng["world_size"] != world or len(rng["rank_rng"]) != world:
        raise ValueError("Distributed checkpoint world size changed")
    if base._state_digest(rng["rank_rng"]) != marker["rank_rng_digest"]:
        raise ValueError("Distributed checkpoint RNG digest mismatch")
    sample_history = rng.get("sample_history")
    history_digest = marker.get("sample_history_digest")
    if sample_history is not None or history_digest is not None or "sample_history" in configuration:
        if sample_history is None or history_digest != history.digest(sample_history):
            raise ValueError("Distributed checkpoint sample-history digest mismatch")
        history.validate_history(sample_history,
            manifest_digest=configuration.get("sample_history", {}).get("manifest_digest"))
    state = base.restore_checkpoint(checkpoint, model, optimizer, configuration=configuration)
    for field in ("group", "optimizer_updates", "policy_digest", "optimizer_digest"):
        if state[field] != marker[field] or state[field] != rng[field]:
            raise ValueError("Distributed checkpoint metadata mismatch: " + field)
    base.restore_rng_state(rng["rank_rng"][rank])
    result = {key: state[key] for key in ("group", "optimizer_updates", "policy_digest", "optimizer_digest")}
    result["wave"] = marker["wave"]
    if sample_history is not None:
        result["sample_history_digest"] = history_digest
        if include_sample_history:
            result["sample_history"] = sample_history
    return result


def _save(torch, base, coordinator, checkpoint, model, optimizer, configuration,
          *, last_group, optimizer_updates, wave, sample_history=None):
    from . import sample_history as history
    states = coordinator.gather(base.capture_rng_state())
    identity = dict(policy_digest=base.adapter_digest(model),
                    optimizer_digest=base._state_digest(optimizer.state_dict()))

    def write():
        if coordinator.rank != 0:
            return
        if sample_history is not None:
            history.validate_history(sample_history,
                manifest_digest=configuration.get("sample_history", {}).get("manifest_digest"))
        elif "sample_history" in configuration:
            raise ValueError("Distributed checkpoint requires sample history")
        base.save_checkpoint(checkpoint, model, optimizer, group=last_group,
                             optimizer_updates=optimizer_updates, configuration=configuration)
        rng_path, marker_path = _checkpoint_paths(checkpoint)
        metadata = dict(schema_version=1, group=last_group, optimizer_updates=optimizer_updates,
                        wave=wave, world_size=coordinator.world, **identity)
        payload = dict(**metadata, rank_rng=states)
        marker = dict(**metadata, rank_rng_digest=base._state_digest(states))
        if sample_history is not None:
            payload["sample_history"] = sample_history
            marker["sample_history_digest"] = history.digest(sample_history)
        _atomic_torch_save(torch, rng_path, payload)
        base.atomic_json(marker_path, marker)
    coordinator.phase("checkpoint_save", write, details={"wave": wave})
    return identity


def _sample_and_judge(base, loaded, recorder, judge, example, cfg, root, group,
                      optimizer_updates, policy_digest, workers, overlap, coordinator,
                      *, sample_attempt=None, paired_configuration=None):
    """Judge completed files on CPU threads while the GPU samples later rollouts."""
    if paired_configuration is not None:
        return _sample_and_judge_paired(base, loaded, judge, example, cfg, root, group,
            optimizer_updates, policy_digest, workers, overlap, coordinator,
            sample_attempt=sample_attempt, paired_configuration=paired_configuration)
    from PIL import Image
    folder = root / f"group_{group:06d}"
    folder.mkdir(parents=True, exist_ok=False)
    base.atomic_json(folder / "group.json", dict(example_id=example.get("example_id", example["task"]),
        group=group, rank=coordinator.rank, policy_digest=policy_digest,
        prior_optimizer_updates=optimizer_updates))
    windows = base.shared_windows(cfg["seed"] + group, cfg["native_settings"]["max_rounds"],
                                  cfg["native_settings"]["num_timesteps"], cfg["sde_window_size"])
    folders, futures, intervals = [], [], []
    started = time.monotonic()

    def assess(rollout_folder):
        began = time.monotonic()
        try:
            return base.judge_group(example, [rollout_folder], judge, write=True, workers=1)[0]
        finally:
            # Each task appends one immutable interval; list append is atomic.
            intervals.append((began, time.monotonic()))

    try:
        with ThreadPoolExecutor(max_workers=min(workers, 8)) as pool:
            for n in range(8):
                seed = cfg["seed"] + group * 1000 + n
                if sample_attempt is not None:
                    sample_attempt.begin_rollout(n, windows)
                base.seed_everything(seed)
                rollout_folder = folder / f"rollout_{n:02d}"
                with base.NativeFlowBridge(loaded[0], windows, cfg["noise_level"], cfg["sde_window_size"]) as bridge:
                    recorder.begin_recording(rollout_folder, bridge)
                    with Image.open(example["problem_image"]["path"]) as image:
                        recorder.interleave_inference([image.convert("RGB"), example["question"]],
                                                      **cfg["native_settings"])
                    base.atomic_json(rollout_folder / "sampling.json", dict(seed=seed, windows=windows,
                        group=group, policy_digest=policy_digest, prior_optimizer_updates=optimizer_updates,
                        base_checkpoint=cfg["checkpoint"], native_revision=cfg["native_revision"],
                        rollout_configuration_digest=base._state_digest(cfg),
                        example_id=example.get("example_id", example["task"]), rank=coordinator.rank))
                if sample_attempt is not None:
                    sample_attempt.finish_rollout(n, rollout_folder)
                folders.append(rollout_folder)
                if overlap:
                    futures.append(pool.submit(assess, rollout_folder))
                coordinator.status("sampling_and_judging" if overlap else "sampling",
                                   group=group, rollouts_completed=n + 1)
                print(json.dumps(dict(event="rollout_complete", rank=coordinator.rank, group=group,
                                      rollout=n, seed=seed, seconds=time.monotonic() - started)), flush=True)
            recorder.events = []
            sampling_finished = time.monotonic()
            if not overlap:
                futures = [pool.submit(assess, rollout_folder) for rollout_folder in folders]
            coordinator.status("waiting_for_judge", group=group, rollouts_completed=8)
            judgments, errors = [], []
            for future in futures:
                try:
                    judgments.append(future.result())
                except Exception as error:
                    judgments.append(None)
                    errors.append(error)
            # A semantic failure in one rollout must not hide a provider or
            # programming failure in another when skipping unscorable groups.
            fatal = next((error for error in errors if not isinstance(error,base.JudgeAssessmentError)),None)
            if fatal is not None:
                raise fatal
            if errors:
                raise UnscorableJudgeGroup(str(errors[0])) from errors[0]
    finally:
        recorder.events = []
    finished = time.monotonic()
    return folders, judgments, dict(sampling_seconds=sampling_finished - started,
        judge_seconds=finished - sampling_finished, judge_tail_seconds=finished - sampling_finished,
        sampling_and_judging_seconds=finished - started,
        judge_worker_seconds=sum(end - begin for begin, end in intervals),
        judge_span_seconds=max(end for _, end in intervals) - min(begin for begin, _ in intervals),
        judge_overlapped_with_sampling=overlap)


def _sample_and_judge_paired(base, loaded, judge, example, cfg, root, group,
                              optimizer_updates, policy_digest, workers, overlap, coordinator,
                              *, sample_attempt, paired_configuration):
    """Four text pairs remain one N8 question, with per-rollout judge overlap."""
    from .paired_rollout import generate_group
    futures, intervals, started_rollouts, completed = {}, [], set(), set()
    started = time.monotonic()

    def assess(folder):
        began = time.monotonic()
        try:
            return base.judge_group(example, [folder], judge, write=True, workers=1)[0]
        finally:
            intervals.append((began, time.monotonic()))

    with ThreadPoolExecutor(max_workers=min(workers, 8)) as pool:
        def begin(number, seed, windows):
            if (number != len(started_rollouts) or number in started_rollouts or
                    seed != cfg["seed"] + group * 1000 + number):
                raise ValueError("Paired rollout start/seed order differs from the fixed N8 recipe")
            if sample_attempt is not None:
                sample_attempt.begin_rollout(number, windows)
            started_rollouts.add(number)

        def finish(folder, number, seed):
            if number not in started_rollouts or number in completed:
                raise ValueError("Paired rollout completion was missing or repeated")
            if seed != cfg["seed"] + group * 1000 + number:
                raise ValueError("Completed paired rollout seed changed")
            if sample_attempt is not None:
                sample_attempt.finish_rollout(number, folder)
            completed.add(number)
            if overlap:
                futures[number] = pool.submit(assess, folder)
            coordinator.status("sampling_and_judging" if overlap else "sampling",
                               group=group, rollouts_completed=len(completed))
            print(json.dumps(dict(event="rollout_complete", rank=coordinator.rank, group=group,
                rollout=number, seed=seed, seconds=time.monotonic() - started)), flush=True)

        def progress(event):
            coordinator.status("sampling_and_judging" if overlap else "sampling", group=group,
                rollouts_completed=len(completed), generation_event=event.get("event"),
                active_pair=event.get("pair"), active_text_step=event.get("step"))
            print(json.dumps(dict(event, rank=coordinator.rank, group=group)), flush=True)

        result = generate_group(loaded, cfg=cfg, example=example, group_index=group,
            output=root / f"group_{group:06d}", experiment_configuration=paired_configuration,
            policy_digest=policy_digest, prior_optimizer_updates=optimizer_updates,
            progress=progress, on_rollout_start=begin, on_rollout_complete=finish)
        folders = result["folders"]
        if started_rollouts != set(range(8)) or completed != set(range(8)) or len(folders) != 8:
            raise ValueError("Paired generation did not complete exactly eight rollout audits")
        sampling_finished = time.monotonic()
        if not overlap:
            futures = {number: pool.submit(assess, folder) for number, folder in enumerate(folders)}
        if set(futures) != set(range(8)):
            raise ValueError("Paired generation did not schedule every rollout judgment")
        coordinator.status("waiting_for_judge", group=group, rollouts_completed=8)
        judgments, errors = [], []
        for number in range(8):
            try:
                judgments.append(futures[number].result())
            except Exception as error:
                judgments.append(None)
                errors.append(error)
        fatal = next((error for error in errors if not isinstance(error, base.JudgeAssessmentError)), None)
        if fatal is not None:
            raise fatal
        if errors:
            raise UnscorableJudgeGroup(str(errors[0])) from errors[0]
    finished = time.monotonic()
    return folders, judgments, dict(sampling_seconds=sampling_finished - started,
        judge_seconds=finished - sampling_finished, judge_tail_seconds=finished - sampling_finished,
        sampling_and_judging_seconds=finished - started,
        judge_worker_seconds=sum(end - begin for begin, end in intervals),
        judge_span_seconds=max(end for _, end in intervals) - min(begin for begin, _ in intervals),
        judge_overlapped_with_sampling=overlap, text_rollout_batch_size=2,
        paired_text_batches=len(result["text_batches"]))


def main(argv=None):
    # This must precede torch.cuda, native imports and distributed initialization.
    selected_gpu = pin_visible_gpu()
    import torch
    import torch.distributed as dist
    from . import train as base
    from . import sample_history as history
    from .distributed import synchronize_gradients

    parser = base.make_parser()
    parser.description = __doc__
    parser.add_argument("--rollout-batch-size", type=int, choices=(1, 2), default=1,
                        help="Concurrent text rollout lanes per GPU; question group size remains eight")
    parser.add_argument("--no-overlap-judge", action="store_true",
                        help="Wait until all eight samples finish before judging (timing baseline)")
    parser.add_argument("--gradient-bucket-mb", type=float, default=16.0)
    parser.add_argument("--distributed-timeout-minutes", type=float, default=60.0)
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--resume-audit-directory", action="append", default=[],
                        help="Additional sample_history directory from a failed prior run; repeatable, requires --resume")
    args = parser.parse_args(argv)
    if args.resume_audit_directory and not args.resume:
        raise ValueError("--resume-audit-directory requires --resume")
    cfg = json.loads(Path(args.preview_config).read_text())
    recipe = json.loads(Path(args.recipe).read_text())
    text_backend = recipe.get("text_likelihood_backend", base.TEXT_LIKELIHOOD_BACKEND)
    if text_backend not in (base.TEXT_LIKELIHOOD_BACKEND, base.PARALLEL_TEXT_LIKELIHOOD_BACKEND):
        raise ValueError("Unsupported experimental text likelihood backend")
    if text_backend == base.PARALLEL_TEXT_LIKELIHOOD_BACKEND and args.rollout_batch_size != 2:
        raise ValueError("Parallel old/new text replay requires --rollout-batch-size 2")
    examples = json.loads(Path(args.manifest).read_text())
    if not 0 < args.max_groups <= len(examples):
        raise ValueError("max-groups must select a nonempty manifest prefix")
    if args.max_updates is not None and args.max_updates < 1:
        raise ValueError("max-updates must be positive")
    if args.judge_workers < 1 or args.cpu_threads < 1:
        raise ValueError("Worker/thread counts must be positive")
    if not math.isfinite(args.gradient_bucket_mb) or args.gradient_bucket_mb <= 0:
        raise ValueError("gradient-bucket-mb must be finite and positive")
    if not math.isfinite(args.distributed_timeout_minutes) or args.distributed_timeout_minutes <= 0:
        raise ValueError("distributed-timeout-minutes must be finite and positive")
    base.require_no_cfg(cfg["native_settings"], "Distributed training configuration")
    if cfg["group_size"] != 8 or cfg["sde_window_size"] != 3:
        raise ValueError("Distributed integration preserves N=8/window=3")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Each distributed worker must see exactly one GPU")
    torch.cuda.set_device(0)
    torch.set_num_threads(args.cpu_threads)
    timeout = timedelta(minutes=args.distributed_timeout_minutes)
    dist.init_process_group("nccl", timeout=timeout)
    control = dist.new_group(backend="gloo", timeout=timeout)
    rank, world = dist.get_rank(), dist.get_world_size()
    root = Path(args.output)
    coordinator = Coordinator(root, rank, world, control, dist, base.atomic_json)
    optimizer = None
    sample_attempt = None
    summary = dict(status="running", phase="initializing", started_at_unix=time.time(),
                   world_size=world, groups=[], waves=[], optimizer_updates=0,
                   max_groups=args.max_groups, resume=args.resume, target_reached=False,
                   groups_attempted_this_run=0, unscorable_groups_skipped=0)
    try:
        coordinator.phase("create_output", lambda: root.mkdir(parents=True, exist_ok=False) if rank == 0 else None)
        coordinator.ready, coordinator.summary = True, summary
        device = dict(host=socket.gethostname(), visible_gpu=selected_gpu,
                      gpu_uuid=str(getattr(torch.cuda.get_device_properties(0), "uuid", selected_gpu)),
                      gpu_name=torch.cuda.get_device_name(0))
        devices = coordinator.gather(device)
        if len({(item["host"], item["gpu_uuid"]) for item in devices}) != world:
            raise ValueError("Two ranks were assigned the same physical GPU")
        coordinator.same("input configuration", base._state_digest((cfg, recipe, examples)))
        manifest_digest = history.digest(examples)
        run_id = coordinator.gather(uuid.uuid4().hex if rank == 0 else None)[0]

        def prepare_judge():
            value = base.make_judge(args, audit_directory=(root / "judge_rpc" / f"rank_{rank:05d}")
                                    if args.audit_judge_rpc else None)
            value.verify_model()
            base.validate_judge_recipe(recipe, value, text_likelihood_backend=text_backend)
            return value
        judge = coordinator.phase("verifying_judge", prepare_judge)
        identity = base.judge_identity(judge)
        coordinator.same("judge identity", identity)
        configuration = dict(rollout=cfg, recipe=recipe, examples=examples,
            source_hashes=base.source_hashes(), judge=identity, judge_workers=args.judge_workers,
            judge_timeout=args.judge_timeout, judge_observation_mode=args.judge_observation_mode,
            skip_unscorable_groups=args.skip_unscorable_groups,
            sample_history=dict(schema_version=history.SCHEMA_VERSION, manifest_digest=manifest_digest),
            distributed=dict(world_size=world, question_group_size=8, questions_per_wave=world,
                rollouts_per_full_wave=8 * world, normalization="within_question_N8",
                gradient_reduction="mean_of_valid_question_objectives_including_flat_groups",
                gradient_bucket_bytes=int(args.gradient_bucket_mb * 2**20),
                overlap_judge=not args.no_overlap_judge, cpu_threads=args.cpu_threads))
        paired_configuration = None
        if args.rollout_batch_size == 2:
            from .paired_rollout import ROLLOUT_EXECUTION, make_experiment_configuration
            configuration["rollout_execution"] = dict(ROLLOUT_EXECUTION)
            paired_configuration = make_experiment_configuration(base, cfg, recipe,
                base._state_digest(configuration))
            paired_configuration["optimizer_or_judge_enabled"] = True
            # The sampler still records native incremental probabilities. Its
            # artifact field remains truthful and distinct from trial training.
            paired_configuration["training_text_likelihood_backend"] = text_backend
            paired_configuration["training_text_denominator_source"] = (
                "parallel_old_policy_snapshot" if text_backend == base.PARALLEL_TEXT_LIKELIHOOD_BACKEND
                else "recorded_sampling")
            coordinator.same("paired generation configuration", base._state_digest(paired_configuration))
        coordinator.same("resolved configuration", base._state_digest(configuration))
        coordinator.phase("writing_configuration", lambda: base.atomic_json(root / "resolved_config.json",
            {**configuration, "invocation": vars(args), "devices": devices}) if rank == 0 else None)

        def initialize():
            base.seed_everything(cfg["seed"])
            loaded = base.load_native(cfg["source"], cfg["checkpoint"])
            groups = base.attach_lora(loaded[0], **recipe["lora"])
            opt = recipe["optimizer"]
            optimizer = torch.optim.AdamW([
                {"params": groups["image"], "lr": opt["image_learning_rate"]},
                {"params": groups["text"], "lr": opt["text_learning_rate"]}],
                weight_decay=opt["weight_decay"])
            recorder = base.recorded_inferencer(loaded[-1])(*loaded[:-1])
            replay = loaded[-1](*loaded[:-1])
            trainer_class = base.NativeGRPOTrainer
            if text_backend == base.PARALLEL_TEXT_LIKELIHOOD_BACKEND:
                from .parallel_trainer import ParallelGRPOTrainer
                trainer_class = ParallelGRPOTrainer
            elif args.rollout_batch_size == 2:
                from .paired_trainer import PairedGRPOTrainer
                trainer_class = PairedGRPOTrainer
            trainer = trainer_class(replay, optimizer, base.GRPOConfig(**recipe["grpo"]))
            return loaded, optimizer, recorder, trainer
        loaded, optimizer, recorder, trainer = coordinator.phase("loading_model_and_adapters", initialize)
        model = loaded[0]

        def state_identity():
            return dict(policy_digest=base.adapter_digest(model),
                        optimizer_digest=base._state_digest(optimizer.state_dict()))
        initial_identity = coordinator.phase("checking_initial_state", state_identity)
        coordinator.same("initial adapters/optimizer", initial_identity)
        start_group, optimizer_updates, wave = 0, 0, 0
        sample_history = history.new_history(manifest_digest)
        if args.resume:
            state = coordinator.phase("restoring_checkpoint", lambda: _restore(torch, base, args.resume,
                model, optimizer, configuration, rank, world, include_sample_history=True))
            stored_history = state.pop("sample_history")
            coordinator.same("restored checkpoint", state)
            start_group, optimizer_updates, wave = state["group"] + 1, state["optimizer_updates"], state["wave"] + 1
            initial_identity = {key: state[key] for key in ("policy_digest", "optimizer_digest")}
            imported = coordinator.phase("restoring_sample_history", lambda: history.resume_history(
                stored_history, examples, Path(args.resume).resolve().parent.parent / "sample_history",
                additional_audit_directories=args.resume_audit_directory)
                if rank == 0 else None)
            sample_history = coordinator.gather(imported)[0]
        coordinator.same("sample history", history.digest(sample_history))
        coordinator.phase("writing_sample_history", lambda: history.write_history(root, sample_history)
                          if rank == 0 else None)
        max_updates = args.max_updates if args.max_updates is not None else (
            optimizer_updates + math.ceil(max(0, args.max_groups - start_group) / world))
        summary.update(target_optimizer_updates=max_updates, optimizer_updates=optimizer_updates,
                       first_group=start_group, initial_state=initial_identity)
        coordinator.phase("writing_initial_state", lambda: base.atomic_json(root / "initial_state.json",
            dict(**initial_identity, first_group=start_group, optimizer_updates=optimizer_updates,
                 world_size=world, devices=devices, configuration_digest=base._state_digest(configuration),
                 verified_identical_across_ranks=True, resume=args.resume, audit_run_id=run_id,
                 sample_history_digest=history.digest(sample_history))) if rank == 0 else None)

        for first_group in range(start_group, args.max_groups, world):
            if optimizer_updates >= max_updates:
                break
            dist.barrier(group=control)
            wave_started = time.monotonic()
            group = first_group + rank
            assigned = group < args.max_groups
            sample_attempt = None
            summary.update(active_wave=wave, active_group_range=[first_group, min(first_group + world, args.max_groups)])
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            before = coordinator.phase("checking_sampling_policy", state_identity)
            coordinator.same("sampling policy/optimizer", before)
            local = dict(rank=rank, group=group if assigned else None, assigned=assigned,
                         accepted=False, gradients_ready=False, policy_digest_before=before["policy_digest"])
            def replay_progress(event):
                coordinator.status("replay_backward", group=group, **event)
                print(json.dumps(dict(event="replay_progress",rank=rank,group=group,**event)),flush=True)
            trainer.progress_callback = replay_progress

            def do_local_work():
                nonlocal sample_attempt
                if not assigned:
                    return
                example = examples[group]
                local["example_id"] = example.get("example_id", example["task"])
                sample_attempt = history.SampleAttempt(root, run_id=run_id, manifest_digest=manifest_digest,
                    identity=history.example_identity(examples, group), seed=cfg["seed"], rank=rank,
                    world_size=world, policy_digest=before["policy_digest"],
                    prior_optimizer_updates=optimizer_updates)
                local["sample_attempt_id"] = sample_attempt.entry["attempt_id"]
                began = time.monotonic()
                try:
                    folders, judgments, timings = _sample_and_judge(base, loaded, recorder, judge, example,
                        cfg, root, group, optimizer_updates, before["policy_digest"], args.judge_workers,
                        not args.no_overlap_judge, coordinator, sample_attempt=sample_attempt,
                        paired_configuration=paired_configuration)
                except UnscorableJudgeGroup as error:
                    if not args.skip_unscorable_groups:
                        raise
                    local.update(skip_reason="unscorable_judge_group", error=str(error),
                                 sampling_and_judging_seconds=time.monotonic() - began)
                    sample_attempt.progress("unscorable")
                    base.atomic_json(root / f"group_{group:06d}" / "group_skip.json", local)
                    return
                local.update(timings, accepted=True)
                sample_attempt.progress("replay_backward")
                coordinator.status("replay_backward", group=group)
                replay_started = time.monotonic()
                metrics = trainer.update(folders, judgments, optimize=True, defer_step=True)
                local.update(metrics=metrics, gradients_ready=bool(metrics.get("gradients_ready")),
                             replay_backward_seconds=time.monotonic() - replay_started)
                sample_attempt.progress("gradients_ready" if local["gradients_ready"] else "flat")

            def local_work():
                began = time.monotonic()
                try:
                    do_local_work()
                finally:
                    local["local_work_seconds"] = time.monotonic() - began
                    local["_work_finished_at"] = time.monotonic()
                coordinator.status("waiting_for_peers", wave=wave,
                    group=group if assigned else None, accepted=local["accepted"],
                    gradients_ready=local["gradients_ready"])

            coordinator.phase("sampling_judging_and_replay", local_work, details={"wave": wave, "group": group if assigned else None})
            local["readiness_wait_seconds"] = time.monotonic() - local.pop("_work_finished_at")
            readiness = coordinator.gather({key: local[key] for key in ("rank", "assigned", "accepted", "gradients_ready")})
            denominator = sum(item["accepted"] for item in readiness)
            any_gradients = any(item["gradients_ready"] for item in readiness)
            sync_metrics, finish_metrics = {}, dict(optimized=False, gradient_norm_before_clip=0.)
            sync_seconds, step_seconds = 0., 0.
            if any_gradients:
                if denominator < 1:
                    raise ValueError("Gradients exist without any accepted question")
                started = time.monotonic()
                # All ranks have voted success, including inactive/flat ranks.
                sync_metrics = synchronize_gradients(model, denominator=denominator,
                    bucket_bytes=int(args.gradient_bucket_mb * 2**20))
                torch.cuda.synchronize()
                sync_seconds = time.monotonic() - started
                started = time.monotonic()
                finish_metrics = coordinator.phase("optimizer_step", lambda: trainer.finish_update(optimize=True))
                torch.cuda.synchronize()
                step_seconds = time.monotonic() - started
                optimizer_updates += 1
            else:
                optimizer.zero_grad(set_to_none=True)
                finish_metrics["skip_reason"] = "all_groups_flat_or_unscorable"
            after = coordinator.phase("checking_synchronized_update", state_identity)
            states = coordinator.same("updated adapters/optimizer", after)
            local.update(optimized=any_gradients, optimizer_updates=optimizer_updates,
                policy_digest_after=after["policy_digest"], gradient_synchronization=sync_metrics,
                gradient_sync_seconds=sync_seconds, optimizer_step_seconds=step_seconds,
                optimizer_metrics=finish_metrics,
                peak_cuda_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                peak_cuda_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
            if assigned:
                coordinator.phase("writing_group_metrics", lambda: base.atomic_json(root / f"group_{group:06d}" / "update.json", local))
            else:
                coordinator.phase("writing_group_metrics", lambda: None)
            wave_folder = root / f"wave_{wave:06d}"
            coordinator.phase("creating_wave_directory", lambda: wave_folder.mkdir(parents=True, exist_ok=False) if rank == 0 else None)
            checkpoint = wave_folder / "checkpoint.pt"
            checkpoint_started = time.monotonic()
            entry = coordinator.phase("preparing_sample_commit", lambda: sample_attempt.for_checkpoint(
                checkpoint=checkpoint, wave=wave, optimizer_updates=optimizer_updates,
                global_step_applied=any_gradients, accepted=local["accepted"],
                contributed_gradients=local["gradients_ready"], **after) if sample_attempt is not None else None)
            entries = coordinator.gather(entry)
            next_history = coordinator.phase("merging_sample_history", lambda: history.add_attempts(sample_history, entries))
            identity = _save(torch, base, coordinator, checkpoint, model, optimizer, configuration,
                             last_group=min(first_group + world, args.max_groups) - 1,
                             optimizer_updates=optimizer_updates, wave=wave, sample_history=next_history)
            sample_history = next_history
            def publish_history():
                if sample_attempt is not None:
                    sample_attempt.accept_checkpoint(entry)
                if rank == 0:
                    history.write_history(root, sample_history)
            coordinator.phase("publishing_sample_history", publish_history)
            if args.verify_checkpoint_reload:
                reloaded = coordinator.phase("checkpoint_reload", lambda: _restore(torch, base, checkpoint,
                    model, optimizer, configuration, rank, world))
                coordinator.same("checkpoint reload", reloaded)
                coordinator.phase("writing_checkpoint_validation", lambda: base.atomic_json(
                    wave_folder / "checkpoint_validation.json", dict(reload_passed=True,
                    per_rank_rng_restored=True, optimizer_state_exact=True, world_size=world, **identity)) if rank == 0 else None)
            checkpoint_seconds = time.monotonic() - checkpoint_started
            workers = coordinator.gather(local)
            record = dict(wave=wave, first_group=first_group, last_group=min(first_group + world, args.max_groups) - 1,
                accepted_question_count=denominator, gradient_question_count=sum(item["gradients_ready"] for item in readiness),
                optimized=any_gradients, optimizer_updates=optimizer_updates, workers=workers,
                policy_digest_before=before["policy_digest"], **identity,
                adapters_and_optimizer_identical_across_ranks=True, checkpoint=str(checkpoint),
                sample_history_digest=history.digest(sample_history), sample_history_counts=history.history_counts(sample_history),
                checkpoint_reload_verified=args.verify_checkpoint_reload,
                checkpoint_seconds=checkpoint_seconds, wall_seconds=time.monotonic() - wave_started)
            def write_wave():
                if rank != 0:
                    return
                base.atomic_json(wave_folder / "update.json", record)
                summary["waves"].append({key: record[key] for key in ("wave", "first_group", "last_group",
                    "accepted_question_count", "gradient_question_count", "optimized", "optimizer_updates",
                    "policy_digest", "optimizer_digest", "checkpoint", "wall_seconds")})
                for item in workers:
                    if item["assigned"]:
                        summary["groups"].append({key: item.get(key) for key in
                            ("group", "rank", "example_id", "accepted", "gradients_ready", "skip_reason")})
                summary.update(optimizer_updates=optimizer_updates,
                    sample_history_counts=record["sample_history_counts"],
                    groups_attempted_this_run=len(summary["groups"]),
                    unscorable_groups_skipped=sum(item.get("skip_reason") == "unscorable_judge_group" for item in summary["groups"]))
                base.atomic_json(root / "summary.json", summary)
                print(json.dumps(dict(event="wave_complete", wave=wave, optimizer_updates=optimizer_updates,
                    accepted_questions=denominator, optimized=any_gradients, wall_seconds=record["wall_seconds"],
                    synchronized=True, checkpoint=str(checkpoint))), flush=True)
            coordinator.phase("writing_wave_summary", write_wave)
            wave += 1
        def complete():
            if rank == 0:
                summary.update(status="completed", phase="finished", finished_at_unix=time.time(),
                    optimizer_updates=optimizer_updates, target_reached=optimizer_updates >= max_updates,
                    stop_reason="target_updates_reached" if optimizer_updates >= max_updates else "max_groups_exhausted")
                base.atomic_json(root / "summary.json", summary)
        coordinator.phase("finalizing", complete)
    except BaseException as error:
        if sample_attempt is not None:
            try:
                sample_attempt.fail(error)
            except Exception:
                traceback.print_exc()  # Preserve the original training/collective error.
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        if rank == 0 and coordinator.ready:
            summary.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                           finished_at_unix=time.time(), error={"type": type(error).__name__, "message": str(error)})
            base.atomic_json(root / "summary.json", summary)
        raise
    finally:
        # torchrun terminates peers if a rank disappears; no barrier on failure.
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
