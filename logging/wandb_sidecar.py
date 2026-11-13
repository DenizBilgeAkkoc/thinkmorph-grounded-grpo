#!/usr/bin/env python3
"""Online CPU-only W&B reader for the fresh parallel ThinkMorph run.

Training files are read-only. Writes are confined to this run's wandb_live
directory and W&B_Status.json. This process imports no Torch/native model,
starts no GPU work, runs no SSH commands and never uploads code or weights.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time
import uuid

from wandb_metrics import aggregate_wave, numeric
from wandb_media import build_media


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def retry_read(operation, *, optional=False):
    """Retry stale NFS handles, atomic-rename races and transient partial JSON."""
    error = None
    for attempt in range(4):
        try:
            return operation()
        except (OSError, json.JSONDecodeError) as current:
            error = current
            transient = isinstance(current, json.JSONDecodeError) or getattr(current, "errno", None) in (errno.ESTALE, errno.ENOENT, errno.EINTR)
            if not transient:
                raise
            if attempt < 3:
                time.sleep(.05 * 2**attempt)
    if optional and isinstance(error, FileNotFoundError):
        return None
    raise error


def read_json(path, optional=False):
    return retry_read(lambda: json.loads(Path(path).read_text()), optional=optional)


def read_image(path, side):
    from PIL import Image
    def load():
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((side, side), Image.Resampling.LANCZOS)
            return image.copy()
    return retry_read(load)


def segments(root):
    document = read_json(root/"Segments.json", optional=True)
    if document is None:
        return [], None
    found = []
    for item in document["segments"]:
        path = Path(item["path"]).resolve()
        if root.resolve() not in path.parents:
            raise ValueError("Segment must belong to this fresh run root")
        found.append(dict(item, path=path))
    return found, document


def wave_identity(wave):
    return {name: wave[name] for name in ("wave", "first_group", "last_group", "optimizer_updates",
        "policy_digest", "optimizer_digest", "sample_history_digest")}


def committed_waves(root):
    result = {}
    listed, _ = segments(root)
    for segment in listed:
        for path in sorted(segment["path"].glob("wave_*/update.json")):
            marker = read_json(path.parent/"checkpoint.pt.complete.json", optional=True)
            validation = read_json(path.parent/"checkpoint_validation.json", optional=True)
            if marker is None or validation is None:
                continue
            wave = read_json(path)
            if wave["first_group"] < segment["start_group"] or wave["last_group"] >= segment["stop_group"]:
                continue
            for name in ("policy_digest", "optimizer_digest", "optimizer_updates", "sample_history_digest"):
                if wave[name] != marker[name]:
                    raise ValueError("Committed wave/marker identity mismatch")
            if marker["group"] != wave["last_group"] or marker["wave"] != wave["wave"]:
                raise ValueError("Committed wave indexes disagree")
            if not all(validation.get(key) is True for key in ("reload_passed", "per_rank_rng_restored", "optimizer_state_exact")):
                raise ValueError("Checkpoint reload proof is incomplete")
            if any(validation.get(key) != marker[key] for key in ("policy_digest", "optimizer_digest")):
                raise ValueError("Checkpoint reload identity differs")
            key = wave["wave"]
            if key in result and wave_identity(result[key]["wave"]) != wave_identity(wave):
                raise ValueError("Conflicting committed global wave across segments")
            result.setdefault(key, dict(wave=wave, segment=segment, path=path))
    return [result[key] for key in sorted(result)]


class Ledger:
    """Single-writer durable history step and global-wave cursor.

    A pending event is persisted before run.log. Reconnection reconciles it with
    the SDK's resumed history step; otherwise the same explicit step is retried.
    SDK return is acceptance into its transaction stream, not server ACK. NFS
    W&B transaction files are retained for recovery; root verifies remote media.
    """
    def __init__(self, directory, project, entity, requested_id=None):
        self.path = directory/"Wandb_Run_State.json"
        self.state = read_json(self.path, optional=True)
        if self.state is None:
            self.state = dict(schema_version=2, project=project, entity=entity,
                run_id=requested_id or uuid.uuid4().hex[:12], created_at_utc=utc(),
                completed_waves={}, next_history_step=0, pending=None, heartbeats=0)
            self.save()
        if self.state["project"] != project or self.state["entity"] != entity:
            raise ValueError("Existing sidecar cursor belongs to another W&B project/entity")
        if requested_id and self.state["run_id"] != requested_id:
            raise ValueError("Requested W&B run ID differs from the durable fresh-run cursor")

    def save(self):
        atomic_json(self.path, self.state)

    def needs_wave(self, wave):
        previous = self.state["completed_waves"].get(str(wave["wave"]))
        if previous is not None and previous["identity"] != wave_identity(wave):
            raise ValueError("A previously logged committed wave changed")
        return previous is None

    def begin(self, kind, identity, sdk_step):
        pending = self.state["pending"]
        if pending is not None:
            if pending["kind"] != kind or pending["identity"] != identity:
                raise ValueError("Pending W&B event must be resolved before logging another event")
            return pending["history_step"]
        step = max(self.state["next_history_step"], int(sdk_step))
        self.state["pending"] = dict(kind=kind, identity=identity, history_step=step)
        self.save()
        return step

    def accept(self, metadata=None):
        pending = self.state["pending"]
        if pending is None:
            return
        if pending["kind"] == "wave":
            self.state["completed_waves"][str(pending["identity"]["wave"])] = dict(
                identity=pending["identity"], history_step=pending["history_step"],
                accepted_at_utc=utc(), media=metadata or {})
        self.state["next_history_step"] = pending["history_step"] + 1
        self.state["pending"] = None
        self.save()

    def reconcile(self, sdk_step):
        pending = self.state["pending"]
        if pending and int(sdk_step) > pending["history_step"]:
            self.accept({"reconciled_from_resumed_history_step": True})


class Heartbeats:
    def __init__(self):
        self.cache = {}

    def collect(self, root, controller, sequence, elapsed):
        listed, _ = segments(root)
        output = {"live/elapsed_seconds": elapsed, "live/heartbeat": sequence,
            "live/controller_status": controller.get("status", "unknown"),
            "live/controller_phase": controller.get("phase", "unknown")}
        for segment in listed:
            for path in (segment["path"]/"sample_history").glob("attempt_*.json"):
                stamp = retry_read(lambda: (path.stat().st_mtime_ns, path.stat().st_size))
                old = self.cache.get(str(path))
                if old is None or old[0] != stamp:
                    self.cache[str(path)] = (stamp, read_json(path))
        entries = [item[1] for item in self.cache.values()]
        output.update({"live/questions_sampling_started": len({e["identity"]["manifest_index"] for e in entries if e.get("rollouts")}),
            "live/rollout_attempts_started": sum(len(e.get("rollouts", [])) for e in entries),
            "live/rollout_attempts_completed": sum(r.get("status") == "sampled" for e in entries for r in e.get("rollouts", [])),
            "live/committed_questions": controller.get("committed_questions", 0),
            "live/optimizer_updates": controller.get("optimizer_updates", 0)})
        current = (Path(controller["active_phase_directory"]) if controller.get("active_phase_directory")
                   else listed[-1]["path"] if listed else None)
        if current is not None and root.resolve() not in current.resolve().parents:
            raise ValueError("Live run directory escapes the fresh run root")
        phases = Counter()
        for path in sorted(current.glob("ranks/rank_*/status.json")) if current is not None else []:
            value = read_json(path); rank = value.get("rank")
            if rank is None:
                continue
            phase = value.get("phase", "unknown"); phases[phase] += 1
            output[f"live/rank/{rank}/phase"] = phase
            for name in ("group", "rollouts_completed", "completed_text_actions", "total_text_actions",
                         "completed_image_transitions", "sampling_replay_delta", "action_seconds"):
                if numeric(value.get(name)):
                    output[f"live/rank/{rank}/{name}"] = value[name]
            if numeric(value.get("updated_at_unix")):
                output[f"live/rank/{rank}/status_age_seconds"] = max(0, time.time()-value["updated_at_unix"])
        output.update({"live/ranks_in_phase/"+phase: count for phase,count in phases.items()})
        return output


class Sidecar:
    def __init__(self, args, wandb_module=None):
        self.args = args; self.root = args.run_root.resolve()
        self.directory = self.root/"wandb_live"; self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory/"sidecar.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.ledger = Ledger(self.directory, args.project, args.entity, args.run_id)
        self.run = None; self.wandb = wandb_module; self.secret = None; self.stop_requested = False
        self.heartbeats = Heartbeats()
        manifest = read_json(args.manifest)
        if not isinstance(manifest, list):
            raise ValueError("The fixed training manifest must be a JSON list")
        self.examples = dict(enumerate(manifest))
        self.manifest_digest = hashlib.sha256(retry_read(args.manifest.read_bytes)).hexdigest()
        self.status = dict(status="starting", mode="online", run_id=self.ledger.state["run_id"],
            project=args.project, entity=args.entity, sidecar_pid=os.getpid(), training_process_unchanged=True,
            media_policy="All question input thumbnails and rollout summaries; one rotating N8 question's generated steps per committed wave")
        self.publish()

    def publish(self, **updates):
        self.status.update(updates, updated_at_utc=utc(),
                           logged_committed_waves=len(self.ledger.state["completed_waves"]))
        atomic_json(self.root/"W&B_Status.json", self.status)

    def error_text(self, error):
        message = str(error)
        if self.secret:
            message = message.replace(self.secret, "[REDACTED]")
        return dict(type=type(error).__name__, message=message[:1500])

    def connect(self):
        if self.wandb is None:
            import wandb
            self.wandb = wandb
        self.secret = retry_read(self.args.key_file.read_text).strip()
        if not self.secret or any(character.isspace() for character in self.secret):
            raise ValueError("W&B key file must contain one nonempty token")
        settings = self.wandb.Settings(api_key=self.secret, mode="online", init_timeout=60, login_timeout=30,
            console="off", disable_code=True, disable_git=True, disable_job_creation=True,
            x_disable_meta=True, x_disable_stats=True, x_disable_machine_info=True, x_save_requirements=False,
            quiet=True, silent=True, summary_timeout=30, x_service_wait=30,
            x_graphql_retry_max=3, x_graphql_timeout_seconds=20,
            x_file_stream_retry_max=3, x_file_stream_timeout_seconds=20,
            x_file_transfer_retry_max=3, x_file_transfer_timeout_seconds=20)
        self.run = self.wandb.init(project=self.args.project, entity=self.args.entity,
            id=self.ledger.state["run_id"], name=self.args.name, resume="allow", mode="online",
            dir=str(self.directory), save_code=False, settings=settings,
            config=dict(fresh_training_run=segments(self.root)[0][0]['start_group']==0,
                initialization="native_base_or_explicit_full_checkpoint",
                starting_manifest_index=segments(self.root)[0][0]['start_group'],
                execution_profile=read_json(self.root/'config/run_contract.json')['profile'],
                text_likelihood_backend="parallel_full_sequence_consistent_old_new",
                text_denominator_source="parallel_old_policy_snapshot",
                manifest_sha256=self.manifest_digest,
                target_questions=read_json(self.root/'Controller_Status.json')['target_questions'],
                rollouts_per_question=8, logging_method="independent_committed_artifact_reader",
                media_image_side=self.args.image_side, media_input_side=self.args.input_side,
                media_max_generated_images=self.args.max_generated_images, media_text_limit=self.args.text_limit,
                media_question_policy="one_complete_N8_question_per_wave_rotating_rank",
                media_all_question_inputs=True, media_all_rollout_summaries=True,
                uploads_code=False, uploads_weights=False,
                scalar_step_axis="actual_optimizer_updates; global_wave also recorded",
                objective_reduction="mean_normalized_N8_question_objective_including_flat_accepted_questions"))
        if self.run is None or self.run.settings.mode != "online":
            raise RuntimeError("W&B did not initialize in online mode")
        self.run.define_metric("train/global_step")
        for namespace in ("train", "reward", "timing", "memory", "rank", "optimizer", "fidelity", "diagnostic", "integrity", "task", "media", "samples"):
            self.run.define_metric(namespace+"/*", step_metric="train/global_step", step_sync=False)
        self.run.define_metric("live/elapsed_seconds")
        self.run.define_metric("live/*", step_metric="live/elapsed_seconds", step_sync=False)
        self.ledger.reconcile(getattr(self.run, "step", 0))
        self.ledger.state["url"] = self.run.url; self.ledger.save()
        self.publish(status="online", url=self.run.url, connected_at_utc=utc())

    def sync_waves(self):
        _, document = segments(self.root)
        if document is not None and document.get("fixed_manifest_sha256", self.manifest_digest) != self.manifest_digest:
            raise ValueError("Logger manifest differs from the controller's fixed manifest")
        pending = self.ledger.state["pending"]
        if pending and pending["kind"] == "heartbeat":
            # A heartbeat has no training claim; retry its reserved step first.
            self.heartbeat(read_json(self.root/"Controller_Status.json", optional=True) or {}, retry_identity=pending["identity"])
        for record in committed_waves(self.root):
            wave, segment = record["wave"], record["segment"]
            if not self.ledger.needs_wave(wave):
                continue
            configuration = read_json(segment["path"]/"resolved_config.json")
            payload = aggregate_wave(wave, configuration, self.examples)
            payload["train/segment"] = segment["id"]
            media, metadata = build_media(self.wandb, wave, segment["path"], self.examples, read_json, read_image,
                image_side=self.args.image_side, input_side=self.args.input_side,
                max_generated_images=self.args.max_generated_images, text_limit=self.args.text_limit)
            payload.update(media)
            identity = wave_identity(wave)
            step = self.ledger.begin("wave", identity, getattr(self.run, "step", 0))
            self.run.log(payload, step=step, commit=True)
            self.ledger.accept(metadata)
            self.publish(last_logged_wave=wave["wave"], last_global_step=wave["optimizer_updates"],
                last_media_counts=metadata, last_logged_history_step=step)

    def heartbeat(self, controller, retry_identity=None):
        created = self.ledger.state["created_at_utc"]
        elapsed = max(0, time.time()-datetime.fromisoformat(created).timestamp())
        if retry_identity is None:
            self.ledger.state["heartbeats"] += 1
            retry_identity = {"sequence": self.ledger.state["heartbeats"]}
        payload = self.heartbeats.collect(self.root, controller, retry_identity["sequence"], elapsed)
        step = self.ledger.begin("heartbeat", retry_identity, getattr(self.run, "step", 0))
        self.run.log(payload, step=step, commit=True); self.ledger.accept()
        self.publish(last_heartbeat_at_utc=utc(), controller_status=controller.get("status", "waiting"),
                     controller_phase=controller.get("phase", "waiting"))

    def finish(self, status):
        if self.run is not None:
            self.run.summary["controller_final_status"] = status
            self.run.summary["logged_committed_waves"] = len(self.ledger.state["completed_waves"])
            self.run.summary["logger_finished_at_utc"] = utc()
            self.run.finish(exit_code=0 if status=="completed" else 1, quiet=True); self.run = None
        self.publish(status="finished", controller_status=status)

    def execute(self):
        failures = 0
        while not self.stop_requested and failures < self.args.max_failures:
            try:
                if self.run is None:
                    self.connect()
                self.sync_waves()
                controller = read_json(self.root/"Controller_Status.json", optional=True) or {}
                self.heartbeat(controller)
                final = controller.get("status") in ("completed", "failed") or (
                    controller.get("status") == "interrupted" and not controller.get("recovery_allowed"))
                if final:
                    self.finish(controller["status"])
                    return 0 if controller["status"] == "completed" else 1
                deadline = time.monotonic()+self.args.interval_seconds
                while not self.stop_requested and time.monotonic() < deadline:
                    time.sleep(min(1, max(0, deadline-time.monotonic())))
            except Exception as error:
                failures += 1
                self.publish(status="retrying" if failures<self.args.max_failures else "failed",
                             failures=failures, error=self.error_text(error))
                if self.run is not None:
                    try:
                        self.run.finish(exit_code=1, quiet=True)
                    except Exception:
                        pass
                    self.run = None
                deadline = time.monotonic()+min(60, 5*2**(failures-1))
                while not self.stop_requested and failures<self.args.max_failures and time.monotonic()<deadline:
                    time.sleep(1)
        if self.stop_requested:
            self.finish("sidecar_interrupted_training_continues")
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--project", required=True); parser.add_argument("--entity", required=True)
    parser.add_argument("--name", default="thinkmorph-grounded-grpo")
    parser.add_argument("--run-id")
    parser.add_argument("--interval-seconds", type=float, default=15)
    parser.add_argument("--max-failures", type=int, default=10)
    parser.add_argument("--image-side", type=int, default=640)
    parser.add_argument("--input-side", type=int, default=256)
    parser.add_argument("--max-generated-images", type=int, default=64)
    parser.add_argument("--text-limit", type=int, default=12000)
    args = parser.parse_args()
    if not math.isfinite(args.interval_seconds) or args.interval_seconds < 5 or not 1<=args.max_failures<=20:
        parser.error("interval must be >=5 seconds; failure budget must be 1..20")
    if not 64<=args.input_side<=512 or not 128<=args.image_side<=1024 or not 1<=args.max_generated_images<=128 or not 1000<=args.text_limit<=50000:
        parser.error("Media bounds are outside the supported limits")
    sidecar = Sidecar(args)
    def stop(_signum, _frame):
        sidecar.stop_requested = True
    signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
    return sidecar.execute()


if __name__ == "__main__":
    raise SystemExit(main())
