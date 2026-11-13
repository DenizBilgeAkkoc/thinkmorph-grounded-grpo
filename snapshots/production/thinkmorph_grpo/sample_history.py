"""Explicit sample attempts and checkpoint-bound training history.

An attempt can be sampled without contributing gradients. Only a digest-bound
checkpoint marks an optimizer contribution committed; live audit files alone do
not establish training provenance when resuming an earlier checkpoint.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import time

from .bridge import atomic_json

SCHEMA_VERSION = 1
SOURCE_FIELDS = ("repo", "revision", "dataset", "source_dataset", "split", "source_split",
                 "row_index", "row_id", "source_row_index", "pid", "id", "content_sha256",
                 "source_row_sha256", "input_sha256", "source_parquet")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def example_identity(examples, index):
    example = examples[index]
    image = example["problem_image"]
    actual = file_digest(image["path"])
    if image.get("sha256") is not None and image["sha256"] != actual:
        raise ValueError(f"Manifest image SHA256 mismatch at index {index}")
    source = {key: deepcopy(example[key]) for key in SOURCE_FIELDS if key in example}
    metadata = example.get("metadata", {})
    if isinstance(metadata, dict):
        source.update({"metadata." + key: deepcopy(metadata[key]) for key in SOURCE_FIELDS if key in metadata})
    return dict(manifest_index=index, example_id=example.get("example_id", example.get("task")),
                task=example.get("task"), source=source, image_path=image["path"], image_sha256=actual,
                declared_pixel_sha256=image.get("pixel_sha256"),
                source_encoded_sha256=image.get("source_encoded_sha256"),
                question_sha256=hashlib.sha256(example["question"].encode()).hexdigest())


def new_history(manifest_digest):
    return dict(schema_version=SCHEMA_VERSION, manifest_digest=manifest_digest, attempts=[])


def validate_history(history, *, manifest_digest=None, examples=None):
    if not isinstance(history, dict) or history.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported sample-history schema")
    if manifest_digest is not None and history.get("manifest_digest") != manifest_digest:
        raise ValueError("Sample history belongs to a different manifest")
    attempts = history.get("attempts")
    if not isinstance(attempts, list):
        raise ValueError("Sample history must contain a list of attempts")
    seen, identities = set(), {}
    for entry in attempts:
        attempt_id = entry.get("attempt_id")
        if not isinstance(attempt_id, str) or not attempt_id or attempt_id in seen:
            raise ValueError("Missing or duplicated sample attempt ID")
        seen.add(attempt_id)
        if entry.get("manifest_digest") != history["manifest_digest"]:
            raise ValueError("Sample attempt manifest mismatch")
        identity = entry.get("identity", {})
        index = identity.get("manifest_index")
        if type(index) is not int or index < 0:
            raise ValueError("Invalid sample manifest index")
        seeds = entry.get("planned_seeds")
        if not isinstance(seeds, list) or len(seeds) != 8 or any(type(seed) is not int for seed in seeds):
            raise ValueError("Sample history requires all eight planned seeds")
        rollouts = entry.get("rollouts")
        if not isinstance(rollouts, list) or len(rollouts) > 8:
            raise ValueError("Invalid sample rollout history")
        for number, rollout in enumerate(rollouts):
            if rollout.get("rollout") != number or rollout.get("seed") != seeds[number]:
                raise ValueError("Sample rollout order/seed mismatch")
            if rollout.get("status") not in ("sampling", "sampled"):
                raise ValueError("Unknown sample rollout status")
            if rollout["status"] == "sampled" and not all(rollout.get(key) for key in
                    ("trace_sha256", "sampling_sha256", "artifact_folder")):
                raise ValueError("Completed rollout is missing artifact provenance")
        committed = entry.get("optimizer_commit")
        checkpoint_commit = entry.get("checkpoint_commit")
        if checkpoint_commit is not None:
            included = bool(checkpoint_commit.get("included_in_objective"))
            contributed = bool(checkpoint_commit.get("contributed_gradients"))
            expected_outcome = "trained" if contributed else "flat" if included else "unscorable"
            if entry.get("outcome") != expected_outcome:
                raise ValueError("Checkpoint outcome does not match its objective contribution")
            if len(rollouts) != 8 or any(r["status"] != "sampled" for r in rollouts):
                raise ValueError("Incomplete group cannot be checkpoint committed")
        if committed is not None:
            if entry.get("outcome") != "trained" or entry.get("checkpoint_commit") != committed:
                raise ValueError("Optimizer contribution lacks a matching checkpoint commit")
            if not committed.get("global_step_applied") or not committed.get("contributed_gradients"):
                raise ValueError("Invalid optimizer-contribution claim")
            if len(rollouts) != 8 or any(r["status"] != "sampled" for r in rollouts):
                raise ValueError("Incomplete group cannot be marked trained")
        elif entry.get("outcome") == "trained":
            raise ValueError("Trained sample lacks optimizer commit")
        if examples is not None:
            if index >= len(examples):
                raise ValueError("History index is outside current manifest")
            if index not in identities:
                identities[index] = example_identity(examples, index)
            if identity != identities[index]:
                raise ValueError(f"Sample identity/content changed at manifest index {index}")
    return history


def add_attempts(history, entries):
    result = deepcopy(history)
    known = {entry["attempt_id"] for entry in result["attempts"]}
    for entry in entries:
        if entry is None:
            continue
        if entry["attempt_id"] in known:
            raise ValueError("Attempt already exists in checkpoint history")
        known.add(entry["attempt_id"])
        result["attempts"].append(deepcopy(entry))
    return validate_history(result)


def history_counts(history):
    entries = history["attempts"]
    return dict(attempts=len(entries),
                questions_with_sampling_started=len({e["identity"]["manifest_index"] for e in entries if e["rollouts"]}),
                sampled_rollouts=sum(r["status"] == "sampled" for e in entries for r in e["rollouts"]),
                gradient_contributing_attempts=sum(e.get("optimizer_commit") is not None for e in entries),
                flat_attempts=sum(e.get("outcome") == "flat" for e in entries),
                unscorable_attempts=sum(e.get("outcome") == "unscorable" for e in entries),
                uncommitted_attempts=sum(e.get("checkpoint_commit") is None for e in entries))


def write_history(root, history):
    validate_history(history)
    path = Path(root) / "sample_history.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, history)
    atomic_json(Path(root) / "sample_history_summary.json", history_counts(history))


class SampleAttempt:
    def __init__(self, root, *, run_id, manifest_digest, identity, seed, rank, world_size,
                 policy_digest, prior_optimizer_updates):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise ValueError("Invalid audit run ID")
        index = identity["manifest_index"]
        self.path = Path(root) / "sample_history" / f"attempt_{run_id}_{index:08d}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise ValueError("Refusing to overwrite a sample attempt")
        self.entry = dict(attempt_id=f"{run_id}:{index:08d}", run_id=run_id,
            manifest_digest=manifest_digest, identity=deepcopy(identity), source_run_root=str(Path(root).resolve()),
            rank=rank, world_size=world_size, policy_digest_before=policy_digest,
            prior_optimizer_updates=prior_optimizer_updates, created_at_unix=time.time(),
            planned_seeds=[seed + index * 1000 + n for n in range(8)], rollouts=[],
            processing_status="planned", outcome="pending", checkpoint_commit=None, optimizer_commit=None)
        self.flush()

    def flush(self):
        atomic_json(self.path, self.entry)

    def begin_rollout(self, number, windows):
        if number != len(self.entry["rollouts"]) or not 0 <= number < 8:
            raise ValueError("Rollouts must start exactly once and in order")
        self.entry["processing_status"] = "sampling"
        self.entry["sde_windows"] = deepcopy(windows)
        self.entry["rollouts"].append(dict(rollout=number, seed=self.entry["planned_seeds"][number],
                                          status="sampling", started_at_unix=time.time()))
        self.flush()

    def finish_rollout(self, number, folder):
        rollout = self.entry["rollouts"][number]
        if rollout["status"] != "sampling":
            raise ValueError("Rollout was already completed")
        folder = Path(folder)
        rollout.update(status="sampled", completed_at_unix=time.time(), artifact_folder=str(folder.resolve()),
                       trace_sha256=file_digest(folder / "trace.json"),
                       sampling_sha256=file_digest(folder / "sampling.json"))
        all_sampled = (len(self.entry["rollouts"]) == 8 and
                       all(item["status"] == "sampled" for item in self.entry["rollouts"]))
        self.entry["processing_status"] = "sampled" if all_sampled else "sampling"
        self.flush()

    def progress(self, status):
        self.entry["processing_status"] = status
        if status in ("gradients_ready", "flat", "unscorable"):
            self.entry["local_outcome"] = status
        self.flush()

    def fail(self, error):
        failure = dict(type=type(error).__name__, message=str(error), at_unix=time.time())
        if self.entry.get("checkpoint_commit") is not None:
            self.entry["post_commit_error"] = failure
        else:
            self.entry.update(outcome="failed", processing_status="failed", error=failure)
        self.flush()

    def for_checkpoint(self, *, checkpoint, wave, optimizer_updates, global_step_applied,
                       accepted, contributed_gradients, policy_digest, optimizer_digest):
        result = deepcopy(self.entry)
        commit = dict(checkpoint=str(checkpoint), wave=wave, optimizer_updates=optimizer_updates,
            global_step_applied=bool(global_step_applied), included_in_objective=bool(accepted),
            contributed_gradients=bool(contributed_gradients), policy_digest=policy_digest,
            optimizer_digest=optimizer_digest)
        if contributed_gradients and (not accepted or not global_step_applied):
            raise ValueError("Gradient contribution requires an accepted committed global step")
        result.update(checkpoint_commit=commit, optimizer_commit=commit if contributed_gradients else None,
                      outcome="trained" if contributed_gradients else "flat" if accepted else "unscorable",
                      processing_status="checkpoint_committed")
        validate_history(dict(schema_version=SCHEMA_VERSION, manifest_digest=result["manifest_digest"], attempts=[result]))
        return result

    def accept_checkpoint(self, committed_entry):
        if committed_entry["attempt_id"] != self.entry["attempt_id"]:
            raise ValueError("Checkpoint belongs to a different sample attempt")
        self.entry = deepcopy(committed_entry)
        self.flush()


def resume_history(stored, examples, source_audit_directory, *, additional_audit_directories=()):
    """Keep explicit prior attempts without promoting later-branch optimizer work.

    Additional directories preserve failed retries which never wrote their own
    checkpoint. They are operational audit inputs, not sources of policy state.
    The selected checkpoint remains authoritative for every attempt it stores.
    """
    expected = digest(examples)
    result = deepcopy(stored)
    validate_history(result, manifest_digest=expected, examples=examples)
    known = {entry["attempt_id"] for entry in result["attempts"]}
    extras = []
    additional = [Path(directory) for directory in additional_audit_directories]
    for directory in additional:
        if not directory.is_dir():
            raise ValueError("Explicit resume audit directory does not exist: " + str(directory))
    for directory in (Path(source_audit_directory), *additional):
        for path in sorted(directory.glob("attempt_*.json")):
            entry = json.loads(path.read_text())
            validate_history(dict(schema_version=SCHEMA_VERSION, manifest_digest=expected, attempts=[entry]),
                             manifest_digest=expected, examples=examples)
            if entry["attempt_id"] in known:
                continue  # Digest-bound checkpoint entries and earlier imports win.
            if entry.get("checkpoint_commit") is not None:
                entry["prior_branch_checkpoint_commit"] = entry["checkpoint_commit"]
            entry.update(checkpoint_commit=None, optimizer_commit=None, outcome="uncommitted_before_resume",
                         processing_status="preserved_from_previous_attempt", imported_audit_path=str(path))
            extras.append(entry)
            known.add(entry["attempt_id"])
    return add_attempts(result, extras)
