"""Reproducibility contracts for sampled versus checkpoint-committed questions."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from thinkmorph_grpo import sample_history as history
from thinkmorph_grpo import train as base
from thinkmorph_grpo.train_distributed import _checkpoint_paths, _restore, _save
import test_distributed_runner as runner_tests


class SampleHistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.image = self.root / "image.png"
        self.image.write_bytes(b"fixed original image content")
        self.examples = [dict(example_id=f"example-{i}", task="Visual_Search", question=f"Question {i}?",
            repo="owner/data", revision="a" * 40, split="train", row_index=100 + i, pid=f"pid-{i}",
            problem_image=dict(path=str(self.image), sha256=history.file_digest(self.image))) for i in range(3)]
        self.manifest_digest = history.digest(self.examples)

    def attempt(self, index=0, run_id="first-run", root=None):
        return history.SampleAttempt(root or self.root, run_id=run_id, manifest_digest=self.manifest_digest,
            identity=history.example_identity(self.examples, index), seed=41, rank=index, world_size=3,
            policy_digest="policy-before", prior_optimizer_updates=0)

    def finish(self, attempt, count=8):
        for number in range(count):
            attempt.begin_rollout(number, [[2, 3, 4]])
            folder = self.root / "artifacts" / attempt.entry["attempt_id"].replace(":", "_") / str(number)
            folder.mkdir(parents=True)
            (folder / "trace.json").write_text(json.dumps([{"text": "answer"}]))
            (folder / "sampling.json").write_text(json.dumps({"seed": attempt.entry["planned_seeds"][number]}))
            attempt.finish_rollout(number, folder)
        return attempt

    def commit(self, attempt, *, accepted=True, gradients=True, applied=True, update=1):
        return attempt.for_checkpoint(checkpoint=self.root / f"wave_{update:06d}/checkpoint.pt",
            wave=update, optimizer_updates=update, global_step_applied=applied,
            accepted=accepted, contributed_gradients=gradients,
            policy_digest=f"policy-{update}", optimizer_digest=f"optimizer-{update}")

    def test_partial_failed_attempt_preserves_actual_starts_and_all_planned_seeds(self):
        attempt = self.finish(self.attempt(index=1), count=1)
        attempt.begin_rollout(1, [[2, 3, 4]])
        attempt.fail(RuntimeError("native generation failed"))
        saved = json.loads(attempt.path.read_text())
        self.assertEqual(saved["planned_seeds"], list(range(1041, 1049)))
        self.assertEqual([r["status"] for r in saved["rollouts"]], ["sampled", "sampling"])
        self.assertEqual(saved["outcome"], "failed")
        self.assertIsNone(saved["optimizer_commit"])
        self.assertIsNone(saved["checkpoint_commit"])
        self.assertEqual(saved["identity"]["source"]["row_index"], 101)
        self.assertEqual(saved["identity"]["image_sha256"], history.file_digest(self.image))
        with self.assertRaisesRegex(ValueError, "Incomplete group"):
            self.commit(attempt)

    def test_trained_flat_and_unscorable_have_distinct_committed_meanings(self):
        trained = self.finish(self.attempt(index=0))
        flat = self.finish(self.attempt(index=1))
        skipped = self.finish(self.attempt(index=2))
        entries = [self.commit(trained), self.commit(flat, gradients=False),
                   self.commit(skipped, accepted=False, gradients=False)]
        self.assertTrue(all(a.entry["checkpoint_commit"] is None for a in (trained, flat, skipped)))
        self.assertEqual([e["outcome"] for e in entries], ["trained", "flat", "unscorable"])
        self.assertIsNotNone(entries[0]["optimizer_commit"])
        self.assertIsNone(entries[1]["optimizer_commit"])
        self.assertTrue(entries[1]["checkpoint_commit"]["included_in_objective"])
        self.assertFalse(entries[2]["checkpoint_commit"]["included_in_objective"])
        snapshot = history.add_attempts(history.new_history(self.manifest_digest), entries)
        self.assertEqual(history.history_counts(snapshot), dict(attempts=3, questions_with_sampling_started=3,
            sampled_rollouts=24, gradient_contributing_attempts=1, flat_attempts=1,
            unscorable_attempts=1, uncommitted_attempts=0))
        trained.accept_checkpoint(entries[0])
        trained.fail(RuntimeError("summary failed after checkpoint commit"))
        self.assertEqual(trained.entry["outcome"], "trained")
        self.assertIn("post_commit_error", trained.entry)

    def test_all_flat_wave_commits_data_progress_without_an_optimizer_step(self):
        flat = self.finish(self.attempt())
        entry = self.commit(flat, gradients=False, applied=False, update=0)
        self.assertEqual(entry["outcome"], "flat")
        self.assertFalse(entry["checkpoint_commit"]["global_step_applied"])
        self.assertEqual(entry["checkpoint_commit"]["optimizer_updates"], 0)
        with self.assertRaisesRegex(ValueError, "accepted committed global step"):
            self.commit(flat, gradients=True, applied=False)

    def test_repeated_question_has_distinct_attempt_ids_and_cannot_overwrite(self):
        first = self.attempt(run_id="run-one")
        second = self.attempt(run_id="run-two")
        self.assertNotEqual(first.entry["attempt_id"], second.entry["attempt_id"])
        self.assertEqual(first.entry["planned_seeds"], second.entry["planned_seeds"])
        with self.assertRaisesRegex(ValueError, "overwrite"):
            self.attempt(run_id="run-one")

    def test_auditing_does_not_consume_policy_rng(self):
        with patch("torch.cuda.is_available", return_value=False):
            base.seed_everything(12345)
            before = base._state_digest(base.capture_rng_state())
            self.finish(self.attempt()).progress("gradients_ready")
            self.assertEqual(base._state_digest(base.capture_rng_state()), before)

    def test_resume_preserves_partial_and_later_branch_work_as_uncommitted(self):
        committed = self.finish(self.attempt(index=0))
        committed_entry = self.commit(committed)
        committed.accept_checkpoint(committed_entry)
        stored = history.add_attempts(history.new_history(self.manifest_digest), [committed_entry])
        partial = self.finish(self.attempt(index=1), count=2)
        partial.begin_rollout(2, [[2, 3, 4]])
        partial.fail(RuntimeError("interrupted"))
        later_branch = self.finish(self.attempt(index=2))
        later_branch.accept_checkpoint(self.commit(later_branch, update=2))
        restored = history.resume_history(stored, self.examples, self.root / "sample_history")
        self.assertEqual(restored["attempts"][0], committed_entry)
        self.assertEqual(len(restored["attempts"]), 3)
        for entry in restored["attempts"][1:]:
            self.assertEqual(entry["outcome"], "uncommitted_before_resume")
            self.assertIsNone(entry["checkpoint_commit"])
            self.assertIsNone(entry["optimizer_commit"])
        self.assertEqual(restored["attempts"][1]["rollouts"][2]["status"], "sampling")
        self.assertEqual(restored["attempts"][2]["prior_branch_checkpoint_commit"]["optimizer_updates"], 2)
        new_root = self.root / "new-output-directory"
        history.write_history(new_root, restored)
        self.assertEqual(json.loads((new_root / "sample_history.json").read_text()), restored)
        next_attempt = self.finish(self.attempt(index=1, run_id="resumed-run", root=new_root))
        next_history = history.add_attempts(restored, [self.commit(next_attempt, update=2)])
        self.assertEqual(len(next_history["attempts"]), 4)
        self.assertEqual(history.history_counts(next_history)["gradient_contributing_attempts"], 2)

    def test_resume_rejects_changed_image_and_question_or_untrusted_audit_identity(self):
        attempt = self.attempt()
        empty = history.new_history(self.manifest_digest)
        wrong = deepcopy(attempt.entry)
        wrong["identity"]["question_sha256"] = "wrong"
        attempt.path.write_text(json.dumps(wrong))
        with self.assertRaisesRegex(ValueError, "identity/content changed"):
            history.resume_history(empty, self.examples, self.root / "sample_history")
        attempt.flush()
        self.image.write_bytes(b"image was replaced")
        with self.assertRaisesRegex(ValueError, "image SHA256 mismatch"):
            history.resume_history(empty, self.examples, self.root / "sample_history")

    def test_consecutive_failed_retry_keeps_explicit_audits_and_deduplicates_attempts(self):
        # A has the selected checkpoint. B resumes A but fails before writing
        # any checkpoint, so C must explicitly carry B's audit directory.
        first = self.finish(self.attempt(index=0, run_id="run-a"))
        committed = self.commit(first)
        first.accept_checkpoint(committed)
        stored = history.add_attempts(history.new_history(self.manifest_digest), [committed])
        root_b, root_c = self.root / "run-b", self.root / "run-c"
        failed = self.finish(self.attempt(index=1, run_id="run-b", root=root_b), count=1)
        failed.begin_rollout(1, [[2, 3, 4]])
        failed.fail(RuntimeError("B failed before its first checkpoint"))
        duplicate_audits = self.root / "duplicate-audits"
        duplicate_audits.mkdir()
        (duplicate_audits / failed.path.name).write_text(failed.path.read_text())
        # Later audit edits never replace A's digest-bound checkpoint entry.
        first.fail(RuntimeError("post-commit audit error"))
        resumed = history.resume_history(stored, self.examples, self.root / "sample_history",
            additional_audit_directories=[root_b / "sample_history", duplicate_audits])
        self.assertEqual(len(resumed["attempts"]), 2)
        self.assertEqual(resumed["attempts"][0], committed)
        imported = resumed["attempts"][1]
        self.assertEqual(imported["attempt_id"], failed.entry["attempt_id"])
        self.assertEqual(imported["outcome"], "uncommitted_before_resume")
        self.assertIsNone(imported["optimizer_commit"])
        self.assertIsNone(imported["checkpoint_commit"])
        self.assertEqual([r["status"] for r in imported["rollouts"]], ["sampled", "sampling"])
        self.assertEqual(imported["error"]["message"], "B failed before its first checkpoint")
        self.assertEqual(imported["imported_audit_path"], str(failed.path))
        retried = self.finish(self.attempt(index=1, run_id="run-c", root=root_c))
        completed = history.add_attempts(resumed, [self.commit(retried, update=2)])
        self.assertEqual(len(completed["attempts"]), 3)
        self.assertEqual(history.history_counts(completed)["gradient_contributing_attempts"], 2)

    def test_explicit_resume_audit_directories_must_exist_and_match_inputs(self):
        empty = history.new_history(self.manifest_digest)
        with self.assertRaisesRegex(ValueError, "Explicit resume audit directory"):
            history.resume_history(empty, self.examples, self.root / "sample_history",
                additional_audit_directories=[self.root / "missing-audits"])
        extra_root = self.root / "extra"
        attempt = self.attempt(root=extra_root)
        wrong = deepcopy(attempt.entry)
        wrong["identity"]["question_sha256"] = "different question"
        attempt.path.write_text(json.dumps(wrong))
        with self.assertRaisesRegex(ValueError, "identity/content changed"):
            history.resume_history(empty, self.examples, self.root / "sample_history",
                additional_audit_directories=[extra_root / "sample_history"])

    def test_resume_rejects_changed_manifest_even_when_image_is_same(self):
        changed = deepcopy(self.examples)
        changed[0]["question"] = "different question"
        with self.assertRaisesRegex(ValueError, "different manifest"):
            history.resume_history(history.new_history(self.manifest_digest), changed, self.root / "sample_history")


class HistoryCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cpu = patch("torch.cuda.is_available", return_value=False)
        self.cpu.start()
        self.addCleanup(self.cpu.stop)
        self.model = runner_tests.TinyPolicy()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=.01)
        sum(p.square().sum() for p in self.model.parameters()).backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.history = history.new_history("manifest-sha256")
        self.configuration = {"sample_history": {"schema_version": 1, "manifest_digest": "manifest-sha256"}}
        self.path = Path(self.tmp.name) / "checkpoint.pt"
        self.coordinator = runner_tests.SaveCoordinator([base.capture_rng_state(), base.capture_rng_state()])
        self.save()

    def save(self):
        _save(torch, base, self.coordinator, self.path, self.model, self.optimizer, self.configuration,
              last_group=7, optimizer_updates=1, wave=0, sample_history=self.history)

    def restore(self, include=True):
        return _restore(torch, base, self.path, self.model, self.optimizer, self.configuration,
                        0, 2, include_sample_history=include)

    def test_history_travels_inside_checkpoint_sidecar_and_digest_marker(self):
        _, marker_path = _checkpoint_paths(self.path)
        self.assertEqual(json.loads(marker_path.read_text())["sample_history_digest"], history.digest(self.history))
        result = self.restore()
        self.assertEqual(result["sample_history"], self.history)
        self.assertEqual(result["sample_history_digest"], history.digest(self.history))
        self.assertNotIn("sample_history", self.restore(include=False))

    def test_corrupted_or_removed_history_is_rejected_before_model_restore(self):
        rng_path, _ = _checkpoint_paths(self.path)
        for remove in (False, True):
            with self.subTest(remove=remove):
                self.save()
                value = torch.load(rng_path, weights_only=True)
                if remove:
                    value.pop("sample_history")
                else:
                    value["sample_history"]["manifest_digest"] = "altered"
                torch.save(value, rng_path)
                with torch.no_grad():
                    self.model.lora_text.add_(1.)
                expected = base.adapter_digest(self.model)
                with self.assertRaisesRegex(ValueError, "sample-history digest mismatch"):
                    self.restore()
                self.assertEqual(base.adapter_digest(self.model), expected)

    def test_new_configuration_cannot_save_without_required_history(self):
        with self.assertRaisesRegex(ValueError, "requires sample history"):
            _save(torch, base, self.coordinator, self.path, self.model, self.optimizer, self.configuration,
                  last_group=7, optimizer_updates=1, wave=0)


class SamplingHistoryIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = runner_tests.SamplingJudgingBoundaryTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        examples = [self.fixture.example]
        self.attempt = history.SampleAttempt(self.fixture.root, run_id="integration",
            manifest_digest=history.digest(examples), identity=history.example_identity(examples, 0),
            seed=self.fixture.cfg["seed"], rank=0, world_size=1,
            policy_digest="policy_digest", prior_optimizer_updates=0)

    def test_real_sampler_boundary_records_all_completed_rollouts_without_training_claim(self):
        self.fixture.run_group(sample_attempt=self.attempt)
        entry = json.loads(self.attempt.path.read_text())
        self.assertEqual([r["status"] for r in entry["rollouts"]], ["sampled"] * 8)
        self.assertIsNone(entry["optimizer_commit"])
        self.assertTrue(all(r["sampling_sha256"] for r in entry["rollouts"]))

    def test_failed_sampler_retains_started_rollout_without_completion_or_training(self):
        with self.assertRaisesRegex(RuntimeError, "failed sampler"):
            self.fixture.run_group(sampler_error=RuntimeError("failed sampler"), sample_attempt=self.attempt)
        entry = json.loads(self.attempt.path.read_text())
        self.assertEqual(len(entry["rollouts"]), 1)
        self.assertEqual(entry["rollouts"][0]["status"], "sampling")
        self.assertIsNone(entry["optimizer_commit"])


if __name__ == "__main__":
    unittest.main()
