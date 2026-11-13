"""Training skips only exhausted judgments, preserving the complete N=8 contract."""
from contextlib import ExitStack, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image
import torch

from thinkmorph_grpo import train
from thinkmorph_grpo.judge_v2 import JudgeAssessmentError


class SkipUnscorableGroupsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        original = self.root / "original.png"
        Image.new("RGB", (2, 2), "white").save(original)
        cfg = {"group_size": 8, "sde_window_size": 3, "seed": 42,
            "source": "/unused/native", "checkpoint": "/unused/checkpoint",
            "native_revision": "test", "noise_level": .7,
            "native_settings": {"cfg_text_scale": 1., "cfg_img_scale": 1.,
                                "max_rounds": 1, "num_timesteps": 15}}
        recipe = {"judge_version": "test", "terminal_policy_version": "test",
            "lora": {"rank": 1, "alpha": 2}, "grpo": {},
            "optimizer": {"image_learning_rate": 5e-5, "text_learning_rate": 3e-5, "weight_decay": 0.}}
        examples = [{"task": "test", "question": "q", "answer": "a",
            "example_id": str(i), "problem_image": {"path": str(original)}} for i in range(3)]
        for name, value in (("config", cfg), ("recipe", recipe), ("manifest", examples)):
            (self.root / (name + ".json")).write_text(json.dumps(value))

    def arguments(self, output):
        return ["--preview-config", str(self.root / "config.json"),
            "--recipe", str(self.root / "recipe.json"), "--manifest", str(self.root / "manifest.json"),
            "--output", str(output), "--judge-url", "http://unused"]

    def invoke(self, output, judgments, *, skip=True, max_groups=3, max_updates=1,
               update_error=None, sampling_error=None):
        judge = SimpleNamespace(version="test", terminal_version="test", verify_model=MagicMock())
        recorder = MagicMock()
        recorder.begin_recording.side_effect = lambda folder, bridge: folder.mkdir()
        if sampling_error:
            recorder.interleave_inference.side_effect = sampling_error
        self.trainer = MagicMock()
        self.trainer.update.return_value = {"optimized": True, "gradient_norm_before_clip": 1.}
        if update_error:
            self.trainer.update.side_effect = update_error
        inferencer = MagicMock(return_value=SimpleNamespace(model=object()))
        args = ["train", *self.arguments(output), "--max-groups", str(max_groups),
                "--max-updates", str(max_updates)]
        if skip:
            args.append("--skip-unscorable-groups")
        self.stdout = StringIO()
        with ExitStack() as stack:
            stack.enter_context(patch("sys.argv", args))
            stack.enter_context(redirect_stdout(self.stdout))
            stack.enter_context(patch.object(train, "seed_everything"))
            stack.enter_context(patch.object(train.torch, "set_num_threads"))
            stack.enter_context(patch.object(train.torch.cuda, "is_available", return_value=False))
            stack.enter_context(patch.object(train, "GroundedQwenJudge", return_value=judge))
            stack.enter_context(patch.object(train, "load_native", return_value=(object(), inferencer)))
            stack.enter_context(patch.object(train, "attach_lora", return_value={"image": [], "text": []}))
            stack.enter_context(patch.object(train.torch.optim, "AdamW"))
            stack.enter_context(patch.object(train, "recorded_inferencer", return_value=MagicMock(return_value=recorder)))
            stack.enter_context(patch.object(train, "NativeGRPOTrainer", return_value=self.trainer))
            stack.enter_context(patch.object(train, "NativeFlowBridge"))
            stack.enter_context(patch.object(train, "shared_windows", return_value=[1]))
            stack.enter_context(patch.object(train, "adapter_digest", return_value="same-policy"))
            self.judge_group = stack.enter_context(patch.object(train, "judge_group", side_effect=judgments))
            self.checkpoint = stack.enter_context(patch.object(train, "save_checkpoint", return_value="next-policy"))
            train.main()
        return json.loads((output / "summary.json").read_text())

    def test_skipping_is_opt_in(self):
        args = self.arguments(self.root / "out")
        self.assertFalse(train.parse_args(args).skip_unscorable_groups)
        self.assertTrue(train.parse_args(args + ["--skip-unscorable-groups"]).skip_unscorable_groups)

    def test_skipped_group_has_no_update_and_next_complete_group_can_train(self):
        output = self.root / "out"
        failure = JudgeAssessmentError("uncertain factual support after retries")
        summary = self.invoke(output, [failure, [{"complete": True}] * 8])
        self.assertEqual(self.judge_group.call_count, 2)
        self.assertTrue(all(len(call.args[1]) == 8 for call in self.judge_group.call_args_list))
        self.trainer.update.assert_called_once()
        self.assertEqual(len(self.trainer.update.call_args.args[0]), 8)
        self.assertTrue(all("group_000001" in str(p) for p in self.trainer.update.call_args.args[0]))
        self.checkpoint.assert_called_once()
        self.assertEqual(self.checkpoint.call_args.kwargs["group"], 1)
        self.assertFalse((output / "group_000000/update.json").exists())
        self.assertFalse((output / "group_000000/checkpoint.pt").exists())
        skipped = json.loads((output / "group_000000/group_skip.json").read_text())
        self.assertFalse(skipped["optimized"])
        self.assertEqual(skipped["optimizer_updates"], 0)
        self.assertEqual(skipped["error"]["type"], "JudgeAssessmentError")
        self.assertIn(str(failure), skipped["traceback"])
        self.assertNotIn("rewards", skipped)
        self.assertEqual(summary["unscorable_groups_skipped"], 1)
        self.assertEqual(summary["optimizer_updates"], 1)
        self.assertEqual(summary["groups_attempted_this_run"], 2)
        self.assertTrue(summary["target_reached"])
        self.assertEqual(summary["stop_reason"], "target_updates_reached")
        config = json.loads((output / "resolved_config.json").read_text())
        self.assertTrue(config["skip_unscorable_groups"])
        self.assertTrue(config["invocation"]["skip_unscorable_groups"])
        self.assertTrue(self.checkpoint.call_args.kwargs["configuration"]["skip_unscorable_groups"])
        events = [json.loads(line) for line in self.stdout.getvalue().splitlines()]
        self.assertEqual(sum(e["event"] == "group_skipped" for e in events), 1)

    def test_all_unscorable_groups_exhaust_budget_without_claiming_updates(self):
        output = self.root / "all_skipped"
        summary = self.invoke(output, [JudgeAssessmentError("uncertain")] * 3)
        self.trainer.update.assert_not_called()
        self.checkpoint.assert_not_called()
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["optimizer_updates"], 0)
        self.assertEqual(summary["target_optimizer_updates"], 1)
        self.assertEqual(summary["unscorable_groups_skipped"], 3)
        self.assertEqual(summary["groups_attempted_this_run"], 3)
        self.assertFalse(summary["target_reached"])
        self.assertEqual(summary["stop_reason"], "max_groups_exhausted")
        self.assertEqual(len(list(output.glob("group_*"))), 3)

    def test_default_still_fails_on_unscorable_assessment(self):
        output = self.root / "strict"
        failure = JudgeAssessmentError("uncertain")
        with self.assertRaises(JudgeAssessmentError) as caught:
            self.invoke(output, [failure], skip=False)
        self.assertIs(caught.exception, failure)
        self.trainer.update.assert_not_called()
        self.checkpoint.assert_not_called()
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["unscorable_groups_skipped"], 0)
        self.assertFalse(list(output.glob("group_*/group_skip.json")))

    def test_transport_and_programming_errors_are_not_swallowed(self):
        for n, failure in enumerate((TimeoutError("connection timed out"),
                                     ValueError("bad native trace"), RuntimeError("service unavailable"))):
            with self.subTest(type=type(failure).__name__):
                output = self.root / f"error_{n}"
                with self.assertRaises(type(failure)) as caught:
                    self.invoke(output, [failure])
                self.assertIs(caught.exception, failure)
                self.trainer.update.assert_not_called()
                summary = json.loads((output / "summary.json").read_text())
                self.assertEqual(summary["status"], "failed")
                self.assertEqual(summary["unscorable_groups_skipped"], 0)
                self.assertFalse(list(output.glob("group_*/group_skip.json")))

    def test_assessment_error_outside_judging_is_not_swallowed(self):
        for phase in ("sampling", "updating"):
            with self.subTest(phase=phase):
                output = self.root / phase
                failure = JudgeAssessmentError("wrong phase")
                options = {"sampling_error" if phase == "sampling" else "update_error": failure}
                with self.assertRaises(JudgeAssessmentError) as caught:
                    self.invoke(output, [[{}] * 8], **options)
                self.assertIs(caught.exception, failure)
                summary = json.loads((output / "summary.json").read_text())
                self.assertEqual(summary["phase"], phase)
                self.assertEqual(summary["status"], "failed")
                self.assertEqual(summary["unscorable_groups_skipped"], 0)

    def test_checkpoint_rejects_changed_skip_policy_before_mutating_adapters(self):
        class Model(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.lora_test = torch.nn.Parameter(torch.tensor([value]))
        model, target = Model(1.), Model(2.)
        optimizer = torch.optim.AdamW(model.parameters())
        target_optimizer = torch.optim.AdamW(target.parameters())
        path = self.root / "checkpoint.pt"
        train.save_checkpoint(path, model, optimizer, group=0, optimizer_updates=0,
                              configuration={"skip_unscorable_groups": True})
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            train.restore_checkpoint(path, target, target_optimizer,
                                     configuration={"skip_unscorable_groups": False})
        self.assertEqual(float(target.lora_test.detach()), 2.)


if __name__ == "__main__":
    unittest.main()
