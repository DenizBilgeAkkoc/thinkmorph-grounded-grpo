import ast
import json
from pathlib import Path
import tempfile
import typing
from types import SimpleNamespace
import unittest
import torch
from PIL import Image
from thinkmorph_grpo.sde import schedule, shared_windows, sde_step, ode_step
from thinkmorph_grpo.objective import GRPOConfig, group_advantages, normalize_advantages, ppo_loss
from thinkmorph_grpo.judge import StepEvidence, process_messages, validate_process, reward_components, METRICS
from thinkmorph_grpo.guidance import require_no_cfg

# Compare against the frozen historical implementation, independent of this package.
ROOT = Path(__file__).resolve().parent / "fixtures"


def original_function(path, name):
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    namespace = {"torch": torch, "Optional": typing.Optional, "SimpleNamespace": SimpleNamespace}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class CoreTests(unittest.TestCase):
    def test_rl_rejects_cfg_or_unspecified_guidance(self):
        require_no_cfg({"cfg_text_scale":1.,"cfg_img_scale":1.})
        for settings in ({}, {"cfg_text_scale":4.,"cfg_img_scale":2.},
                         {"cfg_text_scale":1.,"cfg_img_scale":2.},
                         {"cfg_text_scale":True,"cfg_img_scale":1.}):
            with self.assertRaises(ValueError): require_no_cfg(settings)

    def test_selected_custom_preset_does_not_enable_guidance(self):
        make_config = original_function(ROOT / "historical_math.py", "make_grpo_config")
        preset = json.loads((ROOT / "historical_preset.json").read_text())
        config = make_config(SimpleNamespace(**preset))
        self.assertFalse(hasattr(config.sample, "guidance"))
        sample_guidance = original_function(ROOT / "historical_math.py", "sample_guidance")
        self.assertEqual(sample_guidance(None, config, None, None, None), {})

    def test_guided_rollout_cannot_enter_unguided_replay(self):
        from thinkmorph_grpo.replay import flow_inputs, flow_logprob
        guided={"settings":{"cfg_text_scale":4.,"cfg_img_scale":2.}}
        with self.assertRaises(ValueError): flow_inputs(None,guided,None)
        off={"cfg_text_scale":1.,"cfg_img_scale":1.}
        # Even relabeling trace settings cannot make old/unmarked transitions valid.
        with self.assertRaises(ValueError): flow_logprob(None,off,{"settings":off},{},None)

    def test_sde_matches_actual_custom_code_and_gradients(self):
        original = original_function(ROOT / "historical_math.py", "_sde_step_with_logprob")
        ts = schedule()
        for i in (0, 2, 6):
            g = torch.Generator().manual_seed(10)
            x, v = torch.randn(5, 8, generator=g), torch.randn(5, 8, generator=g)
            a = sde_step(v, ts[i], ts[i+1], x, sigma_max=ts[1], generator=torch.Generator().manual_seed(5))
            b = original(None, v, ts[i], ts[i+1], ts[i+1]-ts[i], x,
                sigma_max=ts[1], noise_level=1.0, generator=torch.Generator().manual_seed(5))
            for actual, expected in zip(a, b):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            new_v = v.clone().requires_grad_()
            _, lp, _, _ = sde_step(new_v, ts[i], ts[i+1], x, sigma_max=ts[1], previous_sample=a[0])
            lp.backward()
            self.assertTrue(torch.isfinite(new_v.grad).all())
            self.assertGreater(new_v.grad.abs().sum().item(), 0)

    def test_15_points_means_14_transitions_and_shared_three_step_window(self):
        self.assertEqual(len(schedule())-1, 14)
        windows = shared_windows(42)
        self.assertEqual(windows, shared_windows(42))
        self.assertTrue(all(0 <= w <= 4 for w in windows))
        self.assertEqual(windows, [2, 2, 1])

    def test_ode_fp32_matches_custom_zero_noise_branch(self):
        original = original_function(ROOT / "historical_math.py", "_sde_step_with_logprob")
        ts=schedule()
        x=torch.randn(5,8)
        v=torch.randn(5,8).bfloat16()
        expected=original(None,v,ts[2],ts[3],ts[3]-ts[2],x,sigma_max=ts[1],noise_level=0)[0]
        torch.testing.assert_close(ode_step(v,ts[2],ts[3],x),expected,rtol=0,atol=0)

    def test_normalization_matches_custom_trainer(self):
        original = original_function(ROOT / "historical_math.py", "normalize_advantages")
        for r in (torch.tensor([1.]), torch.ones(8), torch.arange(8).float()):
            torch.testing.assert_close(normalize_advantages(r, .02), original(r, .02))

    def test_masked_variable_length_group(self):
        valid = torch.tensor([[1,1,0],[1,0,0],[1,1,1]], dtype=torch.bool)
        r = torch.tensor([[1.,0,0],[0,0,0],[.5,.5,1]])
        adv = group_advantages(r, r, valid, torch.tensor([1.,0,1]), GRPOConfig())
        self.assertTrue(torch.isfinite(adv).all())
        self.assertEqual(adv[2,2], 0)  # one member cannot define a group advantage
        self.assertTrue((adv[~valid] == 0).all())

    def test_ratio_identity_and_nonzero_gradient(self):
        lp = torch.tensor(-12000., requires_grad=True)
        loss, ratio = ppo_loss(lp, lp.detach(), 1.0, GRPOConfig())
        self.assertEqual(ratio.item(), 1.0)
        loss.backward()
        self.assertEqual(lp.grad.item(), -1.0)

    def test_process_judge_has_no_reference_or_final_answer(self):
        with tempfile.TemporaryDirectory() as d:
            image = Path(d) / "image.png"
            Image.new("RGB", (2, 2), "white").save(image)
            e = StepEvidence("problem", str(image), str(image), str(image),
                "inspect this <answer>SECRET_ANSWER</answer>", "claim <answer>SECRET_UNFINISHED", ("prior",))
            serialized = json.dumps(process_messages(e))
            self.assertNotIn("SECRET", serialized)
            self.assertNotIn("expected_answer", serialized)
            with self.assertRaises(TypeError):
                StepEvidence(**{**e.__dict__, "reference_trace": "SECRET"})

    def test_invalid_or_uncertain_judgments_do_not_reward(self):
        value = {"scores": {m:4 for m in METRICS}, "assessable": True, "issues": [], "evidence":"visible"}
        self.assertEqual(reward_components(value), {"text":1.,"image":1.,"post_text":1.})
        for change in ({"assessable":False}, {"scores":{m:5 for m in METRICS}}, {"scores":{m:True for m in METRICS}}):
            with self.assertRaises(ValueError):
                validate_process({**value, **change})
        value["scores"]["step_consistency"] = 0
        self.assertEqual(reward_components(value), {"text":0.,"image":0.,"post_text":0.})

    def test_after_image_feedback_reaches_following_text_action(self):
        from thinkmorph_grpo.trainer import judge_group
        class FakeJudge:
            def process(self, evidence):
                scores={m:4 for m in METRICS}
                scores["observation_grounding"]=1
                return {"scores":scores,"assessable":True,"issues":[],"evidence":"diagnostic"}
            def terminal(self,*args):
                return {"correct":True,"assessable":True,"evidence":"diagnostic"}
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            trace=[{"kind":"text","text":"inspect <image_start>"},
                   {"kind":"image","file":"after.png"},
                   {"kind":"text","text":"claim <answer>x</answer>"}]
            (folder/"trace.json").write_text(json.dumps(trace))
            example={"question":"problem","problem_image":{"path":"original.png"},"answer":"x"}
            result=judge_group(example,[folder],FakeJudge())[0]
            self.assertEqual(result["steps"][0]["text"],1.)
            self.assertEqual(result["steps"][1]["text"],.25)
            self.assertEqual(result["steps"][1]["image"],0.)


if __name__ == "__main__":
    unittest.main()
