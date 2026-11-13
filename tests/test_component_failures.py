"""Confirmed component zeros and honest partial results on judge failure."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from urllib.error import URLError

from thinkmorph_grpo import practical_judge
from thinkmorph_grpo.judge import reward_components
from thinkmorph_grpo.judge_v2 import (
    GroundedQwenJudge, JudgeAssessmentError, validate_visual_with_checks,
)
from thinkmorph_grpo.trainer import judge_group
from test_coverage_normalization import draft
from test_navigation_predicates import assertion
from test_practical_judge import annotation_assessment, verdict


def missing_bar_checks():
    return {
        "pixel_diagnostics": {"original_bar_regions": [
            {"original_bbox_xyxy_pixels": [0, i, 10, i + 1],
             "after_colored_fraction": 0 if i == 0 else 1} for i in range(4)]},
        "detail_crops": [{"region_index": 0, "original_mark_xyxy_pixels": [0, 0, 10, 1]}],
        "focused_chart_checks": [{"region_index": 0, "target_bar_visible": False}],
    }


class ComponentFailureTests(unittest.TestCase):
    def test_confirmed_chart_failure_zeros_only_image_and_keeps_raw_disagreement(self):
        value = annotation_assessment("The label is still highlighted.")
        value["checks"].update(missing_bar_checks())
        original = deepcopy(value)
        value["visual"] = validate_visual_with_checks(value["visual"], value["checks"], "grounded_v9")
        self.assertIn("Confirmed image component failure", value["visual"]["issues"][-1])
        self.assertEqual(value["visual"]["preservation"], "intact")  # Audit disagreement stays visible.
        self.assertEqual(reward_components(verdict(value)), dict(text=1., image=0., post_text=1.))
        self.assertEqual(original["visual"]["issues"], [])
        with self.assertRaisesRegex(ValueError, "Conflicting visual evidence"):
            validate_visual_with_checks(original["visual"], original["checks"], "grounded_v8")

    def test_weak_chart_registration_cannot_zero_a_useful_partial(self):
        value = annotation_assessment()
        value["checks"].update(missing_bar_checks())
        value["checks"]["pixel_diagnostics"]["original_bar_regions"][1]["after_colored_fraction"] = 0
        value["visual"] = validate_visual_with_checks(value["visual"], value["checks"], "grounded_v9")
        self.assertEqual(reward_components(verdict(value))["image"], .75)

    def test_confirmed_image_failure_does_not_hide_invalid_other_fields(self):
        for mutation in (lambda v: v.update(assessable=False),
                         lambda v: v.update(execution="nonsense"),
                         lambda v: v["objects"][0].update(status="uncertain")):
            value = annotation_assessment()
            mutation(value["visual"])
            with self.assertRaises(ValueError):
                validate_visual_with_checks(value["visual"], missing_bar_checks(), "grounded_v9")

    def test_prose_route_loses_symbolic_proof_but_literal_route_keeps_it(self):
        prose, units = draft("The correct path should go up, right, and then down.",
            nav=[assertion("goal_reached", scope="stated_full_route", route="U,R,D")])
        literal, extra = draft(r"\boxed{U,R,D}",
            nav=[assertion("goal_reached", scope="stated_full_route", route="U,R,D")])
        literal["claims"][0]["claim_id"] = 1
        extra[0]["claim_id"] = 1
        prose["claims"].extend(literal["claims"]); units.extend(extra)
        original = deepcopy(prose)
        result = practical_judge.validate_semantics(prose, units)
        self.assertEqual(prose, original)
        self.assertEqual(result["claims"][0]["navigation_claims"][0]["scope"], "unverified")
        self.assertFalse(result["claims"][0]["all_factual_content_covered"])
        self.assertEqual(result["claims"][1]["navigation_claims"][0]["route"], "U,R,D")

    def test_prose_recovery_does_not_hide_malformed_claim(self):
        for changes in ({"all_factual_content_covered": "yes"}, {"nonfactual_only": 1}):
            raw, units = draft("Go up, right and down.",
                nav=[assertion("goal_reached", scope="stated_full_route", route="U,R,D")])
            raw["claims"][0].update(changes)
            with self.assertRaises(ValueError):
                practical_judge.validate_semantics(raw, units)

    def test_exhausted_terminal_is_skippable_but_transport_is_not(self):
        class Judge(GroundedQwenJudge):
            def complete(self, *args, **kwargs):
                self.calls += 1
                if self.transport:
                    raise URLError("server unavailable")
                return {"correct": False, "assessable": False, "evidence": "Unreadable"}

        with tempfile.TemporaryDirectory() as tmp:
            from PIL import Image
            path = Path(tmp) / "image.png"; Image.new("RGB", (4, 4)).save(path)
            judge = Judge("unused"); judge.calls = 0; judge.transport = False
            with self.assertRaises(JudgeAssessmentError):
                judge.terminal("Question", str(path), "answer", "expected")
            self.assertEqual(judge.calls, 2)
            judge.calls = 0; judge.transport = True
            with self.assertRaises(URLError):
                judge.terminal("Question", str(path), "answer", "expected")
            self.assertEqual(judge.calls, 1)

    def test_process_failure_preserves_terminal_without_fake_zero_verdict(self):
        class Judge:
            protocol_revision = "test"
            def terminal(self, *args):
                return {"correct": True, "assessable": True, "evidence": "Independent answer"}
            def process(self, evidence):
                raise JudgeAssessmentError("Unresolved image assessment")

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "trace.json").write_text(json.dumps([
                {"kind": "text", "text": "Draw the box."}, {"kind": "image", "file": "after.png"},
                {"kind": "text", "text": "<answer>76</answer>"}]))
            example = dict(question="Question", answer="76", problem_image={"path": "original.png"})
            with self.assertRaisesRegex(JudgeAssessmentError, "process step 0"):
                judge_group(example, [folder], Judge())
            partial = json.loads((folder / "judge_partial.json").read_text())
            self.assertTrue(partial["terminal"]["correct"])
            self.assertFalse(partial["missing_components_imputed"])
            self.assertEqual(partial["completed_process_components"], [])
            self.assertFalse((folder / "judge.json").exists())
            self.assertTrue((folder / "terminal.json").exists())


if __name__ == "__main__":
    unittest.main()
