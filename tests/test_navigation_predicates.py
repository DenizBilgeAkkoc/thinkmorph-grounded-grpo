"""Navigation claim regressions discovered by the frozen v5b control suite.

These test schema/derivation with explicit model assessments, not a model's
ability to extract the semantics. Live multimodal controls remain necessary.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from thinkmorph_grpo.checks import check_path
from thinkmorph_grpo.judge import StepEvidence, reward_components, validate_process
from thinkmorph_grpo.judge_v2 import (
    GroundedQwenJudge, derive_scores, resolve_claim_ids, text_units, validate_observations,
    validate_visual_with_checks,
)


GRID = {"rows": 3, "cols": 3, "start": [2, 0], "goal": [2, 2],
        "holes": [[0, 0], [0, 2], [2, 1]]}


def assertion(predicate, asserted=True, scope="drawn_route", route=""):
    return dict(scope=scope, route=route, predicate=predicate, asserted=asserted)


def assessment(quote, assertions, moves=("U",), kind="other", support="supported"):
    return {
        "proposal": {"evidence": "Draw the requested route", "operation": "draw route",
            "operation_kind": "navigate", "claims": [{"quote": "Draw a route.",
                "kind": "intention", "support": "not_applicable", "evidence": "Future action"}],
            "moves": [], "validity": "valid", "assessable": True, "issues": []},
        "visual": {"evidence": "Route visible on preserved grid", "observed_operation": "route drawn",
            "objects": [{"name": "grid", "origin": "original", "preservation_required": True,
                "status": "present", "evidence": "Grid remains visible"}],
            "annotation": "present", "target_match": "yes", "path_present": True,
            "path_moves": list(moves), "execution": "complete", "preservation": "intact",
            "transition": "valid", "utility": "useful", "assessable": True, "issues": []},
        "observations": {"evidence": "Claims independently checked", "claims": [
            {"quote": quote, "kind": kind, "support": support, "evidence": "Frozen visual assessment",
             "navigation_claims": assertions}], "assessable": True, "issues": []},
        "checks": {"verified_original_grid": deepcopy(GRID), "observed_path": check_path(list(moves), GRID)},
    }


def scores(value):
    return derive_scores(value, require_origins=True, schema_version="grounded_v6")


class NavigationPredicateTests(unittest.TestCase):
    def test_public_dispatch_preserves_honest_text_credit_after_failed_image(self):
        value = assessment("This path is unsafe.", [assertion("safe", False)], moves=("R", "R"))
        verdict = {"schema_version": "grounded_v6", "assessment": value, "scores": scores(value),
                   "assessable": True, "issues": [], "evidence": "Original grid and explicit assertions checked"}
        self.assertIs(validate_process(verdict), verdict)
        self.assertEqual(reward_components(verdict), {"text": 1.0, "image": 0.0, "post_text": 1.0})
        verdict["scores"]["observation_grounding"] = 0
        with self.assertRaises(ValueError):
            reward_components(verdict)

    def test_process_integrates_predicates_without_extra_request_or_empty_path_proof(self):
        for path_present in (True, False):
            with self.subTest(path_present=path_present), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "original.png"
                Image.new("RGB", (12, 12), "white").save(path)
                value = assessment("The route is safe and reaches the goal.",
                    [assertion("safe"), assertion("goal_reached")])
                value["visual"].update(path_present=path_present, path_moves=["U"] if path_present else [])
                drafts = [deepcopy(value[k]) for k in ("proposal", "visual", "observations")]
                for draft in (drafts[0], drafts[2]):
                    draft["claims"][0].pop("quote")
                    draft["claims"][0]["claim_id"] = 0

                class Judge(GroundedQwenJudge):
                    version = "grounded_v6"
                    def complete(self, messages, **kwargs):
                        self.requests.append(messages)
                        return drafts.pop(0)

                judge = Judge("unused")
                judge.requests = []
                constraints = dict(source="verified_original_image",
                    image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), grid=deepcopy(GRID))
                result = judge.process(StepEvidence("Find a legal route.", str(path), str(path), str(path),
                    "Draw a route.", "The route is safe and reaches the goal.",
                    task="navigation", original_constraints=constraints))
                self.assertEqual(len(judge.requests), 3)
                self.assertIs(validate_process(result), result)
                self.assertEqual(reward_components(result)["post_text"], 0 if path_present else 1)
                checks = result["assessment"]["checks"]
                self.assertEqual(checks["verified_original_grid"], GRID)
                self.assertEqual("observed_path" in checks, path_present)
                self.assertEqual(result["scores"]["observation_grounding"], 0 if path_present else 4)
                self.assertIn("navigation_claims", judge.requests[2][0]["content"])
                fields = json.loads(judge.requests[2][1]["content"][0]["text"])
                self.assertIn("text_after_units", fields)

    def test_compound_safety_does_not_hide_false_arrival(self):
        value = assessment("The drawn arrow completes a safe route all the way to the goal.",
            [assertion("safe"), assertion("goal_reached")], kind="path_safety")
        original = deepcopy(value)
        self.assertEqual(scores(value)["observation_grounding"], 0)
        self.assertEqual(value, original)  # Raw evidence remains auditable.

    def test_truthful_negative_safety_survives_coarse_kind(self):
        value = assessment("This drawn path is not safe.", [assertion("safe", False)],
            moves=("R", "R"), kind="path_safety")
        result = scores(value)
        self.assertEqual(result["observation_grounding"], 4)
        self.assertEqual(result["image_execution"], 0)

    def test_negative_goal_truth_depends_on_actual_route(self):
        for moves, expected in [(("U",), 4), (("U", "R", "R", "D"), 0)]:
            with self.subTest(moves=moves):
                value = assessment("The goal has not been reached.",
                    [assertion("goal_reached", False)], moves=moves, kind="goal_reached")
                self.assertEqual(scores(value)["observation_grounding"], expected)

    def test_two_negative_predicates_and_double_negation(self):
        value = assessment("The route is unsafe and has not reached the goal.",
            [assertion("safe", False), assertion("goal_reached", False)], moves=("R", "R"))
        self.assertEqual(scores(value)["observation_grounding"], 4)
        value = assessment("The route does not fail to reach the goal.", [assertion("goal_reached")])
        self.assertEqual(scores(value)["observation_grounding"], 0)

    def test_future_and_negated_claim_do_not_create_current_assertions(self):
        for quote in ("I will extend the route to reach the goal.",
                      "If extended, the route may reach the goal.",
                      "I do not claim that the route is unsafe."):
            with self.subTest(quote=quote):
                value = assessment(quote, [], kind="intention", support="not_applicable")
                self.assertEqual(scores(value)["observation_grounding"], 0)
        mixed = assessment("The goal has not been reached; I will extend the route next.",
            [assertion("goal_reached", False)])
        self.assertEqual(scores(mixed)["observation_grounding"], 4)

    def test_valid_stated_alternative_is_not_the_incomplete_or_unsafe_drawing(self):
        route = "U, R, R, D"
        for moves in (("U",), ("R", "R")):
            with self.subTest(moves=moves):
                value = assessment(f"The full route {route} is safe and reaches the goal.",
                    [assertion(p, scope="stated_full_route", route=route) for p in ("safe", "goal_reached")],
                    moves=moves, kind="goal_reached")
                self.assertEqual(scores(value)["observation_grounding"], 4)

    def test_invalid_stated_alternative_is_not_repaired_by_correct_drawing(self):
        value = assessment("The proposed full route R,R is safe and reaches the goal.",
            [assertion(p, scope="stated_full_route", route="R,R") for p in ("safe", "goal_reached")],
            moves=("U", "R", "R", "D"))
        self.assertEqual(scores(value)["observation_grounding"], 0)

    def test_different_scopes_can_coexist_in_one_claim(self):
        value = assessment("The drawn route has not reached the goal but full route U,R,R,D reaches it.",
            [assertion("goal_reached", False),
             assertion("goal_reached", scope="stated_full_route", route="U,R,R,D")])
        self.assertEqual(scores(value)["observation_grounding"], 4)
        value["observations"]["claims"][0]["navigation_claims"][0]["asserted"] = True
        self.assertEqual(scores(value)["observation_grounding"], 0)

    def test_unverified_scope_or_missing_path_retains_model_judgment(self):
        value = assessment("This incremental segment is safe.",
            [assertion("safe", scope="unverified")], moves=("R", "R"), kind="path_safety")
        self.assertEqual(scores(value)["observation_grounding"], 4)
        value = assessment("The route reaches the goal.", [assertion("goal_reached")], kind="goal_reached")
        del value["checks"]["observed_path"]
        self.assertEqual(scores(value)["observation_grounding"], 4)
        value = assessment("The full route R,R is safe.",
            [assertion("safe", scope="stated_full_route", route="R,R")])
        del value["checks"]["verified_original_grid"]
        self.assertEqual(scores(value)["observation_grounding"], 4)

    def test_true_predicates_do_not_promote_other_unsupported_facts(self):
        value = assessment("The route is safe and the box is blue.", [assertion("safe")], support="unsupported")
        self.assertEqual(scores(value)["observation_grounding"], 0)

    def test_claim_ids_keep_exact_text_and_require_complete_navigation_schema(self):
        units = text_units("The route is safe. The goal has not been reached.")
        raw = {"evidence": "Checked", "assessable": True, "issues": [], "claims": [
            {"claim_id": i, "evidence": "Supported by image", "kind": "other", "support": "supported",
             "navigation_claims": [assertion("safe" if i == 0 else "goal_reached", i == 0)]}
            for i in range(2)]}
        resolved = resolve_claim_ids(raw, units, navigation=True)
        validate_observations(resolved, navigation=True)
        self.assertEqual(resolved["claims"][1]["quote"], units[1]["text"])
        del raw["claims"][0]["navigation_claims"]
        with self.assertRaises(ValueError):
            resolve_claim_ids(raw, units, navigation=True)

    def test_structural_validation_rejects_invented_or_truncated_routes(self):
        for route in ("R,R", "U,R", "U,R,R,D,D"):
            with self.subTest(route=route):
                value = assessment("The full route U,R,R,D is safe.",
                    [assertion("safe", scope="stated_full_route", route=route)])
                with self.assertRaises(ValueError):
                    scores(value)
        for quote, route in [("The full route Q, U,R is safe.", "U,R"),
                             ("The full route U,R,Q is safe.", "U,R")]:
            with self.subTest(quote=quote):
                value = assessment(quote, [assertion("safe", scope="stated_full_route", route=route)])
                with self.assertRaises(ValueError):
                    scores(value)
        value = assessment("The route is safe.", [assertion("safe", 1)])
        with self.assertRaises(ValueError):
            scores(value)
        value = assessment("I will draw a safe path.", [assertion("safe")], kind="intention", support="not_applicable")
        with self.assertRaises(ValueError):
            scores(value)

    def test_v6_keeps_preservation_noop_invalid_proposal_and_annotation_gates(self):
        value = assessment("The route is safe.", [assertion("safe")])
        value["visual"]["objects"].append({"name": "white divider", "origin": "original",
            "preservation_required": False, "status": "missing", "evidence": "Removed layout"})
        self.assertEqual(scores(value)["image_preservation"], 4)
        validate_visual_with_checks(value["visual"], {}, "grounded_v6")
        value["checks"]["identical_before_after"] = True
        self.assertEqual(scores(value)["reasoning_utility"], 0)
        del value["checks"]["identical_before_after"]
        value["proposal"]["validity"] = "invalid"
        self.assertEqual(scores(value)["reasoning_utility"], 0)
        value["proposal"].update(validity="valid", operation_kind="annotation")
        value["visual"]["annotation"] = "missing"
        value["observations"]["claims"][0].update(quote="The box is present.", kind="edit_success", navigation_claims=[])
        self.assertEqual(scores(value)["observation_grounding"], 0)
        value["visual"]["objects"][0]["status"] = "missing"
        self.assertEqual(scores(value)["image_preservation"], 0)

    def test_legacy_v5_derivation_remains_unchanged(self):
        value = assessment("This drawn path is not safe.", [assertion("safe", False)],
            moves=("R", "R"), kind="path_safety")
        del value["observations"]["claims"][0]["navigation_claims"]
        # Historical result is intentionally preserved for reproducible comparison.
        self.assertEqual(derive_scores(value, True, "grounded_v5")["observation_grounding"], 0)


if __name__ == "__main__":
    unittest.main()
