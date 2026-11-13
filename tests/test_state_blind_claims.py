"""State-blind semantics, proof boundaries, and exact v6 failure regressions."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from thinkmorph_grpo.judge import StepEvidence, reward_components, validate_process
from thinkmorph_grpo.judge_v2 import (
    GroundedQwenJudge, derive_scores, resolve_claim_ids, validate_claim_semantics,
    validate_observations_with_semantics,
)
from test_navigation_predicates import GRID, assertion, assessment


def meaning(quote, nav=(), edits=(), covered=False, nonfactual=False):
    return dict(quote=quote, evidence="Exact linguistic assertion", navigation_claims=list(nav),
        edit_claims=[dict(predicate="intended_edit_complete", asserted=flag) for flag in edits],
        all_factual_content_covered=covered, nonfactual_only=nonfactual)


def with_semantics(quote, nav=(), edits=(), covered=False, nonfactual=False,
                   moves=("U",), kind="other", support="supported"):
    value = assessment(quote, [], moves=moves, kind=kind, support=support)
    value["observations"]["claims"][0].pop("navigation_claims")
    value["checks"]["state_blind_claims"] = dict(evidence="Text-only assertion extraction", assessable=True,
        issues=[], claims=[meaning(quote, nav, edits, covered, nonfactual)])
    return value


def scores(value):
    return derive_scores(value, True, "grounded_v7")


class StateBlindClaimTests(unittest.TestCase):
    def test_actual_fruit_false_success_cannot_be_rewritten_as_honest_failure(self):
        quote = "With the banana now highlighted by the red box, I can identify it."
        value = with_semantics(quote, edits=[True], kind="failure_report")
        value["proposal"]["operation_kind"] = "annotation"
        value["visual"].update(annotation="present", target_match="no", execution="wrong", transition="invalid")
        result = scores(value)
        self.assertEqual(result["observation_grounding"], 0)
        self.assertEqual(result["image_execution"], 0)
        self.assertEqual(value["observations"]["claims"][0]["support"], "supported")

    def test_honest_different_edit_is_not_intended_success(self):
        value = with_semantics("The box is around the mug.")
        value["proposal"]["operation_kind"] = "annotation"
        value["visual"].update(target_match="no", execution="wrong", transition="invalid")
        self.assertEqual(scores(value)["observation_grounding"], 4)

    def test_negative_edit_polarity_and_no_visual_support_promotion(self):
        value = with_semantics("The requested box was not added.", edits=[False], covered=True, kind="edit_success")
        value["proposal"]["operation_kind"] = "annotation"
        value["visual"].update(annotation="missing", execution="absent")
        self.assertEqual(scores(value)["observation_grounding"], 4)
        value["visual"].update(annotation="present", execution="complete")
        self.assertEqual(scores(value)["observation_grounding"], 0)
        value = with_semantics("The intended edit succeeded.", edits=[True], covered=True, support="unsupported")
        self.assertEqual(scores(value)["observation_grounding"], 0)

    def test_navigation_false_success_uses_claim_polarity_not_corrective_evidence(self):
        value = with_semantics("The drawn route reaches the goal safely without crossing any hole.",
            [assertion("safe"), assertion("goal_reached")], moves=("R", "R"), kind="failure_report")
        self.assertEqual(scores(value)["observation_grounding"], 0)

    def test_both_polarities_and_compound_arrival_are_checked(self):
        value = with_semantics("The route is safe and reaches the goal.",
            [assertion("safe"), assertion("goal_reached")], kind="path_safety")
        self.assertEqual(scores(value)["observation_grounding"], 0)
        for moves, expected in [(("U",), 4), (("U", "R", "R", "D"), 0)]:
            value = with_semantics("The goal has not been reached.", [assertion("goal_reached", False)], moves=moves)
            self.assertEqual(scores(value)["observation_grounding"], expected)
        value = with_semantics("This path is not safe.", [assertion("safe", False)], moves=("R", "R"), kind="path_safety")
        self.assertEqual(scores(value)["observation_grounding"], 4)

    def test_explicit_complete_route_proof_can_correct_absent_drawing_rejection(self):
        route = "U,R,R,D"
        value = with_semantics(f"The full route {route} is safe and reaches the goal.",
            [assertion(p, scope="stated_full_route", route=route) for p in ("safe", "goal_reached")],
            covered=True, support="unsupported")
        self.assertEqual(scores(value)["observation_grounding"], 4)
        value["checks"].pop("verified_original_grid")
        self.assertEqual(scores(value)["observation_grounding"], 0)

    def test_correct_route_does_not_promote_independent_false_fact(self):
        value = with_semantics("The full route U,R,R,D is safe and the grid has four rows.",
            [assertion("safe", scope="stated_full_route", route="U,R,R,D")], covered=False, support="unsupported")
        self.assertEqual(scores(value)["observation_grounding"], 0)

    def test_drawn_or_prose_routes_never_upgrade_visual_support(self):
        for scope in ("drawn_route", "unverified"):
            value = with_semantics("The route is safe.", [assertion("safe", scope=scope)], covered=True, support="unsupported")
            self.assertEqual(scores(value)["observation_grounding"], 0)

    def test_invalid_explicit_alternative_not_rescued_by_complete_drawing(self):
        value = with_semantics("The proposed full route R,R is safe and reaches the goal.",
            [assertion(p, scope="stated_full_route", route="R,R") for p in ("safe", "goal_reached")],
            covered=True, moves=("U", "R", "R", "D"))
        self.assertEqual(scores(value)["observation_grounding"], 0)

    def test_future_and_negated_claim_have_no_factual_reward_but_mixed_fact_remains(self):
        for quote in ("I will draw the box.", "If extended, the route may reach the goal.",
                      "I do not claim the path is unsafe."):
            value = with_semantics(quote, nonfactual=True)
            self.assertEqual(scores(value)["observation_grounding"], 0)
        value = with_semantics("With the banana now highlighted, I will inspect it.", edits=[True])
        value["visual"].update(target_match="no", execution="wrong")
        self.assertEqual(scores(value)["observation_grounding"], 0)

    def test_honest_partial_report_is_not_penalized_by_wrong_proposal(self):
        value = with_semantics("The drawn route is safe but has not reached the goal.",
            [assertion("safe"), assertion("goal_reached", False)])
        value["proposal"]["validity"] = "invalid"
        result = scores(value)
        self.assertEqual(result["reasoning_utility"], 0)
        self.assertEqual(result["observation_grounding"], 4)

    def test_missing_misaligned_or_factual_intention_support_abstains(self):
        value = with_semantics("The route is safe.", [assertion("safe")])
        semantics = value["checks"].pop("state_blind_claims")
        with self.assertRaises(ValueError): scores(value)
        value["checks"]["state_blind_claims"] = semantics
        semantics["claims"][0]["quote"] = "Different claim."
        with self.assertRaises(ValueError): scores(value)
        value = with_semantics("The route is safe.", [assertion("safe")], kind="intention", support="not_applicable")
        with self.assertRaises(ValueError): scores(value)

    def test_semantics_schema_exact_coverage_and_literal_provenance(self):
        value = with_semantics("The full route U,R,R,D is safe.",
            [assertion("safe", scope="stated_full_route", route="R,R")])
        with self.assertRaises(ValueError): scores(value)
        semantics = dict(evidence="", assessable=True, issues=[], claims=[meaning("I will draw.", nonfactual=True, covered=True)])
        with self.assertRaises(ValueError): validate_claim_semantics(semantics)
        units = [{"claim_id": 0, "text": "The goal has not been reached."}]
        raw = dict(evidence="", assessable=True, issues=[], claims=[dict(claim_id=0,
            **{k:v for k,v in meaning(units[0]["text"], [assertion("goal_reached", False)]).items() if k != "quote"})])
        resolved = validate_claim_semantics(resolve_claim_ids(raw, units, semantics=True))
        self.assertEqual(resolved["claims"][0]["quote"], units[0]["text"])

    def test_public_v7_dispatch_keeps_honest_failure_credit(self):
        value = with_semantics("The requested annotation is absent.", edits=[False])
        value["visual"].update(annotation="missing", execution="absent")
        verdict = dict(schema_version="grounded_v7", assessment=value, scores=scores(value),
                       assessable=True, issues=[], evidence="Independent semantics")
        self.assertIs(validate_process(verdict), verdict)
        self.assertEqual(reward_components(verdict), dict(text=1.0, image=0.0, post_text=1.0))

    def test_process_text_blinding_and_visual_no_future_claim_leak(self):
        for operation_kind in ("navigate", "annotation"):
            with self.subTest(operation_kind=operation_kind), tempfile.TemporaryDirectory() as directory:
                image = Path(directory) / "original.png"
                Image.new("RGB", (12, 12), "white").save(image)
                quote = "The AFTER_SENTINEL path reaches the goal."
                value = with_semantics(quote, [assertion("goal_reached")])
                value["proposal"]["operation_kind"] = operation_kind
                drafts = [deepcopy(value["proposal"]), deepcopy(value["checks"]["state_blind_claims"]),
                          deepcopy(value["visual"]), deepcopy(value["observations"])]
                for draft in (drafts[0], drafts[1], drafts[3]):
                    draft["claims"][0].pop("quote")
                    draft["claims"][0]["claim_id"] = 0
                class Judge(GroundedQwenJudge):
                    # This fixture supplies the four v7 replies. The v8
                    # five-call flow has its own navigation-perception tests.
                    version = "grounded_v7"
                    def complete(self, messages, **kwargs):
                        self.requests.append(messages)
                        return drafts.pop(0)
                judge = Judge("unused")
                judge.requests = []
                constraints = dict(source="verified_original_image", image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(), grid=deepcopy(GRID))
                result = judge.process(StepEvidence("QUESTION_SENTINEL", str(image), str(image), str(image),
                    "Draw a route.", quote, task="Spatial_Navigation", original_constraints=constraints))
                self.assertEqual(len(judge.requests), 4)
                blind = judge.requests[1]
                fields = json.loads(blind[1]["content"][0]["text"])
                self.assertEqual(set(fields), {"text_units"} if operation_kind == "navigate" else {"text_units", "intended_operation"})
                self.assertEqual(len(blind[1]["content"]), 1)
                self.assertNotIn("QUESTION_SENTINEL", json.dumps(blind))
                self.assertNotIn("verified_original_grid", json.dumps(blind))
                visual_text = judge.requests[2][1]["content"][0]["text"]
                self.assertNotIn("AFTER_SENTINEL", visual_text)
                self.assertNotIn("state_blind_claims", visual_text)
                self.assertIn("state_blind_claims", result["assessment"]["checks"])
                self.assertEqual(reward_components(result)["post_text"], 0)


if __name__ == "__main__":
    unittest.main()
