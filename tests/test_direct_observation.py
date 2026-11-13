"""Direct observation contracts and call isolation; these are not accuracy tests."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from urllib.error import URLError

from PIL import Image, ImageDraw

from thinkmorph_grpo import practical_judge
from thinkmorph_grpo.judge import StepEvidence, reward_components, validate_process
from thinkmorph_grpo.judge_v2 import (
    DirectObservationQwenJudge, GroundedQwenJudge, JudgeAssessmentError,
    derive_scores, validate_visual_with_checks,
)
from test_component_failures import missing_bar_checks
from test_navigation_perception import attach
from test_practical_judge import annotation_assessment, observation as legacy_observation, verdict
from test_state_blind_claims import with_semantics


def observation(support="supported", scope=None):
    value = legacy_observation(support)
    value["claims"][0]["scope"] = scope or ("nonfactual" if support == "not_applicable" else "current_state")
    return value


def direct_assessment(quote="Some of the fruit is highlighted.", support="supported"):
    value = annotation_assessment(quote)
    value["checks"].pop("state_blind_claims")
    value["observations"] = practical_judge.validate_direct_observations(
        observation(support), [{"claim_id": 0, "text": quote}])
    return value


class DirectObservationTests(unittest.TestCase):
    def test_nonfactual_scope_excludes_accidental_support_with_audit(self):
        quote = "I will extend this route to reach the goal."
        raw = observation("supported", scope="nonfactual")
        original = deepcopy(raw)
        result = practical_judge.validate_direct_observations(raw, [{"claim_id": 0, "text": quote}])
        self.assertEqual(raw, original)
        self.assertEqual(result["claims"][0]["support"], "not_applicable")
        self.assertEqual(result["claims"][0]["kind"], "intention")
        self.assertIn("Same-call scope identifies nonfactual", result["issues"][0])
        value = direct_assessment(quote)
        value["observations"] = result
        self.assertEqual(reward_components(verdict(value, "grounded_v10"))["post_text"], 0.)

    def test_factual_scope_cannot_silently_exclude_its_assertion(self):
        for scope in ("current_state", "stated_route", "other_fact"):
            raw = observation("not_applicable", scope=scope)
            original = deepcopy(raw)
            with self.subTest(scope=scope), self.assertRaisesRegex(ValueError, "Factual scope"):
                practical_judge.validate_direct_observations(raw, [{"claim_id": 0, "text": "A fact is asserted."}])
            self.assertEqual(raw, original)

    def test_unknown_or_missing_scope_requires_assessment_instead_of_guessing(self):
        for scope in ("future", True, None):
            raw = observation()
            if scope is None:
                raw["claims"][0].pop("scope")
            else:
                raw["claims"][0]["scope"] = scope
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                practical_judge.validate_direct_observations(raw, [{"claim_id": 0, "text": "Some text."}])

    def test_stated_route_scope_alone_does_not_promote_partial_or_false_claim(self):
        quote = "The full route U,R,R,D is safe and the grid has five rows."
        for support, expected in (("partial", .5), ("unsupported", 0.)):
            raw = observation(support, scope="stated_route")
            result = practical_judge.validate_direct_observations(raw, [{"claim_id": 0, "text": quote}])
            value = direct_assessment(quote)
            value["observations"] = result
            with self.subTest(support=support):
                self.assertEqual(reward_components(verdict(value, "grounded_v10"))["post_text"], expected)

    def test_exact_source_quotes_and_order_are_restored_without_mutating_raw_reply(self):
        units = [{"claim_id": 0, "text": "The box covers part of the fruit."},
                 {"claim_id": 1, "text": "I will inspect it next."}]
        raw = observation("not_applicable")
        raw["claims"][0]["claim_id"] = 1
        raw["claims"].append(dict(claim_id=0, scope="current_state", evidence="Partial box visible", support="supported"))
        original = deepcopy((raw, units))
        result = practical_judge.validate_direct_observations(raw, units)
        self.assertEqual((raw, units), original)
        self.assertEqual([claim["quote"] for claim in result["claims"]], [u["text"] for u in units])
        self.assertEqual([claim["kind"] for claim in result["claims"]], ["other", "intention"])
        self.assertEqual([claim["support"] for claim in result["claims"]], ["supported", "not_applicable"])

    def test_missing_duplicate_foreign_and_boolean_ids_are_rejected(self):
        units = [{"claim_id": 0, "text": "The fruit is yellow."},
                 {"claim_id": 1, "text": "The box is red."}]
        for ids in ([0], [0, 0], [0, 2], [0, True]):
            raw = observation()
            raw["claims"] = [dict(claim_id=i, scope="current_state", evidence="Visible", support="supported") for i in ids]
            original = deepcopy(raw)
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                practical_judge.validate_direct_observations(raw, units)
            self.assertEqual(raw, original)

    def test_uncertain_and_unassessable_observations_never_become_zero_or_positive(self):
        units = [{"claim_id": 0, "text": "The fruit is highlighted."}]
        for raw in (observation("uncertain"), dict(observation(), assessable=False)):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                practical_judge.validate_direct_observations(raw, units)

    def test_explicit_trailing_support_recovery_preserves_audit(self):
        raw = observation()
        raw["claims"][0].pop("support")
        raw["claims"][0]["evidence"] = "Box stops before the target edge; support=partial."
        original = deepcopy(raw)
        result = practical_judge.validate_direct_observations(raw, [{"claim_id": 0, "text": "The whole fruit is boxed."}])
        self.assertEqual(raw, original)
        self.assertEqual(result["claims"][0]["support"], "partial")
        self.assertTrue(result["issues"])

    def test_bad_image_does_not_erase_supported_failure_report(self):
        value = direct_assessment("The requested box was not added.")
        value["visual"].update(annotation="missing", execution="absent", utility="none")
        result = verdict(value, version="grounded_v10")
        self.assertEqual(reward_components(result), dict(text=1., image=0., post_text=1.))
        self.assertEqual(validate_process(result), result)
        self.assertNotIn("state_blind_claims", result["assessment"]["checks"])

    def test_useful_partial_image_and_nonfactual_text_keep_independent_rewards(self):
        value = direct_assessment("I will inspect the fruit next.", "not_applicable")
        self.assertEqual(reward_components(verdict(value, "grounded_v10")),
                         dict(text=1., image=.75, post_text=0.))

    def test_direct_unsupported_claim_is_not_promoted_by_a_good_image(self):
        value = direct_assessment("The entire fruit is covered.", "unsupported")
        self.assertEqual(reward_components(verdict(value, "grounded_v10"))["post_text"], 0.)
        self.assertEqual(value["observations"]["claims"][0]["support"], "unsupported")

    def test_confirmed_chart_failure_retains_v9_component_zero_rule(self):
        value = direct_assessment("The numeric label remains visible.")
        value["checks"].update(missing_bar_checks())
        raw_visual = deepcopy(value["visual"])
        value["visual"] = validate_visual_with_checks(value["visual"], value["checks"], "grounded_v10")
        self.assertEqual(raw_visual["issues"], [])
        self.assertIn("Confirmed image component failure", value["visual"]["issues"][-1])
        self.assertEqual(reward_components(verdict(value, "grounded_v10")),
                         dict(text=1., image=0., post_text=1.))

    def test_navigation_image_disagreement_still_fails_without_semantics_stage(self):
        value = attach(with_semantics("The arrow is visible."), [[2, 0], [1, 0]])
        value["checks"].pop("state_blind_claims")
        value["visual"]["path_moves"] = ["U", "R"]
        with self.assertRaisesRegex(ValueError, "Conflicting navigation perception"):
            derive_scores(value, True, "grounded_v10")

    def test_process_uses_three_calls_and_keeps_future_text_out_of_image_assessment(self):
        with tempfile.TemporaryDirectory() as directory:
            before = Path(directory) / "before.png"
            after = Path(directory) / "after.png"
            image = Image.new("RGB", (24, 24), "white"); image.save(before)
            ImageDraw.Draw(image).rectangle((4, 4, 14, 14), outline="red", width=2); image.save(after)
            quote = "Some of the AFTER_SENTINEL fruit is highlighted."
            value = direct_assessment(quote)
            drafts = [deepcopy(value["proposal"]), deepcopy(value["visual"]), observation()]
            drafts[0]["claims"][0].pop("quote")
            drafts[0]["claims"][0].pop("kind")
            drafts[0]["claims"][0]["claim_id"] = 0
            original = deepcopy(drafts)

            class Judge(DirectObservationQwenJudge):
                def complete(self, messages, **kwargs):
                    self.requests.append(messages)
                    return drafts[len(self.requests) - 1]

            judge = Judge("unused"); judge.requests = []
            result = judge.process(StepEvidence("QUESTION_SENTINEL", str(before), str(before), str(after),
                "Draw a red box around the fruit.", quote, task="Visual_Search"))
            self.assertEqual(len(judge.requests), 3)
            self.assertEqual(drafts, original)
            self.assertEqual(result["schema_version"], "grounded_v10")
            self.assertEqual(reward_components(result)["image"], .75)
            self.assertNotIn("state_blind_claims", result["assessment"]["checks"])
            for request in judge.requests[:2]:
                self.assertNotIn("AFTER_SENTINEL", json.dumps(request))
            self.assertIn(practical_judge.VISUAL_ADDENDUM, judge.requests[1][0]["content"])
            self.assertEqual(judge.requests[2][0]["content"], practical_judge.DIRECT_OBSERVATION_PROMPT)
            fields = json.loads(judge.requests[2][1]["content"][0]["text"])
            self.assertEqual(fields["text_after_units"], [{"claim_id": 0, "text": quote}])
            self.assertIn("intended_operation", fields)
            self.assertIn("independent_visual_assessment", fields)
            self.assertNotIn("state_blind_claims", fields["independent_checks"])

    def test_bounded_direct_observation_retry_stays_typed_and_transport_propagates(self):
        class Judge(DirectObservationQwenJudge):
            def complete(self, messages, **kwargs):
                self.calls += 1
                if self.transport:
                    raise URLError("endpoint unavailable")
                return observation("uncertain")

        judge = Judge("unused"); judge.calls = 0; judge.transport = False
        units = [{"claim_id": 0, "text": "The target is highlighted."}]
        messages = [{"role": "system", "content": "Assess observation"}]
        validator = lambda raw: practical_judge.validate_direct_observations(raw, units)
        with self.assertRaises(JudgeAssessmentError):
            judge._assess(messages, validator)
        self.assertEqual(judge.calls, 2)
        judge.calls = 0; judge.transport = True
        with self.assertRaises(URLError):
            judge._assess(messages, validator)
        self.assertEqual(judge.calls, 1)

    def test_terminal_implementation_and_baseline_protocol_are_unchanged(self):
        self.assertIs(DirectObservationQwenJudge.terminal, GroundedQwenJudge.terminal)
        self.assertIs(DirectObservationQwenJudge.terminal_with_constraints, GroundedQwenJudge.terminal_with_constraints)
        self.assertEqual(DirectObservationQwenJudge.terminal_version, GroundedQwenJudge.terminal_version)
        self.assertEqual(GroundedQwenJudge.version, "grounded_v9")
        self.assertEqual(DirectObservationQwenJudge.version, "grounded_v10")
        self.assertEqual(DirectObservationQwenJudge.protocol_revision, "direct_observation_scoped_json_20261110")

    def test_direct_training_mode_requires_a_matching_judge_recipe(self):
        from thinkmorph_grpo.train import parse_args, validate_judge_recipe
        required = ["--preview-config", "preview.json", "--recipe", "recipe.json",
                    "--manifest", "manifest.json", "--output", "output", "--judge-url", "unused"]
        self.assertEqual(parse_args(required).judge_observation_mode, "separate")
        self.assertEqual(parse_args(required + ["--judge-observation-mode", "direct"]).judge_observation_mode,
                         "direct")
        baseline, direct = GroundedQwenJudge("unused"), DirectObservationQwenJudge("unused")
        recipe = dict(judge_version=baseline.version, terminal_policy_version=baseline.terminal_version,
                      judge_protocol_revision=baseline.protocol_revision)
        validate_judge_recipe(recipe, baseline)
        with self.assertRaisesRegex(ValueError, "versions differ"):
            validate_judge_recipe(recipe, direct)
        recipe["judge_version"] = direct.version
        with self.assertRaisesRegex(ValueError, "protocol revisions differ"):
            validate_judge_recipe(recipe, direct)
        recipe["judge_protocol_revision"] = direct.protocol_revision
        validate_judge_recipe(recipe, direct)


if __name__ == "__main__":
    unittest.main()
