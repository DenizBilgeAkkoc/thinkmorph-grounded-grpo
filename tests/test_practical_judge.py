"""V9 contract/reward regressions; supplied assessments are not model accuracy tests."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image, ImageDraw

from thinkmorph_grpo import practical_judge
from thinkmorph_grpo.checks import check_path
from thinkmorph_grpo.judge import StepEvidence, reward_components, validate_process
from thinkmorph_grpo.judge_v2 import (
    GroundedQwenJudge, OBSERVATION_PROMPT_V7, derive_scores,
    validate_claim_semantics_units, validate_observations_with_semantics,
    validate_proposal_units,
)
from test_coverage_normalization import draft
from test_navigation_perception import attach
from test_navigation_predicates import GRID, assertion
from test_proposal_grounding import proposal
from test_state_blind_claims import meaning, with_semantics


def observation(support="supported", **extra):
    return dict(evidence="Visible evidence", assessable=True, issues=[], claims=[
        dict(claim_id=0, evidence="Exact assertion inspected", support=support, **extra)])


def annotation_assessment(quote="Some of the fruit is highlighted."):
    value = with_semantics(quote)
    value["proposal"].update(operation_kind="annotation", operation="Box the whole fruit")
    value["proposal"]["claims"][0]["quote"] = "Draw a red box around the fruit."
    value["visual"].update(path_present=False, path_moves=[], target_match="partial", execution="partial")
    value["visual"]["objects"][0].update(name="fruit", evidence="Fruit remains visible")
    value["checks"].pop("verified_original_grid")
    value["checks"].pop("observed_path")
    return value


def verdict(value, version="grounded_v9"):
    return dict(schema_version=version, assessment=value,
        scores=derive_scores(value, True, version), assessable=True, issues=[],
        evidence="Controlled categorical evidence, not a live model judgment")


class PracticalContractTests(unittest.TestCase):
    def test_support_only_observation_has_one_factual_modality_owner(self):
        raw, units = draft("The fruit is yellow.", covered=False)
        semantics = practical_judge.validate_semantics(raw, units)
        for extra in ({}, {"kind": "intention"}, {"kind": "failure_report"}):
            with self.subTest(extra=extra):
                response = observation("partial", **extra)
                original = deepcopy((response, semantics, units))
                result = practical_judge.validate_observations(response, units, semantics)
                self.assertEqual(result["claims"][0]["kind"], "other")
                self.assertEqual(result["claims"][0]["support"], "partial")
                self.assertEqual((response, semantics, units), original)

    def test_factual_not_applicable_is_rejected_not_promoted(self):
        raw, units = draft("With the banana now highlighted, I will inspect it.",
            edit=[dict(predicate="intended_edit_complete", asserted=True)])
        semantics = practical_judge.validate_semantics(raw, units)
        self.assertFalse(semantics["claims"][0]["nonfactual_only"])
        for extra in ({}, {"kind": "intention"}):
            response = observation("not_applicable", **extra)
            original = deepcopy(response)
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                practical_judge.validate_observations(response, units, semantics)
            self.assertEqual(response, original)

    def test_nonfactual_observation_does_not_become_positive_factual_credit(self):
        raw, units = draft("I will draw the box.", covered=False, nonfactual=True)
        semantics = practical_judge.validate_semantics(raw, units)
        response = observation("supported", kind="other")
        original = deepcopy(response)
        result = practical_judge.validate_observations(response, units, semantics)
        self.assertEqual(response, original)
        self.assertEqual((result["claims"][0]["kind"], result["claims"][0]["support"]),
                         ("intention", "not_applicable"))
        value = annotation_assessment(units[0]["text"])
        value["observations"] = result
        value["checks"]["state_blind_claims"] = semantics
        self.assertEqual(reward_components(verdict(value))["post_text"], 0)

    def test_bare_answer_introduction_only_is_nonfactual_and_raw_is_preserved(self):
        for quote in ("Therefore, the answer is:", "The solution is:", "Thus, final answer:"):
            with self.subTest(quote=quote):
                raw, units = draft(quote)
                original = deepcopy(raw)
                result = practical_judge.validate_semantics(raw, units)
                self.assertEqual(raw, original)
                self.assertTrue(result["claims"][0]["nonfactual_only"])
                self.assertFalse(result["claims"][0]["all_factual_content_covered"])
                self.assertEqual(result["claims"][0]["quote"], quote)
                self.assertTrue(result["issues"])
        for quote in ("The answer is: banana.", "The solution is: U,R,R,D.",
                      "With the banana now highlighted, I will inspect it."):
            raw, units = draft(quote)
            with self.subTest(quote=quote):
                self.assertFalse(practical_judge.validate_semantics(raw, units)["claims"][0]["nonfactual_only"])

    def test_missing_ids_and_malformed_types_still_fail_before_normalization(self):
        raw, units = draft("Therefore, the answer is:")
        mutations = [lambda v: v["claims"][0].pop("claim_id"),
                     lambda v: v["claims"].clear(),
                     lambda v: v["claims"][0].update(claim_id=True),
                     lambda v: v["claims"][0].update(nonfactual_only="false"),
                     lambda v: v["claims"][0].update(all_factual_content_covered=1),
                     lambda v: v["claims"][0].update(evidence=4),
                     lambda v: v["claims"][0].update(navigation_claims="")]
        for index, mutate in enumerate(mutations):
            value = deepcopy(raw); mutate(value); original = deepcopy(value)
            with self.subTest(case=index), self.assertRaises(ValueError):
                practical_judge.validate_semantics(value, units)
            self.assertEqual(value, original)
        semantics = practical_judge.validate_semantics(*draft("The fruit is yellow.", covered=False))
        units = [{"claim_id": 0, "text": "The fruit is yellow."}]
        for changes in ({"claim_id": True}, {"evidence": 7}, {"support": True}):
            value = observation(); value["claims"][0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                practical_judge.validate_observations(value, units, semantics)
        value = observation(); value["claims"] = []
        with self.assertRaises(ValueError):
            practical_judge.validate_observations(value, units, semantics)

    def test_duplicate_support_keeps_worst_and_uncertainty_cannot_be_hidden(self):
        raw, units = draft("The fruit is fully boxed.", covered=False)
        semantics = practical_judge.validate_semantics(raw, units)
        value = observation("supported")
        value["claims"].append(dict(claim_id=0, evidence="One edge is outside", support="unsupported"))
        original = deepcopy(value)
        self.assertEqual(practical_judge.validate_observations(value, units, semantics)["claims"][0]["support"], "unsupported")
        self.assertEqual(value, original)
        value["claims"][0]["support"] = "uncertain"
        with self.assertRaisesRegex(ValueError, "Uncertain"):
            practical_judge.validate_observations(value, units, semantics)

    def test_proposal_support_only_keeps_false_premise_and_literal_provenance(self):
        raw, units = proposal("The target is already boxed.", [])
        raw["operation_kind"] = "annotation"
        raw["claims"][0].update(kind="intention", support="unsupported")
        original = deepcopy(raw)
        result = practical_judge.validate_proposal(raw, units)
        self.assertEqual(raw, original)
        self.assertEqual((result["claims"][0]["kind"], result["claims"][0]["support"]),
                         ("other", "unsupported"))
        value = annotation_assessment(); value["proposal"] = result
        self.assertEqual(reward_components(verdict(value))["text"], 0)
        for moves in (["U", "R", "R", "D"], ["R", "R"]):
            raw, units = proposal("Draw the route U,R,R,D.", moves)
            raw["claims"][0].pop("kind")
            if moves[0] == "U":
                self.assertEqual(practical_judge.validate_proposal(raw, units)["moves"], moves)
            else:
                with self.assertRaisesRegex(ValueError, "whole literal"):
                    practical_judge.validate_proposal(raw, units)

    def test_proposal_missing_units_and_invalid_support_still_rejected(self):
        for mutation in (lambda v: v["claims"][0].pop("claim_id"),
                         lambda v: v["claims"].clear(),
                         lambda v: v["claims"][0].update(claim_id=True),
                         lambda v: v["claims"][0].update(support=True)):
            raw, units = proposal("Draw the box.", [])
            raw["claims"][0].pop("kind"); mutation(raw)
            with self.assertRaises(ValueError):
                practical_judge.validate_proposal(raw, units)

    def test_malformed_root_shapes_and_issue_types_raise_validation_errors(self):
        semantic_raw, units = draft("Therefore, the answer is:")
        proposal_raw, proposal_units = proposal("Draw the box.", [])
        proposal_raw["claims"][0].pop("kind")
        semantics = practical_judge.validate_semantics(semantic_raw, units)
        contracts = (
            (semantic_raw, lambda value: practical_judge.validate_semantics(value, units)),
            (proposal_raw, lambda value: practical_judge.validate_proposal(value, proposal_units)),
            (observation(), lambda value: practical_judge.validate_observations(value, units, semantics)))
        for contract, (valid, validate) in enumerate(contracts):
            invalid = [None, [], "bad root", {}]
            for issues in ("bad", None, [7]):
                value = deepcopy(valid); value["issues"] = issues; invalid.append(value)
            for claims in (None, [None]):
                value = deepcopy(valid); value["claims"] = claims; invalid.append(value)
            for index, value in enumerate(invalid):
                original = deepcopy(value)
                with self.subTest(contract=contract, malformed=index), self.assertRaises(ValueError):
                    validate(value)
                self.assertEqual(value, original)

    def test_assess_retries_malformed_json_shapes_and_keeps_original_messages(self):
        semantic_raw, units = draft("Therefore, the answer is:")
        proposal_raw, proposal_units = proposal("Draw the box.", [])
        proposal_raw["claims"][0].pop("kind")
        for valid, validate in (
            (semantic_raw, lambda value: practical_judge.validate_semantics(value, units)),
            (proposal_raw, lambda value: practical_judge.validate_proposal(value, proposal_units))):
            for malformed in (None, [], dict(deepcopy(valid), issues="bad")):
                responses = [malformed, deepcopy(valid)]
                original_responses = deepcopy(responses)
                messages = [{"role": "system", "content": "Assess supplied evidence."}]
                original_messages = deepcopy(messages)

                class Judge(GroundedQwenJudge):
                    def complete(self, request, **kwargs):
                        self.requests.append((request, kwargs))
                        return responses[len(self.requests) - 1]

                judge = Judge("unused"); judge.requests = []
                with self.subTest(malformed=malformed):
                    result = judge._assess(messages, validate, max_tokens=173)
                    self.assertTrue(result["assessable"])
                    self.assertEqual(len(judge.requests), 2)
                    self.assertIn("Previous assessment failed validation", judge.requests[1][0][0]["content"])
                    self.assertEqual([call[1]["max_tokens"] for call in judge.requests], [173, 173])
                    self.assertEqual(messages, original_messages)
                    self.assertEqual(responses, original_responses)

    def test_assess_exhausted_malformed_attempts_never_return_a_default_verdict(self):
        raw, units = draft("The fruit is yellow.", covered=False)
        responses = [None, dict(raw, issues="bad")]

        class Judge(GroundedQwenJudge):
            def complete(self, messages, **kwargs):
                return responses.pop(0)

        judge = Judge("unused")
        with self.assertRaisesRegex(ValueError, "Assessment failed after two attempts"):
            judge._assess([{"role": "system", "content": "Assess evidence."}],
                          lambda value: practical_judge.validate_semantics(value, units))
        self.assertEqual(responses, [])

    def test_scope_downgrade_keeps_safe_segment_despite_unsafe_whole_route(self):
        quote = "The first move U is safe."
        raw, units = draft(quote, nav=[assertion("safe", route="U")])
        original = deepcopy(raw)
        semantics = practical_judge.validate_semantics(raw, units)
        self.assertEqual(raw, original)
        normalized = semantics["claims"][0]["navigation_claims"][0]
        self.assertEqual(normalized, assertion("safe", scope="unverified"))
        self.assertIn("symbolic check disabled", semantics["issues"][0])
        value = attach(with_semantics(quote), [[2, 0], [1, 0], [0, 0]])
        value["checks"]["observed_path"] = check_path(["U", "U"], GRID)
        self.assertFalse(value["checks"]["observed_path"]["legal"])
        value["checks"]["state_blind_claims"] = semantics
        value["observations"] = practical_judge.validate_observations(observation(), units, semantics)
        original = deepcopy(value)
        self.assertEqual(reward_components(verdict(value)), dict(text=1.0, image=0.0, post_text=1.0))
        self.assertEqual(value, original)
        # The SAME unsafe drawing does cap a genuine whole-route safety claim.
        quote = "The entire drawn route is safe."
        raw, units = draft(quote, nav=[assertion("safe")])
        semantics = practical_judge.validate_semantics(raw, units)
        value["checks"]["state_blind_claims"] = semantics
        value["observations"] = practical_judge.validate_observations(observation(), units, semantics)
        self.assertEqual(reward_components(verdict(value))["post_text"], 0)

    def test_scope_recovery_never_promotes_unsupported_segment(self):
        raw, units = draft("The first U reaches the goal.", nav=[assertion("goal_reached", route="U")])
        semantics = practical_judge.validate_semantics(raw, units)
        value = attach(with_semantics(units[0]["text"]), [[2, 0], [1, 0], [1, 1], [1, 2], [2, 2]])
        value["checks"]["observed_path"] = check_path(["U", "R", "R", "D"], GRID)
        value["checks"]["state_blind_claims"] = semantics
        value["observations"] = practical_judge.validate_observations(observation("unsupported"), units, semantics)
        self.assertEqual(reward_components(verdict(value))["post_text"], 0)

    def test_false_explicit_complete_route_still_zero_and_substitution_rejected(self):
        quote = "The complete route R,R is safe and reaches the goal."
        nav = [assertion(p, scope="stated_full_route", route="R,R") for p in ("safe", "goal_reached")]
        raw, units = draft(quote, nav=nav)
        semantics = practical_judge.validate_semantics(raw, units)
        self.assertEqual(semantics["claims"][0]["navigation_claims"], nav)
        value = attach(with_semantics(quote), [[2, 0], [1, 0], [1, 1], [1, 2], [2, 2]])
        value["checks"]["state_blind_claims"] = semantics
        value["observations"] = practical_judge.validate_observations(observation(), units, semantics)
        self.assertEqual(reward_components(verdict(value))["post_text"], 0)
        raw["claims"][0]["navigation_claims"][0]["route"] = "U,R,R,D"
        with self.assertRaisesRegex(ValueError, "whole literal"):
            practical_judge.validate_semantics(raw, units)

    def test_strict_v8_rejects_scope_modality_repairs_and_keeps_intro_factual(self):
        raw, units = draft("The first U is safe.", nav=[assertion("safe", route="U")])
        with self.assertRaisesRegex(ValueError, "Only stated_full_route"):
            validate_claim_semantics_units(raw, units, normalize_unverified=True)
        raw, units = draft("Therefore, the answer is:")
        self.assertFalse(validate_claim_semantics_units(raw, units)["claims"][0]["nonfactual_only"])
        raw, units = draft("The fruit is yellow.", covered=False)
        semantics = validate_claim_semantics_units(raw, units)
        with self.assertRaises(ValueError):
            validate_observations_with_semantics(observation("partial", kind="intention"), units, semantics)
        raw, units = proposal("Draw the box.", [])
        raw["claims"][0].pop("kind")
        with self.assertRaises(ValueError):
            validate_proposal_units(raw, units, literal_moves=True)


def focused_support_fixture():
    texts = ["I will inspect the box.", "Parking meters usually collect parking fees.",
             "The fruit is fully boxed.", "Bananas are often yellow."]
    units = [dict(claim_id=index, text=text) for index, text in enumerate(texts)]
    semantics = dict(evidence="Text-only meaning", assessable=True, issues=[], claims=[])
    for index, text in enumerate(texts):
        claim = meaning(text, nonfactual=index == 0)
        claim.pop("quote"); claim["claim_id"] = index
        semantics["claims"].append(claim)
    observations = dict(evidence="Scene and claims inspected", assessable=True, issues=[], claims=[
        dict(claim_id=index, evidence="Original support evidence", support=support)
        for index, support in enumerate(("not_applicable", "not_applicable", "unsupported", "not_applicable"))])
    return units, semantics, observations


class FocusedSupportRepairTests(unittest.TestCase):
    def test_disputed_indices_are_only_factual_sorted_unique_and_raw_unchanged(self):
        units, raw_semantics, raw = focused_support_fixture()
        semantics = practical_judge.validate_semantics(raw_semantics, units)
        raw["claims"].reverse()
        raw["claims"].append(deepcopy(raw["claims"][2]))  # Duplicate unit1 NA.
        original = deepcopy((raw, semantics, units))
        with self.assertRaises(practical_judge.FactualSupportRequired) as raised:
            practical_judge.validate_observations(raw, units, semantics)
        self.assertEqual(raised.exception.indices, [1, 3])
        self.assertEqual((raw, semantics, units), original)
        raw["claims"] = [claim for claim in raw["claims"] if claim["claim_id"] != 2]
        with self.assertRaises(ValueError) as missing:
            practical_judge.validate_observations(raw, units, semantics)
        self.assertNotIsInstance(missing.exception, practical_judge.FactualSupportRequired)

    def test_merge_changes_only_requested_na_units_preserving_raw_and_classification(self):
        units, raw_semantics, raw = focused_support_fixture()
        semantics = practical_judge.validate_semantics(raw_semantics, units)
        repair = dict(claims=[dict(claim_id=3, evidence="Typical fruit color", support="partial"),
                            dict(claim_id=1, evidence="Typical meter purpose", support="supported")])
        original = deepcopy((raw, repair, semantics))
        merged = practical_judge.merge_factual_support(raw, repair, [1, 3])
        result = practical_judge.validate_observations(merged, units, semantics)
        self.assertEqual((raw, repair, semantics), original)
        self.assertEqual(merged["claims"][0], raw["claims"][0])
        self.assertEqual(merged["claims"][2], raw["claims"][2])
        self.assertEqual([claim["support"] for claim in result["claims"]],
                         ["not_applicable", "supported", "unsupported", "partial"])
        self.assertEqual([claim["kind"] for claim in result["claims"]],
                         ["intention", "other", "other", "other"])
        self.assertEqual(len(merged["issues"]), 2)

    def test_repair_requires_exact_ids_and_factual_support_types(self):
        _, _, raw = focused_support_fixture()
        valid = dict(claims=[dict(claim_id=index, evidence="Independent support", support="supported")
                            for index in (1, 3)])
        invalid = [None, [], {}, dict(claims=None), dict(claims=[])]
        for edits in ({"claim_id": 0}, {"claim_id": True}, {"claim_id": "1"},
                      {"support": "not_applicable"}, {"support": 4}, {"evidence": None},
                      {"kind": "intention"}):
            value = deepcopy(valid); value["claims"][0].update(edits); invalid.append(value)
        value = deepcopy(valid); value["claims"].append(deepcopy(value["claims"][0])); invalid.append(value)
        for index, repair in enumerate(invalid):
            original = deepcopy((raw, repair))
            with self.subTest(case=index), self.assertRaises(ValueError):
                practical_judge.merge_factual_support(raw, repair, [1, 3])
            self.assertEqual((raw, repair), original)

    def test_duplicate_negative_evidence_survives_na_repair_and_uncertain_rejects(self):
        units, raw_semantics, raw = focused_support_fixture()
        semantics = practical_judge.validate_semantics(raw_semantics, units)
        raw["claims"].append(dict(claim_id=1, evidence="Existing contrary evidence", support="unsupported"))
        repair = dict(claims=[dict(claim_id=index, evidence="Repair sees support", support="supported")
                            for index in (1, 3)])
        original = deepcopy(raw)
        merged = practical_judge.merge_factual_support(raw, repair, [1, 3])
        result = practical_judge.validate_observations(merged, units, semantics)
        self.assertEqual(result["claims"][1]["support"], "unsupported")
        self.assertIn("Existing contrary evidence", result["claims"][1]["evidence"])
        self.assertEqual(raw, original)
        repair["claims"][0]["support"] = "uncertain"
        merged = practical_judge.merge_factual_support(raw, repair, [1, 3])
        with self.assertRaisesRegex(ValueError, "Uncertain"):
            practical_judge.validate_observations(merged, units, semantics)

    def test_known_unrequested_context_replies_cannot_overwrite_existing_negatives(self):
        units, raw_semantics, raw = focused_support_fixture()
        semantics = practical_judge.validate_semantics(raw_semantics, units)
        raw["claims"][1].update(support="unsupported", evidence="Existing negative general-fact judgment")
        repair = dict(claims=[dict(claim_id=index, evidence="Overcomplete reply", support="supported")
                            for index in (3, 0, 2, 1)])
        original = deepcopy((raw, repair, semantics))
        merged = practical_judge.merge_factual_support(raw, repair, [3])
        for index in (0, 1, 2):
            self.assertEqual(merged["claims"][index], raw["claims"][index])
        self.assertEqual(merged["claims"][3]["support"], "supported")
        result = practical_judge.validate_observations(merged, units, semantics)
        self.assertEqual([claim["support"] for claim in result["claims"]],
                         ["not_applicable", "unsupported", "unsupported", "supported"])
        self.assertEqual((raw, repair, semantics), original)

    def test_overcomplete_repair_still_requires_requested_id_once_and_known_ids(self):
        _, _, raw = focused_support_fixture()
        answer = dict(claim_id=3, evidence="Requested support", support="supported")
        context = dict(claim_id=0, evidence="Known context", support="supported")
        repairs = [dict(claims=[context]),
                   dict(claims=[answer, deepcopy(answer), context]),
                   dict(claims=[answer, dict(context, claim_id=99)]),
                   dict(claims=[answer, context, deepcopy(context)])]
        for index, repair in enumerate(repairs):
            original = deepcopy((raw, repair))
            with self.subTest(case=index), self.assertRaises(ValueError):
                practical_judge.merge_factual_support(raw, repair, [3])
            self.assertEqual((raw, repair), original)

    def run_focused_process(self, need_repair=True, uncertain=False, marker_only=False):
        with tempfile.TemporaryDirectory() as directory:
            before = Path(directory) / "before.png"; after = Path(directory) / "after.png"
            image = Image.new("RGB", (24, 24), "white"); image.save(before)
            ImageDraw.Draw(image).rectangle((4, 4, 14, 14), outline="red", width=2); image.save(after)
            units, semantic_response, observation_response = focused_support_fixture()
            value = annotation_assessment()
            proposal_response = deepcopy(value["proposal"])
            proposal_response["claims"][0].pop("quote")
            proposal_response["claims"][0].pop("kind")
            proposal_response["claims"][0]["claim_id"] = 0
            repair = dict(claims=[dict(claim_id=index, evidence="Fresh focused assessment",
                                      support="uncertain" if uncertain else "supported") for index in (1, 3)])
            if not need_repair:
                for index in (1, 3): observation_response["claims"][index]["support"] = "supported"
            if marker_only:
                for claim in observation_response["claims"]:
                    claim["evidence"] += "; support=" + claim.pop("support")
            responses = [proposal_response, semantic_response, deepcopy(value["visual"]), observation_response]
            if need_repair: responses.append(repair)
            if uncertain or marker_only:
                responses.extend([deepcopy(observation_response), deepcopy(repair)])
            original = deepcopy(responses)

            class Judge(GroundedQwenJudge):
                def complete(self, messages, **kwargs):
                    self.requests.append((messages, kwargs))
                    return responses[len(self.requests) - 1]

                def terminal(self, *args, **kwargs):
                    raise AssertionError("Process repair must never request a terminal label")

            judge = Judge("unused"); judge.version = "grounded_v9"; judge.requests = []
            evidence = StepEvidence("QUESTION_SENTINEL", str(before), str(before), str(after),
                "Draw a red box around the fruit. <answer>BEFORE_TERMINAL_SECRET</answer>",
                "\n".join(unit["text"] for unit in units) + "\n<answer>AFTER_TERMINAL_SECRET</answer>",
                history=("Earlier inspection. <answer>HISTORY_TERMINAL_SECRET</answer>",), task="Visual_Search")
            result = None
            if uncertain:
                with self.assertRaisesRegex(ValueError, "Assessment failed after two attempts"):
                    judge.process(evidence)
            else:
                result = judge.process(evidence)
            self.assertEqual(responses, original)
            return result, judge.requests, units

    def test_process_requests_all_disputed_units_once_without_terminal_leak(self):
        result, requests, units = self.run_focused_process()
        self.assertEqual(len(requests), 5)
        focused, kwargs = requests[4]
        self.assertEqual(focused[0]["content"], practical_judge.FOCUSED_SUPPORT_PROMPT)
        self.assertEqual(kwargs["max_tokens"], 800)
        fields = json.loads(focused[1]["content"][0]["text"])
        self.assertEqual(set(fields), {"problem", "task", "prior_generated_text", "text_units",
                                     "context_after_units", "intended_operation", "independent_visual_assessment"})
        self.assertEqual(fields["text_units"], [units[1], units[3]])
        self.assertEqual(fields["context_after_units"], units)
        self.assertEqual(fields["intended_operation"], "Box the whole fruit")
        self.assertEqual(sum(block["type"] == "image_url" for block in focused[1]["content"]), 2)
        for secret in ("BEFORE_TERMINAL_SECRET", "AFTER_TERMINAL_SECRET", "HISTORY_TERMINAL_SECRET"):
            self.assertNotIn(secret, json.dumps(requests))
        semantics_request = requests[1][0]
        self.assertEqual(len(semantics_request[1]["content"]), 1)
        self.assertNotIn("QUESTION_SENTINEL", json.dumps(semantics_request))
        self.assertEqual([claim["nonfactual_only"] for claim in result["assessment"]["checks"]["state_blind_claims"]["claims"]],
                         [True, False, False, False])
        self.assertEqual(reward_components(result)["post_text"], 0)  # Existing unsupported scene claim survives.
        no_repair_result, ordinary_requests, _ = self.run_focused_process(need_repair=False)
        self.assertEqual(len(ordinary_requests), 4)
        self.assertTrue(all(request[0][0]["content"] != practical_judge.FOCUSED_SUPPORT_PROMPT for request in ordinary_requests))
        self.assertEqual(reward_components(no_repair_result), reward_components(result))

    def test_uncertain_focused_calls_exhaust_outer_two_attempts_without_default_reward(self):
        result, requests, _ = self.run_focused_process(uncertain=True)
        self.assertIsNone(result)
        self.assertEqual(len(requests), 7)  # Three prior stages, then observation+repair twice.
        focused = [request for request, _ in requests if request[0]["content"] == practical_judge.FOCUSED_SUPPORT_PROMPT]
        self.assertEqual(len(focused), 2)
        self.assertIn("Previous assessment failed validation", requests[5][0][0]["content"])

    def test_recovered_na_markers_compose_with_focused_repair_without_losing_other_labels(self):
        result, requests, units = self.run_focused_process(marker_only=True)
        self.assertEqual(len(requests), 5)
        fields = json.loads(requests[4][0][1]["content"][0]["text"])
        self.assertEqual(fields["text_units"], [units[1], units[3]])
        self.assertEqual([claim["support"] for claim in result["assessment"]["observations"]["claims"]],
                         ["not_applicable", "supported", "unsupported", "supported"])
        self.assertEqual(reward_components(result)["post_text"], 0)


class ExplicitSupportRecoveryTests(unittest.TestCase):
    def test_exact_trailing_markers_recover_both_contracts_without_mutating_raw(self):
        for support in ("supported", "partial", "unsupported", "not_applicable"):
            for suffix in ("", ".", ". \n"):
                nonfactual = support == "not_applicable"
                text = "I will inspect the fruit." if nonfactual else "The fruit is yellow."
                semantic_raw, units = draft(text, covered=False, nonfactual=nonfactual)
                semantics = practical_judge.validate_semantics(semantic_raw, units)
                raw = observation(); raw["claims"][0].pop("support")
                raw["claims"][0]["evidence"] = "Independent assessment; support=" + support + suffix
                proposed, proposal_units = proposal(text, [])
                proposed["claims"] = deepcopy(raw["claims"])
                original = deepcopy((raw, proposed))
                with self.subTest(support=support, suffix=suffix):
                    observed = practical_judge.validate_observations(raw, units, semantics)
                    resolved_proposal = practical_judge.validate_proposal(proposed, proposal_units)
                    self.assertEqual(observed["claims"][0]["support"], support)
                    self.assertEqual(resolved_proposal["claims"][0]["support"], support)
                    self.assertEqual(observed["claims"][0]["evidence"], raw["claims"][0]["evidence"])
                    self.assertTrue(any("Recovered explicit trailing support" in issue for issue in observed["issues"]))
                    self.assertEqual((raw, proposed), original)

    def test_ambiguous_conflicting_nontrailing_and_prose_labels_are_not_recovered(self):
        raw_semantics, units = draft("The fruit is yellow.", covered=False)
        semantics = practical_judge.validate_semantics(raw_semantics, units)
        for evidence in ("support=supported; support=supported", "support=unsupported; support=supported",
                         "support=supported but one edge is missing", "This claim is supported.",
                         "support: supported", "support=invalid", "The target is visible."):
            raw = observation(); raw["claims"][0].pop("support"); raw["claims"][0]["evidence"] = evidence
            proposed, proposal_units = proposal(units[0]["text"], []); proposed["claims"] = deepcopy(raw["claims"])
            original = deepcopy((raw, proposed))
            with self.subTest(evidence=evidence):
                with self.assertRaises(ValueError):
                    practical_judge.validate_observations(raw, units, semantics)
                with self.assertRaises(ValueError):
                    practical_judge.validate_proposal(proposed, proposal_units)
                self.assertEqual((raw, proposed), original)

    def test_existing_support_field_is_never_overwritten_by_evidence_marker(self):
        raw_semantics, units = draft("The fruit is fully boxed.", covered=False)
        semantics = practical_judge.validate_semantics(raw_semantics, units)
        raw = observation("unsupported")
        raw["claims"][0]["evidence"] = "Conflicting formatting marker support=supported"
        proposed, proposal_units = proposal(units[0]["text"], [])
        proposed["claims"] = deepcopy(raw["claims"])
        original = deepcopy((raw, proposed))
        self.assertEqual(practical_judge.validate_observations(raw, units, semantics)["claims"][0]["support"], "unsupported")
        self.assertEqual(practical_judge.validate_proposal(proposed, proposal_units)["claims"][0]["support"], "unsupported")
        self.assertEqual((raw, proposed), original)
        raw["claims"][0]["support"] = None  # Present but invalid is not an absent field.
        with self.assertRaises(ValueError):
            practical_judge.validate_observations(raw, units, semantics)

    def test_uncertain_marker_still_rejects_and_factual_na_requires_model_assessment(self):
        raw_semantics, units = draft("The fruit is yellow.", covered=False)
        semantics = practical_judge.validate_semantics(raw_semantics, units)
        for support in ("uncertain", "not_applicable"):
            raw = observation(); raw["claims"][0].pop("support")
            raw["claims"][0]["evidence"] = "Explicit judge label support=" + support
            original = deepcopy(raw)
            exception = practical_judge.FactualSupportRequired if support == "not_applicable" else ValueError
            with self.subTest(support=support), self.assertRaises(exception):
                practical_judge.validate_observations(raw, units, semantics)
            self.assertEqual(raw, original)
        proposed, proposal_units = proposal(units[0]["text"], [])
        proposed["claims"] = [dict(claim_id=0, evidence="Explicit judge label support=uncertain")]
        with self.assertRaisesRegex(ValueError, "Uncertain"):
            practical_judge.validate_proposal(proposed, proposal_units)


class PracticalRewardTests(unittest.TestCase):
    def test_useful_partial_annotation_between_complete_and_weak_or_failed(self):
        partial = annotation_assessment(); before = deepcopy(partial)
        partial_verdict = verdict(partial)
        self.assertIs(validate_process(partial_verdict), partial_verdict)
        self.assertEqual(partial_verdict["scores"]["image_execution"], 3)
        self.assertEqual(reward_components(partial_verdict)["image"], .75)
        self.assertEqual(partial, before)
        complete = deepcopy(partial); complete["visual"].update(execution="complete", target_match="yes")
        weak = deepcopy(partial); weak["visual"]["utility"] = "limited"
        self.assertEqual(reward_components(verdict(complete))["image"], 1)
        self.assertEqual(reward_components(verdict(weak))["image"], .375)
        for updates in ({"execution": "wrong", "target_match": "no"},
                        {"annotation": "missing", "execution": "absent"}):
            failed = deepcopy(partial); failed["visual"].update(updates)
            self.assertEqual(reward_components(verdict(failed))["image"], 0)
        for status in ("missing", "duplicated"):
            destroyed = deepcopy(partial); destroyed["visual"]["objects"][0]["status"] = status
            self.assertEqual(reward_components(verdict(destroyed))["image"], 0)
        tampered = deepcopy(partial_verdict); tampered["scores"]["image_execution"] = 4
        with self.assertRaisesRegex(ValueError, "contradict"):
            reward_components(tampered)

    def test_partial_extent_does_not_automatically_upgrade_transition_or_utility(self):
        value = annotation_assessment()
        value["visual"].update(transition="partial", utility="limited")
        result = verdict(value)
        self.assertEqual(result["scores"]["step_consistency"], 2)
        self.assertEqual(result["scores"]["reasoning_utility"], 2)
        self.assertEqual(reward_components(result)["image"], .1875)

    def test_higher_partial_extent_is_v9_non_grid_annotation_only(self):
        value = annotation_assessment()
        self.assertEqual(verdict(value, "grounded_v8")["scores"]["image_execution"], 2)
        self.assertEqual(reward_components(verdict(value, "grounded_v8"))["image"], .5)
        for operation in ("navigate", "crop", "rearrange"):
            other = deepcopy(value); other["proposal"]["operation_kind"] = operation
            with self.subTest(operation=operation):
                self.assertEqual(verdict(other)["scores"]["image_execution"], 2)
        grid = attach(with_semantics("The arrow is partial."), [[2, 0], [1, 0]])
        grid["proposal"]["operation_kind"] = "annotation"
        grid["visual"].update(execution="partial", target_match="partial")
        self.assertEqual(verdict(grid)["scores"]["image_execution"], 2)

    def test_honest_failure_survives_bad_image_but_mixed_success_remains_factual(self):
        for quote, asserted, expected in (
            ("The requested box was not added.", False, 1),
            ("With the banana now highlighted, I will inspect it.", True, 0)):
            raw, units = draft(quote, edit=[dict(predicate="intended_edit_complete", asserted=asserted)])
            semantics = practical_judge.validate_semantics(raw, units)
            value = annotation_assessment(quote)
            value["visual"].update(annotation="missing", execution="absent")
            value["checks"]["state_blind_claims"] = semantics
            value["observations"] = practical_judge.validate_observations(
                observation("supported", kind="intention"), units, semantics)
            self.assertFalse(semantics["claims"][0]["nonfactual_only"])
            self.assertEqual(value["observations"]["claims"][0]["kind"], "other")
            with self.subTest(quote=quote):
                result = reward_components(verdict(value))
                self.assertEqual(result["image"], 0)
                self.assertEqual(result["post_text"], expected)

    def test_process_wires_support_only_contract_and_visual_extent_prompt_only_in_v9(self):
        for version, expected_image in (("grounded_v8", .5), ("grounded_v9", .75)):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as directory:
                original = Path(directory) / "before.png"
                after = Path(directory) / "after.png"
                image = Image.new("RGB", (24, 24), "white"); image.save(original)
                ImageDraw.Draw(image).rectangle((4, 4, 14, 14), outline="red", width=2); image.save(after)
                quote = "Some of the AFTER_SENTINEL fruit is highlighted."
                value = annotation_assessment(quote)
                drafts = [deepcopy(value["proposal"]), deepcopy(value["checks"]["state_blind_claims"]),
                          deepcopy(value["visual"]), deepcopy(value["observations"])]
                for index in (0, 1, 3):
                    drafts[index]["claims"][0].pop("quote")
                    drafts[index]["claims"][0]["claim_id"] = 0
                if version == "grounded_v9":
                    for index in (0, 3): drafts[index]["claims"][0].pop("kind")
                saved = deepcopy(drafts)

                class Judge(GroundedQwenJudge):
                    def complete(self, messages, **kwargs):
                        self.requests.append(messages)
                        return drafts[len(self.requests) - 1]

                judge = Judge("unused"); judge.version = version; judge.requests = []
                result = judge.process(StepEvidence("QUESTION_SENTINEL", str(original), str(original), str(after),
                    "Draw a red box around the fruit.", quote, task="Visual_Search"))
                self.assertEqual(len(judge.requests), 4)
                self.assertEqual(drafts, saved)
                self.assertEqual(result["schema_version"], version)
                self.assertEqual(reward_components(result)["image"], expected_image)
                visual_prompt = judge.requests[2][0]["content"]
                self.assertEqual(practical_judge.VISUAL_ADDENDUM in visual_prompt, version == "grounded_v9")
                expected_prompt = practical_judge.OBSERVATION_PROMPT if version == "grounded_v9" else OBSERVATION_PROMPT_V7
                self.assertEqual(judge.requests[3][0]["content"], expected_prompt)
                semantic_request = judge.requests[1]
                self.assertEqual(len(semantic_request[1]["content"]), 1)
                fields = json.loads(semantic_request[1]["content"][0]["text"])
                self.assertEqual(set(fields), {"text_units", "intended_operation"})
                self.assertNotIn("QUESTION_SENTINEL", json.dumps(semantic_request))
                self.assertNotIn("AFTER_SENTINEL", judge.requests[2][1]["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
