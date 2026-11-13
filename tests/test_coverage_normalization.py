"""Coverage normalization only removes proof eligibility; raw/meaning stay intact."""
from copy import deepcopy
import json
import unittest

from thinkmorph_grpo.judge_v2 import (validate_claim_semantics_units,
                                    validate_claim_semantics, derive_scores)
from test_navigation_predicates import assertion
from test_state_blind_claims import with_semantics


def draft(text="The fruit has a bright yellow peel.", covered=True, nav=(), edit=(), nonfactual=False):
    units = [{"claim_id": 0, "text": text}]
    raw = {"evidence": "linguistic assessment", "assessable": True, "issues": [],
           "claims": [{"claim_id": 0, "evidence": "fruit has bright yellow peel",
                       "navigation_claims": list(nav), "edit_claims": list(edit),
                       "all_factual_content_covered": covered, "nonfactual_only": nonfactual}]}
    return raw, units


class CoverageDowngradeTests(unittest.TestCase):
    def test_empty_predicates_clear_only_coverage_and_keep_original_rpc_unchanged(self):
        raw, units = draft()
        original = deepcopy(raw)
        value = validate_claim_semantics_units(raw, units)
        self.assertEqual(raw, original)
        expected = {key: item for key, item in original["claims"][0].items() if key != "claim_id"}
        expected.update(quote=units[0]["text"], all_factual_content_covered=False)
        self.assertEqual(value["claims"][0], expected)
        self.assertEqual(len(value["issues"]), 1)
        log = json.loads(value["issues"][0].split(": ", 1)[1])
        self.assertEqual(log["claim_id"], 0)
        self.assertIs(log["original_value"], True)
        self.assertIs(log["normalized_value"], False)
        self.assertIs(validate_claim_semantics(value), value)

    def test_observation_support_is_never_promoted(self):
        for support, expected in (("unsupported", 0), ("partial", 2), ("supported", 4)):
            raw, units = draft()
            value = with_semantics(units[0]["text"], support=support)
            value["checks"]["state_blind_claims"] = validate_claim_semantics_units(raw, units)
            before = deepcopy(value)
            self.assertEqual(derive_scores(value, True, "grounded_v7")["observation_grounding"], expected)
            self.assertEqual(value, before)

    def test_explicit_routes_polarity_scope_and_false_route_proof_unchanged(self):
        quote = "The proposed full route R,R is safe and reaches the goal."
        nav = [assertion(p, scope="stated_full_route", route="R,R") for p in ("safe", "goal_reached")]
        raw, units = draft(quote, nav=nav)
        resolved = validate_claim_semantics_units(raw, units)
        self.assertTrue(resolved["claims"][0]["all_factual_content_covered"])
        self.assertEqual(resolved["claims"][0]["navigation_claims"], nav)
        self.assertEqual(resolved["issues"], [])
        value = with_semantics(quote, nav, covered=True, moves=("U", "R", "R", "D"))
        value["checks"]["state_blind_claims"] = resolved
        self.assertEqual(derive_scores(value, True, "grounded_v7")["observation_grounding"], 0)
        raw["claims"][0]["navigation_claims"][0]["route"] = "U,R,R,D"
        with self.assertRaises(ValueError):
            validate_claim_semantics_units(raw, units)

    def test_edit_polarity_nonfactual_flags_and_quotes_stay_unchanged(self):
        quote = "The requested highlight was not added."
        edits = [{"predicate": "intended_edit_complete", "asserted": False}]
        raw, units = draft(quote, edit=edits)
        resolved = validate_claim_semantics_units(raw, units)
        self.assertEqual(resolved["claims"][0]["edit_claims"], edits)
        self.assertEqual(resolved["claims"][0]["quote"], quote)
        self.assertFalse(resolved["claims"][0]["nonfactual_only"])
        self.assertEqual(resolved["issues"], [])
        raw, units = draft("I will draw the box.", nonfactual=True)
        resolved = validate_claim_semantics_units(raw, units)
        self.assertTrue(resolved["claims"][0]["nonfactual_only"])
        self.assertFalse(resolved["claims"][0]["all_factual_content_covered"])

    def test_malformed_types_uncertainty_and_id_coverage_still_reject(self):
        mutations = [lambda r: r["claims"][0].update(all_factual_content_covered="true"),
                     lambda r: r["claims"][0].update(nonfactual_only="false"),
                     lambda r: r["claims"][0].update(navigation_claims=None),
                     lambda r: r["claims"][0].update(edit_claims={}),
                     lambda r: r["claims"][0].update(claim_id=True),
                     lambda r: r["claims"].append(deepcopy(r["claims"][0])),
                     lambda r: r.update(claims=[]), lambda r: r.update(assessable=False)]
        for change in mutations:
            raw, units = draft()
            change(raw)
            original = deepcopy(raw)
            with self.subTest(raw=raw), self.assertRaises((ValueError, TypeError, KeyError)):
                validate_claim_semantics_units(raw, units)
            self.assertEqual(raw, original)

    def test_logged_ids_match_original_units_when_raw_order_is_reversed(self):
        raw, units = draft()
        units.append({"claim_id": 1, "text": "The fruit is curved."})
        raw["claims"].insert(0, {**deepcopy(raw["claims"][0]), "claim_id": 1})
        value = validate_claim_semantics_units(raw, units)
        logs = [json.loads(item.split(": ", 1)[1]) for item in value["issues"]]
        self.assertEqual([item["claim_id"] for item in logs], [0, 1])
        self.assertEqual([c["quote"] for c in value["claims"]], [u["text"] for u in units])


if __name__ == "__main__":
    unittest.main()
