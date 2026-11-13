"""Prevent invented or truncated proposal paths from becoming symbolic proof."""
import unittest
from thinkmorph_grpo.judge_v2 import validate_proposal_units, text_units


def proposal(text, moves):
    units = text_units(text)
    value = {"evidence": "Proposed operation", "operation": text, "operation_kind": "navigate",
             "claims": [{"claim_id": u["claim_id"], "kind": "intention",
                         "support": "not_applicable", "evidence": "Future action"} for u in units],
             "moves": moves, "validity": "valid", "assessable": True, "issues": []}
    return value, units


class ProposalProvenanceTests(unittest.TestCase):
    def test_layout_preamble_cannot_supply_invented_moves(self):
        value, units = proposal("Let me break down the maze layout in detail. Time to plot a route.", ["R", "R"])
        result = validate_proposal_units(value, units, literal_moves=True)
        self.assertEqual(result["moves"], [])
        self.assertIn("Discarded inferred", result["issues"][-1])
        self.assertEqual(result["validity"], "valid")
        self.assertEqual(value["moves"], ["R", "R"])

    def test_prose_plan_remains_semantically_assessable_without_simulated_moves(self):
        value, units = proposal("Move up, then right twice and down.", [])
        self.assertEqual(validate_proposal_units(value, units, literal_moves=True)["validity"], "valid")
        value["moves"] = ["U", "R", "R", "D"]
        result = validate_proposal_units(value, units, literal_moves=True)
        self.assertEqual(result["moves"], [])
        self.assertEqual(result["validity"], "valid")

    def test_invalid_prose_plan_stays_invalid_when_unverified_moves_are_removed(self):
        value, units = proposal("Move right twice through the hole.", ["R", "R"])
        value["validity"] = "invalid"
        result = validate_proposal_units(value, units, literal_moves=True)
        self.assertEqual(result["moves"], [])
        self.assertEqual(result["validity"], "invalid")

    def test_whole_literal_sequence_is_retained(self):
        value, units = proposal("Draw the route U, R, R, D.", ["U", "R", "R", "D"])
        self.assertEqual(validate_proposal_units(value, units, literal_moves=True)["moves"], value["moves"])

    def test_subsequence_is_not_a_complete_proposed_route(self):
        value, units = proposal("Draw the route U,R,R,D.", ["R", "R"])
        with self.assertRaisesRegex(ValueError, "whole literal"):
            validate_proposal_units(value, units, literal_moves=True)

    def test_malformed_chain_cannot_be_repaired_by_omission(self):
        for text in ("Draw Q,U,R.", "Draw U,R,Q."):
            value, units = proposal(text, ["U", "R"])
            with self.subTest(text=text):
                # No valid literal candidate exists: the malformed source cannot
                # establish symbolic proof, and validity is still judged by Qwen.
                result = validate_proposal_units(value, units, literal_moves=True)
                self.assertEqual(result["moves"], [])


if __name__ == "__main__":
    unittest.main()
