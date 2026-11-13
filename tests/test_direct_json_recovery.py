"""Lossless missing-array-close recovery, bounded to direct model replies."""
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from thinkmorph_grpo import practical_judge
from thinkmorph_grpo.judge import MODEL
from thinkmorph_grpo.judge_v2 import (
    DirectObservationQwenJudge, GroundedQwenJudge, JudgeAssessmentError,
    PROPOSAL_PROMPT, VISUAL_PROMPT,
)


FIXTURE = Path(__file__).parent / "fixtures/direct_observation_missing_array_close.json"


def draft(support="supported"):
    return {"claims": [{"claim_id": 0, "scope": "current_state", "evidence": "Visible target",
                        "support": support}],
            "evidence": "Image inspected", "assessable": True, "issues": []}


def missing_close(value=None):
    raw = json.dumps(draft() if value is None else value, separators=(",", ":"))
    return raw.replace('}],"evidence":', '},"evidence":', 1)


def reply(raw, finish="stop"):
    return {"choices": [{"finish_reason": finish, "message": {"content": raw}}]}


def messages(prompt=None):
    return [{"role": "system", "content": practical_judge.DIRECT_OBSERVATION_PROMPT if prompt is None else prompt},
            {"role": "user", "content": "Untrusted observation data"}]


class DirectJSONRecoveryTests(unittest.TestCase):
    def test_real_complete_reply_recovers_twelve_claims_without_changing_content(self):
        record = json.loads(FIXTURE.read_text())
        raw = record["raw_content"]
        before = raw[:]
        with self.assertRaises(json.JSONDecodeError):
            json.loads(raw)
        expected = json.loads(raw.replace('},"evidence":', '}],"evidence":', 1))
        repaired = practical_judge.recover_claims_array_close(raw)
        self.assertEqual(raw, before)
        self.assertEqual(len(repaired["claims"]), 12)
        self.assertEqual([c["claim_id"] for c in repaired["claims"]], list(range(12)))
        self.assertEqual([c["support"] for c in repaired["claims"]], ["supported"] * 10 + ["unsupported"] * 2)
        self.assertEqual(repaired["claims"], expected["claims"])
        self.assertEqual({k: v for k, v in repaired.items() if k != "issues"},
                         {k: v for k, v in expected.items() if k != "issues"})
        self.assertEqual(len(repaired["issues"]), len(expected["issues"]) + 1)
        self.assertIn("inserted one missing claims-array", repaired["issues"][-1])
        validated = practical_judge.validate_direct_observations(repaired, record["text_after_units"])
        self.assertEqual([c["quote"] for c in validated["claims"]],
                         [unit["text"] for unit in record["text_after_units"]])
        self.assertEqual([c["support"] for c in validated["claims"]],
                         [c["support"] for c in expected["claims"]])

    def test_only_missing_array_close_is_recovered(self):
        good = json.dumps(draft(), separators=(",", ":"))
        bad = missing_close()
        missing_field = draft(); missing_field["claims"][0].pop("support")
        negatives = [None, good, bad[:-1], bad + " trailing text",
                     bad.replace('"scope":', '"scope" '),
                     bad.replace('"Visible target"', '"Visible "target""'),
                     bad.replace('},"evidence":', '},,"evidence":'),
                     bad.replace('"evidence":"Image inspected",', ''),
                     bad.replace('"claims":[{', '"claims":[3,{'),
                     bad.replace('"claim_id":0,', '"claim_id":0,"claim_id":1,'),
                     bad.replace('"issues":[]', '"issues":[],"issues":[]'),
                     missing_close(missing_field),
                     '{"evidence":"first","claims":[{"claim_id":0},"assessable":true,"issues":[]}',
                     '{"claims":[,"evidence":"empty","assessable":true,"issues":[]}']
        for raw in negatives:
            with self.subTest(raw=raw):
                self.assertIsNone(practical_judge.recover_claims_array_close(raw))

    def test_decoder_handles_whitespace_and_delimiter_text_inside_evidence(self):
        value = draft()
        value["claims"][0]["evidence"] = 'Literal text },"evidence": is not a JSON boundary.'
        # The only edit to the valid serialization removes its array close.
        raw = json.dumps(value, indent=2).replace('  ],\n  "evidence":', '  ,\n  "evidence":', 1)
        result = practical_judge.recover_claims_array_close(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result["claims"], value["claims"])

    def test_complete_uses_one_unchanged_request_and_preserves_raw_rpc_audit(self):
        raw = missing_close()
        body = json.dumps(reply(raw)).encode()
        response = BytesIO(body); response.status = 200
        with tempfile.TemporaryDirectory() as directory:
            judge = DirectObservationQwenJudge("http://unused/v1", audit_directory=directory)
            judge._verified = True
            sent = messages()
            with patch("thinkmorph_grpo.judge.urllib.request.urlopen", return_value=response) as request:
                result = judge.complete(sent, max_tokens=333)
            self.assertEqual(request.call_count, 1)
            payload = json.loads(request.call_args.args[0].data)
            self.assertEqual(payload, {"model": MODEL, "messages": sent, "temperature": 0,
                                      "max_tokens": 333, "response_format": {"type": "json_object"}})
            paths = list(Path(directory).glob("*.json")); self.assertEqual(len(paths), 1)
            audit = json.loads(paths[0].read_text())
            self.assertEqual(audit["response"]["choices"][0]["message"]["content"], raw)
            self.assertEqual(audit["raw_response_text"], body.decode())
            self.assertEqual(result["claims"], draft()["claims"])
            self.assertTrue(result["issues"])

    def test_baseline_and_nonexact_observation_prompts_never_recover(self):
        prompts = [PROPOSAL_PROMPT, VISUAL_PROMPT, "Evaluate FINAL answer correctness",
                   practical_judge.DIRECT_OBSERVATION_PROMPT + "\nPrevious assessment failed validation"]
        cases = [(GroundedQwenJudge, practical_judge.DIRECT_OBSERVATION_PROMPT)]
        cases.extend((DirectObservationQwenJudge, prompt) for prompt in prompts)
        for kind, prompt in cases:
            judge = kind("unused"); judge._verified = True
            with self.subTest(kind=kind, prompt=prompt), patch.object(judge, "_request", return_value=reply(missing_close())) as request:
                with self.assertRaises(json.JSONDecodeError):
                    judge.complete(messages(prompt))
                self.assertEqual(request.call_count, 1)

    def test_truncated_and_transport_json_errors_are_not_repaired(self):
        judge = DirectObservationQwenJudge("unused"); judge._verified = True
        with patch.object(judge, "_request", return_value=reply(missing_close(), finish="length")):
            with self.assertRaisesRegex(ValueError, "truncated"):
                judge.complete(messages())
        for verified in (False, True):
            judge._verified = verified
            error = json.JSONDecodeError("bad wire JSON", missing_close(), 0)
            with self.subTest(verified=verified), patch.object(judge, "_request", side_effect=error) as request:
                with self.assertRaises(json.JSONDecodeError) as caught:
                    judge.complete(messages())
                self.assertIs(caught.exception, error)
                self.assertEqual(request.call_count, 1)

    def test_invalid_label_remains_a_typed_assessment_failure_after_syntax_recovery(self):
        raw = missing_close(draft("invented_support"))
        result = practical_judge.recover_claims_array_close(raw)
        self.assertEqual(result["claims"][0]["support"], "invented_support")
        judge = DirectObservationQwenJudge("unused"); judge._verified = True
        units = [{"claim_id": 0, "text": "The target is visible."}]
        with patch.object(judge, "_request", return_value=reply(raw)) as request:
            with self.assertRaises(JudgeAssessmentError):
                judge._assess(messages(), lambda value: practical_judge.validate_direct_observations(value, units))
            self.assertEqual(request.call_count, 2)  # Existing bounded semantic retry, no repair RPC.

    def test_syntax_recovery_does_not_repair_source_ids_or_missing_units(self):
        for ids in ([1], [0, 0]):
            value = draft()
            value["claims"] = [dict(value["claims"][0], claim_id=index) for index in ids]
            raw = missing_close(value)
            repaired = practical_judge.recover_claims_array_close(raw)
            self.assertEqual(repaired["claims"], value["claims"])
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                practical_judge.validate_direct_observations(repaired, [{"claim_id": 0, "text": "Original text."}])


if __name__ == "__main__":
    unittest.main()
