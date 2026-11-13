"""Gemini transport correctness and secret-safe concurrent audit tests (offline)."""
import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.error

from thinkmorph_grpo.gemini_judge import GeminiJudge, GeminiTransportError, native_payload


KEY = "unit-test-credential-do-not-log"
MODEL = "gemini-3.8-flash"
MESSAGES = [{"role": "system", "content": "Return the requested JSON."},
            {"role": "user", "content": "Judge this evidence."}]
METADATA = {"name": "models/" + MODEL, "version": "unit-version",
            "supportedGenerationMethods": ["generateContent"]}


class Response(io.BytesIO):
    status = 200

    def __init__(self, value):
        super().__init__(value if isinstance(value, bytes) else json.dumps(value).encode())


def completion(text='{"score": 1}', finish="STOP"):
    return {"modelVersion": MODEL, "candidates": [{"finishReason": finish,
        "content": {"parts": [{"text": "private reasoning", "thought": True}, {"text": text}]}}],
        "usageMetadata": {"promptTokenCount": 110, "candidatesTokenCount": 5,
                          "thoughtsTokenCount": 7, "cachedContentTokenCount": 20,
                          "totalTokenCount": 122}}


def transport_error(status, body, headers=None):
    return urllib.error.HTTPError("https://generativelanguage.googleapis.com/test", status,
                                  "failure", headers or {}, io.BytesIO(body.encode()))


class GeminiJudgeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)

    def judge(self, **kwargs):
        return GeminiJudge(KEY, audit_directory=self.path / "audit", **kwargs)

    def audits(self):
        return [json.loads(path.read_text()) for path in sorted((self.path / "audit").glob("*.json"))]

    def test_native_conversion_preserves_images_labels_roles_and_prompt_strings(self):
        first, second = base64.b64encode(b"first-image-bytes").decode(), base64.b64encode(b"second-image-bytes").decode()
        messages = [{"role": "system", "content": "Never reveal an expected answer to the process judge."},
                    {"role": "user", "content": [
                        {"type": "text", "text": "ORIGINAL"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + first}},
                        {"type": "text", "text": "AFTER"},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + second}},
                        {"type": "text", "text": "Exact observation text."}]},
                    {"role": "assistant", "content": "Previous assessment"},
                    {"role": "user", "content": "Correct the assessment"}]
        result = native_payload(messages, thinking="low", max_output_tokens=8192)
        self.assertEqual(result["systemInstruction"]["parts"], [{"text": messages[0]["content"]}])
        parts = result["contents"][0]["parts"]
        self.assertEqual([p.get("text") for p in parts], ["ORIGINAL", None, "AFTER", None, "Exact observation text."])
        self.assertEqual(parts[1]["inlineData"], {"mimeType": "image/png", "data": first})
        self.assertEqual(parts[3]["inlineData"], {"mimeType": "image/jpeg", "data": second})
        self.assertEqual([c["role"] for c in result["contents"]], ["user", "model", "user"])
        self.assertEqual(result["generationConfig"]["responseMimeType"], "application/json")
        self.assertEqual(result["generationConfig"]["temperature"], 0)
        self.assertEqual(result["generationConfig"]["maxOutputTokens"], 8192)

    def test_remote_image_and_unsupported_role_are_rejected_before_transport(self):
        for messages in ([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.org/image.png"}}]}],
                         [{"role": "tool", "content": "tool"}]):
            with self.subTest(messages=messages), self.assertRaises((ValueError, KeyError)):
                native_payload(messages, thinking="low", max_output_tokens=8192)

    def test_native_completion_ignores_thoughts_and_records_usage_and_model_identity(self):
        judge = self.judge()
        calls = []
        def respond(request, timeout):
            calls.append(request)
            self.assertEqual(timeout, 180)
            self.assertEqual(request.get_header("X-goog-api-key"), KEY)
            self.assertNotIn(KEY, request.full_url)
            return Response(METADATA if request.data is None else completion())
        with patch("urllib.request.urlopen", side_effect=respond):
            self.assertEqual(judge.complete(MESSAGES, max_tokens=2600), {"score": 1})
        self.assertEqual(len(calls), 2)
        sent = json.loads(calls[1].data)
        self.assertEqual(sent["generationConfig"]["maxOutputTokens"], 8192)
        self.assertEqual(sent["systemInstruction"]["parts"][0]["text"], MESSAGES[0]["content"])
        identity = judge.identity()
        self.assertEqual(identity["verified_model_version"], "unit-version")
        self.assertEqual(identity["judge_version"], "grounded_v10")
        self.assertEqual(identity["provider"], "google_gemini")
        self.assertNotIn(KEY, json.dumps(identity))
        stats = judge.stats()
        self.assertEqual(stats["generation_requests"], 1)
        self.assertEqual((stats["prompt_tokens"], stats["candidate_tokens"], stats["thought_tokens"], stats["cached_tokens"]), (110, 5, 7, 20))
        generation = [r for r in self.audits() if r["method"] == "POST"][0]
        self.assertEqual(generation["usage"]["thoughtsTokenCount"], 7)
        self.assertEqual(generation["response"]["modelVersion"], MODEL)

    def test_audit_redacts_credential_and_both_original_and_native_image_payloads(self):
        judge = self.judge()
        judge._verified = True
        encoded = base64.b64encode(b"a-unique-binary-image-payload").decode()
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "Allowed evidence"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}}]}]
        response = completion(json.dumps({"evidence": KEY}))
        response["echo"] = {"inlineData": {"mimeType": "image/png", "data": encoded}}
        with patch("urllib.request.urlopen", return_value=Response(response)):
            judge.complete(messages)
        audit_text = "".join(p.read_text() for p in (self.path/"audit").glob("*.json"))
        self.assertNotIn(KEY, audit_text)
        self.assertNotIn(encoded, audit_text)
        record = self.audits()[0]
        self.assertEqual(record["native_request"]["contents"][0]["parts"][1]["inlineData"]["bytes"], 29)
        self.assertIn("[REDACTED]", audit_text)

    def test_transient_retry_is_bounded_and_honors_capped_retry_after(self):
        judge = self.judge()
        judge._verified = True
        error = transport_error(429, "rate limited " + KEY, {"Retry-After": "120"})
        with patch("urllib.request.urlopen", side_effect=[error, Response(completion())]) as opener, \
             patch("time.sleep") as sleep:
            self.assertEqual(judge.complete(MESSAGES), {"score": 1})
        self.assertEqual(opener.call_count, 2)
        sleep.assert_called_once_with(30.)
        self.assertEqual(judge.stats()["http_429"], 1)
        self.assertEqual(judge.stats()["retry_requests"], 1)
        self.assertEqual(len(self.audits()), 2)
        self.assertNotIn(KEY, json.dumps(self.audits()))

    def test_terminal_http_failure_never_enters_semantic_retry_or_reward_fallback(self):
        judge = self.judge()
        judge._verified = True
        error = transport_error(403, "bad credential " + KEY)
        with patch("urllib.request.urlopen", side_effect=error) as opener:
            with self.assertRaises(GeminiTransportError) as raised:
                judge._assess(MESSAGES, lambda value: value)
        self.assertEqual(opener.call_count, 1)
        self.assertNotIn(KEY, str(raised.exception))
        self.assertNotIn(KEY, json.dumps(self.audits()))

    def test_connection_failure_exhausts_only_transport_retry_budget(self):
        judge = self.judge(max_retries=2)
        judge._verified = True
        with patch("urllib.request.urlopen", side_effect=TimeoutError("timeout " + KEY)) as opener, \
             patch("time.sleep") as sleep:
            with self.assertRaises(GeminiTransportError) as raised:
                judge._assess(MESSAGES, lambda value: value)
        self.assertEqual(opener.call_count, 3)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [2., 4.])
        self.assertEqual(judge.stats()["transport_errors"], 3)
        self.assertNotIn(KEY, str(raised.exception))

    def test_invalid_http_json_is_transport_failure_not_semantic_zero(self):
        judge = self.judge()
        judge._verified = True
        with patch("urllib.request.urlopen", return_value=Response(b"upstream unavailable")) as opener:
            with self.assertRaises(GeminiTransportError):
                judge._assess(MESSAGES, lambda value: value)
        self.assertEqual(opener.call_count, 1)

    def test_malformed_usage_metadata_is_a_transport_failure_not_typeerror_fallback(self):
        for usage in ("invalid", {"promptTokenCount": "110"}):
            judge = self.judge()
            judge._verified = True
            response = completion()
            response["usageMetadata"] = usage
            with self.subTest(usage=usage), patch("urllib.request.urlopen", return_value=Response(response)) as opener:
                with self.assertRaises(GeminiTransportError):
                    judge._assess(MESSAGES, lambda value: value)
            self.assertEqual(opener.call_count, 1)

    def test_error_body_inline_images_and_key_are_redacted(self):
        judge = self.judge()
        judge._verified = True
        encoded = base64.b64encode(b"unique-error-image-content").decode()
        body = json.dumps({"message": KEY, "inlineData": {"mimeType": "image/png", "data": encoded}})
        with patch("urllib.request.urlopen", side_effect=transport_error(403, body)):
            with self.assertRaises(GeminiTransportError) as raised:
                judge.complete(MESSAGES)
        self.assertNotIn(KEY, str(raised.exception))
        self.assertNotIn(encoded, str(raised.exception))
        self.assertNotIn(encoded, json.dumps(self.audits()))

    def test_truncated_candidate_is_rejected_even_if_content_looks_valid(self):
        judge = self.judge()
        judge._verified = True
        with patch("urllib.request.urlopen", return_value=Response(completion(finish="MAX_TOKENS"))):
            with self.assertRaisesRegex(ValueError, "truncated"):
                judge.complete(MESSAGES)
        self.assertEqual(self.audits()[0]["finish_reasons"], ["MAX_TOKENS"])

    def test_wrong_model_metadata_is_a_setup_failure(self):
        judge = self.judge()
        with patch("urllib.request.urlopen", return_value=Response(dict(METADATA, name="models/wrong"))):
            with self.assertRaises(GeminiTransportError):
                judge.verify_model()
        self.assertFalse(judge._verified)

    def test_concurrent_requests_verify_once_and_write_distinct_audits(self):
        judge = self.judge()
        barrier = threading.Barrier(4)
        count_lock = threading.Lock()
        count = {"metadata": 0}
        def respond(request, timeout):
            if request.data is None:
                with count_lock:
                    count["metadata"] += 1
                return Response(METADATA)
            barrier.wait(timeout=10)
            return Response(completion())
        with patch("urllib.request.urlopen", side_effect=respond):
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(lambda _: judge.complete(MESSAGES), range(4)))
        self.assertEqual(results, [{"score": 1}]*4)
        self.assertEqual(count["metadata"], 1)
        records = self.audits()
        self.assertEqual(len(records), 5)
        self.assertEqual(len({r["request_id"] for r in records}), 5)
        self.assertEqual(judge.stats()["max_concurrent_requests"], 4)
        self.assertEqual(judge.stats()["prompt_tokens"], 440)

    def test_key_file_and_environment_are_supported_without_identity_leaks(self):
        path = self.path/"fake_key"
        path.write_text(KEY+"\n")
        judge = GeminiJudge(api_key_file=path)
        self.assertEqual(judge.api_key, KEY)
        with patch.dict("os.environ", {"GEMINI_API_KEY": KEY}, clear=True):
            self.assertEqual(GeminiJudge().api_key, KEY)
        with self.assertRaisesRegex(ValueError, "either"):
            GeminiJudge(KEY, api_key_file=path)
        self.assertNotIn(KEY, json.dumps(judge.identity()))


if __name__ == "__main__":
    unittest.main()
