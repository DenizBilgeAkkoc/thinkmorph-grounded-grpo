"""Native Gemini transport for the unchanged grounded V10 judge rubric.

The model identity is separate from ``version``: the latter selects the rubric
inside judge_v2.py and must remain grounded_v10. No API credential is serialized.
"""
import base64
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import time
import urllib.error
import urllib.request
import uuid

from .judge import _audit_value, _write_rpc_audit
from .judge_v2 import DirectObservationQwenJudge


class GeminiTransportError(RuntimeError):
    """A provider/connection failure, never a missing visual-reward label."""


def native_payload(messages, *, thinking, max_output_tokens):
    """Preserve prompt strings, image bytes, labels and content ordering."""
    system, contents = [], []
    for message in messages:
        role, content = message["role"], message["content"]
        if role == "system":
            if contents or not isinstance(content, str):
                raise ValueError("Gemini judge requires leading text system messages")
            system.append({"text": content})
            continue
        if role not in ("user", "assistant"):
            raise ValueError("Unsupported Gemini judge message role")
        parts = []
        items = [{"type": "text", "text": content}] if isinstance(content, str) else content
        if not isinstance(items, list):
            raise ValueError("Unsupported Gemini judge message content")
        for item in items:
            if item["type"] == "text" and isinstance(item["text"], str):
                parts.append({"text": item["text"]})
            elif item["type"] == "image_url":
                url = item["image_url"]["url"]
                header, encoded = url.split(",", 1)
                if not header.startswith("data:image/") or not header.endswith(";base64"):
                    raise ValueError("Gemini judge accepts original inline images only")
                base64.b64decode(encoded, validate=True)
                parts.append({"inlineData": {"mimeType": header[5:-7], "data": encoded}})
            else:
                raise ValueError("Unsupported Gemini judge content part")
        contents.append({"role": "model" if role == "assistant" else "user", "parts": parts})
    if not contents:
        raise ValueError("Gemini judge requires user content")
    return {"systemInstruction": {"parts": system}, "contents": contents,
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json",
                "maxOutputTokens": max_output_tokens, "thinkingConfig": {"thinkingLevel": thinking}}}


class GeminiJudge(DirectObservationQwenJudge):
    provider = "google_gemini"
    transport_revision = "gemini_native_grounded_v10_20261111"
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, api_key="", *, api_key_file=None, model="gemini-3.8-flash",
                 thinking="low", timeout=180, audit_directory=None,
                 max_output_tokens=8192, max_retries=2):
        if api_key and api_key_file:
            raise ValueError("Choose either a Gemini API key or a key file")
        key = (Path(api_key_file).read_text() if api_key_file else
               api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")).strip()
        if not key or any(character.isspace() for character in key):
            raise ValueError("Gemini API key must be a single nonempty token")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", model):
            raise ValueError("Invalid Gemini model ID")
        if thinking not in ("minimal", "low", "medium", "high"):
            raise ValueError("Unsupported Gemini thinking level")
        if model == "gemini-3.8-flash" and thinking == "minimal":
            raise ValueError("Gemini 3.8 Flash requires low, medium or high thinking")
        if not math.isfinite(float(timeout)) or float(timeout) <= 0:
            raise ValueError("Gemini timeout must be finite and positive")
        if type(max_output_tokens) is not int or max_output_tokens < 1:
            raise ValueError("Gemini output token budget must be a positive integer")
        if type(max_retries) is not int or not 0 <= max_retries <= 5:
            raise ValueError("Gemini retries must be an integer from zero through five")
        super().__init__(self.BASE_URL, key, timeout=float(timeout), audit_directory=audit_directory)
        self.model = self.model_id = model
        self.thinking = thinking
        self.max_output_tokens, self.max_retries = max_output_tokens, max_retries
        self.model_version = None
        self._verify_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._active_requests = 0
        self._stats = {"requests": 0, "generation_requests": 0, "responses": 0,
                       "transport_errors": 0, "retry_requests": 0, "http_429": 0,
                       "validation_retry_requests": 0,
                       "prompt_tokens": 0, "candidate_tokens": 0, "thought_tokens": 0,
                       "cached_tokens": 0, "request_seconds": 0., "max_concurrent_requests": 0}

    def identity(self):
        """Configuration identity suitable for checkpoints; contains no secrets."""
        return {"provider": self.provider, "model": self.model, "thinking": self.thinking,
                "temperature": 0, "max_output_tokens": self.max_output_tokens,
                "transport_revision": self.transport_revision,
                "judge_version": self.version, "judge_protocol_revision": self.protocol_revision,
                "terminal_policy_version": self.terminal_version,
                "verified_model_version": self.model_version}

    def stats(self):
        with self._stats_lock:
            return dict(self._stats)

    def _safe(self, value):
        if isinstance(value, dict):
            output = {}
            for key, item in value.items():
                if key in ("inlineData", "inline_data") and isinstance(item, dict) and "data" in item:
                    encoded = item["data"]
                    try:
                        raw = base64.b64decode(encoded, validate=True)
                        output[key] = {"mimeType": item.get("mimeType", item.get("mime_type")),
                                       "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                    except (ValueError, TypeError):
                        output[key] = {"encoded_sha256": hashlib.sha256(str(encoded).encode()).hexdigest()}
                else:
                    output[self._safe(key)] = self._safe(item)
            return output
        if isinstance(value, (list, tuple)):
            return [self._safe(item) for item in value]
        if isinstance(value, str):
            return _audit_value(value.replace(self.api_key, "[REDACTED]"))
        return value

    @staticmethod
    def _retry_delay(error, attempt):
        try:
            requested = float(error.headers.get("Retry-After", ""))
            if math.isfinite(requested):
                return min(30., max(0., requested))
        except (AttributeError, ValueError, TypeError):
            pass
        return min(30., float(2 ** (attempt + 1)))

    def _safe_error_message(self, text):
        try:
            value = json.loads(text)
        except (ValueError, TypeError):
            value = str(text)
        value = self._safe(value)
        return value if isinstance(value, str) else json.dumps(value)

    def _native_request(self, path, payload=None, *, request_audit=None):
        body = json.dumps(payload).encode() if payload is not None else None
        for attempt in range(self.max_retries + 1):
            started = time.monotonic()
            record = {"schema_version": "gemini_judge_rpc_v1",
                "request_id": f"rpc_{time.time_ns()}_{uuid.uuid4().hex}",
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "thread_id": threading.get_ident(), "provider": self.provider,
                "model": self.model, "thinking": self.thinking,
                "base_url": self.BASE_URL, "path": path,
                "method": "POST" if body is not None else "GET",
                "request": self._safe(request_audit if request_audit is not None else payload),
                "native_request": self._safe(payload), "request_bytes": len(body or b""),
                "request_body_sha256": hashlib.sha256(body or b"").hexdigest(),
                "transport_attempt": attempt, "timeout_seconds": self.timeout,
                "judge_version": self.version, "judge_protocol_revision": self.protocol_revision,
                "transport_revision": self.transport_revision}
            request = urllib.request.Request(self.BASE_URL + path, data=body,
                headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key})
            retry_delay, raw, value = None, None, None
            with self._stats_lock:
                self._active_requests += 1
                self._stats["requests"] += 1
                self._stats["generation_requests"] += int(path.endswith(":generateContent"))
                self._stats["retry_requests"] += int(attempt > 0)
                self._stats["validation_retry_requests"] += int(
                    "Previous assessment failed validation:" in str(request_audit or {}))
                self._stats["max_concurrent_requests"] = max(self._stats["max_concurrent_requests"], self._active_requests)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    record["http_status"] = response.status
                    raw = response.read()
                try:
                    value = json.loads(raw)
                except (ValueError, TypeError, UnicodeError):
                    raise GeminiTransportError("Gemini returned a non-JSON HTTP response") from None
                if not isinstance(value, dict):
                    raise GeminiTransportError("Gemini returned an invalid response envelope")
                usage = value.get("usageMetadata", {})
                if not isinstance(usage, dict) or any(
                        type(usage.get(field, 0)) is not int or usage.get(field, 0) < 0
                        for field in ("promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount", "cachedContentTokenCount")):
                    raise GeminiTransportError("Gemini returned invalid token usage metadata")
                record["response"] = self._safe(value)
                record["outcome"] = "response_received"
                record["usage"] = value.get("usageMetadata")
                record["finish_reasons"] = [c.get("finishReason") for c in value.get("candidates", [])]
                return value
            except urllib.error.HTTPError as error:
                record["http_status"] = error.code
                try:
                    raw = error.read()
                except Exception:
                    raw = b""
                message = self._safe_error_message(raw.decode(errors="replace"))
                record["error"] = {"type": "HTTPError", "message": message[:6000]}
                record["outcome"] = "transport_error"
                if error.code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    retry_delay = self._retry_delay(error, attempt)
                else:
                    raise GeminiTransportError(f"Gemini HTTP {error.code}: {message[:1200]}") from None
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                message = self._safe_error_message(str(error))
                record["error"] = {"type": type(error).__name__, "message": message}
                record["outcome"] = "transport_error"
                if attempt < self.max_retries:
                    retry_delay = self._retry_delay(error, attempt)
                else:
                    raise GeminiTransportError("Gemini connection failed: " + message) from None
            except Exception as error:
                record["error"] = {"type": type(error).__name__, "message": self._safe(str(error))}
                record["outcome"] = "transport_error"
                if isinstance(error, GeminiTransportError):
                    raise
                raise GeminiTransportError("Gemini response failed: " + self._safe_error_message(str(error))) from None
            finally:
                elapsed = time.monotonic() - started
                record["elapsed_seconds"] = elapsed
                if raw is not None:
                    record["response_bytes"] = len(raw)
                    record["response_body_sha256"] = hashlib.sha256(raw).hexdigest()
                with self._stats_lock:
                    self._active_requests -= 1
                    self._stats["request_seconds"] += elapsed
                    self._stats["responses"] += int(value is not None)
                    self._stats["transport_errors"] += int(record.get("outcome") == "transport_error")
                    self._stats["http_429"] += int(record.get("http_status") == 429)
                    usage = value.get("usageMetadata", {}) if isinstance(value, dict) else {}
                    usage = usage if isinstance(usage, dict) else {}
                    for target, source in (("prompt_tokens", "promptTokenCount"), ("candidate_tokens", "candidatesTokenCount"),
                                           ("thought_tokens", "thoughtsTokenCount"), ("cached_tokens", "cachedContentTokenCount")):
                        count = usage.get(source, 0)
                        self._stats[target] += count if type(count) is int and count >= 0 else 0
                if self._rpc_audit_directory is not None:
                    _write_rpc_audit(self._rpc_audit_directory, self._safe(record))
            if retry_delay is not None:
                time.sleep(retry_delay)
        raise GeminiTransportError("Gemini transport retries exhausted")

    def verify_model(self):
        with self._verify_lock:
            if self._verified:
                return
            value = self._native_request("/models/" + self.model)
            if value.get("name") != "models/" + self.model:
                raise GeminiTransportError("Gemini metadata returned a different model")
            if "generateContent" not in value.get("supportedGenerationMethods", []):
                raise GeminiTransportError("Gemini model does not support generateContent")
            self.model_version = value.get("version")
            self._verified = True

    def complete(self, messages, *, max_tokens=1200):
        if not self._verified:
            self.verify_model()
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("Judge max_tokens must be a positive integer")
        payload = native_payload(messages, thinking=self.thinking,
                                 max_output_tokens=max(self.max_output_tokens, max_tokens))
        response = self._native_request("/models/" + self.model + ":generateContent", payload,
            request_audit={"model": self.model, "messages": messages, "max_tokens": max_tokens})
        candidates = response.get("candidates", [])
        if not candidates:
            raise ValueError("Gemini produced no complete candidate: " + str(self._safe(response.get("promptFeedback", {}))))
        choice = candidates[0]
        if choice.get("finishReason") != "STOP":
            raise ValueError("Gemini response was truncated or otherwise incomplete: " + str(choice.get("finishReason")))
        parts = choice.get("content", {}).get("parts", [])
        content = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        return self.parse_completion_content(content, messages=messages)
