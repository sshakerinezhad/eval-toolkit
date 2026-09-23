import asyncio
from types import SimpleNamespace as NS

import httpx
import openai
import pytest

import llm


# ---------- helpers ----------

def _err(cls, status, message="boom", headers=None, code=None):
    req = httpx.Request("POST", "http://x")
    resp = httpx.Response(status, request=req, headers=headers or {})
    body = {"error": {"code": code}} if code else None
    return cls(message, response=resp, body=body)


class FakeCompletions:
    """Scripted chat.completions.create: each item is an exception to raise or a response to return."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class FakeClient:
    def __init__(self, script):
        self.chat = NS(completions=FakeCompletions(script))


def _resp(content: str | None = "hi", tokens_in=10, tokens_out=5, finish="stop"):
    choice = NS(message=NS(content=content), finish_reason=finish)
    usage = NS(prompt_tokens=tokens_in, completion_tokens=tokens_out)
    return NS(choices=[choice], usage=usage)


def _call(client, **kw):
    args = dict(provider="openai", model="gpt-4.1-mini", system_prompt="sys", prompt="p",
                temperature=0.5, max_tokens=100, max_retries=4)
    args.update(kw)
    return asyncio.run(llm.call(client, **args))


@pytest.fixture
def no_sleep(monkeypatch):
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr(llm.asyncio, "sleep", fake_sleep)
    return sleeps


# ---------- parse_model ----------

def test_parse_model_splits_provider_and_model():
    assert llm.parse_model("anthropic:claude-haiku-4-5") == ("anthropic", "claude-haiku-4-5")


def test_parse_model_keeps_slashes_and_colons_in_model_name():
    assert llm.parse_model("openrouter:openai/gpt-5-mini:free") == ("openrouter", "openai/gpt-5-mini:free")


def test_parse_model_rejects_unknown_provider():
    with pytest.raises(ValueError, match="unknown provider"):
        llm.parse_model("gemini:x")


def test_parse_model_rejects_missing_provider():
    with pytest.raises(ValueError):
        llm.parse_model("gpt-5-mini")


# ---------- build_params ----------

def test_build_params_normal_model_sends_temperature_and_max_tokens():
    assert llm.build_params("openai", "gpt-4.1-mini", 0.7, 100) == {"temperature": 0.7, "max_tokens": 100}


def test_build_params_null_temperature_is_omitted():
    assert llm.build_params("anthropic", "claude-haiku-4-5", None, 100) == {"max_tokens": 100}


def test_build_params_openai_reasoning_model_drops_temp_uses_max_completion_tokens():
    assert llm.build_params("openai", "gpt-5-mini", 0.7, 100) == {"max_completion_tokens": 100}


@pytest.mark.parametrize("model", ["gpt-5.6-sol", "gpt-6-luna", "o4-mini"])
def test_build_params_reasoning_prefixes_cover_gpt5x_gpt6_and_o_series(model):
    assert llm.build_params("openai", model, 0.7, 100) == {"max_completion_tokens": 100}


def test_build_params_reasoning_prefix_only_applies_to_openai_provider():
    assert llm.build_params("openrouter", "openai/gpt-5-mini", 0.7, 100) == {"temperature": 0.7, "max_tokens": 100}


# ---------- cache ----------

def test_cache_key_is_deterministic_sha256_hex():
    a = llm.cache_key("openai", "m", "s", "p", 0, 0.5, 10)
    b = llm.cache_key("openai", "m", "s", "p", 0, 0.5, 10)
    assert a == b and len(a) == 64 and int(a, 16)


def test_cache_key_changes_with_each_field():
    base = ("openai", "m", "s", "p", 0, 0.5, 10)
    variants = [
        ("anthropic", "m", "s", "p", 0, 0.5, 10),
        ("openai", "m2", "s", "p", 0, 0.5, 10),
        ("openai", "m", "s2", "p", 0, 0.5, 10),
        ("openai", "m", "s", "p2", 0, 0.5, 10),
        ("openai", "m", "s", "p", 1, 0.5, 10),
        ("openai", "m", "s", "p", 0, None, 10),
        ("openai", "m", "s", "p", 0, 0.5, 11),
    ]
    keys = {llm.cache_key(*base)} | {llm.cache_key(*v) for v in variants}
    assert len(keys) == 8


def test_cache_miss_returns_none():
    assert llm.cache_get("0" * 64) is None


def test_cache_roundtrip_and_no_tmp_left_behind():
    rec = {"response": "x", "tokens_in": 1, "tokens_out": 2, "finish_reason": "stop"}
    llm.cache_put("abc", rec)
    assert llm.cache_get("abc") == rec
    assert not list(llm.CACHE_DIR.glob("*.tmp"))


# ---------- call: success ----------

def test_call_success_returns_response_tokens_finish_and_no_errors():
    r = _call(FakeClient([_resp("hello", 12, 3, "stop")]))
    assert r.response == "hello"
    assert (r.tokens_in, r.tokens_out, r.finish_reason) == (12, 3, "stop")
    assert r.errors == [] and r.fatal is False
    assert r.params_sent == {"temperature": 0.5, "max_tokens": 100}


def test_call_sends_model_params_system_and_user_messages():
    c = FakeClient([_resp()])
    _call(c)
    sent = c.chat.completions.calls[0]
    assert sent["model"] == "gpt-4.1-mini"
    assert sent["temperature"] == 0.5 and sent["max_tokens"] == 100
    assert sent["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "p"}]


def test_call_omits_system_message_when_system_prompt_empty():
    c = FakeClient([_resp()])
    _call(c, system_prompt="")
    assert c.chat.completions.calls[0]["messages"] == [{"role": "user", "content": "p"}]


# ---------- call: retries ----------

def test_call_retries_on_rate_limit_then_succeeds(no_sleep):
    c = FakeClient([_err(openai.RateLimitError, 429, code="rate_limit_exceeded"), _resp("ok")])
    r = _call(c)
    assert r.response == "ok"
    assert len(r.errors) == 1
    e = r.errors[0]
    assert (e.attempt, e.status, e.type, e.code, e.retryable) == (1, 429, "RateLimitError", "rate_limit_exceeded", True)
    assert len(no_sleep) == 1 and 2 <= no_sleep[0] <= 3


def test_call_backoff_doubles_and_caps_at_64(no_sleep, monkeypatch):
    monkeypatch.setattr(llm.random, "random", lambda: 0.0)
    script = [_err(openai.InternalServerError, 500)] * 6 + [_resp("ok")]
    r = _call(FakeClient(script), max_retries=6)
    assert r.response == "ok"
    assert no_sleep == [2, 4, 8, 16, 32, 64]


def test_call_honours_retry_after_header(no_sleep):
    c = FakeClient([_err(openai.RateLimitError, 429, headers={"retry-after": "7"}), _resp()])
    _call(c)
    assert no_sleep == [7.0]


def test_call_gives_up_after_max_retries(no_sleep):
    c = FakeClient([_err(openai.RateLimitError, 429)] * 5)
    r = _call(c, max_retries=4)
    assert r.response is None and r.tokens_in is None
    assert len(r.errors) == 5 and len(c.chat.completions.calls) == 5
    assert r.fatal is False


def test_call_timeout_and_connection_errors_are_retryable(no_sleep):
    req = httpx.Request("POST", "http://x")
    c = FakeClient([openai.APITimeoutError(req), openai.APIConnectionError(request=req), _resp("ok")])
    r = _call(c)
    assert r.response == "ok"
    assert [e.type for e in r.errors] == ["APITimeoutError", "APIConnectionError"]
    assert all(e.retryable and e.status is None for e in r.errors)


def test_call_bad_request_is_not_retried():
    c = FakeClient([_err(openai.BadRequestError, 400, "Unsupported parameter: temperature")])
    r = _call(c)
    assert r.response is None
    assert len(c.chat.completions.calls) == 1
    e = r.errors[0]
    assert (e.status, e.retryable, r.fatal) == (400, False, False)
    assert "temperature" in e.message


def test_call_auth_error_is_fatal_and_not_retried():
    c = FakeClient([_err(openai.AuthenticationError, 401)])
    r = _call(c)
    assert r.fatal is True and len(c.chat.completions.calls) == 1
    assert r.errors[0].status == 401 and r.errors[0].retryable is False


def test_call_permission_error_is_fatal():
    r = _call(FakeClient([_err(openai.PermissionDeniedError, 403)]))
    assert r.fatal is True


def test_call_empty_content_is_nonretryable_error():
    c = FakeClient([_resp(content=None)])
    r = _call(c)
    assert r.response is None
    assert r.errors[0].type == "EmptyResponse" and r.errors[0].retryable is False
    assert len(c.chat.completions.calls) == 1


# ---------- smoke ----------

def test_smoke_reports_ok_latency_and_content(monkeypatch):
    monkeypatch.setattr(llm, "make_client", lambda provider, timeout: FakeClient([_resp("OK", 5, 1)]))
    [r] = asyncio.run(llm.smoke(["openai:gpt-4.1-mini"], timeout=10))
    assert r.spec == "openai:gpt-4.1-mini" and r.ok is True and r.error is None
    assert r.content == "OK" and r.latency_s >= 0


def test_smoke_reports_failure_with_error_details(monkeypatch):
    monkeypatch.setattr(llm, "make_client",
                        lambda provider, timeout: FakeClient([_err(openai.NotFoundError, 404, "no such model")]))
    [r] = asyncio.run(llm.smoke(["openai:nope"], timeout=10))
    assert r.ok is False and r.error.status == 404 and "no such model" in r.error.message


def test_smoke_passes_on_200_even_with_empty_content(monkeypatch):
    monkeypatch.setattr(llm, "make_client",
                        lambda provider, timeout: FakeClient([_resp(content="", tokens_in=5, tokens_out=200)]))
    [r] = asyncio.run(llm.smoke(["openai:gpt-5-mini"], timeout=10))
    assert r.ok is True and r.content == ""


def test_smoke_fails_when_key_missing(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    [r] = asyncio.run(llm.smoke(["openrouter:x/y"], timeout=10))
    assert r.ok is False and r.error.type == "MissingAPIKey"


# ---------- extra_body passthrough (grader needs reasoning_effort) ----------

def test_build_params_extra_body_is_forwarded():
    p = llm.build_params("anthropic", "claude-opus-5-5", None, 100, extra_body={"reasoning_effort": "high"})
    assert p == {"max_tokens": 100, "extra_body": {"reasoning_effort": "high"}}


def test_build_params_extra_body_none_leaves_params_unchanged():
    assert llm.build_params("anthropic", "claude-opus-5-5", None, 100, extra_body=None) == {"max_tokens": 100}


def test_cache_key_unchanged_when_extra_body_none_and_changes_when_set():
    base = llm.cache_key("openai", "m", "s", "p", 0, 0.5, 10)
    assert llm.cache_key("openai", "m", "s", "p", 0, 0.5, 10, extra_body=None) == base
    assert llm.cache_key("openai", "m", "s", "p", 0, 0.5, 10, extra_body={"reasoning_effort": "high"}) != base


def test_call_sends_extra_body_to_sdk():
    client = FakeClient([_resp("ok")])
    _call(client, extra_body={"reasoning_effort": "high"})
    assert client.chat.completions.calls[0]["extra_body"] == {"reasoning_effort": "high"}
