"""ONE CALL. Everything about making a single chat completion correctly.

- provider registry (all reached through the OpenAI SDK, different base URLs)
- disk cache keyed by the full call signature
- call() with retries, exact error records, and param adaptation
- smoke() for pre-run sanity checks

Reusable by anything that needs to hit a model (runner, graders, judges).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import openai
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# provider -> (base_url or None for OpenAI default, env var holding the key)
PROVIDERS: dict[str, tuple[str | None, str]] = {
    "openai": (None, "OPENAI_API_KEY"),
    "anthropic": ("https://api.anthropic.com/v1/", "ANTHROPIC_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
}

# OpenAI-direct reasoning models reject `temperature` and want `max_completion_tokens`.
# Only applies to provider "openai"; other providers get the plain params.
REASONING_PREFIXES = ("gpt-5", "gpt-6", "o1", "o3", "o4")

RETRYABLE_STATUS = {408, 429} | set(range(500, 600))
FATAL_STATUS = {401, 403}  # bad key / no access: never retry, kill the run
# Out of credit arrives as a 429 but will never recover by waiting: treat like a bad key.
FATAL_CODES = {"insufficient_quota"}

CACHE_DIR = Path(".cache")
BACKOFF_BASE = 2.0
BACKOFF_CAP = 64.0
SMOKE_PROMPT = "Reply with the single word OK."
SMOKE_MAX_TOKENS = 256  # reasoning models spend budget thinking; keep room


class MissingAPIKey(Exception):
    pass


# ---------- model spec / params ----------

def parse_model(spec: str) -> tuple[str, str]:
    """'anthropic:claude-haiku-4-5' -> ('anthropic', 'claude-haiku-4-5'). Split on FIRST colon only."""
    if ":" not in spec:
        raise ValueError(f"model spec {spec!r} must be 'provider:model'")
    provider, model = spec.split(":", 1)
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r} in {spec!r}; known: {sorted(PROVIDERS)}")
    if not model:
        raise ValueError(f"empty model name in {spec!r}")
    return provider, model


def make_client(provider: str, timeout: float) -> AsyncOpenAI:
    base_url, env = PROVIDERS[provider]
    key = os.getenv(env)
    if not key:
        raise MissingAPIKey(f"{env} is blank in .env (needed for provider {provider!r})")
    # max_retries=0: WE own the retry loop so every attempt is recorded.
    return AsyncOpenAI(api_key=key, base_url=base_url, timeout=timeout, max_retries=0)


def build_params(provider: str, model: str, temperature: float | None, max_tokens: int) -> dict:
    """Exactly what gets sent. Reasoning adapt lives here and nowhere else."""
    if provider == "openai" and model.startswith(REASONING_PREFIXES):
        return {"max_completion_tokens": max_tokens}
    params: dict = {}
    if temperature is not None:
        params["temperature"] = temperature
    params["max_tokens"] = max_tokens
    return params


# ---------- cache ----------

def cache_key(provider: str, model: str, system_prompt: str, prompt: str,
              sample_index: int, temperature: float | None, max_tokens: int) -> str:
    sig = json.dumps(
        {"provider": provider, "model": model, "system_prompt": system_prompt, "prompt": prompt,
         "sample_index": sample_index, "temperature": temperature, "max_tokens": max_tokens},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(sig.encode("utf-8")).hexdigest()


def cache_get(key: str) -> dict | None:
    p = CACHE_DIR / f"{key}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def cache_put(key: str, record: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    final = CACHE_DIR / f"{key}.json"
    tmp = CACHE_DIR / f"{key}.tmp"
    tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, final)  # atomic: a crash mid-write never leaves a half file at the real name


# ---------- call ----------

@dataclass
class Attempt:
    attempt: int
    status: int | None
    type: str
    code: str | None
    message: str
    retryable: bool


@dataclass
class CallResult:
    response: str | None
    tokens_in: int | None
    tokens_out: int | None
    finish_reason: str | None
    errors: list[Attempt] = field(default_factory=list)
    params_sent: dict = field(default_factory=dict)
    fatal: bool = False  # 401/403 or insufficient_quota seen


def _classify(exc: BaseException, attempt: int) -> Attempt:
    status = getattr(exc, "status_code", None)
    code = None
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        code = (body.get("error") or {}).get("code") if isinstance(body.get("error"), dict) else body.get("code")
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)):
        retryable = True
    elif status in FATAL_STATUS or code in FATAL_CODES:
        retryable = False
    else:
        retryable = status in RETRYABLE_STATUS
    return Attempt(attempt=attempt, status=status, type=type(exc).__name__, code=code,
                   message=str(exc), retryable=retryable)


def _retry_after(exc: BaseException) -> float | None:
    resp = getattr(exc, "response", None)
    val = resp.headers.get("retry-after") if resp is not None else None
    try:
        return float(val) if val is not None else None
    except ValueError:
        return None


def _backoff(attempt: int) -> float:
    return min(BACKOFF_CAP, BACKOFF_BASE * 2 ** (attempt - 1)) + random.random()


async def call(client, provider: str, model: str, system_prompt: str, prompt: str,
               temperature: float | None, max_tokens: int, max_retries: int) -> CallResult:
    params = build_params(provider, model, temperature, max_tokens)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    result = CallResult(None, None, None, None, params_sent=params)
    for attempt in range(1, max_retries + 2):  # max_retries retries = max_retries+1 attempts
        try:
            r = await client.chat.completions.create(model=model, messages=messages, **params)
        except Exception as exc:  # noqa: BLE001 - every failure must be recorded, never swallowed
            a = _classify(exc, attempt)
            result.errors.append(a)
            if a.status in FATAL_STATUS or a.code in FATAL_CODES:
                result.fatal = True
                return result
            if not a.retryable or attempt > max_retries:
                return result
            await asyncio.sleep(_retry_after(exc) or _backoff(attempt))
            continue

        content = r.choices[0].message.content if r.choices else None
        usage = getattr(r, "usage", None)
        if content is None:
            result.errors.append(Attempt(attempt, None, "EmptyResponse", None,
                                         "no choices or null content in 200 response", False))
            return result
        result.response = content
        result.finish_reason = r.choices[0].finish_reason
        result.tokens_in = getattr(usage, "prompt_tokens", None)
        result.tokens_out = getattr(usage, "completion_tokens", None)
        return result
    return result  # unreachable, keeps type checkers happy


# ---------- smoke ----------

@dataclass
class SmokeResult:
    spec: str
    ok: bool
    latency_s: float
    error: Attempt | None
    content: str

    def line(self) -> str:
        if self.ok:
            return f"OK    {self.spec:40s} {self.latency_s:5.2f}s  -> {self.content[:40]!r}"
        e = self.error
        return f"FAIL  {self.spec:40s} {e.type} status={e.status} code={e.code}: {e.message[:200]}"


async def _smoke_one(spec: str, timeout: float) -> SmokeResult:
    provider, model = parse_model(spec)
    t0 = time.perf_counter()
    try:
        client = make_client(provider, timeout)
    except MissingAPIKey as e:
        return SmokeResult(spec, False, 0.0, Attempt(1, None, "MissingAPIKey", None, str(e), False), "")
    r = await call(client, provider, model, "", SMOKE_PROMPT, None, SMOKE_MAX_TOKENS, max_retries=0)
    latency = time.perf_counter() - t0
    # pass = a 200 with a content string. Empty string is fine (reasoning models may spend the
    # whole tiny budget thinking); a null/missing content is a real problem and fails.
    if r.response is not None:
        return SmokeResult(spec, True, latency, None, r.response.strip())
    return SmokeResult(spec, False, latency, r.errors[0], "")


async def smoke(specs: list[str], timeout: float) -> list[SmokeResult]:
    """One tiny call per spec, concurrently. Never uses the cache."""
    return list(await asyncio.gather(*(_smoke_one(s, timeout) for s in specs)))
