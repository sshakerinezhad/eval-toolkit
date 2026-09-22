"""P1a smoke test: one chat call per provider whose key is set in .env.

Run:  python smoke.py
Blank keys are skipped. Override a model with OPENAI_MODEL / ANTHROPIC_MODEL /
GEMINI_MODEL / OPENROUTER_MODEL in .env if a default name is rejected.
"""
import os
from dotenv import load_dotenv

load_dotenv()
PROMPT = "Reply with the single word OK."


def openai_call():
    from openai import OpenAI
    client = OpenAI()
    r = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        messages=[{"role": "user", "content": PROMPT}],
    )
    return r.choices[0].message.content


def anthropic_call():
    import anthropic
    client = anthropic.Anthropic()
    r = client.messages.create(
        model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"),
        max_tokens=16,
        messages=[{"role": "user", "content": PROMPT}],
    )
    return "".join(b.text for b in r.content if b.type == "text")


def gemini_call():
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    r = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        contents=PROMPT,
    )
    return r.text


def openrouter_call():
    from openai import OpenAI
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    r = client.chat.completions.create(
        model=os.getenv("OPENROUTER_MODEL", "openai/gpt-5-mini"),
        messages=[{"role": "user", "content": PROMPT}],
    )
    return r.choices[0].message.content


PROVIDERS = [
    ("openai", "OPENAI_API_KEY", openai_call),
    ("anthropic", "ANTHROPIC_API_KEY", anthropic_call),
    ("gemini", "GEMINI_API_KEY", gemini_call),
    ("openrouter", "OPENROUTER_API_KEY", openrouter_call),
]

if __name__ == "__main__":
    passed = 0
    for name, env, fn in PROVIDERS:
        if not os.getenv(env):
            print(f"{name:11s} SKIP  ({env} blank)")
            continue
        try:
            out = (fn() or "").strip()
            print(f"{name:11s} OK    -> {out[:40]!r}")
            passed += 1
        except Exception as e:  # noqa: BLE001 - report everything, this is a smoke test
            print(f"{name:11s} FAIL  {type(e).__name__}: {str(e)[:160]}")
    print(f"\n{passed}/4 providers passed")
