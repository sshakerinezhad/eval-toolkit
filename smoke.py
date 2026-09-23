"""Smoke test: one tiny call per model through the SAME client path the runner uses.

    python smoke.py configs/x.yaml            # every model in that run config
    python smoke.py --model openai:gpt-5-mini --model anthropic:claude-haiku-4-5

Exit 0 if all pass, 1 otherwise. Never touches the cache. Costs a few tokens per model.
"""
from __future__ import annotations

import argparse
import asyncio
import sys

import llm
import run


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", help="run config yaml; its models are smoked")
    ap.add_argument("--model", action="append", default=[], help="provider:model (repeatable)")
    ap.add_argument("--timeout", type=float, default=60)
    args = ap.parse_args(argv)

    specs = list(args.model)
    if args.config:
        specs += run.load_config(args.config, {}).models
    if not specs:
        ap.error("give a config file or at least one --model")

    results = asyncio.run(llm.smoke(specs, args.timeout))
    passed = 0
    for r in results:
        passed += r.ok
        print(r.line())
    print(f"\n{passed}/{len(results)} models passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
