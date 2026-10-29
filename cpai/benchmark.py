"""Run a bounded CPU inference benchmark and emit JSON."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
from pathlib import Path
from typing import Any

from .engine import DEFAULT_MODEL, LlamaCppEngine, extract_metrics


PROMPT = """You are an expert Python developer. Write a clear, efficient Python function.
Implement a function named deduplicate_records(records) that removes duplicate dictionaries
while preserving their original order. Explain the time complexity in a short comment.
Return only Python code.
"""


def cpu_description() -> str:
    try:
        result = subprocess.run(
            ["bash", "-lc", "lscpu | grep -E 'Model name|CPU\(s\)' | head -2"],
            check=False, capture_output=True, text=True, timeout=5,
        )
        description = " | ".join(line.split(":", 1)[-1].strip() for line in result.stdout.splitlines())
        if description:
            return description
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine()


def run_benchmark(model: str, threads: int, tokens: int, runs: int) -> dict[str, Any]:
    engine = LlamaCppEngine(model=model, threads=threads)
    results: list[dict[str, float | int]] = []
    try:
        with engine:
            # Short warm-up is excluded from reported measurements.
            engine.complete(PROMPT, max_tokens=8)
            for _ in range(runs):
                response = engine.complete(PROMPT, max_tokens=tokens)
                results.append(extract_metrics(response))
    finally:
        engine.close()

    generation_rates = [float(item["generation_tokens_per_second"]) for item in results]
    prompt_rates = [float(item["prompt_tokens_per_second"]) for item in results]
    return {
        "schema_version": 1,
        "model": model,
        "backend": "llama.cpp",
        "device": "CPU",
        "cpu": cpu_description(),
        "logical_cpus": os.cpu_count(),
        "threads": threads,
        "requested_generation_tokens": tokens,
        "runs": results,
        "summary": {
            "generation_tokens_per_second_median": round(statistics.median(generation_rates), 3),
            "generation_tokens_per_second_min": round(min(generation_rates), 3),
            "generation_tokens_per_second_max": round(max(generation_rates), 3),
            "prompt_tokens_per_second_median": round(statistics.median(prompt_rates), 3),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a GGUF model on CPU using llama.cpp.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--threads", type=int, default=min(os.cpu_count() or 4, 4))
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.json"))
    args = parser.parse_args()

    if args.tokens < 16 or args.runs < 1:
        parser.error("--tokens must be >= 16 and --runs must be >= 1")

    result = run_benchmark(args.model, args.threads, args.tokens, args.runs)
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
