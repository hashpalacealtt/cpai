"""Reproducible CPU benchmark with a small thread-count sweep."""

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
            ["lscpu"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        fields = {}
        for line in result.stdout.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                if key.strip() in {"Model name", "Architecture", "CPU(s)", "Core(s) per socket", "Thread(s) per core"}:
                    fields[key.strip()] = value.strip()
        if fields:
            return "; ".join(f"{key}={value}" for key, value in fields.items())
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine()


def run_configuration(
    model: str, threads: int, tokens: int, runs: int,
    batch_size: int, ubatch_size: int,
) -> dict[str, Any]:
    engine = LlamaCppEngine(
        model=model,
        threads=threads,
        threads_batch=threads,
        batch_size=batch_size,
        ubatch_size=ubatch_size,
    )
    results: list[dict[str, float | int]] = []
    try:
        with engine:
            engine.complete(PROMPT, max_tokens=16)
            for _ in range(runs):
                response = engine.complete(PROMPT, max_tokens=tokens)
                results.append(extract_metrics(response))
    finally:
        engine.close()

    rates = [float(item["generation_tokens_per_second"]) for item in results]
    prompt_rates = [float(item["prompt_tokens_per_second"]) for item in results]
    return {
        "threads": threads,
        "threads_batch": threads,
        "batch_size": batch_size,
        "ubatch_size": ubatch_size,
        "runs": results,
        "summary": {
            "generation_tokens_per_second_median": round(statistics.median(rates), 3),
            "generation_tokens_per_second_min": round(min(rates), 3),
            "generation_tokens_per_second_max": round(max(rates), 3),
            "prompt_tokens_per_second_median": round(statistics.median(prompt_rates), 3),
        },
    }


def run_benchmark(
    model: str, thread_candidates: list[int], tokens: int, runs: int,
    batch_size: int = 512, ubatch_size: int = 128,
) -> dict[str, Any]:
    configurations = [
        run_configuration(model, threads, tokens, runs, batch_size, ubatch_size)
        for threads in thread_candidates
    ]
    configurations.sort(
        key=lambda item: item["summary"]["generation_tokens_per_second_median"],
        reverse=True,
    )
    return {
        "schema_version": 2,
        "model": model,
        "backend": "llama.cpp",
        "device": "CPU",
        "cpu": cpu_description(),
        "logical_cpus": os.cpu_count(),
        "requested_generation_tokens": tokens,
        "runs_per_configuration": runs,
        "configurations": configurations,
        "best_configuration": configurations[0],
    }


def parse_thread_candidates(value: str, cpu_count: int | None = None) -> list[int]:
    maximum = cpu_count or (os.cpu_count() or 4)
    candidates = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not candidates or any(item < 1 or item > maximum for item in candidates):
        raise argparse.ArgumentTypeError(f"thread counts must be between 1 and {maximum}")
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a GGUF model on CPU using llama.cpp.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--threads", type=int, default=None, help="Run one configuration with this thread count.")
    parser.add_argument("--thread-candidates", default="1,2,3,4")
    parser.add_argument("--tokens", type=int, default=160)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--ubatch-size", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.json"))
    args = parser.parse_args()

    if args.tokens < 32 or args.runs < 2:
        parser.error("--tokens must be >= 32 and --runs must be >= 2")
    if args.threads is not None:
        candidates = parse_thread_candidates(str(args.threads))
    else:
        candidates = parse_thread_candidates(args.thread_candidates)
    if args.ubatch_size > args.batch_size:
        parser.error("--ubatch-size must not exceed --batch-size")

    result = run_benchmark(
        args.model, candidates, args.tokens, args.runs,
        args.batch_size, args.ubatch_size,
    )
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\\n", encoding="utf-8")


if __name__ == "__main__":
    main()
