"""Reproducible CPU benchmark with scheduling and speculative-decoding sweeps."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import statistics
import subprocess
from pathlib import Path
from typing import Any

from .engine import DEFAULT_MODEL, LlamaCppEngine, extract_metrics

PROMPT = """You are an expert Python developer. Write a clear, efficient Python function.
Implement deduplicate_records(records) that removes duplicate dictionaries while
preserving their original order. Explain the time complexity in a short comment.
Return only Python code. Include sensible handling for empty input and repeated records.
"""


def cpu_description() -> str:
    try:
        result = subprocess.run(
            ["lscpu"], check=False, capture_output=True, text=True, timeout=5
        )
        wanted = {
            "Model name", "Architecture", "CPU(s)", "On-line CPU(s) list",
            "Core(s) per socket", "Thread(s) per core", "NUMA node(s)",
            "CPU max MHz", "CPU min MHz",
        }
        fields = {}
        for line in result.stdout.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                if key.strip() in wanted:
                    fields[key.strip()] = value.strip()
        if fields:
            return "; ".join(f"{key}={value}" for key, value in fields.items())
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine()


def run_configuration(
    model: str, threads: int, tokens: int, runs: int,
    batch_size: int, ubatch_size: int,
    cpu_strict: bool = False, poll: int = 50,
    speculative_type: str = "none", speculative_tokens: int = 64,
) -> dict[str, Any]:
    engine = LlamaCppEngine(
        model=model,
        threads=threads,
        threads_batch=threads,
        batch_size=batch_size,
        ubatch_size=ubatch_size,
        cpu_strict=cpu_strict,
        poll=poll,
        speculative_type=speculative_type,
        speculative_tokens=speculative_tokens,
    )
    results: list[dict[str, float | int]] = []
    try:
        with engine:
            engine.complete(PROMPT, max_tokens=16, ignore_eos=True, cache_prompt=False)
            for _ in range(runs):
                response = engine.complete(
                    PROMPT, max_tokens=tokens, ignore_eos=True, cache_prompt=False
                )
                results.append(extract_metrics(response))
    finally:
        engine.close()

    rates = [float(item["generation_tokens_per_second"]) for item in results]
    prompt_rates = [float(item["prompt_tokens_per_second"]) for item in results]
    prompt_counts = [int(item["prompt_tokens"]) for item in results]
    return {
        "threads": threads,
        "threads_batch": threads,
        "batch_size": batch_size,
        "ubatch_size": ubatch_size,
        "cpu_strict": cpu_strict,
        "poll": poll,
        "cache_prompt": False,
        "speculative_type": speculative_type,
        "speculative_tokens": speculative_tokens if speculative_type != "none" else None,
        "runs": results,
        "summary": {
            "generation_tokens_per_second_median": round(statistics.median(rates), 3),
            "generation_tokens_per_second_min": round(min(rates), 3),
            "generation_tokens_per_second_max": round(max(rates), 3),
            "prompt_tokens_per_second_median": round(statistics.median(prompt_rates), 3),
            "prompt_tokens_median": round(statistics.median(prompt_counts), 1),
        },
    }


def run_benchmark(
    model: str, thread_candidates: list[int], tokens: int, runs: int,
    batch_candidates: list[int], ubatch_candidates: list[int],
    cpu_strict_candidates: list[int], poll_candidates: list[int],
    speculative_candidates: list[str], speculative_tokens: int = 64,
) -> dict[str, Any]:
    combinations = [
        (threads, batch, ubatch, bool(strict), poll, speculative)
        for threads, batch, ubatch, strict, poll, speculative in itertools.product(
            thread_candidates, batch_candidates, ubatch_candidates,
            cpu_strict_candidates, poll_candidates, speculative_candidates,
        )
        if ubatch <= batch
    ]
    configurations = [
        run_configuration(
            model, threads, tokens, runs, batch, ubatch,
            cpu_strict=strict, poll=poll,
            speculative_type=speculative,
            speculative_tokens=speculative_tokens,
        )
        for threads, batch, ubatch, strict, poll, speculative in combinations
    ]
    configurations.sort(
        key=lambda item: item["summary"]["generation_tokens_per_second_median"],
        reverse=True,
    )
    return {
        "schema_version": 4,
        "model": model,
        "backend": "llama.cpp",
        "device": "CPU",
        "cpu": cpu_description(),
        "logical_cpus": os.cpu_count(),
        "requested_generation_tokens": tokens,
        "runs_per_configuration": runs,
        "prompt_cache_disabled_for_measurement": True,
        "tested_thread_counts": thread_candidates,
        "tested_cpu_strict": cpu_strict_candidates,
        "tested_poll_levels": poll_candidates,
        "tested_speculative_types": speculative_candidates,
        "tested_batch_sizes": batch_candidates,
        "tested_ubatch_sizes": ubatch_candidates,
        "configurations": configurations,
        "best_configuration": configurations[0],
    }


def parse_thread_candidates(value: str, cpu_count: int | None = None) -> list[int]:
    maximum = cpu_count if cpu_count is not None else (os.cpu_count() or 4) * 2
    candidates = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not candidates or any(item < 1 or item > maximum for item in candidates):
        raise argparse.ArgumentTypeError(
            f"thread counts must be between 1 and {maximum} (2x logical CPUs by default)"
        )
    return candidates


def parse_size_candidates(value: str, name: str) -> list[int]:
    try:
        candidates = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated integers") from exc
    if not candidates or any(item < 1 for item in candidates):
        raise argparse.ArgumentTypeError(f"{name} values must be positive")
    return candidates


def parse_choice_candidates(value: str, allowed: set[str], name: str) -> list[str]:
    candidates = sorted({part.strip() for part in value.split(",") if part.strip()})
    if not candidates or any(item not in allowed for item in candidates):
        raise argparse.ArgumentTypeError(f"{name} choices must be among: {', '.join(sorted(allowed))}")
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark a GGUF model on CPU using llama.cpp."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--thread-candidates", default="1,2,3,4")
    parser.add_argument("--cpu-strict-candidates", default="0,1")
    parser.add_argument("--poll-candidates", default="0,50,100")
    parser.add_argument("--speculative-candidates", default="none,ngram-mod")
    parser.add_argument("--speculative-tokens", type=int, default=64)
    parser.add_argument("--batch-candidates", default="512")
    parser.add_argument("--ubatch-candidates", default="128")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.json"))
    args = parser.parse_args()

    if args.tokens < 32 or args.runs < 2:
        parser.error("--tokens must be >= 32 and --runs must be >= 2")
    if args.speculative_tokens < 1:
        parser.error("--speculative-tokens must be >= 1")
    threads = parse_thread_candidates(args.thread_candidates)
    strict = parse_size_candidates(args.cpu_strict_candidates, "--cpu-strict-candidates")
    if any(item not in {0, 1} for item in strict):
        parser.error("--cpu-strict-candidates values must be 0 or 1")
    polls = parse_size_candidates(args.poll_candidates, "--poll-candidates")
    if any(item > 100 for item in polls):
        parser.error("--poll-candidates values must be between 0 and 100")
    speculative = parse_choice_candidates(
        args.speculative_candidates, {"none", "ngram-mod"}, "--speculative-candidates"
    )
    batches = parse_size_candidates(args.batch_candidates, "--batch-candidates")
    ubatches = parse_size_candidates(args.ubatch_candidates, "--ubatch-candidates")
    if not any(ubatch <= batch for batch in batches for ubatch in ubatches):
        parser.error("at least one ubatch candidate must be <= a batch candidate")

    result = run_benchmark(
        args.model, threads, args.tokens, args.runs, batches, ubatches,
        strict, polls, speculative, args.speculative_tokens,
    )
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
