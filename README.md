# cpai

**Fast, reproducible CPU inference for small language models.**

cpai is an experimental CPU-first inference runner and benchmark harness. The first reference workload is **Qwen2.5-1.5B-Instruct**, using the official GGUF **Q4_K_M** quantization and the CPU backend from [llama.cpp](https://github.com/ggml-org/llama.cpp).

## Goals

- Run instruction models without requiring a GPU.
- Keep setup small: Python standard library + llama.cpp.
- Measure generation throughput in **tokens/second**, prompt throughput, latency, and peak RSS where available.
- Produce machine-readable JSON so results can be compared across CPUs and CI runs.
- Keep benchmarks honest: report the runner CPU and thread count with every result.

## Quick start

1. Install a CPU build of [llama.cpp](https://github.com/ggml-org/llama.cpp) and make `llama-server` available on your `PATH`.
2. Use Python 3.10 or newer.
3. Run:

```bash
python -m cpai.benchmark
```

By default cpai downloads/loads:

```text
Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M
```

The first run downloads approximately 1.12 GB of model weights. Subsequent runs use the Hugging Face cache.

### Options

```bash
python -m cpai.benchmark --threads 4 --tokens 128 --runs 3
python -m cpai.benchmark --model Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q5_K_M
python -m cpai.benchmark --output results/local.json
```

You can also set `CPAI_LLAMA_SERVER` to the path of your llama-server executable.

## Metrics

Each run records:

- `generation_tokens_per_second`: generated tokens divided by generation time.
- `prompt_tokens_per_second`: prompt tokens processed per second.
- `generation_ms` and `prompt_ms`.
- `generated_tokens`, model identifier, CPU information, thread count, and run count.

The benchmark performs a short warm-up, then takes the median of repeated measured runs. CI results are measurements of that particular hosted runner, not universal hardware guarantees.

## GitHub Actions

The workflow runs on pushes and pull requests that modify cpai, and can also be started manually. It builds llama.cpp for CPU, runs unit tests, downloads the model, executes a bounded benchmark, and uploads the JSON result as a workflow artifact.

No scheduled jobs, GPU requests, mining, or unrelated workloads are used.

## License

The cpai source code is licensed under the MIT License. The Qwen model has its own Apache 2.0 license; see the [official model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) and [GGUF repository](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF). Model weights are downloaded from Hugging Face and are not bundled with cpai.

## Status

Early prototype. Initial benchmark numbers will be published after the first successful CI run.
