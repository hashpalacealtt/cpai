# cpai

**Fast, reproducible CPU inference for small language models.**

cpai is an experimental CPU-first inference runner and benchmark harness. The reference workload is **Qwen2.5-1.5B-Instruct**, using the official GGUF **Q4_K_M** quantization and the CPU backend from [llama.cpp](https://github.com/ggml-org/llama.cpp).

## Goals

- Run instruction models without a GPU.
- Keep setup small: Python standard library + llama.cpp.
- Measure generation throughput, prompt throughput, and latency.
- Sweep CPU thread counts and preserve per-configuration results.
- Report CPU details and settings with every result.

## Quick start

1. Build a CPU version of [llama.cpp](https://github.com/ggml-org/llama.cpp) and make `llama-server` available on `PATH`.
2. Use Python 3.10 or newer.
3. Run:

```bash
python -m cpai.benchmark
```

The default model is `Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M`. The first run downloads the model weights; subsequent runs use the Hugging Face cache.

## Tuning

```bash
# Compare 1, 2, 3 and 4 CPU threads
python -m cpai.benchmark --thread-candidates 1,2,3,4

# Benchmark one specific configuration
python -m cpai.benchmark --threads 4 --tokens 256 --runs 5

# Change prompt-processing batch sizes
python -m cpai.benchmark --threads 4 --batch-size 512 --ubatch-size 128
```

The benchmark warms up each configuration, measures repeated completions, and ranks configurations by median generation tokens/second. It does not assume that more threads are always faster.

## CPU build

For a benchmark running on the same machine that compiles llama.cpp, enable native CPU optimizations:

```bash
cmake -S llama.cpp -B llama.cpp/build \
  -DGGML_NATIVE=ON \
  -DGGML_OPENMP=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build -j2 --target llama-server
```

Do not distribute a `GGML_NATIVE=ON` binary to machines with different CPU capabilities. For portable builds, use `GGML_NATIVE=OFF`.

## Metrics and fairness

Each configuration records every run plus median/min/max generation throughput and median prompt throughput. Results are specific to the reported runner and its CPU allocation. Compare results only when model revision, quantization, prompt, token limit, backend revision, and hardware allocation match.

GitHub Actions runs bounded tests on pushes, pull requests and manual dispatches. It does not run scheduled jobs or unrelated workloads.

## License

cpai source code is licensed under MIT. Qwen model weights have their own license; see the [official model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) and [GGUF repository](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF).
