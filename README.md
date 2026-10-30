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
# Compare CPU thread counts
python -m cpai.benchmark --thread-candidates 1,2,3,4

# Benchmark one specific configuration
python -m cpai.benchmark --threads 4 --tokens 256 --runs 5

# Tune prompt-processing batch sizes
python -m cpai.benchmark --threads 4 --batch-candidates 512 --ubatch-candidates 128
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

## GitHub Actions

Run **CPU inference checks** manually from the Actions tab. Enable the `run_benchmark` input to additionally build llama.cpp, download the reference model, and run a short CPU benchmark. Benchmark JSON and runner metadata are uploaded as the `cpai-cpu-benchmark` artifact. Benchmark jobs are opt-in and never run on ordinary pushes.

## Metrics and fairness

Each configuration records every run plus median/min/max generation throughput and median prompt throughput. Results are specific to the reported runner and its CPU allocation. Compare results only when model revision, quantization, prompt, token limit, backend revision, and hardware allocation match.

## License

cpai source code is licensed under MIT. Qwen model weights have their own license; see the [official model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) and [GGUF repository](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF).


## CPU tuning notes (Qwen3.5 2B Q4_K_M)

The October 2026 EPYC 7763 run measured a best median of 22.402 generated tokens/s with 4 generation threads, non-strict CPU placement, polling at 100, and speculative decoding disabled. `ngram-mod` was slower in this workload, so it is no longer the engine default. These are measurements on a 4-vCPU shared runner, not universal defaults.

The benchmark now tests exactly eight focused configurations. Generation remains at the measured 4-thread winner while prompt/batch threads (2 or 4), strict CPU placement (off/on), and polling (50/100) are varied independently. llama.cpp exposes `--threads-batch` separately from generation threads, so this isolates prompt-processing tuning without multiplying the entire search space.

Builds use Release mode, native CPU instruction selection (`GGML_NATIVE=ON`), and OpenMP (`GGML_OPENMP=ON`). The runtime sets a fixed OpenMP thread count and core binding for reproducibility. Results should be remeasured on each CPU model.


The next tuning pass tests two versus four generation threads, two versus four prompt/batch threads, and polling disabled versus fully enabled. OpenMP thread count is no longer pinned globally by the workflow: cpai sets it per configuration so the thread-count comparison is fair. The native Release build also enables link-time optimization (`GGML_LTO=ON`); its effect is measured rather than assumed.
