# x86 quantized dot-product patch

This patch vendors the experimental x86 AVX2 quantized dot-product optimization from [llama.cpp PR #31061](https://github.com/ggml-org/llama.cpp/pull/31061).

- Original candidate commit: `b77e97fcf3ab296391db149d9a38628913e181d6`
- Exact upstream base commit: `7894e8b92481a8bb8ffe94f73b656ffe0dd01e10`
- Modified source: `ggml/src/ggml-cpu/arch/x86/quants.c`
- Focus: Q4_0, Q4_K, Q5_K and Q6_K quantized dot-product kernels, including loop unrolling and independent accumulators.

The patch is experimental. The CI applies it to the pinned upstream base, builds both versions with native CPU flags, runs quantization tests, and compares end-to-end Qwen 1.5B Q4_K_M generation throughput on the same runner. Do not treat microkernel claims as an end-to-end speedup; retain the patch only while the controlled benchmark remains positive and tests pass.
