Small F32/F16 weight matrices with N ≤ 64 rows (hyper-connection inject/gate projections `[K, 4]`, GDN β/α
projections `[2560, 32]` in `qwen4exp`) were dispatched to cuBLAS for decode-sized batches: a TF32 GEMM plus a
split-K reduction kernel, ~8 µs and two launches per call, 4 calls per layer. `ggml_cuda_should_use_mmvf` already
accepted N ≤ 16 for these types; this raises the limit to 64. The vector kernel does them in one launch with full
FP32 accumulation and is not slower for any of the shapes we measured on sm_80.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
