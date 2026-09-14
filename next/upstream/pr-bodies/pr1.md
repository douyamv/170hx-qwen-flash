On CUDA builds with CCCL < 3.2 (no `cub::DeviceTopK`, e.g. CUDA 12.4) `ggml_cuda_op_top_k` falls back to
sorting the whole row with `DeviceRadixSort` and copying the first k indices. Models with a sparse-attention indexer
(`qwen4exp`, `deepseek4`, GLM) call this for every query row of every sparse layer with `ncols = n_kv` (up to 262144)
and `k ≈ 2048`, so at long context the sorts became a large part of the decode step.

This PR enables the radix-select kernels that already exist in `top-k.cu` (they were compiled only for
`!GGML_CUDA_USE_CUB && GGML_USE_HIP`) for CUDA when `DeviceTopK` is unavailable, and makes the result deterministic
and identical to the argsort path:

- ties at the k-th value are broken by ascending index (what a stable descending sort produces) via per-block
  counts and an exclusive prefix over the blocks of a row, instead of the previous `atomicAdd` gather;
- the k selected indices are sorted by (value desc, index asc) with an in-block bitonic sort (k ≤ 4096);
- fixes the per-block offset computation for `blocks_per_row > 32` (rows longer than 32768 elements).

Used for `ncols > 1024 && k <= 4096 && 4*k <= ncols`; `GGML_CUDA_TOPK_FORCE_ARGSORT=1` restores the previous
path. When `DeviceTopK` is available nothing changes.

Measurements (NVIDIA CMP 170HX, sm_80, CUDA 12.4, `ggml_top_k` on f32 rows, 20 iterations, output hashes identical
to the argsort path for all cases including ones with many tied values and `-inf` entries):

| shape | k | argsort path | radix select |
|---|---|---|---|
| [70000 × 5] | 2051 | 0.66 ms | 0.155 ms |
| [262144 × 5] | 2051 | 2.12 ms | 0.183 ms |
| [262144 × 1] | 2051 | 0.105 ms | 0.156 ms |
| [16384 × 5] | 2051 | 0.177 ms | 0.141 ms |
| [70000 × 5] | 100 | 0.59 ms | 0.111 ms |

For a single row the argsort path is still slightly faster (fewer launches); the gain comes from batching all query
rows of a layer into one call, which `qwen4exp` now does end to end (see the second table in the linked repo).

Tested: `test-backend-ops -o TOP_K` on CUDA (529/529, including the added `[70000 × {1,5}]`, k = 2051 cases with and without ties) and `-o ARGSORT` (98/98), plus a standalone comparison program with tied inputs. Related:
#28497 (non-deterministic cell sets from `DeviceTopK` on tied block scores) — a deterministic path may be useful
there as well.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
