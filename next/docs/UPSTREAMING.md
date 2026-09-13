# Upstreaming plan

Goal: get the reusable parts merged into [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) as **small,
independently reviewable PRs**, each with a clear motivation, numbers, and `test-backend-ops` coverage where a
kernel is involved. The model-specific graph work for `qwen4exp` is coordinated separately because upstream's
`qwen4exp` graph (dense mask attention, `// TODO: enable sparse attention`) differs from the compact path this
fork inherited from the Unsloth branch.

Conventions to follow (from `CONTRIBUTING.md` / `docs/development`): one topic per PR, `clang-format` on touched
code (see `.clang-format`), no unrelated whitespace changes, keep the `ggml` API stable, add or extend
`tests/test-backend-ops.cpp` cases for new CUDA paths, and state the hardware + numbers in the PR description.
The `NEXT_*` environment switches and `$NEXT_OPT_DIR` runtime files used in this fork are for operations here;
upstream versions should use existing mechanisms (env vars already used by ggml-cuda, or none).

## PR 1 — CUDA: deterministic radix top-k when `cub::DeviceTopK` is unavailable

- File: `ggml/src/ggml-cuda/top-k.cu` (the radix kernels already exist for HIP; this enables them for CUDA with
  CCCL < 3.2, adds a deterministic gather + in-block sort so the result is bit-identical to the argsort path,
  and fixes the block-offset bug for `blocks_per_row > 32`).
- Why: for `ncols > 1024` the CUDA fallback sorts the whole row per top-k call. `qwen4exp` calls it for every
  query row of every QSA layer (k = 2051 of up to 262144).
- Numbers (CMP 170HX / sm_80): `[70000×5]` 0.66 → 0.155 ms, `[262144×5]` 2.1 → 0.18 ms, `[16384×5]` 0.18 → 0.14 ms.
- Test: `tests/test-backend-ops.cpp` TOP_K cases with large ncols and tied values; `next/tools/bench/topk_test.cpp`.
- Also relevant to upstream issue #28497 (non-deterministic cell sets with `DeviceTopK` on tied block scores):
  the deterministic path could be offered as an option there too.

## PR 2 — llama: do not recreate the backend scheduler on re-reserve

- File: `src/llama-context.cpp` (`sched_reserve`).
- Why: `llama_set_sampler()` sets `sched_need_reserve`; `llama-server` calls it for every request. Recreating the
  scheduler frees and reallocates every compute buffer and the pinned host input buffer (0.6–0.9 s per request at
  `-ub 1024`, 262K context; 0.2–0.4 s at `-ub 256`), then the first decode steps page-fault on the fresh buffers.
  `ggml_backend_sched_reserve` on the existing scheduler re-plans and only grows buffers.
- Numbers: TTFT of a 4-token continuation 0.8–1.1 s → 0.25–0.4 s (production), identical outputs.

## PR 3 — llama: honor `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` in `llama_kv_cache` state I/O

- File: `src/llama-kv-cache.cpp` (`state_write`, `state_read_sinfo`).
- Why: server context checkpoints use `PARTIAL_ONLY` for both the target and the draft context; the KV cache
  ignored the flag, so every request copied the MTP draft's full KV (about 2 KB/token, 400–900 MB at 200K) to the
  host. The draft KV needs no checkpoint (`common_memory::seq_rm` rolls it back).
- Numbers: TTFT at 200K 3.5 s → < 1 s.

## PR 4 — ggml-backend: copy user inputs before cross-device inputs in `compute_splits`

- File: `ggml/src/ggml-backend.cpp`.
- Why: without pipeline parallelism (`n_copies == 1`, e.g. whenever `--override-tensor` is used) the user-input
  copy synchronizes the split backend's stream; if the cross-device wait for the previous split was already
  queued on that stream, the host blocks until the previous GPU finishes its whole graph — at every split boundary.
  Ordering the copies removes the stall with no extra memory.

## PR 5 — CUDA: aligned split-layout GEMV for Q8_0/Q4_0 decode batches

- Files: `ggml/src/ggml-cuda/q8a.{cu,cuh}`, hook in `mmvq.cu`.
- Why: Q8_0 blocks are 34 bytes and Q4_0 18 bytes, so `mmvq` can only issue 2-byte aligned loads; on parts with
  few SMs per byte of bandwidth (CMP 170HX: 70 SMs, 1.39 TB/s) the kernel is instruction bound at ~40% of
  bandwidth. Repacking once into `int8[N][K] + half[N][K/32]` allows 16-byte loads.
- Numbers (ggml op level, same error as mmvq): 2560×6144 B=5 40 → 29 µs, 2560×12288 B=4 70 → 47 µs,
  2560×248320 B=5 1027 → 675 µs, Q4_0 2560×248320 B=1 601 → 318 µs.
- Open questions for upstream: the extra VRAM (a second copy of each repacked tensor) and whether to gate it by
  device (A100/H100 with many SMs reach a higher fraction of bandwidth with the existing kernel). Data from those
  GPUs is needed before proposing this; an alternative is a new tensor type with the aligned layout produced at
  quantization time, which would also serve MMQ.

## PR 6 — MTP: acceptance-adaptive draft length

- File: `common/speculative.cpp` (MTP driver).
- Why: with fixed `n_max` the verify batch grows with the draft length whether or not drafts get accepted; an EMA of
  accepted/drafted per sequence chooses n_max / n_max−1 / n_max−2. Cheap, generic, could apply to other drafters.
- Needs: an A/B on a standard model (the numbers here are content dependent: prose ~0.5 acceptance vs code ~0.8).

## PR 7 — qwen4exp graph work (coordinate with upstream first)

Upstream's `qwen4exp` decode still runs dense attention over all `n_kv` with a mask (issue #28734 documents the
resulting depth decay). The items below assume the compact path; they should be discussed in that issue / with
the author of the `levers` patches before opening PRs:

1. persistent pooled block-key cache (lukolszewski's lever2/2b, ported here with an O(1) fast path for one
   contiguous sequence) — credit and co-author accordingly;
2. fused hyper-connection ops (`hc_mix_tail`, `hc_combine`) — implemented here as `GGML_OP_CUSTOM` kinds; an
   upstream version would add proper ops (or generalize the existing DeepSeek-V4 `dsv4_hc_*` ops, whose
   semantics differ: Qwen's mix gate is element-wise, not per stream);
3. fused gather+dequant+cast for the compact K/V;
4. MMQ MoE tile grid sized by the busiest expert (`mmq.cu`, generic — could be its own small PR with the routing
   statistics as justification);
5. strided conv-state store (no `ggml_cont`), `rms_norm+mul` written in the fusable order.

## Not proposed

- `moea` (expert-grouped MoE GEMV): correct, but not faster than `mmvq` yet on this GPU; kept off by default.
- `token_embd`/`per_layer_token_embd` placement, runtime toggle files, systemd/proxy scripts: deployment specific.
