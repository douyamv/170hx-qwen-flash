# Optimizations, measurements and where the code lives

All numbers: 4× NVIDIA CMP 170HX (sm_80, 70 SMs @ 1.14 GHz, 1.39 TB/s HBM2e, PCIe Gen2, no P2P), 2× E5-2630 v4,
15 GB RAM, Qwen3.8-Flash-Next UD-Q4_K_XL + MTP GGUF, CUDA 12.4. "Step" = one MTP verify pass (batch of 1 + up to 4
draft tokens) plus the draft passes; one step produces ~2.5–3.5 tokens depending on acceptance.

Method: a CUPTI tracer (`next/tools/trace/next-trace.cpp`, `LD_PRELOAD`, toggled by a flag file) records every
kernel/copy/API call with per-step markers; `an2.py`/`an_layer.py` aggregate kernels per device per step and print
the per-layer kernel sequence. `perf record --call-graph lbr` on the main thread gives the host-side picture
without the tracer's overhead. Every kernel change was first measured standalone on a spare GPU (`next/tools/bench`)
and then end to end on a 4-layer mini model (`next/tools/mini`) with token-level output comparison before touching
the 29-minute production load.

## Where a decode step went (70K context, before Phase C)

GPU kernels 46 ms of a ~52 ms step (89%); host ~6 ms.

| Bucket | ms/step (3 GPUs) | Notes |
|---|---|---|
| dense Q8_0 GEMV (166 calls/GPU) | 13.5 | ~40% of HBM bandwidth: Q8_0 blocks are 34 bytes → 2-byte aligned loads, instruction bound on 70 SMs |
| ~3000 tiny kernels (2–3 µs each) | ~11 | hyper-connection mix/combine, activation quantization, norms, scales, conv-state copies |
| MoE expert GEMV (32 calls/GPU) | 9 | ~600 GB/s, Q4_K up/gate + Q5_1 down |
| hyper-connection small matmuls (mmvf/mul_mat_f/cuBLAS) | 3.6 | [K×4] and [2560×32] shapes, launch bound |
| QSA at 70K (sort, gather, cast, pad, FA) | 3.5 | grows with context |
| MTP draft LM head (Q4_0, 3–4 calls) | 1.9 | 627 µs each |

## Phase A/B (deployed as opt-v3/v4)

| # | Change | Files | Measured |
|---|---|---|---|
| 1 | MoE MMQ (prefill) tile grid sized by the busiest expert; `mmq_stats` counter | `ggml/src/ggml-cuda/mmq.cu` | prefill 8K 402 → 523 tok/s; 70K prefill +20%; real routing max/avg load ≈ 10× |
| 2 | QSA decode fast path threshold made a runtime file (`qsa_min_kv`, default 131072 → 4096) | `src/llama-next-opt.h`, `src/models/qwen4exp.cpp`, `src/llama-memory-hybrid-idx.cpp` | 70K decode 44.5 → 57.4 tok/s |
| 3 | `token_embd` kept in VRAM (`--override-tensor "^token_embd\.weight$=CUDA2"`) | launch flags | removes 10–40 ms HDD page faults per uncached token id (1.2 s TTFT outliers) |
| 4 | Radix top-k on CUDA for `GGML_OP_TOP_K` (CCCL < 3.2 has no `DeviceTopK`; the fallback sorted the whole row per query). Deterministic tie-break = stable sort; output sorted (value desc, index asc) → bit-identical to the argsort path. Batched over all query rows. | `ggml/src/ggml-cuda/top-k.cu`, `src/models/qwen4exp.cpp` | 70K×5 rows 0.66 → 0.155 ms; 262K×5 2.1 → 0.18 ms |
| 5 | Scheduler: copy user inputs before cross-device inputs | `ggml/src/ggml-backend.cpp` | removes the host stall at each GPU boundary (with n_copies=1 the input path synchronized a stream that already waited on the previous device) |
| 6 | Persistent pooled block-key cache (port of lukolszewski's lever2/2b) + O(1) `qsa_prepare` fast path for one contiguous sequence | `src/llama-memory-hybrid-idx.{cpp,h}`, `src/llama-kv-cache.{cpp,h}`, `src/models/qwen4exp.cpp` | no per-token re-pool/re-rope of all blocks; H2D per step 2.3 → 1.8 MB at 70K; output identical to the non-cached path |
| 7 | KV cache honors `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` | `src/llama-kv-cache.cpp` | server checkpoints stopped copying the MTP draft KV (2 KB/token, 400–900 MB at 200K) on every request |
| 8 | Keep the `ggml_backend_sched` on re-reserve (`llama_set_sampler` is called per request and forced a full recreate: `cudaFreeHost`+`cudaMallocHost` of the 512 MB pinned input buffer) | `src/llama-context.cpp` | TTFT 0.8–1.1 s → 0.25–0.4 s |
| 9 | MTP: draft-only Q4_0 LM head added to the MTP GGUF (`make_mtp_q4head.py`), runtime `spec_n_max`/`spec_p_min` | `common/next-opt-common.h`, `common/speculative.cpp`, `src/models/qwen4exp.cpp` | +3–4% (head), n_max 3→4 +4% sampled / +17% greedy at 2K |
| 10 | GDN l2norm fix (backport of #28068), CUDA-graph key includes node count (MTP drafts no longer re-warm), mmvf for tiny-N F32 weights, `--batch-size/--ubatch-size 1024`, `--backend-sampling` | various | prefill 70K 159 → 104 s; GPU sampling lossless (same seed → identical tokens) and +5% |

## Phase C (mini-validated, pending deployment)

| # | Change | Files | Measured |
|---|---|---|---|
| 11 | `q8a`: Q8_0 (and Q4_0) weights repacked once into a split layout (int8 quants contiguous, fp16 scales contiguous, 16-byte aligned) attached via `tensor->extra`; dp4a GEMV, 2 rows/warp, deterministic split-K for K ≥ 4096; hooked at the top of `ggml_cuda_mul_mat_vec_q`. Off: `NEXT_Q8A=0`; size cap `NEXT_Q8A_MAX_MB` (default 400) | `ggml/src/ggml-cuda/q8a.{cu,cuh}`, `mmvq.cu` | ggml-level, same error as mmvq: 2560×6144 B=5 40 → 29 µs, 2560×12288 B=4 70 → 47, 6144×2560 B=5 42 → 34, Q8 LM head B=5 1027 → 675–800, Q4_0 draft head B=1 601 → 318 µs (1.1 TB/s) |
| 12 | Hyper-connection fused ops: `ggml_hc_mix_tail` (sigmoid gate · norm · mean over streams: 7 kernels → 1) and `ggml_hc_combine` (2·sigmoid(inject/hc) fan-out + residual add: 6 → 1); `rms_norm+mul` written so the CUDA fusion applies. Off: `hc_fuse_off` | `ggml/src/ggml-qsa.{c,h}`, `ggml/src/ggml-cuda/qsa.cu`, `src/models/qwen4exp.cpp` | bit-identical tokens fused vs unfused; mini −4.7% ms/token |
| 13 | QSA compact path: fused gather+dequant+cast (Q8_0 rows → padded F16) replaces get_rows+pad+cast | `qsa.cu`, `qwen4exp.cpp` | bit-identical; ~170 µs → ~8 µs per K/V per layer |
| 14 | Conv-state rollback slots written with one strided copy each (no `ggml_cont`) | `qwen4exp.cpp` | 6–12 fewer launches per GDN layer |
| 15 | MTP acceptance-adaptive draft length (EMA of accepted/drafted per sequence; n_max / n_max−1 / n_max−2 at ≥0.70 / ≥0.50 / below). Off: `spec_adaptive_off` | `common/speculative.cpp`, `common/next-opt-common.h` | content dependent (Chinese prose accepts ~0.5, code/English ~0.8) |
| 16 | `moea` v3: expert-grouped MoE GEMV (`mul_mat_id`, ≤ 8 tokens). `mm_ids_helper` groups the (token, slot) pairs per expert; a compact list of active experts sizes the grid (with 512 experts and 5 tokens, 90% of the per-expert blocks used to exit empty); a warp owns 4 rows (Q4_K, 8 lanes per row, 16-byte loads) or 8 rows (Q5_1, 4 lanes per row, 8-byte loads) and processes the expert's tokens 2 per pass, so the kernels stay at 80 registers (3 blocks/SM). Activations are quantized once per column to int8 with an fp32 {scale, sum} per 32-block (per token for up/gate, per (token, slot) for down); fused up·silu(gate). Unfused Q4_K stays on mmvq (no gain there). Off: `NEXT_MOEA=0` at startup or the runtime file `$NEXT_OPT_DIR/moea_off` | `ggml/src/ggml-cuda/moea.{cu,cuh}`, `mmvq.cu` | on the real experts of the model (mini GGUF layers 1/3, T=5): up/gate+swiglu 212 → 185 µs graph, same error as mmvq (mean rel 1.77e-2 vs the FP32 reference for both); down (Q5_1) error halved (1.14e-2 vs 2.34e-2: exact int block sums instead of fp16). Synthetic E=64/128: Q4_K gate T=5 134 → 101 µs, T=8 218 → 136; Q5_1 T=5 96 → 73, T=8 139 → 89. v1/v2 (8 tokens preloaded, 120 regs, 2 rows/warp) were 1.5–10× slower than mmvq: register pressure + wave quantization, found with `ncu` |
| 17 | mmvf for F32/F16 weights with N ≤ 64 (GDN β/α projections went through cuBLAS TF32 + split-K) | `ggml/src/ggml-cuda/mmvf.cu` | 2 launches → 1 per projection, full FP32 |

## Phase D (opt-v6, 2026-09-14)

| # | Change | Files | Measured |
|---|---|---|---|
| 18 | `q8a` v2: rows-per-warp is a template parameter (1–4), the R weight loads of an iteration are issued together; the (R, K-splits) plan per shape comes from a measured table (`q8a_measured`, built with `next/tools/bench/q8a_test2` + `sweep2.sh`), overridable at runtime with `$NEXT_OPT_DIR/q8a_plans` (`N K B R splitk` per line) and for experiments with `NEXT_Q8A_RPW` / `NEXT_Q8A_SPLITK`; `NEXT_Q8A_VERBOSE=1` logs the plan per shape. A first analytical wave-quantization model was wrong (partial last waves cost far less than a full wave), hence the table | `ggml/src/ggml-cuda/q8a.cu` | Q8A_V2_NUMBERS |
| 19 | top-k without the final in-block sort (`NEXT_TOPK_NOSORT=1`): `ggml_top_k` promises no order and the QSA consumers (gather + mask select) are order-independent; the sort existed only to be bit-identical to the argsort fallback | `ggml/src/ggml-cuda/top-k.cu` | 4 × 92 µs per device per step at 70K (k = 2051 → the 4096-element bitonic sort ran on one SM per row) |
| 20 | asynchronous graph-input uploads in the scheduler (`NEXT_SCHED_ASYNC_INPUTS=1`): pinned host inputs go through `ggml_backend_tensor_set_async` instead of copy + per-tensor stream synchronize (llama.cpp never rewrites an input before the graph that reads it has completed) | `ggml/src/ggml-backend.cpp` | ASYNC_NUMBERS |
| 21 | runtime A/B switches without a reload: `$NEXT_OPT_DIR/moea_off` (moea → mmvq; the CUDA graph is re-captured), `$NEXT_OPT_DIR/q8a_plans` | `moea.cu`, `q8a.cu` | — |

Diagnostics that drove Phase D: `ncu` (needs `sudo` on this box: `ERR_NVGPUCTRPERM`) on `moea_test` showed the v1/v2
kernels at 25% theoretical occupancy (120 registers), 0.76 eligible warps per scheduler and 18 waves of mostly empty
blocks; `an_host.py` on the CUPTI trace showed ~320 `cudaStreamSynchronize` and ~114 `cudaMemcpyAsync` per step
(per-tensor input uploads for 3 devices + 4 MTP steps) and ~6 MB/step of H2D at 70K; `LLAMA_GRAPH_RESULT_DEBUG=1`
showed llama's graph reuse working (132 of 136 decodes on the mini), so host time is not graph building.

Mini model end-to-end (4 layers, one GPU, ms per generated token, greedy, no MTP): baseline 12.85 (2K) / 14.66 (16K);
Phase C 11.07 (2K) / 12.73 (16K) → −13%. The mini has one QSA layer and one GPU, so its numbers understate the
production gain from 4 QSA layers per GPU and 5-row MTP batches.

## Things that did not help (measured)

- `GGML_CUDA_GRAPH_OPT=1` (concurrent streams inside a graph): no change on this model.
- `-sm tensor`: not implemented for `qwen4exp`.
- Raising MTP `p_min` to 0.8–0.9: slower (fewer drafts) at every context.
- F16 dense weights instead of Q8_0: slower (2× bytes).
- An earlier naive fused gather kernel: slower than get_rows+pad+cast; the rewrite in #13 fixed it.
- DFlash/DFlash2: no drafter exists for Qwen3.8-Flash-Next and MoE/hybrid targets gain 0.6–1.9× at best; the model's own MTP head is the better drafter here.
- vLLM/SGLang on this box: every recipe keeps the 51B n-gram table in host RAM (51–102 GB); with 15 GB RAM only the IQ4_NL table on GPU3 (27 GB) fits.
