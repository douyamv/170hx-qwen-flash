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
| 15 | MTP acceptance-adaptive draft length (EMA of accepted/drafted per sequence; n_max / n_max−1 / n_max−2 at ≥0.70 / ≥0.50 / below). Off: `spec_adaptive_off` | `common/speculative.cpp`, `common/next-opt-common.h` | **was never live**: `libllama-common` was not rebuilt after 2026-09-13 18:46 (only `llama`/`ggml-cuda` targets were built), found on 2026-09-14 when a rebuilt library changed the mini's stream. The production sweeps show a fixed draft length of 4 beats 3 at every context and acceptance seen (2K: 101.9 vs 84.6 tok/s; 70K: 81.4 vs 79.4) — a draft step costs ~2.5 ms against a ~45 ms target step, so shortening drafts below 4 only pays off under ~20% acceptance. Kept disabled (`opt/spec_adaptive_off`) in opt-v6.1 |
| 16 | `moea` v3: expert-grouped MoE GEMV (`mul_mat_id`, ≤ 8 tokens). `mm_ids_helper` groups the (token, slot) pairs per expert; a compact list of active experts sizes the grid (with 512 experts and 5 tokens, 90% of the per-expert blocks used to exit empty); a warp owns 4 rows (Q4_K, 8 lanes per row, 16-byte loads) or 8 rows (Q5_1, 4 lanes per row, 8-byte loads) and processes the expert's tokens 2 per pass, so the kernels stay at 80 registers (3 blocks/SM). Activations are quantized once per column to int8 with an fp32 {scale, sum} per 32-block (per token for up/gate, per (token, slot) for down); fused up·silu(gate). Unfused Q4_K also goes through moea (no speed gain there, but it keeps the numerics independent of whether the scheduler fused up/gate/swiglu — fusion depends on node order, which differs between device splits; with the unfused case on mmvq a 1-GPU and a 2-GPU run of the mini diverged while every other feature was split-invariant). Off: `NEXT_MOEA=0` at startup or the runtime file `$NEXT_OPT_DIR/moea_off` | `ggml/src/ggml-cuda/moea.{cu,cuh}`, `mmvq.cu` | on the real experts of the model (mini GGUF layers 1/3, T=5): up/gate+swiglu 212 → 185 µs graph, same error as mmvq (mean rel 1.77e-2 vs the FP32 reference for both); down (Q5_1) error halved (1.14e-2 vs 2.34e-2: exact int block sums instead of fp16). Synthetic E=64/128: Q4_K gate T=5 134 → 101 µs, T=8 218 → 136; Q5_1 T=5 96 → 73, T=8 139 → 89. v1/v2 (8 tokens preloaded, 120 regs, 2 rows/warp) were 1.5–10× slower than mmvq: register pressure + wave quantization, found with `ncu` |
| 17 | mmvf for F32/F16 weights with N ≤ 64 (GDN β/α projections went through cuBLAS TF32 + split-K) | `ggml/src/ggml-cuda/mmvf.cu` | 2 launches → 1 per projection, full FP32 |

## Phase D (opt-v6, deployed 2026-09-14 11:17 as lib-new21 + launch.json.v4b)

opt-v6 = opt-v5 + moea v3.2 (default on) + q8a v2 plan table + `NEXT_TOPK_NOSORT=1` + the SM clock pin; the asynchronous
input upload (row 20) stays off. Backups on the box: `bin.prev7`, `launch.json.prev7` (= opt-v5).

Measured on production right after the deploy (`post10.sh`; opt-v5 numbers in brackets, both with the SM clock pinned):

| | opt-v6 | opt-v5 |
|---|---|---|
| 2K greedy 4/q4 · 5/q4 · 3/q4 | 101.9 · 92.7 · 84.6 tok/s | 75.3 · 72.4 · 65.8 |
| 2K sampled 4/q4 · 3/q4 · 5/q4 | 103.9 · 92–93 · 95.3 | 72.2 · 66.5–68 · 63.3 |
| 70K greedy 4/q4 · 5/q4 · 3/q4 | 81.4 · 80.2 · 79.4 | 71.3 · 55.0 · 59.1 |
| 70K sampled 5/q4 · 3/q8 · 4/q4 | 81.9 · 77.7 · 75.0 | 58.2 · 58.9 · 57.0 |
| 70K moea A/B (greedy 200 tokens, `opt/moea_off`) | 87.3 on / 79.9 off (identical text for 37 tokens, then a numerics divergence; on/on repeat identical 200/200) | — |
| TTFT, 4–17 new tokens at 70K | 0.24–0.43 s | 0.24–0.45 s |
| prefill 70K | 102 s | 102 s |
| GPU busy per step, 70K trace | 28.8 ms (8.3 / 7.9 / 12.6) | 38.3 ms (11.1 / 10.8 / 16.5) |

200K (probe right after the deploy, `next/tools/validate/prod_200k.py`): prefill 200,014 tokens 408 s; greedy 4/q4 59.6
tok/s, sampled 46.6; TTFT of a 16-token turn 0.64 s (3-token turn 0.39 s). 200K trace: GPU busy 33.4 ms per step
(9.1 / 8.8 / 15.5) of a ~57 ms step, i.e. ~24 ms per step are host work and gaps: `QSA_CPU_INPUT` 1.4 ms, ~6 MB of
per-step H2D (dense `kq_mask` 2 MB per QSA device, `cell_blk`, `bias`), five draft-context decodes (4 drafts + the
catch-up over the accepted tokens; their dense attention over 200K costs 8 × 189 µs + 1 × 487 µs of `flash_attn`
alone), sampling readbacks and two device boundaries.

Per device per step at 70K: `q8a_gemv` 1.8 ms (130 launches, 13.5 µs each ≈ 1.1 TB/s), `moea_q4k` 1.0 ms (16 × 64 µs ≈
1.0 TB/s), `moea_q51` 0.5 ms (15 × 35 µs ≈ 1.2 TB/s), `mul_mat_vec_f` 0.6 ms (72 tiny F32 matmuls), `q8a_quantize`
0.33 ms (166 launches), radix top-k 0.35 ms on the QSA device, no sort. The biggest remaining buckets are the ~1000
small kernels per device (~3.6 ms "other" + cpy/binbcast/norm/unary ≈ 40% of GPU time) and the host/transfer gaps
(step ≈ 42 ms at 70K vs 28.8 ms of GPU work).

| # | Change | Files | Measured |
|---|---|---|---|
| 18 | `q8a` v2: rows-per-warp is a template parameter (1–4), the R weight loads of an iteration are issued together; the (R, K-splits) plan per shape comes from a measured table (`q8a_measured`, built with `next/tools/bench/q8a_test2` + `sweep2.sh`), overridable at runtime with `$NEXT_OPT_DIR/q8a_plans` (`N K B R splitk` per line) and for experiments with `NEXT_Q8A_RPW` / `NEXT_Q8A_SPLITK`; `NEXT_Q8A_VERBOSE=1` logs the plan per shape. A first analytical wave-quantization model was wrong (partial last waves cost far less than a full wave), hence the table | `ggml/src/ggml-cuda/q8a.cu` | sweep with the SM clock pinned (min of 7×100 graph evaluations), best plan vs the v5 rule (R=2): 2560×6144 B=5 23.0 → 21.0 µs (R=3), B=4 22.1 → 19.0 (R=3, 4 splits); 2560×12288 B=4 36.6 → 33.7, B=5 37.9 → 35.9 (R=3); 2560×10240 B=5 31.5 → 29.2 (R=4), B=1 21.9 → 19.8 (R=1, 4 splits); 2560×2560 B=5 28.8 → 17.0 (R=4, 4 splits: the rule left the GPU half empty); N=320/640 rows 10–14% (R=1); K=6144 shapes unchanged. 22 table entries, everything else keeps the rule |
| 19 | top-k without the final in-block sort (`NEXT_TOPK_NOSORT=1`): `ggml_top_k` promises no order and the QSA consumers (gather + mask select) are order-independent; the sort existed only to be bit-identical to the argsort fallback | `ggml/src/ggml-cuda/top-k.cu` | 4 × 92 µs per device per step at 70K (k = 2051 → the 4096-element bitonic sort ran on one SM per row) |
| 20 | asynchronous graph-input uploads in the scheduler (`NEXT_SCHED_ASYNC_INPUTS=1`, **off in production**): pinned host inputs go through `ggml_backend_tensor_set_async` instead of copy + per-tensor stream synchronize. Not safe as is: within one `llama_decode` the next ubatch's `set_inputs` rewrites the same pinned input tensors right after the previous compute was enqueued (llama only synchronizes first when `pipeline_parallel` is on — see the comment in `llama_context::process_ubatch`), so the copy of ubatch *i* can read ubatch *i+1*'s data; upstream's synchronous copy is what makes that safe. A correct version needs double-buffered inputs or an event wait before `set_inputs` | `ggml/src/ggml-backend.cpp` | mini: tokens bit-identical with the flag on and off (host-only change); ~114 `cudaMemcpyAsync` + per-tensor synchronizes per step no longer serialize the host; production effect measured with opt-v6 |
| 23 | GPU-generated causal KQ mask (`NEXT_DEVICE_MASK=1`): the KV cache keeps an I32 cell-position tensor on every device that holds KV layers, `apply_ubatch` writes only the ubatch's cells (a few bytes per step), structural changes (`seq_rm`, `clear`, state restore, …) mark it dirty for one full re-upload; the attention input builds one `ggml_kq_mask_dev` op per device (kind 9 of the custom-op family: keep iff `cell_pos ≥ 0 && cell_pos ≤ pos[t]`) instead of a host-filled `[n_kv, n_tokens]` F16 input that was rebuilt and uploaded to every device each step. Layers use the mask of their own device (`get_kq_mask_l(il)`); the MTP draft context gets it too. Conditions: causal, one stream, one sequence, no SWA / ALiBi; `qwen4exp` batches carry M-RoPE (4-component) positions, which the host code treats as "2-D" — `NEXT_DEVICE_MASK=2` accepts them because for text tokens the rule is exactly `cell.pos ≤ token.pos` on the first component (the extra x/y check only concerns image tokens sharing a position). Found with `NEXT_DEVICE_MASK_VERBOSE=1` after a CUPTI kernel listing showed no `kq_mask_dev` launches — an equality test that passes trivially is worthless, always prove the new path runs. Scheduler: pass 1 of `ggml_backend_sched` places a node only by its own buffer, the INPUT flag or a WEIGHTS source; the mask op has none, so the expansion passes put it next to the token-embedding lookup, and on a layer split the cell table and the mask would cross devices every step — a targeted rule (`1.mask`: custom op kind 9 runs where its cell table lives) fixes that, verified with `GGML_SCHED_DEBUG=1`. Validation trap: on the mini the *smaller* footprint of the device mask let `q8a`'s opportunistic 644 MiB draft-head repack succeed, and the CUDA pools then ran out of memory in the draft's first decode; the first round therefore 'failed' with `CUDA error: out of memory` for a reason unrelated to the mask (mini runs now pin `NEXT_Q8A_MAX_MB=300` for baseline and test alike) Why the outputs are not bit-identical anyway: with `NEXT_DEVICE_MASK_CHECK=1` (debug readback, masks kept alive with `ggml_set_output`) every one of >1000 ubatches in 2K/16K prefills, decodes with and without MTP drafts and checkpoint rewinds matched the host rule exactly (0 mismatching entries), and a CUPTI kernel-sequence diff showed the same kernel multiset but a different *order*: ggml-cuda's fusions (e.g. `topk_moe_cuda` vs the unfused softmax/argsort/get_rows/sum/div chain) pass through `ggml_cuda_check_fusion_memory_ranges`, which compares compute-buffer *addresses*, so any change of the graph's tensor set (here: a computed mask instead of an uploaded one) moves allocations and flips a few fusion decisions — the host path itself alternates them from ubatch to ubatch as `n_kv` changes. Rounding-level: measured against the codebase's own accepted variants on the same prompts, the device mask is the *smallest* perturbation — first-token log-probs identical to 9 digits (host vs device), while fusion on vs off, or the dense vs the compact QSA path, already differ by ~1e-2 in the first token's log-probs and flip the mini's greedy stream after 3–12 tokens; the device mask flips it after 8–48 tokens, and even with `GGML_CUDA_DISABLE_FUSION=1` on both sides a residual (address-dependent kernel choices elsewhere, e.g. cuBLAS/MMQ split decisions) remains after 25–38 tokens. The mask ops are expanded into the graph before the layers (`ggml_build_forward_expand` in the two `llm_graph_context` builders) so they never sit between a fusable pair | `src/llama-kv-cache.{h,cpp}`, `src/llama-graph.{h,cpp}`, `src/llama-context.cpp` (check), `ggml/src/ggml-qsa.{h,c}`, `ggml/src/ggml-cuda/qsa.cu`, `ggml/src/ggml-backend.cpp`, `qwen4exp.cpp` | mini: masks bit-identical to the host rule (readback check, >1000 ubatches); greedy streams identical for 48 tokens in several scenarios, diverging after 8–38 tokens in others on the mini's flat distribution — below the fusion-on/off and dense/compact reference noise (3–12 tokens); production (opt-v6.2, 17:37): VRAM −216/−218/−262 MB on the layer GPUs; 2K greedy 80–95 tok/s right after load (trajectory-dependent acceptance); 70K/200K timelines pending |
| 24 | Multi-slot decode with the fast paths kept: with `--parallel N` (one KV stream per slot, the default non-unified layout) the QSA custom kernels `qsa_pool_norm` / `qsa_expand` / `qsa_mask_select` / `qsa_gather_f16` gained a stream dimension (grid over streams, strides from `nb[]`; the compact path passes the flattened `[n_top_k, n_tps*n_stream]` index rows), the GPU mask gets a `self_kvs` I32 input mapping each mask plane to its KV stream (`seq_to_stream`) and produces `[n_kv, n_tps, 1, n_stream]`, `device_mask_ok` accepts `n_stream == n_seq_max`, and the graph gates changed from `n_stream == 1 && n_tps <= 8` to `n_tps <= 8` (tokens per stream). A unified KV (`--kv-unified`) is *not* supported by the fork's block-key logic (a 4-cell block would mix sequences). `q8a` (B ≤ 8) and `moea` (≤ 8 tokens) still fall back to MMQ/mmvq when several slots decode in one step (3 × (1+4) = 15 tokens). Per 262K slot: ≈ 1.5 GB KV+indexer per layer GPU, 0.6 GB draft KV, 0.11 GB GDN state — the draft, `token_embd` and the output head moved to the spare (PLE) GPU to make room on the last layer GPU | `ggml/src/ggml-qsa.{h,c}`, `ggml/src/ggml-cuda/qsa.cu`, `src/llama-graph.{h,cpp}`, `src/llama-kv-cache.{h,cpp}`, `qwen4exp.cpp` | mini, 3 slots: concurrent requests keep their own caches (round 2: 4 prompt tokens, identical output), per-stream mask readback 0 mismatches over 500+ ubatches, fast paths on vs off agree for the first 6–9 tokens (same as the single-slot dense/compact gap); production opt-v8 (3 × 262K): MSLOT_NUMBERS |
| 22 | SM clock pinned at 1410 MHz (`nvidia-smi -lgc 1410,1410` in `ExecStartPre`): `nvidia-smi` sampled during a decode showed GPUs 0/1/3 at 1140 MHz with 20–35% utilization each — the governor never boosts a GPU that idles two thirds of every step | systemd unit | the same cold 200-token request 44.4 → 70.5 tok/s (identical tokens); `prod_sweep` numbers (sustained, already boosted) unchanged: 2K greedy 4/q4 75.3, 70K 71.3 |
| 21 | runtime A/B switches without a reload: `$NEXT_OPT_DIR/moea_off` (moea → mmvq; the CUDA graph is re-captured), `$NEXT_OPT_DIR/q8a_plans` | `moea.cu`, `q8a.cu` | — |

### opt-v6.1 (2026-09-14 13:32, lib-new22 + launch.json.v5)

opt-v6 + `--spec-draft-backend-sampling` (the MTP draft's top-k sampling on the GPU instead of a 1 MB logits copy and a CPU
softmax per draft step; validated on the mini: target streams identical) + host-timeline markers (`llama_trace_local`
RAII markers in `llama_context::{set_inputs, graph_compute, synchronize}`, the server's decode / accept / draft /
checkpoint / send phases and the MTP driver's process / draft / sample) + a markers-only mode of the CUPTI tracer
(`touch $NEXT_TRACE_FLAG.cpu`: 15 s of CPU markers with no activity tracing, so the host timeline is not inflated) +
`opt/spec_adaptive_off` (see row 15). Numbers (same prompts as opt-v6): 2K greedy 4/q4 102.1 (unchanged), **70K greedy 4/q4 89.2 tok/s (opt-v6 81.4)** — the
GPU-sampled draft stops earlier when unsure (267 drafted / 210 accepted = 79% vs 351 / 210 = 60%), 200K greedy 54.4
(opt-v6 59.6; different acceptance 57% vs 61% on that run — not a like-for-like A/B, the flag is startup-only).
Host timeline (markers only, 70K / 200K, ms per step of 37.8 / 48.3): the target decode 20.7 / 28.2 of which
`SET_INPUTS` 0.5 / 2.2, `QSA_CPU_INPUT` 0.24 / 1.1 and 17.5 / 18.4 inside `ggml_backend_sched_graph_compute_async`
(the host blocks there at both device boundaries: no P2P, the cross-device copy is a blocking host-staged memcpy after a
device synchronize); then 9.3 / 9.4 ms waiting for the last GPU, the draft catch-up decode 1.0 / 2.25, four draft
decodes 4.3 / 6.1 (≈ 0.9 / 1.5 each incl. their GPU time), and 45 `llama_synchronize` calls per step (every
`*_ith` getter synchronizes three CUDA streams). Host-only work is therefore ~6 ms per step at 70K and ~9 ms at 200K;
the rest of the non-GPU time is the serial dependency chain target → catch-up → 4 drafts → target.

Diagnostics that drove Phase D: `ncu` (needs `sudo` on this box: `ERR_NVGPUCTRPERM`) on `moea_test` showed the v1/v2
kernels at 25% theoretical occupancy (120 registers), 0.76 eligible warps per scheduler and 18 waves of mostly empty
blocks; `an_host.py` on the CUPTI trace showed ~320 `cudaStreamSynchronize` and ~114 `cudaMemcpyAsync` per step
(per-tensor input uploads for 3 devices + 4 MTP steps) and ~6 MB/step of H2D at 70K; `LLAMA_GRAPH_RESULT_DEBUG=1`
showed llama's graph reuse working (132 of 136 decodes on the mini), so host time is not graph building.

Two more facts that settle "is the CPU the bottleneck?": during a 2K decode the server's main thread uses 6% of one
core and the whole process 13% (`ps -L`, `/proc/<pid>/stat`) — the host is blocked in `cudaStreamSynchronize`
waiting for the GPUs, not computing; and the decode is already graph-launched (7 `cudaGraphLaunch` per step: 3 target
splits + 4 MTP draft steps). What remains on the host side is one re-capture of a ~242-node graph every step
(1 `cudaStreamBeginCapture` + 242 `cudaLaunchKernel`) and a full 4325-node re-capture whenever the verify batch size
changes (about every 25 steps) — ~1.4 ms per step, the concrete target for the next round. Per-GPU utilization of
20–35% is the layer-split pipeline (one GPU works at a time), not CPU starvation; only tensor parallelism changes it.

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
