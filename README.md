# 170hx-qwen-flash

Serving **Qwen3.8-Flash-Next** (125B MoE, 6B active, 262K context, Qwen Sparse Attention + Gated DeltaNet +
hyper-connections + MTP) with **llama.cpp on four NVIDIA CMP 170HX mining cards** (40 GB HBM2e each, PCIe Gen2,
no P2P, 15 GB host RAM), and the CUDA/scheduler work that took single-stream decode from ~45 tok/s to ~70 tok/s at 70K context (and 27 → 60+ at long contexts).

This repository is a fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (via the Unsloth
`qwen4exp` branch) plus everything needed to reproduce the deployment: kernels, graph changes, launch files, the
tracing/benchmark tooling and a 4-layer "mini model" harness used to validate every change before touching the
29-minute production load. Model weights are **not** included — see [Deployment](#deployment).

中文摘要见文末 [中文说明](#中文说明).

## Results

Single request (`--parallel 1`), UD-Q4_K_XL weights, q8_0 KV cache, MTP speculative decoding (draft length 4,
GPU sampling), 4× CMP 170HX. Decode = generated tokens / second as reported by `llama-server` (`print_timing`).

| Context | Baseline fork (2026-09-13) | This repo (opt-v4, 2026-09-14 01:25) | This repo (opt-v5, 2026-09-14 07:23, measured) |
|---|---|---|---|
| 2K | 46–55 tok/s | 60–72 | **64–73 (greedy 70–73, sampled 64–72; 48-token turns 82–87)** |
| 70K | 27–45 | 45–52 | **58–70 (greedy 63–70, sampled 58–67)** |
| 200K | 27–37 | 35–45 | not yet re-measured (expected 45–55) |
| Time to first token, 4–17 new tokens at 70K | 0.4–1.2 s (up to 3.5 s at 200K) | **0.25–0.4 s** | 0.24–0.45 s |
| Prefill 70K prompt | 159–186 s | **104 s** | 102 s |
| Full 262K load from USB HDD | ~29 min | ~29 min | same |

Reference points from the community for the same GGUF: 5×RTX 3090 (layer split, no MTP) 54–57 tok/s short /
42 tok/s at 250K ([issue #28734](https://github.com/ggml-org/llama.cpp/issues/28734)); one RTX PRO 6000:
108 tok/s without draft, 144–183 with MTP ([PR #28123](https://github.com/ggml-org/llama.cpp/pull/28123)).

## What is in here

Every optimization is small, has an off-switch, and was checked for **bit-identical or reference-identical output**
on the mini model before deployment. Details, measurements and file pointers: [next/docs/OPTIMIZATIONS.md](next/docs/OPTIMIZATIONS.md).

| Area | Change | Effect |
|---|---|---|
| CUDA GEMV | `q8a`: Q8_0/Q4_0 weights repacked once into a 16-byte-aligned split layout (int8 quants + fp16 scales) and a dp4a GEMV for decode batches ≤ 8 | dense projections 450 → 650–1000 GB/s; Q4_0 draft head 601 → 318 µs |
| CUDA MoE | `moea`: expert-grouped GEMV for `mul_mat_id` (Q4_K up/gate with fused SwiGLU, Q5_1 down): only experts that received tokens get blocks, a warp streams 4 (Q4_K) / 8 (Q5_1) rows with 16/8-byte loads, tokens sharing an expert read it once (2 per pass), int8 activations with fp32 per-32 scale+sum | 1.3–1.6× faster than mmvq on the real experts (Q4_K up/gate T=5 134 → 101 µs, T=8 218 → 136; Q5_1 down T=5 96 → 73), same error as mmvq for Q4_K and 2× lower for Q5_1 |
| CUDA top-k | deterministic radix select for `GGML_OP_TOP_K` (bit-identical to the argsort fallback incl. ties) + batched top-k over all query rows | QSA top-k 0.66 → 0.16 ms (70K), 2.1 → 0.18 ms (262K) per layer |
| CUDA fused ops | hyper-connection mix tail and combine (13 kernels → 2 per block, 2 blocks per layer); fused gather+dequant+cast for the QSA compact path; `rms_norm+mul` in fusable form | ~1,300 fewer kernel launches per step |
| MoE prefill | MMQ tile grid sized by the busiest expert instead of the token count | prefill +20% |
| QSA memory | port of lukolszewski's persistent pooled block-key cache (recompute only dirty blocks) with an O(1) fast path for the single-stream contiguous case | no per-token re-pooling / re-rope of all blocks; 60% less H2D per step |
| Scheduler | copy user inputs before cross-device inputs (no host stall at every GPU boundary); keep the scheduler on re-reserve instead of recreating it (was 0.6–0.9 s of `cudaMallocHost` per request) | TTFT 0.8–1.1 s → 0.25–0.4 s |
| KV cache | honor `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` in `state_write/read` (server checkpoints were copying the whole MTP draft KV, up to 900 MB, every request) | TTFT at 200K 3.5 s → <1 s |
| MTP | draft-only Q4_0 LM head (`make_mtp_q4head.py`), runtime `spec_n_max`/`spec_p_min`, acceptance-adaptive draft length | +4–17% depending on content |
| Misc | GDN l2norm fix backported from upstream (#28068); `token_embd` kept in VRAM (was page-faulting from the HDD); strided conv-state store; mmvf for tiny-N F32 weights; CUDA-graph key stability for MTP | stability / jitter |

Runtime toggles live in `$NEXT_OPT_DIR` as plain files (see [next/deploy/README.md](next/deploy/README.md)):
`spec_n_max`, `spec_p_min`, `spec_adaptive_off`, `qsa_min_kv`, `hc_fuse_off`, `qsa_gather_unfused`,
`qsa_topk_rows`, `mmq_grid_off`, `draft_head_target`. Environment kill-switches: `NEXT_Q8A=0`, `NEXT_MOEA=0`,
`NEXT_TOPK_SORT=1`, `NEXT_QSA_NO_BLKCACHE=1`, `NEXT_SCHED_RECREATE=1`.

## Deployment

Hardware assumptions: 4 GPUs with ≥ 40 GB each (160 GB total), sm_80. Host RAM can be as small as 15 GB because
the 51B-parameter n-gram table is stored as IQ4_NL (27 GB) **on the fourth GPU** instead of host memory — this is
the reason llama.cpp fits where the vLLM/SGLang recipes (which keep a 51–102 GB table in host RAM) do not.

1. Build (CUDA 12.4, sm_80):
   ```bash
   cmake -S . -B build-sm80 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 -DGGML_CUDA_GRAPHS=ON -DCMAKE_BUILD_TYPE=Release
   cmake --build build-sm80 --target llama-server -j
   ```
2. Weights: [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) `UD-Q4_K_XL`
   (104 GiB) and the MTP GGUF from the same repo; optionally create the draft-only Q4 head:
   `python next/tools/gguf/make_mtp_q4head.py`.
3. Launch: copy `next/deploy/launch.json.example` and adjust paths/GPU UUIDs; `run-server.py` starts `llama-server`
   with those arguments and `pin-main.py` pins the main thread. The important flags are in the example:
   layer split `16,16,17,0`, `--override-tensor "^per_layer_token_embd\.weight$=CUDA3,^token_embd\.weight$=CUDA2"`,
   `--spec-type draft-mtp --spec-draft-device CUDA2`, `--batch-size 1024 --ubatch-size 1024`, `--cache-ram 0`,
   `--backend-sampling`. Systemd units and the small OpenAI-compatible proxy are in `next/deploy/`.
4. Validate before going live: `next/tools/mini/` builds a 4-layer GGUF from the full model (`make_mini.py`) and
   runs the exact production flags on one spare GPU in two minutes; `next/tools/bench/` has the sweeps and TTFT probes.

## Upstreaming

The plan to contribute the reusable parts back to llama.cpp, split into small PRs, is in
[next/docs/UPSTREAMING.md](next/docs/UPSTREAMING.md).

## Credits

- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) and the Unsloth `qwen4exp` implementation
  (Daniel Han) this fork is based on; upstream fixes #28023, #28068, #28123.
- Łukasz Olszewski's block-key cache and compact attention work
  ([issue #28734](https://github.com/ggml-org/llama.cpp/issues/28734), branches `issue/28734`, `q8-compact`).
- The QSA compact decode path and the CUPTI tracer were first written with OpenAI Codex; the rest of the
  optimization, tooling and validation in this repository was done with Claude Code.

License: MIT (same as llama.cpp).

## 中文说明

本仓库是在 4 张 NVIDIA CMP 170HX 矿卡（每卡 40 GB HBM2e，PCIe Gen2，无 P2P，主机内存仅 15 GB）上用 llama.cpp
部署 Qwen3.8-Flash-Next（125B MoE，262K 上下文）的完整代码：CUDA 内核与调度器优化、启动/部署脚本、
trace 与基准工具、以及用于上线前验证的 4 层 mini 模型工具链。**不包含模型权重**。

单请求 decode 速度：2K 上下文 46–55 → 60–72（本轮预计 72–77）tok/s；70K 27–45 → 45–52（预计 55–62）；
200K 27–37 → 35–45（预计 42–52）；每轮首 token 延迟从 0.4–3.5 s 降到 0.25–0.4 s；70K prefill 从 159 s 降到 104 s。

主要优化（每项都有开关，上线前都在 mini 模型上做过逐 token 等价性验证）：对齐重排的 Q8_0/Q4_0 解码 GEMV、
按专家分组的 MoE GEMV、确定性的 CUDA radix top-k、超连接与 QSA 路径的算子融合、MoE prefill 的专家负载网格、
lukolszewski 的块键缓存移植、调度器跨卡输入顺序与复用修复、KV 检查点标志修复、MTP 的 Q4 草稿头与自适应草稿长度。
详见 `next/docs/OPTIMIZATIONS.md`；部署步骤见 `next/deploy/README.md`；向上游提交的拆分计划见 `next/docs/UPSTREAMING.md`。
