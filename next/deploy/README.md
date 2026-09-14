# Deployment (4× CMP 170HX, 15 GB host RAM)

Files in this directory are the production configuration used on the reference machine, with hostnames and paths
as examples — adjust `launch.json.example`, the systemd units and `proxy.py` to your layout.

## Layout

| Device (as seen by CUDA with the UUID order below) | Contents |
|---|---|
| CUDA0 (bus 02, Gen2 x16) | layers 0–15 |
| CUDA1 (bus 01, Gen2 x8) | layers 16–31 |
| CUDA2 (bus 82, Gen2 x16) | layers 32–47, output head, `token_embd`, MTP draft model + its KV |
| CUDA3 (bus 81, Gen2 x2) | the 27 GB IQ4_NL n-gram (`per_layer_token_embd`) table only |

Select devices by UUID (`CUDA_VISIBLE_DEVICES=GPU-…`) — integer indices reorder between boots. The card on the x2
link only serves embedding-row lookups (3 kernels per step), so its slow link does not matter; do not put layers on it.

## launch.json.example

`run-server.py` reads `launch.json` (`command` + `environment`), starts `pin-main.py` to pin the main thread, and
`exec`s `llama-server`. Notable arguments:

- `--ctx-size 262144 --parallel 1` — one slot; a second slot would need a second 262K KV set.
- `--split-mode layer --tensor-split 16,16,17,0 --fit off --lazy-mode off --load-mode mmap`
- `--override-tensor "^per_layer_token_embd\.weight$=CUDA3,^token_embd\.weight$=CUDA2"` — the n-gram table on the
  spare GPU; `token_embd` in VRAM (llama.cpp keeps it on the CPU by default, mmap'd from disk: on a USB HDD every
  uncached token id cost a 10–40 ms page fault).
- `--flash-attn on --cache-type-k q8_0 --cache-type-v q8_0`
- `--batch-size 1024 --ubatch-size 1024` — prefill 70K: 159 s → 104 s vs 256; needs ~2 GB more VRAM per GPU.
- `--spec-type draft-mtp --spec-draft-model .../mtp-...-Q8_0-q4head.gguf --spec-draft-n-max 5 --spec-draft-device CUDA2 --spec-draft-type-k f16 --spec-draft-type-v f16`
- `--cache-ram 0 --ctx-checkpoints 8` — the default 8 GB prompt cache pushed a 15 GB host into swap.
- `--backend-sampling` — sampling on the GPU (lossless; +5%).
- `--no-context-shift --metrics --jinja`

Environment: `NEXT_QSA_OPT=1` (compact QSA decode path), `NEXT_OPT_DIR=/path/to/opt` (runtime toggle files),
`LLAMA_NO_MMAP_PREFETCH=1`, optionally `LD_PRELOAD=libnext-trace.so` + `NEXT_TRACE_FLAG`/`NEXT_TRACE_LOG` for the
CUPTI tracer (`../tools/trace/next-trace.cpp`; passive until the flag file exists).

## Runtime toggles (`$NEXT_OPT_DIR`, re-read about once per second)

| File | Meaning |
|---|---|
| `spec_n_max` (int) | MTP draft length, clamped to `--spec-draft-n-max` (4 is the sweet spot here) |
| `spec_p_min` (float) | draft stop probability |
| `spec_adaptive_off` (exists) | disable acceptance-adaptive draft length (present in production: fixed 4 drafts measured best) |
| `moea_off` (exists) | fall back from the expert-grouped MoE GEMV to `mmvq` |
| `qsa_min_kv` (int) | context length from which the compact QSA decode path is used (4096) |
| `hc_fuse_off`, `qsa_gather_unfused`, `qsa_topk_rows`, `mmq_grid_off`, `draft_head_target` (exist) | fall back to the unfused / original code paths |

Environment switches (startup only): `NEXT_Q8A=0`, `NEXT_Q8A_MAX_MB` (per-tensor cap for the aligned-layout repack, default
400 MB), `NEXT_MOEA=0` (expert-grouped MoE GEMV is on by default; `moea_off` in `$NEXT_OPT_DIR` switches it off at runtime),
`NEXT_TOPK_NOSORT=1` (skip the sort of the top-k indices — `ggml_top_k` promises no order; in production since opt-v6),
`NEXT_DEVICE_MASK=2` (GPU-generated causal KQ mask, opt-v6.2; `NEXT_DEVICE_MASK_VERBOSE=1` logs the decision),
`NEXT_QSA_NO_BLKCACHE=1`, `NEXT_QSA_PREP_GENERIC=1`, `NEXT_SCHED_RECREATE=1`.

## GPU clocks (do this first)

With a single request in flight the layer-split pipeline keeps each GPU busy only 20–35% of the time, and the NVIDIA
driver never raises the SM clock above the 1140 MHz idle level (memory stays at 1215 MHz). The same 200-token request
decoded at 44 tok/s cold and 70 tok/s with the clock pinned:

```
sudo nvidia-smi -lgc 1410,1410     # all GPUs; undo with nvidia-smi -rgc
```

The service unit does this in `ExecStartPre`. Idle power rises from ~42 W to ~55 W per card.

## Services

- `qwen38-flashnext-opt-262k.service` — the server (needs `TimeoutStopSec` because the HDD load takes ~29 min;
  `ExecStartPre` re-applies PCIe Gen2 on the CMP cards after a cold boot with `cmp170hx-gen2-hotload.py`).
- `qwen38-lb.service` — `proxy.py`, a tiny OpenAI-compatible reverse proxy on :8080 that rewrites model aliases
  and injects `"backend_sampling": true`.

## What the 4-layer mini can and cannot validate

The mini's next-token distribution is almost flat (top-1 vs top-2 probability 4.9e-4 vs 4.8e-4 at the first
position of the 2K test prompt), so its greedy stream flips on differences of ~1e-5. It is a strong test for
crashes, shapes, the checkpoint/rewind logic and for changes that must be bit-identical (fusions, scheduler
ordering, split placement). It cannot judge numerics-level kernel changes (moea, q8a plans): for those compare the
kernel against a FP32 reference on the real expert weights (`../tools/bench/moea_test3.cpp`, `q8a_test.cpp`) and
A/B on production with the runtime kill files. Observed: with `moea` on, moving `token_embd`/the MTP draft to another
GPU changed the mini's stream after 11 tokens while every other feature was placement-invariant — the kernel itself is
bitwise deterministic across GPUs and runs; the flip is a near-tie amplified by a placement-dependent ulp difference.


## Swapping the model for a variant with the same quantization layout

Any GGUF whose tensor list (names, types, dims) matches the Unsloth UD-Q4_K_XL layout is a drop-in for this
deployment (same VRAM, same kernels); `../tools/check_struct.py` compares a download with the base shard by shard
(sizes must be compared with the Hub API's exact `lfs.size`, the headers may be a few hundred bytes longer). The MTP
draft has no counterpart in such repos: keep the base draft layer and rebuild its Q4 head from the new model's
`output.weight` (`../tools/after_dl.sh` generates the `make_mtp_q4head_*` / `make_mini_*` variants, builds the 4-layer
mini and smoke-tests it), then switch `--model` / `--spec-draft-model` in `launch.json` and restart. Done on
2026-09-14 for `huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF` (opt-v7).

Downloading 100 GB through `hf-mirror.com` from this network: the mirror redirects to Hugging Face's xet CDN, which
throttles many-connection clients (aria2c bursts to ~100 MB/s, then collapses below 1 MB/s and stays there); one
`curl -L -C -` stream per shard holds 10–22 MB/s, about 30 MB/s in total (`../tools/dl_huihui.sh`).


## Several slots (opt-v8)

`--parallel 3 --ctx-size 786432` gives three slots of 262K each; the KV total triples, so the draft model, `token_embd`
and the output head go to the spare GPU (`--spec-draft-device CUDA3`,
`--override-tensor "^per_layer_token_embd\.weight$=CUDA3,^token_embd\.weight$=CUDA3,^output\.weight$=CUDA3"` — the three
must share a device) and each layer GPU pays ≈ 1.5 GB per extra slot. Keep the default per-slot KV streams (do not use
`--kv-unified` with this fork). The fast paths are stream-aware from lib-new29 on (OPTIMIZATIONS row 24); with several
slots decoding in the same step the dense GEMVs run on MMQ instead of `q8a`/`moea`. Concurrent agents then keep their own
prompt caches instead of re-prefilling each other's context on every turn (`--parallel 1` re-prefills a 150K context
for ~3.5 min at every switch). Tool-call requests with a grammar disable GPU sampling (llama.cpp limitation).

## Validation before restarting production

A load takes ~29 minutes, so nothing goes live untested:

1. `../tools/gguf/make_mini.py` builds a 4-layer GGUF (3 GDN + 1 QSA layer, PLE removed) from the full model,
   `make_mini_mtp.py` the matching MTP draft; both load in ~2 minutes on one GPU.
2. `../tools/mini/mini_run3.sh` starts `llama-server` with the production flags on chosen GPUs;
   `stream_md5.py` compares greedy token streams between builds/toggles; `mini_ckpt_test.py` checks checkpoint
   rewind; `validate_all.sh` runs the whole matrix (single GPU, toggles off, 3-GPU split, 2K–200K).
3. After deployment: `../tools/validate/deploy_monitor2.sh` (short sweep + TTFT probe) and `post4.sh`
   (70K sweep, TTFT, decode trace).

Limits of the mini, learned the hard way:

- Its next-token distribution is almost flat (top-1 and top-2 differ by ~1e-5), so sampled streams flip on ulp-level
  differences (device placement, GPU vs CPU sampling). Compare CPU-sampling top-k probabilities (`probe_probs.py`,
  `backend_sampling=false`) for equality claims; sampled streams only catch crashes and gross errors.
- On the reference machine it shares a 40 GB GPU with production's 28 GB n-gram table and sits ~0.8 GB from the edge.
  `q8a`'s opportunistic repacks (up to `NEXT_Q8A_MAX_MB` per tensor) can take that headroom, after which the CUDA
  memory pools fail *later*, inside a decode, with `CUDA error: out of memory` — a change that merely *frees* memory
  (the device-side mask did) can therefore appear to crash. Pin `NEXT_Q8A_MAX_MB=300` for every mini run, baseline
  and test alike.
- Prove that a new path actually runs (CUPTI kernel listing, `GGML_SCHED_DEBUG=1` for placement, a verbose switch)
  before trusting an equality test: a disabled path passes every comparison.
- Do not expect bit-identical outputs from a change that only *moves* a computation: ggml-cuda decides some fusions
  by comparing compute-buffer addresses (`ggml_cuda_check_fusion_memory_ranges`), so a different tensor set shifts
  allocations and flips fusion decisions here and there (rounding-level), and other address-dependent kernel choices
  remain even with `GGML_CUDA_DISABLE_FUSION=1`. Verify the moved computation directly (the device mask has
  `NEXT_DEVICE_MASK_CHECK=1`, a readback that compares every mask entry with the host rule) and compare the residual
  against the codebase's own reference noise (fusion on/off, dense vs compact path) on the same prompts.
- `GGML_SCHED_DEBUG` output does not reach the server log; a CUPTI kernel listing per device (`libnext-trace.so`) shows
  where an op runs. Reading an intermediate tensor back after the graph ran needs `ggml_set_output` on it, otherwise
  its memory has already been reused.
