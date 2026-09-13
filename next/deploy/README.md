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
| `spec_adaptive_off` (exists) | disable acceptance-adaptive draft length |
| `qsa_min_kv` (int) | context length from which the compact QSA decode path is used (4096) |
| `hc_fuse_off`, `qsa_gather_unfused`, `qsa_topk_rows`, `mmq_grid_off`, `draft_head_target` (exist) | fall back to the unfused / original code paths |

Environment kill-switches (startup only): `NEXT_Q8A=0`, `NEXT_Q8A_MAX_MB`, `NEXT_MOEA=1` (opt-in), `NEXT_TOPK_SORT=1`,
`NEXT_QSA_NO_BLKCACHE=1`, `NEXT_QSA_PREP_GENERIC=1`, `NEXT_SCHED_RECREATE=1`.

## Services

- `qwen38-flashnext-opt-262k.service` — the server (needs `TimeoutStopSec` because the HDD load takes ~29 min;
  `ExecStartPre` re-applies PCIe Gen2 on the CMP cards after a cold boot with `cmp170hx-gen2-hotload.py`).
- `qwen38-lb.service` — `proxy.py`, a tiny OpenAI-compatible reverse proxy on :8080 that rewrites model aliases
  and injects `"backend_sampling": true`.

## Validation before restarting production

A load takes ~29 minutes, so nothing goes live untested:

1. `../tools/gguf/make_mini.py` builds a 4-layer GGUF (3 GDN + 1 QSA layer, PLE removed) from the full model,
   `make_mini_mtp.py` the matching MTP draft; both load in ~2 minutes on one GPU.
2. `../tools/mini/mini_run3.sh` starts `llama-server` with the production flags on chosen GPUs;
   `stream_md5.py` compares greedy token streams between builds/toggles; `mini_ckpt_test.py` checks checkpoint
   rewind; `validate_all.sh` runs the whole matrix (single GPU, toggles off, 3-GPU split, 2K–200K).
3. After deployment: `../tools/validate/deploy_monitor2.sh` (short sweep + TTFT probe) and `post4.sh`
   (70K sweep, TTFT, decode trace).
