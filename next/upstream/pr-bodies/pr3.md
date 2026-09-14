`llama-server` writes its context checkpoints with `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` for the target and, when
speculative decoding runs a draft context, for the draft context too. `llama_kv_cache::state_write` and
`state_read_sinfo` ignored the flag, so with an MTP draft model every request serialized the draft's whole KV cache
(about 2 KB per token, 400–900 MB at 200K context) to the host and read it back on restore.

A full-attention KV cache is rolled back with `seq_rm` (`common_memory::seq_rm` already does this for both
contexts), so a partial checkpoint carries nothing for it. This PR returns early in both functions when the flag is
set and the cache is not SWA (`hparams.is_swa_any()`), mirroring the existing gates for the indexer caches.

Effect: with Qwen3.8-Flash-Next at 200K context on a 4-GPU box the time to first token of a new turn dropped from
3.5 s to under 1 s; checkpoint rewinds (edited history) still restore correctly (tested by re-sending a prompt whose
last 300 tokens differ and checking the restored position and the generated tokens).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
