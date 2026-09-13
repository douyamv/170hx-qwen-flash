#pragma once

#include "llama-memory-hybrid.h"

#include <memory>
#include <vector>

//
// llama_memory_hybrid_idx
//

// llama_memory_hybrid plus a third cache with one indexer key per token, for block-sparse attention (qwen4exp QSA)
// the indexer is a side buffer over the attention cells: same size, padding, streams and slots, so cell j is one token in both

class llama_memory_hybrid_idx : public llama_memory_hybrid {
public:
    llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
                            /* the indexer cache exists only if this is given */
    const layer_filter_cb & filter_idx);

    ~llama_memory_hybrid_idx() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    //
    // llama_memory_hybrid_idx specific API
    //

    llama_kv_cache * get_mem_idx() const;   // nullptr when the model carries no indexer
    llama_kv_cache * get_mem_blk() const;   // pooled+normed+roped block keys; nullptr without indexer

    // Per-ubatch preparation for the QSA graph: grouping of cells into blocks, plus the
    // delta ("dirty" blocks) against the persistent block-key cache. Computed once per
    // apply() epoch and shared by the graph builder (shapes) and set_input (arrays).
    struct qsa_stream_prep {
        uint32_t stream_id = 0;              // global stream in the caches
        llama_seq_id seq0  = 0;

        bool one_seq = true;
        bool ranked  = false;

        std::vector<int32_t> blk_of;         // [n_kv] cell -> bid (-1 unpooled)
        std::vector<int32_t> cell_grp;       // [n_kv]
        std::vector<int32_t> bid_idx;        // [n_bid] first idx (pb*r) of each block
        std::vector<int32_t> bid_cell;       // [n_bid]
        std::vector<int32_t> bid_slot0;      // [n_bid]
        std::vector<int32_t> blk_cells;      // [r*n_blocks] member cells (0-filled elsewhere)
        std::vector<int32_t> idx_head;       // [r*n_blocks] idx -> cell chains
        std::vector<int32_t> idx_next;       // [n_kv]
        std::vector<int32_t> order;          // ranked mode
        std::vector<int32_t> rank;

        int32_t n_bid = 0;
        bool    oor   = false;

        std::vector<int32_t> dirty;          // bids to (re)compute this ubatch, sorted unique
    };

    struct qsa_prep {
        uint64_t epoch = ~0ull;              // valid when == qsa_epoch
        uint32_t r     = 0;

        int64_t n_kv     = 0;
        int64_t n_blocks = 0;
        int64_t n_dirty  = 0;                // bucketed (pow2, >=4) max over streams; 0 = no blk cache

        std::vector<qsa_stream_prep> streams;
    };

    // runs (or returns the cached) preparation for this apply() epoch.
    // reserve == true computes worst-case shapes only and must not touch any state.
    const qsa_prep & qsa_prepare(const llama_ubatch * ubatch, int64_t n_kv, int64_t n_ns, uint32_t ratio, bool reserve) const;

    // bumped once per context apply; invalidates the qsa_prep cache
    mutable uint64_t qsa_epoch = 0;

    // block-compressed sparse attention (qwen4exp QSA) over the cells of the indexer cache.
    // Blocks cut the position line, not the cell array, so no caller assumes a contiguous layout:
    //   cell_blk  I32 [n_kv, ns]           block each cell belongs to
    //   blk_cells I32 [ratio*n_blocks, ns] cells making up each block
    //   blk_pos   I32 [4*n_blocks*ns]      mrope position rows of each block's first token
    //   bias      F32 [n_kv, n_tokens/ns, ns] -inf where invisible, large where always visible
    // blk_bias asks for the bias per block instead: [n_blocks, n_tokens/ns, ns]
    // the caller then adds the attention mask, the only part of the bias that varies within a block
    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, ggml_tensor * extra_cells, const llama_ubatch * ubatch,
                       int64_t n_kv, uint32_t ratio, bool blk_bias) const;

    // fills the dirty-block recompute inputs (member cells, rope positions, destination
    // rows in the flattened block-key cache) from the current qsa_prep
    void set_input_qsa_dirty(ggml_tensor * dirty_cells, ggml_tensor * dirty_pos,
                             ggml_tensor * dirty_dst, const llama_ubatch * ubatch,
                             int64_t n_kv, uint32_t ratio) const;

private:
    // forget seq_id (all of it if seq_id < 0) in every cache at once, so a failed restore cannot leave the caches out of step
    // seq_id < 0 drops the whole context, as the caches themselves do on a failed restore
    void state_drop(llama_seq_id seq_id);

    // the indexer cache holds one key head per layer, so it needs its own hparams:
    // llama_kv_cache keeps a reference to what it is given
    llama_hparams hparams_idx;
    llama_hparams hparams_blk;

    const std::unique_ptr<llama_kv_cache> mem_idx;

    // block-key cache: one f32 [indexer_head_size] entry per full block per QSA layer,
    // holding the pooled+RMS-normed+roped indexer key. Slot b of stream s mirrors bid b of
    // that stream's grouping; validity is tracked host-side in blk_books.
    const std::unique_ptr<llama_kv_cache> mem_blk;

    // per-global-stream snapshot of the grouping the GPU cache currently reflects:
    // the r member cells of each of n_bid blocks. Compared each ubatch; mismatching or new
    // blocks are recomputed. Any cache mutation (defrag, seq ops, rollback) changes cells
    // and is caught by the compare - no invalidation hooks needed.
    struct blk_book {
        std::vector<int32_t> cells;   // [r*n_bid]
        int32_t n_bid = 0;
    };
    mutable std::vector<blk_book> blk_books;

    mutable qsa_prep prep;
};

class llama_memory_hybrid_idx_context : public llama_memory_hybrid_context {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // used for errors
    explicit llama_memory_hybrid_idx_context(llama_memory_status status);

    // used to create a full-cache context
    explicit llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem);

    // used to create an update context
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                      llama_context * lctx,
                               bool   optimize);

    // used to create a batch processing context from a batch
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                    slot_info_vec_t   sinfos_attn,
                    slot_info_vec_t   sinfos_idx,
          std::vector<llama_ubatch>   ubatches);

    ~llama_memory_hybrid_idx_context() = default;

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    //
    // llama_memory_hybrid_idx_context specific API
    //

    // nullptr with no indexer
    const llama_kv_cache_context * get_idx() const;

    // streams in the current slot info, the `ns` of get_k/get_v; 1 if unified
    uint32_t get_n_stream() const;

    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, ggml_tensor * extra_cells, const llama_ubatch * ubatch,
                       uint32_t ratio, bool blk_bias) const;

    // block-key cache plumbing for the graph builder
    const llama_memory_hybrid_idx::qsa_prep & qsa_prepare(const llama_ubatch * ubatch, uint32_t ratio) const;
    ggml_tensor * blk_k_storage(int32_t il) const;             // [idx_dim, size_blk, n_stream_total]
    ggml_tensor * blk_v_storage(int32_t il) const;             // F32 [ratio, size_blk, n_stream_total] member cells
    uint32_t      blk_stream0(llama_seq_id seq) const;
    bool          blk_available() const;

    void set_input_qsa_dirty(ggml_tensor * dirty_cells, ggml_tensor * dirty_pos,
                             ggml_tensor * dirty_dst, const llama_ubatch * ubatch, uint32_t ratio) const;

private:
    const llama_memory_hybrid_idx * mem = nullptr;

    // streams per ubatch, read from the slot infos before ctx_idx takes them
    // declared first, so it is initialised while sinfos_idx is still intact
    const std::vector<uint32_t> ns_ubatch;

    // null unless the model has an indexer
    const llama_memory_context_ptr ctx_idx;

    // true for the init_full (graph reserve) context: worst-case shapes, no state updates
    bool is_full = false;

    // mirrors the base class's ubatch cursor, which is private there
    size_t i_cur = 0;
};
