#include "llama-trace-local.h"
#include "llama-next-opt.h"
#include <cstdlib>
#include "llama-memory-hybrid-idx.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"


#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <stdexcept>

//
// llama_memory_hybrid_idx
//

llama_memory_hybrid_idx::llama_memory_hybrid_idx(
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
    const layer_filter_cb & filter_idx) :
    llama_memory_hybrid(
        model,
        type_k, type_v, v_trans, kv_size, n_pad, n_swa, swa_type,
        type_r, type_s, rs_size,
        n_seq_max, n_rs_seq, offload, unified,
        filter_attn, filter_recr),
    hparams_idx(model.hparams),
    mem_idx(filter_idx == nullptr ? nullptr : [&] {
        // MQA with a single key head of indexer_head_size, as llama_kv_cache_dsa shapes its own
        std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
        hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size;

        // the cached indexer keys are raw, rotation happens after pooling at read time, so a
        // K-shift must not rotate them while the stream copies in the same update still apply
        hparams_idx.rope_type = LLAMA_ROPE_TYPE_NONE;

        // fool llama_kv_cache into thinking this is a MLA cache, so it won't cache V tensors
        hparams_idx.n_embd_head_k_mla_impl = model.hparams.indexer_head_size;
        hparams_idx.n_embd_head_v_mla_impl = model.hparams.indexer_head_size;

        LLAMA_LOG_INFO("%s: creating indexer KV cache, size = %u cells\n", __func__, kv_size);

        return new llama_kv_cache(
            model, hparams_idx, type_k, type_v, v_trans, offload, unified,
            kv_size, n_seq_max, n_pad, n_swa, swa_type,
            nullptr, filter_idx, nullptr, nullptr, "idx_");
    }()),
    mem_blk(filter_idx == nullptr || getenv("NEXT_QSA_NO_BLKCACHE") ? nullptr : [&]() -> llama_kv_cache * { // NEXT: env disables the block-key cache (full recompute path)
        // the host-side books track one grouping per stream, so a single distinct
        // compress ratio across the indexer layers is required for the cache
        uint32_t r_min = 0;
        uint32_t r_max = 0;
        for (uint32_t il = 0; il < model.hparams.n_layer(); ++il) {
            const uint32_t r = model.hparams.dsv4_compress_ratios[il];
            if (filter_idx(il) && r > 0) {
                r_min = (r_min == 0 || r < r_min) ? r : r_min;
                r_max = std::max(r_max, r);
            }
        }
        if (r_min == 0 || r_min != r_max) {
            return nullptr;
        }

        hparams_blk = hparams_idx;

        // unlike the indexer cache, the blk cache also carries a V tensor: the r member
        // cell indices of each block, stored as f32 (set_rows takes no i32 values), so the
        // top-k expansion gathers on-device instead of re-uploading blk_cells every token
        hparams_blk.n_embd_head_k_mla_impl = 0;
        hparams_blk.n_embd_head_v_mla_impl = 0;
        hparams_blk.n_embd_head_v_full     = r_min;
        hparams_blk.n_embd_head_v_swa      = r_min;

        const uint32_t size_blk = (kv_size + r_min - 1)/r_min;

        LLAMA_LOG_INFO("%s: creating pooled block-key cache, size = %u blocks\n", __func__, size_blk);

        auto * res = new llama_kv_cache(
            model, hparams_blk, GGML_TYPE_F32, GGML_TYPE_F32, false, offload, unified,
            size_blk, n_seq_max, 1, 0, LLAMA_SWA_TYPE_NONE,
            nullptr, filter_idx, nullptr, nullptr, "blk_");

        // slots are addressed directly by block id, never through find_slot, and stale
        // uninitialized VRAM could hold NaNs that poison the block top-k
        res->clear(true);

        return res;
    }()) {}

llama_memory_context_ptr llama_memory_hybrid_idx::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    // note: repeats llama_memory_hybrid::init_batch, as the indexer needs the attention slot infos that the base context hides
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (get_mem_attn()->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = get_mem_recr()->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!get_mem_recr()->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = get_mem_attn()->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // the indexer uses the attention cache's slot layout; a separate one can drift from it
        llama_kv_cache::slot_info_vec_t heads_idx;
        if (mem_idx) {
            heads_idx = heads_attn;
        }

        return std::make_unique<llama_memory_hybrid_idx_context>(
                this, std::move(heads_attn), std::move(heads_idx), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_full() {
    return std::make_unique<llama_memory_hybrid_idx_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_idx_context>(this, lctx, optimize);
}

void llama_memory_hybrid_idx::clear(bool data) {
    llama_memory_hybrid::clear(data);

    if (mem_idx) {
        mem_idx->clear(data);
    }
}

bool llama_memory_hybrid_idx::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // same order as llama_memory_hybrid::seq_rm: the recurrent cache can refuse, so try it first
    if (!get_mem_recr()->seq_rm(seq_id, p0, p1)) {
        return false;
    }

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, p0, p1);
    }

    return get_mem_attn()->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid_idx::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    llama_memory_hybrid::seq_cp(seq_id_src, seq_id_dst, p0, p1);

    if (mem_idx) {
        mem_idx->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    }
}

void llama_memory_hybrid_idx::seq_keep(llama_seq_id seq_id) {
    llama_memory_hybrid::seq_keep(seq_id);

    if (mem_idx) {
        mem_idx->seq_keep(seq_id);
    }
}

void llama_memory_hybrid_idx::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    llama_memory_hybrid::seq_add(seq_id, p0, p1, shift);

    if (mem_idx) {
        mem_idx->seq_add(seq_id, p0, p1, shift);
    }
}

void llama_memory_hybrid_idx::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    llama_memory_hybrid::seq_div(seq_id, p0, p1, d);

    if (mem_idx) {
        mem_idx->seq_div(seq_id, p0, p1, d);
    }
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_idx::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = llama_memory_hybrid::memory_breakdown();

    if (mem_idx) {
        for (const auto & buft_size : mem_idx->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }

    return mb;
}

void llama_memory_hybrid_idx::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    llama_memory_hybrid::state_write(io, seq_id, flags);

    // [TAG_HYBRID_IDX_STATE] the indexer section goes last, so it is a pure suffix: an old reader stops early instead of misparsing it
    // The indexer mirrors the attention cache, so it uses the same PARTIAL_ONLY gate.
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        if (mem_idx) {
            mem_idx->state_write(io, seq_id, flags);
        }
    }

}

void llama_memory_hybrid_idx::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // note: repeats llama_memory_hybrid::state_read
    // the indexer needs the attention cache's cells, and a half-failed restore must leave all three caches alike

    // [TAG_HYBRID_IDX_SINFO]
    // the indexer restore adopts the attention cache's layout instead of searching for cells of its own
    // two find_slot calls agree only while both caches see the same occupancy, which a restore cannot promise
    llama_kv_cache::slot_info_vec_t sinfos_attn;

    try {
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            get_mem_attn()->state_read_sinfo(io, seq_id, flags, mem_idx ? &sinfos_attn : nullptr, nullptr);
        }

        get_mem_recr()->state_read(io, seq_id, flags);

        // [TAG_HYBRID_IDX_STATE] must mirror the write order in state_write
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            if (mem_idx) {
                mem_idx->state_read_sinfo(io, seq_id, flags, nullptr, &sinfos_attn);
            }
        }

    } catch (...) {
        // a half-restored context is the one state the indexer cannot fix by itself: attention holds new cells, the indexer old ones
        // drop what was being restored from all of them, which is a state they do agree on.
        state_drop(seq_id);

        throw;
    }
}

void llama_memory_hybrid_idx::state_drop(llama_seq_id seq_id) {
    // dropped directly, not via seq_rm: the recurrent cache may refuse it and then only the other two get cleared
    if (seq_id < 0) {
        clear(true);

        return;
    }

    get_mem_attn()->seq_rm(seq_id, -1, -1);
    get_mem_recr()->seq_rm(seq_id, -1, -1);

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, -1, -1);
    }
}

llama_kv_cache * llama_memory_hybrid_idx::get_mem_idx() const {
    return mem_idx.get();
}

llama_kv_cache * llama_memory_hybrid_idx::get_mem_blk() const {
    return mem_blk.get();
}

const llama_memory_hybrid_idx::qsa_prep & llama_memory_hybrid_idx::qsa_prepare(
        const llama_ubatch * ubatch,
        int64_t n_kv,
        int64_t n_ns,
        uint32_t ratio,
        bool reserve) const {
    GGML_ASSERT(ratio > 0);
    GGML_ASSERT(get_mem_idx() != nullptr);

    const int64_t r        = ratio;
    const int64_t n_blocks = (n_kv + r - 1)/r;

    if (reserve) {
        // worst-case shapes for graph reservation; no state may change here
        prep.epoch    = ~0ull;
        prep.r        = ratio;
        prep.n_kv     = n_kv;
        prep.n_blocks = n_blocks;
        prep.n_dirty  = mem_blk ? n_blocks : 0;
        prep.streams.clear();
        return prep;
    }

    if (prep.epoch == qsa_epoch && prep.r == ratio && prep.n_kv == n_kv) {
        return prep;
    }

    const int64_t n_tokens = ubatch->n_tokens;

    // a block is keyed on (sequence set, index bucket): a unified cache counts every sequence
    // from zero, so the bucket alone would pool two sequences into one block
    GGML_ASSERT(r <= 64);
    const uint64_t slots_full = r == 64 ? ~uint64_t(0) : ((uint64_t(1) << r) - 1);

    GGML_ASSERT(n_ns >= 1);
    GGML_ASSERT(n_tokens % n_ns == 0);
    const int64_t n_tps = n_tokens/n_ns;

    prep.epoch    = qsa_epoch;
    prep.r        = ratio;
    prep.n_kv     = n_kv;
    prep.n_blocks = n_blocks;
    prep.streams.resize(n_ns);

    if (mem_blk && blk_books.empty()) {
        blk_books.resize(get_mem_idx()->get_n_stream());
    }

    // NEXT: fast path. one stream whose cells are exactly positions 0..valid-1 (no shift, one sequence):
    // block b is cells [b*r, b*r+r) and the tail is the incomplete last block, so every array is
    // arithmetic and the dirty set is the blocks this ubatch writes plus any block the cache lacks.
    if (n_ns == 1 && getenv("NEXT_QSA_OPT") && !getenv("NEXT_QSA_PREP_GENERIC")) {
        const int64_t n_tokens = ubatch->n_tokens;
        const llama_seq_id seq0 = ubatch->seq_id[0][0];
        const auto & cells = get_mem_idx()->get_cells(seq0);
        const int64_t valid = cells.get_used();
        bool eligible = valid > 0 && !cells.get_has_shift() && cells.used_min() == 0 &&
            cells.used_max_p1() == valid && valid <= n_kv && !ubatch->is_pos_2d();
        int seq_count = 0;
        for (int sq = 0; sq < LLAMA_MAX_SEQ && seq_count < 2; ++sq) seq_count += cells.seq_pos_min(sq) >= 0;
        eligible &= seq_count == 1;
        if (eligible) eligible = cells.seq_has(0, seq0);
        for (int64_t i = 1; eligible && i < n_tokens; ++i) eligible = ubatch->seq_id[i][0] == seq0;
        for (int64_t j = 0; eligible && j < valid; ++j) eligible = cells.pos_get(j) == j;
        if (eligible) {
            auto & sp = prep.streams[0];
            sp.seq0 = seq0; sp.stream_id = get_mem_idx()->get_stream_of(seq0);
            sp.one_seq = true; sp.ranked = false; sp.oor = false;
            const int64_t full = valid/r;
            sp.n_bid = (int32_t) full;
            sp.blk_of.assign(n_kv, -1); sp.cell_grp.assign(n_kv, -1);
            for (int64_t j = 0; j < full*r; ++j) { sp.blk_of[j] = (int32_t) (j/r); sp.cell_grp[j] = (int32_t) (j/r); }
            sp.blk_cells.assign(r*n_blocks, 0);
            for (int64_t j = 0; j < full*r; ++j) sp.blk_cells[j] = (int32_t) j;
            sp.idx_head.assign(r*n_blocks, -1); sp.idx_next.assign(n_kv, -1);
            for (int64_t j = 0; j < std::min<int64_t>(valid, r*n_blocks); ++j) sp.idx_head[j] = (int32_t) j;
            sp.bid_idx.resize(full); sp.bid_cell.resize(full); sp.bid_slot0.resize(full);
            for (int64_t b = 0; b < full; ++b) { sp.bid_idx[b] = (int32_t) (b*r); sp.bid_cell[b] = (int32_t) (b*r); sp.bid_slot0[b] = (int32_t) (b*r); }
            sp.dirty.clear();
            int64_t n_dirty_max = 0;
            if (mem_blk) {
                auto & book = blk_books[sp.stream_id];
                const int32_t n_cmp = std::min(sp.n_bid, book.n_bid);
                for (int32_t b = 0; b < n_cmp; ++b) {
                    if (memcmp(sp.blk_cells.data() + (size_t) b*r, book.cells.data() + (size_t) b*r, r*sizeof(int32_t)) != 0) sp.dirty.push_back(b);
                }
                for (int32_t b = n_cmp; b < sp.n_bid; ++b) sp.dirty.push_back(b);
                llama_pos p_min = INT32_MAX, p_max = INT32_MIN;
                for (int64_t ii = 0; ii < n_tokens; ++ii) { p_min = std::min(p_min, ubatch->pos[ii]); p_max = std::max(p_max, ubatch->pos[ii]); }
                const int32_t b_lo = (int32_t) (p_min/r), b_hi = (int32_t) (p_max/r);
                for (int32_t b = std::max(0, b_lo); b <= b_hi && b < sp.n_bid; ++b) sp.dirty.push_back(b);
                std::sort(sp.dirty.begin(), sp.dirty.end());
                sp.dirty.erase(std::unique(sp.dirty.begin(), sp.dirty.end()), sp.dirty.end());
                book.n_bid = sp.n_bid;
                book.cells.assign(sp.blk_cells.begin(), sp.blk_cells.begin() + (size_t) sp.n_bid*r);
                n_dirty_max = (int64_t) sp.dirty.size();
                int64_t bucket = 4;
                while (bucket < n_dirty_max) bucket *= 2;
                prep.n_dirty = std::min<int64_t>(bucket, n_blocks);
                prep.n_dirty = std::max<int64_t>(prep.n_dirty, std::min<int64_t>(4, n_blocks));
            } else {
                prep.n_dirty = 0;
            }
            return prep;
        }
    }

    // scratch shared across streams
    std::vector<int32_t>  grp_head(n_blocks);
    std::vector<int32_t>  grp_next;
    std::vector<int32_t>  grp_first;
    std::vector<int32_t>  grp_slot0;
    std::vector<uint64_t> grp_slots;
    std::vector<int32_t>  grp_bid;

    int64_t n_dirty_max = 0;

    for (int64_t s = 0; s < n_ns; ++s) {
        auto & sp = prep.streams[s];

        const llama_seq_id seq_of_stream = ubatch->seq_id[s*n_tps][0];
        const auto & cells = get_mem_idx()->get_cells(seq_of_stream);

        sp.seq0      = seq_of_stream;
        sp.stream_id = get_mem_idx()->get_stream_of(seq_of_stream);

        sp.blk_of  .assign(n_kv, -1);
        sp.cell_grp.assign(n_kv, -1);
        sp.blk_cells.assign(r*n_blocks, 0);
        sp.idx_head.assign(r*n_blocks, -1);
        sp.idx_next.assign(n_kv, -1);
        sp.bid_idx  .clear();
        sp.bid_cell .clear();
        sp.bid_slot0.clear();
        sp.dirty    .clear();

        int n_seq_present = 0;

        for (int sq = 0; sq < LLAMA_MAX_SEQ && n_seq_present < 2; ++sq) {
            if (cells.seq_pos_min(sq) >= 0) {
                n_seq_present++;
            }
        }

        sp.one_seq = n_seq_present <= 1;

        bool oor = false;
        bool dup = false;

        sp.ranked = false;

        auto group_cells = [&]() {
            std::fill(sp.blk_of.begin(),   sp.blk_of.end(),   -1);
            std::fill(sp.cell_grp.begin(), sp.cell_grp.end(), -1);
            std::fill(grp_head.begin(),    grp_head.end(),    -1);

            grp_next .clear();
            grp_first.clear();
            grp_slot0.clear();
            grp_slots.clear();
            grp_bid  .clear();

            oor = false;
            dup = false;

            for (int64_t j = 0; j < n_kv; ++j) {
                if (cells.is_empty(j)) {
                    continue;
                }

                const int64_t idx = sp.ranked ? sp.rank[j] : cells.pos_get(j);
                const int64_t pb  = idx/r;

                if (pb >= n_blocks) {
                    oor = true;
                    continue;
                }

                int32_t g = -1;

                for (int32_t c = grp_head[pb]; c >= 0; c = grp_next[c]) {
                    if (sp.one_seq || cells.seq_get_all((uint32_t) grp_first[c]) == cells.seq_get_all((uint32_t) j)) {
                        g = c;
                        break;
                    }
                }

                if (g < 0) {
                    g = (int32_t) grp_first.size();

                    grp_next .push_back(grp_head[pb]);
                    grp_first.push_back((int32_t) j);
                    grp_slot0.push_back(-1);
                    grp_slots.push_back(0);
                    grp_bid  .push_back(-1);

                    grp_head[pb] = g;
                }

                const uint64_t bit = uint64_t(1) << (idx%r);

                dup |= (grp_slots[g] & bit) != 0;

                sp.cell_grp[j]  = g;
                grp_slots[g]   |= bit;

                if (idx%r == 0) {
                    grp_slot0[g] = (int32_t) j;
                }
            }
        };

        group_cells();

        // mrope repeats one position across an image, so rank cells instead of using the position
        if (dup && ubatch->is_pos_2d() && sp.one_seq) {
            sp.order.clear();
            sp.order.reserve(n_kv);

            for (int64_t j = 0; j < n_kv; ++j) {
                if (!cells.is_empty(j)) {
                    sp.order.push_back((int32_t) j);
                }
            }

            std::sort(sp.order.begin(), sp.order.end(), [&cells](int32_t a, int32_t b) {
                const llama_pos pa = cells.pos_get(a);
                const llama_pos pb = cells.pos_get(b);

                if (pa != pb) {
                    return pa < pb;
                }

                const auto & ea = cells.ext_get(a);

                return cells.ext_get(b).is_2d_gt(ea.x, ea.y);
            });

            sp.rank.assign(n_kv, -1);

            for (int64_t k = 0; k < (int64_t) sp.order.size(); ++k) {
                sp.rank[sp.order[k]] = (int32_t) k;
            }

            sp.ranked = true;

            group_cells();
        }

        sp.oor = oor;

        sp.n_bid = 0;

        for (int64_t pb = 0; pb < n_blocks; ++pb) {
            for (int32_t g = grp_head[pb]; g >= 0; g = grp_next[g]) {
                if (grp_slots[g] != slots_full) {
                    continue;
                }

                grp_bid[g] = sp.n_bid++;

                sp.bid_idx  .push_back((int32_t) (pb*r));
                sp.bid_cell .push_back(grp_first[g]);
                sp.bid_slot0.push_back(grp_slot0[g]);
            }
        }

        GGML_ASSERT(sp.n_bid <= n_blocks);

        for (int64_t j = 0; j < n_kv; ++j) {
            const int32_t g = sp.cell_grp[j];

            sp.blk_of[j] = g < 0 ? -1 : grp_bid[g];

            if (g >= 0) {
                const int64_t idx = sp.ranked ? sp.rank[j] : cells.pos_get(j);

                if (sp.blk_of[j] >= 0) {
                    sp.blk_cells[sp.blk_of[j]*r + (idx%r)] = (int32_t) j;
                }

                if (idx < r*n_blocks) {
                    sp.idx_next[j] = sp.idx_head[idx];
                    sp.idx_head[idx] = (int32_t) j;
                }
            }
        }

        // ---- block-key cache delta ------------------------------------------------
        if (mem_blk) {
            auto & book = blk_books[sp.stream_id];

            if (sp.ranked) {
                // rank-based grouping can reshuffle wholesale; recompute everything
                for (int32_t b = 0; b < sp.n_bid; ++b) {
                    sp.dirty.push_back(b);
                }
            } else {
                // structural mismatches against what the GPU cache holds
                const int32_t n_cmp = std::min(sp.n_bid, book.n_bid);

                for (int32_t b = 0; b < n_cmp; ++b) {
                    if (memcmp(sp.blk_cells.data() + (size_t) b*r, book.cells.data() + (size_t) b*r, r*sizeof(int32_t)) != 0) {
                        sp.dirty.push_back(b);
                    }
                }

                for (int32_t b = n_cmp; b < sp.n_bid; ++b) {
                    sp.dirty.push_back(b);
                }

                // blocks whose member cells this ubatch (re)writes: raw keys change even
                // when the cell layout does not
                llama_pos p_min = INT32_MAX;
                llama_pos p_max = INT32_MIN;

                for (int64_t ii = 0; ii < n_tps; ++ii) {
                    const llama_pos p = ubatch->pos[s*n_tps + ii];
                    p_min = std::min(p_min, p);
                    p_max = std::max(p_max, p);
                }

                const int32_t pb_lo = (int32_t) (p_min/r)*r;
                const int32_t pb_hi = (int32_t) (p_max/r)*r;

                for (int32_t b = 0; b < sp.n_bid; ++b) {
                    if (sp.bid_idx[b] >= pb_lo && sp.bid_idx[b] <= pb_hi) {
                        sp.dirty.push_back(b);
                    }
                }

                std::sort(sp.dirty.begin(), sp.dirty.end());
                sp.dirty.erase(std::unique(sp.dirty.begin(), sp.dirty.end()), sp.dirty.end());
            }

            // the book now describes the state the cache will be in after this graph runs
            book.n_bid = sp.n_bid;
            book.cells.assign(sp.blk_cells.begin(), sp.blk_cells.begin() + (size_t) sp.n_bid*r);

            n_dirty_max = std::max<int64_t>(n_dirty_max, (int64_t) sp.dirty.size());
        }

    }

    if (mem_blk) {
        // fixed buckets keep the graph topology stable across decode steps (CUDA graphs)
        int64_t bucket = 4;
        while (bucket < n_dirty_max) {
            bucket *= 2;
        }
        prep.n_dirty = std::min<int64_t>(bucket, n_blocks);
        prep.n_dirty = std::max<int64_t>(prep.n_dirty, std::min<int64_t>(4, n_blocks));
    } else {
        prep.n_dirty = 0;
    }

    return prep;
}

void llama_memory_hybrid_idx::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        ggml_tensor * extra_cells,
        const llama_ubatch * ubatch,
        int64_t n_kv,
        uint32_t ratio,
        bool blk_bias) const {
    GGML_ASSERT(ratio > 0);
    GGML_ASSERT(get_mem_idx() != nullptr);

    GGML_ASSERT(blk_cells == nullptr || ggml_backend_buffer_is_host(blk_cells->buffer));
    GGML_ASSERT(blk_pos == nullptr || ggml_backend_buffer_is_host(blk_pos->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(bias->buffer));
    GGML_ASSERT(cell_blk    == nullptr || ggml_backend_buffer_is_host(cell_blk->buffer));
    GGML_ASSERT(extra_cells == nullptr || ggml_backend_buffer_is_host(extra_cells->buffer));

    const int64_t n_ns     = bias->ne[2];            // streams in this ubatch
    const int64_t r        = ratio;
    const int64_t n_blocks = blk_bias ? bias->ne[0] : blk_cells->ne[0]/r;
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(n_tokens % n_ns == 0);
    const int64_t n_tps = n_tokens/n_ns;             // tokens per stream

    llama_trace_local trace("QSA_CPU_INPUT");

    const auto & pr = qsa_prepare(ubatch, n_kv, n_ns, ratio, false);
    GGML_ASSERT((int64_t) pr.streams.size() == n_ns);
    GGML_ASSERT(pr.n_blocks == n_blocks);

    int32_t * dst_cell_blk  = cell_blk    ? (int32_t *) cell_blk->data    : nullptr;
    int32_t * dst_blk_cells = blk_cells   ? (int32_t *) blk_cells->data   : nullptr;
    int32_t * dst_blk_pos   = blk_pos ? (int32_t *) blk_pos->data : nullptr;
    float   * dst_bias      = (float   *) bias->data;
    int32_t * dst_extra     = extra_cells ? (int32_t *) extra_cells->data : nullptr;

    if (dst_blk_pos) {
        std::fill(dst_blk_pos, dst_blk_pos + 4*n_blocks*n_ns, 0);
    }

    for (int64_t s = 0; s < n_ns; ++s) {
        const auto & sp = pr.streams[s];

        const llama_seq_id seq_of_stream = ubatch->seq_id[s*n_tps][0];
        const auto & cells = get_mem_idx()->get_cells(seq_of_stream);

        GGML_ASSERT((!blk_bias || !sp.oor) && "qsa: cell position runs past the cell window");

        if (dst_blk_cells) {
            memcpy(dst_blk_cells + s*(r*n_blocks), sp.blk_cells.data(), r*n_blocks*sizeof(int32_t));
        }

        for (int32_t b = 0; dst_blk_pos && b < sp.n_bid; ++b) {
            int32_t sec_pos[4] = { sp.bid_idx[b], sp.bid_idx[b], sp.bid_idx[b], sp.bid_idx[b] };

            if (sp.ranked) {
                const int32_t   c = sp.bid_slot0[b];
                const llama_pos p = cells.pos_get(c);
                const auto &    e = cells.ext_get(c);

                sec_pos[0] = p;
                sec_pos[1] = e.y;
                sec_pos[2] = e.x;
                sec_pos[3] = p;
            }

            for (int64_t sec = 0; sec < 4; ++sec) {
                dst_blk_pos[sec*(n_blocks*n_ns) + s*n_blocks + b] = sec_pos[sec];
            }
        }

        // unpooled cells all point at one spare block. a spare block exists only when some
        // cell is unpooled: n_bid == n_blocks means every cell sits in a full block.
        const bool     have_dead = sp.n_bid < n_blocks;
        const int32_t  dead_bid  = have_dead ? sp.n_bid : (int32_t) n_blocks - 1;

        if (dst_cell_blk) {
            for (int64_t j = 0; j < n_kv; ++j) {
                dst_cell_blk[s*n_kv + j] = sp.blk_of[j] < 0 ? dead_bid : sp.blk_of[j];
            }
        }

        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t      i      = s*n_tps + ii;
            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            int64_t q = ubatch->pos[i];

            if (sp.ranked) {
                const llama_pos qt = ubatch->pos[i];
                const llama_pos qy = ubatch->pos[i + n_tokens];
                const llama_pos qx = ubatch->pos[i + n_tokens*2];

                int64_t lo = 0;
                int64_t hi = (int64_t) sp.order.size();

                while (lo < hi) {
                    const int64_t   mid = (lo + hi)/2;
                    const int32_t   c   = sp.order[mid];
                    const llama_pos pc  = cells.pos_get(c);

                    if (pc < qt || (pc == qt && !cells.ext_get(c).is_2d_gt(qx, qy))) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }

                q = lo - 1;
            }

            // the tail is an incomplete block and is always visible, as in the reference
            const int64_t tail_start = (q + 1)/r*r;

            if (blk_bias) {
                // a block sits wholly inside or outside the tail, so one value covers it
                // the top-k here runs over blocks, so a fully-future block must be -inf: the
                // attention mask is only added after the selection and cannot reclaim a
                // wasted budget slot. a block straddling q keeps the tail value; the mask
                // still drops its future cells after expansion.
                float * cur_blk_bias = dst_bias + i*n_blocks;

                for (int64_t b = 0; b < n_blocks; ++b) {
                    if (b >= sp.n_bid || !cells.seq_has((uint32_t) sp.bid_cell[b], seq_id)) {
                        cur_blk_bias[b] = -INFINITY;
                        continue;
                    }

                    if (sp.bid_idx[b] > q) {
                        cur_blk_bias[b] = -INFINITY;
                        continue;
                    }

                    // finite, so it can never meet a -inf and produce a nan
                    cur_blk_bias[b] = sp.bid_idx[b] >= tail_start ? 1e9f : 0.0f;
                }

                // the spare block holds the unpooled cells, which are the incomplete tail, so
                // it gets the tail value. it must stay finite: a sequence with fewer than
                // `ratio` cells owns no full block, and a row of -inf only gives a nan.
                if (have_dead) {
                    cur_blk_bias[dead_bid] = 1e9f;
                }

                if (dst_extra) {
                    // the spare block's blk_cells row is zero-filled, so the tail cells the
                    // reference always attends must ride separately. padded by repetition;
                    // duplicates only unmask the same mask cell twice.
                    int32_t * cur_extra = dst_extra + i*r;

                    int64_t n_extra = 0;
                    for (int64_t idx = tail_start; idx <= q && n_extra < r; ++idx) {
                        for (int32_t c = sp.idx_head[idx]; c >= 0; c = sp.idx_next[c]) {
                            if (cells.seq_has((uint32_t) c, seq_id)) {
                                cur_extra[n_extra++] = c;
                                break;
                            }
                        }
                    }

                    if (n_extra == 0) {
                        // empty tail: fall back to the token's own cell, always visible on
                        // the diagonal. cell 0 if even that fails; the mask decides then.
                        int32_t own = 0;
                        if (q >= 0 && q < r*n_blocks) {
                            for (int32_t c = sp.idx_head[q]; c >= 0; c = sp.idx_next[c]) {
                                if (cells.seq_has((uint32_t) c, seq_id)) {
                                    own = c;
                                    break;
                                }
                            }
                        }
                        cur_extra[n_extra++] = own;
                    }

                    while (n_extra < r) {
                        cur_extra[n_extra] = cur_extra[n_extra - 1];
                        ++n_extra;
                    }
                }

                continue;
            }

            float * cur_bias = dst_bias + i*n_kv;

            for (int64_t j = 0; j < n_kv; ++j) {
                float v = -INFINITY;

                if (!cells.is_empty(j) && cells.seq_has(j, seq_id)) {
                    const int64_t idx = sp.ranked ? sp.rank[j] : cells.pos_get(j);

                    if (idx <= q) {
                        // finite, so it can never meet a -inf and produce a nan
                        v = idx >= tail_start ? 1e9f : (sp.blk_of[j] < 0 ? -INFINITY : 0.0f);
                    }
                }

                cur_bias[j] = v;
            }
        }
    }
}

void llama_memory_hybrid_idx::set_input_qsa_dirty(
        ggml_tensor * dirty_cells,
        ggml_tensor * dirty_pos,
        ggml_tensor * dirty_dst,
        const llama_ubatch * ubatch,
        int64_t n_kv,
        uint32_t ratio) const {
    GGML_ASSERT(mem_blk != nullptr);
    GGML_ASSERT(ggml_backend_buffer_is_host(dirty_cells->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(dirty_pos->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(dirty_dst->buffer));

    // set_input_qsa has always prepared this epoch already (it runs first in set_input)
    GGML_ASSERT(prep.epoch == qsa_epoch);
    const auto & pr = prep;

    const int64_t r       = ratio;
    const int64_t n_ns    = (int64_t) pr.streams.size();
    const int64_t n_dirty = pr.n_dirty;

    GGML_ASSERT(dirty_dst->ne[0]   == n_dirty*n_ns);
    GGML_ASSERT(dirty_cells->ne[0] == r*n_dirty);
    GGML_ASSERT(dirty_cells->ne[1] == n_ns);
    GGML_ASSERT(dirty_pos->ne[0]   == 4*n_dirty*n_ns);

    int32_t * dst_cells = (int32_t *) dirty_cells->data;
    int32_t * dst_pos   = (int32_t *) dirty_pos->data;
    int32_t * dst_dst   = (int32_t *) dirty_dst->data;

    const int64_t size_blk = mem_blk->get_size();

    std::fill(dst_pos, dst_pos + 4*n_dirty*n_ns, 0);

    for (int64_t s = 0; s < n_ns; ++s) {
        const auto & sp = pr.streams[s];
        const auto & cells = get_mem_idx()->get_cells(sp.seq0);

        // the graph's cache read view assumes the ubatch streams are contiguous ascending
        GGML_ASSERT(sp.stream_id == pr.streams[0].stream_id + (uint32_t) s);

        for (int64_t d = 0; d < n_dirty; ++d) {
            // pad by repeating the first dirty block (idempotent rewrite); with no dirty
            // and no full blocks at all, write slot 0 from cell 0 - the slot is not valid
            // in the book, its bias is -inf, and the cells are zero-initialized
            int32_t b = 0;

            if (d < (int64_t) sp.dirty.size()) {
                b = sp.dirty[d];
            } else if (!sp.dirty.empty()) {
                b = sp.dirty[0];
            } else if (sp.n_bid > 0) {
                b = 0;
            } else {
                dst_dst[s*n_dirty + d] = (int32_t) (sp.stream_id*size_blk);
                for (int64_t k = 0; k < r; ++k) {
                    dst_cells[s*(r*n_dirty) + d*r + k] = 0;
                }
                continue;
            }

            dst_dst[s*n_dirty + d] = (int32_t) (sp.stream_id*size_blk + b);

            memcpy(dst_cells + s*(r*n_dirty) + d*r, sp.blk_cells.data() + (size_t) b*r, r*sizeof(int32_t));

            int32_t sec_pos[4] = { sp.bid_idx[b], sp.bid_idx[b], sp.bid_idx[b], sp.bid_idx[b] };

            if (sp.ranked) {
                const int32_t   c = sp.bid_slot0[b];
                const llama_pos p = cells.pos_get(c);
                const auto &    e = cells.ext_get(c);

                sec_pos[0] = p;
                sec_pos[1] = e.y;
                sec_pos[2] = e.x;
                sec_pos[3] = p;
            }

            for (int64_t sec = 0; sec < 4; ++sec) {
                dst_pos[sec*(n_dirty*n_ns) + s*n_dirty + d] = sec_pos[sec];
            }
        }
    }
}

//
// llama_memory_hybrid_idx_context
//

// streams in each ubatch's slot info, matching get_k/get_v's `ns`
static std::vector<uint32_t> llama_memory_hybrid_idx_ns(const llama_kv_cache::slot_info_vec_t & sinfos) {
    std::vector<uint32_t> res;
    res.reserve(sinfos.size());

    for (const auto & sinfo : sinfos) {
        res.push_back(sinfo.s1 - sinfo.s0 + 1);
    }

    return res;
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_status status) :
    llama_memory_hybrid_context(status) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem) :
    llama_memory_hybrid_context(mem),
    mem(mem),
    // graph reservation walks a full context, and qwen4exp builds the sparse attention only when this is set
    // without it the reserved worst case is the dense graph, so ggml-alloc must grow the buffer on the first decode
    ns_ubatch(mem->get_mem_idx() == nullptr ?
        std::vector<uint32_t>() : std::vector<uint32_t>{ mem->get_mem_idx()->get_n_stream() }),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx())),
    is_full(true) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                  llama_context * lctx,
                           bool   optimize) :
    llama_memory_hybrid_context(mem, lctx, optimize),
    mem(mem),
    // update() applies a pending cross-stream seq_cp, else the copy keeps stale indexer keys
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        mem->get_mem_idx()->init_update(lctx, optimize)) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                slot_info_vec_t   sinfos_attn,
                slot_info_vec_t   sinfos_idx,
      std::vector<llama_ubatch>   ubatches) :
    // note: the base copies the ubatches; ctx_idx gets a copy of its own
    llama_memory_hybrid_context(mem, std::move(sinfos_attn), ubatches),
    mem(mem),
    ns_ubatch(llama_memory_hybrid_idx_ns(sinfos_idx)),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx(), std::move(sinfos_idx), ubatches)) {}

bool llama_memory_hybrid_idx_context::next() {
    if (ctx_idx) {
        ctx_idx->next();
    }

    ++i_cur;

    return llama_memory_hybrid_context::next();
}

bool llama_memory_hybrid_idx_context::apply() {
    bool res = llama_memory_hybrid_context::apply();

    if (ctx_idx) {
        res = res & ctx_idx->apply();
    }

    // cells may have changed; the next qsa_prepare must recompute
    if (mem) {
        mem->qsa_epoch++;
    }

    return res;
}

const llama_kv_cache_context * llama_memory_hybrid_idx_context::get_idx() const {
    return static_cast<const llama_kv_cache_context *>(ctx_idx.get());
}

uint32_t llama_memory_hybrid_idx_context::get_n_stream() const {
    GGML_ASSERT(i_cur < ns_ubatch.size());

    return ns_ubatch[i_cur];
}

void llama_memory_hybrid_idx_context::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        ggml_tensor * extra_cells,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias) const {
    GGML_ASSERT(mem != nullptr);
    GGML_ASSERT(get_idx() != nullptr);

    mem->set_input_qsa(cell_blk, blk_cells, blk_pos, bias, extra_cells, ubatch,
            get_idx()->get_n_kv(), ratio, blk_bias);
}

const llama_memory_hybrid_idx::qsa_prep & llama_memory_hybrid_idx_context::qsa_prepare(
        const llama_ubatch * ubatch,
        uint32_t ratio) const {
    GGML_ASSERT(mem != nullptr);
    GGML_ASSERT(get_idx() != nullptr);

    return mem->qsa_prepare(ubatch, get_idx()->get_n_kv(), get_n_stream(), ratio, is_full);
}

ggml_tensor * llama_memory_hybrid_idx_context::blk_k_storage(int32_t il) const {
    GGML_ASSERT(mem != nullptr && mem->get_mem_blk() != nullptr);

    return mem->get_mem_blk()->get_k_storage(il);
}

ggml_tensor * llama_memory_hybrid_idx_context::blk_v_storage(int32_t il) const {
    GGML_ASSERT(mem != nullptr && mem->get_mem_blk() != nullptr);

    return mem->get_mem_blk()->get_v_storage(il);
}

uint32_t llama_memory_hybrid_idx_context::blk_stream0(llama_seq_id seq) const {
    GGML_ASSERT(mem != nullptr && mem->get_mem_idx() != nullptr);

    return mem->get_mem_idx()->get_stream_of(seq);
}

bool llama_memory_hybrid_idx_context::blk_available() const {
    return mem != nullptr && mem->get_mem_blk() != nullptr;
}

void llama_memory_hybrid_idx_context::set_input_qsa_dirty(
        ggml_tensor * dirty_cells,
        ggml_tensor * dirty_pos,
        ggml_tensor * dirty_dst,
        const llama_ubatch * ubatch,
        uint32_t ratio) const {
    GGML_ASSERT(mem != nullptr);
    GGML_ASSERT(get_idx() != nullptr);

    mem->set_input_qsa_dirty(dirty_cells, dirty_pos, dirty_dst, ubatch,
            get_idx()->get_n_kv(), ratio);
}
