# cell_pos uploads: seq_rm (the per-step draft roll-back) marks a per-stream cell range instead of forcing a full
# [kv_size x n_stream] re-upload on every device (3 MB x 3 devices + the draft's per step with 3 slots = ~10 ms/step)
S = '/home/douya/src/llama.cpp-flashnext-20260913/'
def rd(p): return open(S + p).read()
def wr(p, s): open(S + p, 'w').write(s)
def rep(s, old, new, tag):
    assert s.count(old) == 1, (tag, old[:80]); return s.replace(old, new, 1)
h = rd('src/llama-kv-cache.h')
h = h if 'cell_pos_touch' in h else rep(h, '    mutable bool cell_pos_dirty = true;\n',
        '    mutable bool cell_pos_dirty = true;                                       // full re-upload pending\n    mutable std::vector<std::pair<uint32_t, uint32_t>> cell_pos_dirty_rng;    // per stream: [lo, hi] cells changed since the last upload (lo > hi: none)\n    void cell_pos_touch(uint32_t strm, uint32_t i) const;\n', 'h')
wr('src/llama-kv-cache.h', h)
c = rd('src/llama-kv-cache.cpp')
# seq_rm: range marks instead of the blanket flag
c = rep(c, '''bool llama_kv_cache::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    cell_pos_dirty = true;
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return true;
    }''', '''bool llama_kv_cache::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        cell_pos_dirty = true;
        return true;
    }''', 'seq_rm head')
c = rep(c, '''            if (cells.seq_has(i, seq_id) && cells.seq_rm(i, seq_id)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }''', '''            if (cells.seq_has(i, seq_id) && cells.seq_rm(i, seq_id)) {
                cell_pos_touch(seq_to_stream[seq_id], i); // NEXT: the device position table forgets this cell
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }''', 'seq_rm loop')
# seq_rm tail (the "match any sequence" branch) + the touch helper, applied inside the seq_rm function only
i0 = c.index('bool llama_kv_cache::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {')
i1 = c.index('\n}\n', i0) + 3
fn = c[i0:i1]
old_rm = '                cells.rm(i);\n'
assert fn.count(old_rm) == 1, 'seq_rm any-branch'
fn = fn.replace(old_rm, '                cells.rm(i);\n                cell_pos_touch(s, i); // NEXT\n', 1)
touch = '''
void llama_kv_cache::cell_pos_touch(uint32_t strm, uint32_t i) const {
    if (cell_pos_dirty || cell_pos_dev.empty()) {
        return; // a full upload is pending anyway
    }
    if (cell_pos_dirty_rng.size() < n_stream) {
        cell_pos_dirty_rng.assign(n_stream, {UINT32_MAX, 0});
    }
    auto & r = cell_pos_dirty_rng[strm];
    r.first  = std::min(r.first, i);
    r.second = std::max(r.second, i);
}
'''
c = c[:i0] + fn + touch + c[i1:]
# upload: ranges when no full upload is pending
c = rep(c, '''void llama_kv_cache::upload_cell_pos() const {
    if (!cell_pos_dirty) {
        return;
    }
    const uint32_t kv_size = v_cells[0].size();''', '''void llama_kv_cache::upload_cell_pos() const {
    const uint32_t kv_size = v_cells[0].size();
    if (!cell_pos_dirty) {
        // partial: only the cell ranges that seq_rm emptied since the last upload
        if (cell_pos_dirty_rng.size() < n_stream) {
            return;
        }
        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & r = cell_pos_dirty_rng[s];
            if (r.first > r.second) {
                continue;
            }
            const uint32_t lo = r.first, hi = std::min<uint32_t>(r.second, kv_size - 1);
            std::vector<int32_t> host(hi - lo + 1);
            const auto & cells = v_cells[s];
            for (uint32_t i = lo; i <= hi; ++i) {
                host[i - lo] = cells.is_empty(i) ? -1 : cells.pos_get(i);
            }
            for (const auto & e : cell_pos_dev) {
                if (e.second->data) {
                    ggml_backend_tensor_set(e.second, host.data(), ((size_t) s * kv_size + lo) * sizeof(int32_t), host.size() * sizeof(int32_t));
                }
            }
            r = {UINT32_MAX, 0};
        }
        return;
    }''', 'upload head')
c = rep(c, '''    cell_pos_dirty = false;
}''', '''    cell_pos_dirty = false;
    cell_pos_dirty_rng.assign(n_stream, {UINT32_MAX, 0});
}''', 'upload tail')
wr('src/llama-kv-cache.cpp', c)
print('cell_pos incremental patch applied')
