#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

// Speculative decoding overrides read from $NEXT_OPT_DIR at most once per second.
inline bool next_opt_read_double(const char * name, double & out) {
    const char * dir = getenv("NEXT_OPT_DIR");
    if (!dir) {
        return false;
    }
    FILE * f = fopen((std::string(dir) + "/" + name).c_str(), "r");
    if (!f) {
        return false;
    }
    double v = 0.0;
    const bool ok = fscanf(f, "%lf", &v) == 1;
    fclose(f);
    if (ok) {
        out = v;
    }
    return ok;
}

inline void next_spec_overrides(int32_t n_max_cfg, float p_min_cfg, int32_t & n_max, float & p_min) {
    static std::chrono::steady_clock::time_point next_check{};
    static double n_max_file = -1.0;
    static double p_min_file = -1.0;
    const auto now = std::chrono::steady_clock::now();
    if (now >= next_check) {
        next_check = now + std::chrono::seconds(1);
        double v;
        n_max_file = next_opt_read_double("spec_n_max", v) ? v : -1.0;
        p_min_file = next_opt_read_double("spec_p_min", v) ? v : -1.0;
    }
    n_max = n_max_cfg;
    p_min = p_min_cfg;
    if (n_max_file >= 1.0) {
        n_max = std::clamp<int32_t>((int32_t) n_max_file, 1, n_max_cfg);
    }
    if (p_min_file >= 0.0) {
        p_min = (float) std::min(p_min_file, 1.0);
    }
}

// Acceptance-adaptive draft length: with n_max=4 the verify batch costs ~4 ms more than n=2 per step, which only
// pays off when most drafts are accepted (English reasoning/code ~0.75) and loses on Chinese prose (~0.5).
// Disabled by an existing file $NEXT_OPT_DIR/spec_adaptive_off.
inline bool next_spec_adaptive() {
    static std::chrono::steady_clock::time_point next_check{};
    static bool enabled = true;
    const auto now = std::chrono::steady_clock::now();
    if (now >= next_check) {
        next_check = now + std::chrono::seconds(1);
        const char * dir = getenv("NEXT_OPT_DIR");
        enabled = true;
        if (dir) {
            FILE * f = fopen((std::string(dir) + "/spec_adaptive_off").c_str(), "r");
            if (f) { enabled = false; fclose(f); }
        }
    }
    return enabled;
}

inline int32_t next_spec_adapt_n(int32_t n_max, float acc_rate) {
    if (acc_rate >= 0.70f) return n_max;
    if (acc_rate >= 0.50f) return std::max<int32_t>(2, n_max - 1);
    return std::max<int32_t>(2, n_max - 2);
}
