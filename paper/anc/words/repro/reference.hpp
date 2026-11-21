#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace reference {
constexpr int64_t limit = 0x7fffffff;
struct Solution { uint32_t base; int width_sum; };

// Each threshold with a nonempty feasible interval contributes one unit
// of coverage. The upper endpoint is inclusive; use 64 bits for right+1.
inline Solution solve(const uint32_t* words) {
    std::vector<std::pair<int64_t, int>> events;
    events.reserve(2 * 16 * 32);
    auto interval = [&](int64_t left, int64_t right) {
        left = std::max<int64_t>(0, left);
        right = std::min(limit, right);
        if (left <= right) {
            events.emplace_back(left, 1);
            events.emplace_back(right + 1, -1);
        }
    };
    for (int g = 0; g < 16; ++g) {
        int64_t lo = limit, hi = 0;
        for (int j = 0; j < 8; ++j) {
            int64_t a = words[8*g+j] & uint32_t(limit);
            lo = std::min(lo, a);
            hi = std::max(hi, a);
        }
        if (lo == hi) interval(lo, lo);
        for (int t = 1; t < 32; ++t) {
            int64_t half = int64_t(1) << (t-1);
            interval(hi - half + 1, lo + half);
        }
    }
    std::sort(events.begin(), events.end());
    int depth = 0, best = 0;
    uint32_t base = 0;
    for (size_t i = 0; i < events.size();) {
        int64_t point = events[i].first;
        do { depth += events[i++].second; }
        while (i < events.size() && events[i].first == point);
        if (point <= limit && depth > best) {
            best = depth;
            base = static_cast<uint32_t>(point);
        }
    }
    return {base, 16*32 - best};
}
}
