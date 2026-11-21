// Bind the audited local row codec for the two-format head experiment.
#include "row-format.hpp"
#include <cstddef>
#include <cstdint>

extern "C" size_t write_local(const float* words, uint8_t* frame) {
    using namespace gdn_local;
    size_t at = 5;
    bool all_positive_zero = true;
    frame[4] = head_tag;
    for (size_t row = 0; row < 128; ++row) {
        size_t count = encode_row(words + 128*row, frame + at, Policy::screened);
        if ((frame[at] & 7) != 4) all_positive_zero = false;
        at += count;
    }
    if (all_positive_zero) at = 4;
    put32(frame, static_cast<uint32_t>(at-4));
    return at;
}

extern "C" int read_local(const uint8_t* frame, size_t count, float* words) {
    if (count < 4 || gdn_local::get32(frame) != count-4) return 1;
    try { gdn_local::decode(frame+4, count-4, words); }
    catch (...) { return 2; }
    return 0;
}
