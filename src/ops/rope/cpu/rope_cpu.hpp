#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void rope(std::byte *out, std::byte *in, std::byte *pos_ids, llaisysDataType_t type, size_t seq_len, size_t nkvhead, size_t d, float theta);
}