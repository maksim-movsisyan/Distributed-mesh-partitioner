#pragma once

#include <cstdint>

#if defined(_MSC_VER)
    #define CFD_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
    #define CFD_RESTRICT __restrict__
#else
    #define CFD_RESTRICT
#endif

namespace cfd {

#if defined(CFD_INDEX_64BIT)
using GlobalIndex = std::int64_t;
#else
using GlobalIndex = std::int32_t;
#endif

// Local entity IDs within an MPI rank / partition (up to 2 billion elements)
using LocalIndex = std::int32_t;

// Canonical element topology indices (0..7)
using ElementLocalNode = std::int8_t;

inline constexpr GlobalIndex kInvalidGlobalIndex = -1;
inline constexpr LocalIndex  kInvalidLocalIndex  = -1;


} // namespace cfd