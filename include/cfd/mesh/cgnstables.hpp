#pragma once

#include <cstddef>
#include <cstdint>
#include <array>
#include "cfd/core/types.hpp"

namespace cfd::mesh {

enum class CellType : std::uint8_t {
    TET   = 0,
    PYRA  = 1,
    PRISM = 2,
    HEXA  = 3,
    TRI   = 4,
    QUAD  = 5,
};

// Nodes per element of each type.
constexpr int kNodesPerType[6] = {4, 5, 6, 8, 3, 4};
// Faces per volume element type.
constexpr int kFacesPerType[4] = {4, 5, 5, 6};
// Nodes per face (triangle = 3, quadrilateral = 4).
constexpr int kFaceNodes[4][6] = {
    {3, 3, 3, 3, 0, 0},  // TET
    {4, 3, 3, 3, 3, 0},  // PYRA
    {3, 4, 4, 4, 3, 0},  // PRISM
    {4, 4, 4, 4, 4, 4},  // HEXA
};

// Local nodes of each face (0-based indices into the cell node list);
// the order yields the outward normal by the right-hand rule.
constexpr int kFaceTable[4][6][4] = {
    // TET: (n1,n2,n3) base, n4 above it (det[n1-n0,n2-n0,n3-n0] > 0)
    {{1, 2, 3, -1},  // opposite n0
     {0, 3, 2, -1},  // opposite n1
     {0, 1, 3, -1},  // opposite n2
     {0, 2, 1, -1},  // opposite n3
     {-1, -1, -1, -1},
     {-1, -1, -1, -1}},

    // PYRA: base n0..n3 (CCW seen from the apex n4), apex n4
    {{0, 3, 2, 1},  // base (normal down)
     {0, 1, 4, -1},
     {1, 2, 4, -1},
     {2, 3, 4, -1},
     {3, 0, 4, -1},
     {-1, -1, -1, -1}},

    // PRISM: bottom triangle n0,n1,n2 (CCW seen from top), n3 above n0
    {{0, 2, 1, -1},  // bottom
     {0, 1, 4, 3},
     {1, 2, 5, 4},
     {2, 0, 3, 5},
     {3, 4, 5, -1},  // top
     {-1, -1, -1, -1}},

    // HEXA: bottom n0..n3 (CCW seen from top), n4 above n0
    {{0, 3, 2, 1},  // bottom
     {0, 1, 5, 4},
     {1, 2, 6, 5},
     {2, 3, 7, 6},
     {3, 0, 4, 7},
     {4, 5, 6, 7}},  // top
};

// Node permutation that flips the element orientation to the opposite one
// (used to fix a negative volume).
constexpr int kOrientationFlip[4][8] = {
    {0, 2, 1, 3, -1, -1, -1, -1},  // TET
    {0, 3, 2, 1, 4, -1, -1, -1},   // PYRA
    {0, 2, 1, 3, 5, 4, -1, -1},    // PRISM
    {0, 3, 2, 1, 4, 7, 6, 5},      // HEXA
};

// VTK cell types for VTU output.
constexpr int vtk_cell_type(CellType t) {
    switch (t) {
        case CellType::TET:
            return 10;
        case CellType::PYRA:
            return 14;
        case CellType::PRISM:
            return 13;
        case CellType::HEXA:
            return 12;
        case CellType::TRI:
            return 5;
        case CellType::QUAD:
            return 9;
    }
    return -1;
}

inline bool is_volume_type(CellType t) { return static_cast<int>(t) < 4; }

inline const char* cell_type_name(CellType t) {
    switch (t) {
        case CellType::TET:
            return "TET";
        case CellType::PYRA:
            return "PYRA";
        case CellType::PRISM:
            return "PRISM";
        case CellType::HEXA:
            return "HEXA";
        case CellType::TRI:
            return "TRI";
        case CellType::QUAD:
            return "QUAD";
    }
    return "?";
}

// Face key: sorted global node ids (4 slots, -1 = unused).
struct FaceKey {
    std::array<GlobalIndex, 4> v = {
        kInvalidGlobalIndex, 
        kInvalidGlobalIndex, 
        kInvalidGlobalIndex, 
        kInvalidGlobalIndex
    };

    constexpr bool operator==(const FaceKey& o) const noexcept {
        return v[0] == o.v[0] && v[1] == o.v[1] && v[2] == o.v[2] && v[3] == o.v[3];
    }

    constexpr bool operator<(const FaceKey& o) const noexcept {
        for (std::size_t i = 0; i < 4; ++i) {
            if (v[i] != o.v[i]) return v[i] < o.v[i];
        }
        return false;
    }
};

struct FaceKeyHash {
    [[nodiscard]] std::size_t operator()(const FaceKey& k) const noexcept {
        std::uint64_t h = 1469598103934665603ULL;
        for (std::size_t i = 0; i < 4; ++i) {
            h ^= static_cast<std::uint64_t>(k.v[i]);
            h *= 1099511628211ULL;
        }
        return static_cast<std::size_t>(h);
    }
};

} //namespace cfd::mesh