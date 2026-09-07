#pragma once

#include <cstdint>
#include <vector>
#include <span>
#include <cassert>

#include "cfd/core/types.hpp"
#include "cfd/mesh/localmesh.hpp"

namespace cfd::mesh {

enum class AuxConnType : uint32_t {
    None             = 0,
    CellFaces        = 1 << 0, // cell -> faces
    CellCellsByFace  = 1 << 1, // cell -> cell neighbors sharing a face
    NodeCells        = 1 << 2, // node -> cells sharing this node
    NodeFaces        = 1 << 3, // node -> faces sharing this node
    CellCellsByNode  = 1 << 4  // cell -> cell neighbors sharing at least one node (stencil for LSQ, ram-heavy)
};

inline AuxConnType operator|(AuxConnType a, AuxConnType b) noexcept {
    return static_cast<AuxConnType>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

inline AuxConnType operator&(AuxConnType a, AuxConnType b) noexcept {
    return static_cast<AuxConnType>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}

inline bool has_flag(AuxConnType mask, AuxConnType flag) noexcept {
    return (static_cast<uint32_t>(mask) & static_cast<uint32_t>(flag)) != 0;
}

struct MeshAuxConnectivity {
    AuxConnType active_mask = AuxConnType::None;

    // --- Sizes (owned local capacity: owned) ---
    LocalIndex n_faces = 0;
    LocalIndex n_cells_own = 0;
    LocalIndex n_nodes_own = 0;


    // --- Cell -> Faces (CSR) ---
    std::vector<LocalIndex> cell_faces_offsets;         // size = n_cells_own + 1
    std::vector<LocalIndex> cell_faces;                 // size = sum_cells_(faces per cell)

    // --- Cell -> Cells through faces (CSR) ---
    std::vector<LocalIndex> cell_cells_face_offsets;    // size = n_cells_own + 1
    std::vector<LocalIndex> cell_cells_face;            // size = sum_cells_(neighbors per cell through faces) << (ghost cells included (!!))

    // --- Cell -> Cells through nodes (CSR) ---
    std::vector<LocalIndex> cell_cells_node_offsets;    // size = n_cells_own + 1
    std::vector<LocalIndex> cell_cells_node;            // size = sum_cells_(neighbors per cell through nodes) << (ghost cells included (!!))

    // --- Node -> Cells (CSR) ---
    std::vector<LocalIndex> node_cells_offsets;         // size = n_nodes_own + 1
    std::vector<LocalIndex> node_cells;                 // size = sum_nodes_(cells per node)                   << (ghost cells included (!!))

    // --- Node -> Faces (CSR) ---
    std::vector<LocalIndex> node_faces_offsets;         // size = n_nodes_own + 1
    std::vector<LocalIndex> node_faces;                 // size = sum_nodes_(faces per node)

    // --- Add new connectivity ---
    void add_connectivity(const MeshPart& mp, AuxConnType requested);

    // --- Status Checks ---
    [[nodiscard]] bool has(AuxConnType flag) const noexcept { return has_flag(active_mask, flag); }
    [[nodiscard]] bool has_cell_faces() const noexcept { return !cell_faces_offsets.empty(); }
    [[nodiscard]] bool has_cell_cells_face() const noexcept { return !cell_cells_face_offsets.empty(); }
    [[nodiscard]] bool has_cell_cells_node() const noexcept { return !cell_cells_node_offsets.empty(); }
    [[nodiscard]] bool has_node_cells() const noexcept { return !node_cells_offsets.empty(); }
    [[nodiscard]] bool has_node_faces() const noexcept { return !node_faces_offsets.empty(); }

    // --- Fast Span Views (Zero-overhead range access) ---
    [[nodiscard]] std::span<const LocalIndex> cell_faces_of(LocalIndex c) const noexcept {
        assert(c >= 0 && c < n_cells_own && "cell_faces queried for invalid or ghost cell!");
        return {cell_faces.data() + cell_faces_offsets[static_cast<std::size_t>(c)], 
                static_cast<std::size_t>(cell_faces_offsets[static_cast<std::size_t>(c) + 1] - cell_faces_offsets[static_cast<std::size_t>(c)])};
    }

    [[nodiscard]] std::span<const LocalIndex> cell_neighbors_face_of(LocalIndex c) const noexcept {
        assert(c >= 0 && c < n_cells_own && "cell_cells_face queried for invalid or ghost cell!");
        return {cell_cells_face.data() + cell_cells_face_offsets[static_cast<std::size_t>(c)], 
                static_cast<std::size_t>(cell_cells_face_offsets[static_cast<std::size_t>(c) + 1] - cell_cells_face_offsets[static_cast<std::size_t>(c)])};
    }

    [[nodiscard]] std::span<const LocalIndex> cell_neighbors_node_of(LocalIndex c) const noexcept {
        assert(c >= 0 && c < n_cells_own && "cell_cells_node queried for invalid or ghost cell!");
        return {cell_cells_node.data() + cell_cells_node_offsets[static_cast<std::size_t>(c)], 
                static_cast<std::size_t>(cell_cells_node_offsets[static_cast<std::size_t>(c) + 1] - cell_cells_node_offsets[static_cast<std::size_t>(c)])};
    }

    [[nodiscard]] std::span<const LocalIndex> node_cells_of(LocalIndex n) const noexcept {
        assert(n >= 0 && n < n_nodes_own && "node_cells queried for invalid or ghost node!");
        return {node_cells.data() + node_cells_offsets[static_cast<std::size_t>(n)], 
                static_cast<std::size_t>(node_cells_offsets[static_cast<std::size_t>(n) + 1] - node_cells_offsets[static_cast<std::size_t>(n)])};
    }

    [[nodiscard]] std::span<const LocalIndex> node_faces_of(LocalIndex n) const noexcept {
        assert(n >= 0 && n < n_nodes_own && "node_faces queried for invalid or ghost node!");
        return {node_faces.data() + node_faces_offsets[static_cast<std::size_t>(n)], 
                static_cast<std::size_t>(node_faces_offsets[static_cast<std::size_t>(n) + 1] - node_faces_offsets[static_cast<std::size_t>(n)])};
    }

    // --- Cleanup ---
    void clear() noexcept {
        active_mask = AuxConnType::None;
        n_cells_own = 0;
        n_faces = 0;
        n_nodes_own = 0;

        auto free_vec = [](auto& v) {
            v.clear();
            v.shrink_to_fit();
        };

        free_vec(cell_faces_offsets);
        free_vec(cell_faces);
        free_vec(cell_cells_face_offsets);
        free_vec(cell_cells_face);
        free_vec(cell_cells_node_offsets);
        free_vec(cell_cells_node);
        free_vec(node_cells_offsets);
        free_vec(node_cells);
        free_vec(node_faces_offsets);
        free_vec(node_faces);
    }
};

MeshAuxConnectivity build_aux_connectivity(const MeshPart& mp, AuxConnType mask);

void build_cell_faces_conn(const MeshPart& mp, std::vector<LocalIndex>& cell_faces_offsets, std::vector<LocalIndex>& cell_faces);
void build_cell_cells_face_conn(const MeshPart& mp, std::vector<LocalIndex>& cell_cells_face_offsets, std::vector<LocalIndex>& cell_cells_face);
void build_cell_cells_node_conn(const MeshPart& mp, std::vector<LocalIndex>& cell_cells_node_offsets, std::vector<LocalIndex>& cell_cells_node);
void build_cell_cells_node_conn(const MeshPart& mp,
                                const std::vector<LocalIndex>& node_cells_offsets, const std::vector<LocalIndex>& node_cells,
                                std::vector<LocalIndex>& cell_cells_node_offsets, std::vector<LocalIndex>& cell_cells_node);

void build_node_faces_conn(const MeshPart& mp, std::vector<LocalIndex>& node_faces_offsets, std::vector<LocalIndex>& node_faces);
void build_node_cells_conn(const MeshPart& mp, std::vector<LocalIndex>& node_cells_offsets, std::vector<LocalIndex>& node_cells);

} //namespace cfd::mesh