#pragma once

#include <vector>
#include <string>
#include <utility>
#include <cstdint>

#include <mpi.h>
#include "cfd/core/types.hpp"
#include "cfd/mesh/cgnstables.hpp"

namespace cfd::mesh {

using PatchId = std::int32_t;
inline constexpr PatchId kInvalidPatchId = -1;

struct FileMeta {
    float cgns_version{};
    int file_integer_precision{};
    int storage_type{};
};


struct SectionMeta {
    std::string name;
    mesh::CellType type = mesh::CellType::HEXA; // CellType for uniform sections
    bool is_mixed = false;                      // True if MIXED
    GlobalIndex start = 0;                      // Global (1-based) first element ID in CGNS
    GlobalIndex end = 0;                        // Global (1-based) last element ID in CGNS
    GlobalIndex cell_offset = 0;                // 0-based global ID offset for volume cells
    int sec_idx = 0;                            // 1-based section index in CGNS file
};


struct BCMeta {
    std::string name;                 // FamilyName or BC node name
    std::string cgns_type;            // BCType string
    std::vector<GlobalIndex> eids;    // Global boundary element IDs (1-based)
};


// Boundary element representation for topological BC matching
struct SurfElem {
    mesh::FaceKey key;
    GlobalIndex eid = 0;              // Global 1-based boundary element ID
    PatchId patch = kInvalidPatchId;  // Index into patch_list
};


// Read mesh. Cells are block-distributed by global id (gid is a dense
// renumbering of the volume sections in order); node coordinates are block-distributed.
struct RawMesh {
    int nprocs = 0;
    int rank = 0;
    MPI_Comm comm = MPI_COMM_NULL;

    GlobalIndex n_cells_g = 0;       // global number of cells
    GlobalIndex n_nodes_g = 0;       // global number of nodes

    // Displacements of size (nprocs + 1)
    std::vector<GlobalIndex> cell_displ;
    std::vector<GlobalIndex> node_displ;

    // Local cell attributes (size n_local)
    std::vector<mesh::CellType> ctype;

    // Flattened cell-to-node connectivity table (size sum(nodes_per_cell))
    std::vector<LocalIndex> cnodes_offsets;
    std::vector<GlobalIndex> cnodes;

    // Coordinates of local node slice [my_node_begin(), my_node_end()) in SoA layout
    std::vector<double> my_node_coords_x;
    std::vector<double> my_node_coords_y;
    std::vector<double> my_node_coords_z;

    // Local slice of boundary surface elements
    std::vector<SurfElem> surf_elems;

    // Replicated metadata across all ranks
    FileMeta gfm;
    std::vector<SectionMeta> vol_secs;
    std::vector<SectionMeta> surf_secs;
    std::vector<BCMeta> bcs;
    std::vector<std::pair<std::string, std::string>> patch_list;

// Helper queries
    [[nodiscard]] LocalIndex n_local_cells() const noexcept {
        return static_cast<LocalIndex>(
            cell_displ[static_cast<std::size_t>(rank) + 1] - 
            cell_displ[static_cast<std::size_t>(rank)]
        );
    }

    [[nodiscard]] LocalIndex n_local_nodes() const noexcept {
        return static_cast<LocalIndex>(
            node_displ[static_cast<std::size_t>(rank) + 1] - 
            node_displ[static_cast<std::size_t>(rank)]
        );
    }

    [[nodiscard]] GlobalIndex global_cell_id(LocalIndex local_idx) const noexcept {
        return cell_displ[static_cast<std::size_t>(rank)] + local_idx;
    }

    [[nodiscard]] GlobalIndex my_node_begin() const noexcept { 
        return node_displ[static_cast<std::size_t>(rank)]; 
    }

    [[nodiscard]] GlobalIndex my_node_end() const noexcept { 
        return node_displ[static_cast<std::size_t>(rank) + 1]; 
    }
};    

}