// Final rank-local mesh: owned cells plus a one-hop ghost layer (face and
// vertex-sharing neighbours), locality-oriented renumbering, faces and the
// communication maps. This is exactly the structure the solver loads from file.
#pragma once

#include <string>
#include <vector>

#include "cfd/core/types.hpp"
#include "cfd/mesh/raw_mesh.hpp"
#include "cfd/mesh/faces.hpp"
#include "cfd/partition/partition.hpp"
#include "cfd/mesh/cgnstables.hpp"

namespace cfd::mesh {

struct MeshPart {
    int rank = 0;
    int nprocs = 0;

    // --- Global metadata (replicated on all ranks) ---
    GlobalIndex n_cells_g = 0;
    GlobalIndex n_nodes_g = 0;
    GlobalIndex n_faces_g = 0;
    GlobalIndex n_bfaces_g = 0;
    double bbox_lo[3] = {0, 0, 0};
    double bbox_hi[3] = {0, 0, 0};

    // --- Cells ---
    // [0, n_own) are owned cells (SFC or RCM ordered for cache locality).
    // [n_own, n_cells) are ghost cells received from neighbours.
    LocalIndex n_own = 0;
    LocalIndex n_cells = 0;
    
    std::vector<CellType> cell_type;             //                                         << size = n_cells
    std::vector<GlobalIndex> cell_gid;           // Global ID (strictly for VTK/output)     << size = n_cells
    std::vector<int> cell_donor;                 // -1 if owned; otherwise donor MPI rank   << size = n_cells
    std::vector<LocalIndex> cell_nodes_offsets;  // CSR offsets for cell-node connectivity  << size = n_cells + 1
    std::vector<LocalIndex> cell_nodes;          // Local node indices                      << size = sum_cells_(number of cell nodes)

    std::vector<double> cell_centroid_x;         //                                         << size = n_cells 
    std::vector<double> cell_centroid_y;         //                                         << size = n_cells 
    std::vector<double> cell_centroid_z;         //                                         << size = n_cells 
    std::vector<double> cell_volume;             //                                         << size = n_cells 

    // --- Nodes ---
    // [0, n_nodes_own) are nodes used by at least one owned cell.
    // [n_nodes_own, n_nodes) are nodes used EXCLUSIVELY by ghost cells.
    LocalIndex n_nodes_own = 0;
    LocalIndex n_nodes = 0;
    
    std::vector<GlobalIndex> node_gid;           // Global ID                               << size = n_nodes
    std::vector<double> node_x;                  //                                         << size = n_nodes
    std::vector<double> node_y;                  //                                         << size = n_nodes
    std::vector<double> node_z;                  //                                         << size = n_nodes

    // --- Faces ---
    // Strict orientation constraint: normal always points from `face_owner` to `face_neigh`.
    LocalIndex n_faces = 0;                      // Total number of faces
    LocalIndex n_internal_faces = 0;             // Total number of purley internal faces (that touch only internal cells)
    LocalIndex n_inner_faces = 0;                // Total number of internal faces (that may touch ghost cells)

    std::vector<LocalIndex> face_owner;          // Local index of the owned cell                                           << size = n_faces
    std::vector<LocalIndex> face_neigh;          // Local index (owned or ghost cell) or kInvalidLocalIndex if boundary     << size = n_faces
    std::vector<CellType> face_type;             // Face topology (TRI or QUAD)                                             << size = n_faces
    std::vector<LocalIndex> face_nodes_offsets;  // CSR offsets                                                             << size = n_faces + 1
    std::vector<LocalIndex> face_nodes;          // Local node indices in outward-pointing order                            << size = sum_faces_(number of face nodes)

    std::vector<double> face_centroid_x;         //                                                                         << size = n_face
    std::vector<double> face_centroid_y;         
    std::vector<double> face_centroid_z;         
    std::vector<double> face_normal_x;           // Outward normal (direction from owner to neighbor)                       << size = n_face
    std::vector<double> face_normal_y;
    std::vector<double> face_normal_z;
    std::vector<double> face_area;               //                                                                         << size = n_face
    std::vector<PatchId> face_patch;             // BC patch id or kInvalidPatchId for interior                             << size = n_face

    // --- Communication maps (Neighbours sorted by rank) ---
    std::vector<int> nb_ranks;                   // MPI ranks of neighbours
    std::vector<LocalIndex> recv_offsets;        // Size: n_neighbors + 1, offsets into recv_ghost_local
    std::vector<LocalIndex> recv_ghost_local;    // My ghost cell indices grouped by sending neighbour
    std::vector<LocalIndex> send_offsets;        // Size: n_neighbors + 1, offsets into send_owned_local
    std::vector<LocalIndex> send_owned_local;    // My owned cell indices needed by neighbours (to be sent)

    // --- BC patches (Replicated global list) ---
    struct Patch {
        std::string name;
        std::string cgns_type;
    };

    // NOT USED DUE TO FACE SOTRING SO BOUNDARY FACES ARE STRORED IN CONTIGUOUS RANGES
    std::vector<Patch> patches;
    std::vector<LocalIndex> patch_face_offsets;  // Size: n_patches + 1
    std::vector<LocalIndex> patch_faces;         // Local face indices mapped to patches

    // --- Preprocessing statistics ---
    long long n_flipped = 0;  // Cells with a fixed orientation (negative volume corrected)

    int n_neighbors() const noexcept { 
        return static_cast<int>(nb_ranks.size()); 
    }
};

// Structure sanity (invariants); used by tests.
bool meshpart_sane(const MeshPart& mp, std::string& err);

// Data Migration & Topology
// Consumes `RawMesh` and `faces` destructively (via std::move) to prevent 2x memory peak.
// Performs migration to final ranks, establishes the ghost layer, and renumbers locally.
void migrate_local_mesh(RawMesh&& m, std::vector<FaceRec>&& faces, const partition::PartitionResult& pr, MeshPart& mp);
} // namespace cfd::mesh