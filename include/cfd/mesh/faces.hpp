// Distributed construction of face-cell connectivity from cell-node connectivity.
//
// Algorithm (no replicated global arrays anywhere):
//  1. every rank generates the faces of its own cells from the canonical tables;
//  2. records are sent to the owner of the face's minimal node (block
//     distribution of nodes) and deduplicated in a hash table keyed by the sorted
//     node set; a face with two cells is interior, with one it is a boundary face;
//  3. each face is sent to the owner of its smaller cell (block distribution
//     of cells); the dual graph (CSR) for the partitioner is assembled there too.
#pragma once

#include <vector>
#include <cstdint>

#include "cfd/core/types.hpp"
#include "cfd/mesh/cgnstables.hpp"
#include "cfd/mesh/raw_mesh.hpp"

namespace cfd::mesh {

// Face record stored on the owning rank of `cell_a` (smaller cell GID).
struct FaceRec {
    FaceKey key;                     // Sorted global node IDs (canonical)
    GlobalIndex cell_a = -1;         // Smaller owner cell GID (local to this rank's block)
    GlobalIndex cell_b = -1;         // Neighbour cell GID (-1 if boundary face)
    PatchId patch = kInvalidPatchId; // BC Patch ID (if boundary face, else kInvalidPatchId)
    std::uint8_t lface_a = 0;        // Local face index in cell_a [0..5]
    std::uint8_t lface_b = 0;        // Local face index in cell_b [0..5] (255 if boundary)
};

// Dual graph in CSR form: this rank's cells adjacency by global GID.
// Compatible with ParMETIS / PT-Scotch / KaHIP.
struct DualGraph {
    std::vector<LocalIndex> offsets;    // Offset array of size (n_local_cells + 1)
    std::vector<GlobalIndex> adj;       // Neighbour global GIDs (symmetric interior adjacency)
};

struct FaceStats {
    GlobalIndex n_faces_g = 0;   // total faces
    GlobalIndex n_bfaces_g = 0;  // boundary faces
    GlobalIndex n_ifaces_g = 0;  // interior faces
    GlobalIndex n_dg_edges = 0;  // number of edges in adjecency graph
};

struct BuildFacesResult {
    std::vector<FaceRec> faces;
    DualGraph graph;
    FaceStats stats;
};

// Distributed construction of face-cell connectivity and dual graph CSR.
// Algorithmic steps:
//  1. Generate local half-faces and dispatch to rendezvous ranks via FaceKeyHash % nprocs.
//  2. Deduplicate on rendezvous ranks, match interior pairs and match boundary faces
//     against RawMesh::surf_elems.
//  3. Dispatch matched faces to the rank owning cell_a; assemble DualGraph and FaceRec list.
BuildFacesResult build_faces(const RawMesh& m);

} //namespace cfd::mesh