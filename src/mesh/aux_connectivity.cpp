#include "cfd/mesh/aux_connectivity.hpp"

#include <cstddef>
#include <vector>
#include <algorithm>

#include "cfd/core/types.hpp"

namespace cfd::mesh {

void build_cell_faces_conn(const MeshPart& mp, 
                           std::vector<LocalIndex>& cell_faces_offsets, 
                           std::vector<LocalIndex>& cell_faces) {
    const std::size_t n_own   = static_cast<std::size_t>(mp.n_own);
    const std::size_t n_faces = static_cast<std::size_t>(mp.n_faces);

    if (n_own == 0) {
        cell_faces_offsets.assign(1, 0);
        cell_faces.clear();
        return;
    }

    // allocate offsets memory
    cell_faces_offsets.assign(n_own + 1, 0);

    const LocalIndex* CFD_RESTRICT owner_ptr = mp.face_owner.data();
    const LocalIndex* CFD_RESTRICT neigh_ptr = mp.face_neigh.data();
    LocalIndex* CFD_RESTRICT offsets_ptr     = cell_faces_offsets.data();

    // -------------------------------------------------------------------------
    // Pass 1: Count incident faces for each owned cell
    // -------------------------------------------------------------------------
    for (std::size_t f = 0; f < n_faces; ++f) {
        const LocalIndex c_own = owner_ptr[f];
        if (c_own >= 0 && c_own < mp.n_own) {
            ++offsets_ptr[static_cast<std::size_t>(c_own)];
        }

        const LocalIndex c_neigh = neigh_ptr[f];
        if (c_neigh >= 0 && c_neigh < mp.n_own) {
            ++offsets_ptr[static_cast<std::size_t>(c_neigh)];
        }
    }

    // -------------------------------------------------------------------------
    // Prefix Sum: Convert counters into starting write positions (cursors)
    // -------------------------------------------------------------------------
    LocalIndex total_cell_faces = 0;
    for (std::size_t c = 0; c < n_own; ++c) {
        const LocalIndex count = offsets_ptr[c];
        offsets_ptr[c] = total_cell_faces;
        total_cell_faces += count;
    }
    offsets_ptr[n_own] = total_cell_faces;


    cell_faces.resize(static_cast<std::size_t>(total_cell_faces));
    LocalIndex* CFD_RESTRICT faces_ptr = cell_faces.data();

    // -------------------------------------------------------------------------
    // Pass 2: Populate cell_faces.
    // offsets_ptr[c] is incremented upon each insertion and, at the end of the step,
    // will point to the end of cell c's range (which is the start of cell c + 1).
    // -------------------------------------------------------------------------
    for (std::size_t f = 0; f < n_faces; ++f) {
        const LocalIndex f_idx = static_cast<LocalIndex>(f);

        const LocalIndex c_own = owner_ptr[f];
        if (c_own >= 0 && c_own < mp.n_own) {
            faces_ptr[offsets_ptr[static_cast<std::size_t>(c_own)]++] = f_idx;
        }

        const LocalIndex c_neigh = neigh_ptr[f];
        if (c_neigh >= 0 && c_neigh < mp.n_own) {
            faces_ptr[offsets_ptr[static_cast<std::size_t>(c_neigh)]++] = f_idx;
        }
    }

    // -------------------------------------------------------------------------
    // CSR Restoration: Shift the offsets array to the right by 1 position
    // -------------------------------------------------------------------------
    for (std::size_t c = n_own; c > 0; --c) {
        offsets_ptr[c] = offsets_ptr[c - 1];
    }
    offsets_ptr[0] = 0;
}

void build_cell_cells_face_conn(const MeshPart& mp, 
                                std::vector<LocalIndex>& cell_cells_face_offsets, 
                                std::vector<LocalIndex>& cell_cells_face) {
    const std::size_t n_own   = static_cast<std::size_t>(mp.n_own);
    const std::size_t n_faces = static_cast<std::size_t>(mp.n_faces);

    // 1. Fast path for empty subdomain
    if (n_own == 0) {
        cell_cells_face_offsets.assign(1, 0);
        cell_cells_face.clear();
        return;
    }

    // 2. Pre-allocate and zero-out offsets
    cell_cells_face_offsets.assign(n_own + 1, 0);

    const LocalIndex* CFD_RESTRICT owner_ptr   = mp.face_owner.data();
    const LocalIndex* CFD_RESTRICT neigh_ptr   = mp.face_neigh.data();
    LocalIndex* CFD_RESTRICT offsets_ptr       = cell_cells_face_offsets.data();

    // -------------------------------------------------------------------------
    // Pass 1: Count valid adjacent cell neighbors for each owned cell.
    // Boundary faces (face_neigh < 0) have no neighbor cell and are skipped.
    // Ghost neighbors (face_neigh >= n_own) are counted for the owned cell.
    // -------------------------------------------------------------------------
    for (std::size_t f = 0; f < n_faces; ++f) {
        const LocalIndex c_own   = owner_ptr[f];
        const LocalIndex c_neigh = neigh_ptr[f];

        // Skip boundary faces (no adjacent cell)
        if (c_neigh < 0) {
            continue;
        }

        // Increment neighbor count for owner cell
        if (c_own >= 0 && c_own < mp.n_own) {
            ++offsets_ptr[static_cast<std::size_t>(c_own)];
        }

        // Increment neighbor count for neighbor cell (only if owned locally)
        if (c_neigh < mp.n_own) {
            ++offsets_ptr[static_cast<std::size_t>(c_neigh)];
        }
    }

    // -------------------------------------------------------------------------
    // Prefix Sum: Convert counts into in-place insertion cursors
    // -------------------------------------------------------------------------
    LocalIndex total_neighbors = 0;
    for (std::size_t c = 0; c < n_own; ++c) {
        const LocalIndex count = offsets_ptr[c];
        offsets_ptr[c] = total_neighbors;
        total_neighbors += count;
    }
    offsets_ptr[n_own] = total_neighbors;

    // Allocate exact flat storage for cell neighbors
    cell_cells_face.resize(static_cast<std::size_t>(total_neighbors));
    LocalIndex* CFD_RESTRICT cells_ptr = cell_cells_face.data();

    // -------------------------------------------------------------------------
    // Pass 2: Populate neighbor indices.
    // offsets_ptr[c] is incremented on write and acts as an in-place cursor.
    // -------------------------------------------------------------------------
    for (std::size_t f = 0; f < n_faces; ++f) {
        const LocalIndex c_own   = owner_ptr[f];
        const LocalIndex c_neigh = neigh_ptr[f];

        if (c_neigh < 0) {
            continue;
        }

        if (c_own >= 0 && c_own < mp.n_own) {
            cells_ptr[offsets_ptr[static_cast<std::size_t>(c_own)]++] = c_neigh;
        }

        if (c_neigh < mp.n_own) {
            cells_ptr[offsets_ptr[static_cast<std::size_t>(c_neigh)]++] = c_own;
        }
    }

    // -------------------------------------------------------------------------
    // Restore CSR offsets: Shift offsets array right by 1 element
    // -------------------------------------------------------------------------
    for (std::size_t c = n_own; c > 0; --c) {
        offsets_ptr[c] = offsets_ptr[c - 1];
    }
    offsets_ptr[0] = 0;

    // -------------------------------------------------------------------------
    // Monotonic Sorting: Sort each cell's neighbor list in ascending order.
    // Ensures monotonic cache-friendly memory access during field reconstructions,
    // gradient calculations (Green-Gauss / LSQ), and limiter stencils.
    // -------------------------------------------------------------------------
    for (std::size_t c = 0; c < n_own; ++c) {
        const std::size_t start = static_cast<std::size_t>(offsets_ptr[c]);
        const std::size_t end   = static_cast<std::size_t>(offsets_ptr[c + 1]);
        std::sort(cells_ptr + start, cells_ptr + end);
    }
}

void build_node_cells_conn(const MeshPart& mp, 
                           std::vector<LocalIndex>& node_cells_offsets, 
                           std::vector<LocalIndex>& node_cells) {
    const std::size_t n_nodes_own = static_cast<std::size_t>(mp.n_nodes_own);
    const std::size_t n_cells     = static_cast<std::size_t>(mp.n_cells);

    // 1. Fast path for empty subdomain
    if (n_nodes_own == 0 || n_cells == 0) {
        node_cells_offsets.assign(1, 0);
        node_cells.clear();
        return;
    }

    // 2. Pre-allocate and zero-out offsets for owned nodes
    node_cells_offsets.assign(n_nodes_own + 1, 0);

    const LocalIndex* CFD_RESTRICT cell_nodes_off = mp.cell_nodes_offsets.data();
    const LocalIndex* CFD_RESTRICT cell_nodes     = mp.cell_nodes.data();
    LocalIndex* CFD_RESTRICT offsets_ptr          = node_cells_offsets.data();

    // -------------------------------------------------------------------------
    // Pass 1: Count incident cells for each owned node.
    // We loop over ALL cells [0, n_cells) so owned boundary nodes register both
    // owned and ghost cells sharing the vertex.
    // -------------------------------------------------------------------------
    for (std::size_t c = 0; c < n_cells; ++c) {
        const std::size_t k_start = static_cast<std::size_t>(cell_nodes_off[c]);
        const std::size_t k_end   = static_cast<std::size_t>(cell_nodes_off[c + 1]);

        for (std::size_t k = k_start; k < k_end; ++k) {
            const LocalIndex nid = cell_nodes[k];
            if (nid >= 0 && nid < mp.n_nodes_own) {
                ++offsets_ptr[static_cast<std::size_t>(nid)];
            }
        }
    }

    // -------------------------------------------------------------------------
    // Prefix Sum: Convert counts into in-place insertion cursors
    // -------------------------------------------------------------------------
    LocalIndex total_node_cells = 0;
    for (std::size_t n = 0; n < n_nodes_own; ++n) {
        const LocalIndex count = offsets_ptr[n];
        offsets_ptr[n] = total_node_cells;
        total_node_cells += count;
    }
    offsets_ptr[n_nodes_own] = total_node_cells;

    // Allocate flat storage for incident cells
    node_cells.resize(static_cast<std::size_t>(total_node_cells));
    LocalIndex* CFD_RESTRICT cells_ptr = node_cells.data();

    // -------------------------------------------------------------------------
    // Pass 2: Populate node -> cells connectivity.
    // Because c increases monotonically from 0 to n_cells - 1, cell indices
    // for every node are inherently written in strictly ascending order.
    // -------------------------------------------------------------------------
    for (std::size_t c = 0; c < n_cells; ++c) {
        const LocalIndex c_idx    = static_cast<LocalIndex>(c);
        const std::size_t k_start = static_cast<std::size_t>(cell_nodes_off[c]);
        const std::size_t k_end   = static_cast<std::size_t>(cell_nodes_off[c + 1]);

        for (std::size_t k = k_start; k < k_end; ++k) {
            const LocalIndex nid = cell_nodes[k];
            if (nid >= 0 && nid < mp.n_nodes_own) {
                cells_ptr[offsets_ptr[static_cast<std::size_t>(nid)]++] = c_idx;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Restore CSR offsets: Shift offsets array right by 1 element
    // -------------------------------------------------------------------------
    for (std::size_t n = n_nodes_own; n > 0; --n) {
        offsets_ptr[n] = offsets_ptr[n - 1];
    }
    offsets_ptr[0] = 0;
}

void build_node_faces_conn(const MeshPart& mp, 
                           std::vector<LocalIndex>& node_faces_offsets, 
                           std::vector<LocalIndex>& node_faces) {
    const std::size_t n_nodes_own = static_cast<std::size_t>(mp.n_nodes_own);
    const std::size_t n_faces     = static_cast<std::size_t>(mp.n_faces);

    // 1. Fast path for empty subdomain
    if (n_nodes_own == 0 || n_faces == 0) {
        node_faces_offsets.assign(1, 0);
        node_faces.clear();
        return;
    }

    // 2. Pre-allocate and zero-out offsets for owned nodes
    node_faces_offsets.assign(n_nodes_own + 1, 0);

    const LocalIndex* CFD_RESTRICT face_nodes_off = mp.face_nodes_offsets.data();
    const LocalIndex* CFD_RESTRICT face_nodes     = mp.face_nodes.data();
    LocalIndex* CFD_RESTRICT offsets_ptr          = node_faces_offsets.data();

    // -------------------------------------------------------------------------
    // Pass 1: Count incident faces for each owned node.
    // We loop over ALL local faces [0, n_faces) (interior, boundary, and interface).
    // Faces are simple non-self-intersecting polygons (TRI/QUAD), so each node
    // appears at most once per face (no deduplication needed).
    // -------------------------------------------------------------------------
    for (std::size_t f = 0; f < n_faces; ++f) {
        const std::size_t k_start = static_cast<std::size_t>(face_nodes_off[f]);
        const std::size_t k_end   = static_cast<std::size_t>(face_nodes_off[f + 1]);

        for (std::size_t k = k_start; k < k_end; ++k) {
            const LocalIndex nid = face_nodes[k];
            if (nid >= 0 && nid < mp.n_nodes_own) {
                ++offsets_ptr[static_cast<std::size_t>(nid)];
            }
        }
    }

    // -------------------------------------------------------------------------
    // Prefix Sum: Convert counts into in-place insertion cursors
    // -------------------------------------------------------------------------
    LocalIndex total_node_faces = 0;
    for (std::size_t n = 0; n < n_nodes_own; ++n) {
        const LocalIndex count = offsets_ptr[n];
        offsets_ptr[n] = total_node_faces;
        total_node_faces += count;
    }
    offsets_ptr[n_nodes_own] = total_node_faces;

    // Allocate flat storage for incident faces
    node_faces.resize(static_cast<std::size_t>(total_node_faces));
    LocalIndex* CFD_RESTRICT faces_ptr = node_faces.data();

    // -------------------------------------------------------------------------
    // Pass 2: Populate node -> faces connectivity.
    // Because face index f strictly increases monotonically (0, 1, ..., n_faces - 1),
    // incident faces for each vertex are inherently written in ascending order.
    // -------------------------------------------------------------------------
    for (std::size_t f = 0; f < n_faces; ++f) {
        const LocalIndex f_idx    = static_cast<LocalIndex>(f);
        const std::size_t k_start = static_cast<std::size_t>(face_nodes_off[f]);
        const std::size_t k_end   = static_cast<std::size_t>(face_nodes_off[f + 1]);

        for (std::size_t k = k_start; k < k_end; ++k) {
            const LocalIndex nid = face_nodes[k];
            if (nid >= 0 && nid < mp.n_nodes_own) {
                faces_ptr[offsets_ptr[static_cast<std::size_t>(nid)]++] = f_idx;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Restore CSR offsets: Shift offsets array right by 1 element
    // -------------------------------------------------------------------------
    for (std::size_t n = n_nodes_own; n > 0; --n) {
        offsets_ptr[n] = offsets_ptr[n - 1];
    }
    offsets_ptr[0] = 0;
}

void build_cell_cells_node_conn(const MeshPart& mp,
                                const std::vector<LocalIndex>& node_cells_offsets,
                                const std::vector<LocalIndex>& node_cells,
                                std::vector<LocalIndex>& cell_cells_node_offsets, 
                                std::vector<LocalIndex>& cell_cells_node) {
    const std::size_t n_own   = static_cast<std::size_t>(mp.n_own);
    const std::size_t n_cells = static_cast<std::size_t>(mp.n_cells);

    if (n_own == 0 || mp.n_nodes_own == 0) {
        cell_cells_node_offsets.assign(1, 0);
        cell_cells_node.clear();
        return;
    }

    const LocalIndex* CFD_RESTRICT node_off_ptr   = node_cells_offsets.data();
    const LocalIndex* CFD_RESTRICT node_cells_ptr = node_cells.data();
    const LocalIndex* CFD_RESTRICT cell_nodes_off = mp.cell_nodes_offsets.data();
    const LocalIndex* CFD_RESTRICT cell_nodes     = mp.cell_nodes.data();

    // Pass 1: Count unique vertex-sharing neighbors per owned cell
    cell_cells_node_offsets.assign(n_own + 1, 0);
    LocalIndex* CFD_RESTRICT offsets_ptr = cell_cells_node_offsets.data();

    std::vector<LocalIndex> stamp(n_cells, -1);
    LocalIndex* CFD_RESTRICT stamp_ptr = stamp.data();

    for (std::size_t c = 0; c < n_own; ++c) {
        const LocalIndex tag = static_cast<LocalIndex>(c);
        LocalIndex count = 0;

        const std::size_t k_start = static_cast<std::size_t>(cell_nodes_off[c]);
        const std::size_t k_end   = static_cast<std::size_t>(cell_nodes_off[c + 1]);

        for (std::size_t k = k_start; k < k_end; ++k) {
            const std::size_t n       = static_cast<std::size_t>(cell_nodes[k]);
            const std::size_t j_start = static_cast<std::size_t>(node_off_ptr[n]);
            const std::size_t j_end   = static_cast<std::size_t>(node_off_ptr[n + 1]);

            for (std::size_t j = j_start; j < j_end; ++j) {
                const LocalIndex cand = node_cells_ptr[j];
                const std::size_t cand_sz = static_cast<std::size_t>(cand);

                if (cand == tag || stamp_ptr[cand_sz] == tag) {
                    continue;
                }

                stamp_ptr[cand_sz] = tag;
                ++count;
            }
        }
        offsets_ptr[c] = count;
    }

    // Prefix sum: compute exact CSR bounds
    LocalIndex total_neighbors = 0;
    for (std::size_t c = 0; c < n_own; ++c) {
        const LocalIndex count = offsets_ptr[c];
        offsets_ptr[c] = total_neighbors;
        total_neighbors += count;
    }
    offsets_ptr[n_own] = total_neighbors;

    cell_cells_node.resize(static_cast<std::size_t>(total_neighbors));
    LocalIndex* CFD_RESTRICT cells_ptr = cell_cells_node.data();

    // Pass 2: Populate unique neighbors directly into the target buffer
    std::fill(stamp.begin(), stamp.end(), -1);

    for (std::size_t c = 0; c < n_own; ++c) {
        const LocalIndex tag = static_cast<LocalIndex>(c);
        std::size_t cursor = static_cast<std::size_t>(offsets_ptr[c]);

        const std::size_t k_start = static_cast<std::size_t>(cell_nodes_off[c]);
        const std::size_t k_end   = static_cast<std::size_t>(cell_nodes_off[c + 1]);

        for (std::size_t k = k_start; k < k_end; ++k) {
            const std::size_t n       = static_cast<std::size_t>(cell_nodes[k]);
            const std::size_t j_start = static_cast<std::size_t>(node_off_ptr[n]);
            const std::size_t j_end   = static_cast<std::size_t>(node_off_ptr[n + 1]);

            for (std::size_t j = j_start; j < j_end; ++j) {
                const LocalIndex cand = node_cells_ptr[j];
                const std::size_t cand_sz = static_cast<std::size_t>(cand);

                if (cand == tag || stamp_ptr[cand_sz] == tag) {
                    continue;
                }

                stamp_ptr[cand_sz] = tag;
                cells_ptr[cursor++] = cand;
            }
        }

        const std::size_t start_idx = static_cast<std::size_t>(offsets_ptr[c]);
        const std::size_t end_idx   = static_cast<std::size_t>(offsets_ptr[c + 1]);
        std::sort(cells_ptr + start_idx, cells_ptr + end_idx);
    }
}

void build_cell_cells_node_conn(const MeshPart& mp, 
                                std::vector<LocalIndex>& cell_cells_node_offsets, 
                                std::vector<LocalIndex>& cell_cells_node) {
    std::vector<LocalIndex> temp_node_cells_offsets;
    std::vector<LocalIndex> temp_node_cells;

    build_node_cells_conn(mp, temp_node_cells_offsets, temp_node_cells);
    build_cell_cells_node_conn(mp, temp_node_cells_offsets, temp_node_cells, 
                               cell_cells_node_offsets, cell_cells_node);
}


// -----------------------------------------------------------------------------
// Master Dispatcher: Builds requested auxiliary connectivities according to mask
// -----------------------------------------------------------------------------
MeshAuxConnectivity build_aux_connectivity(const MeshPart& mp, AuxConnType mask) {
    MeshAuxConnectivity aux;
    aux.active_mask = mask;

    // Cache subdomain dimension sizes
    aux.n_faces     = mp.n_faces;
    aux.n_cells_own = mp.n_own;
    aux.n_nodes_own = mp.n_nodes_own;

    // Fast return if no connectivities are requested
    if (mask == AuxConnType::None) {
        return aux;
    }

    // 1. Cell -> Faces
    if (has_flag(mask, AuxConnType::CellFaces)) {
        build_cell_faces_conn(mp, aux.cell_faces_offsets, aux.cell_faces);
    }

    // 2. Cell -> Cells (via face sharing)
    if (has_flag(mask, AuxConnType::CellCellsByFace)) {
        build_cell_cells_face_conn(mp, aux.cell_cells_face_offsets, aux.cell_cells_face);
    }

    // 3. Node -> Faces
    if (has_flag(mask, AuxConnType::NodeFaces)) {
        build_node_faces_conn(mp, aux.node_faces_offsets, aux.node_faces);
    }

    // 4. Node -> Cells & Cell -> Cells (via node sharing)
    const bool need_node_cells       = has_flag(mask, AuxConnType::NodeCells);
    const bool need_cell_cells_node  = has_flag(mask, AuxConnType::CellCellsByNode);

    if (need_node_cells) {
        build_node_cells_conn(mp, aux.node_cells_offsets, aux.node_cells);
    }

    if (need_cell_cells_node) {
        if (need_node_cells) {
            // Zero duplicate work: reuse existing Node -> Cells connectivity
            build_cell_cells_node_conn(mp, 
                                       aux.node_cells_offsets, 
                                       aux.node_cells, 
                                       aux.cell_cells_node_offsets, 
                                       aux.cell_cells_node);
        } else {
            // Node -> Cells wasn't requested in output mask, allocate as temporary
            build_cell_cells_node_conn(mp, 
                                       aux.cell_cells_node_offsets, 
                                       aux.cell_cells_node);
        }
    }

    return aux;
}

} // namespace cfd::mesh