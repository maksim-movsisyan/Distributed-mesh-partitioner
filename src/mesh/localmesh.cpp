#include "cfd/mesh/localmesh.hpp"

#include <algorithm>
#include <cassert>
#include <sstream>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "cfd/mesh/cgnstables.hpp"
#include "cfd/mpi/log.hpp"
#include "cfd/mpi/mpi_util.hpp"

namespace cfd::mesh {

namespace {

// =============================================================================
// Helper Constants and Node Count Resolution
// =============================================================================

constexpr LocalIndex kInvalidLocal = static_cast<LocalIndex>(-1);
constexpr GlobalIndex kInvalidGlobal = static_cast<GlobalIndex>(-1);
constexpr PatchId kInvalidPatch = static_cast<PatchId>(-1);

inline uint8_t get_cell_node_count(CellType t) noexcept {
    return static_cast<uint8_t>(kNodesPerType[static_cast<std::size_t>(t)]);
}

inline int find_owner_rank(GlobalIndex gid, const std::vector<GlobalIndex>& displ) noexcept {
    auto it = std::upper_bound(displ.begin(), displ.end(), gid);
    return static_cast<int>(std::distance(displ.begin(), it) - 1);
}

// =============================================================================
// Packed POD Structures for MPI Communications
// =============================================================================

struct alignas(8) TargetReqMsg {
    GlobalIndex cell_gid;
    LocalIndex face_idx;
};

struct alignas(8) TargetRespMsg {
    LocalIndex face_idx;
    int target_rank;
};

struct alignas(8) CellMigrateMsg {
    GlobalIndex gid;
    CellType type;
    uint8_t num_nodes;
    GlobalIndex nodes[8];
};

struct alignas(8) FaceMigrateMsg {
    GlobalIndex owner_gid;  // Global ID of the owner cell on the target rank
    GlobalIndex neigh_gid;  // Global ID of the neighbor cell (-1 if boundary face)
    PatchId patch;          // Boundary condition patch ID (-1 if internal or inter-domain face)
    uint8_t lface;          // Local face index within the owner cell (typically 0..5)
    int donor_rank;         // Donor MPI rank for neigh_gid (-1 if boundary or local to target)
};

struct alignas(8) GhostTopoReqMsg {
    GlobalIndex cell_gid;
    LocalIndex ghost_slot;
};

struct alignas(8) GhostTopoRespMsg {
    LocalIndex ghost_slot;
    GlobalIndex gid;
    CellType type;
    uint8_t num_nodes;
    GlobalIndex nodes[8];
};

struct alignas(8) NodeCoordReqMsg {
    GlobalIndex node_gid;
    LocalIndex local_slot;
};

struct alignas(8) NodeCoordRespMsg {
    LocalIndex local_slot;
    double x;
    double y;
    double z;
};

struct alignas(8) GhostHandshakeMsg {
    GlobalIndex cell_gid;
};

} // anonymous namespace

bool meshpart_sane(const MeshPart& mp, std::string& err) {
    if (mp.n_own < 0 || mp.n_own > mp.n_cells) {
        err = "Invalid cell counts: n_own > n_cells";
        return false;
    }
    if (mp.n_nodes_own < 0 || mp.n_nodes_own > mp.n_nodes) {
        err = "Invalid node counts: n_nodes_own > n_nodes";
        return false;
    }
    if (mp.cell_type.size() != static_cast<std::size_t>(mp.n_cells) ||
        mp.cell_gid.size() != static_cast<std::size_t>(mp.n_cells) ||
        mp.cell_donor.size() != static_cast<std::size_t>(mp.n_cells) ||
        mp.cell_nodes_offsets.size() != static_cast<std::size_t>(mp.n_cells + 1)) {
        err = "Cell array size mismatch";
        return false;
    }

    // Check cell donors
    for (LocalIndex i = 0; i < mp.n_own; ++i) {
        if (mp.cell_donor[static_cast<std::size_t>(i)] != -1) {
            err = "Owned cell has donor != -1";
            return false;
        }
    }
    for (LocalIndex i = mp.n_own; i < mp.n_cells; ++i) {
        const int donor = mp.cell_donor[static_cast<std::size_t>(i)];
        if (donor < 0 || donor >= mp.nprocs || donor == mp.rank) {
            err = "Ghost cell has invalid donor rank";
            return false;
        }
    }

    // Check nodes indexing in cells
    for (LocalIndex c = 0; c < mp.n_cells; ++c) {
        const LocalIndex off_start = mp.cell_nodes_offsets[static_cast<std::size_t>(c)];
        const LocalIndex off_end   = mp.cell_nodes_offsets[static_cast<std::size_t>(c + 1)];
        for (LocalIndex k = off_start; k < off_end; ++k) {
            const LocalIndex nid = mp.cell_nodes[static_cast<std::size_t>(k)];
            if (nid < 0 || nid >= mp.n_nodes) {
                err = "Cell references out-of-range local node index";
                return false;
            }
            if (c < mp.n_own && nid >= mp.n_nodes_own) {
                err = "Owned cell references a ghost-exclusive node index";
                return false;
            }
        }
    }

    // Check faces
    if (mp.face_owner.size() != static_cast<std::size_t>(mp.n_faces) ||
        mp.face_neigh.size() != static_cast<std::size_t>(mp.n_faces) ||
        mp.face_nodes_offsets.size() != static_cast<std::size_t>(mp.n_faces + 1)) {
        err = "Face array size mismatch";
        return false;
    }

    for (LocalIndex f = 0; f < mp.n_faces; ++f) {
        const auto f_sz = static_cast<std::size_t>(f);
        const LocalIndex owner = mp.face_owner[f_sz];
        const LocalIndex neigh = mp.face_neigh[f_sz];

        if (owner < 0 || owner >= mp.n_own) {
            err = "Face owner is not a valid owned cell";
            return false;
        }
        if (neigh != kInvalidLocal && (neigh < 0 || neigh >= mp.n_cells)) {
            err = "Face neigh is invalid";
            return false;
        }
    }

    // Check communication maps
    const auto n_nb = static_cast<std::size_t>(mp.n_neighbors());
    if (mp.recv_offsets.size() != n_nb + 1 || mp.send_offsets.size() != n_nb + 1) {
        err = "Comm offsets size mismatch with nb_ranks";
        return false;
    }
    if (static_cast<LocalIndex>(mp.recv_ghost_local.size()) != (mp.n_cells - mp.n_own)) {
        err = "recv_ghost_local count does not match total ghost cells";
        return false;
    }

    // Check BC patches
    if (mp.patch_face_offsets.size() != mp.patches.size() + 1) {
        err = "patch_face_offsets size mismatch with patches";
        return false;
    }

    return true;
}

void migrate_local_mesh(
    RawMesh&& m, std::vector<FaceRec>&& faces,
    const partition::PartitionResult& pr, MeshPart& mp) {

    // get local rank index and total ranks count
    const int rank = m.rank;
    const int nprocs = m.nprocs;
    const MPI_Comm comm = m.comm;
    const std::size_t nprocs_sz = static_cast<std::size_t>(nprocs);

    // initialize result
    mp.rank = rank;
    mp.nprocs = nprocs;
    mp.n_cells_g = m.n_cells_g;
    mp.n_nodes_g = m.n_nodes_g;

    const GlobalIndex my_cell_start = m.cell_displ[static_cast<std::size_t>(rank)];
    const GlobalIndex my_cell_end = m.cell_displ[static_cast<std::size_t>(rank + 1)];



    // -------------------------------------------------------------------------
    // Step 1: Resolve Target Ranks for Remote cell_b in faces
    // -------------------------------------------------------------------------
    std::vector<int> target_b_resolved(faces.size(), -1);
    std::vector<int> req_counts(nprocs_sz, 0);
    struct RemoteQuery {
        GlobalIndex gid;
        LocalIndex face_idx;
    };
    std::vector<RemoteQuery> queries;

    // loop over all my faces
    for (std::size_t i = 0; i < faces.size(); ++i) {
        // get neighbor cell index
        const GlobalIndex cb = faces[i].cell_b;

        if (cb == kInvalidGlobal) {
            // boundary face
            target_b_resolved[i] = -1;
        } else if (cb >= my_cell_start && cb < my_cell_end) {
            // my interior face
            target_b_resolved[i] = pr.cell_target_rank[static_cast<std::size_t>(cb - my_cell_start)];
        } else {
            // inter-rank interior face (i don't know which rank it belongs => request to owner needed)
            const int owner = find_owner_rank(cb, m.cell_displ);
            ++req_counts[static_cast<std::size_t>(owner)];
            queries.push_back({cb, static_cast<LocalIndex>(i)});
        }
    } // end loop ove all my faces

    // creating displacements
    std::vector<int> req_sdispls(nprocs_sz + 1, 0);
    for (std::size_t i = 0; i < nprocs_sz; ++i) {
        req_sdispls[i + 1] = req_sdispls[i] + req_counts[i];
    }

    // fill request buffer
    std::vector<TargetReqMsg> req_send(queries.size());
    std::vector<int> req_curs = req_sdispls;
    for (const auto& q : queries) {
        const int owner = find_owner_rank(q.gid, m.cell_displ);
        req_send[static_cast<std::size_t>(req_curs[static_cast<std::size_t>(owner)]++)]
            = TargetReqMsg{q.gid, q.face_idx};
    }

    // perform requests exchange
    // after that step req_recv contains
    // requests from other ranks that i should send
    std::vector<TargetReqMsg> req_recv;
    mpi::alltoallv_packed(comm, nprocs, req_counts, req_send, req_recv);

    // prepare response array (number of resp = numer of reqs)
    std::vector<TargetRespMsg> resp_send(req_recv.size());
    // loop over all requsts
    for (std::size_t i = 0; i < req_recv.size(); ++i) {
        const auto local_c = static_cast<LocalIndex>(req_recv[i].cell_gid - my_cell_start);
        resp_send[i] = TargetRespMsg{
            req_recv[i].face_idx,
            pr.cell_target_rank[static_cast<std::size_t>(local_c)]
        };
    } // end loop over all requsest

    std::vector<int> resp_send_counts(nprocs_sz, 0);
    MPI_Alltoall(req_counts.data(), 1, MPI_INT, resp_send_counts.data(), 1, MPI_INT, comm);

    std::vector<TargetRespMsg> resp_recv;
    mpi::alltoallv_packed(comm, nprocs, resp_send_counts, resp_send, resp_recv);

    for (const auto& r : resp_recv) {
        target_b_resolved[static_cast<std::size_t>(r.face_idx)] = r.target_rank;
    }



    // -------------------------------------------------------------------------
    // Step 2: Migrate Volume Cells (Owned cells to target ranks)
    // -------------------------------------------------------------------------
    std::vector<int> cell_send_counts(nprocs_sz, 0);
    const LocalIndex n_raw_cells = m.n_local_cells();

    // loop over all local cells
    for (LocalIndex c = 0; c < n_raw_cells; ++c) {
        // get cell target rank
        const int target = pr.cell_target_rank[static_cast<std::size_t>(c)];

        // increase number of cells to send
        ++cell_send_counts[static_cast<std::size_t>(target)];
    } // end loop over all local cells

    // create displacementes
    std::vector<int> cell_sdispls(nprocs_sz + 1, 0);
    for (std::size_t i = 0; i < nprocs_sz; ++i) {
        cell_sdispls[i + 1] = cell_sdispls[i] + cell_send_counts[i];
    }

    // prepare send buffer
    std::vector<CellMigrateMsg> cell_send_buf(static_cast<std::size_t>(n_raw_cells));
    std::vector<int> cell_curs = cell_sdispls;

    // loop over all local cells (fill send buffer)
    for (LocalIndex c = 0; c < n_raw_cells; ++c) {
        // get cell local index and target (owner) rank
        const std::size_t c_sz = static_cast<std::size_t>(c);
        const int target = pr.cell_target_rank[c_sz];

        // get cell type, number of nodes and node indices offset
        const CellType type = m.ctype[c_sz];
        const uint8_t nnodes = get_cell_node_count(type);
        const LocalIndex off = m.cnodes_offsets[c_sz];

        CellMigrateMsg msg{};
        msg.gid = m.global_cell_id(c);
        msg.type = type;
        msg.num_nodes = nnodes;
        for (uint8_t k = 0; k < nnodes; ++k) {
            msg.nodes[k] = m.cnodes[static_cast<std::size_t>(off + k)];
        }

        cell_send_buf[static_cast<std::size_t>(cell_curs[static_cast<std::size_t>(target)]++)] = msg;
    } // end loop over all local cells

    // Free raw cells memory immediately
    m.ctype.clear(); m.ctype.shrink_to_fit();
    m.cnodes.clear(); m.cnodes.shrink_to_fit();
    m.cnodes_offsets.clear(); m.cnodes_offsets.shrink_to_fit();

    // receive all my cells
    std::vector<CellMigrateMsg> cell_recv_buf;
    mpi::alltoallv_packed(comm, nprocs, cell_send_counts, cell_send_buf, cell_recv_buf);
    cell_send_buf.clear(); cell_send_buf.shrink_to_fit();

    const LocalIndex n_owned_cells = static_cast<LocalIndex>(cell_recv_buf.size());
    const std::size_t n_owned_sz = static_cast<std::size_t>(n_owned_cells);
    mp.n_own = n_owned_cells;

    // Temporary storage for owned cells
    std::vector<GlobalIndex> owned_cell_gids(n_owned_sz);
    std::vector<CellType> owned_cell_types(n_owned_sz);
    std::vector<LocalIndex> owned_cell_offsets(n_owned_sz + 1, 0);
    for (std::size_t i = 0; i < n_owned_sz; ++i) {
        owned_cell_offsets[i + 1] = owned_cell_offsets[i] + cell_recv_buf[i].num_nodes;
    }
    std::vector<GlobalIndex> owned_cell_nodes_flat(static_cast<std::size_t>(owned_cell_offsets.back()));
    std::unordered_map<GlobalIndex, LocalIndex> owned_gid_to_local;
    owned_gid_to_local.reserve(static_cast<std::size_t>(n_owned_cells));

    // loop over all my cells (that i'v received)
    for (LocalIndex i = 0; i < n_owned_cells; ++i) {
        const std::size_t i_sz = static_cast<std::size_t>(i);

        // get cell
        const auto& c = cell_recv_buf[i_sz];
        owned_cell_gids[i_sz] = c.gid;
        owned_cell_types[i_sz] = c.type;

        std::size_t off = static_cast<std::size_t>(owned_cell_offsets[i_sz]);
        for (std::size_t k = 0; k < static_cast<std::size_t>(c.num_nodes); ++k) {
            owned_cell_nodes_flat[off + k] = c.nodes[k]; 
        }
        owned_gid_to_local[c.gid] = i;
    } // end loop over all my cells
    cell_recv_buf.clear(); cell_recv_buf.shrink_to_fit();



    // -------------------------------------------------------------------------
    // Step 3: Migrate Faces & Detect Ghost Cells
    // -------------------------------------------------------------------------
    std::vector<int> face_send_counts(nprocs_sz, 0);

    // loop over all local faces
    for (std::size_t i = 0; i < faces.size(); ++i) {
        const auto& f = faces[i];
        const int target_a = pr.cell_target_rank[static_cast<std::size_t>(f.cell_a - my_cell_start)];
        const int target_b = target_b_resolved[i];

        if (target_a == target_b || target_b == -1) {
            ++face_send_counts[static_cast<std::size_t>(target_a)];
        } else {
            // Inter-partition cut: send Face message to both partition owners
            ++face_send_counts[static_cast<std::size_t>(target_a)];
            ++face_send_counts[static_cast<std::size_t>(target_b)];
        }
    }

    // creating face displacements
    std::vector<int> face_sdispls(nprocs_sz + 1, 0);
    for (std::size_t i = 0; i < nprocs_sz; ++i) {
        face_sdispls[i + 1] = face_sdispls[i] + face_send_counts[i];
    }

    std::vector<FaceMigrateMsg> face_send_buf(static_cast<std::size_t>(face_sdispls.back()));
    std::vector<int> face_curs = face_sdispls;

    // loop over all local faces (fill send face buffer)
    for (std::size_t i = 0; i < faces.size(); ++i) {
        const auto& f = faces[i];
        const int target_a = pr.cell_target_rank[static_cast<std::size_t>(f.cell_a - my_cell_start)];
        const int target_b = target_b_resolved[i];

        if (target_a == target_b || target_b == -1) {
            // Interior local face or boundary face
            FaceMigrateMsg msg{};
            msg.owner_gid = f.cell_a;
            msg.neigh_gid = f.cell_b;
            msg.patch = f.patch;
            msg.lface = static_cast<uint8_t>(f.lface_a);
            msg.donor_rank = -1;

            face_send_buf[static_cast<std::size_t>(face_curs[static_cast<std::size_t>(target_a)]++)] = msg;
        } else {
            // inter rank face
            FaceMigrateMsg msg_a{};
            msg_a.owner_gid = f.cell_a;
            msg_a.neigh_gid = f.cell_b;
            msg_a.patch = kInvalidPatch;
            msg_a.lface = static_cast<uint8_t>(f.lface_a);
            msg_a.donor_rank = target_b;
            face_send_buf[static_cast<std::size_t>(face_curs[static_cast<std::size_t>(target_a)]++)] = msg_a;

            FaceMigrateMsg msg_b{};
            msg_b.owner_gid = f.cell_b;
            msg_b.neigh_gid = f.cell_a;
            msg_b.patch = kInvalidPatch;
            msg_b.lface = static_cast<uint8_t>(f.lface_b);
            msg_b.donor_rank = target_a;
            face_send_buf[static_cast<std::size_t>(face_curs[static_cast<std::size_t>(target_b)]++)] = msg_b;
        }
    } // end loop over all local faces
    faces.clear(); faces.shrink_to_fit();

    // perform exchange, after that step each rank get their faces
    std::vector<FaceMigrateMsg> face_recv_buf;
    mpi::alltoallv_packed(comm, nprocs, face_send_counts, face_send_buf, face_recv_buf);
    face_send_buf.clear(); face_send_buf.shrink_to_fit();

    // Group received ghost cells by donor rank
    struct GhostEntry {
        GlobalIndex gid;
        int donor;
    };
    std::vector<GhostEntry> unique_ghosts;
    std::unordered_map<GlobalIndex, LocalIndex> ghost_gid_to_local;

    // loop over all my faces
    for (const auto& f : face_recv_buf) {
        // get only inter-rank faecs
        if (f.donor_rank != -1 && f.neigh_gid != kInvalidGlobal) {
            if (ghost_gid_to_local.find(f.neigh_gid) == ghost_gid_to_local.end()) {
                ghost_gid_to_local[f.neigh_gid] = 0;                                     // mark seen
                unique_ghosts.push_back({f.neigh_gid, f.donor_rank});
            }
        }
    } // end loop ove all my faces

    // Sort ghosts by donor rank to guarantee contiguous halo slices
    std::sort(unique_ghosts.begin(), unique_ghosts.end(), [](const GhostEntry& a, const GhostEntry& b) {
        if (a.donor != b.donor) return a.donor < b.donor;
        return a.gid < b.gid;
    });

    const LocalIndex n_ghosts = static_cast<LocalIndex>(unique_ghosts.size());
    mp.n_cells = mp.n_own + n_ghosts;

    // loop over all ghost cells
    for (LocalIndex i = 0; i < n_ghosts; ++i) {
        const auto g_sz = static_cast<std::size_t>(i);
        ghost_gid_to_local[unique_ghosts[g_sz].gid] = mp.n_own + i;
    } // end loop over all ghosts


    // -------------------------------------------------------------------------
    // Step 4: Fetch Ghost Cell Topologies (CellType & node lists)
    // -------------------------------------------------------------------------
    std::vector<int> ghost_req_counts(nprocs_sz, 0);
    for (const auto& g : unique_ghosts) {
        ++ghost_req_counts[static_cast<std::size_t>(g.donor)];
    }

    std::vector<int> ghost_req_sdispls(nprocs_sz + 1, 0);
    for (std::size_t i = 0; i < nprocs_sz; ++i) {
        ghost_req_sdispls[i + 1] = ghost_req_sdispls[i] + ghost_req_counts[i];
    }
    std::vector<GhostTopoReqMsg> ghost_req_send(static_cast<std::size_t>(n_ghosts));
    std::vector<int> ghost_req_curs = ghost_req_sdispls;

    for (LocalIndex i = 0; i < n_ghosts; ++i) {
        const auto& g = unique_ghosts[static_cast<std::size_t>(i)];
        ghost_req_send[static_cast<std::size_t>(ghost_req_curs[static_cast<std::size_t>(g.donor)]++)] = GhostTopoReqMsg{
            g.gid, i
        };
    }

    std::vector<GhostTopoReqMsg> ghost_req_recv;
    mpi::alltoallv_packed(comm, nprocs, ghost_req_counts, ghost_req_send, ghost_req_recv);

    // Prepare response buffer
    std::vector<GhostTopoRespMsg> ghost_resp_send(ghost_req_recv.size());
    for (std::size_t i = 0; i < ghost_req_recv.size(); ++i) {
        const auto& req = ghost_req_recv[i];
        const LocalIndex loc_c = owned_gid_to_local[req.cell_gid];
        const std::size_t loc_c_sz = static_cast<std::size_t>(loc_c);
        const CellType t = owned_cell_types[loc_c_sz];
        const uint8_t nnodes = get_cell_node_count(t);

        GhostTopoRespMsg resp{};
        resp.ghost_slot = req.ghost_slot;
        resp.gid = req.cell_gid;
        resp.type = t;
        resp.num_nodes = nnodes;

        const std::size_t off = static_cast<std::size_t>(owned_cell_offsets[loc_c_sz]);
        for (uint8_t k = 0; k < nnodes; ++k) {
            resp.nodes[k] = owned_cell_nodes_flat[off + static_cast<std::size_t>(k)];
        }
        ghost_resp_send[i] = resp;
    }

    std::vector<int> ghost_resp_send_counts(nprocs_sz, 0);
    MPI_Alltoall(ghost_req_counts.data(), 1, MPI_INT, ghost_resp_send_counts.data(), 1, MPI_INT, comm);

    std::vector<GhostTopoRespMsg> ghost_resp_recv;
    mpi::alltoallv_packed(comm, nprocs, ghost_resp_send_counts, ghost_resp_send, ghost_resp_recv);

    // Allocate ghost metadata
    std::vector<GlobalIndex> ghost_cell_gids(static_cast<std::size_t>(n_ghosts));
    std::vector<CellType> ghost_cell_types(static_cast<std::size_t>(n_ghosts));
    std::vector<int> ghost_cell_donors(static_cast<std::size_t>(n_ghosts));
    std::vector<LocalIndex> ghost_cell_offsets(static_cast<std::size_t>(n_ghosts) + 1, 0);

    // Pass 1: Map received cell metadata to slots
    for (const auto& resp : ghost_resp_recv) {
        const std::size_t slot = static_cast<std::size_t>(resp.ghost_slot);
        ghost_cell_gids[slot] = resp.gid;
        ghost_cell_types[slot] = resp.type;
        ghost_cell_donors[slot] = unique_ghosts[slot].donor;
    }

    // Pass 2: Prefix sum for ghost node offsets
    for (std::size_t i = 0; i < static_cast<std::size_t>(n_ghosts); ++i) {
        ghost_cell_offsets[i + 1] = ghost_cell_offsets[i] + get_cell_node_count(ghost_cell_types[i]);
    }

    // Pass 3: Fill flat ghost node IDs
    std::vector<GlobalIndex> ghost_cell_nodes_flat(static_cast<std::size_t>(ghost_cell_offsets.back()));
    for (const auto& resp : ghost_resp_recv) {
        const std::size_t slot = static_cast<std::size_t>(resp.ghost_slot);
        const std::size_t off = static_cast<std::size_t>(ghost_cell_offsets[slot]);

        for (std::size_t k = 0; k < static_cast<std::size_t>(resp.num_nodes); ++k) {
            ghost_cell_nodes_flat[off + k] = resp.nodes[k]; 
        }
    }



    // -------------------------------------------------------------------------
    // Step 5: Collect Unique Node GIDs & Fetch Coordinates (x, y, z)
    // -------------------------------------------------------------------------
    std::vector<GlobalIndex> owned_node_gids = owned_cell_nodes_flat;
    std::sort(owned_node_gids.begin(), owned_node_gids.end());
    owned_node_gids.erase(std::unique(owned_node_gids.begin(), owned_node_gids.end()), owned_node_gids.end());

    std::unordered_map<GlobalIndex, LocalIndex> node_gid_to_local;
    node_gid_to_local.reserve(owned_node_gids.size() * 2);

    mp.n_nodes_own = static_cast<LocalIndex>(owned_node_gids.size());
    for (LocalIndex i = 0; i < mp.n_nodes_own; ++i) {
        node_gid_to_local[owned_node_gids[static_cast<std::size_t>(i)]] = i;
    }

    std::vector<GlobalIndex> ghost_only_node_gids;
    for (const auto nid : ghost_cell_nodes_flat) {
        if (node_gid_to_local.find(nid) == node_gid_to_local.end()) {
            ghost_only_node_gids.push_back(nid);
        }
    }
    std::sort(ghost_only_node_gids.begin(), ghost_only_node_gids.end());
    ghost_only_node_gids.erase(std::unique(ghost_only_node_gids.begin(), ghost_only_node_gids.end()), ghost_only_node_gids.end());

    for (std::size_t i = 0; i < ghost_only_node_gids.size(); ++i) {
        node_gid_to_local[ghost_only_node_gids[i]] = mp.n_nodes_own + static_cast<LocalIndex>(i);
    }

    mp.n_nodes = mp.n_nodes_own + static_cast<LocalIndex>(ghost_only_node_gids.size());

    mp.node_gid.resize(static_cast<std::size_t>(mp.n_nodes));
    for (LocalIndex i = 0; i < mp.n_nodes_own; ++i) {
        mp.node_gid[static_cast<std::size_t>(i)] = owned_node_gids[static_cast<std::size_t>(i)];
    }
    for (std::size_t i = 0; i < ghost_only_node_gids.size(); ++i) {
        mp.node_gid[static_cast<std::size_t>(mp.n_nodes_own) + i] = ghost_only_node_gids[i];
    }

    // Fetch coordinates from initial node owners
    std::vector<int> node_req_counts(nprocs_sz, 0);
    for (LocalIndex i = 0; i < mp.n_nodes; ++i) {
        const GlobalIndex nid = mp.node_gid[static_cast<std::size_t>(i)];
        const int owner = find_owner_rank(nid, m.node_displ);
        ++node_req_counts[static_cast<std::size_t>(owner)];
    }

    std::vector<int> node_req_sdispls(nprocs_sz + 1, 0);
    for (std::size_t i = 0; i < nprocs_sz; ++i) {
        node_req_sdispls[i + 1] = node_req_sdispls[i] + node_req_counts[i];
    }
    std::vector<NodeCoordReqMsg> node_req_send(static_cast<std::size_t>(mp.n_nodes));
    std::vector<int> node_req_curs = node_req_sdispls;

    for (LocalIndex i = 0; i < mp.n_nodes; ++i) {
        const GlobalIndex nid = mp.node_gid[static_cast<std::size_t>(i)];
        const int owner = find_owner_rank(nid, m.node_displ);
        node_req_send[static_cast<std::size_t>(node_req_curs[static_cast<std::size_t>(owner)]++)] = NodeCoordReqMsg{
            nid, i
        };
    }

    std::vector<NodeCoordReqMsg> node_req_recv;
    mpi::alltoallv_packed(comm, nprocs, node_req_counts, node_req_send, node_req_recv);

    const GlobalIndex my_node_start = m.node_displ[static_cast<std::size_t>(rank)];
    std::vector<NodeCoordRespMsg> node_resp_send(node_req_recv.size());

    for (std::size_t i = 0; i < node_req_recv.size(); ++i) {
        const std::size_t local_n = static_cast<std::size_t>(node_req_recv[i].node_gid - my_node_start);
        node_resp_send[i] = NodeCoordRespMsg{
            node_req_recv[i].local_slot,
            m.my_node_coords_x[local_n],
            m.my_node_coords_y[local_n],
            m.my_node_coords_z[local_n]
        };
    }

    // Free raw coordinates
    m.my_node_coords_x.clear(); m.my_node_coords_x.shrink_to_fit();
    m.my_node_coords_y.clear(); m.my_node_coords_y.shrink_to_fit();
    m.my_node_coords_z.clear(); m.my_node_coords_z.shrink_to_fit();

    std::vector<int> node_resp_send_counts(nprocs_sz, 0);
    MPI_Alltoall(node_req_counts.data(), 1, MPI_INT, node_resp_send_counts.data(), 1, MPI_INT, comm);

    std::vector<NodeCoordRespMsg> node_resp_recv;
    mpi::alltoallv_packed(comm, nprocs, node_resp_send_counts, node_resp_send, node_resp_recv);

    mp.node_x.resize(static_cast<std::size_t>(mp.n_nodes));
    mp.node_y.resize(static_cast<std::size_t>(mp.n_nodes));
    mp.node_z.resize(static_cast<std::size_t>(mp.n_nodes));

    for (const auto& resp : node_resp_recv) {
        const auto slot = static_cast<std::size_t>(resp.local_slot);
        mp.node_x[slot] = resp.x;
        mp.node_y[slot] = resp.y;
        mp.node_z[slot] = resp.z;
    }



    // -------------------------------------------------------------------------
    // Step 6: Assemble Final Local Cells
    // -------------------------------------------------------------------------
    const std::size_t n_cells_sz = static_cast<std::size_t>(mp.n_cells);
    mp.cell_type.resize(n_cells_sz);
    mp.cell_gid.resize(n_cells_sz);
    mp.cell_donor.resize(n_cells_sz);
    mp.cell_nodes_offsets.resize(n_cells_sz + 1, 0);

    for (LocalIndex i = 0; i < mp.n_own; ++i) {
        const std::size_t i_sz = static_cast<std::size_t>(i);
        mp.cell_type[i_sz] = owned_cell_types[i_sz];
        mp.cell_gid[i_sz] = owned_cell_gids[i_sz];
        mp.cell_donor[i_sz] = -1;
        mp.cell_nodes_offsets[i_sz + 1] = mp.cell_nodes_offsets[i_sz] + 
            static_cast<LocalIndex>(owned_cell_offsets[i_sz + 1] - owned_cell_offsets[i_sz]);
    }

    for (LocalIndex i = 0; i < n_ghosts; ++i) {
        const std::size_t g_sz = static_cast<std::size_t>(mp.n_own + i);
        const std::size_t slot = static_cast<std::size_t>(i);
        mp.cell_type[g_sz] = ghost_cell_types[slot];
        mp.cell_gid[g_sz] = ghost_cell_gids[slot];
        mp.cell_donor[g_sz] = ghost_cell_donors[slot];
        mp.cell_nodes_offsets[g_sz + 1] = mp.cell_nodes_offsets[g_sz] + 
            static_cast<LocalIndex>(ghost_cell_offsets[slot + 1] - ghost_cell_offsets[slot]);
    }

    mp.cell_nodes.resize(static_cast<std::size_t>(mp.cell_nodes_offsets.back()));
    for (LocalIndex i = 0; i < mp.n_own; ++i) {
        const std::size_t i_sz = static_cast<std::size_t>(i);
        const LocalIndex off = mp.cell_nodes_offsets[i_sz];
        const std::size_t n_nodes_in_cell = static_cast<std::size_t>(owned_cell_offsets[i_sz + 1] - owned_cell_offsets[i_sz]);
        for (std::size_t k = 0; k < n_nodes_in_cell; ++k) {
            mp.cell_nodes[static_cast<std::size_t>(off) + k] = 
                node_gid_to_local[owned_cell_nodes_flat[static_cast<std::size_t>(owned_cell_offsets[i_sz]) + k]];
        }
    }
    for (LocalIndex i = 0; i < n_ghosts; ++i) {
        const std::size_t g_sz = static_cast<std::size_t>(mp.n_own + i);
        const std::size_t slot = static_cast<std::size_t>(i);
        const LocalIndex off = mp.cell_nodes_offsets[g_sz];
        const std::size_t n_nodes_in_cell = static_cast<std::size_t>(ghost_cell_offsets[slot + 1] - ghost_cell_offsets[slot]);
        for (std::size_t k = 0; k < n_nodes_in_cell; ++k) {
            mp.cell_nodes[static_cast<std::size_t>(off) + k] = 
                node_gid_to_local[ghost_cell_nodes_flat[static_cast<std::size_t>(ghost_cell_offsets[slot]) + k]];
        }
    }



    // -------------------------------------------------------------------------
    // Step 7: Assemble & Sort Faces
    // -------------------------------------------------------------------------
    struct LocalFaceTemp {
        LocalIndex owner;
        LocalIndex neigh;
        PatchId patch;
        CellType type;
        uint8_t num_nodes;
        LocalIndex nodes[4];

        bool operator<(const LocalFaceTemp& o) const noexcept {
            if (owner != o.owner) return owner < o.owner;
            return neigh < o.neigh;
        }
    };

    std::vector<LocalFaceTemp> local_faces(face_recv_buf.size());

    // loop over all my faces
    for (std::size_t i = 0; i < face_recv_buf.size(); ++i) {
        const auto& f = face_recv_buf[i];
        LocalFaceTemp lf{};
        lf.owner = owned_gid_to_local[f.owner_gid];
        
        if (f.neigh_gid == kInvalidGlobal) {
            lf.neigh = kInvalidLocal;
        } else if (f.donor_rank == -1) {
            lf.neigh = owned_gid_to_local[f.neigh_gid];
        } else {
            lf.neigh = ghost_gid_to_local[f.neigh_gid];
        }
        
        lf.patch = f.patch;

        const std::size_t owner_sz = static_cast<std::size_t>(lf.owner);
        const CellType cell_t = owned_cell_types[owner_sz];
        const std::size_t c_idx = static_cast<std::size_t>(cell_t);
        const std::size_t lf_idx = static_cast<std::size_t>(f.lface);

        const int fnodes_count = kFaceNodes[c_idx][lf_idx];
        lf.num_nodes = static_cast<uint8_t>(fnodes_count);
        lf.type = (fnodes_count == 3) ? CellType::TRI : CellType::QUAD;

        const LocalIndex cell_node_off = owned_cell_offsets[owner_sz];

        for (int k = 0; k < fnodes_count; ++k) {
            const int local_node_in_cell = kFaceTable[c_idx][lf_idx][k];
            const GlobalIndex node_gid = owned_cell_nodes_flat[static_cast<std::size_t>(cell_node_off + local_node_in_cell)];
            lf.nodes[k] = node_gid_to_local[node_gid];
        }

        local_faces[i] = lf;
    } // end loop over all my faces

    std::sort(local_faces.begin(), local_faces.end());

    mp.n_faces = static_cast<LocalIndex>(local_faces.size());
    const std::size_t n_faces_sz = static_cast<std::size_t>(mp.n_faces);
    mp.face_owner.resize(n_faces_sz);
    mp.face_neigh.resize(n_faces_sz);
    mp.face_patch.resize(n_faces_sz);
    mp.face_type.resize(n_faces_sz);
    mp.face_nodes_offsets.resize(n_faces_sz + 1, 0);

    for (std::size_t i = 0; i < n_faces_sz; ++i) {
        mp.face_owner[i] = local_faces[i].owner;
        mp.face_neigh[i] = local_faces[i].neigh;
        mp.face_patch[i] = local_faces[i].patch;
        mp.face_type[i]  = local_faces[i].type;
        mp.face_nodes_offsets[i + 1] = mp.face_nodes_offsets[i] + local_faces[i].num_nodes;
    }

    mp.face_nodes.resize(static_cast<std::size_t>(mp.face_nodes_offsets.back()));
    for (std::size_t i = 0; i < n_faces_sz; ++i) {
        const LocalIndex off = mp.face_nodes_offsets[i];
        for (uint8_t k = 0; k < local_faces[i].num_nodes; ++k) {
            mp.face_nodes[static_cast<std::size_t>(off + k)] = local_faces[i].nodes[k];
        }
    }

    


    // -------------------------------------------------------------------------
    // Step 8: Build Communication Maps (Symmetric Send/Recv)
    // -------------------------------------------------------------------------
    std::vector<int> nb_ranks_set;
    for (const auto& g : unique_ghosts) {
        nb_ranks_set.push_back(g.donor);
    }
    std::sort(nb_ranks_set.begin(), nb_ranks_set.end());
    nb_ranks_set.erase(std::unique(nb_ranks_set.begin(), nb_ranks_set.end()), nb_ranks_set.end());

    mp.nb_ranks = nb_ranks_set;
    const std::size_t n_nb = mp.nb_ranks.size();
    mp.recv_offsets.resize(n_nb + 1, 0);
    mp.send_offsets.resize(n_nb + 1, 0);

    std::vector<int> nb_ghost_counts(n_nb, 0);
    for (const auto& g : unique_ghosts) {
        auto it = std::lower_bound(mp.nb_ranks.begin(), mp.nb_ranks.end(), g.donor);
        const std::size_t idx = static_cast<std::size_t>(std::distance(mp.nb_ranks.begin(), it));
        ++nb_ghost_counts[idx];
    }

    for (std::size_t i = 0; i < n_nb; ++i) {
        mp.recv_offsets[i + 1] = mp.recv_offsets[i] + nb_ghost_counts[i];
    }

    mp.recv_ghost_local.resize(static_cast<std::size_t>(n_ghosts));
    for (LocalIndex i = 0; i < n_ghosts; ++i) {
        mp.recv_ghost_local[static_cast<std::size_t>(i)] = mp.n_own + i;
    }

    // Handshake send list: send requested GIDs to neighbours
    std::vector<int> hs_send_counts(nprocs_sz, 0);
    for (std::size_t i = 0; i < n_nb; ++i) {
        hs_send_counts[static_cast<std::size_t>(mp.nb_ranks[i])] = nb_ghost_counts[i];
    }

    std::vector<GhostHandshakeMsg> hs_send_buf(static_cast<std::size_t>(n_ghosts));
    for (LocalIndex i = 0; i < n_ghosts; ++i) {
        hs_send_buf[static_cast<std::size_t>(i)] = GhostHandshakeMsg{unique_ghosts[static_cast<std::size_t>(i)].gid};
    }

    std::vector<int> hs_recv_counts(nprocs_sz, 0);
    MPI_Alltoall(hs_send_counts.data(), 1, MPI_INT, hs_recv_counts.data(), 1, MPI_INT, comm);

    std::vector<GhostHandshakeMsg> hs_recv_buf;
    mpi::alltoallv_packed(comm, nprocs, hs_send_counts, hs_send_buf, hs_recv_buf);

    for (std::size_t i = 0; i < n_nb; ++i) {
        const int neigh = mp.nb_ranks[i];
        mp.send_offsets[i + 1] = mp.send_offsets[i] + hs_recv_counts[static_cast<std::size_t>(neigh)];
    }

    mp.send_owned_local.resize(hs_recv_buf.size());
    for (std::size_t i = 0; i < hs_recv_buf.size(); ++i) {
        mp.send_owned_local[i] = owned_gid_to_local[hs_recv_buf[i].cell_gid];
    }



    
    // -------------------------------------------------------------------------
    // Step 9: Boundary Condition Patches & Global Statistics (Flat CSR)
    // -------------------------------------------------------------------------
    mp.patches.clear();
    for (const auto& bc : m.bcs) {
        mp.patches.push_back(MeshPart::Patch{bc.name, bc.cgns_type});
    }

    const std::size_t n_patches_sz = mp.patches.size();
    mp.patch_face_offsets.assign(n_patches_sz + 1, 0);

    // Pass 1: Count local boundary faces per patch
    std::vector<LocalIndex> patch_counts(n_patches_sz, 0);
    LocalIndex total_local_bfaces = 0;

    for (LocalIndex f = 0; f < mp.n_faces; ++f) {
        const PatchId p = mp.face_patch[static_cast<std::size_t>(f)];
        if (p != kInvalidPatch) {
            const std::size_t p_sz = static_cast<std::size_t>(p);
            if (p_sz < n_patches_sz) {
                ++patch_counts[p_sz];
                ++total_local_bfaces;
            }
        }
    }

    // Prefix sum to compute CSR offsets
    for (std::size_t p = 0; p < n_patches_sz; ++p) {
        mp.patch_face_offsets[p + 1] = mp.patch_face_offsets[p] + patch_counts[p];
    }

    // Allocate flat contiguous array for all patch faces (single allocation)
    mp.patch_faces.resize(static_cast<std::size_t>(mp.patch_face_offsets.back()));
    std::vector<LocalIndex> patch_cursors = mp.patch_face_offsets;

    // Pass 2: Fill flat patch_faces using cursors
    for (LocalIndex f = 0; f < mp.n_faces; ++f) {
        const PatchId p = mp.face_patch[static_cast<std::size_t>(f)];
        if (p != kInvalidPatch) {
            const std::size_t p_sz = static_cast<std::size_t>(p);
            if (p_sz < n_patches_sz) {
                const std::size_t write_pos = static_cast<std::size_t>(patch_cursors[p_sz]++);
                mp.patch_faces[write_pos] = f;
            }
        }
    }




    // Bounding Box
    double loc_bbox_lo[3] = {std::numeric_limits<double>::max(), 
                             std::numeric_limits<double>::max(), 
                             std::numeric_limits<double>::max()};
    double loc_bbox_hi[3] = {-std::numeric_limits<double>::max(), 
                             -std::numeric_limits<double>::max(), 
                             -std::numeric_limits<double>::max()};

    for (LocalIndex i = 0; i < mp.n_nodes_own; ++i) {
        const auto i_sz = static_cast<std::size_t>(i);
        loc_bbox_lo[0] = std::min(loc_bbox_lo[0], mp.node_x[i_sz]);
        loc_bbox_lo[1] = std::min(loc_bbox_lo[1], mp.node_y[i_sz]);
        loc_bbox_lo[2] = std::min(loc_bbox_lo[2], mp.node_z[i_sz]);

        loc_bbox_hi[0] = std::max(loc_bbox_hi[0], mp.node_x[i_sz]);
        loc_bbox_hi[1] = std::max(loc_bbox_hi[1], mp.node_y[i_sz]);
        loc_bbox_hi[2] = std::max(loc_bbox_hi[2], mp.node_z[i_sz]);
    }

    MPI_Allreduce(loc_bbox_lo, mp.bbox_lo, 3, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(loc_bbox_hi, mp.bbox_hi, 3, MPI_DOUBLE, MPI_MAX, comm);

    // Global counts
    GlobalIndex local_counts[2] = {
        static_cast<GlobalIndex>(mp.n_faces),
        static_cast<GlobalIndex>(total_local_bfaces)
    };
    GlobalIndex global_counts[2] = {0, 0};
    MPI_Allreduce(local_counts, global_counts, 2, MPI_INT64_T, MPI_SUM, comm);

    mp.n_faces_g = global_counts[0] - pr.global_edge_cut;
    mp.n_bfaces_g = global_counts[1];

    std::string err;
    if (!meshpart_sane(mp, err)) {
        std::stringstream ss;
        ss << "MeshPart sanity failed on rank " << rank << ": " << err;
        std::string result = ss.str();
        mpi::fatal(comm, result);
    }

    if (rank == 0) {
        mpi::log_stat("INFO[Mesh migration]: Mesh migration complete. Total cells=%lld, Total nodes=%lld, Total faces=%lld, Total boundary faces=%lld",
                      static_cast<long long>(mp.n_cells_g),
                      static_cast<long long>(mp.n_nodes_g),
                      static_cast<long long>(mp.n_faces_g),
                      static_cast<long long>(mp.n_bfaces_g));

        mpi::log_stat("INFO[Mesh migration]: Mesh bounding box: [%.3f;%.3f]x[%.3f;%.3f]x[%.3f;%.3f]",
                      mp.bbox_lo[0], mp.bbox_hi[0],
                      mp.bbox_lo[1], mp.bbox_hi[1],
                      mp.bbox_lo[2], mp.bbox_hi[2]);
    }
}

} // namespace cfd::mesh