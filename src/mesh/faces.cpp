#include "cfd/mesh/faces.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "cfd/core/types.hpp"
#include "cfd/mesh/cgnstables.hpp"
#include "cfd/mpi/log.hpp"
#include "cfd/mpi/mpi_util.hpp"

namespace cfd::mesh {

namespace {

// =============================================================================
// HPC Communication Buffers & POD Messages
// =============================================================================

// Phase 1 message: Sent to the rendezvous owner computed via FaceKeyHash
struct alignas(8) HalfFaceMsg {
    FaceKey key;
    GlobalIndex cell_id = kInvalidGlobalIndex; // Volume cell GID (or kInvalidGlobalIndex if SurfElem)
    PatchId patch = kInvalidPatchId;          // BC Patch ID (if SurfElem)
    std::uint8_t lface = 0;                   // Local face index [0..5]
    std::uint8_t is_surf = 0;                 // 1 if surface element, 0 if volume face

    bool operator<(const HalfFaceMsg& o) const noexcept {
        return key < o.key;
    }
};

// Phase 2 message: Directed dual graph adjacency edge (for remote cell_b owner)
struct alignas(8) DualEdgeMsg {
    GlobalIndex cell_u = kInvalidGlobalIndex; // Local cell on target rank
    GlobalIndex cell_v = kInvalidGlobalIndex; // Adjacent neighbour cell
};

// Binary search rank locator over monotonic (nprocs + 1) displacement array
inline int find_owner_rank(GlobalIndex gid, const std::vector<GlobalIndex>& displ) noexcept {
    auto it = std::upper_bound(displ.begin(), displ.end(), gid);
    return static_cast<int>(std::distance(displ.begin(), it) - 1);
}

// Compute deterministic rendezvous rank via FNV-1a FaceKeyHash
inline int find_rendezvous_rank(const FaceKey& key, int nprocs) noexcept {
    return static_cast<int>(FaceKeyHash{}(key) % static_cast<std::size_t>(nprocs));
}

// Construct canonical sorted FaceKey using fixed 4-slot array with -1 sentinel padding
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
inline FaceKey make_sorted_face_key(const GlobalIndex* raw_nodes, int node_count) noexcept {
    FaceKey k{}; // Initializes all slots to -1 (kInvalidGlobalIndex)
    const int count = std::min(node_count, 4);
    for (int i = 0; i < count; ++i) {
        k.v[static_cast<std::size_t>(i)] = raw_nodes[i];
    }

    if (count > 1) {
        std::sort(k.v.begin(), k.v.begin() + count);
    }
    return k;
}
#pragma GCC diagnostic pop

} // namespace

BuildFacesResult build_faces(const RawMesh& m) {
    // get local rank index and total ranks count
    const int nprocs = m.nprocs;
    const int rank = m.rank;
    const MPI_Comm comm = m.comm;

    // get local number of cells
    const LocalIndex n_loc_cells = m.n_local_cells();


    
    // -------------------------------------------------------------------------
    // Phase 1: Generate Half-Faces & Surface Elements -> Rendezvous by FaceKeyHash
    // -------------------------------------------------------------------------
    std::vector<int> p1_send_counts(static_cast<std::size_t>(nprocs), 0);

    // Pass 1.1: Count send items per rank (Volume cells)
    std::size_t cnode_offset = 0;
    // loop over all local cells
    for (std::size_t c = 0; c < static_cast<std::size_t>(n_loc_cells); ++c) {
        // get cell type
        const std::size_t type_idx = static_cast<std::size_t>(m.ctype[c]);
        if (!is_volume_type(m.ctype[c])) {
            mpi::fatal(comm, "Cell type must be 3D");
        }

        // get number of cell faces
        const std::size_t num_faces = static_cast<std::size_t>(kFacesPerType[type_idx]);

        // loop over all cell faces
        for (std::size_t f = 0; f < num_faces; ++f) {
            // get number of nodes per face
            const std::size_t fn_count = static_cast<std::size_t>(kFaceNodes[type_idx][f]);

            std::array<GlobalIndex, 4> fnodes{};

            // loop over all face nodes
            for (std::size_t i = 0; i < fn_count; ++i) {
                // get local node index (in cell nodes array)
                const std::size_t local_node = static_cast<std::size_t>(kFaceTable[type_idx][f][i]);

                // write global face-node index
                fnodes[i] = m.cnodes[cnode_offset + local_node];
            } // end loop over face nodes

            // creating face key
            const FaceKey key = make_sorted_face_key(fnodes.data(), static_cast<int>(fn_count));

            // find rendezvous destination of face via hash
            const int dest = find_rendezvous_rank(key, nprocs);

            // increase number of faces, owned by each rank
            ++p1_send_counts[static_cast<std::size_t>(dest)];
        } // end loop over all cell faces

        // increase offset
        cnode_offset += static_cast<std::size_t>(kNodesPerType[type_idx]);
    } // end loop over all local cells

    // Pass 1.2: Count send items for surface elements
    for (const auto& surf : m.surf_elems) {
        const int dest = find_rendezvous_rank(surf.key, nprocs);
        ++p1_send_counts[static_cast<std::size_t>(dest)];
    }


    // Allocate Phase 1 send buffer with prefix cursors
    std::vector<int> p1_sdispls(static_cast<std::size_t>(nprocs + 1), 0);
    for (int i = 0; i < nprocs; ++i) {
        std::size_t ii = static_cast<std::size_t>(i);
        p1_sdispls[ii + 1] = p1_sdispls[ii] + p1_send_counts[ii];
    }
    std::vector<HalfFaceMsg> p1_send_buf(static_cast<std::size_t>(p1_sdispls[static_cast<std::size_t>(nprocs)]));
    std::vector<int> p1_cursors = p1_sdispls;

    // Fill Phase 1 send buffer (Volume cells)
    cnode_offset = 0;
    // loop over all local cells
    for (std::size_t c = 0; c < static_cast<std::size_t>(n_loc_cells); ++c) {
        // get cell global index, type and face count
        const GlobalIndex cell_gid = m.global_cell_id(static_cast<LocalIndex>(c));
        const std::size_t type_idx = static_cast<std::size_t>(m.ctype[c]);
        const std::size_t num_faces = static_cast<std::size_t>(kFacesPerType[type_idx]);

        // loop over all cell faces
        for (std::size_t f = 0; f < num_faces; ++f) {
            // get number of face nodes
            const std::size_t fn_count = static_cast<std::size_t>(kFaceNodes[type_idx][f]);

            std::array<GlobalIndex, 4> fnodes{};
            // loop over all face nodes
            for (std::size_t i = 0; i < fn_count; ++i) {
                // get local face-node index (in cell nodes)
                const std::size_t local_node = static_cast<std::size_t>(kFaceTable[type_idx][f][i]);

                // write global face-node index
                fnodes[i] = m.cnodes[cnode_offset + local_node];
            } // end loop over all face nodes

            // creating face key
            const FaceKey key = make_sorted_face_key(fnodes.data(), static_cast<int>(fn_count));

            // find rendezvous destination of face
            const int dest = find_rendezvous_rank(key, nprocs);

            // write data in buffer
            p1_send_buf[static_cast<std::size_t>(p1_cursors[static_cast<std::size_t>(dest)]++)] = 
                HalfFaceMsg{
                    key, cell_gid, kInvalidPatchId, static_cast<std::uint8_t>(f), 0
                };
        } // end loop over cell faces

        // increase offsets
        cnode_offset += static_cast<std::size_t>(kNodesPerType[type_idx]);
    }

    // Fill Phase 1 send buffer (Surface elements)
    for (const auto& surf : m.surf_elems) {
        // find rendezvous rank
        const int dest = find_rendezvous_rank(surf.key, nprocs);

        p1_send_buf[static_cast<std::size_t>(p1_cursors[static_cast<std::size_t>(dest)]++)] =
            HalfFaceMsg{
                surf.key, kInvalidGlobalIndex, surf.patch, 0, 1
            };
    }


    // Exchange Phase 1 messages
    std::vector<HalfFaceMsg> p1_recv_buf;
    mpi::alltoallv_packed(comm, nprocs, p1_send_counts, p1_send_buf, p1_recv_buf);

    // Free send buffers early to release memory pressure
    p1_send_buf.clear();
    p1_send_buf.shrink_to_fit();



    // -------------------------------------------------------------------------
    // Phase 2: Deduplication on Rendezvous Ranks & Matching
    // -------------------------------------------------------------------------
    // sort my faces (that i'v received) by key
    std::sort(p1_recv_buf.begin(), p1_recv_buf.end());

    // initialize arrays
    std::vector<int> p2_face_counts(static_cast<std::size_t>(nprocs), 0);
    std::vector<int> p2_edge_counts(static_cast<std::size_t>(nprocs), 0);

    struct MatchedFace {
        FaceRec rec;
        int rank_a = 0;
        int rank_b = -1;
    };
    std::vector<MatchedFace> matched_faces;
    matched_faces.reserve(p1_recv_buf.size());

    const std::size_t total_recv = p1_recv_buf.size();
    // loop over all received faces
    for (std::size_t i = 0; i < total_recv;) {
        // find number of duplicates (exclusive) = size of face group
        std::size_t j = i + 1;
        while (j < total_recv && p1_recv_buf[i].key == p1_recv_buf[j].key) {
            ++j;
        }

        HalfFaceMsg vol_faces[2];
        int vol_count = 0;
        PatchId patch = kInvalidPatchId;

        // loop over all face-group
        for (std::size_t k = i; k < j; ++k) {
            // get received face
            const auto& item = p1_recv_buf[k];
            if (item.is_surf) {
                patch = item.patch;
            } else {
                if (vol_count < 2) {
                    vol_faces[vol_count++] = item;
                } else {
                    mpi::fatal(comm, "HPC Error: Non-manifold mesh detected (>2 cells sharing a face)!");
                }
            }
        } // end loop over all face group

        if (vol_count == 2) {
            // Interior Face
            GlobalIndex c1 = vol_faces[0].cell_id;
            GlobalIndex c2 = vol_faces[1].cell_id;
            std::uint8_t lf1 = vol_faces[0].lface;
            std::uint8_t lf2 = vol_faces[1].lface;

            if (c1 > c2) {
                std::swap(c1, c2);
                std::swap(lf1, lf2);
            }

            const int r_a = find_owner_rank(c1, m.cell_displ);
            const int r_b = find_owner_rank(c2, m.cell_displ);

            matched_faces.push_back({
                FaceRec{vol_faces[0].key, c1, c2, kInvalidPatchId, lf1, lf2},
                r_a, r_b
            });
            ++p2_face_counts[static_cast<std::size_t>(r_a)];

            if (r_a != r_b) {
                ++p2_edge_counts[static_cast<std::size_t>(r_b)]; // Back-edge for dual graph (cell_b -> cell_a)
            }
        } else if (vol_count == 1) {
            // Boundary Face
            const GlobalIndex c1 = vol_faces[0].cell_id;
            const int r_a = find_owner_rank(c1, m.cell_displ);

            matched_faces.push_back({
                FaceRec{vol_faces[0].key, c1, kInvalidGlobalIndex, patch, vol_faces[0].lface, 255},
                r_a, -1
            });
            ++p2_face_counts[static_cast<std::size_t>(r_a)];
        }

        i = j;
    } // end loop over all received faces

    // Free Phase 1 recv buffer
    p1_recv_buf.clear();
    p1_recv_buf.shrink_to_fit();



    // -------------------------------------------------------------------------
    // Phase 3: Dispatch Matched Faces & Graph Edges to Cell Owners
    // -------------------------------------------------------------------------
    std::vector<int> p2_face_sdispls(static_cast<std::size_t>(nprocs) + 1, 0);
    std::vector<int> p2_edge_sdispls(static_cast<std::size_t>(nprocs) + 1, 0);
    for (std::size_t k = 0; k < static_cast<std::size_t>(nprocs); ++k) {
        p2_face_sdispls[k + 1] = p2_face_sdispls[k] + p2_face_counts[k];
        p2_edge_sdispls[k + 1] = p2_edge_sdispls[k] + p2_edge_counts[k];
    }

    const std::size_t total_sfaces = static_cast<std::size_t>(p2_face_sdispls[static_cast<std::size_t>(nprocs)]);
    const std::size_t total_sedges = static_cast<std::size_t>(p2_edge_sdispls[static_cast<std::size_t>(nprocs)]);

    std::vector<FaceRec> p2_face_send(total_sfaces);
    std::vector<DualEdgeMsg> p2_edge_send(total_sedges);
    std::vector<int> p2_face_cursors = p2_face_sdispls;
    std::vector<int> p2_edge_cursors = p2_edge_sdispls;

    // loop over all faces that I own in rendezvous phase
    for (const auto& mf : matched_faces) {
        p2_face_send[static_cast<std::size_t>(p2_face_cursors[static_cast<std::size_t>(mf.rank_a)]++)] = mf.rec;
        if (mf.rank_b != -1 && mf.rank_a != mf.rank_b) {
            p2_edge_send[static_cast<std::size_t>(p2_edge_cursors[static_cast<std::size_t>(mf.rank_b)]++)] 
                = DualEdgeMsg{mf.rec.cell_b, mf.rec.cell_a};
        }
    } // end loop over all faces

    matched_faces.clear();
    matched_faces.shrink_to_fit();

    // perform data exchange: every rank receives its own faces and remote back-edges
    std::vector<FaceRec> local_faces;
    std::vector<DualEdgeMsg> remote_edges;
    mpi::alltoallv_packed(comm, nprocs, p2_face_counts, p2_face_send, local_faces);
    mpi::alltoallv_packed(comm, nprocs, p2_edge_counts, p2_edge_send, remote_edges);

    p2_face_send.clear();
    p2_edge_send.clear();



    // -------------------------------------------------------------------------
    // Phase 4: Assembly of Dual Graph CSR & Face Statistics
    // -------------------------------------------------------------------------
    BuildFacesResult result;
    result.faces = std::move(local_faces);

    const GlobalIndex my_cell_start = m.cell_displ[static_cast<std::size_t>(rank)];
    const GlobalIndex my_cell_end = m.cell_displ[static_cast<std::size_t>(rank) + 1];
    std::vector<LocalIndex> deg(static_cast<std::size_t>(n_loc_cells), 0);

    GlobalIndex local_ifaces = 0;
    GlobalIndex local_bfaces = 0;

    // loop over all my faces
    for (const auto& f : result.faces) {
        if (f.cell_b != kInvalidGlobalIndex) {
            ++local_ifaces;
            // Edge cell_a -> cell_b

            // get local cell index
            const auto u = static_cast<LocalIndex>(f.cell_a - my_cell_start);
            assert(u >= 0 && u < n_loc_cells);
            ++deg[static_cast<std::size_t>(u)];

            // If cell_b is also local, add edge cell_b -> cell_a immediately
            if (f.cell_b >= my_cell_start && f.cell_b < my_cell_end) {
                const auto v = static_cast<LocalIndex>(f.cell_b - my_cell_start);
                assert(v >= 0 && v < n_loc_cells);
                ++deg[static_cast<std::size_t>(v)];
            }
        } else {
            ++local_bfaces;
        }
    } // end loop over all my faces

    // Remote back-edge contributions
    for (const auto& e : remote_edges) {
        // get local cell index
        const auto v = static_cast<LocalIndex>(e.cell_u - my_cell_start);
        assert(v >= 0 && v < n_loc_cells);
        ++deg[static_cast<std::size_t>(v)];
    }

    // Build CSR offsets
    result.graph.offsets.resize(static_cast<std::size_t>(n_loc_cells) + 1, 0);
    for (std::size_t i = 0; i < static_cast<std::size_t>(n_loc_cells); ++i) {
        result.graph.offsets[i + 1] = result.graph.offsets[i] + deg[i];
    }

    // Fill CSR adj array
    result.graph.adj.resize(static_cast<std::size_t>(result.graph.offsets[static_cast<std::size_t>(n_loc_cells)]));
    std::vector<LocalIndex> head = result.graph.offsets;

    // loop over all my faces
    for (const auto& f : result.faces) {
        if (f.cell_b != kInvalidGlobalIndex) {
            // get local cell index
            const auto u = static_cast<LocalIndex>(f.cell_a - my_cell_start);
            result.graph.adj[static_cast<std::size_t>(head[static_cast<std::size_t>(u)]++)] = f.cell_b;

            if (f.cell_b >= my_cell_start && f.cell_b < my_cell_end) {
                const auto v = static_cast<LocalIndex>(f.cell_b - my_cell_start);
                result.graph.adj[static_cast<std::size_t>(head[static_cast<std::size_t>(v)]++)] = f.cell_a;
            }
        }
    } // end loop over all my faces

    for (const auto& e : remote_edges) {
        const auto v = static_cast<LocalIndex>(e.cell_u - my_cell_start);
        result.graph.adj[static_cast<std::size_t>(head[static_cast<std::size_t>(v)]++)] = e.cell_v;
    }

    // Sort adjacencies for each vertex (ParMETIS/Scotch binary search requirement)
    for (LocalIndex i = 0; i < n_loc_cells; ++i) {
        const LocalIndex start = result.graph.offsets[static_cast<std::size_t>(i)];
        const LocalIndex end = result.graph.offsets[static_cast<std::size_t>(i + 1)];
        std::sort(result.graph.adj.begin() + start, result.graph.adj.begin() + end);
    }

    // Single collective reduction for global stats
    const std::int64_t local_stats[4] = {
        static_cast<std::int64_t>(result.faces.size()),
        static_cast<std::int64_t>(local_bfaces),
        static_cast<std::int64_t>(local_ifaces),
        static_cast<std::int64_t>(result.graph.adj.size())
    };
    std::int64_t global_stats[4] = {0, 0, 0, 0};

    MPI_Allreduce(local_stats, global_stats, 4, MPI_INT64_T, MPI_SUM, comm);

    result.stats.n_faces_g = static_cast<GlobalIndex>(global_stats[0]);
    result.stats.n_bfaces_g = static_cast<GlobalIndex>(global_stats[1]);
    result.stats.n_ifaces_g = static_cast<GlobalIndex>(global_stats[2]);
    result.stats.n_dg_edges = static_cast<GlobalIndex>(global_stats[3]);

    mpi::log_stat("INFO[Faces]: Total face count=%lld, Boundary face count=%lld, Interior face count=%lld", 
                  static_cast<long long>(result.stats.n_faces_g), 
                  static_cast<long long>(result.stats.n_bfaces_g), 
                  static_cast<long long>(result.stats.n_ifaces_g));
    mpi::log_stat("INFO[Dual Graph]: Total edges=%lld", static_cast<long long>(result.stats.n_dg_edges));

    return result;
}

} // namespace cfd::mesh