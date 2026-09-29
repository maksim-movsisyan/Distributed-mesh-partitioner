#include "cfd/mesh/localmesh.hpp"

#include <algorithm>
#include <cassert>
#include <sstream>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
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

// Release backing storage at the last use, rather than at function exit.
template<class Container>
void release_storage(Container& c) { Container{}.swap(c); }

// Blocking exchange: the send buffer is dead after this call. Avoid copying it
// through MPI at all in the single-rank case.
template<class T>
void exchange_and_release(MPI_Comm comm, int nprocs, const std::vector<int>& counts,
                          std::vector<T>& send, std::vector<T>& recv) {
    if (nprocs == 1) recv = std::move(send);
    else {
        mpi::alltoallv_packed(comm, nprocs, counts, send, recv);
        release_storage(send);
    }
}

// Dense lookup for compact GID ranges; otherwise a contiguous open-addressed
// table. Unlike unordered_map, the sparse case has no allocation per entry.
// Stored LocalIndex values must be nonnegative; -1 marks an unused slot.
class GidIndex {
public:
    GidIndex(GlobalIndex lo, GlobalIndex hi, std::size_t expected) : lo_(lo) {
        expected = std::max<std::size_t>(expected, 1);
        const auto span = hi >= lo ? static_cast<std::uint64_t>(hi - lo) + 1 : 0;
        dense_ = span / 4 + (span % 4 != 0) <= expected;
        if (dense_) values_.assign(static_cast<std::size_t>(span), kInvalidLocal);
        else {
            std::size_t capacity = 8;
            while (capacity - capacity / 4 < expected) capacity *= 2;
            keys_.resize(capacity);
            values_.assign(capacity, kInvalidLocal);
        }
    }

    bool insert(GlobalIndex gid, LocalIndex value) {
        auto slot = find_slot(gid);
        if (values_[slot] != kInvalidLocal) return false;
        if (!dense_ && size_ == values_.size() - values_.size() / 4) {
            grow();
            slot = find_slot(gid);
        }
        if (!dense_) keys_[slot] = gid;
        values_[slot] = value;
        ++size_;
        return true;
    }

    void assign(GlobalIndex gid, LocalIndex value) {
        const auto slot = find_slot(gid);
        assert(values_[slot] != kInvalidLocal);
        values_[slot] = value;
    }

    LocalIndex at(GlobalIndex gid) const {
        const auto value = values_[find_slot(gid)];
        if (value == kInvalidLocal) throw std::runtime_error("Missing mesh GID in local lookup");
        return value;
    }

    bool is_dense() const noexcept { return dense_; }

    // During node collection, marker 0 means used by owned cells and marker 1
    // means ghost-only. Scanning dense slots emits each group in GID order.
    void write_dense_node_ids(std::vector<GlobalIndex>& ids, std::size_t owned_count) const {
        assert(dense_ && ids.size() == size_);
        std::size_t owned = 0, ghost = owned_count;
        for (std::size_t i = 0; i < values_.size(); ++i) {
            if (values_[i] == 0) ids[owned++] = lo_ + static_cast<GlobalIndex>(i);
            else if (values_[i] == 1) ids[ghost++] = lo_ + static_cast<GlobalIndex>(i);
        }
        assert(owned == owned_count && ghost == ids.size());
    }

    void release() { release_storage(keys_); release_storage(values_); }

private:
    static std::uint64_t hash(GlobalIndex gid) noexcept {
        auto x = static_cast<std::uint64_t>(gid);
        x ^= x >> 30; x *= UINT64_C(0xbf58476d1ce4e5b9);
        x ^= x >> 27; x *= UINT64_C(0x94d049bb133111eb);
        return x ^ (x >> 31);
    }
    std::size_t find_slot(GlobalIndex gid) const {
        if (dense_) {
            assert(gid >= lo_ && static_cast<std::uint64_t>(gid - lo_) < values_.size());
            return static_cast<std::size_t>(gid - lo_);
        }
        const auto mask = values_.size() - 1;
        auto slot = static_cast<std::size_t>(hash(gid)) & mask;
        while (values_[slot] != kInvalidLocal && keys_[slot] != gid) slot = (slot + 1) & mask;
        return slot;
    }
    void grow() {
        auto old_keys = std::move(keys_);
        auto old_values = std::move(values_);
        keys_.resize(old_values.size() * 2);
        values_.assign(keys_.size(), kInvalidLocal);
        for (std::size_t i = 0; i < old_values.size(); ++i) {
            if (old_values[i] == kInvalidLocal) continue;
            const auto slot = find_slot(old_keys[i]);
            keys_[slot] = old_keys[i];
            values_[slot] = old_values[i];
        }
    }
    GlobalIndex lo_;
    bool dense_ = false;
    std::size_t size_ = 0;
    std::vector<GlobalIndex> keys_;
    std::vector<LocalIndex> values_;
};

struct LocalFaceTemp {
    LocalIndex owner;
    LocalIndex neigh;
    PatchId patch;
    uint8_t lface;

    bool operator<(const LocalFaceTemp& o) const noexcept {
        if (owner != o.owner) return owner < o.owner;
        return neigh < o.neigh;
    }
};


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
    std::vector<int> target_b_resolved;
    if (nprocs > 1) {
        target_b_resolved.assign(faces.size(), -1);
        std::vector<int> req_counts(nprocs_sz, 0);
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
            }
        } // end loop ove all my faces

        // creating displacements
        std::vector<int> req_sdispls(nprocs_sz + 1, 0);
        for (std::size_t i = 0; i < nprocs_sz; ++i) {
            req_sdispls[i + 1] = req_sdispls[i] + req_counts[i];
        }

        // fill request buffer
        std::vector<TargetReqMsg> req_send(static_cast<std::size_t>(req_sdispls.back()));
        std::vector<int> req_curs = req_sdispls;
        for (std::size_t i = 0; i < faces.size(); ++i) {
            const auto gid = faces[i].cell_b;
            if (gid == kInvalidGlobal || (gid >= my_cell_start && gid < my_cell_end)) continue;
            const int owner = find_owner_rank(gid, m.cell_displ);
            req_send[static_cast<std::size_t>(req_curs[static_cast<std::size_t>(owner)]++)]
                = TargetReqMsg{gid, static_cast<LocalIndex>(i)};
        }

        // perform requests exchange
        // after that step req_recv contains
        // requests from other ranks that i should send
        std::vector<TargetReqMsg> req_recv;
        exchange_and_release(comm, nprocs, req_counts, req_send, req_recv);

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

        release_storage(req_recv);

        std::vector<int> resp_send_counts(nprocs_sz, 0);
        MPI_Alltoall(req_counts.data(), 1, MPI_INT, resp_send_counts.data(), 1, MPI_INT, comm);

        std::vector<TargetRespMsg> resp_recv;
        exchange_and_release(comm, nprocs, resp_send_counts, resp_send, resp_recv);

        for (const auto& r : resp_recv) {
            target_b_resolved[static_cast<std::size_t>(r.face_idx)] = r.target_rank;
        }


        release_storage(resp_recv);
    }

    // -------------------------------------------------------------------------
    // Step 2: Migrate Volume Cells (Owned cells to target ranks)
    // -------------------------------------------------------------------------
    const LocalIndex n_raw_cells = m.n_local_cells();
    std::vector<GlobalIndex> owned_cell_gids;
    std::vector<CellType> owned_cell_types;
    std::vector<LocalIndex> owned_cell_offsets;
    std::vector<GlobalIndex> owned_cell_nodes_flat;
    if (nprocs == 1) {
        mp.n_own = n_raw_cells;
        owned_cell_gids.resize(static_cast<std::size_t>(n_raw_cells));
        for (LocalIndex c = 0; c < n_raw_cells; ++c)
            owned_cell_gids[static_cast<std::size_t>(c)] = m.global_cell_id(c);
        owned_cell_types = std::move(m.ctype);
        owned_cell_offsets = std::move(m.cnodes_offsets);
        owned_cell_nodes_flat = std::move(m.cnodes);
    } else {
        std::vector<int> cell_send_counts(nprocs_sz, 0);

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
        release_storage(m.ctype);
        release_storage(m.cnodes);
        release_storage(m.cnodes_offsets);

        // receive all my cells
        std::vector<CellMigrateMsg> cell_recv_buf;
        exchange_and_release(comm, nprocs, cell_send_counts, cell_send_buf, cell_recv_buf);

        const LocalIndex n_owned_cells = static_cast<LocalIndex>(cell_recv_buf.size());
        const std::size_t n_owned_sz = static_cast<std::size_t>(n_owned_cells);
        mp.n_own = n_owned_cells;

        // Temporary storage for owned cells
        owned_cell_gids.resize(n_owned_sz);
        owned_cell_types.resize(n_owned_sz);
        owned_cell_offsets.assign(n_owned_sz + 1, 0);
        for (std::size_t i = 0; i < n_owned_sz; ++i) {
            owned_cell_offsets[i + 1] = owned_cell_offsets[i] + cell_recv_buf[i].num_nodes;
        }
        owned_cell_nodes_flat.resize(static_cast<std::size_t>(owned_cell_offsets.back()));

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
        } // end loop over all my cells
        release_storage(cell_recv_buf);
    }
    const auto gid_bounds = std::minmax_element(owned_cell_gids.begin(), owned_cell_gids.end());
    GidIndex owned_gid_to_local(owned_cell_gids.empty() ? 0 : *gid_bounds.first,
                               owned_cell_gids.empty() ? -1 : *gid_bounds.second,
                               owned_cell_gids.size());
    for (LocalIndex i = 0; i < mp.n_own; ++i)
        owned_gid_to_local.insert(owned_cell_gids[static_cast<std::size_t>(i)], i);

    // -------------------------------------------------------------------------
    // Step 3: Migrate Faces & Detect Ghost Cells
    // -------------------------------------------------------------------------
    std::vector<FaceMigrateMsg> face_recv_buf;
    std::vector<LocalFaceTemp> local_faces;
    if (nprocs == 1) {
        local_faces.resize(faces.size());
        for (std::size_t i = 0; i < faces.size(); ++i) {
            const auto& f = faces[i];
            local_faces[i] = {
                owned_gid_to_local.at(f.cell_a),
                f.cell_b == kInvalidGlobal ? kInvalidLocal : owned_gid_to_local.at(f.cell_b),
                f.patch, static_cast<uint8_t>(f.lface_a)
            };
        }
        release_storage(faces);
    } else {
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
        release_storage(faces);
        release_storage(target_b_resolved);

        // perform exchange, after that step each rank get their faces
        exchange_and_release(comm, nprocs, face_send_counts, face_send_buf, face_recv_buf);
    }

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
    exchange_and_release(comm, nprocs, ghost_req_counts, ghost_req_send, ghost_req_recv);

    // Prepare response buffer
    std::vector<GhostTopoRespMsg> ghost_resp_send(ghost_req_recv.size());
    for (std::size_t i = 0; i < ghost_req_recv.size(); ++i) {
        const auto& req = ghost_req_recv[i];
        const LocalIndex loc_c = owned_gid_to_local.at(req.cell_gid);
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

    release_storage(ghost_req_recv);

    std::vector<int> ghost_resp_send_counts(nprocs_sz, 0);
    MPI_Alltoall(ghost_req_counts.data(), 1, MPI_INT, ghost_resp_send_counts.data(), 1, MPI_INT, comm);

    std::vector<GhostTopoRespMsg> ghost_resp_recv;
    exchange_and_release(comm, nprocs, ghost_resp_send_counts, ghost_resp_send, ghost_resp_recv);

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


    release_storage(ghost_resp_recv);

    // -------------------------------------------------------------------------
    // Step 5: Collect Unique Node GIDs
    // -------------------------------------------------------------------------
    GlobalIndex node_lo = std::numeric_limits<GlobalIndex>::max();
    GlobalIndex node_hi = -1;
    for (const auto gid : owned_cell_nodes_flat) { node_lo = std::min(node_lo, gid); node_hi = std::max(node_hi, gid); }
    for (const auto gid : ghost_cell_nodes_flat) { node_lo = std::min(node_lo, gid); node_hi = std::max(node_hi, gid); }
    if (node_hi < node_lo) node_lo = 0;
    const auto expected_nodes = std::max<std::size_t>(static_cast<std::size_t>(mp.n_cells), 1);
    GidIndex node_gid_to_local(node_lo, node_hi, expected_nodes);
    mp.node_gid.clear();
    if (node_gid_to_local.is_dense()) {
        std::size_t owned_count = 0, total_count = 0;
        for (const auto gid : owned_cell_nodes_flat)
            if (node_gid_to_local.insert(gid, 0)) ++owned_count;
        total_count = owned_count;
        for (const auto gid : ghost_cell_nodes_flat)
            if (node_gid_to_local.insert(gid, 1)) ++total_count;
        mp.n_nodes_own = static_cast<LocalIndex>(owned_count);
        mp.n_nodes = static_cast<LocalIndex>(total_count);
        mp.node_gid.resize(total_count);
        node_gid_to_local.write_dense_node_ids(mp.node_gid, owned_count);
    } else {
        mp.node_gid.reserve(expected_nodes);
        for (const auto gid : owned_cell_nodes_flat)
            if (node_gid_to_local.insert(gid, 0)) mp.node_gid.push_back(gid);
        mp.n_nodes_own = static_cast<LocalIndex>(mp.node_gid.size());
        for (const auto gid : ghost_cell_nodes_flat)
            if (node_gid_to_local.insert(gid, 0)) mp.node_gid.push_back(gid);
        mp.n_nodes = static_cast<LocalIndex>(mp.node_gid.size());
        const auto split = mp.node_gid.begin() + mp.n_nodes_own;
        std::sort(mp.node_gid.begin(), split);
        std::sort(split, mp.node_gid.end());
    }
    // Same numbering as before: sorted owned-used GIDs, then sorted ghost-only
    // GIDs. Dense ranges need no sort; sparse ranges sort only unique IDs.
    for (LocalIndex i = 0; i < mp.n_nodes; ++i)
        node_gid_to_local.assign(mp.node_gid[static_cast<std::size_t>(i)], i);

    // -------------------------------------------------------------------------
    // Step 6: Assemble Final Local Cells
    // -------------------------------------------------------------------------
    const auto owned_node_entries = owned_cell_nodes_flat.size();
    mp.cell_type = std::move(owned_cell_types);
    mp.cell_gid = std::move(owned_cell_gids);
    mp.cell_nodes_offsets = std::move(owned_cell_offsets);
    // Reserve the final size before appending ghosts: ordinary vector growth
    // could otherwise double the retained owned-cell metadata capacity.
    mp.cell_type.reserve(static_cast<std::size_t>(mp.n_cells));
    mp.cell_gid.reserve(static_cast<std::size_t>(mp.n_cells));
    mp.cell_nodes_offsets.reserve(static_cast<std::size_t>(mp.n_cells) + 1);
    mp.cell_type.insert(mp.cell_type.end(), ghost_cell_types.begin(), ghost_cell_types.end());
    mp.cell_gid.insert(mp.cell_gid.end(), ghost_cell_gids.begin(), ghost_cell_gids.end());
    mp.cell_donor.assign(static_cast<std::size_t>(mp.n_cells), -1);
    mp.cell_nodes_offsets.resize(static_cast<std::size_t>(mp.n_cells) + 1);
    for (LocalIndex i = 0; i < n_ghosts; ++i) {
        const auto slot = static_cast<std::size_t>(i);
        mp.cell_donor[static_cast<std::size_t>(mp.n_own) + slot] = ghost_cell_donors[slot];
        mp.cell_nodes_offsets[static_cast<std::size_t>(mp.n_own) + slot + 1] =
            static_cast<LocalIndex>(owned_node_entries) + ghost_cell_offsets[slot + 1];
    }
    mp.cell_nodes.resize(owned_node_entries + ghost_cell_nodes_flat.size());
    for (std::size_t i = 0; i < owned_node_entries; ++i)
        mp.cell_nodes[i] = node_gid_to_local.at(owned_cell_nodes_flat[i]);
    release_storage(owned_cell_nodes_flat);
    for (std::size_t i = 0; i < ghost_cell_nodes_flat.size(); ++i)
        mp.cell_nodes[owned_node_entries + i] = node_gid_to_local.at(ghost_cell_nodes_flat[i]);
    release_storage(ghost_cell_nodes_flat);
    release_storage(ghost_cell_gids);
    release_storage(ghost_cell_types);
    release_storage(ghost_cell_donors);
    release_storage(ghost_cell_offsets);
    node_gid_to_local.release();

    // Fetch coordinates only after global connectivity and its lookup are gone.
    const GlobalIndex my_node_start = m.node_displ[static_cast<std::size_t>(rank)];
    if (nprocs == 1 && mp.node_gid.size() == m.my_node_coords_x.size() &&
        (mp.node_gid.empty() || mp.node_gid.front() == my_node_start)) {
        // All original nodes are used in their original sorted order.
        mp.node_x = std::move(m.my_node_coords_x);
        mp.node_y = std::move(m.my_node_coords_y);
        mp.node_z = std::move(m.my_node_coords_z);
    } else {
        mp.node_x.resize(static_cast<std::size_t>(mp.n_nodes));
        mp.node_y.resize(static_cast<std::size_t>(mp.n_nodes));
        mp.node_z.resize(static_cast<std::size_t>(mp.n_nodes));
        std::vector<int> node_req_counts(nprocs_sz, 0);
        for (LocalIndex i = 0; i < mp.n_nodes; ++i) {
            const GlobalIndex nid = mp.node_gid[static_cast<std::size_t>(i)];
            const int owner = find_owner_rank(nid, m.node_displ);
            if (owner == rank) {
                const auto raw = static_cast<std::size_t>(nid - my_node_start);
                const auto slot = static_cast<std::size_t>(i);
                mp.node_x[slot] = m.my_node_coords_x[raw];
                mp.node_y[slot] = m.my_node_coords_y[raw];
                mp.node_z[slot] = m.my_node_coords_z[raw];
            } else ++node_req_counts[static_cast<std::size_t>(owner)];
        }

        std::vector<int> node_req_sdispls(nprocs_sz + 1, 0);
        for (std::size_t i = 0; i < nprocs_sz; ++i) {
            node_req_sdispls[i + 1] = node_req_sdispls[i] + node_req_counts[i];
        }
        std::vector<NodeCoordReqMsg> node_req_send(static_cast<std::size_t>(node_req_sdispls.back()));
        std::vector<int> node_req_curs = node_req_sdispls;

        for (LocalIndex i = 0; i < mp.n_nodes; ++i) {
            const GlobalIndex nid = mp.node_gid[static_cast<std::size_t>(i)];
            const int owner = find_owner_rank(nid, m.node_displ);
            if (owner == rank) continue;
            node_req_send[static_cast<std::size_t>(node_req_curs[static_cast<std::size_t>(owner)]++)] = NodeCoordReqMsg{
                nid, i
            };
        }

        std::vector<NodeCoordReqMsg> node_req_recv;
        exchange_and_release(comm, nprocs, node_req_counts, node_req_send, node_req_recv);


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

        release_storage(node_req_recv);

        // Free raw coordinates
        release_storage(m.my_node_coords_x);
        release_storage(m.my_node_coords_y);
        release_storage(m.my_node_coords_z);

        std::vector<int> node_resp_send_counts(nprocs_sz, 0);
        MPI_Alltoall(node_req_counts.data(), 1, MPI_INT, node_resp_send_counts.data(), 1, MPI_INT, comm);

        std::vector<NodeCoordRespMsg> node_resp_recv;
        exchange_and_release(comm, nprocs, node_resp_send_counts, node_resp_send, node_resp_recv);


        for (const auto& resp : node_resp_recv) {
            const auto slot = static_cast<std::size_t>(resp.local_slot);
            mp.node_x[slot] = resp.x;
            mp.node_y[slot] = resp.y;
            mp.node_z[slot] = resp.z;
        }
    }

    // -------------------------------------------------------------------------
    // Step 7: Assemble & Sort Faces
    // -------------------------------------------------------------------------
    if (nprocs > 1) local_faces.resize(face_recv_buf.size());

    // loop over all my faces
    for (std::size_t i = 0; i < face_recv_buf.size(); ++i) {
        const auto& f = face_recv_buf[i];
        LocalFaceTemp lf{};
        lf.owner = owned_gid_to_local.at(f.owner_gid);

        if (f.neigh_gid == kInvalidGlobal) {
            lf.neigh = kInvalidLocal;
        } else if (f.donor_rank == -1) {
            lf.neigh = owned_gid_to_local.at(f.neigh_gid);
        } else {
            lf.neigh = ghost_gid_to_local.at(f.neigh_gid);
        }

        lf.patch = f.patch;

        lf.lface = f.lface;

        local_faces[i] = lf;
    } // end loop over all my faces

    release_storage(face_recv_buf);
    release_storage(ghost_gid_to_local);
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
        const auto cell_type = static_cast<std::size_t>(mp.cell_type[static_cast<std::size_t>(local_faces[i].owner)]);
        const auto count = kFaceNodes[cell_type][local_faces[i].lface];
        mp.face_type[i] = count == 3 ? CellType::TRI : CellType::QUAD;
        mp.face_nodes_offsets[i + 1] = mp.face_nodes_offsets[i] + count;
    }

    mp.face_nodes.resize(static_cast<std::size_t>(mp.face_nodes_offsets.back()));
    for (std::size_t i = 0; i < n_faces_sz; ++i) {
        const auto& f = local_faces[i];
        const auto owner = static_cast<std::size_t>(f.owner);
        const auto cell_type = static_cast<std::size_t>(mp.cell_type[owner]);
        const auto cell_off = static_cast<std::size_t>(mp.cell_nodes_offsets[owner]);
        const auto face_off = static_cast<std::size_t>(mp.face_nodes_offsets[i]);
        const int count = kFaceNodes[cell_type][f.lface];
        for (int k = 0; k < count; ++k)
            mp.face_nodes[face_off + static_cast<std::size_t>(k)] =
                mp.cell_nodes[cell_off + static_cast<std::size_t>(kFaceTable[cell_type][f.lface][k])];
    }
    release_storage(local_faces);

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
    exchange_and_release(comm, nprocs, hs_send_counts, hs_send_buf, hs_recv_buf);

    for (std::size_t i = 0; i < n_nb; ++i) {
        const int neigh = mp.nb_ranks[i];
        mp.send_offsets[i + 1] = mp.send_offsets[i] + hs_recv_counts[static_cast<std::size_t>(neigh)];
    }

    mp.send_owned_local.resize(hs_recv_buf.size());
    for (std::size_t i = 0; i < hs_recv_buf.size(); ++i) {
        mp.send_owned_local[i] = owned_gid_to_local.at(hs_recv_buf[i].cell_gid);
    }


    release_storage(hs_recv_buf);
    release_storage(unique_ghosts);
    owned_gid_to_local.release();

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
    std::int64_t local_counts[2] = {
        static_cast<std::int64_t>(mp.n_faces),
        static_cast<std::int64_t>(total_local_bfaces)
    };
    std::int64_t global_counts[2] = {0, 0};
    MPI_Allreduce(local_counts, global_counts, 2, MPI_INT64_T, MPI_SUM, comm);

    mp.n_faces_g = static_cast<GlobalIndex>(global_counts[0]) - pr.global_edge_cut;
    mp.n_bfaces_g = static_cast<GlobalIndex>(global_counts[1]);

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