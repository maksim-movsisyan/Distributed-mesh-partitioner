#include "cfd/mesh/faces.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "cfd/core/types.hpp"
#include "cfd/mesh/cgnstables.hpp"
#include "cfd/mpi/log.hpp"

namespace cfd::mesh {
namespace {

struct alignas(8) HalfFaceMsg {
    FaceKey key;
    GlobalIndex cell_id = kInvalidGlobalIndex;
    PatchId patch = kInvalidPatchId;
    std::uint8_t lface = 0;
    std::uint8_t is_surf = 0;
    bool operator<(const HalfFaceMsg& other) const noexcept { return key < other.key; }
};

struct alignas(8) DualEdgeMsg {
    GlobalIndex cell_u = kInvalidGlobalIndex;
    GlobalIndex cell_v = kInvalidGlobalIndex;
};

// Target average half-face send-buffer size per rank per round. Receive sizes
// depend on the key distribution; this is not a cap on total process memory.
#ifndef CFD_FACES_BATCH_BYTES
#define CFD_FACES_BATCH_BYTES (128ULL * 1024ULL * 1024ULL)
#endif
constexpr std::uint64_t kBatchBytes = CFD_FACES_BATCH_BYTES;
static_assert(kBatchBytes >= sizeof(HalfFaceMsg), "Face batch buffer is too small");

// Optional comparison with the actual FaceKeyHash supplied by faces.hpp.
// Enable with -DCFD_FACES_REPORT_OLD_HASH=1 to diagnose legacy rank imbalance.
#ifndef CFD_FACES_REPORT_OLD_HASH
#define CFD_FACES_REPORT_OLD_HASH 0
#endif

constexpr std::uint64_t kSurfaceToken = UINT64_C(1) << 63;

template<class T>
void release_storage(std::vector<T>& data) { std::vector<T>().swap(data); }

[[noreturn]] void fail(MPI_Comm comm, const char* message) {
    mpi::fatal(comm, message);
    std::abort();
}

int find_owner_rank(GlobalIndex gid, const std::vector<GlobalIndex>& displ) noexcept {
    return static_cast<int>(std::upper_bound(displ.begin(), displ.end(), gid) - displ.begin() - 1);
}

// Mix all four slots and avalanche the result before rank/bucket reduction.
// Direct FNV % 2^k can expose weak low bits on regularly numbered quad faces.
std::uint64_t routing_hash(const FaceKey& key) noexcept {
    std::uint64_t h = UINT64_C(14695981039346656037);
    for (const auto node : key.v) {
        h ^= static_cast<std::uint64_t>(node);
        h *= UINT64_C(1099511628211);
    }
    h ^= h >> 33;
    h *= UINT64_C(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h *= UINT64_C(0xc4ceb9fe1a85ec53);
    return h ^ (h >> 33);
}

FaceKey volume_face_key(const RawMesh& m, std::size_t cell, std::size_t face) noexcept {
    FaceKey key{};
    key.v.fill(kInvalidGlobalIndex);
    const auto type = static_cast<std::size_t>(m.ctype[cell]);
    const auto offset = static_cast<std::size_t>(m.cnodes_offsets[cell]);
    const int count = kFaceNodes[type][face];
    for (int i = 0; i < count; ++i)
        key.v[static_cast<std::size_t>(i)] =
            m.cnodes[offset + static_cast<std::size_t>(kFaceTable[type][face][i])];
    auto swap_if = [&](std::size_t a, std::size_t b) {
        if (key.v[a] > key.v[b]) std::swap(key.v[a], key.v[b]);
    };
    if (count == 3) {
        swap_if(0, 1); swap_if(1, 2); swap_if(0, 1);
    } else if (count == 4) {
        swap_if(0, 1); swap_if(2, 3); swap_if(0, 2); swap_if(1, 3); swap_if(1, 2);
    }
    return key;
}

template<class Visit>
void visit_half_faces(const RawMesh& m, Visit&& visit) {
    const auto cells = static_cast<std::size_t>(m.n_local_cells());
    for (std::size_t c = 0; c < cells; ++c) {
        const auto type = static_cast<std::size_t>(m.ctype[c]);
        for (int f = 0; f < kFacesPerType[type]; ++f) {
            const auto token = (static_cast<std::uint64_t>(c) << 3) | static_cast<std::uint64_t>(f);
            visit(volume_face_key(m, c, static_cast<std::size_t>(f)), token);
        }
    }
    for (std::size_t i = 0; i < m.surf_elems.size(); ++i)
        visit(m.surf_elems[i].key, kSurfaceToken | static_cast<std::uint64_t>(i));
}

HalfFaceMsg unpack_half_face(const RawMesh& m, std::uint64_t token) {
    if (token & kSurfaceToken) {
        const auto& surf = m.surf_elems[static_cast<std::size_t>(token & ~kSurfaceToken)];
        return {surf.key, kInvalidGlobalIndex, surf.patch, 0, 1};
    }
    const auto cell = static_cast<std::size_t>(token >> 3);
    const auto face = static_cast<std::uint8_t>(token & 7);
    return {volume_face_key(m, cell, face), m.global_cell_id(static_cast<LocalIndex>(cell)),
            kInvalidPatchId, face, 0};
}

// Checked, blocking POD exchange. Byte counts and byte displacements stay
// within classic MPI_Alltoallv's int limits. Release the send array immediately.
template<class T>
void exchange(MPI_Comm comm, int nprocs, const std::vector<int>& counts,
              std::vector<T>& send, std::vector<T>& recv) {
    static_assert(std::is_trivially_copyable<T>::value, "MPI message must be trivially copyable");
    if (nprocs == 1) {
        recv = std::move(send);
        return;
    }
    const auto p = static_cast<std::size_t>(nprocs);
    std::vector<int> recv_counts(p), send_bytes(p), recv_bytes(p), send_disp(p), recv_disp(p);
    MPI_Alltoall(counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);
    std::size_t ns = 0, nr = 0;
    const auto max_items = static_cast<std::size_t>(std::numeric_limits<int>::max()) / sizeof(T);
    for (std::size_t i = 0; i < p; ++i) {
        if (counts[i] < 0 || recv_counts[i] < 0 ||
            static_cast<std::size_t>(counts[i]) > max_items - ns ||
            static_cast<std::size_t>(recv_counts[i]) > max_items - nr)
            fail(comm, "Face exchange exceeds MPI int byte-count limits; reduce CFD_FACES_BATCH_BYTES");
        send_disp[i] = static_cast<int>(ns * sizeof(T));
        recv_disp[i] = static_cast<int>(nr * sizeof(T));
        send_bytes[i] = static_cast<int>(static_cast<std::size_t>(counts[i]) * sizeof(T));
        recv_bytes[i] = static_cast<int>(static_cast<std::size_t>(recv_counts[i]) * sizeof(T));
        ns += static_cast<std::size_t>(counts[i]);
        nr += static_cast<std::size_t>(recv_counts[i]);
    }
    assert(ns == send.size());
    recv.resize(nr);
    MPI_Alltoallv(send.empty() ? nullptr : send.data(), send_bytes.data(), send_disp.data(), MPI_BYTE,
                  recv.empty() ? nullptr : recv.data(), recv_bytes.data(), recv_disp.data(), MPI_BYTE, comm);
    release_storage(send);
}

// Sorted equal-key groups are complete within one batch because routing and
// batching both depend only on the key. Visit twice to count and then pack;
// do not materialize an additional array of matched FaceRec objects.
template<class Visit>
void visit_matches(const std::vector<HalfFaceMsg>& messages, const RawMesh& m, Visit&& visit) {
    for (std::size_t i = 0; i < messages.size();) {
        std::size_t end = i + 1;
        while (end < messages.size() && messages[i].key == messages[end].key) ++end;
        const HalfFaceMsg* volumes[2] = {nullptr, nullptr};
        int count = 0;
        PatchId patch = kInvalidPatchId;
        for (std::size_t k = i; k < end; ++k) {
            const auto& item = messages[k];
            if (item.is_surf) patch = item.patch;
            else {
                if (count == 2) fail(m.comm, "HPC Error: Non-manifold mesh detected (>2 cells sharing a face)!");
                volumes[count++] = &item;
            }
        }
        if (count != 0) {
            if (count == 2 && volumes[0]->cell_id > volumes[1]->cell_id)
                std::swap(volumes[0], volumes[1]);
            const auto& a = *volumes[0];
            const auto b_gid = count == 2 ? volumes[1]->cell_id : kInvalidGlobalIndex;
            const auto b_face = count == 2 ? volumes[1]->lface : static_cast<std::uint8_t>(255);
            const FaceRec rec{a.key, a.cell_id, b_gid, count == 2 ? kInvalidPatchId : patch, a.lface, b_face};
            const int owner_a = find_owner_rank(a.cell_id, m.cell_displ);
            const int owner_b = count == 2 ? find_owner_rank(b_gid, m.cell_displ) : -1;
            visit(rec, owner_a, owner_b);
        }
        i = end;
    }
}

std::vector<std::size_t> prefix(const std::vector<int>& counts) {
    std::vector<std::size_t> result(counts.size() + 1, 0);
    for (std::size_t i = 0; i < counts.size(); ++i)
        result[i + 1] = result[i] + static_cast<std::size_t>(counts[i]);
    return result;
}

} // namespace

BuildFacesResult build_faces(const RawMesh& m) {
    const int nprocs = m.nprocs;
    const int rank = m.rank;
    const MPI_Comm comm = m.comm;
    const auto p = static_cast<std::size_t>(nprocs);
    const LocalIndex n_loc_cells = m.n_local_cells();

    // Validate the fixed-size message format and count without generating keys.
    if (static_cast<std::uint64_t>(n_loc_cells) >= (kSurfaceToken >> 3) ||
        m.surf_elems.size() >= kSurfaceToken)
        fail(comm, "Local mesh is too large for face source tokens");
    std::uint64_t local_items = static_cast<std::uint64_t>(m.surf_elems.size());
    for (std::size_t c = 0; c < static_cast<std::size_t>(n_loc_cells); ++c) {
        if (!is_volume_type(m.ctype[c])) fail(comm, "Cell type must be 3D");
        const auto type = static_cast<std::size_t>(m.ctype[c]);
        const int count = kFacesPerType[type];
        if (count < 1 || count > 8) fail(comm, "Unsupported volume face count");
        for (int f = 0; f < count; ++f)
            if (kFaceNodes[type][f] != 3 && kFaceNodes[type][f] != 4)
                fail(comm, "Only triangular and quadrilateral faces are supported");
        local_items += static_cast<std::uint64_t>(count);
    }
    std::uint64_t max_items = 0;
    MPI_Allreduce(&local_items, &max_items, 1, MPI_UINT64_T, MPI_MAX, comm);
    constexpr auto per_batch = kBatchBytes / sizeof(HalfFaceMsg);
    const auto rounds = std::max<std::uint64_t>(1, max_items / per_batch + (max_items % per_batch != 0));
    if (rounds > std::numeric_limits<std::size_t>::max() / p)
        fail(comm, "Face bucket count exceeds addressable storage");
    const auto buckets = static_cast<std::size_t>(rounds) * p;

    // Store 8-byte source references, not all 48-byte half-face messages.
    // Bucket b belongs to round b/P and rendezvous rank b%P.
    std::vector<std::size_t> offsets(buckets + 1, 0);
#if CFD_FACES_REPORT_OLD_HASH
    std::vector<std::uint64_t> legacy_local(p, 0), legacy_global(p, 0);
#endif
    visit_half_faces(m, [&](const FaceKey& key, std::uint64_t) {
        ++offsets[static_cast<std::size_t>(routing_hash(key) % buckets) + 1];
#if CFD_FACES_REPORT_OLD_HASH
        ++legacy_local[FaceKeyHash{}(key) % p];
#endif
    });
#if CFD_FACES_REPORT_OLD_HASH
    MPI_Allreduce(legacy_local.data(), legacy_global.data(), nprocs, MPI_UINT64_T, MPI_SUM, comm);
    const auto legacy_bounds = std::minmax_element(legacy_global.begin(), legacy_global.end());
    mpi::log_stat("INFO[Faces]: Legacy hash messages/rank min=%llu max=%llu (max on rank %d)",
                  static_cast<unsigned long long>(*legacy_bounds.first),
                  static_cast<unsigned long long>(*legacy_bounds.second),
                  static_cast<int>(legacy_bounds.second - legacy_global.begin()));
#endif
    for (std::size_t b = 0; b < buckets; ++b) offsets[b + 1] += offsets[b];
    std::vector<std::uint64_t> sources(offsets.back());
    {
        auto cursor = offsets;
        visit_half_faces(m, [&](const FaceKey& key, std::uint64_t token) {
            const auto bucket = static_cast<std::size_t>(routing_hash(key) % buckets);
            sources[cursor[bucket]++] = token;
        });
    }

    std::vector<std::vector<FaceRec>> face_chunks;
    std::vector<std::vector<DualEdgeMsg>> edge_chunks;
    face_chunks.reserve(static_cast<std::size_t>(rounds));
    edge_chunks.reserve(static_cast<std::size_t>(rounds));
    std::size_t total_faces = 0;
    std::uint64_t received_total = 0, largest_receive = 0;

    for (std::size_t round = 0; round < static_cast<std::size_t>(rounds); ++round) {
        std::vector<int> send_counts(p, 0);
        const auto bucket0 = round * p;
        for (std::size_t dest = 0; dest < p; ++dest) {
            const auto count = offsets[bucket0 + dest + 1] - offsets[bucket0 + dest];
            if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                fail(comm, "Face bucket exceeds MPI count limits; reduce CFD_FACES_BATCH_BYTES");
            send_counts[dest] = static_cast<int>(count);
        }
        const auto begin = offsets[bucket0];
        const auto end = offsets[bucket0 + p];
        std::vector<HalfFaceMsg> send(end - begin);
        for (std::size_t i = begin; i < end; ++i) send[i - begin] = unpack_half_face(m, sources[i]);
        std::vector<HalfFaceMsg> received;
        exchange(comm, nprocs, send_counts, send, received);
        received_total += received.size();
        largest_receive = std::max<std::uint64_t>(largest_receive, received.size());
        std::sort(received.begin(), received.end());

        std::vector<int> face_counts(p, 0), edge_counts(p, 0);
        visit_matches(received, m, [&](const FaceRec&, int a, int b) {
            ++face_counts[static_cast<std::size_t>(a)];
            if (b != -1 && a != b) ++edge_counts[static_cast<std::size_t>(b)];
        });
        auto face_cursor = prefix(face_counts);
        auto edge_cursor = prefix(edge_counts);
        std::vector<FaceRec> face_send(face_cursor.back());
        std::vector<DualEdgeMsg> edge_send(edge_cursor.back());
        visit_matches(received, m, [&](const FaceRec& rec, int a, int b) {
            face_send[face_cursor[static_cast<std::size_t>(a)]++] = rec;
            if (b != -1 && a != b)
                edge_send[edge_cursor[static_cast<std::size_t>(b)]++] = {rec.cell_b, rec.cell_a};
        });
        release_storage(received);
        face_chunks.emplace_back();
        exchange(comm, nprocs, face_counts, face_send, face_chunks.back());
        total_faces += face_chunks.back().size();
        edge_chunks.emplace_back();
        exchange(comm, nprocs, edge_counts, edge_send, edge_chunks.back());
    }
    release_storage(sources);
    release_storage(offsets);

    BuildFacesResult result;
    // reserve + append avoids zero-initializing/touching the entire final array
    // while all chunks are still resident. Release each chunk after copying it.
    if (face_chunks.size() == 1) result.faces = std::move(face_chunks.front());
    else {
        result.faces.reserve(total_faces);
        for (auto& chunk : face_chunks) {
            result.faces.insert(result.faces.end(), chunk.begin(), chunk.end());
            release_storage(chunk);
        }
    }
    release_storage(face_chunks);

    const GlobalIndex cell_start = m.cell_displ[static_cast<std::size_t>(rank)];
    const GlobalIndex cell_end = m.cell_displ[static_cast<std::size_t>(rank) + 1];
    result.graph.offsets.assign(static_cast<std::size_t>(n_loc_cells) + 1, 0);
    GlobalIndex local_ifaces = 0, local_bfaces = 0;
    for (const auto& f : result.faces) {
        if (f.cell_b == kInvalidGlobalIndex) { ++local_bfaces; continue; }
        ++local_ifaces;
        const auto u = static_cast<std::size_t>(f.cell_a - cell_start);
        assert(u < static_cast<std::size_t>(n_loc_cells));
        ++result.graph.offsets[u + 1];
        if (f.cell_b >= cell_start && f.cell_b < cell_end)
            ++result.graph.offsets[static_cast<std::size_t>(f.cell_b - cell_start) + 1];
    }
    for (const auto& chunk : edge_chunks)
        for (const auto& e : chunk) {
            const auto u = static_cast<std::size_t>(e.cell_u - cell_start);
            assert(u < static_cast<std::size_t>(n_loc_cells));
            ++result.graph.offsets[u + 1];
        }
    for (std::size_t c = 0; c < static_cast<std::size_t>(n_loc_cells); ++c) {
        if (result.graph.offsets[c] > std::numeric_limits<LocalIndex>::max() - result.graph.offsets[c + 1])
            fail(comm, "Local dual-graph CSR exceeds LocalIndex range");
        result.graph.offsets[c + 1] += result.graph.offsets[c];
    }
    result.graph.adj.resize(static_cast<std::size_t>(result.graph.offsets.back()));
    {
        std::vector<LocalIndex> head(result.graph.offsets.begin(), result.graph.offsets.end() - 1);
        for (const auto& f : result.faces) {
            if (f.cell_b == kInvalidGlobalIndex) continue;
            const auto u = static_cast<std::size_t>(f.cell_a - cell_start);
            result.graph.adj[static_cast<std::size_t>(head[u]++)] = f.cell_b;
            if (f.cell_b >= cell_start && f.cell_b < cell_end) {
                const auto v = static_cast<std::size_t>(f.cell_b - cell_start);
                result.graph.adj[static_cast<std::size_t>(head[v]++)] = f.cell_a;
            }
        }
        for (auto& chunk : edge_chunks) {
            for (const auto& e : chunk) {
                const auto u = static_cast<std::size_t>(e.cell_u - cell_start);
                result.graph.adj[static_cast<std::size_t>(head[u]++)] = e.cell_v;
            }
            release_storage(chunk);
        }
    }
    release_storage(edge_chunks);
    for (LocalIndex c = 0; c < n_loc_cells; ++c) {
        const auto begin = result.graph.offsets[static_cast<std::size_t>(c)];
        const auto end = result.graph.offsets[static_cast<std::size_t>(c) + 1];
        std::sort(result.graph.adj.begin() + begin, result.graph.adj.begin() + end);
    }

    const std::int64_t local_stats[4] = {static_cast<std::int64_t>(result.faces.size()),
        static_cast<std::int64_t>(local_bfaces), static_cast<std::int64_t>(local_ifaces),
        static_cast<std::int64_t>(result.graph.adj.size())};
    std::int64_t global_stats[4] = {};
    MPI_Allreduce(local_stats, global_stats, 4, MPI_INT64_T, MPI_SUM, comm);
    result.stats.n_faces_g = static_cast<GlobalIndex>(global_stats[0]);
    result.stats.n_bfaces_g = static_cast<GlobalIndex>(global_stats[1]);
    result.stats.n_ifaces_g = static_cast<GlobalIndex>(global_stats[2]);
    result.stats.n_dg_edges = static_cast<GlobalIndex>(global_stats[3]);

    std::uint64_t min_received = 0, max_received = 0, max_batch = 0;
    MPI_Allreduce(&received_total, &min_received, 1, MPI_UINT64_T, MPI_MIN, comm);
    MPI_Allreduce(&received_total, &max_received, 1, MPI_UINT64_T, MPI_MAX, comm);
    MPI_Allreduce(&largest_receive, &max_batch, 1, MPI_UINT64_T, MPI_MAX, comm);
    mpi::log_stat("INFO[Faces]: Rendezvous batches=%llu, messages/rank min=%llu max=%llu, largest receive batch=%llu",
                  static_cast<unsigned long long>(rounds), static_cast<unsigned long long>(min_received),
                  static_cast<unsigned long long>(max_received), static_cast<unsigned long long>(max_batch));
    mpi::log_stat("INFO[Faces]: Total face count=%lld, Boundary face count=%lld, Interior face count=%lld",
                  static_cast<long long>(result.stats.n_faces_g), static_cast<long long>(result.stats.n_bfaces_g),
                  static_cast<long long>(result.stats.n_ifaces_g));
    mpi::log_stat("INFO[Dual Graph]: Total edges=%lld", static_cast<long long>(result.stats.n_dg_edges));
    return result;
}

} // namespace cfd::mesh