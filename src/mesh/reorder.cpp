#include "cfd/mesh/reorder.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

#include "cfd/core/types.hpp"
#include "cfd/mesh/cgnstables.hpp"

namespace cfd::mesh {

namespace {

constexpr LocalIndex kInvalidLocal = static_cast<LocalIndex>(-1);
constexpr PatchId kInvalidPatch = static_cast<PatchId>(-1);

// =============================================================================
// Step A: Dual Graph Construction (Owned Cells Only)
// =============================================================================

struct DualGraphCSR {
    std::vector<LocalIndex> offsets;
    std::vector<LocalIndex> adj;
    std::vector<LocalIndex> degrees;
};

DualGraphCSR build_owned_dual_graph(const MeshPart& mp) {
    // get number of local cells
    const LocalIndex n_own = mp.n_own;
    const std::size_t n_own_sz = static_cast<std::size_t>(n_own);

    // initialize result
    DualGraphCSR g;
    g.degrees.assign(n_own_sz, 0);

    // Pass 1: Count symmetric degrees between owned cells
    for (LocalIndex f = 0; f < mp.n_faces; ++f) {
        const std::size_t f_sz = static_cast<std::size_t>(f);

        // get local cell indices
        const LocalIndex u = mp.face_owner[f_sz];
        const LocalIndex v = mp.face_neigh[f_sz];

        // only internal faces (real edges in dual graph)
        if (v != kInvalidLocal && v < n_own) {
            ++g.degrees[static_cast<std::size_t>(u)];
            ++g.degrees[static_cast<std::size_t>(v)];
        }
    } // end loop over all faces

    g.offsets.assign(n_own_sz + 1, 0);
    for (std::size_t i = 0; i < n_own_sz; ++i) {
        g.offsets[i + 1] = g.offsets[i] + g.degrees[i];
    }

    g.adj.resize(static_cast<std::size_t>(g.offsets.back()));
    std::vector<LocalIndex> curs = g.offsets;

    // Pass 2: Fill flat adjacency list
    for (LocalIndex f = 0; f < mp.n_faces; ++f) {
        const std::size_t f_sz = static_cast<std::size_t>(f);
        const LocalIndex u = mp.face_owner[f_sz];
        const LocalIndex v = mp.face_neigh[f_sz];

        if (v != kInvalidLocal && v < n_own) {
            const std::size_t u_sz = static_cast<std::size_t>(u);
            const std::size_t v_sz = static_cast<std::size_t>(v);
            g.adj[static_cast<std::size_t>(curs[u_sz]++)] = v;
            g.adj[static_cast<std::size_t>(curs[v_sz]++)] = u;
        }
    }

    return g;
}



// =============================================================================
// Step B.1: Reverse Cuthill-McKee (RCM) Permutation
// =============================================================================

// Finds a pseudo-peripheral starting node using two-stage BFS
LocalIndex find_pseudo_peripheral(const DualGraphCSR& g, LocalIndex root, std::vector<uint8_t>& visited_global) {
    LocalIndex best_start = root;
    int max_depth = 0;

    for (int iter = 0; iter < 3; ++iter) {
        std::queue<std::pair<LocalIndex, int>> q;
        std::vector<uint8_t> visited_local(g.degrees.size(), 0);

        q.push({best_start, 0});
        visited_local[static_cast<std::size_t>(best_start)] = 1;

        LocalIndex furthest = best_start;
        int deepest = 0;
        LocalIndex min_deg = g.degrees[static_cast<std::size_t>(best_start)];

        while (!q.empty()) {
            auto [u, depth] = q.front();
            q.pop();

            if (depth > deepest || (depth == deepest && g.degrees[static_cast<std::size_t>(u)] < min_deg)) {
                deepest = depth;
                furthest = u;
                min_deg = g.degrees[static_cast<std::size_t>(u)];
            }

            const auto u_sz = static_cast<std::size_t>(u);
            const LocalIndex start_off = g.offsets[u_sz];
            const LocalIndex end_off   = g.offsets[u_sz + 1];

            for (LocalIndex k = start_off; k < end_off; ++k) {
                const LocalIndex v = g.adj[static_cast<std::size_t>(k)];
                const auto v_sz = static_cast<std::size_t>(v);
                if (!visited_local[v_sz] && !visited_global[v_sz]) {
                    visited_local[v_sz] = 1;
                    q.push({v, depth + 1});
                }
            }
        }

        if (deepest <= max_depth) break;
        max_depth = deepest;
        best_start = furthest;
    }

    return best_start;
}

std::vector<LocalIndex> compute_rcm_order(const MeshPart& mp) {
    // get number of local cells (without ghosts)
    const LocalIndex n_own = mp.n_own;
    const std::size_t n_own_sz = static_cast<std::size_t>(n_own);

    // build dual graph
    DualGraphCSR g = build_owned_dual_graph(mp);
    std::vector<uint8_t> visited(n_own_sz, 0);
    std::vector<LocalIndex> cm_order;
    cm_order.reserve(n_own_sz);

    // Process all connected components (loop over all local cells)
    for (LocalIndex i = 0; i < n_own; ++i) {
        const std::size_t i_sz = static_cast<std::size_t>(i);
        if (visited[i_sz]) continue;

        // find start node
        const LocalIndex start_node = find_pseudo_peripheral(g, i, visited);

        std::queue<LocalIndex> q;
        q.push(start_node);
        visited[static_cast<std::size_t>(start_node)] = 1;

        while (!q.empty()) {
            const LocalIndex u = q.front();
            q.pop();
            cm_order.push_back(u);

            const std::size_t u_sz = static_cast<std::size_t>(u);
            const LocalIndex start_off = g.offsets[u_sz];
            const LocalIndex end_off = g.offsets[u_sz + 1];

            // Collect and sort unvisited neighbors by degree ascending
            std::vector<std::pair<LocalIndex, LocalIndex>> unvisited_nbs;
            for (LocalIndex k = start_off; k < end_off; ++k) {
                const LocalIndex v = g.adj[static_cast<std::size_t>(k)];
                const std::size_t v_sz = static_cast<std::size_t>(v);
                if (!visited[v_sz]) {
                    visited[v_sz] = 1;
                    unvisited_nbs.push_back({g.degrees[v_sz], v});
                }
            }

            std::sort(unvisited_nbs.begin(), unvisited_nbs.end());
            for (const auto& [deg, v] : unvisited_nbs) {
                q.push(v);
            }
        }
    }

    // Reverse Cuthill-McKee order
    std::reverse(cm_order.begin(), cm_order.end());
    return cm_order; // new2old mapping
}



// =============================================================================
// Step B.2: 3D Hilbert Space-Filling Curve (SFC) Algorithm
// =============================================================================

// Skilling's method for fast 3D coordinate-to-Hilbert index mapping (21 bits per axis = 63-bit key)
[[nodiscard]] inline uint64_t hilbert_index_3d(uint32_t x, uint32_t y, uint32_t z, int bits = 21) noexcept {
    uint32_t X[3] = {x, y, z};
    const uint32_t M = 1u << (bits - 1);

    // Inverse Gray code
    for (uint32_t q = M; q > 1; q >>= 1) {
        const uint32_t P = q - 1;
        for (int i = 0; i < 3; ++i) {
            if (X[i] & q) {
                X[0] ^= P;
            } else {
                const uint32_t t = (X[0] ^ X[i]) & P;
                X[0] ^= t;
                X[i] ^= t;
            }
        }
    }

    // Gray code
    for (int i = 1; i < 3; ++i) X[i] ^= X[i - 1];
    uint32_t t = 0;
    for (uint32_t q = M; q > 1; q >>= 1) {
        if (X[2] & q) t ^= (q - 1);
    }
    for (int i = 0; i < 3; ++i) X[i] ^= t;

    // Bit-interleaving across 3 dimensions to form the final 64-bit integer
    uint64_t key = 0;
    for (int i = bits - 1; i >= 0; --i) {
        const uint64_t bx = (X[0] >> i) & 1u;
        const uint64_t by = (X[1] >> i) & 1u;
        const uint64_t bz = (X[2] >> i) & 1u;
        key = (key << 3) | (bx << 2) | (by << 1) | bz;
    }

    return key;
}

std::vector<LocalIndex> compute_hilbert_order(const MeshPart& mp) {
    const auto n_own = mp.n_own;
    const auto n_own_sz = static_cast<std::size_t>(n_own);

    // 1. Compute Centroid Bounding Box
    double min_x = std::numeric_limits<double>::max();
    double min_y = std::numeric_limits<double>::max();
    double min_z = std::numeric_limits<double>::max();
    double max_x = -std::numeric_limits<double>::max();
    double max_y = -std::numeric_limits<double>::max();
    double max_z = -std::numeric_limits<double>::max();

    for (LocalIndex c = 0; c < n_own; ++c) {
        const auto c_sz = static_cast<std::size_t>(c);
        min_x = std::min(min_x, mp.cell_centroid_x[c_sz]);
        min_y = std::min(min_y, mp.cell_centroid_y[c_sz]);
        min_z = std::min(min_z, mp.cell_centroid_z[c_sz]);
        max_x = std::max(max_x, mp.cell_centroid_x[c_sz]);
        max_y = std::max(max_y, mp.cell_centroid_y[c_sz]);
        max_z = std::max(max_z, mp.cell_centroid_z[c_sz]);
    }

    // 2. Uniform Aspect Ratio Normalization to [0, 2^21 - 1]
    const double dx = max_x - min_x;
    const double dy = max_y - min_y;
    const double dz = max_z - min_z;
    double max_dim = std::max({dx, dy, dz});
    if (max_dim <= 1e-15) max_dim = 1.0;

    constexpr uint32_t kMaxCoord = (1u << 21) - 1;
    const double inv_scale = static_cast<double>(kMaxCoord) / max_dim;

    // 3. Compute 3D Hilbert Keys
    struct CellKey {
        uint64_t key;
        LocalIndex orig_idx;

        bool operator<(const CellKey& o) const noexcept {
            if (key != o.key) return key < o.key;
            return orig_idx < o.orig_idx; // Deterministic tie-breaker
        }
    };

    std::vector<CellKey> keys(n_own_sz);
    for (LocalIndex c = 0; c < n_own; ++c) {
        const auto c_sz = static_cast<std::size_t>(c);

        const auto ix = static_cast<uint32_t>(std::clamp((mp.cell_centroid_x[c_sz] - min_x) * inv_scale, 0.0, static_cast<double>(kMaxCoord)));
        const auto iy = static_cast<uint32_t>(std::clamp((mp.cell_centroid_y[c_sz] - min_y) * inv_scale, 0.0, static_cast<double>(kMaxCoord)));
        const auto iz = static_cast<uint32_t>(std::clamp((mp.cell_centroid_z[c_sz] - min_z) * inv_scale, 0.0, static_cast<double>(kMaxCoord)));

        keys[c_sz] = CellKey{hilbert_index_3d(ix, iy, iz, 21), c};
    }

    // 4. Sort along the space-filling curve
    std::sort(keys.begin(), keys.end());

    std::vector<LocalIndex> new2old(n_own_sz);
    for (std::size_t i = 0; i < n_own_sz; ++i) {
        new2old[i] = keys[i].orig_idx;
    }

    return new2old;
}



// =============================================================================
// Step C: In-Place Application of Permutation
// =============================================================================

void apply_cell_permutation(MeshPart& mp, const std::vector<LocalIndex>& new2old) {
    const LocalIndex n_own = mp.n_own;

    // 1. Build old2new mapping: old2new[old_c] = new_c
    std::vector<LocalIndex> old2new(static_cast<std::size_t>(mp.n_cells));
    for (LocalIndex new_c = 0; new_c < n_own; ++new_c) {
        old2new[static_cast<std::size_t>(new2old[static_cast<std::size_t>(new_c)])] = new_c;
    }
    // Ghost cells keep their identical index: [n_own, n_cells)
    for (LocalIndex g = n_own; g < mp.n_cells; ++g) {
        old2new[static_cast<std::size_t>(g)] = g;
    }



    // 2. Permute Owned Cell SoA arrays
    std::vector<CellType> new_cell_type(mp.cell_type.size());
    std::vector<GlobalIndex> new_cell_gid(mp.cell_gid.size());
    std::vector<double> new_cell_centroid_x(mp.cell_centroid_x.size());
    std::vector<double> new_cell_centroid_y(mp.cell_centroid_y.size());
    std::vector<double> new_cell_centroid_z(mp.cell_centroid_z.size());
    std::vector<double> new_cell_volume(mp.cell_volume.size());

    // Permute owned cells
    for (LocalIndex new_c = 0; new_c < n_own; ++new_c) {
        const std::size_t new_sz = static_cast<std::size_t>(new_c);
        const std::size_t old_sz = static_cast<std::size_t>(new2old[new_sz]);

        new_cell_type[new_sz] = mp.cell_type[old_sz];
        new_cell_gid[new_sz] = mp.cell_gid[old_sz];
        new_cell_centroid_x[new_sz] = mp.cell_centroid_x[old_sz];
        new_cell_centroid_y[new_sz] = mp.cell_centroid_y[old_sz];
        new_cell_centroid_z[new_sz] = mp.cell_centroid_z[old_sz];
        new_cell_volume[new_sz] = mp.cell_volume[old_sz];
    }

    // Copy ghost cells intact
    for (LocalIndex g = n_own; g < mp.n_cells; ++g) {
        const std::size_t g_sz = static_cast<std::size_t>(g);
        new_cell_type[g_sz] = mp.cell_type[g_sz];
        new_cell_gid[g_sz] = mp.cell_gid[g_sz];
        new_cell_centroid_x[g_sz] = mp.cell_centroid_x[g_sz];
        new_cell_centroid_y[g_sz] = mp.cell_centroid_y[g_sz];
        new_cell_centroid_z[g_sz] = mp.cell_centroid_z[g_sz];
        new_cell_volume[g_sz] = mp.cell_volume[g_sz];
    }

    mp.cell_type = std::move(new_cell_type);
    mp.cell_gid = std::move(new_cell_gid);
    mp.cell_centroid_x = std::move(new_cell_centroid_x);
    mp.cell_centroid_y = std::move(new_cell_centroid_y);
    mp.cell_centroid_z = std::move(new_cell_centroid_z);
    mp.cell_volume = std::move(new_cell_volume);



    // 3. Rebuild Cell-to-Nodes CSR Connectivity
    std::vector<LocalIndex> new_cell_nodes_offsets(mp.cell_nodes_offsets.size(), 0);
    for (LocalIndex new_c = 0; new_c < n_own; ++new_c) {
        const std::size_t new_sz = static_cast<std::size_t>(new_c);
        const std::size_t old_sz = static_cast<std::size_t>(new2old[new_sz]);
        const LocalIndex nnodes = mp.cell_nodes_offsets[old_sz + 1] - mp.cell_nodes_offsets[old_sz];
        new_cell_nodes_offsets[new_sz + 1] = new_cell_nodes_offsets[new_sz] + nnodes;
    }
    for (LocalIndex g = n_own; g < mp.n_cells; ++g) {
        const std::size_t g_sz = static_cast<std::size_t>(g);
        const LocalIndex nnodes = mp.cell_nodes_offsets[g_sz + 1] - mp.cell_nodes_offsets[g_sz];
        new_cell_nodes_offsets[g_sz + 1] = new_cell_nodes_offsets[g_sz] + nnodes;
    }

    std::vector<LocalIndex> new_cell_nodes(static_cast<std::size_t>(new_cell_nodes_offsets.back()));
    for (LocalIndex new_c = 0; new_c < n_own; ++new_c) {
        const std::size_t new_sz = static_cast<std::size_t>(new_c);
        const std::size_t old_sz = static_cast<std::size_t>(new2old[new_sz]);
        const LocalIndex old_off = mp.cell_nodes_offsets[old_sz];
        const LocalIndex new_off = new_cell_nodes_offsets[new_sz];
        const std::size_t nnodes = static_cast<std::size_t>(mp.cell_nodes_offsets[old_sz + 1] - old_off);

        for (std::size_t k = 0; k < nnodes; ++k) {
            new_cell_nodes[static_cast<std::size_t>(new_off) + k] =
                mp.cell_nodes[static_cast<std::size_t>(old_off) + k];
        }
    }
    for (LocalIndex g = n_own; g < mp.n_cells; ++g) {
        const std::size_t g_sz = static_cast<std::size_t>(g);
        const LocalIndex old_off = mp.cell_nodes_offsets[g_sz];
        const LocalIndex new_off = new_cell_nodes_offsets[g_sz];
        const std::size_t nnodes = static_cast<std::size_t>(mp.cell_nodes_offsets[g_sz + 1] - old_off);

        for (std::size_t k = 0; k < nnodes; ++k) {
            new_cell_nodes[static_cast<std::size_t>(new_off) + k] =
                mp.cell_nodes[static_cast<std::size_t>(old_off) + k];
        }
    }

    mp.cell_nodes_offsets = std::move(new_cell_nodes_offsets);
    mp.cell_nodes = std::move(new_cell_nodes);



    // 4. Update Communication Maps: send_owned_local
    for (auto& send_cell : mp.send_owned_local) {
        send_cell = old2new[static_cast<std::size_t>(send_cell)];
    }



    // 5. Update Face Connectivity & 3-Zone Re-sorting
    for (LocalIndex f = 0; f < mp.n_faces; ++f) {
        const std::size_t f_sz = static_cast<std::size_t>(f);
        mp.face_owner[f_sz] = old2new[static_cast<std::size_t>(mp.face_owner[f_sz])];
        if (mp.face_neigh[f_sz] != kInvalidLocal) {
            mp.face_neigh[f_sz] = old2new[static_cast<std::size_t>(mp.face_neigh[f_sz])];
        }
    }

    auto face_category = [&](LocalIndex f) noexcept -> int {
        const LocalIndex neigh = mp.face_neigh[static_cast<std::size_t>(f)];
        if (neigh < 0) return 2;             // Zone 2: Boundary face
        if (neigh >= n_own) return 1;        // Zone 1: Inter-rank / Coupled face
        return 0;                            // Zone 0: Purely internal face
    };

    std::vector<LocalIndex> face_perm(static_cast<std::size_t>(mp.n_faces));
    for (LocalIndex f = 0; f < mp.n_faces; ++f) face_perm[static_cast<std::size_t>(f)] = f;
    
    std::sort(face_perm.begin(), face_perm.end(), [&](LocalIndex a, LocalIndex b) {
        const std::size_t a_sz = static_cast<std::size_t>(a);
        const std::size_t b_sz = static_cast<std::size_t>(b);

        const int cat_a = face_category(a);
        const int cat_b = face_category(b);

        // Sort by contiguous 3-zone layout: Internal -> Coupled -> Boundary
        if (cat_a != cat_b) {
            return cat_a < cat_b;
        }

        // Zone 2 (Boundary): Group by patch_id, then sort by owner
        if (cat_a == 2) {
            if (mp.face_patch[a_sz] != mp.face_patch[b_sz]) {
                return mp.face_patch[a_sz] < mp.face_patch[b_sz];
            }
            return mp.face_owner[a_sz] < mp.face_owner[b_sz];
        }

        // Zone 0 (Internal) and Zone 1 (Coupled):
        // Sorted primarily by owner cell for cache locality in flux loops,
        // then by neigh for deterministic ordering.
        if (mp.face_owner[a_sz] != mp.face_owner[b_sz]) {
            return mp.face_owner[a_sz] < mp.face_owner[b_sz];
        }
        return mp.face_neigh[a_sz] < mp.face_neigh[b_sz];
    });

    // Permute all Face SoA arrays
    const std::size_t n_faces_sz = static_cast<std::size_t>(mp.n_faces);
    std::vector<LocalIndex> new_face_owner(n_faces_sz);
    std::vector<LocalIndex> new_face_neigh(n_faces_sz);
    std::vector<PatchId> new_face_patch(n_faces_sz);
    std::vector<CellType> new_face_type(n_faces_sz);
    std::vector<double> new_face_area(n_faces_sz);
    std::vector<double> new_face_centroid_x(n_faces_sz);
    std::vector<double> new_face_centroid_y(n_faces_sz);
    std::vector<double> new_face_centroid_z(n_faces_sz);
    std::vector<double> new_face_normal_x(n_faces_sz);
    std::vector<double> new_face_normal_y(n_faces_sz);
    std::vector<double> new_face_normal_z(n_faces_sz);

    std::vector<LocalIndex> new_face_nodes_offsets(n_faces_sz + 1, 0);
    for (std::size_t i = 0; i < n_faces_sz; ++i) {
        const std::size_t old_f = static_cast<std::size_t>(face_perm[i]);
        const LocalIndex nnodes = mp.face_nodes_offsets[old_f + 1] - mp.face_nodes_offsets[old_f];
        new_face_nodes_offsets[i + 1] = new_face_nodes_offsets[i] + nnodes;
    }

    std::vector<LocalIndex> new_face_nodes(static_cast<std::size_t>(new_face_nodes_offsets.back()));

    for (std::size_t i = 0; i < n_faces_sz; ++i) {
        const std::size_t old_f = static_cast<std::size_t>(face_perm[i]);

        new_face_owner[i] = mp.face_owner[old_f];
        new_face_neigh[i] = mp.face_neigh[old_f];
        new_face_patch[i] = mp.face_patch[old_f];
        new_face_type[i] = mp.face_type[old_f];
        new_face_area[i] = mp.face_area[old_f];
        new_face_centroid_x[i] = mp.face_centroid_x[old_f];
        new_face_centroid_y[i] = mp.face_centroid_y[old_f];
        new_face_centroid_z[i] = mp.face_centroid_z[old_f];
        new_face_normal_x[i] = mp.face_normal_x[old_f];
        new_face_normal_y[i] = mp.face_normal_y[old_f];
        new_face_normal_z[i] = mp.face_normal_z[old_f];

        const LocalIndex old_off = mp.face_nodes_offsets[old_f];
        const LocalIndex new_off = new_face_nodes_offsets[i];
        const std::size_t nnodes = static_cast<std::size_t>(mp.face_nodes_offsets[old_f + 1] - old_off);

        for (std::size_t k = 0; k < nnodes; ++k) {
            new_face_nodes[static_cast<std::size_t>(new_off) + k] =
                mp.face_nodes[static_cast<std::size_t>(old_off) + k];
        }
    }

    mp.face_owner = std::move(new_face_owner);
    mp.face_neigh = std::move(new_face_neigh);
    mp.face_patch = std::move(new_face_patch);
    mp.face_type = std::move(new_face_type);
    mp.face_area = std::move(new_face_area);
    mp.face_centroid_x = std::move(new_face_centroid_x);
    mp.face_centroid_y = std::move(new_face_centroid_y);
    mp.face_centroid_z = std::move(new_face_centroid_z);
    mp.face_normal_x = std::move(new_face_normal_x);
    mp.face_normal_y = std::move(new_face_normal_y);
    mp.face_normal_z = std::move(new_face_normal_z);
    mp.face_nodes_offsets = std::move(new_face_nodes_offsets);
    mp.face_nodes = std::move(new_face_nodes);



    // 6. Rebuild BC Patch Flat CSR Tables
    const std::size_t n_patches_sz = mp.patches.size();
    mp.patch_face_offsets.assign(n_patches_sz + 1, 0);

    std::vector<LocalIndex> patch_counts(n_patches_sz, 0);
    for (LocalIndex f = 0; f < mp.n_faces; ++f) {
        const PatchId p = mp.face_patch[static_cast<std::size_t>(f)];
        if (p != kInvalidPatch) {
            const std::size_t p_sz = static_cast<std::size_t>(p);
            if (p_sz < n_patches_sz) ++patch_counts[p_sz];
        }
    }

    for (std::size_t p = 0; p < n_patches_sz; ++p) {
        mp.patch_face_offsets[p + 1] = mp.patch_face_offsets[p] + patch_counts[p];
    }

    mp.patch_faces.resize(static_cast<std::size_t>(mp.patch_face_offsets.back()));
    std::vector<LocalIndex> patch_cursors = mp.patch_face_offsets;

    for (LocalIndex f = 0; f < mp.n_faces; ++f) {
        const PatchId p = mp.face_patch[static_cast<std::size_t>(f)];
        if (p != kInvalidPatch) {
            const auto p_sz = static_cast<std::size_t>(p);
            if (p_sz < n_patches_sz) {
                mp.patch_faces[static_cast<std::size_t>(patch_cursors[p_sz]++)] = f;
            }
        }
    }

    // 7. Calculate 3-Zone Face Boundaries
    LocalIndex n_internal_faces = 0;
    LocalIndex n_inner_faces = 0;

    for (LocalIndex f = 0; f < mp.n_faces; ++f) {
        const LocalIndex neigh = mp.face_neigh[static_cast<std::size_t>(f)];
        if (neigh >= 0) {
            ++n_inner_faces;
            if (neigh < n_own) {
                assert(n_inner_faces == n_internal_faces + 1); // Zone 0 must be strictly contiguous
                ++n_internal_faces;
            }
        } else {
            break; // Reached Zone 2 (Boundary faces)
        }
    }

    mp.n_internal_faces = n_internal_faces;
    mp.n_inner_faces = n_inner_faces;
}

} // anonymous namespace


void reorder_local_mesh(MeshPart& mp, ReorderMethod method) {
    if (mp.n_own <= 1) {
        return;
    }

    std::vector<LocalIndex> new2old;
    if (method == ReorderMethod::RCM) {
        new2old = compute_rcm_order(mp);
    } else if (method == ReorderMethod::HILBERT_SFC) {
        new2old = compute_hilbert_order(mp);
    }

    if (new2old.empty()) {
        const LocalIndex n_own = mp.n_own;
        const std::size_t n_own_sz = static_cast<std::size_t>(n_own);
        new2old.resize(n_own_sz);
        for (std::size_t i = 0; i < n_own_sz; ++i) {
            new2old[i] = static_cast<LocalIndex>(i); 
        }
    }

    apply_cell_permutation(mp, new2old);
}

} // namespace cfd::mesh