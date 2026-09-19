#include "cfd/mesh_generator/topology.hpp"

#include <algorithm>
#include <format>
#include <string>

#include "cfd/mesh_generator/grading.hpp"
#include "cfd/mesh_generator/tfi.hpp"
#include "cfd/mpi/log.hpp"

namespace cfd::mesh_generator {

namespace {

[[noreturn]] void fail(MPI_Comm comm, const std::string& what) {
    mpi::fatal(comm, "topology: " + what);
    std::abort();
}

// Canonical edge ordering (u < v)
struct RawEdgeKey {
    std::size_t u{0};
    std::size_t v{0};

    bool operator==(const RawEdgeKey& o) const noexcept {
        return u == o.u && v == o.v;
    }
};

// Maps 4 cyclically ordered quad vertices to a canonical representation:
// starts with minimal vertex index, then chooses second vertex as smaller of the two neighbors
struct CanonicalFaceOrientation {
    std::array<std::size_t, 4> canonical_verts;
    std::uint8_t start_idx; // 0..3
    bool is_flipped;        // true if reversed
};

CanonicalFaceOrientation make_canonical_face(const std::array<std::size_t, 4>& in) {
    std::size_t min_pos = 0;
    for (std::size_t i = 1; i < 4; ++i) {
        if (in[i] < in[min_pos]) {
            min_pos = i;
        }
    }

    const std::size_t next_v = in[(min_pos + 1) % 4];
    const std::size_t prev_v = in[(min_pos + 3) % 4];

    CanonicalFaceOrientation res;
    res.start_idx = static_cast<std::uint8_t>(min_pos);

    if (next_v <= prev_v) {
        res.is_flipped = false;
        for (std::size_t i = 0; i < 4; ++i) {
            res.canonical_verts[i] = in[(min_pos + i) % 4];
        }
    } else {
        res.is_flipped = true;
        for (std::size_t i = 0; i < 4; ++i) {
            res.canonical_verts[i] = in[(min_pos + 4 - i) % 4];
        }
    }
    return res;
}

} // namespace

void MacroTopology::build(const GeneratorConfig& config, MPI_Comm comm) {
    const std::size_t num_blocks = config.blocks.size();
    const std::size_t num_macro_vertices = config.vertices.size();

    block_topos_.resize(num_blocks);
    volumes_.resize(num_blocks);
    cell_offsets_.assign(num_blocks + 1, 0);

    // 1. Calculate cell offsets and total cells
    total_cells_ = 0;
    for (std::size_t b = 0; b < num_blocks; ++b) {
        cell_offsets_[b] = total_cells_;
        const auto& blk = config.blocks[b];
        const std::uint64_t b_cells = static_cast<std::uint64_t>(blk.cells[0]) *
                                      static_cast<std::uint64_t>(blk.cells[1]) *
                                      static_cast<std::uint64_t>(blk.cells[2]);
        total_cells_ += b_cells;
    }
    cell_offsets_[num_blocks] = total_cells_;

    // 2. Discover unique macro-edges and macro-faces across all blocks
    // Local edge vertex pairs for HEXA_8 (v0..v7)
    // Edges along xi (0..3), eta (4..7), zeta (8..11)
    constexpr std::array<std::pair<std::size_t, std::size_t>, 12> hex_edge_pairs = {{
        {0, 1}, {3, 2}, {4, 5}, {7, 6}, // xi-direction
        {0, 3}, {1, 2}, {4, 7}, {5, 6}, // eta-direction
        {0, 4}, {1, 5}, {2, 6}, {3, 7}  // zeta-direction
    }};

    // Local face vertex quads for HEXA_8 (ordered counter-clockwise seen from outside)
    constexpr std::array<std::array<std::size_t, 4>, 6> hex_face_quads = {{
        {0, 3, 2, 1}, // Face 0: bottom (zeta = 0)
        {0, 1, 5, 4}, // Face 1: front  (eta = 0)
        {1, 2, 6, 5}, // Face 2: right  (xi = 1)
        {2, 3, 7, 6}, // Face 3: back   (eta = 1)
        {3, 0, 4, 7}, // Face 4: left   (xi = 0)
        {4, 5, 6, 7}  // Face 5: top    (zeta = 1)
    }};

    for (std::size_t b = 0; b < num_blocks; ++b) {
        const auto& blk = config.blocks[b];

        // Process 12 edges
        for (std::size_t e = 0; e < 12; ++e) {
            const std::size_t u = blk.vertices[hex_edge_pairs[e].first];
            const std::size_t v = blk.vertices[hex_edge_pairs[e].second];

            std::size_t cells = 0;
            GradingType gt = GradingType::UNIFORM;
            double g = 1.0;

            if (e < 4) {
                cells = blk.cells[0];
                gt = blk.grading_type[0];
                g = blk.grading[0];
            } else if (e < 8) {
                cells = blk.cells[1];
                gt = blk.grading_type[1];
                g = blk.grading[1];
            } else {
                cells = blk.cells[2];
                gt = blk.grading_type[2];
                g = blk.grading[2];
            }

            const std::size_t edge_idx = find_or_add_edge(u, v, cells, gt, g, config, comm);
            const bool is_reversed = (u > v);
            block_topos_[b].edges[e] = {edge_idx, is_reversed};
        }

        // Process 6 faces
        for (std::size_t f = 0; f < 6; ++f) {
            std::array<std::size_t, 4> quad_v;
            for (std::size_t i = 0; i < 4; ++i) {
                quad_v[i] = blk.vertices[hex_face_quads[f][i]];
            }

            std::size_t cu = 0, cv = 0;
            GradingType gtu = GradingType::UNIFORM, gtv = GradingType::UNIFORM;
            double gu = 1.0, gv = 1.0;

            if (f == 0 || f == 5) { // zeta-faces: (xi, eta)
                cu = blk.cells[0]; cv = blk.cells[1];
                gtu = blk.grading_type[0]; gtv = blk.grading_type[1];
                gu = blk.grading[0]; gv = blk.grading[1];
            } else if (f == 1 || f == 3) { // eta-faces: (xi, zeta)
                cu = blk.cells[0]; cv = blk.cells[2];
                gtu = blk.grading_type[0]; gtv = blk.grading_type[2];
                gu = blk.grading[0]; gv = blk.grading[2];
            } else { // xi-faces: (eta, zeta)
                cu = blk.cells[1]; cv = blk.cells[2];
                gtu = blk.grading_type[1]; gtv = blk.grading_type[2];
                gu = blk.grading[1]; gv = blk.grading[2];
            }

            const auto orient = make_canonical_face(quad_v);
            const std::size_t face_idx = find_or_add_face(orient.canonical_verts, cu, cv, gtu, gtv, gu, gv, config, comm);
            block_topos_[b].faces[f] = {face_idx, orient.start_idx, orient.is_flipped};
        }

        // Prepare block volume info
        auto& vol = volumes_[b];
        vol.count = (blk.cells[0] - 1) * (blk.cells[1] - 1) * (blk.cells[2] - 1);
        vol.dist_x = Grading::compute_distribution(blk.cells[0], blk.grading_type[0], blk.grading[0]);
        vol.dist_y = Grading::compute_distribution(blk.cells[1], blk.grading_type[1], blk.grading[1]);
        vol.dist_z = Grading::compute_distribution(blk.cells[2], blk.grading_type[2], blk.grading[2]);
    }

    // 3. Assign 1-based global NodeIDs across non-overlapping topological strata
    std::uint64_t current_id = 1;

    // Strata 0: Macro-vertices [1, num_macro_vertices]
    lookups_.reserve(num_macro_vertices + edges_.size() + faces_.size() + num_blocks);
    for (std::size_t v = 0; v < num_macro_vertices; ++v) {
        lookups_.push_back({current_id, current_id, FeatureType::VERTEX, v});
        current_id++;
    }

    // Strata 1: Macro-edges interior nodes
    for (std::size_t e = 0; e < edges_.size(); ++e) {
        auto& edge = edges_[e];
        const std::size_t interior_count = (edge.num_cells > 1) ? (edge.num_cells - 1) : 0;
        if (interior_count > 0) {
            edge.global_start_id = current_id;
            const std::uint64_t end_id = current_id + interior_count - 1;
            lookups_.push_back({current_id, end_id, FeatureType::EDGE, e});
            current_id += interior_count;
        }
    }

    // Strata 2: Macro-faces interior nodes
    for (std::size_t f = 0; f < faces_.size(); ++f) {
        auto& face = faces_[f];
        const std::size_t interior_count = (face.cells_u > 1 && face.cells_v > 1)
                                               ? (face.cells_u - 1) * (face.cells_v - 1)
                                               : 0;
        if (interior_count > 0) {
            face.global_start_id = current_id;
            const std::uint64_t end_id = current_id + interior_count - 1;
            lookups_.push_back({current_id, end_id, FeatureType::FACE, f});
            current_id += interior_count;
        }
    }

    // Strata 3: Block volumes interior nodes
    for (std::size_t b = 0; b < num_blocks; ++b) {
        auto& vol = volumes_[b];
        if (vol.count > 0) {
            vol.global_start_id = current_id;
            const std::uint64_t end_id = current_id + vol.count - 1;
            lookups_.push_back({current_id, end_id, FeatureType::VOLUME, b});
            current_id += vol.count;
        }
    }

    total_nodes_ = current_id - 1;
}

std::size_t MacroTopology::find_or_add_edge(std::size_t u, std::size_t v, std::size_t cells,
                                            GradingType gt, double g, const GeneratorConfig& cfg, MPI_Comm comm) {
    const std::size_t v_min = std::min(u, v);
    const std::size_t v_max = std::max(u, v);

    for (std::size_t i = 0; i < edges_.size(); ++i) {
        if (edges_[i].v0 == v_min && edges_[i].v1 == v_max) {
            // Conformance check: shared edge must have the exact same cell count
            if (edges_[i].num_cells != cells) {
                fail(comm, std::format("Non-conformal mesh! Shared edge ({}, {}) has conflicting cell counts: {} vs {}",
                                       v_min, v_max, edges_[i].num_cells, cells));
            }
            return i;
        }
    }

    CanonicalEdge edge;
    edge.v0 = v_min;
    edge.v1 = v_max;
    edge.num_cells = cells;
    edge.grading_type = gt;
    edge.grading = g;
    edge.distribution = Grading::compute_distribution(cells, gt, g);

    edges_.push_back(std::move(edge));
    return edges_.size() - 1;
}

std::size_t MacroTopology::find_or_add_face(const std::array<std::size_t, 4>& q,
                                            std::size_t cu, std::size_t cv,
                                            GradingType gtu, GradingType gtv,
                                            double gu, double gv,
                                            const GeneratorConfig& cfg, MPI_Comm comm) {
    for (std::size_t i = 0; i < faces_.size(); ++i) {
        if (faces_[i].vertices == q) {
            if ((faces_[i].cells_u != cu || faces_[i].cells_v != cv) &&
                (faces_[i].cells_u != cv || faces_[i].cells_v != cu)) {
                fail(comm, std::format("Non-conformal mesh! Shared face ({},{},{},{}) has conflicting cell counts",
                                       q[0], q[1], q[2], q[3]));
            }
            return i;
        }
    }

    CanonicalFace face;
    face.vertices = q;
    face.cells_u = cu;
    face.cells_v = cv;
    face.grading_type_u = gtu;
    face.grading_type_v = gtv;
    face.grading_u = gu;
    face.grading_v = gv;
    face.dist_u = Grading::compute_distribution(cu, gtu, gu);
    face.dist_v = Grading::compute_distribution(cv, gtv, gv);

    faces_.push_back(std::move(face));
    return faces_.size() - 1;
}

std::uint64_t MacroTopology::get_node_id(std::size_t b, std::size_t i, std::size_t j, std::size_t k) const noexcept {
    const auto& blk = block_topos_[b];
    // Block cell divisions along x, y, z
    const auto& vol = volumes_[b];
    const std::size_t Nx = vol.dist_x.size() - 1;
    const std::size_t Ny = vol.dist_y.size() - 1;
    const std::size_t Nz = vol.dist_z.size() - 1;

    const bool on_x0 = (i == 0), on_x1 = (i == Nx);
    const bool on_y0 = (j == 0), on_y1 = (j == Ny);
    const bool on_z0 = (k == 0), on_z1 = (k == Nz);

    const int bnd_count = (on_x0 || on_x1 ? 1 : 0) +
                          (on_y0 || on_y1 ? 1 : 0) +
                          (on_z0 || on_z1 ? 1 : 0);

    // 1. Macro-vertex (all 3 coords on boundary)
    if (bnd_count == 3) {
        const std::size_t corner = (on_z0 ? 0 : 4) +
                                   (on_y0 ? (on_x0 ? 0 : 1) : (on_x0 ? 3 : 2));
        // Return 1-based vertex index
        const auto& raw_block = lookups_[0]; // base pointer not needed, vertex is simply ID
        (void)raw_block;
        // Vertex range is [1, num_macro_vertices]
        // Look up macro-block corner vertex
        const std::size_t v_idx = (corner == 0) ? (on_y0 ? (on_x0 ? 0 : 1) : (on_x0 ? 3 : 2))
                                                : (on_y0 ? (on_x0 ? 4 : 5) : (on_x0 ? 7 : 6));
        // Retrieve vertex index from lookups
        return lookups_[v_idx].start_id;
    }

    // 2. Macro-edge (2 coords on boundary)
    if (bnd_count == 2) {
        std::size_t e = 0;
        std::size_t step = 0;
        std::size_t max_step = 0;

        if (!on_x0 && !on_x1) { // xi-direction
            step = i;
            max_step = Nx;
            if (on_y0 && on_z0) e = 0;
            else if (on_y1 && on_z0) e = 1;
            else if (on_y0 && on_z1) e = 2;
            else e = 3;
        } else if (!on_y0 && !on_y1) { // eta-direction
            step = j;
            max_step = Ny;
            if (on_x0 && on_z0) e = 4;
            else if (on_x1 && on_z0) e = 5;
            else if (on_x0 && on_z1) e = 6;
            else e = 7;
        } else { // zeta-direction
            step = k;
            max_step = Nz;
            if (on_x0 && on_y0) e = 8;
            else if (on_x1 && on_y0) e = 9;
            else if (on_x1 && on_y1) e = 10;
            else e = 11;
        }

        const auto [edge_idx, is_reversed] = blk.edges[e];
        const auto& edge = edges_[edge_idx];
        const std::size_t interior_idx = is_reversed ? (max_step - step - 1) : (step - 1);
        return edge.global_start_id + interior_idx;
    }

    // 3. Macro-face (1 coord on boundary)
    if (bnd_count == 1) {
        std::size_t f = 0;
        std::size_t u = 0, v = 0;
        std::size_t Nu = 0, Nv = 0;

        if (on_z0) { f = 0; u = i; v = j; Nu = Nx; Nv = Ny; }
        else if (on_z1) { f = 5; u = i; v = j; Nu = Nx; Nv = Ny; }
        else if (on_y0) { f = 1; u = i; v = k; Nu = Nx; Nv = Nz; }
        else if (on_y1) { f = 3; u = i; v = k; Nu = Nx; Nv = Nz; }
        else if (on_x0) { f = 4; u = j; v = k; Nu = Ny; Nv = Nz; }
        else { f = 2; u = j; v = k; Nu = Ny; Nv = Nz; }

        const auto& [face_idx, start_idx, is_flipped] = blk.faces[f];
        const auto& face = faces_[face_idx];

        // Map local (u, v) into canonical quad (p, q)
        std::size_t p = u - 1;
        std::size_t q = v - 1;
        if (is_flipped) {
            p = (Nu - 1) - u;
        }

        return face.global_start_id + q * (face.cells_u - 1) + p;
    }

    // 4. Volume interior node (0 coords on boundary)
    const std::uint64_t v_idx = static_cast<std::uint64_t>(k - 1) * (Nx - 1) * (Ny - 1) +
                                static_cast<std::uint64_t>(j - 1) * (Nx - 1) +
                                static_cast<std::uint64_t>(i - 1);
    return vol.global_start_id + v_idx;
}

Vec3 MacroTopology::evaluate_node_coord(std::uint64_t global_node_id,
                                        const GeneratorConfig& config) const noexcept {
    // Binary search in sorted range lookups: O(log(num_features))
    auto it = std::upper_bound(lookups_.begin(), lookups_.end(), global_node_id,
                               [](std::uint64_t id, const NodeRangeLookup& r) {
                                   return id < r.start_id;
                               });
    if (it != lookups_.begin()) {
        --it;
    }

    const auto& range = *it;
    const std::size_t offset = static_cast<std::size_t>(global_node_id - range.start_id);

    switch (range.type) {
        case FeatureType::VERTEX: {
            return config.vertices[range.feature_idx];
        }
        case FeatureType::EDGE: {
            const auto& edge = edges_[range.feature_idx];
            const double s = edge.distribution[offset + 1];
            return TFI::interpolate_edge(config.vertices[edge.v0], config.vertices[edge.v1], s);
        }
        case FeatureType::FACE: {
            const auto& face = faces_[range.feature_idx];
            const std::size_t nu = face.cells_u - 1;
            const std::size_t p = offset % nu;
            const std::size_t q = offset / nu;

            const double u = face.dist_u[p + 1];
            const double v = face.dist_v[q + 1];
            const std::array<Vec3, 4> corners = {
                config.vertices[face.vertices[0]],
                config.vertices[face.vertices[1]],
                config.vertices[face.vertices[2]],
                config.vertices[face.vertices[3]]
            };
            return TFI::interpolate_quad(corners, u, v);
        }
        case FeatureType::VOLUME: {
            const std::size_t b = range.feature_idx;
            const auto& vol = volumes_[b];
            const auto& blk = config.blocks[b];

            const std::size_t nx = blk.cells[0] - 1;
            const std::size_t ny = blk.cells[1] - 1;

            const std::size_t i = offset % nx;
            const std::size_t j = (offset / nx) % ny;
            const std::size_t k = offset / (nx * ny);

            const double xi   = vol.dist_x[i + 1];
            const double eta  = vol.dist_y[j + 1];
            const double zeta = vol.dist_z[k + 1];

            std::array<Vec3, 8> corners;
            for (std::size_t c = 0; c < 8; ++c) {
                corners[c] = config.vertices[blk.vertices[c]];
            }
            return TFI::interpolate_hex(corners, xi, eta, zeta);
        }
    }
    return {};
}

void MacroTopology::get_cell_block_and_ijk(std::uint64_t global_cell_id,
                                           std::size_t& block_idx,
                                           std::size_t& i,
                                           std::size_t& j,
                                           std::size_t& k) const noexcept {
    auto it = std::upper_bound(cell_offsets_.begin(), cell_offsets_.end(), global_cell_id);
    block_idx = static_cast<std::size_t>(std::distance(cell_offsets_.begin(), it) - 1);

    const std::uint64_t local_c = global_cell_id - cell_offsets_[block_idx];
    const auto& vol = volumes_[block_idx];
    const std::size_t Nx = vol.dist_x.size() - 1;
    const std::size_t Ny = vol.dist_y.size() - 1;

    k = static_cast<std::size_t>(local_c / (Nx * Ny));
    j = static_cast<std::size_t>((local_c % (Nx * Ny)) / Nx);
    i = static_cast<std::size_t>(local_c % Nx);
}

std::array<std::uint64_t, 8> MacroTopology::get_cell_nodes(std::uint64_t global_cell_id) const noexcept {
    std::size_t b = 0, i = 0, j = 0, k = 0;
    get_cell_block_and_ijk(global_cell_id, b, i, j, k);

    // HEXA_8 local node ordering conforming to CGNS/VTK:
    // Bottom (zeta = 0): (i, j, k), (i+1, j, k), (i+1, j+1, k), (i, j+1, k)
    // Top    (zeta = 1): (i, j, k+1), (i+1, j, k+1), (i+1, j+1, k+1), (i, j+1, k+1)
    return {
        get_node_id(b, i,     j,     k),
        get_node_id(b, i + 1, j,     k),
        get_node_id(b, i + 1, j + 1, k),
        get_node_id(b, i,     j + 1, k),
        get_node_id(b, i,     j,     k + 1),
        get_node_id(b, i + 1, j,     k + 1),
        get_node_id(b, i + 1, j + 1, k + 1),
        get_node_id(b, i,     j + 1, k + 1)
    };
}

} // namespace cfd::mesh_generator