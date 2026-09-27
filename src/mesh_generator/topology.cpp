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

} // namespace

void MacroTopology::build(const GeneratorConfig& config, MPI_Comm comm) {
    const std::size_t num_blocks = config.blocks.size();
    const std::size_t num_macro_vertices = config.vertices.size();

    blocks_.resize(num_blocks);
    cell_offsets_.assign(num_blocks + 1, 0);

    total_cells_ = 0;
    for (std::size_t b = 0; b < num_blocks; ++b) {
        cell_offsets_[b] = total_cells_;
        const auto& blk = config.blocks[b];
        total_cells_ += static_cast<std::uint64_t>(blk.cells[0]) * blk.cells[1] * blk.cells[2];
    }
    cell_offsets_[num_blocks] = total_cells_;

    // 12 edges in standard HEXA_8 (0: (0,0,0), 1: (1,0,0), 2: (1,1,0), 3: (0,1,0)...)
    // Ordered strictly from step 0 to step N along xi (0..3), eta (4..7), zeta (8..11)
    constexpr std::array<std::pair<std::size_t, std::size_t>, 12> hex_edge_pairs = {{
        {0, 1}, {3, 2}, {4, 5}, {7, 6}, // xi-direction
        {0, 3}, {1, 2}, {4, 7}, {5, 6}, // eta-direction
        {0, 4}, {1, 5}, {2, 6}, {3, 7}  // zeta-direction
    }};

    // 6 faces in parametric (u, v) order: (0,0), (1,0), (1,1), (0,1)
    constexpr std::array<std::array<std::size_t, 4>, 6> hex_face_quads = {{
        {0, 1, 2, 3}, // Face 0: bottom (zeta = 0, u = xi, v = eta)
        {0, 1, 5, 4}, // Face 1: front  (eta = 0,  u = xi, v = zeta)
        {1, 2, 6, 5}, // Face 2: right  (xi = 1,   u = eta, v = zeta)
        {3, 2, 6, 7}, // Face 3: back   (eta = 1,  u = xi, v = zeta)
        {0, 3, 7, 4}, // Face 4: left   (xi = 0,   u = eta, v = zeta)
        {4, 5, 6, 7}  // Face 5: top    (zeta = 1, u = xi, v = eta)
    }};

    for (std::size_t b = 0; b < num_blocks; ++b) {
        const auto& blk = config.blocks[b];
        auto& bt = blocks_[b];
        bt.vertices = blk.vertices;

        for (std::size_t e = 0; e < 12; ++e) {
            const std::size_t u = blk.vertices[hex_edge_pairs[e].first];
            const std::size_t v = blk.vertices[hex_edge_pairs[e].second];
            const std::size_t cells = (e < 4) ? blk.cells[0] : ((e < 8) ? blk.cells[1] : blk.cells[2]);

            const std::size_t edge_idx = find_or_add_edge(u, v, cells, comm);
            bt.edges[e] = {edge_idx, (u > v)};
        }

        for (std::size_t f = 0; f < 6; ++f) {
            std::array<std::size_t, 4> q_verts;
            for (std::size_t i = 0; i < 4; ++i) {
                q_verts[i] = blk.vertices[hex_face_quads[f][i]];
            }

            std::size_t cu = 0, cv = 0;
            if (f == 0 || f == 5)      { cu = blk.cells[0]; cv = blk.cells[1]; }
            else if (f == 1 || f == 3) { cu = blk.cells[0]; cv = blk.cells[2]; }
            else                       { cu = blk.cells[1]; cv = blk.cells[2]; }

            bt.faces[f] = find_or_add_face(q_verts, cu, cv, comm);
        }

        bt.vol_count = (blk.cells[0] - 1) * (blk.cells[1] - 1) * (blk.cells[2] - 1);
        bt.dist_x = Grading::compute_distribution(blk.cells[0], blk.grading_type[0], blk.grading[0]);
        bt.dist_y = Grading::compute_distribution(blk.cells[1], blk.grading_type[1], blk.grading[1]);
        bt.dist_z = Grading::compute_distribution(blk.cells[2], blk.grading_type[2], blk.grading[2]);
    }

    // Allocate 1-based unique IDs
    std::uint64_t current_id = 1;

    // Strata 0: Macro-vertices [1, num_macro_vertices]
    for (std::size_t v = 0; v < num_macro_vertices; ++v) {
        current_id++;
    }

    // Strata 1: Unique Edges
    for (auto& edge : edges_) {
        const std::size_t count = (edge.num_cells > 1) ? (edge.num_cells - 1) : 0;
        if (count > 0) {
            edge.global_start_id = current_id;
            current_id += count;
        }
    }

    // Strata 2: Unique Faces
    for (auto& face : faces_) {
        const std::size_t count = (face.cells_u > 1 && face.cells_v > 1)
                                      ? (face.cells_u - 1) * (face.cells_v - 1)
                                      : 0;
        if (count > 0) {
            face.global_start_id = current_id;
            current_id += count;
        }
    }

    // Strata 3: Block Volumes
    for (std::size_t b = 0; b < num_blocks; ++b) {
        if (blocks_[b].vol_count > 0) {
            blocks_[b].vol_start_id = current_id;
            current_id += blocks_[b].vol_count;
        }
    }

    total_nodes_ = current_id - 1;

    // Register primary source coordinates for every global node
    node_sources_.resize(total_nodes_);

    for (std::size_t b = 0; b < num_blocks; ++b) {
        const auto& blk = config.blocks[b];
        const std::size_t Nx = blk.cells[0];
        const std::size_t Ny = blk.cells[1];
        const std::size_t Nz = blk.cells[2];

        for (std::size_t k = 0; k <= Nz; ++k) {
            for (std::size_t j = 0; j <= Ny; ++j) {
                for (std::size_t i = 0; i <= Nx; ++i) {
                    const std::uint64_t nid = get_node_id(b, i, j, k);
                    node_sources_[nid - 1] = {b, i, j, k};
                }
            }
        }
    }
}

std::size_t MacroTopology::find_or_add_edge(std::size_t u, std::size_t v, std::size_t cells, MPI_Comm comm) {
    const std::size_t v_min = std::min(u, v);
    const std::size_t v_max = std::max(u, v);

    for (std::size_t i = 0; i < edges_.size(); ++i) {
        if (edges_[i].v0 == v_min && edges_[i].v1 == v_max) {
            if (edges_[i].num_cells != cells) {
                fail(comm, std::format("Shared edge ({},{}) cell count mismatch: {} vs {}",
                                       v_min, v_max, edges_[i].num_cells, cells));
            }
            return i;
        }
    }

    edges_.push_back({v_min, v_max, cells, 0});
    return edges_.size() - 1;
}

std::size_t MacroTopology::find_or_add_face(const std::array<std::size_t, 4>& q_verts,
                                            std::size_t cu, std::size_t cv, MPI_Comm comm) {
    auto sorted_q = q_verts;
    std::sort(sorted_q.begin(), sorted_q.end());

    static_cast<void>(comm);
    
    for (std::size_t i = 0; i < faces_.size(); ++i) {
        auto existing = faces_[i].vertices;
        std::sort(existing.begin(), existing.end());
        if (existing == sorted_q) {
            return i;
        }
    }

    faces_.push_back({q_verts, cu, cv, 0});
    return faces_.size() - 1;
}

std::uint64_t MacroTopology::get_node_id(std::size_t b, std::size_t i, std::size_t j, std::size_t k) const noexcept {
    const auto& bt = blocks_[b];
    const std::size_t Nx = bt.dist_x.size() - 1;
    const std::size_t Ny = bt.dist_y.size() - 1;
    const std::size_t Nz = bt.dist_z.size() - 1;

    const bool on_x0 = (i == 0), on_x1 = (i == Nx);
    const bool on_y0 = (j == 0), on_y1 = (j == Ny);
    const bool on_z0 = (k == 0), on_z1 = (k == Nz);

    const int bnd_count = (on_x0 || on_x1 ? 1 : 0) +
                          (on_y0 || on_y1 ? 1 : 0) +
                          (on_z0 || on_z1 ? 1 : 0);

    // 1. Macro-vertex
    if (bnd_count == 3) {
        const std::size_t corner = (on_z0 ? 0 : 4) +
                                   (on_y0 ? (on_x0 ? 0 : 1) : (on_x0 ? 3 : 2));
        return static_cast<std::uint64_t>(bt.vertices[corner] + 1);
    }

    // 2. Macro-edge
    if (bnd_count == 2) {
        std::size_t e = 0;
        std::size_t step = 0;
        std::size_t max_step = 0;

        if (!on_x0 && !on_x1) {
            step = i; max_step = Nx;
            if (on_y0 && on_z0)      e = 0;
            else if (on_y1 && on_z0) e = 1;
            else if (on_y0 && on_z1) e = 2;
            else                     e = 3;
        } else if (!on_y0 && !on_y1) {
            step = j; max_step = Ny;
            if (on_x0 && on_z0)      e = 4;
            else if (on_x1 && on_z0) e = 5;
            else if (on_x0 && on_z1) e = 6;
            else                     e = 7;
        } else {
            step = k; max_step = Nz;
            if (on_x0 && on_y0)      e = 8;
            else if (on_x1 && on_y0) e = 9;
            else if (on_x1 && on_y1) e = 10;
            else                     e = 11;
        }

        const auto [edge_idx, is_reversed] = bt.edges[e];
        const auto& edge = edges_[edge_idx];
        const std::size_t interior_idx = is_reversed ? (max_step - step - 1) : (step - 1);
        return edge.global_start_id + interior_idx;
    }

    // 3. Macro-face
    if (bnd_count == 1) {
        std::size_t f = 0;
        std::size_t u = 0, v = 0;

        if (on_z0)      { f = 0; u = i; v = j; }
        else if (on_z1) { f = 5; u = i; v = j; }
        else if (on_y0) { f = 1; u = i; v = k; }
        else if (on_y1) { f = 3; u = i; v = k; }
        else if (on_x0) { f = 4; u = j; v = k; }
        else            { f = 2; u = j; v = k; }

        const auto& face = faces_[bt.faces[f]];
        return face.global_start_id + (v - 1) * (face.cells_u - 1) + (u - 1);
    }

    // 4. Volume interior
    const std::uint64_t v_idx = static_cast<std::uint64_t>(k - 1) * (Nx - 1) * (Ny - 1) +
                                static_cast<std::uint64_t>(j - 1) * (Nx - 1) +
                                static_cast<std::uint64_t>(i - 1);
    return bt.vol_start_id + v_idx;
}

Vec3 MacroTopology::evaluate_node_coord(std::uint64_t global_node_id,
                                        const GeneratorConfig& config) const noexcept {
    const auto& src = node_sources_[global_node_id - 1];
    const auto& bt  = blocks_[src.block_idx];
    const auto& blk = config.blocks[src.block_idx];

    const double xi   = bt.dist_x[src.i];
    const double eta  = bt.dist_y[src.j];
    const double zeta = bt.dist_z[src.k];

    std::array<Vec3, 8> corners;
    for (std::size_t c = 0; c < 8; ++c) {
        corners[c] = config.vertices[blk.vertices[c]];
    }

    return TFI::interpolate_hex(corners, xi, eta, zeta);
}

void MacroTopology::get_cell_block_and_ijk(std::uint64_t global_cell_id,
                                           std::size_t& block_idx,
                                           std::size_t& i,
                                           std::size_t& j,
                                           std::size_t& k) const noexcept {
    auto it = std::upper_bound(cell_offsets_.begin(), cell_offsets_.end(), global_cell_id);
    block_idx = static_cast<std::size_t>(std::distance(cell_offsets_.begin(), it) - 1);

    const std::uint64_t local_c = global_cell_id - cell_offsets_[block_idx];
    const auto& bt = blocks_[block_idx];
    const std::size_t Nx = bt.dist_x.size() - 1;
    const std::size_t Ny = bt.dist_y.size() - 1;

    k = static_cast<std::size_t>(local_c / (Nx * Ny));
    j = static_cast<std::size_t>((local_c % (Nx * Ny)) / Nx);
    i = static_cast<std::size_t>(local_c % Nx);
}

std::array<std::uint64_t, 8> MacroTopology::get_block_cell_nodes(std::size_t b,
                                                                 std::size_t i,
                                                                 std::size_t j,
                                                                 std::size_t k) const noexcept {
    // Strictly conforms to CGNS/VTK HEXA_8 (bottom CCW seen from top, top CCW seen from top)
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

std::array<std::uint64_t, 8> MacroTopology::get_cell_nodes(std::uint64_t global_cell_id) const noexcept {
    std::size_t b = 0, i = 0, j = 0, k = 0;
    get_cell_block_and_ijk(global_cell_id, b, i, j, k);
    return get_block_cell_nodes(b, i, j, k);
}

} // namespace cfd::mesh_generator