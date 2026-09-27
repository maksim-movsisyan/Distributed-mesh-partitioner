#include "cfd/mesh_generator/writer.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <string>
#include <vector>

#include <cgnslib.h>
#include <pcgnslib.h>
#include "cfd/mpi/log.hpp"

namespace cfd::mesh_generator {

namespace {

inline void check_cgns(int status, const char* msg, MPI_Comm comm) {
    if (status != CG_OK) {
        mpi::fatal(comm, std::format("CGNS Error [{}]: {}", msg, cg_get_error()));
        std::abort();
    }
}

inline std::pair<std::uint64_t, std::uint64_t> decompose_1d(std::uint64_t total_count,
                                                            int rank,
                                                            int nprocs) noexcept {
    if (total_count == 0) return {0, 0};
    const std::uint64_t base = total_count / static_cast<std::uint64_t>(nprocs);
    const std::uint64_t rem  = total_count % static_cast<std::uint64_t>(nprocs);

    const std::uint64_t start = static_cast<std::uint64_t>(rank) * base +
                                std::min(static_cast<std::uint64_t>(rank), rem);
    const std::uint64_t count = base + (static_cast<std::uint64_t>(rank) < rem ? 1 : 0);
    return {start, start + count};
}

// Canonical HEXA_8 outward face table matching cfd::mesh::kFaceTable
constexpr int kHexaFaceTable[6][4] = {
    {0, 3, 2, 1}, // Face 0: bottom (-z)
    {0, 1, 5, 4}, // Face 1: front  (-y)
    {1, 2, 6, 5}, // Face 2: right  (+x)
    {2, 3, 7, 6}, // Face 3: back   (+y)
    {3, 0, 4, 7}, // Face 4: left   (-x)
    {4, 5, 6, 7}  // Face 5: top    (+z)
};

struct ResolvedPatchFace {
    std::size_t block_idx{0};
    std::size_t local_face_idx{0}; // 0..5
    std::size_t Nu{0};
    std::size_t Nv{0};
    std::uint64_t face_cell_offset{0};
};

} // namespace

void write(const std::string& filepath,
           const GeneratorConfig& config,
           const MacroTopology& topo,
           MPI_Comm comm) {
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    check_cgns(cgp_pio_mode(CGP_COLLECTIVE), "cgp_pio_mode(CGP_COLLECTIVE)", comm);

    int fn = 0;
    check_cgns(cgp_open(filepath.c_str(), CG_MODE_WRITE, &fn), "cgp_open(WRITE)", comm);

    int base_id = 0;
    check_cgns(cg_base_write(fn, "Base_3D", 3, 3, &base_id), "cg_base_write", comm);

    const std::uint64_t total_nodes = topo.total_nodes();
    const std::uint64_t total_cells = topo.total_cells();

    int zone_id = 0;
    cgsize_t zone_sizes[3] = {
        static_cast<cgsize_t>(total_nodes),
        static_cast<cgsize_t>(total_cells),
        0
    };

    check_cgns(cg_zone_write(fn, base_id, "Zone1", zone_sizes, CGNS_ENUMV(Unstructured), &zone_id),
               "cg_zone_write", comm);

    // Coordinates write
    int cx_id = 0, cy_id = 0, cz_id = 0;
    check_cgns(cgp_coord_write(fn, base_id, zone_id, CGNS_ENUMV(RealDouble), "CoordinateX", &cx_id), "cgp_coord_write(X)", comm);
    check_cgns(cgp_coord_write(fn, base_id, zone_id, CGNS_ENUMV(RealDouble), "CoordinateY", &cy_id), "cgp_coord_write(Y)", comm);
    check_cgns(cgp_coord_write(fn, base_id, zone_id, CGNS_ENUMV(RealDouble), "CoordinateZ", &cz_id), "cgp_coord_write(Z)", comm);

    const auto [node_start, node_end] = decompose_1d(total_nodes, rank, nprocs);
    const std::uint64_t local_nodes_cnt = node_end - node_start;
    const bool has_nodes = (local_nodes_cnt > 0);

    std::vector<double> px(local_nodes_cnt);
    std::vector<double> py(local_nodes_cnt);
    std::vector<double> pz(local_nodes_cnt);

    for (std::uint64_t i = 0; i < local_nodes_cnt; ++i) {
        const std::uint64_t global_id = (node_start + 1) + i;
        const Vec3 coord = topo.evaluate_node_coord(global_id, config);
        px[i] = coord.x;
        py[i] = coord.y;
        pz[i] = coord.z;
    }

    const cgsize_t r_coord_s = has_nodes ? static_cast<cgsize_t>(node_start + 1) : 1;
    const cgsize_t r_coord_e = has_nodes ? static_cast<cgsize_t>(node_end) : 0;

    check_cgns(cgp_coord_write_data(fn, base_id, zone_id, cx_id, &r_coord_s, &r_coord_e, has_nodes ? px.data() : nullptr), "cgp_coord_write_data(X)", comm);
    check_cgns(cgp_coord_write_data(fn, base_id, zone_id, cy_id, &r_coord_s, &r_coord_e, has_nodes ? py.data() : nullptr), "cgp_coord_write_data(Y)", comm);
    check_cgns(cgp_coord_write_data(fn, base_id, zone_id, cz_id, &r_coord_s, &r_coord_e, has_nodes ? pz.data() : nullptr), "cgp_coord_write_data(Z)", comm);

    // Volume HEXA_8 section write
    cgsize_t current_global_eid = 1;
    const cgsize_t hex_start = current_global_eid;
    const cgsize_t hex_end   = static_cast<cgsize_t>(total_cells);
    current_global_eid = hex_end + 1;

    int hex_sec_id = 0;
    check_cgns(cgp_section_write(fn, base_id, zone_id, "Volume_HEXA",
                                 CGNS_ENUMV(HEXA_8), hex_start, hex_end, 0, &hex_sec_id),
               "cgp_section_write(HEXA_8)", comm);

    const auto [cell_start, cell_end] = decompose_1d(total_cells, rank, nprocs);
    const std::uint64_t local_cells_cnt = cell_end - cell_start;
    const bool has_cells = (local_cells_cnt > 0);

    std::vector<cgsize_t> hex_conn(local_cells_cnt * 8);
    for (std::uint64_t c = 0; c < local_cells_cnt; ++c) {
        const std::uint64_t global_cell_id = cell_start + c;
        const auto nodes = topo.get_cell_nodes(global_cell_id);
        for (std::size_t k = 0; k < 8; ++k) {
            hex_conn[c * 8 + k] = static_cast<cgsize_t>(nodes[k]);
        }
    }

    const cgsize_t r_hex_s = has_cells ? static_cast<cgsize_t>(cell_start + 1) : 1;
    const cgsize_t r_hex_e = has_cells ? static_cast<cgsize_t>(cell_end) : 0;

    check_cgns(cgp_elements_write_data(fn, base_id, zone_id, hex_sec_id, r_hex_s, r_hex_e, has_cells ? hex_conn.data() : nullptr),
               "cgp_elements_write_data(HEXA_8)", comm);

    // Boundary QUAD_4 sections write
    for (const auto& patch : config.boundaries) {
        std::vector<ResolvedPatchFace> resolved_faces;
        std::uint64_t patch_total_quads = 0;

        for (const auto& mf : patch.faces) {
            bool matched = false;
            for (std::size_t b = 0; b < config.blocks.size(); ++b) {
                const auto& blk = config.blocks[b];
                for (std::size_t f = 0; f < 6; ++f) {
                    const std::array<std::size_t, 4> bf_verts = {
                        blk.vertices[static_cast<std::size_t>(kHexaFaceTable[f][0])],
                        blk.vertices[static_cast<std::size_t>(kHexaFaceTable[f][1])],
                        blk.vertices[static_cast<std::size_t>(kHexaFaceTable[f][2])],
                        blk.vertices[static_cast<std::size_t>(kHexaFaceTable[f][3])]
                    };

                    bool face_match = false;
                    for (std::size_t shift = 0; shift < 4; ++shift) {
                        if (mf.vertices[0] == bf_verts[shift] &&
                            mf.vertices[1] == bf_verts[(shift + 1) % 4] &&
                            mf.vertices[2] == bf_verts[(shift + 2) % 4] &&
                            mf.vertices[3] == bf_verts[(shift + 3) % 4]) {
                            face_match = true;
                            break;
                        }
                    }

                    if (face_match) {
                        std::size_t Nu = 0, Nv = 0;
                        if (f == 0 || f == 5)      { Nu = blk.cells[0]; Nv = blk.cells[1]; }
                        else if (f == 1 || f == 3) { Nu = blk.cells[0]; Nv = blk.cells[2]; }
                        else                       { Nu = blk.cells[1]; Nv = blk.cells[2]; }

                        resolved_faces.push_back({b, f, Nu, Nv, patch_total_quads});
                        patch_total_quads += static_cast<std::uint64_t>(Nu * Nv);
                        matched = true;
                        break;
                    }
                }
                if (matched) break;
            }

            if (!matched) {
                mpi::fatal(comm, std::format("Patch '{}' face [{},{},{},{}] doesn't match any block face",
                                             patch.name, mf.vertices[0], mf.vertices[1], mf.vertices[2], mf.vertices[3]));
            }
        }

        const cgsize_t patch_start = current_global_eid;
        const cgsize_t patch_end   = static_cast<cgsize_t>(static_cast<std::uint64_t>(patch_start) + patch_total_quads - 1);
        current_global_eid = patch_end + 1;

        int bnd_sec_id = 0;
        check_cgns(cgp_section_write(fn, base_id, zone_id, patch.name.c_str(),
                                     CGNS_ENUMV(QUAD_4), patch_start, patch_end, 0, &bnd_sec_id),
                   "cgp_section_write(QUAD_4)", comm);

        const auto [quad_start, quad_end] = decompose_1d(patch_total_quads, rank, nprocs);
        const std::uint64_t local_quads_cnt = quad_end - quad_start;
        const bool has_quads = (local_quads_cnt > 0);

        std::vector<cgsize_t> quad_conn(local_quads_cnt * 4);

        for (std::uint64_t q = 0; q < local_quads_cnt; ++q) {
            const std::uint64_t glob_q = quad_start + q;

            auto it = std::upper_bound(resolved_faces.begin(), resolved_faces.end(), glob_q,
                                       [](std::uint64_t val, const ResolvedPatchFace& f) {
                                           return val < f.face_cell_offset;
                                       });
            if (it != resolved_faces.begin()) --it;

            const auto& rf = *it;
            const std::uint64_t local_sub_q = glob_q - rf.face_cell_offset;
            const std::size_t u = static_cast<std::size_t>(local_sub_q % rf.Nu);
            const std::size_t v = static_cast<std::size_t>(local_sub_q / rf.Nu);

            const auto& blk = config.blocks[rf.block_idx];
            const std::size_t Nx = blk.cells[0];
            const std::size_t Ny = blk.cells[1];
            const std::size_t Nz = blk.cells[2];

            std::size_t ci = 0, cj = 0, ck = 0;
            switch (rf.local_face_idx) {
                case 0: ci = u;      cj = v;      ck = 0;      break; // bottom
                case 1: ci = u;      cj = 0;      ck = v;      break; // front
                case 2: ci = Nx - 1; cj = u;      ck = v;      break; // right
                case 3: ci = u;      cj = Ny - 1; ck = v;      break; // back
                case 4: ci = 0;      cj = u;      ck = v;      break; // left
                case 5: ci = u;      cj = v;      ck = Nz - 1; break; // top
            }

            const auto c_nodes = topo.get_block_cell_nodes(rf.block_idx, ci, cj, ck);
            for (std::size_t k = 0; k < 4; ++k) {
                quad_conn[q * 4 + k] = static_cast<cgsize_t>(c_nodes[static_cast<std::size_t>(kHexaFaceTable[rf.local_face_idx][k])]);
            }
        }

        const cgsize_t r_bnd_s = has_quads ? static_cast<cgsize_t>(static_cast<std::size_t>(patch_start) + quad_start) : 1;
        const cgsize_t r_bnd_e = has_quads ? static_cast<cgsize_t>(static_cast<std::size_t>(patch_start) + quad_end - 1) : 0;

        check_cgns(cgp_elements_write_data(fn, base_id, zone_id, bnd_sec_id, r_bnd_s, r_bnd_e, has_quads ? quad_conn.data() : nullptr),
                   "cgp_elements_write_data(QUAD_4)", comm);

        int boco_id = 0;
        const cgsize_t range[2] = {patch_start, patch_end};

        check_cgns(cg_boco_write(fn, base_id, zone_id, patch.name.c_str(), CGNS_ENUMV(BCTypeUserDefined),
                                 CGNS_ENUMV(PointRange), 2, range, &boco_id),
                   "cg_boco_write", comm);

        check_cgns(cg_boco_gridlocation_write(fn, base_id, zone_id, boco_id, CGNS_ENUMV(FaceCenter)),
                   "cg_boco_gridlocation_write", comm);
    }

    check_cgns(cgp_close(fn), "cgp_close", comm);

    if (rank == 0) {
        mpi::log_stat("Parallel CGNS mesh successfully written to '%s'", filepath.c_str());
    }
}

} // namespace cfd::mesh_generator