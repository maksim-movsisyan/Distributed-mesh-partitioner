#include "cfd/io/cgns/cgns_writer.hpp"

#include <cgnstypes.h>
#include <cgnslib.h>
#include <mpi.h>

#include <cstdint>
#include <string>
#include <vector>
#include <array>

#include "cfd/io/cgns/cgns_reader.hpp"
#include "cfd/mesh/cgnstables.hpp"
#include "cfd/mpi/log.hpp"

namespace cfd::io::cgns {

namespace {

inline ElementType cell_type_to_cgns(mesh::CellType t) {
    switch (t) {
        case mesh::CellType::TET:   return CGNS_ENUMV(TETRA_4);
        case mesh::CellType::PYRA:  return CGNS_ENUMV(PYRA_5);
        case mesh::CellType::PRISM: return CGNS_ENUMV(PENTA_6);
        case mesh::CellType::HEXA:  return CGNS_ENUMV(HEXA_8);
        case mesh::CellType::TRI:   return CGNS_ENUMV(TRI_3);
        case mesh::CellType::QUAD:  return CGNS_ENUMV(QUAD_4);
        default:                    return CGNS_ENUMV(ElementTypeNull);
    }
}

inline BoundaryConditionType bc_type_from_string(const std::string& s) {
    if (s == "SymmetryPlane") return CGNS_ENUMV(BCTypeNull); // Written as UserDefined or Family
    if (s == "BCWall" || s == "Wall") return CGNS_ENUMV(BCWall);
    if (s == "BCInflow" || s == "Inflow") return CGNS_ENUMV(BCInflow);
    if (s == "BCOutflow" || s == "Outflow") return CGNS_ENUMV(BCOutflow);
    return CGNS_ENUMV(BCTypeUserDefined);
}

} // namespace

void write_cgns_parallel(const std::string& path, const mesh::RawMesh& m) {
    const MPI_Comm comm = m.comm;
    const int rank = m.rank;
    // 1. Initialize collective parallel I/O and open file
    check(cgp_pio_mode(CGP_COLLECTIVE), "cgp_pio_mode(CGP_COLLECTIVE)", comm);

    int f_id = 0;
    check(cgp_open(path.c_str(), CG_MODE_WRITE, &f_id), "cgp_open(WRITE)", comm);

    // 2. Base & Zone Definition (Always 3D)
    int base_id = 0;
    check(cg_base_write(f_id, "Base", 3, 3, &base_id), "cg_base_write", comm);

    int zone_id = 0;
    cgsize_t zone_sizes[3] = {
        static_cast<cgsize_t>(m.n_nodes_g),
        static_cast<cgsize_t>(m.n_cells_g),
        0 // Unstructured boundary vertex count
    };
    check(cg_zone_write(f_id, base_id, "Zone1", zone_sizes, CGNS_ENUMV(Unstructured), &zone_id),
          "cg_zone_write", comm);

    // 3. Parallel Coordinate Arrays Creation & Collective Write
    int cx_id = 0, cy_id = 0, cz_id = 0;
    check(cgp_coord_write(f_id, base_id, zone_id, CGNS_ENUMV(RealDouble), "CoordinateX", &cx_id), "cgp_coord_write(X)", comm);
    check(cgp_coord_write(f_id, base_id, zone_id, CGNS_ENUMV(RealDouble), "CoordinateY", &cy_id), "cgp_coord_write(Y)", comm);
    check(cgp_coord_write(f_id, base_id, zone_id, CGNS_ENUMV(RealDouble), "CoordinateZ", &cz_id), "cgp_coord_write(Z)", comm);

    const GlobalIndex nb = m.my_node_begin();
    const GlobalIndex ne = m.my_node_end();
    const bool has_nodes = (nb < ne);

    const cgsize_t r_coord_s = has_nodes ? static_cast<cgsize_t>(nb + 1) : 1;
    const cgsize_t r_coord_e = has_nodes ? static_cast<cgsize_t>(ne) : 0;

    const double* px = has_nodes ? m.my_node_coords_x.data() : nullptr;
    const double* py = has_nodes ? m.my_node_coords_y.data() : nullptr;
    const double* pz = has_nodes ? m.my_node_coords_z.data() : nullptr;

    check(cgp_coord_write_data(f_id, base_id, zone_id, cx_id, &r_coord_s, &r_coord_e, px), "cgp_coord_write_data(X)", comm);
    check(cgp_coord_write_data(f_id, base_id, zone_id, cy_id, &r_coord_s, &r_coord_e, py), "cgp_coord_write_data(Y)", comm);
    check(cgp_coord_write_data(f_id, base_id, zone_id, cz_id, &r_coord_s, &r_coord_e, pz), "cgp_coord_write_data(Z)", comm);

    GlobalIndex current_global_eid = 1;

    // 4. Volume Elements: Partition by Type into Homogeneous Sections
    const std::array<mesh::CellType, 4> vol_types = {
        mesh::CellType::HEXA,
        mesh::CellType::PRISM,
        mesh::CellType::TET,
        mesh::CellType::PYRA
    };

    const LocalIndex nl = m.n_local_cells();
    std::vector<LocalIndex> local_cells_of_type;
    local_cells_of_type.reserve(static_cast<std::size_t>(nl));

    for (mesh::CellType t : vol_types) {
        local_cells_of_type.clear();
        for (LocalIndex i = 0; i < nl; ++i) {
            if (m.ctype[static_cast<std::size_t>(i)] == t) {
                local_cells_of_type.push_back(i);
            }
        }

        const uint64_t loc_cnt = static_cast<uint64_t>(local_cells_of_type.size());
        uint64_t glob_cnt = 0;
        MPI_Allreduce(&loc_cnt, &glob_cnt, 1, MPI_UINT64_T, MPI_SUM, comm);

        if (glob_cnt == 0) continue;

        uint64_t sec_offset = 0;
        MPI_Exscan(&loc_cnt, &sec_offset, 1, MPI_UINT64_T, MPI_SUM, comm);
        if (rank == 0) sec_offset = 0;

        const cgsize_t sec_start = static_cast<cgsize_t>(current_global_eid);
        const cgsize_t sec_end = static_cast<cgsize_t>(static_cast<uint64_t>(current_global_eid) + glob_cnt - 1);
        current_global_eid = static_cast<GlobalIndex>(sec_end + 1);

        const int npt = mesh::kNodesPerType[static_cast<std::size_t>(t)];
        std::vector<cgsize_t> conn(loc_cnt * static_cast<std::size_t>(npt));

        for (std::size_t i = 0; i < loc_cnt; ++i) {
            const LocalIndex c = local_cells_of_type[i];
            const LocalIndex off_start = m.cnodes_offsets[static_cast<std::size_t>(c)];
            for (int k = 0; k < npt; ++k) {
                // CGNS requires 1-based node connectivity
                conn[i * static_cast<std::size_t>(npt) + static_cast<std::size_t>(k)] =
                    static_cast<cgsize_t>(m.cnodes[static_cast<std::size_t>(off_start + k)] + 1);
            }
        }

        int sec_id = 0;
        const std::string sec_name = std::string("Volume_") + mesh::cell_type_name(t);
        check(cgp_section_write(f_id, base_id, zone_id, sec_name.c_str(),
                                cell_type_to_cgns(t), sec_start, sec_end, 0, &sec_id),
              "cgp_section_write(vol)", comm);

        const bool has_elems = (loc_cnt > 0);
        const cgsize_t rs = has_elems ? static_cast<cgsize_t>(static_cast<std::size_t>(sec_start) + sec_offset) : 1;
        const cgsize_t re = has_elems ? static_cast<cgsize_t>(static_cast<std::size_t>(sec_start) + sec_offset + loc_cnt - 1) : 0;
        const cgsize_t* pdata = has_elems ? conn.data() : nullptr;

        check(cgp_elements_write_data(f_id, base_id, zone_id, sec_id, rs, re, pdata),
              "cgp_elements_write_data(vol)", comm);
    }

    // 5. Boundary Sections: Group by Patch and Surface Type (TRI / QUAD)
    int bc_index = 1;

    for (std::size_t p = 0; p < m.patch_list.size(); ++p) {
        const auto& patch_meta = m.patch_list[p];

        const std::array<int, 2> face_node_counts = {4, 3}; // QUAD_4, TRI_3

        for (int npt : face_node_counts) {
            std::vector<const mesh::SurfElem*> local_patch_faces;
            for (const auto& se : m.surf_elems) {
                if (se.patch == static_cast<PatchId>(p)) {
                    const int actual_nodes = (se.key.v[3] == kInvalidGlobalIndex) ? 3 : 4;
                    if (actual_nodes == npt) {
                        local_patch_faces.push_back(&se);
                    }
                }
            }

            const uint64_t loc_cnt = static_cast<uint64_t>(local_patch_faces.size());
            uint64_t glob_cnt = 0;
            MPI_Allreduce(&loc_cnt, &glob_cnt, 1, MPI_UINT64_T, MPI_SUM, comm);

            if (glob_cnt == 0) continue;

            uint64_t sec_offset = 0;
            MPI_Exscan(&loc_cnt, &sec_offset, 1, MPI_UINT64_T, MPI_SUM, comm);
            if (rank == 0) sec_offset = 0;

            const cgsize_t sec_start = static_cast<cgsize_t>(current_global_eid);
            const cgsize_t sec_end = static_cast<cgsize_t>(static_cast<std::size_t>(current_global_eid) + glob_cnt - 1);
            current_global_eid = static_cast<GlobalIndex>(sec_end + 1);

            std::vector<cgsize_t> conn(loc_cnt * static_cast<std::size_t>(npt));
            for (std::size_t i = 0; i < loc_cnt; ++i) {
                for (int k = 0; k < npt; ++k) {
                    conn[i * static_cast<std::size_t>(npt) + static_cast<std::size_t>(k)] =
                        static_cast<cgsize_t>(local_patch_faces[i]->key.v[static_cast<std::size_t>(k)] + 1);
                }
            }

            const ElementType etype = (npt == 3) ? CGNS_ENUMV(TRI_3) : CGNS_ENUMV(QUAD_4);
            std::string sec_name = patch_meta.first;
            if (npt == 3) sec_name += "_tri";

            int sec_id = 0;
            check(cgp_section_write(f_id, base_id, zone_id, sec_name.c_str(),
                                    etype, sec_start, sec_end, 0, &sec_id),
                  "cgp_section_write(bndry)", comm);

            const bool has_elems = (loc_cnt > 0);
            const cgsize_t rs = has_elems ? static_cast<cgsize_t>(static_cast<std::size_t>(sec_start) + sec_offset) : 1;
            const cgsize_t re = has_elems ? static_cast<cgsize_t>(static_cast<std::size_t>(sec_start) + sec_offset + loc_cnt - 1) : 0;
            const cgsize_t* pdata = has_elems ? conn.data() : nullptr;

            check(cgp_elements_write_data(f_id, base_id, zone_id, sec_id, rs, re, pdata),
                  "cgp_elements_write_data(bndry)", comm);

            // Write ZoneBC node for this boundary section
            int boco_id = 0;
            const cgsize_t range[2] = {sec_start, sec_end};
            const BoundaryConditionType bctype = bc_type_from_string(patch_meta.second);

            check(cg_boco_write(f_id, base_id, zone_id, sec_name.c_str(), bctype,
                                CGNS_ENUMV(PointRange), 2, range, &boco_id),
                  "cg_boco_write", comm);

            check(cg_boco_gridlocation_write(f_id, base_id, zone_id, boco_id, CGNS_ENUMV(FaceCenter)),
                  "cg_boco_gridlocation_write", comm);

            ++bc_index;
        }
    }

    check(cgp_close(f_id), "cgp_close", comm);

    if (rank == 0) {
        mpi::log_stat("Successfully written parallel CGNS mesh to '%s'", path.c_str());
    }
}

} // namespace cfd::io::cgns