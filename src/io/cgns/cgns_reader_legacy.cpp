#include "cfd/io/cgns/cgns_reader.hpp"

#include <algorithm>
#include <cstdio>
#include <unordered_map>
#include <vector>
#include <cstddef>

#include "cfd/core/types.hpp"
#include "cfd/mesh/cgnstables.hpp"
#include "cfd/mpi/log.hpp"
#include "cfd/mpi/mpi_util.hpp"

namespace cfd::io::cgns {

namespace {

mesh::CellType cgns_elem_to_type(ElementType e, MPI_Comm comm) {
    switch (e) {
        case CGNS_ENUMV(TETRA_4): return mesh::CellType::TET;
        case CGNS_ENUMV(PYRA_5): return mesh::CellType::PYRA;
        case CGNS_ENUMV(PENTA_6): return mesh::CellType::PRISM;
        case CGNS_ENUMV(HEXA_8): return mesh::CellType::HEXA;
        case CGNS_ENUMV(TRI_3): return mesh::CellType::TRI;
        case CGNS_ENUMV(QUAD_4): return mesh::CellType::QUAD;
        default:
            mpi::fatal(comm, "Unsupported CGNS element type: " + std::to_string(static_cast<int>(e)) +
                        " (MIXED and high-order elements are not supported)");
            return mesh::CellType::HEXA;
    }
}


} // namespace

mesh::RawMesh read_cgns_parallel(const std::string& path, MPI_Comm comm) {
    // get rank index and total process count
    int nprocs = 0, rank = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);


    // Set parallel collective I/O mode across all ranks
    check(cgp_pio_mode(CGP_COLLECTIVE), "cgp_pio_mode(CGP_COLLECTIVE)");

    // Open CGNS file in parallel via RAII wrapper
    File file(path, comm);
    const int f_id = file.id();


    // Initialize result
    mesh::RawMesh m;
    m.nprocs = nprocs;
    m.rank = rank;
    m.comm = comm;


    // Read file and general metadata
    check(cg_version(f_id, &m.gfm.cgns_version), "cg_version");
    check(cg_precision(f_id, &m.gfm.file_integer_precision), "cg_precision");
    check(cg_get_file_type(f_id, &m.gfm.storage_type), "cg_get_file_type");

    const char* storage_str = "Unknown";
    if (m.gfm.storage_type == 1) storage_str = "ADF";
    else if (m.gfm.storage_type == 2) storage_str = "HDF5";
    else if (m.gfm.storage_type == 0) storage_str = "None/Error";
    mpi::log_stat("CGNS: Version='%.2f', Integer precision='%d bits', Storage type='%s'", 
                m.gfm.cgns_version, m.gfm.file_integer_precision, storage_str);


    // Base metadata
    int nbases = 0;
    check(cg_nbases(f_id, &nbases), "cg_nbases");
    if (nbases != 1) { mpi::fatal(comm, "Expected exactly 1 base in CGNS file"); }

    const int B = 1;
    int celldim = 0, physdim = 0;
    char basename[33] = "";
    check(cg_base_read(f_id, B, basename, &celldim, &physdim), "cg_base_read");
    if (celldim != 3 || physdim != 3) { mpi::fatal(comm, "Base must be 3D (CellDim=3, PhysDim=3)"); }


    // Zone metadata 
    int nzones = 0;
    check(cg_nzones(f_id, B, &nzones), "cg_nzones");
    if (nzones != 1) { mpi::fatal(comm, "Expected exactly 1 zone in CGNS file"); }

    const int Z = 1;
    ZoneType zonetype;
    check(cg_zone_type(f_id, B, Z, &zonetype), "cg_zone_type");
    if (zonetype != CGNS_ENUMV(Unstructured)) { mpi::fatal(comm, "Only Unstructured zones are supported"); }

    char zonename[33] = "";
    cgsize_t sizes[9] = {0};
    check(cg_zone_read(f_id, B, Z, zonename, sizes), "cg_zone_read");
    m.n_nodes_g = static_cast<GlobalIndex>(sizes[0]);
    m.n_cells_g = static_cast<GlobalIndex>(sizes[1]);

    mpi::log_stat("CGNS: Base='%s', Zone='%s', Total Nodes=%lld, Total Cells=%lld", 
                      basename, zonename, static_cast<long long>(m.n_nodes_g), static_cast<long long>(m.n_cells_g));


    // Read Element Section Metadata
    int nsecs = 0;
    check(cg_nsections(f_id, B, Z, &nsecs), "cg_nsections");

    // loop over all sections
    for (int S = 1; S <= nsecs; ++S) {
        char secname[33] = "";
        ElementType etype;
        cgsize_t start = 0, end = 0;
        int nbndry = 0, parent_flag = 0;

        check(cg_section_read(f_id, B, Z, S, secname,
              &etype, &start, &end, &nbndry, &parent_flag), 
              "cg_section_read");

        // get section elements type (+ check if element type is supported)
        const mesh::CellType t = cgns_elem_to_type(etype, comm);

        mesh::SectionMeta sm;
        sm.name = secname;
        sm.type = t;
        sm.start = static_cast<GlobalIndex>(start);
        sm.end = static_cast<GlobalIndex>(end);
        sm.sec_idx = S;

        if (mesh::is_volume_type(t)) {
            m.vol_secs.push_back(sm);
        } else {
            m.surf_secs.push_back(sm);
        }
    } // end loop over all sections

    // Sort volume sections by start ID for contiguous global cell numbering
    std::sort(m.vol_secs.begin(), m.vol_secs.end(),
              [](const mesh::SectionMeta& a, const mesh::SectionMeta& b) noexcept { 
                  return a.start < b.start; });

    GlobalIndex total_cells = 0;
    // loop over volume sections for global offsets
    for (auto& s : m.vol_secs) {
        s.cell_offset = total_cells;
        total_cells += (s.end - s.start + 1);
    } // end loop over volume sections for global offsets

    if (m.n_cells_g == 0 || total_cells != m.n_cells_g) { mpi::fatal(comm, "Mismatch or zero volume cells in CGNS zone"); }

    mpi::log_stat("CGNS: Volume sections=%zu, Total Cells=%lld, Boundary sections=%zu", 
                      m.vol_secs.size(), static_cast<long long>(m.n_cells_g), m.surf_secs.size());


    // Read Boundary Conditions (ZoneBC)
    int nbocos = 0;
    check(cg_nbocos(f_id, B, Z, &nbocos), "cg_nbocos");

    // loop over all boundary conditions
    for (int bc = 1; bc <= nbocos; ++bc) {
        char bcname[33] = "";
        BoundaryConditionType btype;
        PointSetType ptype;
        cgsize_t npnts = 0, normallistsize = 0;
        int normalidx[3] = {0, 0, 0};
        DataType ndtype;
        int ndataset = 0;

        check(cg_boco_info(f_id, B, Z, bc, bcname, &btype, &ptype, &npnts, normalidx, 
                           &normallistsize, &ndtype, &ndataset), "cg_boco_info");

        GridLocation loc;
        check(cg_boco_gridlocation_read(f_id, B, Z, bc, &loc), "cg_boco_gridlocation_read");

        if (loc != CGNS_ENUMV(FaceCenter)) {
            if (rank == 0) {
                mpi::log_warn_rank("BC '%s': GridLocation != FaceCenter, skipped", bcname);
            }
            continue;
        }

        std::vector<cgsize_t> pnts(static_cast<std::size_t>(npnts));
        if (npnts > 0) {
            check(cg_boco_read(f_id, B, Z, bc, pnts.data(), nullptr), "cg_boco_read");
        }

        mesh::BCMeta bm;
        bm.cgns_type = cg_BCTypeName(btype);

        if (ptype == CGNS_ENUMV(PointRange) && npnts == 2) {
            for (cgsize_t e = pnts[0]; e <= pnts[1]; ++e) {
                bm.eids.push_back(static_cast<GlobalIndex>(e));
            }
        } else {
            for (cgsize_t i = 0; i < npnts; ++i) {
                bm.eids.push_back(static_cast<GlobalIndex>(pnts[static_cast<std::size_t>(i)]));
            }
        }

        char fam[33] = "";
        if (cg_goto(f_id, B, "Zone_t", Z, "ZoneBC_t", 1, "BC_t", bc, "end") == CG_OK) {
            if (cg_famname_read(fam) == CG_OK && fam[0] != '\0') {
                bm.name = fam;
            }
        }
        if (bm.name.empty()) {
            bm.name = bcname;
        }

        m.bcs.push_back(std::move(bm));
    } // end loop over all boundary conditions


    // Build global patch list
    // loop over processed boundary conditions
    for (const auto& b : m.bcs) {
        m.patch_list.push_back({b.name, b.cgns_type});
        mpi::log_stat("CGNS: BC '%s' Type=%s Faces=%zu -> PatchId=%d", 
                          b.name.c_str(), b.cgns_type.c_str(), b.eids.size(), 
                          static_cast<int>(m.patch_list.size() - 1));
    } // end loop over processed boundary conditions


    // Read Surface Sections (Collective)
    {
        std::unordered_map<GlobalIndex, PatchId> eid2patch;
        // loop over bc metadata for fast map lookup
        for (std::size_t p = 0; p < m.bcs.size(); ++p) {
            for (GlobalIndex e : m.bcs[p].eids) {
                eid2patch[e] = static_cast<PatchId>(p);
            }
        } // end loop over bc metadata for fast map lookup

        // loop over all surface sections
        for (const auto& s : m.surf_secs) {
            const GlobalIndex sec_n = s.end - s.start + 1;

            // replicated on each rank surface section contiguous distribution
            const std::vector<GlobalIndex> d = mpi::block_displ(sec_n, static_cast<std::size_t>(nprocs));

            // get local rank range (0 based, exclusive)
            const GlobalIndex lo = d[static_cast<std::size_t>(rank)];
            const GlobalIndex hi = d[static_cast<std::size_t>(rank) + 1];

            // get number of nodes per section element type
            const std::size_t npt = static_cast<std::size_t>(mesh::kNodesPerType[static_cast<std::size_t>(s.type)]);
            const std::size_t local_count = (lo < hi) ? static_cast<std::size_t>(hi - lo) : 0;
            std::vector<cgsize_t> buf(local_count * npt);

            // In collective mode, all ranks must participate; 0-sized reads pass rs=1, re=0
            const cgsize_t rs = (lo < hi) ? static_cast<cgsize_t>(s.start + lo) : 1;
            const cgsize_t re = (lo < hi) ? static_cast<cgsize_t>(s.start + hi - 1) : 0;

            check(cgp_elements_read_data(f_id, B, Z, s.sec_idx, rs, re, buf.data()),
                  "cgp_elements_read_data(surface)");

            // skip blank ranges
            if (lo >= hi) continue;

            // loop over local surface elements
            for (std::size_t i = 0; i < local_count; ++i) {
                mesh::SurfElem se;
                for (std::size_t k = 0; k < npt; ++k) {
                    se.key.v[k] = static_cast<GlobalIndex>(buf[i * npt + k] - 1); // back to 0-based
                }
                std::sort(se.key.v.begin(), se.key.v.end());

                se.eid = s.start + lo + static_cast<GlobalIndex>(i);
                const auto it = eid2patch.find(se.eid);
                se.patch = (it != eid2patch.end()) ? it->second : kInvalidPatchId;

                m.surf_elems.push_back(se);
            } // end loop over local surface elements
        } // end loop over all surface sections
    }


    // Globla Block Distributions for Volume Cells and Nodes (replicated on each rank)
    m.cell_displ = mpi::block_displ(m.n_cells_g, static_cast<std::size_t>(nprocs));
    m.node_displ = mpi::block_displ(m.n_nodes_g, static_cast<std::size_t>(nprocs));


    // Read Volume Cell Connectivity (Collective)
    const LocalIndex nl = m.n_local_cells();
    m.ctype.reserve(static_cast<std::size_t>(nl));

    // Calculate total local connectivity size
    std::size_t total_conn_entries = 0;
    // loop over volume sections to estimate connectivity memory
    for (const auto& s : m.vol_secs) {
        const GlobalIndex sec_n = s.end - s.start + 1;
        const GlobalIndex cb = m.cell_displ[static_cast<std::size_t>(rank)];
        const GlobalIndex ce = m.cell_displ[static_cast<std::size_t>(rank) + 1];
        const GlobalIndex lo = std::max(s.cell_offset, cb);
        const GlobalIndex hi = std::min(s.cell_offset + sec_n, ce);

        if (lo < hi) {
            const auto npt = static_cast<std::size_t>(mesh::kNodesPerType[static_cast<std::size_t>(s.type)]);
            total_conn_entries += static_cast<std::size_t>(hi - lo) * npt;
        }
    } // end loop over volume sections to estimate connectivity memory
    m.cnodes.reserve(total_conn_entries);
    m.cnodes_offsets.reserve(static_cast<std::size_t>(nl) + 1);
    m.cnodes_offsets.push_back(0);

    // loop over all volume sections
    for (const auto& s : m.vol_secs) {
        const GlobalIndex sec_n = s.end - s.start + 1;
        const GlobalIndex cb = m.cell_displ[static_cast<std::size_t>(rank)];
        const GlobalIndex ce = m.cell_displ[static_cast<std::size_t>(rank) + 1];
        const GlobalIndex lo = std::max(s.cell_offset, cb);
        const GlobalIndex hi = std::min(s.cell_offset + sec_n, ce);

        const auto npt = static_cast<std::size_t>(mesh::kNodesPerType[static_cast<std::size_t>(s.type)]);
        const std::size_t local_count = (lo < hi) ? static_cast<std::size_t>(hi - lo) : 0;
        std::vector<cgsize_t> buf(local_count * npt);

        // In collective mode, all ranks must participate; 0-sized reads pass rs=1, re=0
        const cgsize_t rs = (lo < hi) ? static_cast<cgsize_t>(s.start + (lo - s.cell_offset)) : 1;
        const cgsize_t re = (lo < hi) ? static_cast<cgsize_t>(s.start + (hi - 1 - s.cell_offset)) : 0;

        check(cgp_elements_read_data(f_id, B, Z, s.sec_idx, rs, re, buf.data()),
              "cgp_elements_read_data(volume)");

        if (lo >= hi) continue;

        // loop over local section cells
        for (std::size_t i = 0; i < local_count; ++i) {
            m.ctype.push_back(s.type);
            for (std::size_t k = 0; k < npt; ++k) {
                // CGNS 1-based node IDs converted to 0-based
                m.cnodes.push_back(static_cast<GlobalIndex>(buf[i * npt + k] - 1));
            }
            m.cnodes_offsets.push_back(static_cast<LocalIndex>(m.cnodes.size()));
        } // end loop over local section cells
    } // end loop over all volume sections

    if (m.cnodes_offsets.size() != static_cast<std::size_t>(nl) + 1||
        static_cast<std::size_t>(m.cnodes_offsets[static_cast<std::size_t>(nl)])
            != static_cast<std::size_t>(total_conn_entries)) {
                mpi::fatal(comm, "Consistency error in raw mesh cell nodes CSR table");
        }

    if (m.ctype.size() != static_cast<std::size_t>(nl)) {
        mpi::fatal(comm, "Mismatch in read volume cells: got " + std::to_string(m.ctype.size()) +
                    ", expected " + std::to_string(nl));
    }

    // Read Local Node Slice Coordinates in SoA Layout (Collective)
    const GlobalIndex nb = m.my_node_begin();
    const GlobalIndex ne = m.my_node_end();
    const std::size_t nmy = (nb < ne) ? static_cast<std::size_t>(ne - nb) : 0;

    m.my_node_coords_x.resize(nmy);
    m.my_node_coords_y.resize(nmy);
    m.my_node_coords_z.resize(nmy);

    // In collective mode, all ranks participate with non-blocking/empty bounds
    cgsize_t rs = (nmy > 0) ? static_cast<cgsize_t>(nb + 1) : 1;
    cgsize_t re = (nmy > 0) ? static_cast<cgsize_t>(ne) : 0;

    // Coordinate X (dir = 1)
    check(cgp_coord_read_data(f_id, B, Z, 1, &rs, &re, m.my_node_coords_x.data()),
          "cgp_coord_read_data(X)");

    // Coordinate Y (dir = 2)
    check(cgp_coord_read_data(f_id, B, Z, 2, &rs, &re, m.my_node_coords_y.data()),
          "cgp_coord_read_data(Y)");

    // Coordinate Z (dir = 3)
    check(cgp_coord_read_data(f_id, B, Z, 3, &rs, &re, m.my_node_coords_z.data()),
          "cgp_coord_read_data(Z)");

    mpi::log_rank("CGNS Rank %d: Local Cells=%d, Local Nodes=%zu", 
                  rank, nl, nmy);

    return m;
}

} // namespace cfd::io::cgns