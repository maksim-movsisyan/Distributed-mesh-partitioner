#include "cfd/io/cgns/cgns_reader.hpp"
#include <cgnstypes.h>
#include <cgns_io.h>
#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
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
        case CGNS_ENUMV(PYRA_5):  return mesh::CellType::PYRA;
        case CGNS_ENUMV(PENTA_6): return mesh::CellType::PRISM;
        case CGNS_ENUMV(HEXA_8):  return mesh::CellType::HEXA;
        case CGNS_ENUMV(TRI_3):   return mesh::CellType::TRI;
        case CGNS_ENUMV(QUAD_4):  return mesh::CellType::QUAD;
        default:
            mpi::fatal(comm, "Unsupported homogeneous CGNS element type: " + 
                       std::to_string(static_cast<int>(e)));
            return mesh::CellType::HEXA;
    }
}

inline std::pair<mesh::CellType, int> parse_cgns_element_header(ElementType e, MPI_Comm comm) {
    switch (e) {
        case CGNS_ENUMV(TETRA_4): return {mesh::CellType::TET, 4};
        case CGNS_ENUMV(PYRA_5):  return {mesh::CellType::PYRA, 5};
        case CGNS_ENUMV(PENTA_6): return {mesh::CellType::PRISM, 6};
        case CGNS_ENUMV(HEXA_8):  return {mesh::CellType::HEXA, 8};
        case CGNS_ENUMV(TRI_3):   return {mesh::CellType::TRI, 3};
        case CGNS_ENUMV(QUAD_4):  return {mesh::CellType::QUAD, 4};
        default:
            mpi::fatal(comm, "Unsupported element type in MIXED section: " + 
                       std::to_string(static_cast<int>(e)));
            return {mesh::CellType::HEXA, 8};
    }
}

inline void sort_face_key_nodes(mesh::FaceKey& key, int npt) noexcept {
    if (npt == 3) {
        if (key.v[0] > key.v[1]) std::swap(key.v[0], key.v[1]);
        if (key.v[1] > key.v[2]) std::swap(key.v[1], key.v[2]);
        if (key.v[0] > key.v[1]) std::swap(key.v[0], key.v[1]);
    } else if (npt == 4) {
        std::sort(key.v.begin(), key.v.end());
    }
}

// Low-level metadata query for array width on disk ("I4" / "I8")
int mixed_array_disk_bytes(int f_id, int B, int Z, int sec_idx, const char* array_name, MPI_Comm comm) {
    int cgio = 0;
    check(cg_get_cgio(f_id, &cgio), "cg_get_cgio(MIXED)", comm);

    double node = 0;
    check(cgio_get_root_id(cgio, &node), "cgio_get_root_id(MIXED)", comm);

    // root -> B-th CGNSBase_t -> Z-th Zone_t -> S-th Elements_t
    const struct {
        const char* label;
        int index;
    } path[] = {
        {"CGNSBase_t", B},
        {"Zone_t", Z},
        {"Elements_t", sec_idx},
    };
    for (const auto& step : path) {
        int nchildren = 0;
        check(cgio_number_children(cgio, node, &nchildren), "cgio_number_children(MIXED)", comm);

        std::vector<double> children(static_cast<std::size_t>(nchildren));
        int nret = 0;
        check(cgio_children_ids(cgio, node, 1, nchildren, &nret, children.data()),
              "cgio_children_ids(MIXED)", comm);

        int seen = 0;
        bool found = false;
        for (const double child : children) {
            char label[33] = "";
            check(cgio_get_label(cgio, child, label), "cgio_get_label(MIXED)", comm);
            if (std::strcmp(label, step.label) == 0 && ++seen == step.index) {
                node = child;
                found = true;
                break;
            }
        }
        if (!found) {
            mpi::fatal(comm, std::string("CGNS MIXED: cgio tree has no ") +
                             step.label + " #" + std::to_string(step.index));
        }
    }

    double array_id = 0;
    if (cgio_get_node_id(cgio, node, array_name, &array_id) != CG_OK) {
        return 0;
    }

    char data_type[8] = "";
    check(cgio_get_data_type(cgio, array_id, data_type), "cgio_get_data_type(MIXED)", comm);
    if (std::strcmp(data_type, "I4") == 0) { return 4; }
    if (std::strcmp(data_type, "I8") == 0) { return 8; }

    mpi::fatal(comm, std::string("CGNS MIXED: unexpected on-disk data type '") +
                     data_type + "' of " + array_name);
    return 0;
}

// Normalizes mixed array data to cgsize_t avoiding HDF5 memory buffer overflow
template <typename CgpCall>
std::vector<cgsize_t> read_mixed_width_safe(std::size_t count, int width_bytes, MPI_Comm comm, CgpCall&& cgp_call) {
    std::vector<cgsize_t> out(count);

    if (width_bytes == static_cast<int>(sizeof(cgsize_t))) {
        cgp_call(out.data());
        return out;
    }

    if (width_bytes == 8 && sizeof(cgsize_t) == 4) {
        std::vector<std::int64_t> raw(count);
        cgp_call(reinterpret_cast<cgsize_t*>(raw.data()));
        for (std::size_t i = 0; i < count; ++i) {
            out[i] = static_cast<cgsize_t>(raw[i]);
        }
        return out;
    }

    if (width_bytes == 4 && sizeof(cgsize_t) == 8) {
        std::vector<std::int32_t> raw(count);
        cgp_call(reinterpret_cast<cgsize_t*>(raw.data()));
        for (std::size_t i = 0; i < count; ++i) {
            out[i] = static_cast<cgsize_t>(raw[i]);
        }
        return out;
    }

    mpi::fatal(comm, "CGNS MIXED: unsupported on-disk integer width");
    return out;
}

std::vector<cgsize_t> read_mixed_section_local(int f_id, int B, int Z, int sec_idx,
    cgsize_t rs, cgsize_t re, bool has_data, MPI_Comm comm) {

    const int offset_width = mixed_array_disk_bytes(f_id, B, Z, sec_idx, "ElementStartOffset", comm);
    const int conn_width = mixed_array_disk_bytes(f_id, B, Z, sec_idx, "ElementConnectivity", comm);

    if (offset_width == 0 || conn_width == 0) {
        mpi::fatal(comm, "CGNS MIXED: section has no ElementStartOffset/ElementConnectivity array on disk");
    }

    if (!has_data) {
        check(cgp_poly_elements_read_data_offsets(f_id, B, Z, sec_idx, rs, re, nullptr),
              "cgp_poly_elements_read_data_offsets(MIXED, empty)", comm);
        check(cgp_poly_elements_read_data_elements(f_id, B, Z, sec_idx, rs, re, nullptr, nullptr),
              "cgp_poly_elements_read_data_elements(MIXED, empty)", comm);
        return {};
    }

    const std::size_t n_offsets = static_cast<std::size_t>(re - rs + 2);
    std::vector<cgsize_t> offsets = read_mixed_width_safe(
        n_offsets, offset_width, comm, [&](cgsize_t* buf) {
            check(cgp_poly_elements_read_data_offsets(f_id, B, Z, sec_idx, rs, re, buf),
                  "cgp_poly_elements_read_data_offsets(MIXED)", comm);
        });

    const cgsize_t local_size = offsets.back() - offsets.front();

    std::vector<cgsize_t> elements = read_mixed_width_safe(
        static_cast<std::size_t>(local_size), conn_width, comm, [&](cgsize_t* buf) {
            check(cgp_poly_elements_read_data_elements(f_id, B, Z, sec_idx, rs, re,
                  offsets.data(), buf),
                  "cgp_poly_elements_read_data_elements(MIXED)", comm);
        });

    return elements;
}

// Fallback for MIXED sections written without ElementStartOffset (CGNS < 4.0)
std::vector<cgsize_t> read_mixed_section_legacy_full(int f_id, int B, int Z, const mesh::SectionMeta& s,
    GlobalIndex local_elem_start, GlobalIndex local_elem_end, MPI_Comm comm) {

    const GlobalIndex sec_n = s.end - s.start + 1;
    cgsize_t datasize = 0;
    check(cg_ElementDataSize(f_id, B, Z, s.sec_idx, &datasize), "cg_ElementDataSize", comm);

    if (datasize == 0 || sec_n == 0) {
        return {};
    }

    std::vector<cgsize_t> full_buf(static_cast<std::size_t>(datasize));
    std::vector<cgsize_t> elem_offsets(static_cast<std::size_t>(sec_n) + 1, 0);
    check(cg_poly_elements_read(f_id, B, Z, s.sec_idx,
          full_buf.data(), elem_offsets.data(), nullptr),
          "cg_poly_elements_read(MIXED, legacy)", comm);

    if (local_elem_start >= local_elem_end) {
        return {};
    }

    const std::size_t byte_start = static_cast<std::size_t>(elem_offsets[static_cast<std::size_t>(local_elem_start)]);
    const std::size_t byte_end   = static_cast<std::size_t>(elem_offsets[static_cast<std::size_t>(local_elem_end)]);

    if (byte_end > full_buf.size() || byte_start > byte_end) {
        mpi::fatal(comm, "Malformed offsets in legacy mixed section");
    }

    return std::vector<cgsize_t>(
        full_buf.begin() + static_cast<std::ptrdiff_t>(byte_start),
        full_buf.begin() + static_cast<std::ptrdiff_t>(byte_end));
}

} // namespace

namespace {

std::vector<char> serialize_bcs(const std::vector<mesh::BCMeta>& bcs) {
    std::vector<char> buf;
    auto write_data = [&](const void* ptr, std::size_t size) {
        const char* byte_ptr = reinterpret_cast<const char*>(ptr);
        buf.insert(buf.end(), byte_ptr, byte_ptr + size);
    };

    uint32_t count = static_cast<uint32_t>(bcs.size());
    write_data(&count, sizeof(count));

    for (const auto& bc : bcs) {
        uint32_t name_len = static_cast<uint32_t>(bc.name.size());
        write_data(&name_len, sizeof(name_len));
        write_data(bc.name.data(), name_len);

        uint32_t type_len = static_cast<uint32_t>(bc.cgns_type.size());
        write_data(&type_len, sizeof(type_len));
        write_data(bc.cgns_type.data(), type_len);

        uint64_t n_eids = static_cast<uint64_t>(bc.eids.size());
        write_data(&n_eids, sizeof(n_eids));
        if (n_eids > 0) {
            write_data(bc.eids.data(), n_eids * sizeof(GlobalIndex));
        }
    }
    return buf;
}

std::vector<mesh::BCMeta> deserialize_bcs(const char* buf, std::size_t size) {
    std::vector<mesh::BCMeta> bcs;
    std::size_t ptr = 0;
    static_cast<void>(size);

    auto read_data = [&](void* dst, std::size_t len) {
        std::memcpy(dst, buf + ptr, len);
        ptr += len;
    };

    uint32_t count = 0;
    read_data(&count, sizeof(count));
    bcs.resize(count);

    for (uint32_t i = 0; i < count; ++i) {
        uint32_t name_len = 0;
        read_data(&name_len, sizeof(name_len));
        bcs[i].name.resize(name_len);
        if (name_len > 0) read_data(bcs[i].name.data(), name_len);

        uint32_t type_len = 0;
        read_data(&type_len, sizeof(type_len));
        bcs[i].cgns_type.resize(type_len);
        if (type_len > 0) read_data(bcs[i].cgns_type.data(), type_len);

        uint64_t n_eids = 0;
        read_data(&n_eids, sizeof(n_eids));
        bcs[i].eids.resize(static_cast<std::size_t>(n_eids));
        if (n_eids > 0) {
            read_data(bcs[i].eids.data(), static_cast<std::size_t>(n_eids) * sizeof(GlobalIndex));
        }
    }
    return bcs;
}

std::vector<mesh::BCMeta> read_bcs_serial(const std::string& path, MPI_Comm comm) {
    int s_id = 0;
    if (cg_open(path.c_str(), CG_MODE_READ, &s_id) != CG_OK) {
        mpi::fatal(comm, "Serial cg_open failed on rank 0: " + std::string(cg_get_error()));
    }

    // Base&zone counts
    int nbases = 0, nzones = 0;
    check(cg_nbases(s_id, &nbases), "cg_nbases", comm);
    if (nbases != 1) { mpi::fatal(comm, "Expected exactly 1 base in CGNS file"); }
    const int B = 1;

    check(cg_nzones(s_id, B, &nzones), "cg_nzones", comm);
    if (nzones != 1) { mpi::fatal(comm, "Expected exactly 1 zone in CGNS file"); }
    const int Z = 1;

    ZoneType zonetype;
    check(cg_zone_type(s_id, B, Z, &zonetype), "cg_zone_type", comm);
    if (zonetype != CGNS_ENUMV(Unstructured)) { mpi::fatal(comm, "Only Unstructured zones are supported"); }

    int nbocos = 0;
    if (cg_nbocos(s_id, B, Z, &nbocos) != CG_OK) {
        cg_close(s_id);
        mpi::fatal(comm, "Serial cg_nbocos failed: " + std::string(cg_get_error()));
    }

    std::vector<mesh::BCMeta> bcs;
    for (int bc = 1; bc <= nbocos; ++bc) {
        char bcname[33] = "";
        BoundaryConditionType btype;
        PointSetType ptype;
        cgsize_t npnts = 0, normallistsize = 0;
        int normalidx[3] = {0, 0, 0};
        DataType ndtype;
        int ndataset = 0;

        check(cg_boco_info(s_id, B, Z, bc, bcname, &btype, &ptype, &npnts, normalidx, 
                           &normallistsize, &ndtype, &ndataset), "cg_boco_info", comm);

        GridLocation loc = CGNS_ENUMV(FaceCenter);
        if (cg_boco_gridlocation_read(s_id, B, Z, bc, &loc) == CG_OK) {
            if (loc != CGNS_ENUMV(FaceCenter)) {
                mpi::log_info("BC '%s': GridLocation != FaceCenter, skipped", bcname);
                continue;
            }
        }

        std::vector<cgsize_t> pnts(static_cast<std::size_t>(npnts));
        if (npnts > 0) {
            check(cg_boco_read(s_id, B, Z, bc, pnts.data(), nullptr), "cg_boco_read", comm);
        }

        mesh::BCMeta bm;
        const char* type_str = cg_BCTypeName(btype);
        bm.cgns_type = (type_str != nullptr) ? type_str : "UserDefined";

        if (ptype == CGNS_ENUMV(PointRange) && npnts == 2) {
            const cgsize_t p_min = std::min(pnts[0], pnts[1]);
            const cgsize_t p_max = std::max(pnts[0], pnts[1]);
            bm.eids.reserve(static_cast<std::size_t>(p_max - p_min + 1));
            for (cgsize_t e = p_min; e <= p_max; ++e) {
                bm.eids.push_back(static_cast<GlobalIndex>(e));
            }
        } else {
            bm.eids.reserve(static_cast<std::size_t>(npnts));
            for (cgsize_t i = 0; i < npnts; ++i) {
                bm.eids.push_back(static_cast<GlobalIndex>(pnts[static_cast<std::size_t>(i)]));
            }
        }

        char fam[33] = "";
        if (cg_goto(s_id, B, "Zone_t", Z, "ZoneBC_t", 1, "BC_t", bc, "end") == CG_OK) {
            if (cg_famname_read(fam) == CG_OK && fam[0] != '\0') {
                bm.name = fam;
            }
        }
        if (bm.name.empty()) {
            bm.name = bcname;
        }

        bcs.push_back(std::move(bm));
    }

    cg_close(s_id);
    return bcs;
}

} // namespace



mesh::RawMesh read_cgns_parallel(const std::string& path, MPI_Comm comm) {
    // get rank index and total process count
    int nprocs = 0, rank = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);


    // Initialize result
    mesh::RawMesh m;
    m.nprocs = nprocs;
    m.rank = rank;
    m.comm = comm;

    
    // Read Boundary Conditions (ZoneBC)
    // master rank boundary section read + broadcast
    std::vector<char> bc_buffer;
    uint64_t buf_size = 0;

    if (rank == 0) {
        m.bcs = read_bcs_serial(path, comm);
        bc_buffer = serialize_bcs(m.bcs);
        buf_size = bc_buffer.size();
    }

    MPI_Bcast(&buf_size, 1, MPI_UINT64_T, 0, comm);
    if (rank != 0) {
        bc_buffer.resize(buf_size);
    }
    MPI_Bcast(bc_buffer.data(), static_cast<int>(buf_size), MPI_CHAR, 0, comm);

    if (rank != 0) {
        m.bcs = deserialize_bcs(bc_buffer.data(), buf_size);
    }


    // Set parallel collective I/O mode across all ranks
    check(cgp_pio_mode(CGP_COLLECTIVE), "cgp_pio_mode(CGP_COLLECTIVE)", comm);

    // Open CGNS file in parallel via RAII wrapper
    File file(path, comm);
    const int f_id = file.id();


    // Read file and general metadata
    check(cg_version(f_id, &m.gfm.cgns_version), "cg_version", comm);
    check(cg_precision(f_id, &m.gfm.file_integer_precision), "cg_precision", comm);
    check(cg_get_file_type(f_id, &m.gfm.storage_type), "cg_get_file_type", comm);

    const char* storage_str = "Unknown";
    if (m.gfm.storage_type == 1) storage_str = "ADF";
    else if (m.gfm.storage_type == 2) storage_str = "HDF5";
    else if (m.gfm.storage_type == 0) storage_str = "None/Error";
    mpi::log_stat("CGNS: Version='%.2f', Integer precision='%d bits', Storage type='%s'", 
                  m.gfm.cgns_version, m.gfm.file_integer_precision, storage_str);

    // Files written by CGNS libraries older than ~4.0 store MIXED-section
    // connectivity WITHOUT a separate ElementStartOffset side-array on disk...
    //
    // ADDITIONALLY: cgp_poly_elements_read_data_offsets/_elements (the fast
    // path) pick their HDF5 memory datatype from the FILE's stored integer
    // precision of the array being read, unlike cgp_elements_read_data
    // which picks it from sizeof(cgsize_t) and lets HDF5 convert. Per-array
    // width mismatches (e.g. I8 ElementStartOffset next to I4
    // ElementConnectivity, as in mesh/sphere_v4.cgns) are now handled inside
    // read_mixed_section_local via width-matched scratch buffers; this
    // file-WIDE precision check stays as an extra conservative guard that
    // routes grossly mismatched files to the legacy reader below.
    const bool mixed_needs_legacy_read =
        (m.gfm.cgns_version < 3.99f) ||
        (m.gfm.file_integer_precision != static_cast<int>(sizeof(cgsize_t) * 8));

    if (mixed_needs_legacy_read) {
        if (rank == 0) {
            mpi::log_info("CGNS WARNING: CGNS MIXED sections will be read (if exist) via ram-heavy fall back. See desciption in cgns_read.cpp line 261.");
        }
    }

    
    // Base metadata
    int nbases = 0;
    check(cg_nbases(f_id, &nbases), "cg_nbases", comm);
    if (nbases != 1) { mpi::fatal(comm, "Expected exactly 1 base in CGNS file"); }

    const int B = 1;
    int celldim = 0, physdim = 0;
    char basename[33] = "";
    check(cg_base_read(f_id, B, basename, &celldim, &physdim), "cg_base_read", comm);
    if (celldim != 3 || physdim != 3) { mpi::fatal(comm, "Base must be 3D (CellDim=3, PhysDim=3)"); }


    // Zone metadata 
    int nzones = 0;
    check(cg_nzones(f_id, B, &nzones), "cg_nzones", comm);
    if (nzones != 1) { mpi::fatal(comm, "Expected exactly 1 zone in CGNS file"); }

    const int Z = 1;
    ZoneType zonetype;
    check(cg_zone_type(f_id, B, Z, &zonetype), "cg_zone_type", comm);
    if (zonetype != CGNS_ENUMV(Unstructured)) { mpi::fatal(comm, "Only Unstructured zones are supported"); }

    char zonename[33] = "";
    cgsize_t sizes[9] = {0};
    check(cg_zone_read(f_id, B, Z, zonename, sizes), "cg_zone_read", comm);
    m.n_nodes_g = static_cast<GlobalIndex>(sizes[0]);
    m.n_cells_g = static_cast<GlobalIndex>(sizes[1]);

    mpi::log_stat("CGNS: Base='%s', Zone='%s', Total Nodes=%lld, Total Cells=%lld", 
                  basename, zonename, static_cast<long long>(m.n_nodes_g), static_cast<long long>(m.n_cells_g));


    // Read Element Section Metadata
    int nsecs = 0;
    check(cg_nsections(f_id, B, Z, &nsecs), "cg_nsections", comm);

    // loop over all sections
    for (int S = 1; S <= nsecs; ++S) {
        char secname[33] = "";
        ElementType etype;
        cgsize_t start = 0, end = 0;
        int nbndry = 0, parent_flag = 0;

        check(cg_section_read(f_id, B, Z, S, secname,
              &etype, &start, &end, &nbndry, &parent_flag), 
              "cg_section_read", comm);

        // Skip 1D boundary elements (edges/lines) as documented
        if (etype == CGNS_ENUMV(BAR_2) || etype == CGNS_ENUMV(BAR_3)) {
            if (rank == 0) {
                mpi::log_info("CGNS: Skipping 1D section '%s'", secname);
            }
            continue;
        }
        
        mesh::SectionMeta sm;
        sm.name = secname;
        sm.start = static_cast<GlobalIndex>(start);
        sm.end = static_cast<GlobalIndex>(end);
        sm.sec_idx = S;
        sm.is_mixed = (etype == CGNS_ENUMV(MIXED));

        if (!sm.is_mixed) {
            sm.type = cgns_elem_to_type(etype, comm);
        }

        // Volume section range is [1, n_cells_g]
        if (sm.start <= m.n_cells_g) {
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


    // Build global patch list
    for (const auto& b : m.bcs) {
        m.patch_list.push_back({b.name, b.cgns_type});
        mpi::log_stat("CGNS: BC '%s' Type=%s Faces=%zu -> PatchId=%d", 
                      b.name.c_str(), b.cgns_type.c_str(), b.eids.size(), 
                      static_cast<int>(m.patch_list.size() - 1));
    }

    
    // Read Surface Sections (Collective)
    {
        // loop over all surface sections
        for (const auto& s : m.surf_secs) {
            const GlobalIndex sec_n = s.end - s.start + 1;

            // replicated on each rank surface section contiguous distribution
            const std::vector<GlobalIndex> d = mpi::block_displ(sec_n, static_cast<std::size_t>(nprocs));

            // get local rank range (0 based, exclusive)
            const GlobalIndex lo = d[static_cast<std::size_t>(rank)];
            const GlobalIndex hi = d[static_cast<std::size_t>(rank) + 1];

            const std::size_t local_count = (lo < hi) ? static_cast<std::size_t>(hi - lo) : 0;
            const GlobalIndex sec_start = s.start + lo;
            const GlobalIndex sec_end = s.start + hi;

            // Local patch ID map: O(local_count) memory
            std::vector<PatchId> local_patch(local_count, kInvalidPatchId);
            if (local_count > 0) {
                for (std::size_t p = 0; p < m.bcs.size(); ++p) {
                    for (GlobalIndex e : m.bcs[p].eids) {
                        if (e >= sec_start && e < sec_end) {
                            local_patch[static_cast<std::size_t>(e - sec_start)] = static_cast<PatchId>(p);
                        }
                    }
                }
            }

            if (s.is_mixed) {
                std::vector<cgsize_t> buf;
                if (mixed_needs_legacy_read) {
                    buf = read_mixed_section_legacy_full(f_id, B, Z, s, lo, hi, comm);
                } else {
                    const cgsize_t rs = (lo < hi) ? static_cast<cgsize_t>(s.start + lo) : 1;
                    const cgsize_t re = (lo < hi) ? static_cast<cgsize_t>(s.start + hi - 1) : 0;
                    buf = read_mixed_section_local(f_id, B, Z, s.sec_idx, rs, re, lo < hi, comm);
                }

                std::size_t ptr = 0;
                for (std::size_t i = 0; i < local_count; ++i) {
                    mesh::SurfElem se{};
                    auto raw_type = static_cast<ElementType>(buf[ptr++]);
                    auto [stype, npt] = parse_cgns_element_header(raw_type, comm);

                    for (int k = 0; k < npt; ++k) {
                        se.key.v[static_cast<std::size_t>(k)] = static_cast<GlobalIndex>(buf[ptr++] - 1);
                    }
                    sort_face_key_nodes(se.key, npt);

                    se.eid = sec_start + static_cast<GlobalIndex>(i);
                    se.patch = local_patch[i];

                    m.surf_elems.push_back(se);
                }
            } else {
                // get number of nodes per section element type
                const std::size_t npt = static_cast<std::size_t>(mesh::kNodesPerType[static_cast<std::size_t>(s.type)]);
                std::vector<cgsize_t> buf(local_count * npt);

                // In collective mode, all ranks must participate; 0-sized reads pass rs=1, re=0
                const cgsize_t rs = (lo < hi) ? static_cast<cgsize_t>(s.start + lo) : 1;
                const cgsize_t re = (lo < hi) ? static_cast<cgsize_t>(s.start + hi - 1) : 0;

                cgsize_t* pbuf = (local_count > 0) ? buf.data() : nullptr;

                check(cgp_elements_read_data(f_id, B, Z, s.sec_idx, rs, re, pbuf),
                      "cgp_elements_read_data(surface)", comm);

                // skip blank ranges
                if (lo >= hi) continue;

                // loop over local surface elements
                for (std::size_t i = 0; i < local_count; ++i) {
                    mesh::SurfElem se{};
                    for (std::size_t k = 0; k < npt; ++k) {
                        se.key.v[k] = static_cast<GlobalIndex>(buf[i * npt + k] - 1); // back to 0-based
                    }
                    sort_face_key_nodes(se.key, static_cast<int>(npt));

                    se.eid = sec_start + static_cast<GlobalIndex>(i);
                    se.patch = local_patch[i];

                    m.surf_elems.push_back(se);
                } // end loop over local surface elements
            }
        } // end loop over all surface sections
    }


    // Globla Block Distributions for Volume Cells and Nodes (replicated on each rank)
    m.cell_displ = mpi::block_displ(m.n_cells_g, static_cast<std::size_t>(nprocs));
    m.node_displ = mpi::block_displ(m.n_nodes_g, static_cast<std::size_t>(nprocs));

    
    // Read Volume Cell Connectivity (Collective)
    const LocalIndex nl = m.n_local_cells();
    m.ctype.reserve(static_cast<std::size_t>(nl));

    m.cnodes_offsets.reserve(static_cast<std::size_t>(nl) + 1);
    m.cnodes_offsets.push_back(0);

    // loop over volume sections to estimate connectivity memory
    for (const auto& s : m.vol_secs) {
        const GlobalIndex sec_n = s.end - s.start + 1;
        const GlobalIndex cb = m.cell_displ[static_cast<std::size_t>(rank)];
        const GlobalIndex ce = m.cell_displ[static_cast<std::size_t>(rank) + 1];
        const GlobalIndex lo = std::max(s.cell_offset, cb);
        const GlobalIndex hi = std::min(s.cell_offset + sec_n, ce);
        const std::size_t local_count = (lo < hi) ? static_cast<std::size_t>(hi - lo) : 0;

        if (s.is_mixed) {
            std::vector<cgsize_t> buf;
            if (mixed_needs_legacy_read) {
                buf = read_mixed_section_legacy_full(f_id, B, Z, s, lo - s.cell_offset, hi - s.cell_offset, comm);
            } else {
                const cgsize_t rs = (lo < hi) ? static_cast<cgsize_t>(s.start + (lo - s.cell_offset)) : 1;
                const cgsize_t re = (lo < hi) ? static_cast<cgsize_t>(s.start + (hi - 1 - s.cell_offset)) : 0;
                buf = read_mixed_section_local(f_id, B, Z, s.sec_idx, rs, re, lo < hi, comm);
            }

            std::size_t ptr = 0;
            for (std::size_t i = 0; i < local_count; ++i) {
                auto raw_type = static_cast<ElementType>(buf[ptr++]);
                auto [cell_t, npts] = parse_cgns_element_header(raw_type, comm);

                m.ctype.push_back(cell_t);
                for (int k = 0; k < npts; ++k) {
                    m.cnodes.push_back(static_cast<GlobalIndex>(buf[ptr++] - 1));
                }
                m.cnodes_offsets.push_back(static_cast<LocalIndex>(m.cnodes.size()));
            }
        } else {
            const std::size_t npt = static_cast<std::size_t>(mesh::kNodesPerType[static_cast<std::size_t>(s.type)]);
            std::vector<cgsize_t> buf(local_count * npt);

            const cgsize_t rs = (lo < hi) ? static_cast<cgsize_t>(s.start + (lo - s.cell_offset)) : 1;
            const cgsize_t re = (lo < hi) ? static_cast<cgsize_t>(s.start + (hi - 1 - s.cell_offset)) : 0;

            cgsize_t* pbuf = (local_count > 0) ? buf.data() : nullptr;

            check(cgp_elements_read_data(f_id, B, Z, s.sec_idx, rs, re, pbuf),
                  "cgp_elements_read_data(volume)", comm);

            if (lo >= hi) continue;

            for (std::size_t i = 0; i < local_count; ++i) {
                m.ctype.push_back(s.type);
                for (std::size_t k = 0; k < npt; ++k) {
                    m.cnodes.push_back(static_cast<GlobalIndex>(buf[i * npt + k] - 1));
                }
                m.cnodes_offsets.push_back(static_cast<LocalIndex>(m.cnodes.size()));
            }
        }
    } // end loop over all volume sections

    if (m.cnodes_offsets.size() != static_cast<std::size_t>(nl) + 1) {
        mpi::fatal(comm, "Consistency error in raw mesh cell nodes CSR table");
    }

    if (m.ctype.size() != static_cast<std::size_t>(nl)) {
        mpi::fatal(comm, "Mismatch in read volume cells: got " + std::to_string(m.ctype.size()) +
                    ", expected " + std::to_string(nl));
    }


    const GlobalIndex nb = m.my_node_begin();
    const GlobalIndex ne = m.my_node_end();
    const std::size_t nmy = (nb < ne) ? static_cast<std::size_t>(ne - nb) : 0;

    m.my_node_coords_x.resize(nmy);
    m.my_node_coords_y.resize(nmy);
    m.my_node_coords_z.resize(nmy);

    cgsize_t rs = (nmy > 0) ? static_cast<cgsize_t>(nb + 1) : 1;
    cgsize_t re = (nmy > 0) ? static_cast<cgsize_t>(ne) : 0;

    DataType coord_dtype = CGNS_ENUMV(RealDouble);
    char coord_name[33] = "";
    check(cg_coord_info(f_id, B, Z, 1, &coord_dtype, coord_name), "cg_coord_info", comm);
    
    if (coord_dtype == CGNS_ENUMV(RealSingle)) {
        std::vector<float> tmp_x(nmy), tmp_y(nmy), tmp_z(nmy);
        float* px = (nmy > 0) ? tmp_x.data() : nullptr;
        float* py = (nmy > 0) ? tmp_y.data() : nullptr;
        float* pz = (nmy > 0) ? tmp_z.data() : nullptr;

        check(cgp_coord_read_data(f_id, B, Z, 1, &rs, &re, px), "cgp_coord_read_data(X)", comm);
        check(cgp_coord_read_data(f_id, B, Z, 2, &rs, &re, py), "cgp_coord_read_data(Y)", comm);
        check(cgp_coord_read_data(f_id, B, Z, 3, &rs, &re, pz), "cgp_coord_read_data(Z)", comm);

        for (std::size_t i = 0; i < nmy; ++i) {
            m.my_node_coords_x[i] = static_cast<double>(tmp_x[i]);
            m.my_node_coords_y[i] = static_cast<double>(tmp_y[i]);
            m.my_node_coords_z[i] = static_cast<double>(tmp_z[i]);
        }
    } else {
        double* px = (nmy > 0) ? m.my_node_coords_x.data() : nullptr;
        double* py = (nmy > 0) ? m.my_node_coords_y.data() : nullptr;
        double* pz = (nmy > 0) ? m.my_node_coords_z.data() : nullptr;

        check(cgp_coord_read_data(f_id, B, Z, 1, &rs, &re, px), "cgp_coord_read_data(X)", comm);
        check(cgp_coord_read_data(f_id, B, Z, 2, &rs, &re, py), "cgp_coord_read_data(Y)", comm);
        check(cgp_coord_read_data(f_id, B, Z, 3, &rs, &re, pz), "cgp_coord_read_data(Z)", comm);
    }

    file.close();

    // mpi::log_rank("CGNS Rank %d: Local Cells=%d, Local Nodes=%zu", rank, nl, nmy);

    return m;
}

} // namespace cfd::io::cgns