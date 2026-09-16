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
            if (loc != CGNS_ENUMV(FaceCenter) && loc != CGNS_ENUMV(EdgeCenter)) {
                mpi::log_info("BC '%s': GridLocation != FaceCenter/EdgeCenter, skipped", bcname);
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

// -----------------------------------------------------------------------------
// 2D Extrusion Engine (CellDim=2 -> 1-layer 3D Mesh)
// -----------------------------------------------------------------------------
mesh::RawMesh read_cgns_2d_and_extrude_parallel(const std::string& path, MPI_Comm comm,
                                               double dz = 1.0,
                                               std::vector<mesh::BCMeta> pre_read_bcs = {}) {
    int nprocs = 0, rank = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    mesh::RawMesh m;
    m.nprocs = nprocs;
    m.rank = rank;
    m.comm = comm;

    // Synchronize boundary conditions across ranks
    if (pre_read_bcs.empty()) {
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
    } else {
        m.bcs = std::move(pre_read_bcs);
    }

    check(cgp_pio_mode(CGP_COLLECTIVE), "cgp_pio_mode(CGP_COLLECTIVE)", comm);

    File file(path, comm);
    const int f_id = file.id();

    check(cg_version(f_id, &m.gfm.cgns_version), "cg_version", comm);
    check(cg_precision(f_id, &m.gfm.file_integer_precision), "cg_precision", comm);
    check(cg_get_file_type(f_id, &m.gfm.storage_type), "cg_get_file_type", comm);

    const bool mixed_needs_legacy_read =
        (m.gfm.cgns_version < 3.99f) ||
        (m.gfm.file_integer_precision != static_cast<int>(sizeof(cgsize_t) * 8));

    if (mixed_needs_legacy_read && rank == 0) {
        mpi::log_info("CGNS WARNING: 2D MIXED sections will be read via legacy path");
    }

    const int B = 1;
    const int Z = 1;

    char basename[33] = "";
    int celldim = 0, physdim = 0;
    check(cg_base_read(f_id, B, basename, &celldim, &physdim), "cg_base_read(2D)", comm);
    if (celldim != 2) {
        mpi::fatal(comm, "read_cgns_2d_and_extrude_parallel expects CellDim=2");
    }

    char zonename[33] = "";
    cgsize_t sizes[9] = {0};
    check(cg_zone_read(f_id, B, Z, zonename, sizes), "cg_zone_read(2D)", comm);

    const GlobalIndex n_nodes_2d = static_cast<GlobalIndex>(sizes[0]);
    m.n_cells_g = static_cast<GlobalIndex>(sizes[1]);
    m.n_nodes_g = 2 * n_nodes_2d;

    // -------------------------------------------------------------------------
    // 1. Coordinate Discovery & Reading
    // -------------------------------------------------------------------------
    int ncoords = 0;
    check(cg_ncoords(f_id, B, Z, &ncoords), "cg_ncoords(2D)", comm);

    int idx_x = -1, idx_y = -1, idx_z = -1;
    DataType dt_x = CGNS_ENUMV(RealDouble), dt_y = CGNS_ENUMV(RealDouble), dt_z = CGNS_ENUMV(RealDouble);

    for (int c = 1; c <= ncoords; ++c) {
        char cname[33] = "";
        DataType dt;
        check(cg_coord_info(f_id, B, Z, c, &dt, cname), "cg_coord_info", comm);
        if (std::strcmp(cname, "CoordinateX") == 0) { idx_x = c; dt_x = dt; }
        else if (std::strcmp(cname, "CoordinateY") == 0) { idx_y = c; dt_y = dt; }
        else if (std::strcmp(cname, "CoordinateZ") == 0) { idx_z = c; dt_z = dt; }
    }

    if (idx_x == -1 && ncoords >= 1) { idx_x = 1; }
    if (idx_y == -1 && ncoords >= 2) { idx_y = 2; }
    if (idx_z == -1 && ncoords >= 3) { idx_z = 3; }

    const auto d2d = mpi::block_displ(n_nodes_2d, static_cast<std::size_t>(nprocs));
    const GlobalIndex lo2d = d2d[static_cast<std::size_t>(rank)];
    const GlobalIndex hi2d = d2d[static_cast<std::size_t>(rank) + 1];
    const std::size_t n_loc2d = (lo2d < hi2d) ? static_cast<std::size_t>(hi2d - lo2d) : 0;

    const cgsize_t rs2d = (lo2d < hi2d) ? static_cast<cgsize_t>(lo2d + 1) : 1;
    const cgsize_t re2d = (lo2d < hi2d) ? static_cast<cgsize_t>(hi2d) : 0;

    auto read_coord_component = [&](int c_idx, DataType dt) -> std::vector<double> {
        std::vector<double> res(n_loc2d, 0.0);
        if (c_idx < 1) return res;

        if (dt == CGNS_ENUMV(RealSingle)) {
            std::vector<float> tmp(n_loc2d);
            float* ptr = (n_loc2d > 0) ? tmp.data() : nullptr;
            check(cgp_coord_read_data(f_id, B, Z, c_idx, &rs2d, &re2d, ptr), "cgp_coord_read_data(float)", comm);
            for (std::size_t i = 0; i < n_loc2d; ++i) res[i] = static_cast<double>(tmp[i]);
        } else {
            double* ptr = (n_loc2d > 0) ? res.data() : nullptr;
            check(cgp_coord_read_data(f_id, B, Z, c_idx, &rs2d, &re2d, ptr), "cgp_coord_read_data(double)", comm);
        }
        return res;
    };

    std::vector<double> raw_c1 = read_coord_component(idx_x, dt_x);
    std::vector<double> raw_c2 = read_coord_component(idx_y, dt_y);
    std::vector<double> raw_c3 = read_coord_component(idx_z, dt_z);

    auto get_axis_span = [&](const std::vector<double>& vals) -> std::pair<double, double> {
        double loc_min = std::numeric_limits<double>::infinity();
        double loc_max = -std::numeric_limits<double>::infinity();
        for (double v : vals) {
            loc_min = std::min(loc_min, v);
            loc_max = std::max(loc_max, v);
        }
        double g_min = 0.0, g_max = 0.0;
        MPI_Allreduce(&loc_min, &g_min, 1, MPI_DOUBLE, MPI_MIN, comm);
        MPI_Allreduce(&loc_max, &g_max, 1, MPI_DOUBLE, MPI_MAX, comm);
        return {g_min, g_max};
    };

    auto [min_x, max_x] = get_axis_span(raw_c1);
    auto [min_y, max_y] = get_axis_span(raw_c2);
    auto [min_z, max_z] = (idx_z > 0) ? get_axis_span(raw_c3) : std::make_pair(0.0, 0.0);

    const double span_x = max_x - min_x;
    const double span_y = max_y - min_y;
    const double span_z = max_z - min_z;

    std::vector<double> loc_planar_x(n_loc2d), loc_planar_y(n_loc2d);
    double span_p1 = 0.0, span_p2 = 0.0;

    if (idx_z > 0 && span_x <= 1e-14 * std::max(span_y, span_z)) {
        loc_planar_x = std::move(raw_c2);
        loc_planar_y = std::move(raw_c3);
        span_p1 = span_y; span_p2 = span_z;
    } else if (idx_z > 0 && span_y <= 1e-14 * std::max(span_x, span_z)) {
        loc_planar_x = std::move(raw_c1);
        loc_planar_y = std::move(raw_c3);
        span_p1 = span_x; span_p2 = span_z;
    } else {
        loc_planar_x = std::move(raw_c1);
        loc_planar_y = std::move(raw_c2);
        span_p1 = span_x; span_p2 = span_y;
    }

    const double char_length = std::max(span_p1, span_p2);
    const double actual_dz = (dz > 0.0) ? dz : (0.10 * (char_length > 0.0 ? char_length : 1.0));

    if (rank == 0) {
        mpi::log_stat("2D Bounding Box: PlanarX span=%.4e, PlanarY span=%.4e", span_p1, span_p2);
        mpi::log_stat("Extrusion thickness dz: %.4e (char_length=%.4e)", actual_dz, char_length);
    }

    std::vector<int> counts(static_cast<std::size_t>(nprocs));
    std::vector<int> displs(static_cast<std::size_t>(nprocs));
    for (std::size_t p = 0; p < static_cast<std::size_t>(nprocs); ++p) {
        counts[p] = static_cast<int>(d2d[p + 1] - d2d[p]);
        displs[p] = static_cast<int>(d2d[p]);
    }

    std::vector<double> all_x(static_cast<std::size_t>(n_nodes_2d));
    std::vector<double> all_y(static_cast<std::size_t>(n_nodes_2d));

    MPI_Allgatherv(loc_planar_x.data(), static_cast<int>(n_loc2d), MPI_DOUBLE,
                   all_x.data(), counts.data(), displs.data(), MPI_DOUBLE, comm);
    MPI_Allgatherv(loc_planar_y.data(), static_cast<int>(n_loc2d), MPI_DOUBLE,
                   all_y.data(), counts.data(), displs.data(), MPI_DOUBLE, comm);

    m.node_displ = mpi::block_displ(m.n_nodes_g, static_cast<std::size_t>(nprocs));
    const GlobalIndex nb = m.my_node_begin();
    const GlobalIndex ne = m.my_node_end();
    const std::size_t nmy = (nb < ne) ? static_cast<std::size_t>(ne - nb) : 0;

    m.my_node_coords_x.resize(nmy);
    m.my_node_coords_y.resize(nmy);
    m.my_node_coords_z.resize(nmy);

    for (std::size_t i = 0; i < nmy; ++i) {
        const GlobalIndex gid = nb + static_cast<GlobalIndex>(i);
        if (gid < n_nodes_2d) {
            m.my_node_coords_x[i] = all_x[static_cast<std::size_t>(gid)];
            m.my_node_coords_y[i] = all_y[static_cast<std::size_t>(gid)];
            m.my_node_coords_z[i] = 0.0;
        } else {
            const GlobalIndex orig = gid - n_nodes_2d;
            m.my_node_coords_x[i] = all_x[static_cast<std::size_t>(orig)];
            m.my_node_coords_y[i] = all_y[static_cast<std::size_t>(orig)];
            m.my_node_coords_z[i] = actual_dz;
        }
    }

    // -------------------------------------------------------------------------
    // 2. Sections Reading & Section-Based Patch Discovery
    // -------------------------------------------------------------------------
    int nsecs = 0;
    check(cg_nsections(f_id, B, Z, &nsecs), "cg_nsections(2D)", comm);

    GlobalIndex max_eid_in_file = 0;
    for (int S = 1; S <= nsecs; ++S) {
        char secname[33] = "";
        ElementType etype;
        cgsize_t start = 0, end = 0;
        int nbndry = 0, parent_flag = 0;

        check(cg_section_read(f_id, B, Z, S, secname,
                              &etype, &start, &end, &nbndry, &parent_flag),
              "cg_section_read(2D)", comm);

        max_eid_in_file = std::max(max_eid_in_file, static_cast<GlobalIndex>(end));

        mesh::SectionMeta sm;
        sm.name = secname;
        sm.start = static_cast<GlobalIndex>(start);
        sm.end = static_cast<GlobalIndex>(end);
        sm.sec_idx = S;
        sm.is_mixed = (etype == CGNS_ENUMV(MIXED));

        if (sm.start <= m.n_cells_g) {
            if (!sm.is_mixed) sm.type = cgns_elem_to_type(etype, comm);
            m.vol_secs.push_back(sm);
        } else {
            m.surf_secs.push_back(sm);
        }
    }

    std::sort(m.vol_secs.begin(), m.vol_secs.end(),
              [](const mesh::SectionMeta& a, const mesh::SectionMeta& b) noexcept {
                  return a.start < b.start;
              });

    GlobalIndex total_cells = 0;
    for (auto& s : m.vol_secs) {
        s.cell_offset = total_cells;
        total_cells += (s.end - s.start + 1);
    }
    if (m.n_cells_g == 0 || total_cells != m.n_cells_g) {
        mpi::fatal(comm, "Mismatch or zero volume cells in 2D CGNS zone");
    }

    // -------------------------------------------------------------------------
    // 3. Robust Patch Synthesis & Boundary Synchronization
    // -------------------------------------------------------------------------
    // Fallback: If no ZoneBC nodes exist, synthesize BCs from surface sections
    if (m.bcs.empty()) {
        for (const auto& s : m.surf_secs) {
            mesh::BCMeta bm;
            bm.name = s.name.empty() ? ("Boundary_" + std::to_string(s.sec_idx)) : s.name;
            bm.cgns_type = "UserDefined";
            bm.eids.reserve(static_cast<std::size_t>(s.end - s.start + 1));
            for (GlobalIndex e = s.start; e <= s.end; ++e) {
                bm.eids.push_back(e);
            }
            m.bcs.push_back(std::move(bm));
        }
    }

    // Map each surface section to a guaranteed valid PatchId
    std::vector<PatchId> sec_fallback_patch(m.surf_secs.size(), kInvalidPatchId);
    for (std::size_t i = 0; i < m.surf_secs.size(); ++i) {
        const auto& s = m.surf_secs[i];
        for (std::size_t p = 0; p < m.bcs.size(); ++p) {
            if (m.bcs[p].name == s.name) {
                sec_fallback_patch[i] = static_cast<PatchId>(p);
                break;
            }
        }
        if (sec_fallback_patch[i] == kInvalidPatchId) {
            sec_fallback_patch[i] = static_cast<PatchId>(m.bcs.size());
            mesh::BCMeta bm;
            bm.name = s.name.empty() ? ("Boundary_" + std::to_string(s.sec_idx)) : s.name;
            bm.cgns_type = "UserDefined";
            for (GlobalIndex e = s.start; e <= s.end; ++e) {
                bm.eids.push_back(e);
            }
            m.bcs.push_back(std::move(bm));
        }
    }

    // Synchronize m.patch_list with all physical BCs
    m.patch_list.clear();
    for (const auto& b : m.bcs) {
        m.patch_list.push_back({b.name, b.cgns_type});
    }

    // Append synthesis end-cap patches to BOTH patch_list and bcs
    const PatchId patch_z_min = static_cast<PatchId>(m.patch_list.size());
    m.patch_list.push_back({"Z_MIN", "SymmetryPlane"});
    m.bcs.push_back({"Z_MIN", "SymmetryPlane", {}});

    const PatchId patch_z_max = static_cast<PatchId>(m.patch_list.size());
    m.patch_list.push_back({"Z_MAX", "SymmetryPlane"});
    m.bcs.push_back({"Z_MAX", "SymmetryPlane", {}});

    // -------------------------------------------------------------------------
    // 4. Boundary Section Extrusion (BAR_2 -> QUAD_4)
    // -------------------------------------------------------------------------
    for (std::size_t sec_i = 0; sec_i < m.surf_secs.size(); ++sec_i) {
        const auto& s = m.surf_secs[sec_i];
        const GlobalIndex sec_n = s.end - s.start + 1;
        const auto d = mpi::block_displ(sec_n, static_cast<std::size_t>(nprocs));
        const GlobalIndex lo = d[static_cast<std::size_t>(rank)];
        const GlobalIndex hi = d[static_cast<std::size_t>(rank) + 1];

        const std::size_t local_count = (lo < hi) ? static_cast<std::size_t>(hi - lo) : 0;
        const GlobalIndex sec_start = s.start + lo;
        const GlobalIndex sec_end = s.start + hi;

        // Default to section fallback patch to strictly eliminate -1
        std::vector<PatchId> local_patch(local_count, sec_fallback_patch[sec_i]);
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
                const cgsize_t rs = (lo < hi) ? static_cast<cgsize_t>(sec_start) : 1;
                const cgsize_t re = (lo < hi) ? static_cast<cgsize_t>(sec_start + static_cast<GlobalIndex>(local_count) - 1) : 0;
                buf = read_mixed_section_local(f_id, B, Z, s.sec_idx, rs, re, lo < hi, comm);
            }

            std::size_t ptr = 0;
            for (std::size_t i = 0; i < local_count; ++i) {
                auto raw_type = static_cast<ElementType>(buf[ptr++]);
                if (raw_type != CGNS_ENUMV(BAR_2)) {
                    mpi::fatal(comm, "Unsupported 2D boundary element type in MIXED section");
                }
                const GlobalIndex a = static_cast<GlobalIndex>(buf[ptr++] - 1);
                const GlobalIndex b = static_cast<GlobalIndex>(buf[ptr++] - 1);

                mesh::SurfElem se{};
                se.key.v = {a, b, b + n_nodes_2d, a + n_nodes_2d};
                sort_face_key_nodes(se.key, 4);
                se.eid = sec_start + static_cast<GlobalIndex>(i);
                se.patch = local_patch[i];
                m.surf_elems.push_back(se);
            }
        } else {
            std::vector<cgsize_t> buf(local_count * 2);
            const cgsize_t rs = (lo < hi) ? static_cast<cgsize_t>(sec_start) : 1;
            const cgsize_t re = (lo < hi) ? static_cast<cgsize_t>(sec_start + static_cast<GlobalIndex>(local_count) - 1) : 0;

            cgsize_t* pbuf = (local_count > 0) ? buf.data() : nullptr;
            check(cgp_elements_read_data(f_id, B, Z, s.sec_idx, rs, re, pbuf),
                  "cgp_elements_read_data(BAR_2)", comm);

            if (lo >= hi) continue;

            for (std::size_t i = 0; i < local_count; ++i) {
                const GlobalIndex a = static_cast<GlobalIndex>(buf[i * 2 + 0] - 1);
                const GlobalIndex b = static_cast<GlobalIndex>(buf[i * 2 + 1] - 1);

                mesh::SurfElem se{};
                se.key.v = {a, b, b + n_nodes_2d, a + n_nodes_2d};
                sort_face_key_nodes(se.key, 4);
                se.eid = sec_start + static_cast<GlobalIndex>(i);
                se.patch = local_patch[i];
                m.surf_elems.push_back(se);
            }
        }
    }

    // -------------------------------------------------------------------------
    // 5. Volume Cell Extrusion & Cap Generation
    // -------------------------------------------------------------------------
    m.cell_displ = mpi::block_displ(m.n_cells_g, static_cast<std::size_t>(nprocs));
    const LocalIndex nl = m.n_local_cells();
    m.ctype.reserve(static_cast<std::size_t>(nl));
    m.cnodes_offsets.reserve(static_cast<std::size_t>(nl) + 1);
    m.cnodes_offsets.push_back(0);

    const GlobalIndex cb = m.cell_displ[static_cast<std::size_t>(rank)];
    const GlobalIndex ce = m.cell_displ[static_cast<std::size_t>(rank) + 1];

    auto add_extruded_cell = [&](mesh::CellType type_2d, GlobalIndex* nodes, GlobalIndex global_c_id) {
        const GlobalIndex bot_eid = max_eid_in_file + 1 + global_c_id;
        const GlobalIndex top_eid = max_eid_in_file + 1 + m.n_cells_g + global_c_id;

        if (type_2d == mesh::CellType::TRI) {
            const double ax = all_x[static_cast<std::size_t>(nodes[0])], ay = all_y[static_cast<std::size_t>(nodes[0])];
            const double bx = all_x[static_cast<std::size_t>(nodes[1])], by = all_y[static_cast<std::size_t>(nodes[1])];
            const double cx = all_x[static_cast<std::size_t>(nodes[2])], cy = all_y[static_cast<std::size_t>(nodes[2])];
            const double signed_area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);

            if (signed_area < 0.0) {
                std::swap(nodes[1], nodes[2]);
            }

            m.ctype.push_back(mesh::CellType::PRISM);
            m.cnodes.push_back(nodes[0]);
            m.cnodes.push_back(nodes[1]);
            m.cnodes.push_back(nodes[2]);
            m.cnodes.push_back(nodes[0] + n_nodes_2d);
            m.cnodes.push_back(nodes[1] + n_nodes_2d);
            m.cnodes.push_back(nodes[2] + n_nodes_2d);
            m.cnodes_offsets.push_back(static_cast<LocalIndex>(m.cnodes.size()));

            mesh::SurfElem bot{};
            bot.key.v = {nodes[0], nodes[1], nodes[2], kInvalidGlobalIndex};
            sort_face_key_nodes(bot.key, 3);
            bot.eid = bot_eid;
            bot.patch = patch_z_min;
            m.surf_elems.push_back(bot);

            mesh::SurfElem top{};
            top.key.v = {nodes[0] + n_nodes_2d, nodes[1] + n_nodes_2d, nodes[2] + n_nodes_2d, kInvalidGlobalIndex};
            sort_face_key_nodes(top.key, 3);
            top.eid = top_eid;
            top.patch = patch_z_max;
            m.surf_elems.push_back(top);

        } else if (type_2d == mesh::CellType::QUAD) {
            const double x0 = all_x[static_cast<std::size_t>(nodes[0])], y0 = all_y[static_cast<std::size_t>(nodes[0])];
            const double x1 = all_x[static_cast<std::size_t>(nodes[1])], y1 = all_y[static_cast<std::size_t>(nodes[1])];
            const double x2 = all_x[static_cast<std::size_t>(nodes[2])], y2 = all_y[static_cast<std::size_t>(nodes[2])];
            const double x3 = all_x[static_cast<std::size_t>(nodes[3])], y3 = all_y[static_cast<std::size_t>(nodes[3])];

            const double signed_area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
                                     + (x2 - x0) * (y3 - y0) - (y2 - y0) * (x3 - x0);

            if (signed_area < 0.0) {
                std::swap(nodes[1], nodes[3]);
            }

            m.ctype.push_back(mesh::CellType::HEXA);
            m.cnodes.push_back(nodes[0]);
            m.cnodes.push_back(nodes[1]);
            m.cnodes.push_back(nodes[2]);
            m.cnodes.push_back(nodes[3]);
            m.cnodes.push_back(nodes[0] + n_nodes_2d);
            m.cnodes.push_back(nodes[1] + n_nodes_2d);
            m.cnodes.push_back(nodes[2] + n_nodes_2d);
            m.cnodes.push_back(nodes[3] + n_nodes_2d);
            m.cnodes_offsets.push_back(static_cast<LocalIndex>(m.cnodes.size()));

            mesh::SurfElem bot{};
            bot.key.v = {nodes[0], nodes[1], nodes[2], nodes[3]};
            sort_face_key_nodes(bot.key, 4);
            bot.eid = bot_eid;
            bot.patch = patch_z_min;
            m.surf_elems.push_back(bot);

            mesh::SurfElem top{};
            top.key.v = {nodes[0] + n_nodes_2d, nodes[1] + n_nodes_2d, nodes[2] + n_nodes_2d, nodes[3] + n_nodes_2d};
            sort_face_key_nodes(top.key, 4);
            top.eid = top_eid;
            top.patch = patch_z_max;
            m.surf_elems.push_back(top);
        } else {
            mpi::fatal(comm, "Unsupported 2D cell type during extrusion");
        }
    };

    for (const auto& s : m.vol_secs) {
        const GlobalIndex sec_n = s.end - s.start + 1;
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
                auto [cell_2d_type, npts] = parse_cgns_element_header(raw_type, comm);

                GlobalIndex nodes[4] = {0, 0, 0, 0};
                for (int k = 0; k < npts; ++k) {
                    nodes[k] = static_cast<GlobalIndex>(buf[ptr++] - 1);
                }

                const GlobalIndex global_c_id = lo + static_cast<GlobalIndex>(i);
                add_extruded_cell(cell_2d_type, nodes, global_c_id);
            }
        } else {
            const std::size_t npt = (s.type == mesh::CellType::TRI) ? 3 : 4;
            std::vector<cgsize_t> buf(local_count * npt);

            const cgsize_t rs = (lo < hi) ? static_cast<cgsize_t>(s.start + (lo - s.cell_offset)) : 1;
            const cgsize_t re = (lo < hi) ? static_cast<cgsize_t>(s.start + (hi - 1 - s.cell_offset)) : 0;

            cgsize_t* pbuf = (local_count > 0) ? buf.data() : nullptr;
            check(cgp_elements_read_data(f_id, B, Z, s.sec_idx, rs, re, pbuf),
                  "cgp_elements_read_data(volume 2D)", comm);

            if (lo >= hi) continue;

            for (std::size_t i = 0; i < local_count; ++i) {
                GlobalIndex nodes[4] = {0, 0, 0, 0};
                for (std::size_t k = 0; k < npt; ++k) {
                    nodes[k] = static_cast<GlobalIndex>(buf[i * npt + k] - 1);
                }

                const GlobalIndex global_c_id = lo + static_cast<GlobalIndex>(i);
                add_extruded_cell(s.type, nodes, global_c_id);
            }
        }
    }

    file.close();
    return m;
}

// -----------------------------------------------------------------------------
// Primary Entry Point
// -----------------------------------------------------------------------------
mesh::RawMesh read_cgns_parallel(const std::string& path, MPI_Comm comm) {
    int nprocs = 0, rank = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    mesh::RawMesh m;
    m.nprocs = nprocs;
    m.rank = rank;
    m.comm = comm;

    // Read boundary condition metadata in serial on rank 0 and broadcast
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

    check(cgp_pio_mode(CGP_COLLECTIVE), "cgp_pio_mode(CGP_COLLECTIVE)", comm);

    File file(path, comm);
    const int f_id = file.id();

    check(cg_version(f_id, &m.gfm.cgns_version), "cg_version", comm);
    check(cg_precision(f_id, &m.gfm.file_integer_precision), "cg_precision", comm);
    check(cg_get_file_type(f_id, &m.gfm.storage_type), "cg_get_file_type", comm);

    const char* storage_str = "Unknown";
    if (m.gfm.storage_type == 1) storage_str = "ADF";
    else if (m.gfm.storage_type == 2) storage_str = "HDF5";
    else if (m.gfm.storage_type == 0) storage_str = "None/Error";
    mpi::log_stat("CGNS: Version='%.2f', Integer precision='%d bits', Storage type='%s'", 
                  m.gfm.cgns_version, m.gfm.file_integer_precision, storage_str);

    const bool mixed_needs_legacy_read =
        (m.gfm.cgns_version < 3.99f) ||
        (m.gfm.file_integer_precision != static_cast<int>(sizeof(cgsize_t) * 8));

    if (mixed_needs_legacy_read && rank == 0) {
        mpi::log_info("CGNS WARNING: CGNS MIXED sections will be read via legacy path (work, but it is ram have, if there are no MIXED - ignore this message)");
    }

    int nbases = 0;
    check(cg_nbases(f_id, &nbases), "cg_nbases", comm);
    if (nbases != 1) { mpi::fatal(comm, "Expected exactly 1 base in CGNS file"); }

    const int B = 1;
    int celldim = 0, physdim = 0;
    char basename[33] = "";
    check(cg_base_read(f_id, B, basename, &celldim, &physdim), "cg_base_read", comm);

    // Route 2D meshes to the extrusion procedure
    if (celldim == 2) {
        file.close();
        return read_cgns_2d_and_extrude_parallel(path, comm, /*dz=*/-1.0, std::move(m.bcs));
    }

    if (celldim != 3 || physdim != 3) {
        mpi::fatal(comm, "Base must be 3D (CellDim=3, PhysDim=3) or 2D (CellDim=2)");
    }

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

    int nsecs = 0;
    check(cg_nsections(f_id, B, Z, &nsecs), "cg_nsections", comm);

    for (int S = 1; S <= nsecs; ++S) {
        char secname[33] = "";
        ElementType etype;
        cgsize_t start = 0, end = 0;
        int nbndry = 0, parent_flag = 0;

        check(cg_section_read(f_id, B, Z, S, secname,
                              &etype, &start, &end, &nbndry, &parent_flag), 
              "cg_section_read", comm);

        // Skip 1D boundary elements in pure 3D meshes
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

        if (sm.start <= m.n_cells_g) {
            m.vol_secs.push_back(sm);
        } else {
            m.surf_secs.push_back(sm);
        }
    }

    std::sort(m.vol_secs.begin(), m.vol_secs.end(),
              [](const mesh::SectionMeta& a, const mesh::SectionMeta& b) noexcept { 
                  return a.start < b.start; });

    GlobalIndex total_cells = 0;
    for (auto& s : m.vol_secs) {
        s.cell_offset = total_cells;
        total_cells += (s.end - s.start + 1);
    }

    if (m.n_cells_g == 0 || total_cells != m.n_cells_g) {
        mpi::fatal(comm, "Mismatch or zero volume cells in CGNS zone");
    }

    mpi::log_stat("CGNS: Volume sections=%zu, Total Cells=%lld, Boundary sections=%zu", 
                  m.vol_secs.size(), static_cast<long long>(m.n_cells_g), m.surf_secs.size());

    for (const auto& b : m.bcs) {
        m.patch_list.push_back({b.name, b.cgns_type});
        mpi::log_stat("CGNS: BC '%s' Type=%s Faces=%zu -> PatchId=%d", 
                      b.name.c_str(), b.cgns_type.c_str(), b.eids.size(), 
                      static_cast<int>(m.patch_list.size() - 1));
    }

    // Read surface sections collectively
    for (const auto& s : m.surf_secs) {
        const GlobalIndex sec_n = s.end - s.start + 1;
        const auto d = mpi::block_displ(sec_n, static_cast<std::size_t>(nprocs));

        const GlobalIndex lo = d[static_cast<std::size_t>(rank)];
        const GlobalIndex hi = d[static_cast<std::size_t>(rank) + 1];

        const std::size_t local_count = (lo < hi) ? static_cast<std::size_t>(hi - lo) : 0;
        const GlobalIndex sec_start = s.start + lo;
        const GlobalIndex sec_end = s.start + hi;

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
            const std::size_t npt = static_cast<std::size_t>(mesh::kNodesPerType[static_cast<std::size_t>(s.type)]);
            std::vector<cgsize_t> buf(local_count * npt);

            const cgsize_t rs = (lo < hi) ? static_cast<cgsize_t>(s.start + lo) : 1;
            const cgsize_t re = (lo < hi) ? static_cast<cgsize_t>(s.start + hi - 1) : 0;

            cgsize_t* pbuf = (local_count > 0) ? buf.data() : nullptr;

            check(cgp_elements_read_data(f_id, B, Z, s.sec_idx, rs, re, pbuf),
                  "cgp_elements_read_data(surface)", comm);

            if (lo >= hi) continue;

            for (std::size_t i = 0; i < local_count; ++i) {
                mesh::SurfElem se{};
                for (std::size_t k = 0; k < npt; ++k) {
                    se.key.v[k] = static_cast<GlobalIndex>(buf[i * npt + k] - 1);
                }
                sort_face_key_nodes(se.key, static_cast<int>(npt));

                se.eid = sec_start + static_cast<GlobalIndex>(i);
                se.patch = local_patch[i];

                m.surf_elems.push_back(se);
            }
        }
    }

    m.cell_displ = mpi::block_displ(m.n_cells_g, static_cast<std::size_t>(nprocs));
    m.node_displ = mpi::block_displ(m.n_nodes_g, static_cast<std::size_t>(nprocs));

    const LocalIndex nl = m.n_local_cells();
    m.ctype.reserve(static_cast<std::size_t>(nl));

    m.cnodes_offsets.reserve(static_cast<std::size_t>(nl) + 1);
    m.cnodes_offsets.push_back(0);

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
    }

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
    return m;
}

} // namespace cfd::io::cgns