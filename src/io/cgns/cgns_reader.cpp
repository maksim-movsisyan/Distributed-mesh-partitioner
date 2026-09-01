#include "cfd/io/cgns/cgns_reader.hpp"
#include <cgnstypes.h>
#include <cgns_io.h>
#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
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

// Reads only THIS rank's slice of a MIXED section's flattened connectivity,
// fully in parallel via the cgp_poly_elements_read_data_* API — no rank
// ever downloads the full section, so this scales to grids where a single
// process couldn't hold the whole MIXED block in memory.
//
// [rs, re] are absolute, 1-based, file-space element indices within the
// section (same convention as cgp_elements_read_data's start/end elsewhere
// in this file: for a surface section rs = s.start + lo, for a volume
// section rs = s.start + (lo - s.cell_offset), etc). has_data must be false
// when this rank owns no elements in the section.
//
// Important: unlike cgp_elements_read_data, a degenerate (rs=1, re=0) range
// with a valid dummy buffer is NOT a safe way to signal "no data" here —
// the offset arithmetic in cgp_poly_elements_read_data_offsets does not
// collapse to a zero-size read for that case (verified against the CGNS
// pcgnslib.c source), so it would make HDF5 attempt a real single-element
// read at a bogus offset. The documented, correct way to skip is to pass a
// NULL pointer, which is what the has_data == false branch below does.
//
// On-disk integer widths (CRITICAL): cgp_poly_elements_read_data_offsets/
// _elements pick their HDF5 MEMORY datatype from the FILE's stored datatype
// of the array being read (pcgnslib.c uses section->connect_offset->data_type
// and section->connect->data_type respectively), NOT from sizeof(cgsize_t)
// the way cgp_elements_read_data does — HDF5 packs file-width elements into
// whatever buffer we hand it, with no conversion. And nothing forces an
// exporter to store ElementConnectivity and ElementStartOffset in the same
// width, or in the file-wide index precision: mesh/sphere_v4.cgns (CGNS
// 4.52) stores ElementConnectivity as I4 but ElementStartOffset as I8. With
// 32-bit cgsize_t, reading such offsets into a vector<cgsize_t> makes HDF5
// write 8 bytes per element into a 4-bytes-per-element buffer — a heap
// overflow that surfaces as a segfault deep inside MPI-IO's free(). So each
// array's on-disk width is queried first (mixed_array_disk_bytes below) and
// every read goes through a width-matched scratch buffer before being
// normalized to cgsize_t (read_mixed_width_safe below). The values
// themselves always fit: they index arrays the CGNS library already
// addresses with cgsize_t. Both queries run identically on EVERY rank
// before the has_data branch, keeping the defensive metadata-read lockstep
// the BC block above uses.
//
// This intentionally never touches cgp_pio_mode(): CGP_COLLECTIVE (already
// set once, globally, at the top of read_cgns_parallel) is sufficient for
// this API and keeps every read in this file collective. If a particular
// CGNS/HDF5 build turns out to need CGP_INDEPENDENT for these two calls,
// switch it locally right here (and only here) and flip back to
// CGP_COLLECTIVE immediately after, e.g.:
//     check(cgp_pio_mode(CGP_INDEPENDENT), "cgp_pio_mode(INDEPENDENT)");
//     ... the two cgp_poly_elements_read_data_* calls ...
//     check(cgp_pio_mode(CGP_COLLECTIVE), "cgp_pio_mode(COLLECTIVE)");
// so that only MIXED-section reads are ever affected.

// On-disk element width (4 for "I4", 8 for "I8") of one of a MIXED section's
// child data arrays ("ElementStartOffset" / "ElementConnectivity"), or 0 if
// the array is not present in the file. The mid-level CGNS API has no
// accessor for these datatypes — cg_section_read reports none, and cg_goto
// cannot descend under Elements_t nodes at all (its whitelist allows only
// UserDefinedData_t there) — so this goes through the low-level cgio layer.
// The walk mirrors CGNS's own index semantics (B-th CGNSBase_t under the
// root, Z-th Zone_t, S-th Elements_t, then the child by its fixed array
// name) and uses only independent serial metadata reads, which are safe
// under cgp_open; every rank performs the identical walk.
int mixed_array_disk_bytes(int f_id, int B, int Z, int sec_idx, const char* array_name) {
    int cgio = 0;
    check(cg_get_cgio(f_id, &cgio), "cg_get_cgio(MIXED)");

    double node = 0;
    check(cgio_get_root_id(cgio, &node), "cgio_get_root_id(MIXED)");

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
        check(cgio_number_children(cgio, node, &nchildren), "cgio_number_children(MIXED)");

        std::vector<double> children(static_cast<std::size_t>(nchildren));
        int nret = 0;
        check(cgio_children_ids(cgio, node, 1, nchildren, &nret, children.data()),
              "cgio_children_ids(MIXED)");

        int seen = 0;
        bool found = false;
        for (const double child : children) {
            char label[33] = "";
            check(cgio_get_label(cgio, child, label), "cgio_get_label(MIXED)");
            if (std::strcmp(label, step.label) == 0 && ++seen == step.index) {
                node = child;
                found = true;
                break;
            }
        }
        if (!found) {
            throw std::runtime_error(std::string("CGNS MIXED: cgio tree has no ") +
                                     step.label + " #" + std::to_string(step.index));
        }
    }

    double array_id = 0;
    if (cgio_get_node_id(cgio, node, array_name, &array_id) != CG_OK) {
        return 0;
    }

    char data_type[8] = "";
    check(cgio_get_data_type(cgio, array_id, data_type), "cgio_get_data_type(MIXED)");
    if (std::strcmp(data_type, "I4") == 0) { return 4; }
    if (std::strcmp(data_type, "I8") == 0) { return 8; }
    throw std::runtime_error(std::string("CGNS MIXED: unexpected on-disk data type '") +
                             data_type + "' of " + array_name);
}

// Performs one collective cgp_poly_elements_read_data_* call for `count`
// values whose elements are `width_bytes` wide ON DISK (cgp_call receives
// the raw buffer, reinterpreted as cgsize_t*, and must run the call and
// check() it), then returns the values normalized to cgsize_t. When the
// on-disk width differs from sizeof(cgsize_t), the call goes through a
// scratch vector of the exact on-disk width first: pcgnslib.c pins the HDF5
// memory datatype to the FILE's datatype for these two calls, so handing it
// a cgsize_t buffer of the other width would either overflow it (I8 file,
// 32-bit cgsize_t) or fill it with interleaved halves (I4 file, 64-bit
// cgsize_t).
template <typename CgpCall>
std::vector<cgsize_t> read_mixed_width_safe(std::size_t count, int width_bytes, CgpCall&& cgp_call) {
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

    throw std::runtime_error("CGNS MIXED: unsupported on-disk integer width");
}

std::vector<cgsize_t> read_mixed_section_local(
    int f_id, int B, int Z, int sec_idx,
    cgsize_t rs, cgsize_t re, bool has_data) {

    // Queried by every rank, empty ones included, BEFORE the empty-rank
    // early return — same defensive lockstep of identical serial metadata
    // reads on all ranks as the BC block above.
    const int offset_width = mixed_array_disk_bytes(f_id, B, Z, sec_idx, "ElementStartOffset");
    const int conn_width = mixed_array_disk_bytes(f_id, B, Z, sec_idx, "ElementConnectivity");

    if (offset_width == 0 || conn_width == 0) {
        throw std::runtime_error(
            "CGNS MIXED: section has no ElementStartOffset/ElementConnectivity "
            "array on disk (unexpected for a CGNS >= 4.0 file)");
    }

    if (!has_data) {
        check(cgp_poly_elements_read_data_offsets(f_id, B, Z, sec_idx, rs, re, nullptr),
              "cgp_poly_elements_read_data_offsets(MIXED, empty)");
        check(cgp_poly_elements_read_data_elements(f_id, B, Z, sec_idx, rs, re, nullptr, nullptr),
              "cgp_poly_elements_read_data_elements(MIXED, empty)");
        return {};
    }

    // offsets holds the local slice of ElementStartOffset for elements
    // [rs, re]: (re - rs + 2) cumulative counts in FILE space (i.e. relative
    // to the whole section's connectivity, not to this rank's slice), one
    // more than the number of elements, so we can size and index the
    // flattened data. The file-space convention is load-bearing:
    // cgp_poly_elements_read_data_elements plugs offsets[0] and
    // offsets[end-start+1] straight into the global ElementConnectivity
    // hyperslab, so the normalized values below must stay file-space.
    const std::size_t n_offsets = static_cast<std::size_t>(re - rs + 2);
    std::vector<cgsize_t> offsets = read_mixed_width_safe(
        n_offsets, offset_width, [&](cgsize_t* buf) {
            check(cgp_poly_elements_read_data_offsets(f_id, B, Z, sec_idx, rs, re, buf),
                  "cgp_poly_elements_read_data_offsets(MIXED)");
        });

    const cgsize_t local_size = offsets.back() - offsets.front();

    std::vector<cgsize_t> elements = read_mixed_width_safe(
        static_cast<std::size_t>(local_size), conn_width, [&](cgsize_t* buf) {
            check(cgp_poly_elements_read_data_elements(f_id, B, Z, sec_idx, rs, re,
                  offsets.data(), buf),
                  "cgp_poly_elements_read_data_elements(MIXED)");
        });

    return elements;
}

// Fallback for MIXED sections written by CGNS libraries that predate the
// ElementStartOffset side-array (SIDS/CGNS_VERSION < 4.0). Such files only
// store the flat, type-tagged element stream ("ElementConnectivity") with
// no separate offset dataset on disk at all, so cgp_poly_elements_read_data_
// offsets/_elements above cannot work — it isn't a bug in how we call it,
// the required HDF5 dataset simply does not exist in the file, and the
// parallel API refuses outright ("H5Dopen2() failed") rather than
// reconstructing offsets on the fly the way the serial cg_poly_elements_read
// does. So for these files every rank reads the WHOLE section here (same as
// the original, pre-refactor code) and keeps only its own local slice.
// This is correct but not memory-scalable — the fix for that is not more
// clever code, it's re-exporting the mesh with a CGNS >= 4.x library so the
// file actually contains ElementStartOffset and the fast path can be used.
//
// local_elem_start/local_elem_end are 0-based, SECTION-RELATIVE element
// indices (i.e. offsets from s.start), matching elem_offsets' own indexing.
std::vector<cgsize_t> read_mixed_section_legacy_full(
    int f_id, int B, int Z, const mesh::SectionMeta& s,
    GlobalIndex local_elem_start, GlobalIndex local_elem_end) {

    const GlobalIndex sec_n = s.end - s.start + 1;
    cgsize_t datasize = 0;
    // Called unconditionally by every rank, even ones with an empty local
    // range below — cg_ElementDataSize/cg_poly_elements_read appear to need
    // collective participation on this file (same lesson learned the hard
    // way with the boundary-condition read: skipping these calls on some
    // ranks while others call them deadlocks).
    check(cg_ElementDataSize(f_id, B, Z, s.sec_idx, &datasize), "cg_ElementDataSize");

    if (datasize == 0 || sec_n == 0) {
        return {};
    }

    std::vector<cgsize_t> full_buf(static_cast<std::size_t>(datasize));
    std::vector<cgsize_t> elem_offsets(static_cast<std::size_t>(sec_n) + 1, 0);
    check(cg_poly_elements_read(f_id, B, Z, s.sec_idx,
          full_buf.data(), elem_offsets.data(), nullptr),
          "cg_poly_elements_read(MIXED, legacy)");

    if (local_elem_start >= local_elem_end) {
        return {};
    }

    const std::size_t byte_start = static_cast<std::size_t>(elem_offsets[static_cast<std::size_t>(local_elem_start)]);
    const std::size_t byte_end   = static_cast<std::size_t>(elem_offsets[static_cast<std::size_t>(local_elem_end)]);

    return std::vector<cgsize_t>(
        full_buf.begin() + static_cast<std::ptrdiff_t>(byte_start),
        full_buf.begin() + static_cast<std::ptrdiff_t>(byte_end));
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
            mpi::log_stat("CGNS WARNING: CGNS MIXED sections will be read (if exist) via ram-heavy fall back. See desciption in sgns_read.cpp line 205.");
        }
    }

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


    // Read Boundary Conditions (ZoneBC)
    //
    // NOTE: a rank-0-reads-then-broadcasts version of this block was tried
    // and reliably deadlocked: the file is open via cgp_open(), and on this
    // CGNS/HDF5 build the plain cg_* metadata/navigation calls used here
    // (cg_nbocos, cg_boco_info, cg_goto, ...) apparently still require every
    // rank to participate in lockstep — having only rank 0 call them while
    // the others skipped straight to the broadcast left rank 0 blocked
    // inside the HDF5 layer waiting for peers that never showed up. BC
    // point-lists are normally tiny compared to the volume mesh anyway, so
    // this loop stays fully collective (every rank reads identically).
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

            const std::size_t local_count = (lo < hi) ? static_cast<std::size_t>(hi - lo) : 0;

            if (s.is_mixed) {
                std::vector<cgsize_t> buf;
                if (mixed_needs_legacy_read) {
                    buf = read_mixed_section_legacy_full(f_id, B, Z, s, lo, hi);
                } else {
                    const cgsize_t rs = (lo < hi) ? static_cast<cgsize_t>(s.start + lo) : 1;
                    const cgsize_t re = (lo < hi) ? static_cast<cgsize_t>(s.start + hi - 1) : 0;
                    buf = read_mixed_section_local(f_id, B, Z, s.sec_idx, rs, re, lo < hi);
                }

                std::size_t ptr = 0;
                for (std::size_t i = 0; i < local_count; ++i) {
                    mesh::SurfElem se;
                    auto raw_type = static_cast<ElementType>(buf[ptr++]);
                    auto [stype, npt] = parse_cgns_element_header(raw_type, comm);

                    for (int k = 0; k < npt; ++k) {
                        se.key.v[static_cast<std::size_t>(k)] = static_cast<GlobalIndex>(buf[ptr++] - 1);
                    }
                    sort_face_key_nodes(se.key, npt);

                    se.eid = s.start + lo + static_cast<GlobalIndex>(i);
                    const auto it = eid2patch.find(se.eid);
                    se.patch = (it != eid2patch.end()) ? it->second : kInvalidPatchId;

                    m.surf_elems.push_back(se);
                }
            } else {
                // get number of nodes per section element type
                const std::size_t npt = static_cast<std::size_t>(mesh::kNodesPerType[static_cast<std::size_t>(s.type)]);
                std::vector<cgsize_t> buf(local_count * npt);

                // In collective mode, all ranks must participate; 0-sized reads pass rs=1, re=0
                const cgsize_t rs = (lo < hi) ? static_cast<cgsize_t>(s.start + lo) : 1;
                const cgsize_t re = (lo < hi) ? static_cast<cgsize_t>(s.start + hi - 1) : 0;

                //cgsize_t dummy_elem = 0;
                cgsize_t* pbuf = (local_count > 0) ? buf.data() : nullptr;

                check(cgp_elements_read_data(f_id, B, Z, s.sec_idx, rs, re, pbuf),
                      "cgp_elements_read_data(surface)");

                // skip blank ranges
                if (lo >= hi) continue;

                // loop over local surface elements
                for (std::size_t i = 0; i < local_count; ++i) {
                    mesh::SurfElem se;
                    for (std::size_t k = 0; k < npt; ++k) {
                        se.key.v[k] = static_cast<GlobalIndex>(buf[i * npt + k] - 1); // back to 0-based
                    }
                    sort_face_key_nodes(se.key, static_cast<int>(npt));

                    se.eid = s.start + lo + static_cast<GlobalIndex>(i);
                    const auto it = eid2patch.find(se.eid);
                    se.patch = (it != eid2patch.end()) ? it->second : kInvalidPatchId;

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
                buf = read_mixed_section_legacy_full(f_id, B, Z, s, lo - s.cell_offset, hi - s.cell_offset);
            } else {
                const cgsize_t rs = (lo < hi) ? static_cast<cgsize_t>(s.start + (lo - s.cell_offset)) : 1;
                const cgsize_t re = (lo < hi) ? static_cast<cgsize_t>(s.start + (hi - 1 - s.cell_offset)) : 0;
                buf = read_mixed_section_local(f_id, B, Z, s.sec_idx, rs, re, lo < hi);
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

            //cgsize_t dummy_elem = 0;
            cgsize_t* pbuf = (local_count > 0) ? buf.data() : nullptr;

            check(cgp_elements_read_data(f_id, B, Z, s.sec_idx, rs, re, pbuf),
                  "cgp_elements_read_data(volume)");

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

    double dummy_coord = 0.0;
    double* px = (nmy > 0) ? m.my_node_coords_x.data() : &dummy_coord;
    double* py = (nmy > 0) ? m.my_node_coords_y.data() : &dummy_coord;
    double* pz = (nmy > 0) ? m.my_node_coords_z.data() : &dummy_coord;

    check(cgp_coord_read_data(f_id, B, Z, 1, &rs, &re, px), "cgp_coord_read_data(X)");
    check(cgp_coord_read_data(f_id, B, Z, 2, &rs, &re, py), "cgp_coord_read_data(Y)");
    check(cgp_coord_read_data(f_id, B, Z, 3, &rs, &re, pz), "cgp_coord_read_data(Z)");

    mpi::log_rank("CGNS Rank %d: Local Cells=%d, Local Nodes=%zu", rank, nl, nmy);

    return m;
}

} // namespace cfd::io::cgns