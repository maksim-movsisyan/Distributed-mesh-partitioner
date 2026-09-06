#include "cfd/io/solver_mesh/hdf5_reader.hpp"

#include <hdf5.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <sstream>
#include <type_traits>
#include <vector>
#include <cassert>

#include "cfd/core/types.hpp"
#include "cfd/mpi/log.hpp"

namespace cfd::io::solver_mesh {

namespace {

// Compile-time HDF5 type resolver for core types
template <typename T>
[[nodiscard]] inline hid_t get_h5_type() noexcept {
    if constexpr (std::is_same_v<T, float>)                  return H5T_NATIVE_FLOAT;
    else if constexpr (std::is_same_v<T, double>)            return H5T_NATIVE_DOUBLE;
    else if constexpr (std::is_same_v<T, int8_t>)            return H5T_NATIVE_INT8;
    else if constexpr (std::is_same_v<T, uint8_t>)           return H5T_NATIVE_UINT8;
    else if constexpr (std::is_same_v<T, int16_t>)           return H5T_NATIVE_INT16;
    else if constexpr (std::is_same_v<T, uint16_t>)          return H5T_NATIVE_UINT16;
    else if constexpr (std::is_same_v<T, int32_t>)           return H5T_NATIVE_INT32;
    else if constexpr (std::is_same_v<T, uint32_t>)          return H5T_NATIVE_UINT32;
    else if constexpr (std::is_same_v<T, int64_t>)           return H5T_NATIVE_INT64;
    else if constexpr (std::is_same_v<T, uint64_t>)          return H5T_NATIVE_UINT64;
    else if constexpr (sizeof(T) == 4 && std::is_integral_v<T>) return H5T_NATIVE_INT32;
    else if constexpr (sizeof(T) == 8 && std::is_integral_v<T>) return H5T_NATIVE_INT64;
    else return H5I_INVALID_HID;
}

// RAII Guard for HDF5 Handles
struct H5Id {
    hid_t id = H5I_INVALID_HID;

    H5Id() noexcept = default;
    explicit H5Id(hid_t h) noexcept : id(h) {}
    ~H5Id() { reset(); }

    H5Id(const H5Id&) = delete;
    H5Id& operator=(const H5Id&) = delete;

    H5Id(H5Id&& o) noexcept : id(o.id) { o.id = H5I_INVALID_HID; }
    H5Id& operator=(H5Id&& o) noexcept {
        if (this != &o) {
            reset();
            id = o.id;
            o.id = H5I_INVALID_HID;
        }
        return *this;
    }

    [[nodiscard]] operator hid_t() const noexcept { return id; }

    void reset() noexcept {
        if (id > 0) {
            H5I_type_t type = H5Iget_type(id);
            if (type == H5I_FILE)            H5Fclose(id);
            else if (type == H5I_GROUP)      H5Gclose(id);
            else if (type == H5I_DATASET)    H5Dclose(id);
            else if (type == H5I_DATASPACE)  H5Sclose(id);
            else if (type == H5I_DATATYPE)   H5Tclose(id);
            else if (type == H5I_GENPROP_LST) H5Pclose(id);
            else if (type == H5I_ATTR)       H5Aclose(id);
            id = H5I_INVALID_HID;
        }
    }
};

template <typename T>
void read_dataset_1d(
    hid_t loc_id,
    const std::string& name,
    uint64_t offset_val,
    uint64_t count_val,
    T* out_buf,
    hid_t dxpl_id) {
    H5Id dset(H5Dopen2(loc_id, name.c_str(), H5P_DEFAULT));
    H5Id filespace(H5Dget_space(dset));
    hid_t h5_type = get_h5_type<T>();

    H5Id memspace;
    if (count_val > 0) {
        hsize_t count[1]  = {static_cast<hsize_t>(count_val)};
        hsize_t offset[1] = {static_cast<hsize_t>(offset_val)};
        memspace = H5Id(H5Screate_simple(1, count, nullptr));
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, nullptr, count, nullptr);
    } else {
        memspace = H5Id(H5Screate(H5S_NULL));
        H5Sselect_none(filespace);
    }

    H5Dread(dset, h5_type, memspace, filespace, dxpl_id, out_buf);
}

void read_dataset_2d_vec3(
    hid_t loc_id,
    const std::string& name,
    uint64_t offset_val,
    uint64_t count_val,
    double* out_x,
    double* out_y,
    double* out_z,
    hid_t dxpl_id) {
    H5Id dset(H5Dopen2(loc_id, name.c_str(), H5P_DEFAULT));
    H5Id filespace(H5Dget_space(dset));

    std::vector<double> flat_buf(static_cast<std::size_t>(count_val) * 3);

    H5Id memspace;
    if (count_val > 0) {
        hsize_t count[2]  = {static_cast<hsize_t>(count_val), 3};
        hsize_t offset[2] = {static_cast<hsize_t>(offset_val), 0};
        memspace = H5Id(H5Screate_simple(2, count, nullptr));
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, nullptr, count, nullptr);
    } else {
        memspace = H5Id(H5Screate(H5S_NULL));
        H5Sselect_none(filespace);
    }

    H5Dread(dset, H5T_NATIVE_DOUBLE, memspace, filespace, dxpl_id, flat_buf.data());

    for (std::size_t i = 0; i < static_cast<std::size_t>(count_val); ++i) {
        out_x[i] = flat_buf[i * 3 + 0];
        out_y[i] = flat_buf[i * 3 + 1];
        out_z[i] = flat_buf[i * 3 + 2];
    }
}

} // anonymous namespace

void import_mesh_hdf5(mesh::MeshPart& mp, const std::string& filepath, MPI_Comm comm) {
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    mp.rank = rank;
    mp.nprocs = nprocs;

    // 1. Open Parallel HDF5 Container
    H5Id fapl(H5Pcreate(H5P_FILE_ACCESS));
    H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL);

    H5Id file(H5Fopen(filepath.c_str(), H5F_ACC_RDONLY, fapl));
    if (file < 0) {
        std::stringstream ss;
        ss << "Failed to open HDF5 mesh file for reading: " << filepath.c_str();
        auto result = ss.str();
        mpi::fatal(comm, result);
    }

    H5Id dxpl(H5Pcreate(H5P_DATASET_XFER));
    H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE);

    // 2. Read Root Group Global Attributes
    {
        auto read_attr = [&](const char* name, hid_t type, void* val) {
            H5Id attr(H5Aopen(file, name, H5P_DEFAULT));
            H5Aread(attr, type, val);
        };

        int file_nprocs = 0;
        uint64_t n_cells_g = 0, n_faces_g = 0, n_bfaces_g = 0, n_nodes_g = 0;

        read_attr("nprocs", H5T_NATIVE_INT, &file_nprocs);
        read_attr("n_cells_global", H5T_NATIVE_UINT64, &n_cells_g);
        read_attr("n_faces_global", H5T_NATIVE_UINT64, &n_faces_g);
        read_attr("n_bfaces_global", H5T_NATIVE_UINT64, &n_bfaces_g);
        read_attr("n_nodes_global", H5T_NATIVE_UINT64, &n_nodes_g);

        if (file_nprocs != nprocs) {
            std::stringstream ss;
            ss << "HDF5 mesh file was partitioned for " << file_nprocs << " ranks, but current communicator has " << nprocs << " ranks.";
            auto result = ss.str();
            mpi::fatal(comm, result);
        }

        mp.n_cells_g  = static_cast<GlobalIndex>(n_cells_g);
        mp.n_faces_g  = static_cast<GlobalIndex>(n_faces_g);
        mp.n_bfaces_g = static_cast<GlobalIndex>(n_bfaces_g);
        mp.n_nodes_g  = static_cast<GlobalIndex>(n_nodes_g);

        read_attr("bbox_lo", H5T_NATIVE_DOUBLE, mp.bbox_lo);
        read_attr("bbox_hi", H5T_NATIVE_DOUBLE, mp.bbox_hi);
    }

    // 3. Read Partition Topology Tables
    const auto nprocs_sz = static_cast<std::size_t>(nprocs);
    std::vector<int> rank2part(nprocs_sz);
    std::vector<uint64_t> cell_offsets(nprocs_sz + 1);
    std::vector<uint64_t> cell_nodes_offsets(nprocs_sz + 1);
    std::vector<uint64_t> face_offsets(nprocs_sz + 1);
    std::vector<uint64_t> face_nodes_offsets(nprocs_sz + 1);
    std::vector<uint64_t> node_offsets(nprocs_sz + 1);
    std::vector<uint64_t> comm_nb_offsets(nprocs_sz + 1);

    {
        H5Id g_part(H5Gopen2(file, "partition", H5P_DEFAULT));

        auto read_full_1d = [&](const char* name, hid_t type, void* buf) {
            H5Id dset(H5Dopen2(g_part, name, H5P_DEFAULT));
            H5Dread(dset, type, H5S_ALL, H5S_ALL, dxpl, buf);
        };

        read_full_1d("rank2part", H5T_NATIVE_INT, rank2part.data());
        read_full_1d("cell_offsets", H5T_NATIVE_UINT64, cell_offsets.data());
        read_full_1d("cell_nodes_offsets", H5T_NATIVE_UINT64, cell_nodes_offsets.data());
        read_full_1d("face_offsets", H5T_NATIVE_UINT64, face_offsets.data());
        read_full_1d("face_nodes_offsets", H5T_NATIVE_UINT64, face_nodes_offsets.data());
        read_full_1d("node_offsets", H5T_NATIVE_UINT64, node_offsets.data());
        read_full_1d("comm_nb_offsets", H5T_NATIVE_UINT64, comm_nb_offsets.data());
    }

    // 4. Resolve Topology-Aware Hyperslab Parameters for Current Rank
    const int part_id = rank2part[static_cast<std::size_t>(rank)];
    const auto p_sz   = static_cast<std::size_t>(part_id);

    const uint64_t c_off   = cell_offsets[p_sz];
    const uint64_t c_cnt   = cell_offsets[p_sz + 1] - c_off;

    const uint64_t cn_off  = cell_nodes_offsets[p_sz];
    const uint64_t cn_cnt  = cell_nodes_offsets[p_sz + 1] - cn_off;

    const uint64_t f_off   = face_offsets[p_sz];
    const uint64_t f_cnt   = face_offsets[p_sz + 1] - f_off;

    const uint64_t fn_off  = face_nodes_offsets[p_sz];
    const uint64_t fn_cnt  = face_nodes_offsets[p_sz + 1] - fn_off;

    const uint64_t n_off   = node_offsets[p_sz];
    const uint64_t n_cnt   = node_offsets[p_sz + 1] - n_off;

    const uint64_t nb_off  = comm_nb_offsets[p_sz];
    const uint64_t nb_cnt  = comm_nb_offsets[p_sz + 1] - nb_off;

    mp.n_cells = static_cast<LocalIndex>(c_cnt);
    mp.n_faces = static_cast<LocalIndex>(f_cnt);
    mp.n_nodes = static_cast<LocalIndex>(n_cnt);

    const auto n_cells_sz = static_cast<std::size_t>(mp.n_cells);
    const auto n_faces_sz = static_cast<std::size_t>(mp.n_faces);
    const auto n_nodes_sz = static_cast<std::size_t>(mp.n_nodes);

    // 5. Read /cells Group
    {
        H5Id g_cells(H5Gopen2(file, "cells", H5P_DEFAULT));

        std::vector<uint8_t> cell_types_u8(n_cells_sz);
        mp.cell_gid.resize(n_cells_sz);
        mp.cell_donor.resize(n_cells_sz);
        mp.cell_volume.resize(n_cells_sz);
        mp.cell_centroid_x.resize(n_cells_sz);
        mp.cell_centroid_y.resize(n_cells_sz);
        mp.cell_centroid_z.resize(n_cells_sz);

        mp.cell_nodes_offsets.resize(n_cells_sz + 1);
        mp.cell_nodes.resize(static_cast<std::size_t>(cn_cnt));

        read_dataset_1d(g_cells, "type", c_off, c_cnt, cell_types_u8.data(), dxpl);
        read_dataset_1d(g_cells, "gid", c_off, c_cnt, mp.cell_gid.data(), dxpl);
        read_dataset_1d(g_cells, "donor", c_off, c_cnt, mp.cell_donor.data(), dxpl);
        read_dataset_1d(g_cells, "volume", c_off, c_cnt, mp.cell_volume.data(), dxpl);
        read_dataset_1d(g_cells, "nodes_offsets", c_off, c_cnt, mp.cell_nodes_offsets.data(), dxpl);
        read_dataset_1d(g_cells, "nodes", cn_off, cn_cnt, mp.cell_nodes.data(), dxpl);

        read_dataset_2d_vec3(g_cells, "centroid", c_off, c_cnt,
                             mp.cell_centroid_x.data(), mp.cell_centroid_y.data(), mp.cell_centroid_z.data(), dxpl);

        mp.cell_type.resize(n_cells_sz);
        for (std::size_t i = 0; i < n_cells_sz; ++i) {
            mp.cell_type[i] = static_cast<mesh::CellType>(cell_types_u8[i]);
        }

        mp.cell_nodes_offsets[n_cells_sz] = static_cast<LocalIndex>(cn_cnt);

        LocalIndex own_cnt = 0;
        for (LocalIndex i = 0; i < mp.n_cells; ++i) {
            if (mp.cell_donor[static_cast<std::size_t>(i)] == -1) {
                ++own_cnt;
            } else {
                break;
            }
        }
        mp.n_own = own_cnt;
    }

    // 6. Read /faces Group
    {
        H5Id g_faces(H5Gopen2(file, "faces", H5P_DEFAULT));

        std::vector<uint8_t> face_types_u8(n_faces_sz);
        mp.face_owner.resize(n_faces_sz);
        mp.face_neigh.resize(n_faces_sz);
        mp.face_patch.resize(n_faces_sz);
        mp.face_area.resize(n_faces_sz);
        mp.face_normal_x.resize(n_faces_sz);
        mp.face_normal_y.resize(n_faces_sz);
        mp.face_normal_z.resize(n_faces_sz);
        mp.face_centroid_x.resize(n_faces_sz);
        mp.face_centroid_y.resize(n_faces_sz);
        mp.face_centroid_z.resize(n_faces_sz);

        mp.face_nodes_offsets.resize(n_faces_sz + 1);
        mp.face_nodes.resize(static_cast<std::size_t>(fn_cnt));

        read_dataset_1d(g_faces, "owner", f_off, f_cnt, mp.face_owner.data(), dxpl);
        read_dataset_1d(g_faces, "neigh", f_off, f_cnt, mp.face_neigh.data(), dxpl);
        read_dataset_1d(g_faces, "patch", f_off, f_cnt, mp.face_patch.data(), dxpl);
        read_dataset_1d(g_faces, "type", f_off, f_cnt, face_types_u8.data(), dxpl);
        read_dataset_1d(g_faces, "area", f_off, f_cnt, mp.face_area.data(), dxpl);
        read_dataset_1d(g_faces, "nodes_offsets", f_off, f_cnt, mp.face_nodes_offsets.data(), dxpl);
        read_dataset_1d(g_faces, "nodes", fn_off, fn_cnt, mp.face_nodes.data(), dxpl);

        read_dataset_2d_vec3(g_faces, "normal", f_off, f_cnt,
                             mp.face_normal_x.data(), mp.face_normal_y.data(), mp.face_normal_z.data(), dxpl);
        read_dataset_2d_vec3(g_faces, "centroid", f_off, f_cnt,
                             mp.face_centroid_x.data(), mp.face_centroid_y.data(), mp.face_centroid_z.data(), dxpl);

        mp.face_type.resize(n_faces_sz);
        for (std::size_t i = 0; i < n_faces_sz; ++i) {
            mp.face_type[i] = static_cast<mesh::CellType>(face_types_u8[i]);
        }

        mp.face_nodes_offsets[n_faces_sz] = static_cast<LocalIndex>(fn_cnt);
    }

    // 7. Read /nodes Group
    {
        H5Id g_nodes(H5Gopen2(file, "nodes", H5P_DEFAULT));

        mp.node_gid.resize(n_nodes_sz);
        mp.node_x.resize(n_nodes_sz);
        mp.node_y.resize(n_nodes_sz);
        mp.node_z.resize(n_nodes_sz);

        read_dataset_1d(g_nodes, "gid", n_off, n_cnt, mp.node_gid.data(), dxpl);
        read_dataset_2d_vec3(g_nodes, "coords", n_off, n_cnt,
                             mp.node_x.data(), mp.node_y.data(), mp.node_z.data(), dxpl);

        LocalIndex max_own_nid = 0;
        for (LocalIndex c = 0; c < mp.n_own; ++c) {
            const auto c_idx = static_cast<std::size_t>(c);
            const LocalIndex start = mp.cell_nodes_offsets[c_idx];
            const LocalIndex end   = mp.cell_nodes_offsets[c_idx + 1];
            for (LocalIndex k = start; k < end; ++k) {
                max_own_nid = std::max(max_own_nid, mp.cell_nodes[static_cast<std::size_t>(k)] + 1);
            }
        }
        mp.n_nodes_own = max_own_nid;
    }

    // 8. Read /comm Group
    {
        H5Id g_comm(H5Gopen2(file, "comm", H5P_DEFAULT));

        mp.nb_ranks.resize(static_cast<std::size_t>(nb_cnt));
        read_dataset_1d(g_comm, "nb_ranks", nb_off, nb_cnt, mp.nb_ranks.data(), dxpl);

        const auto n_nb_sz = static_cast<std::size_t>(nb_cnt);
        const uint64_t comm_off_pos = nb_off + static_cast<uint64_t>(part_id);
        const uint64_t comm_off_cnt = nb_cnt + 1;

        mp.send_offsets.resize(n_nb_sz + 1);
        mp.recv_offsets.resize(n_nb_sz + 1);
        read_dataset_1d(g_comm, "send_offsets", comm_off_pos, comm_off_cnt, mp.send_offsets.data(), dxpl);
        read_dataset_1d(g_comm, "recv_offsets", comm_off_pos, comm_off_cnt, mp.recv_offsets.data(), dxpl);

        const uint64_t my_snd_cnt = static_cast<uint64_t>(mp.send_offsets.back());
        const uint64_t my_rcv_cnt = static_cast<uint64_t>(mp.recv_offsets.back());

        uint64_t snd_offset_pos = 0;
        uint64_t rcv_offset_pos = 0;
        MPI_Exscan(&my_snd_cnt, &snd_offset_pos, 1, MPI_UINT64_T, MPI_SUM, comm);
        MPI_Exscan(&my_rcv_cnt, &rcv_offset_pos, 1, MPI_UINT64_T, MPI_SUM, comm);
        if (rank == 0) {
            snd_offset_pos = 0;
            rcv_offset_pos = 0;
        }

        mp.send_owned_local.resize(static_cast<std::size_t>(my_snd_cnt));
        mp.recv_ghost_local.resize(static_cast<std::size_t>(my_rcv_cnt));

        read_dataset_1d(g_comm, "send_owned_cells", snd_offset_pos, my_snd_cnt, mp.send_owned_local.data(), dxpl);
        read_dataset_1d(g_comm, "recv_ghost_cells", rcv_offset_pos, my_rcv_cnt, mp.recv_ghost_local.data(), dxpl);
    }

    // 9. Read /patches Group
    {
        H5Id g_patch(H5Gopen2(file, "patches", H5P_DEFAULT));

        H5Id d_names(H5Dopen2(g_patch, "names", H5P_DEFAULT));
        H5Id s_names(H5Dget_space(d_names));
        hsize_t n_patches_h5 = 0;
        H5Sget_simple_extent_dims(s_names, &n_patches_h5, nullptr);

        const auto n_patches = static_cast<std::size_t>(n_patches_h5);
        constexpr std::size_t kStrLen = 64;
        H5Id str_type(H5Tcopy(H5T_C_S1));
        H5Tset_size(str_type, kStrLen);

        std::vector<std::array<char, kStrLen>> names_buf(n_patches);
        std::vector<std::array<char, kStrLen>> types_buf(n_patches);

        H5Dread(d_names, str_type, H5S_ALL, H5S_ALL, dxpl, names_buf.data());
        H5Dread(H5Id(H5Dopen2(g_patch, "cgns_types", H5P_DEFAULT)), str_type, H5S_ALL, H5S_ALL, dxpl, types_buf.data());

        mp.patches.resize(n_patches);
        for (std::size_t p = 0; p < n_patches; ++p) {
            mp.patches[p].name = std::string(names_buf[p].data());
            mp.patches[p].cgns_type = std::string(types_buf[p].data());
        }

        const uint64_t patch_off_pos = static_cast<uint64_t>(part_id) * (static_cast<uint64_t>(n_patches) + 1);
        const uint64_t patch_off_cnt = static_cast<uint64_t>(n_patches) + 1;

        mp.patch_face_offsets.resize(n_patches + 1);
        read_dataset_1d(g_patch, "patch_face_offsets", patch_off_pos, patch_off_cnt, mp.patch_face_offsets.data(), dxpl);

        const uint64_t my_patch_faces_cnt = static_cast<uint64_t>(mp.patch_face_offsets.back());
        uint64_t patch_faces_offset_pos = 0;
        MPI_Exscan(&my_patch_faces_cnt, &patch_faces_offset_pos, 1, MPI_UINT64_T, MPI_SUM, comm);
        if (rank == 0) patch_faces_offset_pos = 0;

        mp.patch_faces.resize(static_cast<std::size_t>(my_patch_faces_cnt));
        read_dataset_1d(g_patch, "patch_faces", patch_faces_offset_pos, my_patch_faces_cnt, mp.patch_faces.data(), dxpl);
    }

    // 10. Reconstruct 3-zone face partition boundaries (n_internal_faces & n_inner_faces)
    LocalIndex n_internal = 0;
    LocalIndex n_inner = 0;

    for (LocalIndex f = 0; f < mp.n_faces; ++f) {
        const LocalIndex neigh = mp.face_neigh[static_cast<std::size_t>(f)];

        if (neigh >= 0) {
            ++n_inner;
            if (neigh < mp.n_own) {
                // Invariant: Zone 0 (Pure Internal) must strictly precede Zone 1 (Coupled)
                assert(n_inner == n_internal + 1 && "Corrupted face layout: Zone 0 must be contiguous at the start");
                ++n_internal;
            }
        } else {
            break; // Reached Zone 2 (Boundary faces with neigh == -1)
        }
    }

    mp.n_internal_faces = n_internal;
    mp.n_inner_faces = n_inner;

    if (rank == 0) {
        mpi::log_stat("INFO: Mesh successfully loaded from Parallel HDF5: %s", filepath.c_str());
    }
}

} // namespace cfd::io::solver_mesh