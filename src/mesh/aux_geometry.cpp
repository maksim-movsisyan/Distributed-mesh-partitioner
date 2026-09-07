#include "cfd/mesh/aux_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "cfd/core/types.hpp"

namespace cfd::mesh {

void build_face_cell_dist_vectors(const MeshPart& mesh,
                                  std::vector<double>& face_cell_dist_x,
                                  std::vector<double>& face_cell_dist_y,
                                  std::vector<double>& face_cell_dist_z) {
    const std::size_t n_faces       = static_cast<std::size_t>(mesh.n_faces);
    const std::size_t n_inner_faces = static_cast<std::size_t>(mesh.n_inner_faces);

    face_cell_dist_x.resize(n_faces);
    face_cell_dist_y.resize(n_faces);
    face_cell_dist_z.resize(n_faces);

    const LocalIndex* CFD_RESTRICT face_owner = mesh.face_owner.data();
    const LocalIndex* CFD_RESTRICT face_neigh = mesh.face_neigh.data();

    const double* CFD_RESTRICT cx = mesh.cell_centroid_x.data();
    const double* CFD_RESTRICT cy = mesh.cell_centroid_y.data();
    const double* CFD_RESTRICT cz = mesh.cell_centroid_z.data();

    const double* CFD_RESTRICT fx = mesh.face_centroid_x.data();
    const double* CFD_RESTRICT fy = mesh.face_centroid_y.data();
    const double* CFD_RESTRICT fz = mesh.face_centroid_z.data();

    double* CFD_RESTRICT dx = face_cell_dist_x.data();
    double* CFD_RESTRICT dy = face_cell_dist_y.data();
    double* CFD_RESTRICT dz = face_cell_dist_z.data();

    for (std::size_t f = 0; f < n_faces; ++f) {
        const std::size_t u = static_cast<std::size_t>(face_owner[f]);
        if (f < n_inner_faces) {
            const std::size_t v = static_cast<std::size_t>(face_neigh[f]);
            dx[f] = cx[v] - cx[u];
            dy[f] = cy[v] - cy[u];
            dz[f] = cz[v] - cz[u];
        } else {
            // r_ghost = r_in + 2*(r_face - r_in) => dr = 2*(r_face - r_in) (!) can be replaced by orthogonal reflection
            dx[f] = 2.0 * (fx[f] - cx[u]);
            dy[f] = 2.0 * (fy[f] - cy[u]);
            dz[f] = 2.0 * (fz[f] - cz[u]);
        }
    }
}

void build_face_cell_dist(const MeshPart& mesh,
                          const std::vector<double>& face_cell_dist_x,
                          const std::vector<double>& face_cell_dist_y,
                          const std::vector<double>& face_cell_dist_z,
                          std::vector<double>& face_cell_dist) {
    const std::size_t n_faces = static_cast<std::size_t>(mesh.n_faces);
    face_cell_dist.resize(n_faces);

    const double* CFD_RESTRICT dx = face_cell_dist_x.data();
    const double* CFD_RESTRICT dy = face_cell_dist_y.data();
    const double* CFD_RESTRICT dz = face_cell_dist_z.data();
    double* CFD_RESTRICT d = face_cell_dist.data();

    for (std::size_t f = 0; f < n_faces; ++f) {
        d[f] = std::sqrt(dx[f] * dx[f] + dy[f] * dy[f] + dz[f] * dz[f]);
    }
}

void build_face_cell_dist(const MeshPart& mesh,
                          std::vector<double>& face_cell_dist) {
    const std::size_t n_faces = static_cast<std::size_t>(mesh.n_faces);
    const std::size_t n_inner_faces = static_cast<std::size_t>(mesh.n_inner_faces);
    face_cell_dist.resize(n_faces);

    double* CFD_RESTRICT d = face_cell_dist.data();

    const LocalIndex* CFD_RESTRICT face_owner = mesh.face_owner.data();
    const LocalIndex* CFD_RESTRICT face_neigh = mesh.face_neigh.data();

    const double* CFD_RESTRICT cx = mesh.cell_centroid_x.data();
    const double* CFD_RESTRICT cy = mesh.cell_centroid_y.data();
    const double* CFD_RESTRICT cz = mesh.cell_centroid_z.data();

    const double* CFD_RESTRICT fx = mesh.face_centroid_x.data();
    const double* CFD_RESTRICT fy = mesh.face_centroid_y.data();
    const double* CFD_RESTRICT fz = mesh.face_centroid_z.data();

    for (std::size_t f = 0; f < n_faces; ++f) {
        const std::size_t u = static_cast<std::size_t>(face_owner[f]);
        if (f < n_inner_faces) {
            const std::size_t v = static_cast<std::size_t>(face_neigh[f]);
            d[f] = std::sqrt((cx[v] - cx[u]) * (cx[v] - cx[u]) + 
                                (cy[v] - cy[u]) * (cy[v] - cy[u]) +
                                (cz[v] - cz[u]) * (cz[v] - cz[u])); 
        } else {
            // r_ghost = r_in + 2*(r_face - r_in) => dr = 2*(r_face - r_in) (!) can be replaced by orthogonal reflection
            d[f] = std::sqrt(4.0 * (fx[f] - cx[u]) * (fx[f] - cx[u]) + 
                                4.0 * (fy[f] - cy[u]) * (fy[f] - cy[u]) +
                                4.0 * (fz[f] - cz[u]) * (fz[f] - cz[u]));
        }
    }

}

void build_face_cell_dist_inv(const std::vector<double>& face_cell_dist,
                              std::vector<double>& face_cell_dist_inv) {
    const std::size_t n_faces = face_cell_dist.size();
    face_cell_dist_inv.resize(n_faces);

    const double* CFD_RESTRICT d = face_cell_dist.data();
    double* CFD_RESTRICT dinv = face_cell_dist_inv.data();

    for (std::size_t f = 0; f < n_faces; ++f) {
        dinv[f] = 1.0 / std::max(d[f], 1.0e-30);
    }
}

void build_face_cell_dist_inv(const MeshPart& mesh,
                              std::vector<double>& face_cell_dist_inv) {
    const std::size_t n_faces = static_cast<std::size_t>(mesh.n_faces);
    const std::size_t n_inner_faces = static_cast<std::size_t>(mesh.n_inner_faces);
    face_cell_dist_inv.resize(n_faces);

    double* CFD_RESTRICT d_inv = face_cell_dist_inv.data();

    const LocalIndex* CFD_RESTRICT face_owner = mesh.face_owner.data();
    const LocalIndex* CFD_RESTRICT face_neigh = mesh.face_neigh.data();

    const double* CFD_RESTRICT cx = mesh.cell_centroid_x.data();
    const double* CFD_RESTRICT cy = mesh.cell_centroid_y.data();
    const double* CFD_RESTRICT cz = mesh.cell_centroid_z.data();

    const double* CFD_RESTRICT fx = mesh.face_centroid_x.data();
    const double* CFD_RESTRICT fy = mesh.face_centroid_y.data();
    const double* CFD_RESTRICT fz = mesh.face_centroid_z.data();

    for (std::size_t f = 0; f < n_faces; ++f) {
        const std::size_t u = static_cast<std::size_t>(face_owner[f]);
        if (f < n_inner_faces) {
            const std::size_t v = static_cast<std::size_t>(face_neigh[f]);
            const double d = std::sqrt((cx[v] - cx[u]) * (cx[v] - cx[u]) + 
                                (cy[v] - cy[u]) * (cy[v] - cy[u]) +
                                (cz[v] - cz[u]) * (cz[v] - cz[u])); 
            d_inv[f] = 1.0 / d;
        } else {
            // r_ghost = r_in + 2*(r_face - r_in) => dr = 2*(r_face - r_in) (!) can be replaced by orthogonal reflection
            const double d = std::sqrt(4.0 * (fx[f] - cx[u]) * (fx[f] - cx[u]) + 
                                4.0 * (fy[f] - cy[u]) * (fy[f] - cy[u]) +
                                4.0 * (fz[f] - cz[u]) * (fz[f] - cz[u]));
            d_inv[f] = 1.0 / d;
        }
    }

}

void build_interpolation_weights(const MeshPart& mesh,
                                 std::vector<double>& face_interp_weight) {
    const auto n_faces       = static_cast<std::size_t>(mesh.n_faces);
    const auto n_inner_faces = static_cast<std::size_t>(mesh.n_inner_faces);
    face_interp_weight.resize(n_faces);

    const LocalIndex* CFD_RESTRICT face_owner = mesh.face_owner.data();
    const LocalIndex* CFD_RESTRICT face_neigh = mesh.face_neigh.data();

    const double* CFD_RESTRICT nx_ptr = mesh.face_normal_x.data();
    const double* CFD_RESTRICT ny_ptr = mesh.face_normal_y.data();
    const double* CFD_RESTRICT nz_ptr = mesh.face_normal_z.data();

    const double* CFD_RESTRICT cx = mesh.cell_centroid_x.data();
    const double* CFD_RESTRICT cy = mesh.cell_centroid_y.data();
    const double* CFD_RESTRICT cz = mesh.cell_centroid_z.data();

    const double* CFD_RESTRICT fx = mesh.face_centroid_x.data();
    const double* CFD_RESTRICT fy = mesh.face_centroid_y.data();
    const double* CFD_RESTRICT fz = mesh.face_centroid_z.data();

    double* CFD_RESTRICT w = face_interp_weight.data();


    for (std::size_t f = 0; f < n_inner_faces; ++f) {
        const auto u = static_cast<std::size_t>(face_owner[f]); // left
        const auto v = static_cast<std::size_t>(face_neigh[f]); // right

        const double nx = nx_ptr[f];
        const double ny = ny_ptr[f];
        const double nz = nz_ptr[f];

        // d_Cf = face_center - cell_center(left)
        const double dcf_x = fx[f] - cx[u];
        const double dcf_y = fy[f] - cy[u];
        const double dcf_z = fz[f] - cz[u];

        // d_fF = cell_center(right) - face_center
        const double dff_x = cx[v] - fx[f];
        const double dff_y = cy[v] - fy[f];
        const double dff_z = cz[v] - fz[f];


        const double Lf = dcf_x * nx + dcf_y * ny + dcf_z * nz;
        const double Rf = dff_x * nx + dff_y * ny + dff_z * nz;

        const double denom = Lf + Rf;

        // w = Rf / (Lf + Rf) 
        w[f] = (denom > 1.0e-30) ? (Rf / denom) : 0.5;
    }

    for (std::size_t f = n_inner_faces; f < n_faces; ++f) {
        w[f] = 0.5;
    }
}

void build_interpolation_weights(const MeshPart& mesh,
                                 const std::vector<double>& dx_vec,
                                 const std::vector<double>& dy_vec,
                                 const std::vector<double>& dz_vec,
                                 std::vector<double>& face_interp_weight) {
    const auto n_faces       = static_cast<std::size_t>(mesh.n_faces);
    const auto n_inner_faces = static_cast<std::size_t>(mesh.n_inner_faces);
    face_interp_weight.resize(n_faces);

    const LocalIndex* CFD_RESTRICT face_owner = mesh.face_owner.data();

    const double* CFD_RESTRICT nx_ptr = mesh.face_normal_x.data();
    const double* CFD_RESTRICT ny_ptr = mesh.face_normal_y.data();
    const double* CFD_RESTRICT nz_ptr = mesh.face_normal_z.data();

    const double* CFD_RESTRICT cx = mesh.cell_centroid_x.data();
    const double* CFD_RESTRICT cy = mesh.cell_centroid_y.data();
    const double* CFD_RESTRICT cz = mesh.cell_centroid_z.data();

    const double* CFD_RESTRICT fx = mesh.face_centroid_x.data();
    const double* CFD_RESTRICT fy = mesh.face_centroid_y.data();
    const double* CFD_RESTRICT fz = mesh.face_centroid_z.data();

    const double* CFD_RESTRICT dx = dx_vec.data();
    const double* CFD_RESTRICT dy = dy_vec.data();
    const double* CFD_RESTRICT dz = dz_vec.data();

    double* CFD_RESTRICT w = face_interp_weight.data();


    for (std::size_t f = 0; f < n_inner_faces; ++f) {
        const auto u = static_cast<std::size_t>(face_owner[f]);

        const double nx = nx_ptr[f];
        const double ny = ny_ptr[f];
        const double nz = nz_ptr[f];

        // Lf = (face_center - cell_center(left)) · n
        const double Lf = (fx[f] - cx[u]) * nx + 
                          (fy[f] - cy[u]) * ny + 
                          (fz[f] - cz[u]) * nz;

        const double denom = dx[f] * nx + dy[f] * ny + dz[f] * nz;

        // w = Rf / denom = (denom - Lf) / denom
        w[f] = (denom > 1.0e-30) ? ((denom - Lf) / denom) : 0.5;
    }

    for (std::size_t f = n_inner_faces; f < n_faces; ++f) {
        w[f] = 0.5;
    }
}

void build_non_ortho_vectors(const MeshPart& m,
                             const std::vector<double>& face_cell_dist_x,
                             const std::vector<double>& face_cell_dist_y,
                             const std::vector<double>& face_cell_dist_z,
                             std::vector<double>& face_non_ortho_x,
                             std::vector<double>& face_non_ortho_y,
                             std::vector<double>& face_non_ortho_z) {
    const auto n_faces = static_cast<std::size_t>(m.n_faces);
    face_non_ortho_x.resize(n_faces);
    face_non_ortho_y.resize(n_faces);
    face_non_ortho_z.resize(n_faces);

    const double* CFD_RESTRICT dx = face_cell_dist_x.data();
    const double* CFD_RESTRICT dy = face_cell_dist_y.data();
    const double* CFD_RESTRICT dz = face_cell_dist_z.data();

    const double* CFD_RESTRICT area = m.face_area.data();
    const double* CFD_RESTRICT nx   = m.face_normal_x.data();
    const double* CFD_RESTRICT ny   = m.face_normal_y.data();
    const double* CFD_RESTRICT nz   = m.face_normal_z.data();

    double* CFD_RESTRICT tx = face_non_ortho_x.data();
    double* CFD_RESTRICT ty = face_non_ortho_y.data();
    double* CFD_RESTRICT tz = face_non_ortho_z.data();

    // Over-relaxed decomposition: S_f = Delta + T, Delta = (S_f · S_f / (d · S_f)) * d
    for (std::size_t f = 0; f < n_faces; ++f) {
        const double a    = area[f];
        const double sf_x = a * nx[f];
        const double sf_y = a * ny[f];
        const double sf_z = a * nz[f];

        const double sf2     = sf_x * sf_x + sf_y * sf_y + sf_z * sf_z;
        const double d_dot_s = dx[f] * sf_x + dy[f] * sf_y + dz[f] * sf_z;

        const double alpha = sf2 / std::max(d_dot_s, 1.0e-30);

        tx[f] = sf_x - alpha * dx[f];
        ty[f] = sf_y - alpha * dy[f];
        tz[f] = sf_z - alpha * dz[f];
    }
}

void build_non_ortho_vectors(const MeshPart& m,
                             std::vector<double>& face_non_ortho_x,
                             std::vector<double>& face_non_ortho_y,
                             std::vector<double>& face_non_ortho_z) {
    const auto n_faces       = static_cast<std::size_t>(m.n_faces);
    const auto n_inner_faces = static_cast<std::size_t>(m.n_inner_faces);

    face_non_ortho_x.resize(n_faces);
    face_non_ortho_y.resize(n_faces);
    face_non_ortho_z.resize(n_faces);

    const LocalIndex* CFD_RESTRICT face_owner = m.face_owner.data();
    const LocalIndex* CFD_RESTRICT face_neigh = m.face_neigh.data();

    const double* CFD_RESTRICT cx = m.cell_centroid_x.data();
    const double* CFD_RESTRICT cy = m.cell_centroid_y.data();
    const double* CFD_RESTRICT cz = m.cell_centroid_z.data();

    const double* CFD_RESTRICT fx = m.face_centroid_x.data();
    const double* CFD_RESTRICT fy = m.face_centroid_y.data();
    const double* CFD_RESTRICT fz = m.face_centroid_z.data();

    const double* CFD_RESTRICT area = m.face_area.data();
    const double* CFD_RESTRICT nx   = m.face_normal_x.data();
    const double* CFD_RESTRICT ny   = m.face_normal_y.data();
    const double* CFD_RESTRICT nz   = m.face_normal_z.data();

    double* CFD_RESTRICT tx = face_non_ortho_x.data();
    double* CFD_RESTRICT ty = face_non_ortho_y.data();
    double* CFD_RESTRICT tz = face_non_ortho_z.data();

    for (std::size_t f = 0; f < n_inner_faces; ++f) {
        const auto u = static_cast<std::size_t>(face_owner[f]);
        const auto v = static_cast<std::size_t>(face_neigh[f]);

        const double dx = cx[v] - cx[u];
        const double dy = cy[v] - cy[u];
        const double dz = cz[v] - cz[u];

        const double a    = area[f];
        const double sf_x = a * nx[f];
        const double sf_y = a * ny[f];
        const double sf_z = a * nz[f];

        const double sf2     = sf_x * sf_x + sf_y * sf_y + sf_z * sf_z;
        const double d_dot_s = dx * sf_x + dy * sf_y + dz * sf_z;

        const double alpha = sf2 / std::max(d_dot_s, 1.0e-30);

        tx[f] = sf_x - alpha * dx;
        ty[f] = sf_y - alpha * dy;
        tz[f] = sf_z - alpha * dz;
    }

    for (std::size_t f = n_inner_faces; f < n_faces; ++f) {
        const auto u = static_cast<std::size_t>(face_owner[f]);

        const double dx = 2.0 * (fx[f] - cx[u]);
        const double dy = 2.0 * (fy[f] - cy[u]);
        const double dz = 2.0 * (fz[f] - cz[u]);

        const double a    = area[f];
        const double sf_x = a * nx[f];
        const double sf_y = a * ny[f];
        const double sf_z = a * nz[f];

        const double sf2     = sf_x * sf_x + sf_y * sf_y + sf_z * sf_z;
        const double d_dot_s = dx * sf_x + dy * sf_y + dz * sf_z;

        const double alpha = sf2 / std::max(d_dot_s, 1.0e-30);

        tx[f] = sf_x - alpha * dx;
        ty[f] = sf_y - alpha * dy;
        tz[f] = sf_z - alpha * dz;
    }
}

// =============================================================================
// MeshAuxGeometry Method Implementation
// =============================================================================

void MeshAuxGeometry::add_geometry(const MeshPart& mesh, AuxGeomType requested_types) {
    #ifndef NDEBUG
    const std::size_t n_faces = static_cast<std::size_t>(mesh.n_faces);
    if (active_mask != AuxGeomType::None) {
        if (!face_cell_dist.empty())       assert(face_cell_dist.size() == n_faces);
        if (!face_cell_dist_x.empty())     assert(face_cell_dist_x.size() == n_faces);
        if (!face_cell_dist_inv.empty())   assert(face_cell_dist_inv.size() == n_faces);
        if (!face_interp_weight.empty())   assert(face_interp_weight.size() == n_faces);
        if (!face_non_ortho_x.empty())     assert(face_non_ortho_x.size() == n_faces);
    }
    #endif


    if (has_flag(requested_types, AuxGeomType::FaceCellDistanceVector) &&
        !has_face_cell_dist_vector()) {
        build_face_cell_dist_vectors(mesh, face_cell_dist_x, face_cell_dist_y, face_cell_dist_z);
        active_mask = active_mask | AuxGeomType::FaceCellDistanceVector;
    }


    if (has_flag(requested_types, AuxGeomType::FaceCellDistance) &&
        !has_face_cell_dist()) {
        if (has_face_cell_dist_vector()) {

            build_face_cell_dist(mesh, face_cell_dist_x, face_cell_dist_y, face_cell_dist_z, face_cell_dist);
        } else {

            build_face_cell_dist(mesh, face_cell_dist);
        }
        active_mask = active_mask | AuxGeomType::FaceCellDistance;
    }


    if (has_flag(requested_types, AuxGeomType::FaceCellDistanceInv) &&
        !has_face_cell_dist_inv()) {
        if (has_face_cell_dist()) {

            build_face_cell_dist_inv(face_cell_dist, face_cell_dist_inv);
        } else {

            build_face_cell_dist_inv(mesh, face_cell_dist_inv);
        }
        active_mask = active_mask | AuxGeomType::FaceCellDistanceInv;
    }


    if (has_flag(requested_types, AuxGeomType::FaceInterpolationWeights) &&
        !has_face_interp_weights()) {
        if (has_face_cell_dist_vector()) {

            build_interpolation_weights(mesh, face_cell_dist_x, face_cell_dist_y, face_cell_dist_z, face_interp_weight);
        } else {

            build_interpolation_weights(mesh, face_interp_weight);
        }
        active_mask = active_mask | AuxGeomType::FaceInterpolationWeights;
    }


    if (has_flag(requested_types, AuxGeomType::FaceNonOrthogonalityVector) &&
        !has_face_non_ortho()) {
        if (has_face_cell_dist_vector()) {

            build_non_ortho_vectors(mesh, face_cell_dist_x, face_cell_dist_y, face_cell_dist_z,
                                    face_non_ortho_x, face_non_ortho_y, face_non_ortho_z);
        } else {

            build_non_ortho_vectors(mesh,
                                    face_non_ortho_x, face_non_ortho_y, face_non_ortho_z);
        }
        active_mask = active_mask | AuxGeomType::FaceNonOrthogonalityVector;
    }


    if (has_flag(requested_types, AuxGeomType::WallDistance) &&
        !has_wall_dist()) {

        active_mask = active_mask | AuxGeomType::WallDistance;
    }
}

// =============================================================================
// MeshAuxGeometry Add Wall Distance Method Implementation
// =============================================================================
namespace {

constexpr double kInf     = 1.0e30;
constexpr double kMinDist = 1.0e-12;

/**
 * @brief Non-blocking point-to-point halo exchange for scalar field.
 */
inline void exchange_halo_scalar(const MeshPart& mp,
                                 const MPI_Comm comm,
                                 std::vector<double>& send_buf,
                                 std::vector<double>& recv_buf,
                                 std::vector<MPI_Request>& requests,
                                 double* CFD_RESTRICT dist) {
    const std::size_t n_nb = mp.nb_ranks.size();
    if (n_nb == 0) return;

    std::size_t req_count = 0;

    // 1. Post ireceive
    for (std::size_t k = 0; k < n_nb; ++k) {
        const std::size_t beg = static_cast<std::size_t>(mp.recv_offsets[k]);
        const std::size_t end = static_cast<std::size_t>(mp.recv_offsets[k + 1]);
        const int count = static_cast<int>(end - beg);

        if (count > 0) {
            MPI_Irecv(recv_buf.data() + beg, count, MPI_DOUBLE,
                      mp.nb_ranks[k], 0 /*tag*/, comm, &requests[req_count++]);
        }
    }

    // 2. Pack data for sending
    const LocalIndex* CFD_RESTRICT send_idx = mp.send_owned_local.data();
    double* CFD_RESTRICT sbuf               = send_buf.data();
    const std::size_t total_send            = mp.send_owned_local.size();

    for (std::size_t i = 0; i < total_send; ++i) {
        sbuf[i] = dist[send_idx[i]];
    }

    // 3. Send data
    for (std::size_t k = 0; k < n_nb; ++k) {
        const std::size_t beg = static_cast<std::size_t>(mp.send_offsets[k]);
        const std::size_t end = static_cast<std::size_t>(mp.send_offsets[k + 1]);
        const int count = static_cast<int>(end - beg);

        if (count > 0) {
            MPI_Isend(send_buf.data() + beg, count, MPI_DOUBLE,
                      mp.nb_ranks[k], 0 /*tag*/, comm, &requests[req_count++]);
        }
    }

    // 4. Waitall
    int req_count_int = static_cast<int>(req_count);
    MPI_Waitall(req_count_int, requests.data(), MPI_STATUSES_IGNORE);

    // 5. Unpack received data
    const LocalIndex* CFD_RESTRICT recv_idx = mp.recv_ghost_local.data();
    const double* CFD_RESTRICT rbuf         = recv_buf.data();
    const std::size_t total_recv            = mp.recv_ghost_local.size();

    for (std::size_t i = 0; i < total_recv; ++i) {
        dist[recv_idx[i]] = rbuf[i];
    }
}

/**
 * @brief Unified Gauss-Seidel distance relaxer on any CSR cell adjacency graph.
 */
void relax_distance_graph(const MeshPart& mp,
                          const MPI_Comm comm,
                          const LocalIndex* CFD_RESTRICT off,
                          const LocalIndex* CFD_RESTRICT nb,
                          const double* CFD_RESTRICT edge_dist,
                          const std::size_t n_own,
                          const double rel_tol,
                          const int max_sweeps,
                          double* CFD_RESTRICT dist) {
    const std::size_t n_nb = mp.nb_ranks.size();

    std::vector<double> send_buf(mp.send_owned_local.size());
    std::vector<double> recv_buf(mp.recv_ghost_local.size());
    std::vector<MPI_Request> requests(2 * n_nb);

    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        exchange_halo_scalar(mp, comm, send_buf, recv_buf, requests, dist);

        double max_rel_change = 0.0;

        auto relax_cell = [&](const std::size_t c) {
            const double d0 = dist[c];
            double d_new = d0;

            const LocalIndex j_beg = off[c];
            const LocalIndex j_end = off[c + 1];

            for (LocalIndex j = j_beg; j < j_end; ++j) {
                const std::size_t js = static_cast<std::size_t>(j);
                const std::size_t n  = static_cast<std::size_t>(nb[js]);

                const double cand = dist[n] + edge_dist[js];
                d_new = (cand < d_new) ? cand : d_new;
            }

            if (d_new < d0) {
                const double ref = std::max(d0, kMinDist);
                const double rel = (d0 - d_new) / ref;
                max_rel_change = (rel > max_rel_change) ? rel : max_rel_change;
                dist[c] = d_new;
            }
        };

        for (std::size_t c = 0; c < n_own; ++c) {
            relax_cell(c);
        }

        for (std::size_t c = n_own; c-- > 0;) {
            relax_cell(c);
        }

        double global_change = max_rel_change;
        if (n_nb > 0) {
            MPI_Allreduce(&max_rel_change, &global_change, 1, MPI_DOUBLE, MPI_MAX, comm);
        }

        if (global_change <= rel_tol) {
            break;
        }
    }
}

} // anonymous namespace

void MeshAuxGeometry::add_wall_distance(const MeshPart& mesh,
                                        MeshAuxConnectivity& aux_conn,
                                        const MPI_Comm comm,
                                        const std::vector<bool>& wall_patch,
                                        const WallDistStencil stencil,
                                        const int max_sweeps,
                                        const double rel_tol) {
    if (has_wall_dist()) {
        return;  
    }

    const std::size_t n_inner = static_cast<std::size_t>(mesh.n_inner_faces);
    const std::size_t n_faces = static_cast<std::size_t>(mesh.n_faces);
    const std::size_t n_cells = static_cast<std::size_t>(mesh.n_cells);
    const std::size_t n_own   = static_cast<std::size_t>(mesh.n_own);

    cell_wall_dist.assign(n_cells, kInf);
    double* CFD_RESTRICT dist = cell_wall_dist.data();

    const LocalIndex* CFD_RESTRICT face_owner = mesh.face_owner.data();
    const LocalIndex* CFD_RESTRICT face_patch = mesh.face_patch.data();

    const double* CFD_RESTRICT fcx = mesh.face_centroid_x.data();
    const double* CFD_RESTRICT fcy = mesh.face_centroid_y.data();
    const double* CFD_RESTRICT fcz = mesh.face_centroid_z.data();

    const double* CFD_RESTRICT ccx = mesh.cell_centroid_x.data();
    const double* CFD_RESTRICT ccy = mesh.cell_centroid_y.data();
    const double* CFD_RESTRICT ccz = mesh.cell_centroid_z.data();

    for (std::size_t f = n_inner; f < n_faces; ++f) {
        const std::size_t p = static_cast<std::size_t>(face_patch[f]);
        if (p >= wall_patch.size() || !wall_patch[p]) {
            continue;
        }
        const std::size_t c0 = static_cast<std::size_t>(face_owner[f]);

        const double ex = fcx[f] - ccx[c0];
        const double ey = fcy[f] - ccy[c0];
        const double ez = fcz[f] - ccz[c0];
        const double d  = std::sqrt(ex * ex + ey * ey + ez * ez);

        dist[c0] = std::min(dist[c0], d);
    }

    const LocalIndex* CFD_RESTRICT off = nullptr;
    const LocalIndex* CFD_RESTRICT nb  = nullptr;
    std::size_t total_edges = 0;

    if (stencil == WallDistStencil::FaceNeighbors) {
        aux_conn.add_connectivity(mesh, AuxConnType::CellCellsByFace);
        off = aux_conn.cell_cells_face_offsets.data();
        nb  = aux_conn.cell_cells_face.data();
        total_edges = aux_conn.cell_cells_face.size();
    } else {
        aux_conn.add_connectivity(mesh, AuxConnType::CellCellsByNode);
        off = aux_conn.cell_cells_node_offsets.data();
        nb  = aux_conn.cell_cells_node.data();
        total_edges = aux_conn.cell_cells_node.size();
    }

    std::vector<double> edge_dist(total_edges);
    double* CFD_RESTRICT edist = edge_dist.data();

    for (std::size_t c = 0; c < n_own; ++c) {
        const double x0 = ccx[c];
        const double y0 = ccy[c];
        const double z0 = ccz[c];

        const LocalIndex j_beg = off[c];
        const LocalIndex j_end = off[c + 1];

        for (LocalIndex j = j_beg; j < j_end; ++j) {
            const auto js = static_cast<std::size_t>(j);
            const auto n  = static_cast<std::size_t>(nb[js]);

            const double dx = ccx[n] - x0;
            const double dy = ccy[n] - y0;
            const double dz = ccz[n] - z0;

            edist[js] = std::sqrt(dx * dx + dy * dy + dz * dz);
        }
    }

    relax_distance_graph(mesh, comm, off, nb, edist, n_own, rel_tol, max_sweeps, dist);

    active_mask = active_mask | AuxGeomType::WallDistance;
}

} // namespace cfd::mesh