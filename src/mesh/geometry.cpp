#include "cfd/mesh/geometry.hpp"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <limits>
#include <sstream>

#include "cfd/mpi/log.hpp"
#include "cfd/mesh/cgnstables.hpp"

namespace cfd::mesh {

namespace {
struct CellMetrics {
    double volume{0.0};
    double cx{0.0};
    double cy{0.0};
    double cz{0.0};
};

[[nodiscard]] CellMetrics compute_poly_cell_metrics(CellType t,
                                                    const double* x,
                                                    const double* y,
                                                    const double* z) noexcept {
    const std::size_t ti = static_cast<std::size_t>(t);
    const std::size_t nnodes = static_cast<std::size_t>(kNodesPerType[ti]);
    const std::size_t num_faces = static_cast<std::size_t>(kFacesPerType[ti]);

    // finding geometric center
    double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
    for (std::size_t i = 0; i < nnodes; ++i) {
        sum_x += x[i];
        sum_y += y[i];
        sum_z += z[i];
    }
    const double inv_nnodes = 1.0 / static_cast<double>(nnodes);
    const double x0 = sum_x * inv_nnodes;
    const double y0 = sum_y * inv_nnodes;
    const double z0 = sum_z * inv_nnodes;

    double total_vol = 0.0;
    double mx = 0.0, my = 0.0, mz = 0.0;

    // loop over all cell faces
    for (std::size_t f = 0; f < num_faces; ++f) {
        const std::size_t nn = static_cast<std::size_t>(kFaceNodes[ti][f]);

        if (nn == 3) {
            // if face is triangel => centroid == geometric center:
            const std::size_t i0 = static_cast<std::size_t>(kFaceTable[ti][f][0]);
            const std::size_t i1 = static_cast<std::size_t>(kFaceTable[ti][f][1]);
            const std::size_t i2 = static_cast<std::size_t>(kFaceTable[ti][f][2]);

            const double p0x = x[i0], p0y = y[i0], p0z = z[i0];
            const double p1x = x[i1], p1y = y[i1], p1z = z[i1];
            const double p2x = x[i2], p2y = y[i2], p2z = z[i2];

            // triangle area vector: 0.5 * (p1 - p0) x (p2 - p0)
            const double e1x = p1x - p0x, e1y = p1y - p0y, e1z = p1z - p0z;
            const double e2x = p2x - p0x, e2y = p2y - p0y, e2z = p2z - p0z;

            const double Sx = 0.5 * (e1y * e2z - e1z * e2y);
            const double Sy = 0.5 * (e1z * e2x - e1x * e2z);
            const double Sz = 0.5 * (e1x * e2y - e1y * e2x);

            // tetrahedron singed volume (always positive, because S is outward vector): (1/3) * S · (p0 - x0)
            const double v_tet = (1.0 / 3.0) * (Sx * (p0x - x0) + 
                                                Sy * (p0y - y0) + 
                                                Sz * (p0z - z0));

            // tetrahedron centorid == geometric center
            const double c_tet_x = 0.25 * (x0 + p0x + p1x + p2x);
            const double c_tet_y = 0.25 * (y0 + p0y + p1y + p2y);
            const double c_tet_z = 0.25 * (z0 + p0z + p1z + p2z);

            total_vol += v_tet;
            mx += v_tet * c_tet_x;
            my += v_tet * c_tet_y;
            mz += v_tet * c_tet_z;
        } else {
            // if face is general polyhedron => triangulation:
            double f_sum_x = 0.0, f_sum_y = 0.0, f_sum_z = 0.0;
            
            // finding face geometric center (for triangulation)
            for (std::size_t j = 0; j < nn; ++j) {
                const std::size_t idx = static_cast<std::size_t>(kFaceTable[ti][f][j]);
                f_sum_x += x[idx];
                f_sum_y += y[idx];
                f_sum_z += z[idx];
            }
            const double inv_nn = 1.0 / static_cast<double>(nn);
            const double fc_x = f_sum_x * inv_nn;
            const double fc_y = f_sum_y * inv_nn;
            const double fc_z = f_sum_z * inv_nn;

            // loop over all face nodes
            for (std::size_t j = 0; j < nn; ++j) {
                const std::size_t next_j = (j + 1 == nn) ? 0 : (j + 1);

                const std::size_t idx_a = static_cast<std::size_t>(kFaceTable[ti][f][j]);
                const std::size_t idx_b = static_cast<std::size_t>(kFaceTable[ti][f][next_j]);

                const double ax = x[idx_a], ay = y[idx_a], az = z[idx_a];
                const double bx = x[idx_b], by = y[idx_b], bz = z[idx_b];

                // Subtriangel area (fc, a, b): 0.5 * (a - fc) x (b - fc)
                const double e1x = ax - fc_x, e1y = ay - fc_y, e1z = az - fc_z;
                const double e2x = bx - fc_x, e2y = by - fc_y, e2z = bz - fc_z;

                const double tri_Sx = 0.5 * (e1y * e2z - e1z * e2y);
                const double tri_Sy = 0.5 * (e1z * e2x - e1x * e2z);
                const double tri_Sz = 0.5 * (e1x * e2y - e1y * e2x);

                // Subtetrahedron volume (x0, fc, a, b)
                const double v_tet = (1.0 / 3.0) * (tri_Sx * (fc_x - x0) + 
                                                    tri_Sy * (fc_y - y0) + 
                                                    tri_Sz * (fc_z - z0));

                const double c_tet_x = 0.25 * (x0 + fc_x + ax + bx);
                const double c_tet_y = 0.25 * (y0 + fc_y + ay + by);
                const double c_tet_z = 0.25 * (z0 + fc_z + az + bz);

                total_vol += v_tet;
                mx += v_tet * c_tet_x;
                my += v_tet * c_tet_y;
                mz += v_tet * c_tet_z;
            }
        }
    }

    if (total_vol > 1e-15) {
        const double inv_v = 1.0 / total_vol;
        return {total_vol, mx * inv_v, my * inv_v, mz * inv_v};
    }

    return {total_vol, x0, y0, z0};
}

[[nodiscard]] inline double poly_cell_volume(CellType t,
                                             const double* x,
                                             const double* y,
                                             const double* z) noexcept {
    return compute_poly_cell_metrics(t, x, y, z).volume;
}

}



bool validate_face_tables() {
    // idial reference cells ("unit cells")
    struct Ref {
        CellType t;
        std::vector<double> x;
        std::vector<double> y;
        std::vector<double> z;
        double vol;
        const char* name;
    };

    const std::vector<Ref> refs = {
        {
            CellType::TET,
            {0.0, 1.0, 0.0, 0.0},
            {0.0, 0.0, 1.0, 0.0},
            {0.0, 0.0, 0.0, 1.0},
            1.0 / 6.0,
            "TET"
        },
        {
            CellType::PYRA,
            {0.0, 1.0, 1.0, 0.0, 0.5},
            {0.0, 0.0, 1.0, 1.0, 0.5},
            {0.0, 0.0, 0.0, 0.0, 1.0},
            1.0 / 3.0,
            "PYRA"
        },
        {
            CellType::PRISM,
            {0.0, 1.0, 0.0, 0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0, 0.0, 0.0, 1.0},
            {0.0, 0.0, 0.0, 1.0, 1.0, 1.0},
            0.5,
            "PRISM"
        },
        {
            CellType::HEXA,
            {0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0},
            {0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0},
            {0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0},
            1.0,
            "HEXA"
        },
    };

    bool ok = true;

    // loop over reference cell types
    for (const auto& r : refs) {
        const auto ti  = static_cast<std::size_t>(r.t);
        const auto npt = static_cast<std::size_t>(kNodesPerType[ti]);

        // Cell centroid computation
        double cc[3] = {0.0, 0.0, 0.0};

        // loop over cell nodes for centroid
        for (std::size_t i = 0; i < npt; ++i) {
            cc[0] += r.x[i];
            cc[1] += r.y[i];
            cc[2] += r.z[i];
        } // end loop over cell nodes for centroid
        
        const double inv_npt = 1.0 / static_cast<double>(npt);
        
        // loop over spatial dimensions for centroid normalization
        for (std::size_t d = 0; d < 3; ++d) {
            cc[d] *= inv_npt;
        } // end loop over spatial dimensions for centroid normalization

        // Volume check
        const double v = poly_cell_volume(r.t, r.x.data(), r.y.data(), r.z.data());
        if (std::abs(v - r.vol) > 1e-12 || v <= 0.0) {
            std::fprintf(stderr, "TABLES: %s volume %.12e != %.12e\n", r.name, v, r.vol);
            ok = false;
        }

        const auto num_faces = static_cast<std::size_t>(kFacesPerType[ti]);

        // loop over cell faces
        for (std::size_t f = 0; f < num_faces; ++f) {
            const auto nn = static_cast<std::size_t>(kFaceNodes[ti][f]);

            double S[3] = {0.0, 0.0, 0.0};
            double cf[3] = {0.0, 0.0, 0.0};

            // loop over face nodes
            for (std::size_t j = 0; j < nn; ++j) {
                const std::size_t next_j = (j + 1 == nn) ? 0 : (j + 1);

                const auto idx_a = static_cast<std::size_t>(kFaceTable[ti][f][j]);
                const auto idx_b = static_cast<std::size_t>(kFaceTable[ti][f][next_j]);

                const double ax = r.x[idx_a], ay = r.y[idx_a], az = r.z[idx_a];
                const double bx = r.x[idx_b], by = r.y[idx_b], bz = r.z[idx_b];

                // Cross product: a x b
                S[0] += ay * bz - az * by;
                S[1] += az * bx - ax * bz;
                S[2] += ax * by - ay * bx;

                cf[0] += ax;
                cf[1] += ay;
                cf[2] += az;
            } // end loop over face nodes

            const double inv_nn = 1.0 / static_cast<double>(nn);
            // loop over spatial dimensions for face center normalization
            for (std::size_t d = 0; d < 3; ++d) {
                cf[d] *= inv_nn;
            } // end loop over spatial dimensions for face center normalization

            // Dot product between outer normal S and vector (cf - cc)
            const double dot = S[0] * (cf[0] - cc[0]) + 
                               S[1] * (cf[1] - cc[1]) + 
                               S[2] * (cf[2] - cc[2]);

            if (dot <= 0.0) {
                std::fprintf(stderr, "TABLES: %s face %zu inward normal (dot %.3e)\n", 
                             r.name, f, dot);
                ok = false;
            }
        } // end loop over cell faces

        // Permutation volume sign inversion check
        std::vector<double> flipped_x(npt);
        std::vector<double> flipped_y(npt);
        std::vector<double> flipped_z(npt);

        // loop over cell nodes for orientation flip
        for (std::size_t i = 0; i < npt; ++i) {
            const auto src_idx = static_cast<std::size_t>(kOrientationFlip[ti][i]);
            flipped_x[i] = r.x[src_idx];
            flipped_y[i] = r.y[src_idx];
            flipped_z[i] = r.z[src_idx];
        } // end loop over cell nodes for orientation flip

        const double vf = poly_cell_volume(r.t, flipped_x.data(), flipped_y.data(), flipped_z.data());
        if (vf >= 0.0) {
            std::fprintf(stderr, "TABLES: %s orientation flip did not invert volume sign (vol = %.6e)\n", 
                         r.name, vf);
            ok = false;
        }
    } // end loop over reference cell types

    return ok;
}

void compute_mesh_geometry(MeshPart& mp) {
    const std::size_t n_cells_sz = static_cast<std::size_t>(mp.n_cells);
    const std::size_t n_faces_sz = static_cast<std::size_t>(mp.n_faces);

    // -------------------------------------------------------------------------
    // Step 1: Pre-allocate SoA Geometric Arrays
    // -------------------------------------------------------------------------
    mp.cell_centroid_x.resize(n_cells_sz);
    mp.cell_centroid_y.resize(n_cells_sz);
    mp.cell_centroid_z.resize(n_cells_sz);
    mp.cell_volume.resize(n_cells_sz);

    mp.face_centroid_x.resize(n_faces_sz);
    mp.face_centroid_y.resize(n_faces_sz);
    mp.face_centroid_z.resize(n_faces_sz);
    mp.face_normal_x.resize(n_faces_sz);
    mp.face_normal_y.resize(n_faces_sz);
    mp.face_normal_z.resize(n_faces_sz);
    mp.face_area.resize(n_faces_sz);

    // -------------------------------------------------------------------------
    // Step 2: Compute Cell Metrics via Tetrahedralization
    // -------------------------------------------------------------------------
    double min_local_vol = std::numeric_limits<double>::max();
    double max_local_vol = -std::numeric_limits<double>::max();
    double total_local_vol = 0.0;

    double cell_x_buf[8];
    double cell_y_buf[8];
    double cell_z_buf[8];

    for (LocalIndex c = 0; c < mp.n_cells; ++c) {
        const std::size_t c_sz = static_cast<std::size_t>(c);
        const LocalIndex off_start = mp.cell_nodes_offsets[c_sz];
        const LocalIndex off_end   = mp.cell_nodes_offsets[c_sz + 1];
        const std::size_t nnodes = static_cast<std::size_t>(off_end - off_start);
        const CellType type = mp.cell_type[c_sz];

        for (std::size_t k = 0; k < nnodes; ++k) {
            const LocalIndex nid = mp.cell_nodes[static_cast<std::size_t>(off_start) + k];
            const std::size_t nid_sz = static_cast<std::size_t>(nid);

            cell_x_buf[k] = mp.node_x[nid_sz];
            cell_y_buf[k] = mp.node_y[nid_sz];
            cell_z_buf[k] = mp.node_z[nid_sz];
        }

        const CellMetrics metrics = compute_poly_cell_metrics(type, cell_x_buf, cell_y_buf, cell_z_buf);

        // Strict positive volume assertion
        if (metrics.volume <= 1e-15) {
            std::stringstream ss;
            ss << "Degenerate/negative cell volume detected on Rank " << mp.rank
               << " (Local cell: " << c << ", Global GID: " << mp.cell_gid[c_sz]
               << ", Type: " << cell_type_name(type) << ", Volume: " << metrics.volume << ")";
            mpi::fatal(MPI_COMM_WORLD, ss.str());
        }

        mp.cell_centroid_x[c_sz] = metrics.cx;
        mp.cell_centroid_y[c_sz] = metrics.cy;
        mp.cell_centroid_z[c_sz] = metrics.cz;
        mp.cell_volume[c_sz]     = metrics.volume;

        if (c < mp.n_own) {
            min_local_vol = std::min(min_local_vol, metrics.volume);
            max_local_vol = std::max(max_local_vol, metrics.volume);
            total_local_vol += metrics.volume;
        }
    }

    // -------------------------------------------------------------------------
    // Step 3: Compute Face Metrics via Proper Triangulation
    // -------------------------------------------------------------------------
    double min_local_area = std::numeric_limits<double>::max();
    double max_local_area = -std::numeric_limits<double>::max();

    for (LocalIndex f = 0; f < mp.n_faces; ++f) {
        const std::size_t f_sz = static_cast<std::size_t>(f);
        const LocalIndex off_start = mp.face_nodes_offsets[f_sz];
        const LocalIndex off_end   = mp.face_nodes_offsets[f_sz + 1];
        const std::size_t nnodes = static_cast<std::size_t>(off_end - off_start);

        if (nnodes < 3) {
            std::stringstream ss;
            ss << "Face " << f << " has degenerate node count: " << static_cast<int>(nnodes);
            mpi::fatal(MPI_COMM_WORLD, ss.str());
        }

        double Sx = 0.0, Sy = 0.0, Sz = 0.0;
        double fc_x = 0.0, fc_y = 0.0, fc_z = 0.0;

        if (nnodes == 3) {
            const std::size_t n0 = static_cast<std::size_t>(mp.face_nodes[static_cast<std::size_t>(off_start)]);
            const std::size_t n1 = static_cast<std::size_t>(mp.face_nodes[static_cast<std::size_t>(off_start) + 1]);
            const std::size_t n2 = static_cast<std::size_t>(mp.face_nodes[static_cast<std::size_t>(off_start) + 2]);

            const double p0x = mp.node_x[n0], p0y = mp.node_y[n0], p0z = mp.node_z[n0];
            const double p1x = mp.node_x[n1], p1y = mp.node_y[n1], p1z = mp.node_z[n1];
            const double p2x = mp.node_x[n2], p2y = mp.node_y[n2], p2z = mp.node_z[n2];

            const double e1x = p1x - p0x, e1y = p1y - p0y, e1z = p1z - p0z;
            const double e2x = p2x - p0x, e2y = p2y - p0y, e2z = p2z - p0z;

            Sx = 0.5 * (e1y * e2z - e1z * e2y);
            Sy = 0.5 * (e1z * e2x - e1x * e2z);
            Sz = 0.5 * (e1x * e2y - e1y * e2x);

            fc_x = (p0x + p1x + p2x) / 3.0;
            fc_y = (p0y + p1y + p2y) / 3.0;
            fc_z = (p0z + p1z + p2z) / 3.0;
        } else { 
            double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
            for (std::size_t k = 0; k < nnodes; ++k) {
                const std::size_t nid = static_cast<std::size_t>(mp.face_nodes[static_cast<std::size_t>(off_start) + k]);
                sum_x += mp.node_x[nid];
                sum_y += mp.node_y[nid];
                sum_z += mp.node_z[nid];
            }
            const double inv_nn = 1.0 / static_cast<double>(nnodes);
            const double f0_x = sum_x * inv_nn;
            const double f0_y = sum_y * inv_nn;
            const double f0_z = sum_z * inv_nn;

            double weighted_cx = 0.0, weighted_cy = 0.0, weighted_cz = 0.0;
            double total_sub_area = 0.0;

            for (std::size_t k = 0; k < nnodes; ++k) {
                const std::size_t next_k = (k + 1 == nnodes) ? 0 : (k + 1);
                const std::size_t n_curr = static_cast<std::size_t>(mp.face_nodes[static_cast<std::size_t>(off_start) + k]);
                const std::size_t n_next = static_cast<std::size_t>(mp.face_nodes[static_cast<std::size_t>(off_start) + next_k]);

                const double px = mp.node_x[n_curr], py = mp.node_y[n_curr], pz = mp.node_z[n_curr];
                const double qx = mp.node_x[n_next], qy = mp.node_y[n_next], qz = mp.node_z[n_next];

                const double e1x = px - f0_x, e1y = py - f0_y, e1z = pz - f0_z;
                const double e2x = qx - f0_x, e2y = qy - f0_y, e2z = qz - f0_z;

                const double tri_sx = 0.5 * (e1y * e2z - e1z * e2y);
                const double tri_sy = 0.5 * (e1z * e2x - e1x * e2z);
                const double tri_sz = 0.5 * (e1x * e2y - e1y * e2x);

                const double tri_area = std::sqrt(tri_sx * tri_sx + tri_sy * tri_sy + tri_sz * tri_sz);

                Sx += tri_sx;
                Sy += tri_sy;
                Sz += tri_sz; 

                const double tri_cx = (f0_x + px + qx) / 3.0;
                const double tri_cy = (f0_y + py + qy) / 3.0;
                const double tri_cz = (f0_z + pz + qz) / 3.0;

                weighted_cx += tri_area * tri_cx;
                weighted_cy += tri_area * tri_cy;
                weighted_cz += tri_area * tri_cz;
                total_sub_area += tri_area;
            }

            if (total_sub_area > 1e-15) {
                const double inv_sub_area = 1.0 / total_sub_area;
                fc_x = weighted_cx * inv_sub_area;
                fc_y = weighted_cy * inv_sub_area;
                fc_z = weighted_cz * inv_sub_area;
            } else {
                fc_x = f0_x;
                fc_y = f0_y;
                fc_z = f0_z;
            }
        }

        const double area = std::sqrt(Sx * Sx + Sy * Sy + Sz * Sz);

        if (area <= 1e-15) {
            std::stringstream ss;
            ss << "Degenerate zero-area face detected on Rank " << mp.rank
               << " (Face index: " << f << ", Area: " << area << ")";
            mpi::fatal(MPI_COMM_WORLD, ss.str());
        }

        const double inv_area = 1.0 / area;
        const double nx = Sx * inv_area;
        const double ny = Sy * inv_area;
        const double nz = Sz * inv_area;

        mp.face_centroid_x[f_sz] = fc_x;
        mp.face_centroid_y[f_sz] = fc_y;
        mp.face_centroid_z[f_sz] = fc_z;

        mp.face_area[f_sz]     = area;
        mp.face_normal_x[f_sz] = nx;
        mp.face_normal_y[f_sz] = ny;
        mp.face_normal_z[f_sz] = nz;

        min_local_area = std::min(min_local_area, area);
        max_local_area = std::max(max_local_area, area);

        // ---------------------------------------------------------------------
        // Step 4: Verification of Outward Normal Alignment
        // Normal must point from face_owner outward towards face_neigh.
        // ---------------------------------------------------------------------
        const std::size_t owner_sz = static_cast<std::size_t>(mp.face_owner[f_sz]);
        const double oc_x = mp.cell_centroid_x[owner_sz];
        const double oc_y = mp.cell_centroid_y[owner_sz];
        const double oc_z = mp.cell_centroid_z[owner_sz];

        
        const double d_vec_x = fc_x - oc_x;
        const double d_vec_y = fc_y - oc_y;
        const double d_vec_z = fc_z - oc_z;

        const double dot = nx * d_vec_x + ny * d_vec_y + nz * d_vec_z;
        if (dot <= 0.0) {
            std::stringstream ss;
            ss << "Normal orientation mismatch on Rank " << mp.rank
               << " for Face " << f << " (Owner cell: " << mp.face_owner[f_sz]
               << ", normal dot (fc - cc) = " << dot << " <= 0)";
            mpi::fatal(MPI_COMM_WORLD, ss.str());
        }
    }

    // -------------------------------------------------------------------------
    // Step 5: Global Mesh Quality & Consistency Logging
    // -------------------------------------------------------------------------
    double min_glob_vol = 0.0, max_glob_vol = 0.0, total_glob_vol = 0.0;
    double min_glob_area = 0.0, max_glob_area = 0.0;

    MPI_Allreduce(&min_local_vol, &min_glob_vol, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&max_local_vol, &max_glob_vol, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&total_local_vol, &total_glob_vol, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(&min_local_area, &min_glob_area, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&max_local_area, &max_glob_area, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (mp.rank == 0) {
        mpi::log_stat("INFO[Geometry computation]: Geometry verification passed successfully.");
        mpi::log_stat("      Total Domain Volume = %.8e", total_glob_vol);
        mpi::log_stat("      Cell Volumes : min = %.6e, max = %.6e", min_glob_vol, max_glob_vol);
        mpi::log_stat("      Face Areas   : min = %.6e, max = %.6e", min_glob_area, max_glob_area);
    }
}

} //namespace cfd::mesh