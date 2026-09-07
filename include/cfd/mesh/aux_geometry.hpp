#pragma once

#include <cstdint>
#include <vector>

#include <mpi.h>

#include "cfd/mesh/localmesh.hpp"
#include "cfd/mesh/aux_connectivity.hpp"

namespace cfd::mesh {

enum class AuxGeomType : std::uint32_t {
    None                       = 0,
    FaceCellDistance           = 1 << 0, ///< |r_R - r_L|
    FaceCellDistanceVector     = 1 << 1, ///< (r_R - r_L)_x,y,z
    FaceCellDistanceInv        = 1 << 2, ///< 1.0 / |r_R - r_L|
    FaceInterpolationWeights   = 1 << 3, ///< Geometric CD-weights for face interpolation
    WallDistance               = 1 << 4, ///< Closest wall distance (y_dist) for turbulence
    FaceNonOrthogonalityVector = 1 << 5, ///< Non-orthogonal T_f vectors (S_f = Delta_f + T_f)
    All                        = 0xFFFFFFFF
};

[[nodiscard]] constexpr AuxGeomType operator|(AuxGeomType a, AuxGeomType b) noexcept {
    return static_cast<AuxGeomType>(static_cast<std::uint32_t>(a) | static_cast<std::uint32_t>(b));
}

[[nodiscard]] constexpr AuxGeomType operator&(AuxGeomType a, AuxGeomType b) noexcept {
    return static_cast<AuxGeomType>(static_cast<std::uint32_t>(a) & static_cast<std::uint32_t>(b));
}

[[nodiscard]] constexpr bool has_flag(AuxGeomType mask, AuxGeomType flag) noexcept {
    return (mask & flag) == flag;
}

enum class WallDistStencil : std::uint8_t {
    FaceNeighbors, ///< Fast: 4-6 neighbors/cell, ideal for aligned boundary layers
    NodeNeighbors  ///< Robust: 14-26 neighbors/cell, avoids corner metric distortion
};

/**
 * @struct MeshAuxGeometry
 * @brief Precomputed metric fields, allocated strictly on demand.
 */
struct MeshAuxGeometry {
    AuxGeomType active_mask{AuxGeomType::None};

    // -------------------------------------------------------------------------
    // 1. Face-Cell Distance Metrics (Size: n_faces)
    // For ghost cells cell centroid is known as general mesh geometry
    // For boundary ghost cells cell centroid is recomputed from ajecent real cell centroid
    // -------------------------------------------------------------------------
    // Distance between cell centroids sharing the face: |r_R - r_L|
    std::vector<double> face_cell_dist;                                             
    std::vector<double> face_cell_dist_x;        
    std::vector<double> face_cell_dist_y;    
    std::vector<double> face_cell_dist_z;    
    // Inverted distance for diffusion flux: 1.0 / |r_R - r_L|
    std::vector<double> face_cell_dist_inv;

    // -------------------------------------------------------------------------
    // 2. Linear Interpolation Weights (Size: n_faces)
    // For boundary faces the weight is equal 1/2 that consistent with recomputed
    // boundary ghost cells centroids
    // -------------------------------------------------------------------------
    // Weight w: phi_f = w * phi_owner + (1 - w) * phi_neigh
    std::vector<double> face_interp_weight;

    // -------------------------------------------------------------------------
    // 3. Non-Orthogonal Correction Vectors (Size: n_faces)
    // -------------------------------------------------------------------------
    // Decomposition: n * Area = Delta_LR + T
    std::vector<double> face_non_ortho_x;
    std::vector<double> face_non_ortho_y;
    std::vector<double> face_non_ortho_z;

    // -------------------------------------------------------------------------
    // 4. Wall Distance (Size: n_cells)
    // -------------------------------------------------------------------------
    std::vector<double> cell_wall_dist;

    // Unified entry point for metric allocations
    void add_geometry(const MeshPart& mesh, AuxGeomType requested_types);
    void add_wall_distance(const MeshPart& mesh,
                       MeshAuxConnectivity& aux_conn,
                       MPI_Comm comm,
                       const std::vector<bool>& wall_patch,
                       WallDistStencil stencil = WallDistStencil::FaceNeighbors,
                       int max_sweeps = 100,
                       double rel_tol = 1.0e-4);

    // Fast status query helpers
    [[nodiscard]] bool has(AuxGeomType flag) const noexcept {
        return has_flag(active_mask, flag);
    }
    [[nodiscard]] bool has_face_cell_dist() const noexcept {
        return has(AuxGeomType::FaceCellDistance);
    }
    [[nodiscard]] bool has_face_cell_dist_vector() const noexcept {
        return has(AuxGeomType::FaceCellDistanceVector);
    }
    [[nodiscard]] bool has_face_cell_dist_inv() const noexcept {
        return has(AuxGeomType::FaceCellDistanceInv);
    }
    [[nodiscard]] bool has_face_interp_weights() const noexcept {
        return has(AuxGeomType::FaceInterpolationWeights);
    }
    [[nodiscard]] bool has_face_non_ortho() const noexcept {
        return has(AuxGeomType::FaceNonOrthogonalityVector);
    }
    [[nodiscard]] bool has_wall_dist() const noexcept {
        return has(AuxGeomType::WallDistance);
    }
};

void build_face_cell_dist_vectors(const MeshPart& mesh,
                                  std::vector<double>& face_cell_dist_x,
                                  std::vector<double>& face_cell_dist_y,
                                  std::vector<double>& face_cell_dist_z);
void build_face_cell_dist(const MeshPart& mesh,
                          const std::vector<double>& face_cell_dist_x,
                          const std::vector<double>& face_cell_dist_y,
                          const std::vector<double>& face_cell_dist_z,
                          std::vector<double>& face_cell_dist);
void build_face_cell_dist(const MeshPart& mesh,
                          std::vector<double>& face_cell_dist);
void build_face_cell_dist_inv(const std::vector<double>& face_cell_dist,
                              std::vector<double>& face_cell_dist_inv);
void build_face_cell_dist_inv(const MeshPart& mesh,
                              std::vector<double>& face_cell_dist_inv);
void build_interpolation_weights(const MeshPart& mesh,
                                 std::vector<double>& face_interp_weight);
void build_interpolation_weights(const MeshPart& mesh,
                                 const std::vector<double>& dx_vec,
                                 const std::vector<double>& dy_vec,
                                 const std::vector<double>& dz_vec,
                                 std::vector<double>& face_interp_weight);
void build_non_ortho_vectors(const MeshPart& m,
                             const std::vector<double>& face_cell_dist_x,
                             const std::vector<double>& face_cell_dist_y,
                             const std::vector<double>& face_cell_dist_z,
                             std::vector<double>& face_non_ortho_x,
                             std::vector<double>& face_non_ortho_y,
                             std::vector<double>& face_non_ortho_z);
void build_non_ortho_vectors(const MeshPart& m,
                             std::vector<double>& face_non_ortho_x,
                             std::vector<double>& face_non_ortho_y,
                             std::vector<double>& face_non_ortho_z);

void add_wall_distance(const MeshPart& mesh,
                       MeshAuxConnectivity& aux_conn,
                       MPI_Comm comm,
                       const std::vector<bool>& wall_patch,
                       WallDistStencil stencil = WallDistStencil::FaceNeighbors,
                       int max_sweeps = 100,
                       double rel_tol = 1.0e-4);




} // namespace cfd::mesh