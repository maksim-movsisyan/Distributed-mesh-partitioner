#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <mpi.h>
#include "cfd/mesh_generator/config.hpp"

namespace cfd::mesh_generator {

enum class FeatureType : std::uint8_t {
    VERTEX = 0,
    EDGE   = 1,
    FACE   = 2,
    VOLUME = 3
};

// Canonical edge between two macro-vertices (u < v)
struct CanonicalEdge {
    std::size_t v0{0};
    std::size_t v1{0};
    std::size_t num_cells{0};
    GradingType grading_type{GradingType::UNIFORM};
    double grading{1.0};
    std::uint64_t global_start_id{0}; // 1-based start ID for interior nodes
    std::vector<double> distribution; // size: num_cells + 1
};

// Canonical face between 4 macro-vertices (w0 is min, w1 < w3)
struct CanonicalFace {
    std::array<std::size_t, 4> vertices{};
    std::size_t cells_u{0};
    std::size_t cells_v{0};
    GradingType grading_type_u{GradingType::UNIFORM};
    GradingType grading_type_v{GradingType::UNIFORM};
    double grading_u{1.0};
    double grading_v{1.0};
    std::uint64_t global_start_id{0}; // 1-based start ID for interior nodes
    std::vector<double> dist_u;
    std::vector<double> dist_v;
};

// Volume interior information for each block
struct BlockVolume {
    std::uint64_t global_start_id{0};  // 1-based
    std::size_t count{0};              // (Nx - 1) * (Ny - 1) * (Nz - 1)
    std::vector<double> dist_x;
    std::vector<double> dist_y;
    std::vector<double> dist_z;
};

// Range descriptor for O(log N) coordinate lookup from global NodeID
struct NodeRangeLookup {
    std::uint64_t start_id{0};
    std::uint64_t end_id{0}; // inclusive
    FeatureType type{FeatureType::VERTEX};
    std::size_t feature_idx{0};
};

class MacroTopology {
public:
    // Builds and validates the global multi-block topology without communication
    void build(const GeneratorConfig& config, MPI_Comm comm);

    // Returns the 1-based global NodeID for a grid point (i, j, k) in a given block
    [[nodiscard]] std::uint64_t get_node_id(std::size_t block_idx,
                                           std::size_t i,
                                           std::size_t j,
                                           std::size_t k) const noexcept;

    // Evaluates the exact physical Cartesian coordinates for any global 1-based NodeID
    [[nodiscard]] Vec3 evaluate_node_coord(std::uint64_t global_node_id,
                                          const GeneratorConfig& config) const noexcept;

    // Returns the 8 global 1-based NodeIDs for a global cell (in HEXA_8 order)
    [[nodiscard]] std::array<std::uint64_t, 8> get_cell_nodes(std::uint64_t global_cell_id) const noexcept;

    // Maps a global cell index [0, total_cells - 1] to (block_idx, i, j, k)
    void get_cell_block_and_ijk(std::uint64_t global_cell_id,
                                std::size_t& block_idx,
                                std::size_t& i,
                                std::size_t& j,
                                std::size_t& k) const noexcept;

    [[nodiscard]] std::uint64_t total_nodes() const noexcept { return total_nodes_; }
    [[nodiscard]] std::uint64_t total_cells() const noexcept { return total_cells_; }
    [[nodiscard]] std::size_t num_edges() const noexcept { return edges_.size(); }
    [[nodiscard]] std::size_t num_faces() const noexcept { return faces_.size(); }

private:
    std::uint64_t total_nodes_{0};
    std::uint64_t total_cells_{0};

    std::vector<CanonicalEdge> edges_;
    std::vector<CanonicalFace> faces_;
    std::vector<BlockVolume> volumes_;

    // Fast mapping: block -> 12 edges and 6 faces with orientation flags
    struct BlockLocalTopology {
        // [edge_idx, is_reversed]
        std::array<std::pair<std::size_t, bool>, 12> edges;
        // [face_idx, rotation_k, is_flipped]
        std::array<std::tuple<std::size_t, std::uint8_t, bool>, 6> faces;
    };
    std::vector<BlockLocalTopology> block_topos_;

    // Prefix sums of cells across blocks for fast global cell decomposition
    std::vector<std::uint64_t> cell_offsets_;

    // Lookup ranges for coordinate generation
    std::vector<NodeRangeLookup> lookups_;

    // Helpers
    std::size_t find_or_add_edge(std::size_t u, std::size_t v, std::size_t cells,
                                 GradingType gt, double g, const GeneratorConfig& cfg, MPI_Comm comm);
    std::size_t find_or_add_face(const std::array<std::size_t, 4>& quad_verts,
                                 std::size_t cu, std::size_t cv,
                                 GradingType gtu, GradingType gtv,
                                 double gu, double gv,
                                 const GeneratorConfig& cfg, MPI_Comm comm);
};

} // namespace cfd::mesh_generator