#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <mpi.h>
#include "cfd/mesh_generator/config.hpp"

namespace cfd::mesh_generator {

struct NodeSource {
    std::size_t block_idx{0};
    std::size_t i{0};
    std::size_t j{0};
    std::size_t k{0};
};

struct MacroEdge {
    std::size_t v0{0};
    std::size_t v1{0};
    std::size_t num_cells{0};
    std::uint64_t global_start_id{0};
};

struct MacroFaceDesc {
    std::array<std::size_t, 4> vertices{};
    std::size_t cells_u{0};
    std::size_t cells_v{0};
    std::uint64_t global_start_id{0};
};

class MacroTopology {
public:
    void build(const GeneratorConfig& config, MPI_Comm comm);

    [[nodiscard]] std::uint64_t get_node_id(std::size_t block_idx,
                                           std::size_t i,
                                           std::size_t j,
                                           std::size_t k) const noexcept;

    [[nodiscard]] Vec3 evaluate_node_coord(std::uint64_t global_node_id,
                                          const GeneratorConfig& config) const noexcept;

    [[nodiscard]] std::array<std::uint64_t, 8> get_cell_nodes(std::uint64_t global_cell_id) const noexcept;

    [[nodiscard]] std::array<std::uint64_t, 8> get_block_cell_nodes(std::size_t block_idx,
                                                                   std::size_t i,
                                                                   std::size_t j,
                                                                   std::size_t k) const noexcept;

    void get_cell_block_and_ijk(std::uint64_t global_cell_id,
                                std::size_t& block_idx,
                                std::size_t& i,
                                std::size_t& j,
                                std::size_t& k) const noexcept;

    [[nodiscard]] std::uint64_t total_nodes() const noexcept { return total_nodes_; }
    [[nodiscard]] std::uint64_t total_cells() const noexcept { return total_cells_; }

private:
    std::uint64_t total_nodes_{0};
    std::uint64_t total_cells_{0};

    std::vector<MacroEdge> edges_;
    std::vector<MacroFaceDesc> faces_;

    struct BlockTopo {
        std::array<std::size_t, 8> vertices{};
        std::array<std::pair<std::size_t, bool>, 12> edges;
        std::array<std::size_t, 6> faces;
        std::uint64_t vol_start_id{0};
        std::size_t vol_count{0};
        std::vector<double> dist_x;
        std::vector<double> dist_y;
        std::vector<double> dist_z;
    };
    std::vector<BlockTopo> blocks_;
    std::vector<std::uint64_t> cell_offsets_;
    std::vector<NodeSource> node_sources_;

    std::size_t find_or_add_edge(std::size_t u, std::size_t v, std::size_t cells, MPI_Comm comm);
    std::size_t find_or_add_face(const std::array<std::size_t, 4>& quad_v,
                                 std::size_t cu, std::size_t cv, MPI_Comm comm);
};

} // namespace cfd::mesh_generator