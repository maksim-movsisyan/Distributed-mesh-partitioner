#include "cfd/partition/partition.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <dkaminpar.h>
#include <mpi.h>

#include "cfd/mpi/log.hpp"

namespace cfd::partition {

namespace {

// =============================================================================
// Cluster Topology Discovery
// =============================================================================

struct RankTopology {
    int node_id = -1;
    int local_rank = -1;
    int global_rank = -1;

    [[nodiscard]] bool operator<(const RankTopology& o) const noexcept {
        if (node_id != o.node_id) return node_id < o.node_id;
        return local_rank < o.local_rank;
    }
};

static_assert(sizeof(RankTopology) == 3 * sizeof(int), 
              "RankTopology must be tightly packed without padding for MPI byte transfers");

struct TopologyInfo {
    std::vector<int> part2rank;
    std::vector<int> rank_to_node;
};

// Discovers physical cluster compute nodes and constructs part->rank & rank->node tables
TopologyInfo build_topology_info(MPI_Comm comm, int nprocs, int rank) {
    const std::size_t nprocs_sz = static_cast<std::size_t>(nprocs);

    // 1. Discover ranks sharing physical node memory
    MPI_Comm shared_comm = MPI_COMM_NULL;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shared_comm);

    int shared_rank = 0;
    MPI_Comm_rank(shared_comm, &shared_rank);

    // 2. Leader communicator among node master processes (shared_rank == 0)
    MPI_Comm leader_comm = MPI_COMM_NULL;
    MPI_Comm_split(comm, shared_rank == 0 ? 0 : MPI_UNDEFINED, rank, &leader_comm);

    int node_id = -1;
    int num_nodes = 0;
    if (shared_rank == 0) {
        MPI_Comm_rank(leader_comm, &node_id);
        MPI_Comm_size(leader_comm, &num_nodes);
    }

    MPI_Bcast(&node_id, 1, MPI_INT, 0, shared_comm);
    MPI_Bcast(&num_nodes, 1, MPI_INT, 0, shared_comm);

    if (leader_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&leader_comm);
    }
    MPI_Comm_free(&shared_comm);

    // 3. Gather full cluster topology to compute continuous block-to-node placement
    const RankTopology my_topo{node_id, shared_rank, rank};
    std::vector<RankTopology> all_topo(nprocs_sz);

    MPI_Allgather(&my_topo, static_cast<int>(sizeof(RankTopology)), MPI_BYTE,
                  all_topo.data(), static_cast<int>(sizeof(RankTopology)), MPI_BYTE, comm);

    // Build rank_to_node lookup
    std::vector<int> rank_to_node(nprocs_sz);
    for (std::size_t i = 0; i < nprocs_sz; ++i) {
        rank_to_node[i] = all_topo[i].node_id;
    }

    // Sort ranks primarily by physical node, secondarily by core ID
    std::vector<RankTopology> sorted_topo = all_topo;
    std::sort(sorted_topo.begin(), sorted_topo.end());

    std::vector<int> part2rank(nprocs_sz);
    for (std::size_t i = 0; i < nprocs_sz; ++i) {
        part2rank[i] = sorted_topo[i].global_rank;
    }

    mpi::log_stat("INFO[Partitioner]: Detected %d physical compute nodes across %d MPI ranks", num_nodes, nprocs);

    return TopologyInfo{std::move(part2rank), std::move(rank_to_node)};
}

} // anonymous namespace

PartitionResult partition_cells(const mesh::RawMesh& m, const mesh::DualGraph& g, const DKaMinParOptions& dko) {
    // get local rank index and total ranks count
    const int nprocs = m.nprocs;
    const int rank = m.rank;
    const MPI_Comm comm = m.comm;

    // get local cell count
    const LocalIndex n_loc_cells = m.n_local_cells();
    
    // aux cast
    const std::size_t n_loc_cells_sz = static_cast<std::size_t>(n_loc_cells);
    const std::size_t nprocs_sz = static_cast<std::size_t>(nprocs);
    const PartitionBlockId num_blocks = (dko.block_count == 0) ? static_cast<PartitionBlockId>(nprocs) : dko.block_count;


    // -------------------------------------------------------------------------
    // Step 1: Discover Hardware Topology & Construct Part -> Rank Mapping
    // -------------------------------------------------------------------------
    PartitionResult result;
    const auto [part2rank, rank_to_node] = build_topology_info(comm, nprocs, rank);
    result.part2rank = part2rank;
    result.cell_target_rank.resize(n_loc_cells_sz);


    // -------------------------------------------------------------------------
    // Step 2: Prepare 64-bit Distributed CSR Data for dKaMinPar
    // -------------------------------------------------------------------------
    std::vector<kaminpar::dist::GlobalNodeID> vtxdist(nprocs_sz + 1);
    for (std::size_t i = 0; i <= nprocs_sz; ++i) {
        vtxdist[i] = static_cast<kaminpar::dist::GlobalNodeID>(m.cell_displ[i]);
    }

    std::vector<kaminpar::dist::GlobalEdgeID> xadj(n_loc_cells_sz + 1);
    for (std::size_t i = 0; i <= n_loc_cells_sz; ++i) {
        xadj[i] = static_cast<kaminpar::dist::GlobalEdgeID>(g.offsets[i]);
    }

    std::vector<kaminpar::dist::GlobalNodeID> adjncy(g.adj.size());
    for (std::size_t i = 0; i < g.adj.size(); ++i) {
        adjncy[i] = static_cast<kaminpar::dist::GlobalNodeID>(g.adj[i]);
    }


    // -------------------------------------------------------------------------
    // Step 3: Run dKaMinPar Graph Partitioning
    // -------------------------------------------------------------------------
    kaminpar::dKaMinPar::reseed(dko.seed);
    kaminpar::dKaMinPar partitioner(comm, dko.threads_per_rank, kaminpar::dist::create_default_context());

    partitioner.set_output_level(dko.quiet ? kaminpar::OutputLevel::QUIET : kaminpar::OutputLevel::EXPERIMENT);
    partitioner.copy_graph(vtxdist, xadj, adjncy);

    // Release temporary CSR buffers immediately to minimize peak memory pressure during partitioning
    vtxdist.clear();
    vtxdist.shrink_to_fit();
    xadj.clear();
    xadj.shrink_to_fit();
    adjncy.clear();
    adjncy.shrink_to_fit();

    std::vector<kaminpar::dist::BlockID> raw_partition(n_loc_cells_sz, kaminpar::dist::kInvalidBlockID);

    const kaminpar::dist::GlobalEdgeWeight edge_cut_value = partitioner.compute_partition(
        static_cast<kaminpar::dist::BlockID>(num_blocks),
        dko.imbalance_tolerance,
        raw_partition
    );

    result.global_edge_cut = static_cast<GlobalIndex>(edge_cut_value);
    result.inter_node_cut = 0; // Cut evaluation without heavy ghost exchange

    // Map raw partition block IDs to topology-aware target MPI ranks
    for (std::size_t c = 0; c < n_loc_cells_sz; ++c) {
        const std::size_t b = static_cast<std::size_t>(raw_partition[c]);
        result.cell_target_rank[c] = (b < result.part2rank.size()) ? result.part2rank[b] : static_cast<int>(b);
    }

    std::vector<PartitionBlockId> rank2part(static_cast<std::size_t>(nprocs));
    for (int block_id = 0; block_id < nprocs; ++block_id) {
        rank2part[static_cast<std::size_t>(part2rank[static_cast<std::size_t>(block_id)])] = static_cast<PartitionBlockId>(block_id);
    }
    result.rank2part = std::move(rank2part);


    // -------------------------------------------------------------------------
    // Step 4: Distributed Diagnostics & Load Balance Reduction
    // -------------------------------------------------------------------------
    std::vector<std::int64_t> local_ncells_per_rank(nprocs_sz, 0);
    for (std::size_t i = 0; i < n_loc_cells_sz; ++i) {
        const int target = result.cell_target_rank[i];
        ++local_ncells_per_rank[static_cast<std::size_t>(target)];
    }

    std::vector<std::int64_t> global_ncells_per_rank(nprocs_sz, 0);
    MPI_Allreduce(
        local_ncells_per_rank.data(), 
        global_ncells_per_rank.data(), 
        nprocs, 
        MPI_INT64_T, 
        MPI_SUM, 
        comm
    );

    std::int64_t min_cells = global_ncells_per_rank[0];
    std::int64_t max_cells = global_ncells_per_rank[0];
    std::int64_t total_cells = 0;

    for (std::size_t i = 0; i < nprocs_sz; ++i) {
        min_cells = std::min(min_cells, global_ncells_per_rank[i]);
        max_cells = std::max(max_cells, global_ncells_per_rank[i]);
        total_cells += global_ncells_per_rank[i];
    }

    const double avg_cells = static_cast<double>(total_cells) / static_cast<double>(nprocs);
    const double imbalance_pct = (avg_cells > 0.0) ? (static_cast<double>(max_cells) - avg_cells) / avg_cells * 100.0 : 0.0;

    mpi::log_stat(
        "INFO[Partitioner]: Partitioning complete. Global edge-cut = %lld, Cells per rank: [min=%lld, avg=%.0f, max=%lld], Imbalance = %.2f%%",
        static_cast<long long>(result.global_edge_cut),
        static_cast<long long>(min_cells),
        avg_cells,
        static_cast<long long>(max_cells),
        imbalance_pct
    );

    return result;
}

} // namespace cfd::partition