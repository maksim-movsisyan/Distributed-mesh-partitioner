// generate_mesh: Distributed Block Mesh Generator for CFD
//
// Usage:
//   mpirun -np N generate_mesh <config.toml> [out.cgns] [-v|--verbose]

#include <mpi.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include "cfd/mesh_generator/config.hpp"
#include "cfd/mesh_generator/grading.hpp"
#include "cfd/mesh_generator/tfi.hpp"
#include "cfd/mesh_generator/topology.hpp"
#include "cfd/mpi/log.hpp"

static void usage() {
    std::fprintf(stderr,
                 "usage: mpirun -np N generate_mesh <config.toml> [out.cgns] [-v|--verbose]\n");
}

int main(int argc, char** argv) {
    // MPI initialization
    int provided_thread_level = MPI_THREAD_SINGLE;
    const int init_status =
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided_thread_level);

    if (init_status != MPI_SUCCESS) {
        std::cerr << "MPI_Init_thread failed\n";
        return EXIT_FAILURE;
    }

    // Get local rank index and total process number
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Check if target thread level is available
    if (provided_thread_level < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            std::cerr << "MPI did not provide the requested thread level\n";
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Parsing CLI arguments
    std::string config_path;
    std::string out_cgns_path;
    int verbose = 0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-v" || a == "--verbose") {
            verbose = 1;
        } else if (a == "-vv") {
            verbose = 2;
        } else if (config_path.empty()) {
            config_path = a;
        } else if (out_cgns_path.empty()) {
            out_cgns_path = a;
        } else {
            if (rank == 0) usage();
            MPI_Finalize();
            return EXIT_FAILURE;
        }
    }

    if (config_path.empty()) {
        if (rank == 0) usage();
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    cfd::mpi::log_init(verbose);

    const double t0 = MPI_Wtime();

    if (rank == 0) {
        std::cout << "==================================================\n";
        std::cout << "  CFD Block Mesh Generator (Validation Mode)\n";
        std::cout << "  Processes: " << nprocs << "\n";
        std::cout << "  Config:    " << config_path << "\n";
        std::cout << "==================================================\n";
    }

    // Parse configuration and validate all design contracts collectively
    const auto cfg = cfd::mesh_generator::parse_generator_config(config_path, MPI_COMM_WORLD);

    // Compute mesh volume statistics
    std::uint64_t total_volume_cells = 0;
    for (const auto& block : cfg.blocks) {
        total_volume_cells += static_cast<std::uint64_t>(block.cells[0]) *
                              static_cast<std::uint64_t>(block.cells[1]) *
                              static_cast<std::uint64_t>(block.cells[2]);
    }

    std::uint64_t total_boundary_faces = 0;
    for (const auto& patch : cfg.boundaries) {
        total_boundary_faces += patch.faces.size();
    }

    const double parse_time = MPI_Wtime() - t0;

    // Report results on Rank 0
    if (rank == 0) {
        std::cout << "\n[Config Validation Succeeded]\n";
        std::cout << "  Mesh Name:         " << cfg.name << "\n";
        std::cout << "  Scaling Factor:    " << cfg.scale << "\n";
        std::cout << "  Macro Vertices:    " << cfg.vertices.size() << "\n";
        std::cout << "  Macro Blocks:      " << cfg.blocks.size() << "\n";
        std::cout << "  Boundary Patches:  " << cfg.boundaries.size() << "\n";
        std::cout << "  Total Hex Cells:   " << total_volume_cells << "\n";
        std::cout << "  Macro Bnd Faces:   " << total_boundary_faces << "\n";

        if (verbose > 0) {
            std::cout << "\n--- Blocks Breakdown ---\n";
            for (std::size_t i = 0; i < cfg.blocks.size(); ++i) {
                const auto& b = cfg.blocks[i];
                const auto b_cells = static_cast<std::uint64_t>(b.cells[0]) * b.cells[1] * b.cells[2];
                std::cout << "  [" << i << "] " << b.name << ": "
                          << b.cells[0] << "x" << b.cells[1] << "x" << b.cells[2]
                          << " (" << b_cells << " cells)\n";
                std::cout << "      Grading Types: ["
                          << cfd::mesh_generator::to_string(b.grading_type[0]) << ", "
                          << cfd::mesh_generator::to_string(b.grading_type[1]) << ", "
                          << cfd::mesh_generator::to_string(b.grading_type[2]) << "]\n";
                std::cout << "      Grading Values: ["
                          << b.grading[0] << ", " << b.grading[1] << ", " << b.grading[2] << "]\n";
            }

            std::cout << "\n--- Boundary Patches ---\n";
            for (const auto& patch : cfg.boundaries) {
                std::cout << "  Patch '" << patch.name << "': " << patch.faces.size() << " macro-faces\n";
            }
        }

        std::printf("\nConfig parsed, broadcasted and verified in %.4f s\n", parse_time);
        std::cout << "==================================================\n";
    }

    if (rank == 0 && verbose > 0) {
        std::cout << "\n--- Interpolation & Grading Spot Check ---\n";
        for (std::size_t b = 0; b < cfg.blocks.size(); ++b) {
            const auto& blk = cfg.blocks[b];
            std::array<cfd::mesh_generator::Vec3, 8> corners;
            for (std::size_t i = 0; i < 8; ++i) {
                corners[i] = cfg.vertices[blk.vertices[i]];
            }

            const auto dist_x = cfd::mesh_generator::Grading::compute_distribution(blk.cells[0], blk.grading_type[0], blk.grading[0]);
            const auto dist_y = cfd::mesh_generator::Grading::compute_distribution(blk.cells[1], blk.grading_type[1], blk.grading[1]);
            const auto dist_z = cfd::mesh_generator::Grading::compute_distribution(blk.cells[2], blk.grading_type[2], blk.grading[2]);

            const double xi_mid   = dist_x[blk.cells[0] / 2];
            const double eta_mid  = dist_y[blk.cells[1] / 2];
            const double zeta_mid = dist_z[blk.cells[2] / 2];

            const auto p_center = cfd::mesh_generator::TFI::interpolate_hex(corners, xi_mid, eta_mid, zeta_mid);
            std::printf("  Block [%zu] Center Physical Coord: (%.4f, %.4f, %.4f)\n", b, p_center.x, p_center.y, p_center.z);
            std::printf("      First delta_x: %.6f, Last delta_x: %.6f\n", dist_x[1] - dist_x[0], dist_x.back() - dist_x[dist_x.size() - 2]);
        }
    }

    cfd::mesh_generator::MacroTopology topo;
    topo.build(cfg, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "\n[Macro-Topology Analysis Succeeded]\n";
        std::cout << "  Total Unique Nodes: " << topo.total_nodes() << "\n";
        std::cout << "  Total Volume Cells: " << topo.total_cells() << "\n";
        std::cout << "  Unique Macro Edges: " << topo.num_edges() << "\n";
        std::cout << "  Unique Macro Faces: " << topo.num_faces() << "\n";

        // Spot check cell 0 and last cell node IDs
        const auto nodes_first = topo.get_cell_nodes(0);
        const auto nodes_last  = topo.get_cell_nodes(topo.total_cells() - 1);

        std::cout << "  Cell [0] Nodes (1-based): [";
        for (std::size_t i = 0; i < 8; ++i) std::cout << nodes_first[i] << (i < 7 ? ", " : "]\n");

        std::cout << "  Cell [" << topo.total_cells() - 1 << "] Nodes (1-based): [";
        for (std::size_t i = 0; i < 8; ++i) std::cout << nodes_last[i] << (i < 7 ? ", " : "]\n");
    }
    
    MPI_Finalize();
    return EXIT_SUCCESS;
}