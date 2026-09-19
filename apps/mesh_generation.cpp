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

    MPI_Finalize();
    return EXIT_SUCCESS;
}