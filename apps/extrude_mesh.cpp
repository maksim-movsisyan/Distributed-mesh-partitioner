// extrude_mesh: parallel CGNS preprocessing -> 2D CGNS FILE -> 3D CGNS FILE.
// just aux app, extrude is already on in mesh_partitoner (if mesh is 2D), use this app for specific needs
//
// Usage:
//   mpirun -np N preproc <in.cgns> <out.cgns> [--verbose|-v]
#include <mpi.h>

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <string>


#include "cfd/mpi/log.hpp"
#include "cfd/io/cgns/cgns_reader.hpp"
#include "cfd/io/cgns/cgns_writer.hpp"
#include "cfd/mesh/raw_mesh.hpp"
#include "cfd/mesh/geometry.hpp"


static void usage() {
    std::fprintf(stderr,
                 "usage: mpirun -np N mesh_partition <in.cgns> <out.cgns> [-v|--verbose]\n");
}

int main(int argc, char** argv) {
    // MPI initialization 
    int provided_thread_level = MPI_THREAD_SINGLE;
    const int init_status = 
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided_thread_level);

    if (init_status != MPI_SUCCESS) { 
        std::cerr << "MPI_Init_thread failed\n"; return EXIT_FAILURE;
    }


    // get local rank index and total process number
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);


    // check if target thread level is avaliable
    if (provided_thread_level < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            std::cerr << "MPI did not provide the requested thread level\n";
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }


    // parsing arguments
    std::string in, out;
    int verbose = 0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-v" || a == "--verbose")
            verbose = 1;
        else if (a == "-vv")
            verbose = 2;
        else if (in.empty())
            in = a;
        else if (out.empty())
            out = a;
        else {
            if (rank == 0) usage();
            MPI_Finalize();
        }
    }

    if (in.empty() || out.empty()) {
        if (rank == 0) usage();
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    
    cfd::mpi::log_init(verbose);

    double t0 = MPI_Wtime();
    cfd::mpi::log_info("mesh_partition: input=%s output=%s", in.c_str(), out.c_str());


    // self-check of the canonical face tables
    if (!cfd::mesh::validate_face_tables()) {
        if (rank == 0)
            std::fprintf(stderr,
                         "ERROR: canonical face table self-check "
                         "FAILED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cfd::mpi::log_stat("Canonical face table self-check: OK");


    // Parallel CGNS mesh reader.
    // MUST be called collectively by all ranks in `comm` (uses cgp_* collective I/O
    // internally) — calling it on a subset of ranks will deadlock.
    // Precondition: the file must contain exactly one 3D Base and one Unstructured
    // Zone; anything else aborts via fatal(). Boundary conditions are only
    // recognized when GridLocation == FaceCenter; others are skipped (warning only).
    //
    // After this call, each rank holds:
    //  - global metadata: total node/cell counts, CGNS file info (version,
    //    precision, storage type);
    //  - grid metadata: description of volume and surface element sections;
    //  - boundary conditions, fully replicated on every rank (name, CGNS BC type,
    //    global 1-based element ids);
    //  - its own contiguous slice of volume cells (cell types + 0-based
    //    cell->node connectivity), from a simple global index block-split
    //    (NOT graph-partitioned — no locality guarantee);
    //  - its own contiguous slice of nodes (x/y/z coordinates), same
    //    block-split scheme;
    //  - cell_displ / node_displ, replicated on every rank, mapping a global
    //    cell/node index to its owning rank;
    //  - its own contiguous slice of surface elements (patch id, sorted node
    //    key, global 1-based element id). NOTE: surface elements are
    //    block-split independently PER SECTION, so this slice is unrelated
    //    to the cell/node ownership above.
    cfd::mesh::RawMesh m = cfd::io::cgns::read_cgns_parallel(in, MPI_COMM_WORLD);
    cfd::io::cgns::write_cgns_parallel(out, m);

    double t1 = MPI_Wtime() - t0;
    if (rank == 0) std::fprintf(stderr, "Total executional time = %.5f sec\n", t1);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
