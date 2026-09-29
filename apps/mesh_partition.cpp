// mesh_partition: parallel CGNS preprocessing -> custom HDF5 + BC file + VTU.
//
// Usage:
//   mpirun -np N preproc <in.cgns> <out.h5> [--bc <bc.txt>] [--vtu <dir>]
//                        [--verbose|-v] [--reorder <TYPE> (avalivable TYPE=NONE,SFC,RCM)]
#include <mpi.h>

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <string>


#include "cfd/mpi/log.hpp"
#include "cfd/io/cgns/cgns_reader.hpp"
#include "cfd/mesh/raw_mesh.hpp"
#include "cfd/mesh/geometry.hpp"
#include "cfd/mesh/faces.hpp"
#include "cfd/partition/partition.hpp"
#include "cfd/mesh/localmesh.hpp"
#include "cfd/mesh/reorder.hpp"
#include "cfd/mesh/bc_config.hpp"
#include "cfd/mesh/validate.hpp"
#include "cfd/io/solver_mesh/hdf5_writer.hpp"
#include "cfd/io/vtk/vtu.hpp"

static void usage() {
    std::fprintf(stderr,
                 "usage: mpirun -np N mesh_partition <in.cgns> <out.h5> "
                 "[--bc <bc.txt>] [--vtu <dir>] [-v|--verbose] [--reorder <TYPE (avaliable: NONE, RCM, SFC)>]\n");
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
    std::string in, out, bcfile, vtudir;
    int verbose = 0;
    cfd::mesh::ReorderMethod reorder_m = cfd::mesh::ReorderMethod::NONE;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-v" || a == "--verbose")
            verbose = 1;
        else if (a == "-vv")
            verbose = 2;
        else if (a == "--reorder")
            reorder_m = cfd::mesh::parse_reorder_method(argv[++i]);
        else if (a == "--bc" && i + 1 < argc)
            bcfile = argv[++i];
        else if (a == "--vtu" && i + 1 < argc)
            vtudir = argv[++i];
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
    
    if (bcfile.empty()) bcfile = out + ".bc";

    cfd::mpi::log_init(verbose);

    double t0 = MPI_Wtime();
    cfd::mpi::log_info("mesh_partition: input=%s output=%s bcfile=%s, ranks=%d, reorder method=%s", 
            in.c_str(), out.c_str(), bcfile.c_str(), nprocs, cfd::mesh::reorder_method_to_string(reorder_m));


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


    // Distributed construction of unique faces, boundary-patch matching,
    // and a symmetric dual graph in CSR format.
    //
    // Collective operation: every rank in `m.comm`, including ranks with no
    // local cells, must participate. Uses MPI_Alltoall, MPI_Alltoallv,
    // and MPI_Allreduce internally.
    //
    // Preconditions:
    //  - `m` contains valid block-distributed mesh slices and displacement tables;
    //  - Cell connectivity follows canonical CGNS SIDS node ordering;
    //  - Supported volume cells are tetrahedra, pyramids, prisms, and hexahedra;
    //  - Surface elements contain canonical FaceKeys and valid patch assignments;
    //  - More than two volume half-faces sharing a key cause mpi::fatal().
    //
    // Algorithm Overview:
    //
    //  Step 1: Compact Source Index & Batch Planning
    //   - Counts local volume half-faces and surface elements.
    //   - Uses the maximum local count across ranks to select a common number
    //     of communication rounds.
    //   - Builds canonical face keys by sorting their node GIDs.
    //   - Stores 64-bit source references instead of retaining full messages
    //     for every local half-face and surface element.
    //
    //  Step 2: Hash-Based Assignment to Rounds & Rendezvous Ranks
    //   - Mixes all FaceKey slots with a private hash and a final avalanche.
    //   - Maps each key to a bucket identifying both its round and destination.
    //   - Equal keys always reach the same rank in the same round, so matching
    //     never requires carrying unmatched records between rounds.
    //   - Hash mixing improves distribution for regularly numbered meshes;
    //     perfect load balance is not guaranteed.
    //
    //  Step 3: Batched Exchange & Face Matching
    //   - Materializes full messages only for the current round.
    //   - Exchanges them using packed MPI_Alltoallv and releases send storage.
    //   - Sorts received messages by FaceKey and processes equal-key groups:
    //      * Two volume records: interior face, with cell_a < cell_b;
    //      * One volume record: boundary face, using the matching surface patch
    //        when available, otherwise kInvalidPatchId;
    //      * Surface-only group: no mesh face is emitted.
    //   - Counts outputs and then packs them directly, avoiding an intermediate
    //     array containing all matched FaceRec objects.
    //
    //  Step 4: Dispatch to Original Cell Owners
    //   - Sends each FaceRec to the rank owning cell_a in `m.cell_displ`.
    //   - For an interior face whose cells belong to different ranks, sends
    //     a lightweight reverse edge (cell_b -> cell_a) to cell_b's owner.
    //   - Releases the rendezvous receive buffer before these two successive
    //     exchanges; retains received face and edge chunks for final assembly.
    //
    //  Step 5: CSR Assembly & Global Statistics
    //   - Consolidates face chunks, releasing each consumed chunk.
    //   - Counts graph degrees directly in the CSR offsets array, computes
    //     prefix sums, and fills adjacency with global neighboring cell IDs.
    //   - Inserts both directions of every interior connection, using reverse
    //     edge messages for connections crossing original rank boundaries.
    //   - Sorts each cell's adjacency slice by global cell ID.
    //   - Reduces global face counts, directed graph-edge counts, and
    //     rendezvous-distribution diagnostics.
    //
    // Memory Semantics:
    //  - `m` is read-only and remains usable after this call.
    //  - CFD_FACES_BATCH_BYTES defaults to 128 MiB and controls the target
    //    average half-face send-buffer size per rank per round.
    //  - This target is NOT a strict receive-buffer or process-memory limit.
    //  - Temporary buffers are released after their last use; final face and
    //    graph storage still scales with the rank's retained output.
    //
    // After this call:
    //  - result.faces contains exactly one globally owned record per mesh face;
    //    its cell_a belongs to this rank's original cell block.
    //  - result.graph contains symmetric CSR adjacency for local cells,
    //    including connections to cells on other ranks.
    //  - result.stats contains replicated global totals.
    //  - Face-vector order is not a stable global numbering and may change
    //    with the rank count or batch configuration.
    cfd::mesh::BuildFacesResult dual_graph;
    dual_graph = cfd::mesh::build_faces(m);

    
    // Distributed dual-graph partitioning (dKaMinPar) and hardware topology-aware
    // process placement.
    //
    // MUST be called collectively by all ranks in `m.comm` (uses MPI collective
    // topology discovery, dKaMinPar distributed solver, and MPI_Allreduce internally) —
    // calling it on a subset of ranks will deadlock.
    //
    // Preconditions:
    //  - `m` contains valid local cell displacements (`m.cell_displ`);
    //  - `g` is the symmetric, sorted CSR dual graph produced by `build_faces()`;
    //  - `dko.imbalance_tolerance` is within a valid range (typically 0.02 - 0.05).
    //
    // Algorithm Overview:
    //
    //  Step 1: Hardware Topology Discovery & Cluster Mapping
    //   - Identifies co-located ranks sharing physical node memory via
    //     `MPI_Comm_split_type(..., MPI_COMM_TYPE_SHARED)`;
    //   - Elects node master leaders to assign unique physical `node_id`s;
    //   - Sorts all global ranks lexicographically by `(node_id, local_core_id)`
    //     to construct an optimal continuous block-to-hardware mapping (`part2rank`).
    //
    //  Step 2: Distributed CSR Formatting for dKaMinPar
    //   - Converts local CSR graph offsets and adjacencies into 64-bit
    //     distributed node/edge arrays (`vtxdist`, `xadj`, `adjncy`).
    //
    //  Step 3: Distributed Multi-Level Graph Partitioning
    //   - Invokes dKaMinPar with deep recursive bisection to partition the dual graph
    //     into `P` balanced subdomains minimizing global edge cuts;
    //   - Immediately frees temporary CSR buffers to drop memory pressure before
    //     running the heavy solver phases;
    //   - Maps raw partition block IDs to physical MPI ranks using the topology table,
    //     guaranteeing that topologically adjacent blocks (0..K-1) are assigned to
    //     cores on the SAME physical server (Node 0), eliminating network traffic.
    //
    //  Step 4: Distributed Diagnostics & Load Balance Reduction
    //   - Retrieves exact `global_edge_cut` directly from the partitioner without
    //     redundant distributed halo exchanges;
    //   - Computes global cell distribution statistics (min/avg/max cells per rank
    //     and overall imbalance %) via a single lightweight `MPI_Allreduce`.
    //
    // After this call, each rank holds `cfd::partition::PartitionResult`:
    //  - `result.cell_target_rank`: array of size `n_local_cells` defining the target
    //    destination MPI rank for each currently owned volume cell (ready for migration);
    //  - `result.part2rank`: bijection mapping PartitionBlockId -> Target MPI Rank
    //    (used to dispatch cell packages during data migration);
    //  - `result.rank2part`: inverse bijection mapping MPI Rank -> Assigned PartitionBlockId
    //    (used to compute exact deterministic byte offsets in the solver binary file);
    //  - `result.global_edge_cut`: total face cuts across all MPI rank boundaries.
    cfd::partition::PartitionResult pr = cfd::partition::partition_cells(
        m, 
        dual_graph.graph, 
        cfd::partition::DKaMinParOptions{
            .block_count = 0,             // 0 = default to m.nprocs
            .imbalance_tolerance = 0.03,  // 3% standard CFD load imbalance
            .threads_per_rank = 1,        // TBB threads per rank
            .seed = 0,
            .quiet = true
        }
    );


    // Distributed mesh redistribution, one-layer face-neighbor halo construction,
    // local zero-based renumbering, and solver communication-map generation.
    //
    // Collective operation: every rank in `m.comm`, including empty ranks,
    // must participate. Uses packed collective exchanges and global reductions.
    //
    // Memory Semantics:
    //  - Destructively consumes selected buffers from `m` and `faces` through
    //    rvalue references; callers must not rely on their original contents.
    //  - Moves reusable arrays and releases temporary storage after its last use.
    //  - Uses flat connectivity arrays, compact face records, and adaptive
    //    dense/open-addressed GID lookups to reduce allocation overhead.
    //  - Single-rank paths reuse storage and bypass redistribution where possible.
    //  - Temporary send/receive and conversion buffers can still overlap;
    //    there is no fixed bound on peak process memory.
    //
    // Preconditions:
    //  - `m` contains valid raw connectivity, coordinates, BC metadata,
    //    and original cell/node displacement tables.
    //  - `faces` contains unique faces produced by build_faces(), stored on the
    //    original owner rank of cell_a, with valid CGNS local face indices.
    //  - `pr.cell_target_rank` provides a destination MPI rank for each raw
    //    local cell.
    //  - `pr.global_edge_cut` counts unique faces crossing destination ranks,
    //    consistently with the supplied partition.
    //
    // Pipeline Overview:
    //
    //  Step 1: Resolve Remote Cell Destinations
    //   - Reads target ranks directly for locally held cells.
    //   - Requests the destination of remote cell_b entries from their original
    //     cell owners using request and reply exchanges.
    //
    //  Step 2: Redistribute Owned Cells
    //   - Packs cell GIDs, types, and node-GID connectivity by destination rank.
    //   - Releases raw cell arrays after packing and send buffers after exchange.
    //   - Assembles received owned-cell metadata and flat connectivity.
    //   - Builds a GID-to-local lookup using dense storage for compact ranges
    //     or an open-addressed table for sparse ranges.
    //
    //  Step 3: Migrate Faces & Identify Ghost Cells
    //   - Sends boundary faces and faces internal to one destination rank once.
    //   - Sends each face crossing destination ranks to both adjacent owners,
    //     with the local owned cell as face owner on each receiving rank.
    //   - Releases the original FaceRec storage after conversion/packing.
    //   - Deduplicates required ghost cells and sorts them by
    //     (donor_rank, cell_gid), giving contiguous receive-halo groups.
    //
    //  Step 4: Fetch Ghost Cell Connectivity
    //   - Requests types and complete node-GID lists from the new cell owners.
    //   - Builds flat topology arrays for the one-layer ghost-cell set.
    //
    //  Step 5: Collect & Number Required Nodes
    //   - Collects unique node GIDs used by owned and ghost cells.
    //   - Places nodes used by owned cells first, followed by ghost-only nodes;
    //     each group is ordered by global node ID.
    //   - Uses a dense scan for compact GID ranges or sorts unique sparse IDs.
    //   - "Owned nodes" here means nodes referenced by owned cells, not exclusive
    //     global node ownership: neighboring ranks may store the same node.
    //
    //  Step 6: Assemble Local Cells & Fetch Coordinates
    //   - Numbers owned cells in [0, n_own) and ghosts in [n_own, n_cells).
    //   - Moves owned metadata, appends ghost metadata, and converts node GIDs
    //     into local indices in the final cell CSR arrays.
    //   - Releases temporary global connectivity and the node lookup before
    //     allocating/fetching final coordinates.
    //   - Copies coordinates held locally and requests only remote coordinates
    //     from original node owners using `m.node_displ`.
    //   - Releases raw coordinates after their last use; the eligible
    //     single-rank path moves coordinate arrays directly.
    //
    //  Step 7: Reconstruct Local Faces
    //   - Resolves local owner and neighbor indices; boundary neighbors are -1.
    //   - Sorts compact face records lexicographically by (owner, neigh).
    //   - Builds face metadata and flat face-node connectivity directly from
    //     the owner's local face index and canonical CGNS face tables.
    //   - Preserves the tables' owner-relative orientation; outward geometric
    //     orientation assumes valid, correctly oriented input cells.
    //
    //  Step 8: Construct Halo Communication Maps
    //   - Lists neighboring donor ranks and builds contiguous receive groups.
    //   - Exchanges requested ghost-cell GIDs to construct corresponding
    //     send_owned_local and send_offsets arrays on donor ranks.
    //   - Produces matching send/receive index order for later field exchanges;
    //     packing and MPI execution are handled by the solver's halo layer.
    //
    //  Step 9: Build Patches, Statistics & Structural Checks
    //   - Copies BC metadata and groups faces with valid patch IDs into flat
    //     patch-face CSR arrays.
    //   - Reduces the bounding box over nodes referenced by owned cells.
    //   - Computes unique global face count as summed local face counts minus
    //     `pr.global_edge_cut`, accounting for duplicated partition-cut faces.
    //   - Reports the global count of faces assigned to valid BC patches;
    //     unassigned boundary faces are not included in that count.
    //   - Runs meshpart_sane() structural checks and aborts on detected errors.
    //
    // After this call:
    //  - `mp` contains owned cells, their face-neighbor ghosts, required nodes,
    //    local connectivity, boundary patches, and halo communication maps.
    //  - Partition-cut faces occur on both adjacent ranks.
    //  - Geometry metrics and any subsequent reordering are separate operations.
    cfd::mesh::MeshPart mp;
    cfd::mesh::migrate_local_mesh(std::move(m), std::move(dual_graph.faces), pr, mp);


    // Parallel geometric processing, metric computation (volumes, true volume centroids, 
    // face areas, area-weighted face centroids, unit normals), and strict topological 
    // orientation verification.
    //
    // MUST be called collectively by all ranks (performs MPI reductions internally
    // to calculate domain bounding statistics and verify global metric sanity).
    //
    // Preconditions:
    //  - `mp` contains a fully migrated, locally numbered subdomain (produced by `migrate_local_mesh`);
    //  - Vertex coordinates (`mp.node_x`, `mp.node_y`, `mp.node_z`) are populated for all owned and ghost nodes;
    //  - Face node loops follow canonical CGNS winding order (`kFaceTable`).
    //
    // Geometric Contract & Normal Orientation Guarantees:
    //
    //  1. Cell Metrics:
    //     - Centroids: Exact volume centroids (center of mass for uniform density) 
    //       computed via canonical tetrahedral decomposition with a shift-invariant local anchor 
    //       x0 = mean(vertices), eliminating floating-point cancellation on meshes shifted from origin;
    //     - Volumes: Exact signed volume integration over constituent tetrahedra, strictly 
    //       consistent with the centroid moment calculation (supports arbitrary polyhedral types:
    //       TET, PYRA, PRISM, HEXA, and general polyhedra);
    //     - Positivity Guarantee: Asserts volume V > 10^-15 for all cells; aborts with a diagnostic
    //       dump if degenerate or inverted elements are detected.
    //
    //  2. Face Metrics & Normal Vector Convention (The Solver Contract):
    //     - Centroids: Exact area-weighted centroids computed via sub-triangle fan integration 
    //       (preserves 2nd-order accuracy on warped quads and non-regular polygons);
    //     - Areas & Normals: Area vector S is accumulated across sub-triangles (magnitude A = ||S|| > 10^-15),
    //       yielding the true mean surface normal n_hat = S / A;
    //     - Interior Faces: The unit normal vector n_hat = (nx, ny, nz) is strictly directed from
    //       `face_owner` OUTWARD toward `face_neigh` (n_hat · (x_face - x_owner) > 0);
    //     - Boundary Faces (`face_neigh == -1`): Follows the exact same outward contract — the normal
    //       points from `face_owner` toward the exterior, guaranteeing that all boundary normals
    //       point strictly OUTWARD from the computational domain.
    //
    //  3. Alignment & Skewness Verification:
    //     - Performs runtime dot-product assertion n_hat · (x_face - x_owner) > 0 using physical 
    //       centers of mass (x_face, x_owner), mathematically verifying that face winding matches 
    //       CGNS SIDS orientation without false positives on high-aspect-ratio boundary-layer cells.
    //
    // After this call, `mp` is completely populated with all geometric metrics and fully verified,
    // ready for direct serialization into the solver-ready binary mesh format.
    cfd::mesh::compute_mesh_geometry(mp);


    // Local cell and face reordering for CPU cache locality, branch elimination, matrix bandwidth,
    // and non-blocking communication-computation overlap.
    //
    // Optimizes memory access patterns for owned cells [0, n_own):
    //  - HILBERT_SFC: 3D Space-Filling Curve (optimal L1/L2 cache spatial locality for Explicit solvers);
    //  - RCM: Reverse Cuthill-McKee (minimizes sparse matrix bandwidth for Implicit solvers).
    //
    // Guarantees & Invariants:
    //  - Preserves ghost layer contiguous layout intact [n_own, n_cells);
    //  - Automatically updates `send_owned_local` communication indices;
    //  - Partitions and sorts the face array into three contiguous zones:
    //      1. Pure internal faces [0, n_internal_faces): owner < n_own, neigh < n_own.
    //         Sorted monotonically by (owner, neigh) for branchless SIMD/AVX flux loops,
    //         allowing immediate local evaluation while asynchronous MPI halo exchanges are in flight;
    //      2. Coupled / Inter-rank faces [n_internal_faces, n_inner_faces): owner < n_own, neigh >= n_own.
    //         Sorted by (owner, neigh) for cache locality, evaluated immediately after MPI_Waitall;
    //      3. Boundary faces [n_inner_faces, n_faces): neigh == -1.
    //         Grouped contiguously by `patch_id`, then sorted by `owner` cell index for vectorized BC evaluations;
    //  - Reconstructs flat CSR boundary patch tables (`patch_face_offsets`, `patch_faces`);
    //  - Sets `mp.n_internal_faces` to the exact count of pure internal faces;
    //  - Sets `mp.n_inner_faces` to the total count of internal and inter-rank coupled faces (offset where boundary patches begin).
    cfd::mesh::reorder_local_mesh(mp, reorder_m);


    // Exhaustive Sanity Check & Global Diagnostics  
    cfd::mesh::validate_and_log_meshpart(mp);


    // Export Boundary Condition Template Configuration
    cfd::mesh::generate_bc_template_config(mp, bcfile);
    

    // Parallel binary serialization into a single shared topology-aware HDF5 container.
    //
    // MUST be called collectively by all ranks in MPI_COMM_WORLD.
    //
    // Key Properties:
    //  - Uses parallel HDF5 with MPI-IO collective transfers (H5FD_MPIO_COLLECTIVE);
    //  - Subdomains are ordered strictly by partition ID (`pr.rank2part[rank]`), preserving
    //    spatial locality on disk matching the Hilbert curve / dual graph partition;
    //  - Stores full precomputed SoA metrics (volumes, unit normals, areas, centroids)
    //    and communication maps for zero-overhead solver startup.
    cfd::io::solver_mesh::export_mesh_hdf5(mp, pr, out, MPI_COMM_WORLD);


    if (!vtudir.empty()) cfd::io::vtk::write_vtu(mp, vtudir, "part", MPI_COMM_WORLD);

    double t1 = MPI_Wtime() - t0;
    if (rank == 0) std::fprintf(stderr, "Total executional time = %.5f sec\n", t1);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
