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


    // Distributed construction of unique faces, boundary condition matching,
    // and dual graph CSR representation from raw block-distributed mesh slices.
    //
    // MUST be called collectively by all ranks in `m.comm` (uses MPI_Alltoall,
    // MPI_Alltoallv, and MPI_Allreduce internally) — calling it on a subset of
    // ranks will deadlock.
    //
    // Preconditions:
    //  - `m` contains valid contiguous 1D block-split slices of cells, nodes,
    //    and surface elements (as produced by read_cgns_parallel);
    //  - `m.cnodes` conforms to canonical CGNS SIDS element node orderings;
    //  - Volume cells are strictly 3D manifold elements (TET_4, PYRA_5,
    //    PRISM_6, HEXA_8); non-manifold meshes (>2 cells sharing a face)
    //    will abort via mpi::fatal().
    //
    // Algorithm Overview (2-Phase Rendezvous / Owner-Compute Scheme):
    //
    //  Phase 1: Generation & Hash-Based Rendezvous Dispatch
    //   - Every rank iterates over its local volume cells, extracts canonical
    //     sub-faces using CGNS lookup tables, and builds a sorted 4-node `FaceKey`.
    //   - Every rank takes its local slice of `surf_elems` (containing BC PatchIds).
    //   - A rendezvous destination rank is computed deterministically via
    //     `FaceKeyHash(FaceKey) % nprocs`, guaranteeing uniform O(N_faces / P) memory
    //     and network distribution across all ranks (prevents incast / skew on rank 0).
    //   - All half-faces and surface elements are packed and dispatched using
    //     a single `MPI_Alltoallv` exchange.
    //
    //  Phase 2: Deduplication, Matching & Ownership Assignment
    //   - On each rendezvous rank, incoming records are sorted by `FaceKey`
    //     to cluster coincident half-faces in contiguous cache lines.
    //   - Deduplication matches pairs:
    //      * 2 volume half-faces  -> Interior Face (cell_a = min(c1, c2), cell_b = max(c1, c2));
    //      * 1 volume + 1 surface -> Boundary Face (cell_a = c1, cell_b = -1, patch = SurfElem.patch);
    //      * 1 volume only        -> Undefined Boundary Face (cell_a = c1, cell_b = -1, patch = -1).
    //   - Ownership Rule: The rank owning `cell_a` is assigned sole ownership of the `FaceRec`.
    //
    //  Phase 3: Dispatch to Cell Owners & Graph Back-Edges
    //   - Full `FaceRec` structures are dispatched to the owner rank of `cell_a`.
    //   - For inter-rank interior faces (`rank(cell_a) != rank(cell_b)`), a lightweight
    //     `DualEdgeMsg (cell_b -> cell_a)` is dispatched to the owner of `cell_b`
    //     to ensure CSR graph symmetry without duplicating heavy `FaceRec` storage.
    //   - Dispatched via two parallel packed `MPI_Alltoallv` exchanges.
    //
    //  Phase 4: Dual Graph CSR Assembly & Global Stats Reduction
    //   - Receiving ranks assemble the local Dual Graph in CSR format (`offsets` + `adj`).
    //   - Every local vertex's adjacency slice is sorted by global cell ID (required
    //     by ParMETIS / PT-Scotch / KaHIP and enables O(log(deg)) binary searches).
    //   - A single collective `MPI_Allreduce` computes global face statistics.
    //
    // After this call, each rank holds:
    //  - `result.faces`: list of unique faces where `cell_a` is local to this rank
    //    (zero memory duplication across partition boundaries; exactly 1 copy of
    //    each mesh face exists globally across all ranks);
    //  - `result.graph`: symmetric distributed CSR dual graph for local cells
    //    [my_cell_start, my_cell_end), containing all local-local and inter-rank
    //    cell-cell adjacencies;
    //  - `result.stats`: replicated global totals (total unique faces, total interior,
    //    total boundary).
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


    // Distributed mesh migration, ghost-layer halo construction, local 0-based
    // renumbering, and solver communication topology generation.
    //
    // MUST be called collectively by all ranks in `m.comm` (uses Alltoall/Alltoallv
    // point-to-point exchanges and Allreduce reductions internally) — calling it on
    // a subset of ranks will deadlock.
    //
    // Memory Semantics:
    //  - Destructive move (`std::move`): Takes ownership of `m` (RawMesh) and
    //    `dual_graph.faces` (std::vector<FaceRec>), progressively freeing raw data
    //    buffers (`m.ctype`, `m.cnodes`, `m.coords_*`, `faces`) as each migration
    //    phase completes to strictly prevent 2x memory duplication spikes.
    //
    // Preconditions:
    //  - `m` contains valid raw chunked mesh data with displacement tables (`cell_displ`, `node_displ`);
    //  - `dual_graph.faces` contains validated unique interior and boundary faces with CGNS local face IDs;
    //  - `pr` contains valid partitioning maps (`cell_target_rank`, `part2rank`, `rank2part`, `global_edge_cut`).
    //
    // Pipeline Overview (9 Deterministic Stages):
    //
    //  Step 1: Remote Target Rank Resolution
    //   - Identifies destination MPI ranks for remote `cell_b` instances on partition
    //     boundaries via a lightweight 1-round request-reply Alltoallv handshake.
    //
    //  Step 2: Volume Cell Redistribution (Owned Domain)
    //   - Dispatches cell types and node GID connectivity to their assigned target ranks
    //     computed by dKaMinPar; immediately frees raw cell arrays (`m.ctype`, `m.cnodes`).
    //   - Assembles local owned cells `[0, n_own)` in flat CSR format with zero heap fragmentation.
    //
    //  Step 3: Face Migration & Ghost-Layer Cell Detection
    //   - Routes local/boundary faces to their respective owner ranks.
    //   - Duplicates inter-domain cut faces ($msg_a$, $msg_b$) to both adjacent partition owners.
    //   - Detects all required ghost cells and sorts them strictly by `(donor_rank, cell_gid)`
    //     to guarantee contiguous halo slices in memory.
    //
    //  Step 4: Ghost Cell Topology Exchange
    //   - Queries donor ranks for element types and node connectivity lists for all
    //     ghost cells in the one-hop halo layer; packs them into flat ghost CSR tables.
    //
    //  Step 5: Selective Node Coordinate Migration
    //   - Extracts the exact minimal set of unique node GIDs needed for owned + ghost cells;
    //   - Queries original node owners via `m.node_displ` to fetch exact `(x, y, z)` coordinates;
    //   - Destructively frees raw coordinate storage (`m.my_node_coords_*`).
    //
    //  Step 6: Local Contiguous Renumbering & Cell CSR Construction
    //   - Maps global node IDs to local indices: owned nodes `[0, n_nodes_own)`, ghost-only `[n_nodes_own, n_nodes)`;
    //   - Renumbers owned cells to `[0, n_own)` and ghost cells to `[n_own, n_cells)`;
    //   - Builds final flattened `cell_nodes_offsets` and `cell_nodes` arrays.
    //
    //  Step 7: Canonical Face Reconstruction & CGNS Normal Orientation
    //   - Resolves face-to-cell connectivity (`face_owner` and `face_neigh`);
    //   - Reconstructs oriented face node lists using canonical CGNS tables (`kFaceTable`),
    //     strictly guaranteeing outward-pointing normals from `face_owner` toward `face_neigh`.
    //   - Sorts local faces lexicographically by `(owner, neigh)`.
    //
    //  Step 8: Symmetric Communication Map Assembly (Halo Exchange Engine)
    //   - Identifies active neighboring ranks (`nb_ranks`);
    //   - Builds contiguous `recv_offsets` and `recv_ghost_local` index maps;
    //   - Executes a reverse handshake to construct mirror `send_offsets` and `send_owned_local`
    //     maps, enabling zero-copy non-blocking MPI halo exchanges in the solver.
    //
    //  Step 9: Boundary Condition Patches, Global Deduplication & Sanity Verification
    //   - Groups boundary faces into individual BC patches using 2-pass flat CSR tables;
    //   - Computes global Bounding Box (`bbox_lo`, `bbox_hi`);
    //   - Calculates exact unique global face counts by deducting `pr.global_edge_cut` duplicates;
    //   - Runs full structure invariant validation (`meshpart_sane`).
    //
    // After this call, `mp` (MeshPart) contains a fully self-contained, cache-aligned,
    // rank-local subdomain ready for geometric metric calculation and direct parallel binary serialization.
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
