# PARALLEL_CFD — Stage 1: Parallel Mesh Preprocessing

High-performance parallel preprocessing toolkit for an unstructured-mesh CFD solver (compressible viscous Navier–Stokes equations). 

`mesh_partition` reads a CGNS mesh in parallel (PCGNS / parallel HDF5) across all MPI ranks, builds face–cell connectivity in a distributed fashion, partitions the dual graph (dKaMinPar), constructs a one-hop ghost layer (face neighbors + vertex-sharing neighbors for least-squares gradients), computes finite-volume geometric metrics, applies SFC/RCM cache-locality reordering, and exports a single topology-aware Parallel HDF5 container. Additionally, it generates a boundary-condition TOML configuration template and VTU/PVTU files for partition visualization.

`solver_mesh_check` loads the exported HDF5 container using the exact solver pattern (hyperslab collective MPI-IO slicing per rank), runs an exhaustive sanity audit (halo communication graph handshake, volume metrics, face normal orientations, BC coverage), and logs global domain diagnostics.

---

## Project Layout

```text
include/cfd/              Public headers (clean interface separation)
  ├── mpi/                MPI wrappers, collective reductions, structured logging
  ├── core/               Fundamental types (LocalIndex, GlobalIndex, precision)
  ├── mesh/               MeshPart SoA, geometric metrics, ghost layer, reordering
  ├── partition/          dKaMinPar wrapper + Space-Filling Curve rank mapping
  ├── io/cgns/            Parallel CGNS reader (PCGNS cgp_* API)
  ├── io/solver_mesh/     Parallel HDF5 mesh serializer and loader
  └── io/vtk/             Parallel VTU/PVTU export for ParaView visualization
src/                      Library implementation sources mirroring include/
apps/                     Executables (mesh_partition, solver_mesh_check)
tests/                    Unit & integration tests (API & partitioner smoke tests)
mesh/                     Local test meshes (git-ignored)
```

### CMake Targets & Libraries
- `cfd_mpi` — MPI communication wrappers, topology logging, and assertions.
- `cfd_mesh` — Distributed mesh topology, halo exchange layers, geometric metrics, SFC/RCM reordering.
- `cfd_partition` — dKaMinPar dual-graph partitioner and SFC part-to-rank topological mapper.
- `cfd_cgns_io` — Collective parallel CGNS reader (`cgp_*` API).
- `cfd_solver_mesh_io` — High-throughput Parallel HDF5 writer/loader (`H5FD_MPIO_COLLECTIVE`).
- `cfd_vtk_io` — Buffered parallel VTU/PVTU mesh exporters.

---

## Prerequisites

- **Compiler:** GCC >= 11 (C++20 standard required)
- **MPI:** OpenMPI >= 4.1 or MPICH
- **I/O Libraries:** Parallel HDF5 (with MPI-IO support), CGNS >= 4.0 (built with Parallel HDF5)
- **Partitioning & Concurrency:** KaMinPar / dKaMinPar, Intel oneTBB
- **Build System:** CMake >= 3.28, Ninja

For a step-by-step installation guide (WSL2 Ubuntu / Linux HPC clusters), see **[INSTALL-AND-SETUP-EN.md](INSTALL-AND-SETUP-EN.md)**.

**If something went wrong with MIXED section meshes read [INSTALL-AND-SETUP-EN.md](INSTALL-AND-SETUP-EN.md) CGNS warnings first**

---

## Build

```bash
# Configure build preset (Release by default, 32-bit global indexing)
cmake --preset release

# (Optional) Build with 64-bit global indexing for billion-cell meshes:
# cmake --preset release -DCFD_INDEX_64BIT=ON

# Compile with Ninja
cmake --build --preset release -j$(nproc)
```

*Build artifacts will appear in `build/release/`.*

---

## Usage

### 1. Partition and Preprocess Mesh
```bash
mkdir -p out
mpirun -np 4 build/release/mesh_partition mesh/DesktopTest.cgns out/mesh.h5 \
    --verbose \
    --bc out/bc.toml \
    --vtu out/vtu \
    --reorder RCM
```

**Key CLI Options:**
- `--verbose / -v`: Print stage timings and partitioning balance.
- `--bc <file>`: Auto-generate boundary conditions TOML template.
- `--vtu <dir>`: Export partitioned domain to ParaView VTU/PVTU format.
- `--reorder <method>`: Local cache reordering strategy (`NONE`, `RCM`, `HILBERT_SFC`).

### 2. Verify Exported Mesh (Solver Roundtrip Check)
```bash
mpirun -np 4 build/release/solver_mesh_check out/mesh.h5 \
    --verbose \
    --dump-vtu out/vtu/loaded
```

### Visualizing in ParaView
- Open `out/vtu/mesh.pvtu` to inspect the volume mesh. Color by:
  - `rank` — Subdomain spatial decomposition.
  - `local_id` — Cache memory order (shows the Hilbert SFC curve path).
  - `volume` — Cell finite-volume distribution.
- Open `out/vtu/mesh_bnd.pvtu` to inspect boundary patches. Color by `patch_id` to verify boundary condition assignments.

---

## Output File Format (v2, Topology-Aware Flat Parallel Layout)

Global metadata attributes and per-partition offset tables (`cell_offsets`, `face_offsets`, `node_offsets`, `comm_nb_offsets`, `part2rank`, `rank2part`), followed by contiguous 1D/2D SoA datasets.

<details>
<summary><b>Click to expand HDF5 Layout Tree</b></summary>

```text
mesh.h5
├── /metadata/               (Root Group Attributes)
│   ├── nprocs               (int32 / int      - Total MPI partition count)
│   ├── n_cells_global       (uint64           - Total domain owned volume cells)
│   ├── n_faces_global       (uint64           - Total domain unique interior + boundary faces)
│   ├── n_bfaces_global      (uint64           - Total domain boundary faces)
│   ├── n_nodes_global       (uint64           - Total domain unique mesh vertices)
│   ├── bbox_lo              [3] (float64      - Domain minimum bounding box [x_min, y_min, z_min])
│   └── bbox_hi              [3] (float64      - Domain maximum bounding box [x_max, y_max, z_max])
│
├── /partition/              (Partition Topology & Hyperslab Offset Tables for Collective MPI-IO)
│   ├── cell_offsets         [nprocs + 1] (uint64 - Cumulative cell partition slice offsets)
│   ├── cell_nodes_offsets   [nprocs + 1] (uint64 - Cumulative cell connectivity CSR offsets)
│   ├── face_offsets         [nprocs + 1] (uint64 - Cumulative face partition slice offsets)
│   ├── face_nodes_offsets   [nprocs + 1] (uint64 - Cumulative face connectivity CSR offsets)
│   ├── node_offsets         [nprocs + 1] (uint64 - Cumulative node partition slice offsets)
│   ├── comm_nb_offsets      [nprocs + 1] (uint64 - Cumulative MPI neighbor count offsets)
│   ├── part2rank            [nprocs]     (int32  - Spatial partition ID to MPI rank mapping)
│   └── rank2part            [nprocs]     (int32  - MPI rank to spatial partition ID mapping)
│
├── /cells/                  (Volume Cells: Owned [0, n_own) followed by Halo Ghosts [n_own, n_cells))
│   ├── type                 [n_cells_total]    (uint8        - CellType enum: TETRA, HEXA, PRISM, PYRA)
│   ├── gid                  [n_cells_total]    (GlobalIndex  ──► int32 / int64 - Global cell ID)
│   ├── donor                [n_cells_total]    (int32        - Owner rank for halo ghost cells, -1 for owned)
│   ├── volume               [n_cells_total]    (float64      - Precomputed cell volume [m^3])
│   ├── centroid             [n_cells_total, 3] (float64      - Cell centroid coordinates [x, y, z])
│   ├── nodes_offsets        [n_cells_total + 1](LocalIndex   ──► int32 / int64 - CSR node index offsets)
│   └── nodes                [total_cell_nodes] (LocalIndex   ──► int32 / int64 - CSR cell-to-node connectivity)
│
├── /faces/                  (Sorted monotonically by (owner, neigh) for L1/L2 cache locality)
│   ├── owner                [n_faces_total]    (LocalIndex   ──► int32 / int64 - Owned cell index [0, n_own))
│   ├── neigh                [n_faces_total]    (LocalIndex   ──► int32 / int64 - Neighbor cell index, or -1 for boundary)
│   ├── patch                [n_faces_total]    (PatchId      ──► int32 - Boundary patch ID, or -1 for interior)
│   ├── type                 [n_faces_total]    (uint8        - Face CellType enum: TRI, QUAD)
│   ├── area                 [n_faces_total]    (float64      - Face surface area magnitude [m^2])
│   ├── normal               [n_faces_total, 3] (float64      - Outward unit normal vector [nx, ny, nz] from owner)
│   ├── centroid             [n_faces_total, 3] (float64      - Face centroid coordinates [x, y, z])
│   ├── nodes_offsets        [n_faces_total + 1](LocalIndex   ──► int32 / int64 - CSR node index offsets)
│   └── nodes                [total_face_nodes] (LocalIndex   ──► int32 / int64 - CSR face-to-node connectivity)
│
├── /nodes/                  (Local Unique Node Pool)
│   ├── coords               [n_nodes_total, 3] (float64      - Nodal coordinates [x, y, z])
│   └── gid                  [n_nodes_total]    (GlobalIndex  ──► int32 / int64 - Global node ID)
│
├── /comm/                   (Zero-Copy Halo Exchange Communication Graphs for Solver)
│   ├── nb_ranks             [total_neighbors]  (int32        - Neighboring MPI rank IDs)
│   ├── send_offsets         [total_nb + nprocs](LocalIndex   ──► int32 / int64 - CSR offsets into send_owned_cells)
│   ├── recv_offsets         [total_nb + nprocs](LocalIndex   ──► int32 / int64 - CSR offsets into recv_ghost_cells)
│   ├── send_owned_cells     [total_send_cells] (LocalIndex   ──► int32 / int64 - Local owned cell indices to pack & send)
│   └── recv_ghost_cells     [total_recv_cells] (LocalIndex   ──► int32 / int64 - Local ghost cell indices to unpack & recv)
│
└── /patches/                (Boundary Condition Definitions & Surface Face CSR Maps)
    ├── names                [n_patches]        (fixed-string char[64] - Boundary patch names from CGNS)
    ├── cgns_types           [n_patches]        (fixed-string char[64] - CGNS BC types: BCWall, BCInflow, etc.)
    ├── patch_face_offsets   [n_offsets_total]  (LocalIndex   ──► int32 / int64 - Rank-local CSR face offsets per patch)
    └── patch_faces          [total_patch_faces](LocalIndex   ──► int32 / int64 - Local face indices belonging to patch)
```

</details>

Data of rank $r$ with topological partition ID $p = \text{rank2part}[r]$ occupies the continuous range `[offset[p], offset[p+1])` across all hyperslabs. All ranks perform non-interfering I/O using collective MPI-IO (`H5FD_MPIO_COLLECTIVE`).

---

## Status

- Fully verified on unstructured hexahedral, tetrahedral, and prism meshes on 1, 2, 4, 8+ MPI ranks.
- Zero-copy halo exchange communication handshake verified across all neighboring partitions.
- Exact total domain volume and strict outward face normal contracts confirmed.