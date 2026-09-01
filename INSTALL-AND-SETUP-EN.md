<details>
<summary><b>CFD Environment Setup Guide (Click to expand)</b></summary>

# Environment Setup for CFD Project (Ubuntu 22.04 in WSL2)

Target toolchain: GCC + OpenMPI + Parallel HDF5 + CGNS (Parallel I/O) + KaMinPar, C++20, CMake.  
All commands below are executed once during initial setup.

---

## Step 1. Installing WSL2 (Windows Host, PowerShell AS ADMINISTRATOR)

```powershell
wsl --install -d Ubuntu-22.04
```

Then:
1. Reboot the PC.
2. Upon first launch of Ubuntu, set up your username and password (remember the password — it is required for `sudo` inside Ubuntu).
3. Verification (in PowerShell): `wsl -l -v` — should display `Ubuntu-22.04` with `VERSION 2`.

---

## Step 2. Base Packages (inside Ubuntu)

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
    build-essential \
    cmake \
    git \
    ninja-build \
    pkg-config \
    openmpi-bin libopenmpi-dev \
    libhdf5-openmpi-dev hdf5-tools \
    libtbb-dev \
    clangd clang-format
```

Notes:
- `libhdf5-openmpi-dev` is HDF5 built with parallel I/O (PHDF5) over OpenMPI. For development, THERE IS NO NEED TO COMPILE HDF5 FROM SOURCE — the package is already parallel. Compiling from source is only needed on an HPC cluster (instructions in Step 5).
- `libtbb-dev` is a core dependency for KaMinPar shared-memory parallelism.
- `hdf5-tools` provides CLI utilities (`h5dump`, `h5ls`) for inspecting binary output files.

Quick verification that parallel HDF5 is present:
```bash
h5pcc -show        # Should display the wrapper configuration over mpicc
mpicc --version    # GCC 11.x on Ubuntu 22.04, C++20 supported (g++ 11)
```

---

## Step 3. CGNS with Parallel I/O (Build from Source — Mandatory)

The standard `libcgns-dev` package from apt is NOT suitable: it is built without parallel I/O. Build it from source:

```bash
cd $HOME
git clone --depth 1 --branch v4.5.2 https://github.com/CGNS/CGNS.git cgns-src-4-5-2
cmake -B cgns-src-4-5-2/build -S cgns-src-4-5-2 -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=mpicc \
    -DCGNS_ENABLE_HDF5=ON \
    -DCGNS_ENABLE_PARALLEL=ON \
    -DHDF5_NEED_MPI=ON \
    -DHDF5_PREFER_PARALLEL=ON \
    -DCGNS_ENABLE_LFS=ON \
    -DCGNS_BUILD_SHARED=OFF \
    -DCGNS_ENABLE_FORTRAN=OFF \
    -DCGNS_ENABLE_TESTS=OFF \
    -DCGNS_ENABLE_64BIT=OFF \
    -DCMAKE_INSTALL_PREFIX=$HOME/local/cgns-4-5-2
    #-DCMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu/hdf5/openmpi
cmake --build cgns-src-4-5-2/build -j$(nproc)
cmake --install cgns-src-4-5-2/build
```

Key flags:
- `CGNS_ENABLE_PARALLEL=ON` — enables PCGNS (`cgp_*` functions, built with MPI);
- `HDF5_PREFER_PARALLEL=ON` — forces CMake to select parallel HDF5 (otherwise the serial variant is silently chosen, causing link failure);
- `CMAKE_C_COMPILER=mpicc` — CGNS must be compiled using the MPI compiler wrapper;
- `-DCMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu/hdf5/openmpi` — points directly to the OpenMPI HDF5 installation paths in Ubuntu;
- *(Optional)* `-DCGNS_ENABLE_64BIT=ON` — enable if working with large meshes (> 2 billion elements/nodes).

`WARNING-1` : IF YOUR MESH CONTAINS MIXED SECTIONS, THEN -DCGNS_ENABLE_64BIT=OFF/ON - MUST BE CONSISTENT WITH FILE INTEGER PRECSISION, OTHERWISE THE CGNS-READER FALL BACK TO RAM HEAVY READ (IT WILL WORK BUT IT IS SLOWER AND CONTRIBUTE MORE RAM).

`WARNING-2` : IF YOUR MESH CONTAINS MIXED SECTIONS, THEN YOU FILE VERISION (FILE, NOT CGNS) MUST BE MORE OR EQUAL 4.*, OTHERWISE THE CGNS-READER FALL BACK TO RAM HEAVY READ (IT WILL WORK BUT IT IS SLOWER AND CONTRIBUTE MORE RAM). TO UPDATE OLD FILE VERSION USE CGNS OFFICIAL UITILITIES:
```bash
~/local/cgns-4-5-2/bin/cgnsconvert -f -h ~/cfd/mesh-partitioner/mesh/sphere.cgns ~/cfd/mesh-partitioner/mesh/sphere_v4.cgns #convert to h5
~/local/cgns-4-5-2/bin/cgnsupdate ~/cfd/mesh-partitioner/mesh/sphere_v4.cgns #update version
```

Verification: The CMake configuration summary must display `Parallel IO: ON`.

---

## Step 4. KaMinPar (Parallel Partitioner)

```bash
cd $HOME
git clone --recursive https://github.com/KaHIP/KaMinPar.git
cmake -B KaMinPar/build -S KaMinPar \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$HOME/local/kaminpar
cmake --build KaMinPar/build -j$(nproc)
cmake --install KaMinPar/build
```

`--recursive` is mandatory: KaMinPar pulls external submodules (`growt`, `fmt`, etc.).

---

## Step 5. (Cluster Reference Only) Parallel HDF5 from Source

On an HPC Linux cluster where `libhdf5-openmpi-dev` is not available:

```bash
wget https://github.com/HDFGroup/hdf5/releases/download/hdf5_1.14.3/hdf5-1.14.3.tar.gz
tar xf hdf5-1.14.3.tar.gz && cd hdf5-1.14.3
./configure --enable-parallel --enable-shared --with-pic CC=mpicc \
    --prefix=$HOME/local/hdf5
make -j$(nproc) && make install
```

Afterwards, build CGNS with `-DCMAKE_PREFIX_PATH=$HOME/local/hdf5`.

---

## Step 6. Environment Variables (in ~/.bashrc inside Ubuntu)

```bash
export PATH=$HOME/local/cgns/bin:$HOME/local/kaminpar/bin:$PATH
export CMAKE_PREFIX_PATH=$HOME/local/cgns:$HOME/local/kaminpar:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=$HOME/local/cgns/lib:$HOME/local/kaminpar/lib:$HOME/local/kaminpar/lib64:$LD_LIBRARY_PATH
```

After updating: `source ~/.bashrc`.

---

## Important Note on Project Location

- Source code and build artifacts must reside inside the native WSL filesystem (e.g., `~/cfd`), **NOT** under `/mnt/c/...` — builds and runs on `/mnt/c` are significantly slower (Windows filesystem is proxied via 9P).
- The mesh file is accessible from WSL via:
  `/mnt/c/Users/<Username>/Desktop/...`
  (it can be read directly from there or copied into `~/cfd/mesh/` for faster I/O).

---

## Quick Environment Self-Check (after all steps)

```bash
mpirun -np 2 hostname                       # MPI is functional (2 lines of output)
h5pcc -show                                 # Parallel HDF5 is configured
ls $HOME/local/cgns/lib                     # libcgns.a is present
ls $HOME/local/kaminpar/lib                 # KaMinPar libraries/binaries are present
```

</details>
