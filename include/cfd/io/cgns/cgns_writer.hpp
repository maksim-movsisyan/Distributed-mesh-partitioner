#pragma once

#include <string>
#include <mpi.h>

#include "cfd/mesh/raw_mesh.hpp"

namespace cfd::io::cgns {

// Writes a 3D unstructured RawMesh into a parallel CGNS file (HDF5).
// Compatible with standard CFD post-processors (ParaView, Tecplot).
void write_cgns_parallel(const std::string& path, const mesh::RawMesh& m);

} // namespace cfd::io::cgns