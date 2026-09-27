#pragma once

#include <mpi.h>
#include <string>

#include "cfd/mesh_generator/config.hpp"
#include "cfd/mesh_generator/topology.hpp"

namespace cfd::mesh_generator {

// Writes the complete unstructured mesh to CGNS collectively across all MPI ranks
void write(const std::string& filepath,
           const GeneratorConfig& config,
           const MacroTopology& topo,
           MPI_Comm comm);

} // namespace cfd::mesh_generator