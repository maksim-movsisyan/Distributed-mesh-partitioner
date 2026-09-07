// Cell and face geometry plus a self-check of the canonical face tables.
#pragma once

#include "cfd/mesh/localmesh.hpp"

namespace cfd::mesh {

// Self-check of canonical tables on reference elements (TET/PYRA/PRISM/HEXA):
// verifies positive volume, outward normals for all faces, and orientation inversion.
bool validate_face_tables();

void compute_mesh_geometry(MeshPart& mp);

} //namespace cfd::mesh