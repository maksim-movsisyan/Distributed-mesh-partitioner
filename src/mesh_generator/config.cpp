#include "cfd/mesh_generator/config.hpp"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>
#include <toml++/toml.hpp>

namespace cfd::mesh_generator {

namespace {

[[noreturn]] void fail(const MPI_Comm comm, const std::string& what) {
    mpi::fatal(comm, "config: " + what);
    std::abort();
}

[[nodiscard]] std::string broadcast_file_content(const std::string& path, const MPI_Comm comm) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    std::string content;
    int content_size = 0;
    bool read_success = false;

    if (rank == 0) {
        std::ifstream file(path);
        if (file.is_open()) {
            std::stringstream ss;
            ss << file.rdbuf();
            content = ss.str();
            content_size = static_cast<int>(content.size());
            read_success = true;
        }
    }

    // Broadcast file open status
    int status_int = read_success ? 1 : 0;
    MPI_Bcast(&status_int, 1, MPI_INT, 0, comm);

    if (status_int == 0) {
        fail(comm, "cannot open file '" + path + "' on Rank 0");
    }

    // Broadcast string length, allocate buffer on workers, broadcast string bytes
    MPI_Bcast(&content_size, 1, MPI_INT, 0, comm);
    if (rank != 0) {
        content.resize(static_cast<std::size_t>(content_size));
    }
    MPI_Bcast(content.data(), content_size, MPI_CHAR, 0, comm);

    return content;
}

[[nodiscard]] toml::table parse_in_memory_or_die(const std::string& content,
                                                const std::string& source_path,
                                                const MPI_Comm comm) {
    try {
        return toml::parse(content, source_path);
    } catch (const toml::parse_error& err) {
        fail(comm, "cannot parse '" + source_path + "': " + std::string(err.description()));
    }
}

void check_allowed_keys(const toml::table& t,
                        const std::initializer_list<const char*> allowed,
                        const std::string& ctx, const MPI_Comm comm) {
    for (const auto& [k, v] : t) {
        (void)v;
        const std::string_view name = k.str();
        bool known = false;
        for (const char* a : allowed) {
            if (name == std::string_view(a)) {
                known = true;
                break;
            }
        }
        if (!known) {
            fail(comm, ctx + ": unknown key '" + std::string(name) + "'");
        }
    }
}

} // namespace

GeneratorConfig parse_generator_config(const std::string& path, const MPI_Comm comm) {
    GeneratorConfig cfg;

    const std::string raw_content = broadcast_file_content(path, comm);
    const toml::table root = parse_in_memory_or_die(raw_content, path, comm);
    
    check_allowed_keys(root, {"mesh", "blocks", "boundaries"}, "'" + path + "'", comm);

    // -------------------------------------------------------------------------
    // 1. [mesh] section (name, scale, vertices)
    // -------------------------------------------------------------------------
    {
        const auto* mesh_node = root["mesh"].as_table();
        if (!mesh_node) {
            fail(comm, "missing mandatory [mesh] section in '" + path + "'");
        }
        check_allowed_keys(*mesh_node, {"name", "scale", "vertices"}, "[mesh]", comm);

        cfg.name = (*mesh_node)["name"].value_or(std::string("Mesh"));

        if (auto scale = (*mesh_node)["scale"].value<double>()) {
            cfg.scale = *scale;
            if (cfg.scale <= 0.0) {
                fail(comm, "[mesh].scale must be strictly positive, got " + std::to_string(cfg.scale));
            }
        } else {
            cfg.scale = 1.0;
        }

        const auto* vertices_node = (*mesh_node)["vertices"].as_array();
        if (!vertices_node) {
            fail(comm, "missing mandatory 'vertices' array inside [mesh] section");
        }
        if (vertices_node->empty()) {
            fail(comm, "[mesh].vertices array must not be empty");
        }

        cfg.vertices.reserve(vertices_node->size());
        for (std::size_t i = 0; i < vertices_node->size(); ++i) {
            const auto* v_arr = (*vertices_node)[i].as_array();
            if (!v_arr || v_arr->size() != 3) {
                fail(comm, "[mesh].vertices [" + std::to_string(i) + "] must be an array of 3 numbers [x, y, z]");
            }

            auto x = (*v_arr)[0].value<double>();
            auto y = (*v_arr)[1].value<double>();
            auto z = (*v_arr)[2].value<double>();

            if (!x || !y || !z) {
                fail(comm, "[mesh].vertices [" + std::to_string(i) + "] coordinates must be numeric");
            }

            cfg.vertices.emplace_back(*x * cfg.scale, *y * cfg.scale, *z * cfg.scale);
        }
    }

    // -------------------------------------------------------------------------
    // 2. [[blocks]] array
    // -------------------------------------------------------------------------
    {
        const auto* blocks_node = root["blocks"].as_array();
        if (!blocks_node || blocks_node->empty()) {
            fail(comm, "missing or empty [[blocks]] list in '" + path + "'");
        }

        cfg.blocks.reserve(blocks_node->size());
        for (std::size_t b = 0; b < blocks_node->size(); ++b) {
            const auto* b_tbl = (*blocks_node)[b].as_table();
            if (!b_tbl) {
                fail(comm, "block [" + std::to_string(b) + "] must be a table");
            }
            check_allowed_keys(*b_tbl, {"name", "vertices", "cells", "grading_type", "grading"},
                               "[[blocks]] [" + std::to_string(b) + "]", comm);

            MacroBlock block;
            block.name = (*b_tbl)["name"].value_or(std::string("block_") + std::to_string(b));

            // Macro-vertices (8 indices conforming to HEXA_8)
            const auto* v_indices = (*b_tbl)["vertices"].as_array();
            if (!v_indices || v_indices->size() != 8) {
                fail(comm, "block '" + block.name + "': 'vertices' must contain exactly 8 indices");
            }
            for (std::size_t i = 0; i < 8; ++i) {
                auto idx = (*v_indices)[i].value<int64_t>();
                if (!idx || *idx < 0) {
                    fail(comm, "block '" + block.name + "': vertex index [" + std::to_string(i) + "] must be a non-negative integer");
                }
                block.vertices[i] = static_cast<std::size_t>(*idx);
            }

            // Cell counts [Nx, Ny, Nz]
            const auto* c_arr = (*b_tbl)["cells"].as_array();
            if (!c_arr || c_arr->size() != 3) {
                fail(comm, "block '" + block.name + "': 'cells' must contain 3 integers [Nx, Ny, Nz]");
            }
            for (std::size_t i = 0; i < 3; ++i) {
                auto n = (*c_arr)[i].value<int64_t>();
                if (!n || *n <= 0) {
                    fail(comm, "block '" + block.name + "': cell count along axis " + std::to_string(i) + " must be strictly positive");
                }
                block.cells[i] = static_cast<std::size_t>(*n);
            }

            // Grading types (optional, default uniform)
            if (const auto* gt_arr = (*b_tbl)["grading_type"].as_array()) {
                if (gt_arr->size() != 3) {
                    fail(comm, "block '" + block.name + "': 'grading_type' must contain exactly 3 strings");
                }
                for (std::size_t i = 0; i < 3; ++i) {
                    auto gt_str = (*gt_arr)[i].value<std::string_view>();
                    if (!gt_str) {
                        fail(comm, "block '" + block.name + "': invalid string in 'grading_type' [" + std::to_string(i) + "]");
                    }
                    block.grading_type[i] = parse_grading_type(*gt_str, comm);
                }
            } else {
                block.grading_type.fill(GradingType::UNIFORM);
            }

            // Grading strengths (optional, default 1.0)
            if (const auto* g_arr = (*b_tbl)["grading"].as_array()) {
                if (g_arr->size() != 3) {
                    fail(comm, "block '" + block.name + "': 'grading' must contain exactly 3 numbers");
                }
                for (std::size_t i = 0; i < 3; ++i) {
                    auto g_val = (*g_arr)[i].value<double>();
                    if (!g_val) {
                        fail(comm, "block '" + block.name + "': 'grading' values must be numeric");
                    }
                    block.grading[i] = *g_val;
                }
            } else {
                block.grading.fill(1.0);
            }

            cfg.blocks.push_back(std::move(block));
        }
    }

    // -------------------------------------------------------------------------
    // 3. [[boundaries]] array
    // -------------------------------------------------------------------------
    if (const auto* bnd_node = root["boundaries"].as_array()) {
        cfg.boundaries.reserve(bnd_node->size());
        for (std::size_t p = 0; p < bnd_node->size(); ++p) {
            const auto* p_tbl = (*bnd_node)[p].as_table();
            if (!p_tbl) {
                fail(comm, "boundary patch [" + std::to_string(p) + "] must be a table");
            }
            check_allowed_keys(*p_tbl, {"name", "faces"}, "[[boundaries]] [" + std::to_string(p) + "]", comm);

            BoundaryPatch patch;
            patch.name = (*p_tbl)["name"].value_or(std::string("patch_") + std::to_string(p));

            const auto* f_list = (*p_tbl)["faces"].as_array();
            if (!f_list || f_list->empty()) {
                fail(comm, "boundary patch '" + patch.name + "': 'faces' array must not be empty");
            }

            patch.faces.reserve(f_list->size());
            for (std::size_t fi = 0; fi < f_list->size(); ++fi) {
                const auto* f_arr = (*f_list)[fi].as_array();
                if (!f_arr || f_arr->size() != 4) {
                    fail(comm, "boundary patch '" + patch.name + "' face [" + std::to_string(fi) + "] must specify 4 vertex indices");
                }

                MacroFace face;
                for (std::size_t vi = 0; vi < 4; ++vi) {
                    auto v_idx = (*f_arr)[vi].value<int64_t>();
                    if (!v_idx || *v_idx < 0) {
                        fail(comm, "patch '" + patch.name + "' face [" + std::to_string(fi) + "] vertex [" + std::to_string(vi) + "] must be non-negative integer");
                    }
                    face.vertices[vi] = static_cast<std::size_t>(*v_idx);
                }
                patch.faces.push_back(face);
            }
            cfg.boundaries.push_back(std::move(patch));
        }
    }

    // -------------------------------------------------------------------------
    // 4. Contract Validations
    // -------------------------------------------------------------------------
    const std::size_t num_vertices = cfg.vertices.size();

    // Check blocks
    for (const auto& block : cfg.blocks) {
        for (std::size_t i = 0; i < 8; ++i) {
            if (block.vertices[i] >= num_vertices) {
                fail(comm, "block '" + block.name + "' references vertex index " +
                           std::to_string(block.vertices[i]) + " out of bounds [0, " +
                           std::to_string(num_vertices) + ")");
            }
        }

        // Positive Jacobian Invariant: Det[(xi x eta) . zeta] > 0
        const Vec3& v0 = cfg.vertices[block.vertices[0]];
        const Vec3& v1 = cfg.vertices[block.vertices[1]];
        const Vec3& v3 = cfg.vertices[block.vertices[3]];
        const Vec3& v4 = cfg.vertices[block.vertices[4]];

        const Vec3 xi   = v1 - v0;
        const Vec3 eta  = v3 - v0;
        const Vec3 zeta = v4 - v0;

        const double jacobian = xi.cross(eta).dot(zeta);
        if (jacobian <= 1e-14) {
            fail(comm, "block '" + block.name + "' violates positive Jacobian contract! Det[(xi x eta) . zeta] <= 0 (" +
                       std::to_string(jacobian) + "). Verify vertex order conforms to HEXA_8");
        }

        // Grading strengths verification
        for (std::size_t d = 0; d < 3; ++d) {
            if (block.grading_type[d] == GradingType::GEOMETRIC && block.grading[d] <= 0.0) {
                fail(comm, "block '" + block.name + "' direction " + std::to_string(d) +
                           " has geometric grading <= 0 (" + std::to_string(block.grading[d]) + ")");
            }
            if ((block.grading_type[d] == GradingType::TANH_START ||
                 block.grading_type[d] == GradingType::TANH_END   ||
                 block.grading_type[d] == GradingType::TANH_BOTH) && block.grading[d] <= 0.0) {
                fail(comm, "block '" + block.name + "' direction " + std::to_string(d) +
                           " has tanh grading <= 0 (" + std::to_string(block.grading[d]) + ")");
            }
        }
    }

    // Check boundary faces
    for (const auto& patch : cfg.boundaries) {
        for (std::size_t fi = 0; fi < patch.faces.size(); ++fi) {
            const auto& face = patch.faces[fi];
            for (std::size_t vi = 0; vi < 4; ++vi) {
                if (face.vertices[vi] >= num_vertices) {
                    fail(comm, "patch '" + patch.name + "' face [" + std::to_string(fi) +
                               "] references vertex index " + std::to_string(face.vertices[vi]) +
                               " out of bounds [0, " + std::to_string(num_vertices) + ")");
                }
            }

            const Vec3& f0 = cfg.vertices[face.vertices[0]];
            const Vec3& f1 = cfg.vertices[face.vertices[1]];
            const Vec3& f3 = cfg.vertices[face.vertices[3]];
            const Vec3 normal = (f1 - f0).cross(f3 - f0);
            if (normal.norm() <= 1e-14) {
                fail(comm, "patch '" + patch.name + "' face [" + std::to_string(fi) +
                           "] has zero surface area / degenerate vertices");
            }
        }
    }

    return cfg;
}

} // namespace cfd::mesh_generator