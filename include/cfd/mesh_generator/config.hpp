#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <mpi.h>
#include "cfd/mpi/log.hpp"


namespace cfd::mesh_generator {

struct Vec3 {
    double x{0.0};
    double y{0.0};
    double z{0.0};

    constexpr Vec3() noexcept = default;
    constexpr Vec3(double x_, double y_, double z_) noexcept : x(x_), y(y_), z(z_) {}

    constexpr Vec3 operator+(const Vec3& o) const noexcept {
        return {x + o.x, y + o.y, z + o.z};
    }
    constexpr Vec3 operator-(const Vec3& o) const noexcept {
        return {x - o.x, y - o.y, z - o.z};
    }
    constexpr Vec3 operator*(double s) const noexcept {
        return {x * s, y * s, z * s};
    }
    constexpr Vec3 operator/(double s) const noexcept {
        return {x / s, y / s, z / s};
    }

    [[nodiscard]] constexpr double dot(const Vec3& o) const noexcept {
        return x * o.x + y * o.y + z * o.z;
    }

    [[nodiscard]] constexpr Vec3 cross(const Vec3& o) const noexcept {
        return {
            y * o.z - z * o.y,
            z * o.x - x * o.z,
            x * o.y - y * o.x
        };
    }

    [[nodiscard]] double norm() const noexcept {
        return std::sqrt(dot(*this));
    }
};

enum class GradingType : std::uint8_t {
    UNIFORM    = 0,
    GEOMETRIC  = 1,
    TANH_START = 2,
    TANH_END   = 3,
    TANH_BOTH  = 4
};

[[nodiscard]] inline std::string_view to_string(GradingType type) noexcept {
    switch (type) {
        case GradingType::UNIFORM:    return "uniform";
        case GradingType::GEOMETRIC:  return "geometric";
        case GradingType::TANH_START: return "tanh_start";
        case GradingType::TANH_END:   return "tanh_end";
        case GradingType::TANH_BOTH:  return "tanh_both";
    }
    return "unknown";
}

[[nodiscard]] inline GradingType parse_grading_type(std::string_view str, MPI_Comm comm) {
    if (str == "uniform")    return GradingType::UNIFORM;
    if (str == "geometric")  return GradingType::GEOMETRIC;
    if (str == "tanh_start") return GradingType::TANH_START;
    if (str == "tanh_end")   return GradingType::TANH_END;
    if (str == "tanh_both")  return GradingType::TANH_BOTH;
    mpi::fatal(comm, "Unknown grading type: " + std::string(str));
    return static_cast<GradingType>(0);
}

struct MacroBlock {
    std::string name;
    std::array<std::size_t, 8> vertices{};
    std::array<std::size_t, 3> cells{};         // Nx, Ny, Nz
    std::array<GradingType, 3> grading_type{};  // Along xi, eta, zeta
    std::array<double, 3> grading{};            // Strengths / ratios
};

struct MacroFace {
    std::array<std::size_t, 4> vertices{}; // Ordered counter-clockwise (outward normal)
};

struct BoundaryPatch {
    std::string name;
    std::vector<MacroFace> faces;
};

struct GeneratorConfig {
    std::string name{"Mesh"};
    double scale{1.0};
    std::vector<Vec3> vertices;
    std::vector<MacroBlock> blocks;
    std::vector<BoundaryPatch> boundaries;
};

GeneratorConfig parse_generator_config(const std::string& path, const MPI_Comm comm);

} // namespace cfd::mesh_generator