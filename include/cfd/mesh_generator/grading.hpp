#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#include "cfd/mesh_generator/config.hpp"

namespace cfd::mesh_generator {

class Grading {
public:
    // Computes normalized parametric coordinates xi_i in [0.0, 1.0] for i = 0 .. num_cells.
    // Result size is always (num_cells + 1), starting with 0.0 and ending with 1.0.
    [[nodiscard]] static std::vector<double> compute_distribution(std::size_t num_cells,
                                                                 GradingType type,
                                                                 double grading_param) {
        std::vector<double> dist(num_cells + 1, 0.0);
        dist[0] = 0.0;
        dist[num_cells] = 1.0;

        if (num_cells <= 1) {
            return dist;
        }

        const double N = static_cast<double>(num_cells);

        switch (type) {
            case GradingType::UNIFORM: {
                for (std::size_t i = 1; i < num_cells; ++i) {
                    dist[i] = static_cast<double>(i) / N;
                }
                break;
            }

            case GradingType::GEOMETRIC: {
                const double r = grading_param;
                if (std::abs(r - 1.0) < 1e-12) {
                    for (std::size_t i = 1; i < num_cells; ++i) {
                        dist[i] = static_cast<double>(i) / N;
                    }
                } else {
                    // Exact discrete geometric progression:
                    // cell_size[k] = cell_size[0] * alpha^k
                    // ratio = cell_size[N-1] / cell_size[0] = alpha^(N-1) => alpha = r^(1 / (N - 1))
                    const double alpha = std::pow(r, 1.0 / (N - 1.0));
                    const double denom = 1.0 - std::pow(alpha, N);
                    for (std::size_t i = 1; i < num_cells; ++i) {
                        dist[i] = (1.0 - std::pow(alpha, static_cast<double>(i))) / denom;
                    }
                }
                break;
            }

            case GradingType::TANH_START: {
                // Clustering at s = 0 (boundary layer at the beginning)
                const double delta = grading_param;
                const double tanh_d = std::tanh(delta);
                for (std::size_t i = 1; i < num_cells; ++i) {
                    const double s = static_cast<double>(i) / N;
                    dist[i] = 1.0 - std::tanh(delta * (1.0 - s)) / tanh_d;
                }
                break;
            }

            case GradingType::TANH_END: {
                // Clustering at s = 1 (boundary layer at the end)
                const double delta = grading_param;
                const double tanh_d = std::tanh(delta);
                for (std::size_t i = 1; i < num_cells; ++i) {
                    const double s = static_cast<double>(i) / N;
                    dist[i] = std::tanh(delta * s) / tanh_d;
                }
                break;
            }

            case GradingType::TANH_BOTH: {
                // Symmetric clustering towards both s = 0 and s = 1
                const double delta = grading_param;
                const double tanh_d = std::tanh(delta);
                for (std::size_t i = 1; i < num_cells; ++i) {
                    const double s = static_cast<double>(i) / N;
                    dist[i] = 0.5 * (1.0 + std::tanh(delta * (2.0 * s - 1.0)) / tanh_d);
                }
                break;
            }
        }

        return dist;
    }
};

} // namespace cfd::mesh_generator