#ifndef GAUSS_NEWTON_H
#define GAUSS_NEWTON_H

#include <vector>

struct GNConfig {
    double learning_rate; // step scale applied to delta (default 1.0)
    double damping;       // LM damping term (added to diagonal)
    double l2_reg;        // L2 regularization lambda
    int max_iters;
    bool verbose;

    GNConfig()
    : learning_rate(1.0), damping(1e-6), l2_reg(0.0), max_iters(1000), verbose(true) {}
};

class GaussNewtonOptimizer {
public:
    explicit GaussNewtonOptimizer(const GNConfig& cfg);

    // x: parameters (modified in place)
    // JTJ: flattened m*m matrix (row-major) computed externally
    // Jtr: flattened m vector (J^T r) computed externally
    void step(std::vector<double>& x, const std::vector<double>& JTJ, const std::vector<double>& Jtr);

private:
    GNConfig cfg_;
    // helper: solve linear system A * delta = b, A is n x n in row-major
    bool solve_linear_system(std::vector<double>& A, std::vector<double>& b, std::vector<double>& x_out) const;
};

#endif // GAUSS_NEWTON_H