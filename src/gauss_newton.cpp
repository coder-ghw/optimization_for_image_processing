#include "gauss_newton.h"
#include <cmath>
#include <iostream>
#include <algorithm>

// Constructor
GaussNewtonOptimizer::GaussNewtonOptimizer(const GNConfig& cfg) : cfg_(cfg) {}

// Solve A x = b using Gaussian elimination with partial pivoting.
// A is n*n stored row-major in A; b is length n. On success, x_out contains solution and returns true.
// This is a simple implementation intended for small n (kernel sizes like 3x3, 5x5).
bool GaussNewtonOptimizer::solve_linear_system(std::vector<double>& A, std::vector<double>& b, std::vector<double>& x_out) const {
    const int n = static_cast<int>(b.size());
    if (n == 0) return false;
    // Augmented matrix [A | b]
    std::vector<double> aug((n+1)*n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) aug[i*(n+1) + j] = A[i*n + j];
        aug[i*(n+1) + n] = b[i];
    }

    // Gaussian elimination with partial pivoting
    for (int col = 0; col < n; ++col) {
        // find pivot
        int pivot = col;
        double maxv = std::fabs(aug[pivot*(n+1) + col]);
        for (int r = col + 1; r < n; ++r) {
            double v = std::fabs(aug[r*(n+1) + col]);
            if (v > maxv) { maxv = v; pivot = r; }
        }
        if (maxv < 1e-15) {
            // singular or nearly singular
            return false;
        }
        if (pivot != col) {
            // swap rows
            for (int c = col; c <= n; ++c) std::swap(aug[col*(n+1) + c], aug[pivot*(n+1) + c]);
        }
        // normalize and eliminate
        double pivot_val = aug[col*(n+1) + col];
        for (int c = col; c <= n; ++c) aug[col*(n+1) + c] /= pivot_val;
        for (int r = 0; r < n; ++r) {
            if (r == col) continue;
            double factor = aug[r*(n+1) + col];
            if (factor == 0.0) continue;
            for (int c = col; c <= n; ++c) {
                aug[r*(n+1) + c] -= factor * aug[col*(n+1) + c];
            }
        }
    }
    // read solution
    x_out.assign(n, 0.0);
    for (int i = 0; i < n; ++i) x_out[i] = aug[i*(n+1) + n];
    return true;
}

// step: JTJ is m*m row-major, Jtr is m
void GaussNewtonOptimizer::step(std::vector<double>& x, const std::vector<double>& JTJ, const std::vector<double>& Jtr) {
    const int m = static_cast<int>(x.size());
    if (m == 0) return;

    // Build A = JTJ + (l2_reg + damping) * I
    std::vector<double> A = JTJ; // copy
    double diag_add = cfg_.l2_reg + cfg_.damping;
    for (int i = 0; i < m; ++i) A[i*m + i] += diag_add;

    // Build rhs b = Jtr + l2_reg * x
    std::vector<double> b(m, 0.0);
    for (int i = 0; i < m; ++i) b[i] = Jtr[i] + cfg_.l2_reg * x[i];

    // Solve A * delta = b
    std::vector<double> delta;
    bool ok = solve_linear_system(A, b, delta);
    if (!ok) {
        if (cfg_.verbose) std::cerr << "Gauss-Newton: linear solve failed (singular). Skipping update.\n";
        return;
    }

    // update x <- x - lr * delta
    for (int i = 0; i < m; ++i) {
        x[i] -= cfg_.learning_rate * delta[i];
    }
}