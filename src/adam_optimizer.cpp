#include "adam_optimizer.h"
#include <cmath>
#include <iostream>

AdamOptimizer::AdamOptimizer(const AdamConfig& cfg) : cfg_(cfg), t_(0) {}

void AdamOptimizer::reset() {
    m_.clear();
    v_.clear();
    t_ = 0;
}

void AdamOptimizer::step(std::vector<double>& x, const std::vector<double>& grad) {
    const size_t n = x.size();
    if (m_.size() != n) {
        m_.assign(n, 0.0);
        v_.assign(n, 0.0);
    }

    t_ += 1;
    // precompute bias-correction denominators
    double bias_correction1 = 1.0 - std::pow(cfg_.beta1, t_);
    double bias_correction2 = 1.0 - std::pow(cfg_.beta2, t_);

    for (size_t i = 0; i < n; ++i) {
        // incorporate L2 regularization into gradient
        double g = grad[i] + 2.0 * cfg_.l2_reg * x[i];
        // update biased first moment estimate
        m_[i] = cfg_.beta1 * m_[i] + (1.0 - cfg_.beta1) * g;
        // update biased second raw moment estimate
        v_[i] = cfg_.beta2 * v_[i] + (1.0 - cfg_.beta2) * (g * g);
        // compute bias-corrected moments
        double m_hat = m_[i] / bias_correction1;
        double v_hat = v_[i] / bias_correction2;
        // update parameters
        double denom = std::sqrt(v_hat) + cfg_.eps;
        x[i] -= cfg_.learning_rate * (m_hat / denom);
    }
}
