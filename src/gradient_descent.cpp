#include "gradient_descent.h"
#include <cmath>
#include <iostream>

GradientDescent::GradientDescent(const GDConfig& cfg) : cfg_(cfg) {}

void GradientDescent::step(std::vector<double>& x, const std::vector<double>& grad) {
    if (velocity_.size() != x.size()) {
        velocity_.assign(x.size(), 0.0);
    }

    // update velocity and parameters
    for (size_t i = 0; i < x.size(); ++i) {
        // incorporate L2 regularization into effective gradient
        double g = grad[i] + 2.0 * cfg_.l2_reg * x[i];
        velocity_[i] = cfg_.momentum * velocity_[i] + cfg_.learning_rate * g;
        x[i] -= velocity_[i];
    }
}