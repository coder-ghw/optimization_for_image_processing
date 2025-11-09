#ifndef GRADIENT_DESCENT_H
#define GRADIENT_DESCENT_H

#include <vector>

struct GDConfig {
    double learning_rate;
    double momentum; // 0 = vanilla GD
    double l2_reg;   // lambda for L2 regularization
    int max_iters;
    bool verbose;

    GDConfig()
    : learning_rate(0.1), momentum(0.0), l2_reg(0.0), max_iters(1000), verbose(true) {}
};

class GradientDescent {
public:
    explicit GradientDescent(const GDConfig& cfg);

    // x: parameters (modified in place), grad: gradient, returns updated params
    void step(std::vector<double>& x, const std::vector<double>& grad);

private:
    GDConfig cfg_;
    std::vector<double> velocity_; // for momentum
};

#endif // GRADIENT_DESCENT_H