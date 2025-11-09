#ifndef ADAM_OPTIMIZER_H
#define ADAM_OPTIMIZER_H

#include <vector>

struct AdamConfig {
    double learning_rate;
    double beta1;
    double beta2;
    double eps;
    double l2_reg; // lambda for L2 regularization
    int max_iters;
    bool verbose;

    AdamConfig()
    : learning_rate(0.001), beta1(0.9), beta2(0.999), eps(1e-8),
      l2_reg(0.0), max_iters(1000), verbose(true) {}
};

class AdamOptimizer {
public:
    explicit AdamOptimizer(const AdamConfig& cfg);

    // x: parameters (modified in place), grad: gradient (same size as x)
    void step(std::vector<double>& x, const std::vector<double>& grad);

    void reset();

private:
    AdamConfig cfg_;
    std::vector<double> m_; // first moment
    std::vector<double> v_; // second moment
    int t_; // time step
};

#endif // ADAM_OPTIMIZER_H