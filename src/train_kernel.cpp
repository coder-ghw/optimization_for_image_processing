#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <getopt.h>
#include <memory>
#include <random>
#include "gradient_descent.h"
#include "adam_optimizer.h"
#include "gauss_newton.h"

// Simple helper to parse CLI options
struct Options {
    std::string img_path;
    std::string out_path;
    int iters;
    double lr;
    double reg;
    double momentum;
    int kernel_size;
    std::string optimizer;
    double damping; // for GN/LM

    Options()
    : out_path("denoised.png"), iters(1000), lr(0.001), reg(0.0),
      momentum(0.0), kernel_size(3), optimizer("adam"), damping(1e-6) {}
};

static void print_usage() {
    std::cout << "Usage: train_kernel --clean clean.png --noisy noisy.png [options]\n"
              << "Options:\n"
              << "  --out PATH          output denoised image (default denoised.png)\n"
              << "  --iters N           training iterations (default 1000)\n"
              << "  --lr FLOAT          learning rate (default 0.001). For GN, lr scales the GN step (default 1.0 recommended).\n"
              << "  --reg FLOAT         L2 regularization lambda (default 0.0)\n"
              << "  --momentum FLOAT    momentum (only for sgd, default 0.0)\n"
              << "  --kernel-size N     kernel size (odd, default 3)\n"
              << "  --optimizer NAME    optimizer to use: adam, sgd, gn (default adam)\n"
              << "  --damping FLOAT     damping for Gauss-Newton/LM (default 1e-6)\n";
}

// compute convolution (valid) of image 'src' by kernel K (kxk), return same size as src (use padded src)
cv::Mat conv_full(const cv::Mat& padded, const std::vector<double>& K, int ksize) {
    int ph = padded.rows;
    int pw = padded.cols;
    int pad = ksize/2;
    int h = ph - 2*pad;
    int w = pw - 2*pad;
    cv::Mat out(h, w, CV_32F, cv::Scalar(0));
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            double s = 0.0;
            for (int ky = 0; ky < ksize; ++ky) {
                for (int kx = 0; kx < ksize; ++kx) {
                    float v = padded.at<float>(y + ky, x + kx);
                    s += K[ky*ksize + kx] * v;
                }
            }
            out.at<float>(y, x) = static_cast<float>(s);
        }
    }
    return out;
}

// compute MSE loss and gradient with respect to kernel K
double loss_and_gradients(const cv::Mat& noisy_pad, const cv::Mat& target, const std::vector<double>& K, int ksize, std::vector<double>& grad) {
    int h = target.rows;
    int w = target.cols;
    grad.assign(K.size(), 0.0);

    // forward
    cv::Mat pred = conv_full(noisy_pad, K, ksize);

    // compute mse and gradients
    double mse = 0.0;
    int N = h * w;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            double p = pred.at<float>(y, x);
            double t = target.at<float>(y, x);
            double e = p - t;
            mse += e*e;
            // gradient w.r.t kernel element k_ij: sum_over_pixels(2 * e * noisy_patch_value_at_(i,j))
            for (int ky = 0; ky < ksize; ++ky) {
                for (int kx = 0; kx < ksize; ++kx) {
                    double nv = noisy_pad.at<float>(y + ky, x + kx);
                    grad[ky*ksize + kx] += 2.0 * e * nv / N;
                }
            }
        }
    }
    mse /= N;
    return mse;
}

// compute JTJ (m*m) and Jtr (m) for current kernel K
void compute_JTJ_and_Jtr(const cv::Mat& noisy_pad, const cv::Mat& target, const std::vector<double>& K, int ksize,
                        std::vector<double>& JTJ, std::vector<double>& Jtr) {
    int h = target.rows;
    int w = target.cols;
    int N = h * w;
    int m = ksize * ksize;
    JTJ.assign(m * m, 0.0);
    Jtr.assign(m, 0.0);

    // forward prediction
    cv::Mat pred = conv_full(noisy_pad, K, ksize);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            double r = pred.at<float>(y, x) - target.at<float>(y, x); // residual
            // collect patch vector v (length m)
            std::vector<double> v(m);
            for (int ky = 0; ky < ksize; ++ky) {
                for (int kx = 0; kx < ksize; ++kx) {
                    v[ky*ksize + kx] = noisy_pad.at<float>(y + ky, x + kx);
                }
            }
            // accumulate Jtr and JTJ
            for (int i = 0; i < m; ++i) {
                Jtr[i] += v[i] * r / N; // scaled by 1/N
                for (int j = 0; j < m; ++j) {
                    JTJ[i*m + j] += v[i] * v[j] / N; // scaled by 1/N
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    Options opt;
    static struct option long_options[] = {
        {"img", required_argument, 0, 0},
        {"out", required_argument, 0, 0},
        {"iters", required_argument, 0, 0},
        {"lr", required_argument, 0, 0},
        {"reg", required_argument, 0, 0},
        {"momentum", required_argument, 0, 0},
        {"kernel-size", required_argument, 0, 0},
        {"optimizer", required_argument, 0, 0},
        {"damping", required_argument, 0, 0},
        {0,0,0,0}
    };
    int optindex;
    while (true) {
        int c = getopt_long(argc, argv, "", long_options, &optindex);
        if (c == -1) break;
        if (c == 0) {
            std::string name = long_options[optindex].name;
            if (name == "img") opt.img_path = optarg;
            else if (name == "out") opt.out_path = optarg;
            else if (name == "iters") opt.iters = std::stoi(optarg);
            else if (name == "lr") opt.lr = std::stod(optarg);
            else if (name == "reg") opt.reg = std::stod(optarg);
            else if (name == "momentum") opt.momentum = std::stod(optarg);
            else if (name == "kernel-size") opt.kernel_size = std::stoi(optarg);
            else if (name == "optimizer") opt.optimizer = optarg;
            else if (name == "damping") opt.damping = std::stod(optarg);
        }
    }

    if (opt.img_path.empty()) {
        print_usage();
        return 1;
    }
    if (opt.kernel_size % 2 != 1 || opt.kernel_size <= 0) {
        std::cerr << "kernel-size must be positive odd\n";
        return 1;
    }
    std::cout << "img path=" << opt.img_path << "\n";

    cv::Mat clean = cv::imread(opt.img_path, cv::IMREAD_GRAYSCALE);
    cv::Mat noisy;
    cv::GaussianBlur(clean, noisy, cv::Size(5,5), 1.0);

    if (clean.empty() || noisy.empty()) {
        std::cerr << "Failed to load images.\n";
        return 1;
    }
    if (clean.size() != noisy.size()) {
        std::cerr << "Images must have same size.\n";
        return 1;
    }

    // convert to float [0,1]
    cv::Mat clean_f, noisy_f;
    clean.convertTo(clean_f, CV_32F, 1.0/255.0);
    noisy.convertTo(noisy_f, CV_32F, 1.0/255.0);

    cv::Mat n=noisy_f.clone();
    cv::randn(n, 10, 0.05);
    noisy_f = noisy_f + n;

    int ksize = opt.kernel_size;
    int pad = ksize/2;
    cv::Mat noisy_pad;
    cv::copyMakeBorder(noisy_f, noisy_pad, pad, pad, pad, pad, cv::BORDER_REPLICATE);

    // initialize kernel as small average filter + tiny noise using C++11 random
    std::vector<double> K(ksize*ksize, 1.0/(ksize*ksize));
    std::mt19937 rng;
    rng.seed(12345);
    std::uniform_real_distribution<double> dist(0.0, 0.001);
    for (size_t i = 0; i < K.size(); ++i) {
        K[i] += dist(rng);
    }

    // Prepare optimizers using unique_ptr
    bool use_adam = (opt.optimizer == "adam");
    bool use_gn = (opt.optimizer == "gn" || opt.optimizer == "gauss");
    std::unique_ptr<GradientDescent> sgd_ptr;
    std::unique_ptr<AdamOptimizer> adam_ptr;
    std::unique_ptr<GaussNewtonOptimizer> gn_ptr;

    // configure/construct optimizers
    if (use_adam) {
        AdamConfig a_cfg;
        a_cfg.learning_rate = opt.lr;
        a_cfg.beta1 = 0.9;
        a_cfg.beta2 = 0.999;
        a_cfg.eps = 1e-8;
        a_cfg.l2_reg = opt.reg;
        a_cfg.max_iters = opt.iters;
        a_cfg.verbose = true;
        adam_ptr.reset(new AdamOptimizer(a_cfg));
    } else if (use_gn) {
        GNConfig gcfg;
        gcfg.learning_rate = opt.lr;    // step scale for delta
        gcfg.damping = opt.damping;
        gcfg.l2_reg = opt.reg;
        gcfg.max_iters = opt.iters;
        gcfg.verbose = true;
        gn_ptr.reset(new GaussNewtonOptimizer(gcfg));
    } else {
        GDConfig gcfg;
        gcfg.learning_rate = opt.lr;
        gcfg.momentum = opt.momentum;
        gcfg.l2_reg = opt.reg;
        gcfg.max_iters = opt.iters;
        gcfg.verbose = true;
        sgd_ptr.reset(new GradientDescent(gcfg));
    }

    std::vector<double> grad;
    double best_loss = 1e9;
    std::vector<double> bestK = K;
    cv::Mat best_pred;

    int m = ksize * ksize;
    std::vector<double> JTJ; JTJ.reserve(m*m);
    std::vector<double> Jtr; Jtr.reserve(m);

    for (int it = 0; it < opt.iters; ++it) {
        double loss = loss_and_gradients(noisy_pad, clean_f, K, ksize, grad);
        std::cout << "Iter > " << it << " loss=" << loss << "\n";
        // add L2 term to loss
        double l2 = 0.0;
        for (size_t i = 0; i < K.size(); ++i) l2 += K[i]*K[i];
        loss += opt.reg * l2;

        if (loss < best_loss) {
            best_loss = loss;
            bestK = K;
            best_pred = conv_full(noisy_pad, K, ksize);
        }

        if (use_gn) {
            // compute JTJ and Jtr
            compute_JTJ_and_Jtr(noisy_pad, clean_f, K, ksize, JTJ, Jtr);
            gn_ptr->step(K, JTJ, Jtr);
        } else if (use_adam) {
            adam_ptr->step(K, grad);
        } else {
            sgd_ptr->step(K, grad);
        }

        std::cout << "Iter < " << it << " loss=" << loss << "\n";
    }

    // apply best kernel and save result
    if (best_pred.empty()) best_pred = conv_full(noisy_pad, K, ksize);
    cv::Mat out_img;
    best_pred.convertTo(out_img, CV_8U, 255.0);
    cv::imwrite(opt.out_path, out_img);

    // print learned kernel
    std::cout << "Learned kernel (" << ksize << "x" << ksize << "):\n";
    for (int y = 0; y < ksize; ++y) {
        for (int x = 0; x < ksize; ++x) {
            std::printf("%8.6f ", bestK[y*ksize + x]);
        }
        std::printf("\n");
    }
    std::cout << "Best loss: " << best_loss << "\n";
    std::cout << "Saved denoised image to " << opt.out_path << "\n";

    return 0;
}