#include "initial.h"
#include "utils.h"

#include <random>
#include <omp.h>

namespace InitialConditions {

void generate(VectorOfVec2D& pos, VectorOfVec2D& mom) {
    // --- FFTW 初始化 ---
    fftw_complex *phi_f = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * N);
    fftw_complex *phi_real = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * N);
    fftw_plan p_phi_inv = fftw_plan_dft_2d(N, N, phi_f, phi_real, FFTW_BACKWARD, FFTW_ESTIMATE);

    fftw_complex *white_noise_f = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * N);
    fftw_complex *white_noise_real = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * N);
    fftw_plan p_wn_fwd = fftw_plan_dft_2d(N, N, white_noise_real, white_noise_f, FFTW_FORWARD, FFTW_ESTIMATE);

    // --- 生成高斯白噪声 ---
    std::mt19937 gen(SEED);
    std::normal_distribution<> d(0, 1);
    #pragma omp parallel for
    for (int i = 0; i < N * N; ++i) {
        white_noise_real[i][0] = d(gen);
        white_noise_real[i][1] = 0.0;
    }
    fftw_execute(p_wn_fwd);

    // --- 在傅里叶空间应用功率谱 ---
    std::vector<double> k_indices_x(N), k_indices_y(N);
    utils::wave_number(k_indices_x, k_indices_y);

    double k_max = N * M_PI / L;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = i * N + j;
            double kx = k_indices_x[i];
            double ky = k_indices_y[j];
            double k_mag_sq = kx * kx + ky * ky;

            // P(k) = k^n
            double pk = utils::k_pow_safe(k_mag_sq, POWER_LAW_N);
            
            // 应用高斯平滑 (Gaussian smoothing)
            pk *= std::exp(SCALE_SIGMA * SCALE_SIGMA / (BOX_RES * BOX_RES) * (std::cos(kx * BOX_RES) - 1.0));
            pk *= std::exp(SCALE_SIGMA * SCALE_SIGMA / (BOX_RES * BOX_RES) * (std::cos(ky * BOX_RES) - 1.0));
            
            // 截断高频模式
            if (k_mag_sq > k_max * k_max) pk = 0;

            // 得到密度场的傅里叶分量 delta_k ~ sqrt(P(k)) * (Gaussian random)
            double field_amp = std::sqrt(pk);
            white_noise_f[idx][0] *= field_amp;
            white_noise_f[idx][1] *= field_amp;

            // 通过泊松方程求解引力势 phi_k = -delta_k / k^2
            double potential_kernel = -utils::k_pow_safe(k_mag_sq, -2.0);
            phi_f[idx][0] = white_noise_f[idx][0] * potential_kernel;
            phi_f[idx][1] = white_noise_f[idx][1] * potential_kernel;
        }
    }

    // --- 逆变换得到真实空间的引力势 ---
    fftw_execute(p_phi_inv);

    // 归一化并应用振幅
    double norm_factor = FIELD_AMPLITUDE / (N * N);
    #pragma omp parallel for
    for (int i = 0; i < N * N; ++i) {
        phi_real[i][0] *= norm_factor;
    }
    
    // --- 计算位移场 u = -∇φ ---
    VectorOfVec2D u(TOTAL_PARTICLES);
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = i * N + j;
            double du_x = (phi_real[((i + 1) % N) * N + j][0] - phi_real[((i - 1 + N) % N) * N + j][0]) / (2.0 * BOX_RES);
            double du_y = (phi_real[i * N + ((j + 1) % N)][0] - phi_real[i * N + ((j - 1 + N) % N)][0]) / (2.0 * BOX_RES);
            u[idx] << -du_x, -du_y;
        }
    }

    // 
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = i * N + j;
            pos[idx] = Vec2D(i * BOX_RES, j * BOX_RES) + u[idx];
            mom[idx] = A_INIT * u[idx];
        }
    }

    fftw_destroy_plan(p_phi_inv);
    fftw_destroy_plan(p_wn_fwd);
    fftw_free(phi_f);
    fftw_free(phi_real);
    fftw_free(white_noise_f);
    fftw_free(white_noise_real);
}

}
