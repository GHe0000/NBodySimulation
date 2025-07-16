#include "poisson.h"
#include "utils.h"

#include <cmath>
#include <omp.h>

PoissonSolver::PoissonSolver() : k_indices_x(N), k_indices_y(N) {
    delta_grid = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N * N);
    delta_f = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N * N);
    phi_f = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N * N);
    phi_real = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N * N);

    p_fwd = fftw_plan_dft_2d(N, N, delta_grid, delta_f, FFTW_FORWARD, FFTW_ESTIMATE);
    p_bwd = fftw_plan_dft_2d(N, N, phi_f, phi_real, FFTW_BACKWARD, FFTW_ESTIMATE);

    utils::wave_number(k_indices_x, k_indices_y);
}

PoissonSolver::~PoissonSolver() {
    fftw_destroy_plan(p_fwd);
    fftw_destroy_plan(p_bwd);
    fftw_free(delta_grid);
    fftw_free(delta_f);
    fftw_free(phi_f);
    fftw_free(phi_real);
}

double PoissonSolver::da_dt(double a) {
    return H0 * a * std::sqrt(OmegaL + OmegaM * std::pow(a, -3) + OmegaK * std::pow(a, -2));
}

void PoissonSolver::cic(const VectorOfVec2D& pos_grid_units, fftw_complex* target) {
    #pragma omp parallel for
    for (int i = 0; i < N * N; ++i) {
        target[i][0] = 0.0;
        target[i][1] = 0.0;
    }

    std::vector<std::vector<double>> private_targets(omp_get_max_threads(), std::vector<double>(N * N, 0.0));

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        #pragma omp for
        for (int i = 0; i < TOTAL_PARTICLES; ++i) {
            double p_x = pos_grid_units[i].x();
            double p_y = pos_grid_units[i].y();
            int idx0 = static_cast<int>(std::floor(p_x));
            int idx1 = static_cast<int>(std::floor(p_y));
            double f0 = p_x - idx0;
            double f1 = p_y - idx1;

            // 处理周期性边界
            int i0 = idx0 % N; if (i0 < 0) i0 += N;
            int j0 = idx1 % N; if (j0 < 0) j0 += N;
            int i1 = (idx0 + 1) % N; if (i1 < 0) i1 += N;
            int j1 = (idx1 + 1) % N; if (j1 < 0) j1 += N;

            private_targets[thread_id][i0 * N + j0] += (1 - f0) * (1 - f1);
            private_targets[thread_id][i1 * N + j0] += f0 * (1 - f1);
            private_targets[thread_id][i0 * N + j1] += (1 - f0) * f1;
            private_targets[thread_id][i1 * N + j1] += f0 * f1;
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < N * N; ++i) {
        for (int t = 0; t < omp_get_max_threads(); ++t) {
            target[i][0] += private_targets[t][i];
        }
    }
}

Vec2D PoissonSolver::interp(const VectorOfVec2D& data_grid, const Vec2D& x) {
    int idx0 = static_cast<int>(std::floor(x.x()));
    int idx1 = static_cast<int>(std::floor(x.y()));
    double xm = x.x() - idx0;
    double xn = 1.0 - xm;
    double ym = x.y() - idx1;
    double yn = 1.0 - ym;

    // 处理周期性边界
    int i1 = idx0 % N; if (i1 < 0) i1 += N;
    int j1 = idx1 % N; if (j1 < 0) j1 += N;
    int i2 = (idx0 + 1) % N; if (i2 < 0) i2 += N;
    int j2 = (idx1 + 1) % N; if (j2 < 0) j2 += N;

    const Vec2D& f1 = data_grid[i1 * N + j1];
    const Vec2D& f2 = data_grid[i2 * N + j1];
    const Vec2D& f3 = data_grid[i1 * N + j2];
    const Vec2D& f4 = data_grid[i2 * N + j2];
    
    return f1 * xn * yn + f2 * xm * yn + f3 * xn * ym + f4 * xm * ym;
}

VectorOfVec2D PoissonSolver::calculate_acceleration(const VectorOfVec2D& pos, double a) {
    // 1. 将物理坐标转换为网格坐标
    VectorOfVec2D x_grid(TOTAL_PARTICLES);
    #pragma omp parallel for
    for(int i = 0; i < TOTAL_PARTICLES; ++i) {
        x_grid[i] = pos[i] / BOX_RES;
    }

    // 2. CIC 质量分配得到密度场
    cic(x_grid, delta_grid);

    // 3. 计算密度扰动 delta = rho/rho_mean - 1
    double mean_density = static_cast<double>(TOTAL_PARTICLES) / (N * N);
    #pragma omp parallel for
    for(int i = 0; i < N * N; ++i){
        delta_grid[i][0] = delta_grid[i][0] / mean_density - 1.0;
        delta_grid[i][1] = 0.0; // 虚部为0
    }
    
    // 4. FFT 到 k-空间
    fftw_execute(p_fwd);

    // 5. 在 k-空间求解泊松方程
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = i * N + j;
            double kx = k_indices_x[i];
            double ky = k_indices_y[j];
            double k_mag_sq = kx * kx + ky * ky;
            double potential_kernel = -utils::k_pow_safe(k_mag_sq, -2.0);
            phi_f[idx][0] = delta_f[idx][0] * potential_kernel;
            phi_f[idx][1] = delta_f[idx][1] * potential_kernel;
        }
    }
    
    // 6. iFFT 回真实空间得到引力势
    fftw_execute(p_bwd);
    
    // 7. 在网格上计算加速度 (a = -∇φ)
    VectorOfVec2D acc_grid(N * N);
    double G_over_a = G_CONST / a;
    double norm_factor = 1.0 / (N * N); // FFTW 逆变换后的归一化因子
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < N; ++j) {
            double phi_xp1 = phi_real[((i + 1) % N) * N + j][0] * norm_factor;
            double phi_xm1 = phi_real[((i - 1 + N) % N) * N + j][0] * norm_factor;
            double phi_yp1 = phi_real[i * N + ((j + 1) % N)][0] * norm_factor;
            double phi_ym1 = phi_real[i * N + ((j - 1 + N) % N)][0] * norm_factor;
            
            double acc_x = -(phi_xp1 - phi_xm1) / (2.0 * BOX_RES);
            double acc_y = -(phi_yp1 - phi_ym1) / (2.0 * BOX_RES);
            acc_grid[i * N + j] << acc_x * G_over_a, acc_y * G_over_a;
        }
    }
    
    // 8. 将网格上的加速度插值回粒子位置
    VectorOfVec2D acc(TOTAL_PARTICLES);
    #pragma omp parallel for
    for (int i = 0; i < TOTAL_PARTICLES; ++i) {
        acc[i] = interp(acc_grid, x_grid[i]);
    }
    
    return acc;
}
