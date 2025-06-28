#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <random>
#include <string>
#include <fstream>
#include <filesystem>
#include <fftw3.h>
#include <omp.h>
#include <Eigen/Dense>

// ———————————— 模拟参数 ————————————
const int N = 256;              // 粒子网格的一边大小
const double L = 50.0;          // 模拟盒子的物理尺寸 (Mpc/h) 
const double BOX_RES = L / N;   // 盒子分辨率
const int DIM = 2;              // 维度
const int TOTAL_PARTICLES = N * N;

// 宇宙学参数
const double H0 = 68.0;
const double OmegaM = 0.31;
const double OmegaL = 0.69;
const double OmegaK = 1.0 - OmegaM - OmegaL;
const double G_CONST = 3.0 / 2.0 * OmegaM * H0 * H0;

// 模拟时间参数
const double A_INIT = 0.02;    // 初始尺度因子
const double A_FINAL = 4.0;    // 终止尺度因子
const double DT = 0.02;        // 时间步长

// 初始条件参数
const double POWER_LAW_N = -0.5; // 功率谱指数
const double SCALE_SIGMA = 0.2;  // 平滑尺度
const double FIELD_AMPLITUDE = 10.0; // 场振幅
const unsigned int SEED = 4;     // 随机种子

using Vec2D = Eigen::Vector2d;
using VectorOfVec2D = std::vector<Vec2D, Eigen::aligned_allocator<Vec2D>>; // 确保内存对齐


// --- 辅助函数和物理计算 (Helper Functions and Physics Calculations) ---

/**
 * @brief 计算傅里叶空间的波数 k。
 * Computes the wave number k in Fourier space.
 * @param kx 输出的 kx 分量 (Output kx components)
 * @param ky 输出的 ky 分量 (Output ky components)
 */
void wave_number(std::vector<double>& kx, std::vector<double>& ky) {
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        double val = (i > N / 2) ? (i - N) : i;
        val *= 2.0 * M_PI / L;
        kx[i] = val;
        ky[i] = val; // For a square box
    }
}

/**
 * @brief 安全地计算 k 的 n 次方，避免除以零。
 * Safely computes k to the power of n, avoiding division by zero.
 * @param k_sq k的模长平方 (Squared magnitude of k)
 * @param n 指数 (Exponent)
 * @return k^n
 */
double k_pow_safe(double k_sq, double n) {
    if (k_sq == 0.0) return 0.0;
    return std::pow(k_sq, n / 2.0);
}

/**
 * @brief 计算宇宙尺度因子 a 的时间导数。
 * Calculates the time derivative of the scale factor 'a'.
 * @return da/dt
 */
double da_dt(double a) {
    return H0 * a * std::sqrt(OmegaL + OmegaM * std::pow(a, -3) + OmegaK * std::pow(a, -2));
}

/**
 * @brief Cloud-in-Cell (CIC) 质量分配。
 * Cloud-in-Cell (CIC) mass assignment.
 * @param pos 粒子位置 (Particle positions in grid units)
 * @param target 输出的密度网格 (Output density grid)
 */
void md_cic_2d(const VectorOfVec2D& pos, fftw_complex* target) {
    #pragma omp parallel for
    for(int i = 0; i < N * N; ++i) {
        target[i][0] = 0.0;
        target[i][1] = 0.0;
    }

    std::vector<std::vector<double>> private_targets(omp_get_max_threads(), std::vector<double>(N * N, 0.0));

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        #pragma omp for
        for (int i = 0; i < TOTAL_PARTICLES; ++i) {
            double p_x = pos[i].x();
            double p_y = pos[i].y();
            int idx0 = static_cast<int>(std::floor(p_x));
            int idx1 = static_cast<int>(std::floor(p_y));
            double f0 = p_x - idx0;
            double f1 = p_y - idx1;

            int i0 = idx0 % N;
            int j0 = idx1 % N;
            int i1 = (idx0 + 1) % N;
            int j1 = (idx1 + 1) % N;

            if (i0 < 0) i0 += N;
            if (j0 < 0) j0 += N;
            if (i1 < 0) i1 += N;
            if (j1 < 0) j1 += N;

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


/**
 * @brief 二维双线性插值。
 * 2D bilinear interpolation.
 * @param data_grid 输入的网格数据 (Input grid data)
 * @param x 粒子位置 (particle positions in grid units)
 * @return 插值后的值 (Interpolated values)
 */
Vec2D interp_2d(const VectorOfVec2D& data_grid, const Vec2D& x) {
    int idx0 = static_cast<int>(std::floor(x.x()));
    int idx1 = static_cast<int>(std::floor(x.y()));
    double xm = x.x() - idx0;
    double xn = 1.0 - xm;
    double ym = x.y() - idx1;
    double yn = 1.0 - ym;

    int i1 = idx0 % N;
    int j1 = idx1 % N;
    int i2 = (idx0 + 1) % N;
    int j2 = (idx1 + 1) % N;

    if (i1 < 0) i1 += N;
    if (j1 < 0) j1 += N;
    if (i2 < 0) i2 += N;
    if (j2 < 0) j2 += N;

    const Vec2D& f1 = data_grid[i1 * N + j1];
    const Vec2D& f2 = data_grid[i2 * N + j1];
    const Vec2D& f3 = data_grid[i1 * N + j2];
    const Vec2D& f4 = data_grid[i2 * N + j2];
    
    return f1 * xn * yn + f2 * xm * yn + f3 * xn * ym + f4 * xm * ym;
}


/**
 * @brief 保存粒子数据到二进制文件。
 * Saves particle data to a binary file.
 */
void save_data(const VectorOfVec2D& pos, const VectorOfVec2D& mom, double time) {
    int time_ms = static_cast<int>(round(time * 1000));
    char buffer[100];
    snprintf(buffer, sizeof(buffer), "data/x.%05d.bin", time_ms);
    std::ofstream outfile(buffer, std::ios::binary);
    if (!outfile) {
        std::cerr << "Error opening file: " << buffer << std::endl;
        return;
    }
    outfile.write(reinterpret_cast<const char*>(pos.data()), pos.size() * sizeof(Vec2D));
    outfile.write(reinterpret_cast<const char*>(mom.data()), mom.size() * sizeof(Vec2D));
    outfile.close();
}


// --- 主程序 (Main Program) ---
int main() {
    // --- 2. 初始条件生成 (Initial Condition Generation) ---
    std::cout << "Generating initial conditions..." << std::endl;

    fftw_complex *phi_f = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * N);
    fftw_complex *phi_real = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * N);
    fftw_plan p_phi_inv = fftw_plan_dft_2d(N, N, phi_f, phi_real, FFTW_BACKWARD, FFTW_ESTIMATE);

    std::vector<double> k_indices_x(N), k_indices_y(N);
    wave_number(k_indices_x, k_indices_y);

    std::mt19937 gen(SEED);
    std::normal_distribution<> d(0, 1);
    
    fftw_complex *white_noise_f = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * N);
    fftw_complex *white_noise_real = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * N);
    fftw_plan p_wn_fwd = fftw_plan_dft_2d(N, N, white_noise_real, white_noise_f, FFTW_FORWARD, FFTW_ESTIMATE);

    #pragma omp parallel for
    for (int i = 0; i < N * N; ++i) {
        white_noise_real[i][0] = d(gen);
        white_noise_real[i][1] = 0.0;
    }
    fftw_execute(p_wn_fwd);

    double k_max = N * M_PI / L;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = i * N + j;
            double kx = k_indices_x[i];
            double ky = k_indices_y[j];
            double k_mag_sq = kx * kx + ky * ky;

            double pk = k_pow_safe(k_mag_sq, POWER_LAW_N);
            
            pk *= std::exp(SCALE_SIGMA * SCALE_SIGMA / (BOX_RES * BOX_RES) * (std::cos(kx * BOX_RES) - 1.0));
            pk *= std::exp(SCALE_SIGMA * SCALE_SIGMA / (BOX_RES * BOX_RES) * (std::cos(ky * BOX_RES) - 1.0));
            
            if (k_mag_sq > k_max * k_max) pk = 0;

            double field_amp = std::sqrt(pk);
            white_noise_f[idx][0] *= field_amp;
            white_noise_f[idx][1] *= field_amp;

            double potential_kernel = -k_pow_safe(k_mag_sq, -2.0);
            phi_f[idx][0] = white_noise_f[idx][0] * potential_kernel;
            phi_f[idx][1] = white_noise_f[idx][1] * potential_kernel;
        }
    }

    fftw_execute(p_phi_inv);

    #pragma omp parallel for
    for (int i = 0; i < N * N; ++i) {
        phi_real[i][0] *= FIELD_AMPLITUDE / (N * N);
    }
    
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

    VectorOfVec2D positions(TOTAL_PARTICLES);
    VectorOfVec2D momenta(TOTAL_PARTICLES);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = i * N + j;
            positions[idx] = Vec2D(i * BOX_RES, j * BOX_RES) + A_INIT * u[idx];
            momenta[idx] = A_INIT * u[idx];
        }
    }
    
    std::cout << "Initial conditions generated." << std::endl;

    fftw_destroy_plan(p_phi_inv);
    fftw_destroy_plan(p_wn_fwd);
    fftw_free(phi_f);
    fftw_free(phi_real);
    fftw_free(white_noise_f);
    fftw_free(white_noise_real);

    // --- 3. N-Body 模拟循环 (N-Body Simulation Loop) ---
    std::cout << "Starting N-body simulation..." << std::endl;

    double time = A_INIT;

    std::filesystem::create_directory("data");
    save_data(positions, momenta, time);
    
    fftw_complex* delta_grid = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * N);
    fftw_complex* delta_f = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * N);
    fftw_complex* phi_f_loop = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * N);
    fftw_complex* phi_real_loop = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N * N);

    fftw_plan p_fwd_loop = fftw_plan_dft_2d(N, N, delta_grid, delta_f, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan p_bwd_loop = fftw_plan_dft_2d(N, N, phi_f_loop, phi_real_loop, FFTW_BACKWARD, FFTW_ESTIMATE);

    int step_count = 0;
    while (time < A_FINAL) {
        // --- Leap-frog: kick-drift-kick ---
        auto calculate_momentum_update = [&](const VectorOfVec2D& pos, double a) -> VectorOfVec2D {
            VectorOfVec2D x_grid(TOTAL_PARTICLES);
            #pragma omp parallel for
            for(int i=0; i<TOTAL_PARTICLES; ++i) {
                x_grid[i] = pos[i] / BOX_RES;
            }

            md_cic_2d(x_grid, delta_grid);

            double mean_density = TOTAL_PARTICLES / (double)(N*N);
            #pragma omp parallel for
            for(int i=0; i<N*N; ++i){
                delta_grid[i][0] = delta_grid[i][0] / mean_density - 1.0;
                delta_grid[i][1] = 0.0;
            }
            
            fftw_execute(p_fwd_loop);

            #pragma omp parallel for collapse(2)
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    int idx = i * N + j;
                    double kx = k_indices_x[i];
                    double ky = k_indices_y[j];
                    double k_mag_sq = kx * kx + ky * ky;
                    double potential_kernel = -k_pow_safe(k_mag_sq, -2.0);
                    phi_f_loop[idx][0] = delta_f[idx][0] * potential_kernel;
                    phi_f_loop[idx][1] = delta_f[idx][1] * potential_kernel;
                }
            }
            
            fftw_execute(p_bwd_loop);
            
            VectorOfVec2D acc_grid(TOTAL_PARTICLES);
            double G_over_a = G_CONST / a;
            double norm_factor = 1.0 / (N*N);
            #pragma omp parallel for collapse(2)
            for(int i = 0; i < N; ++i) {
                for(int j = 0; j < N; ++j) {
                    double phi_xp1 = phi_real_loop[((i + 1) % N) * N + j][0] * norm_factor;
                    double phi_xm1 = phi_real_loop[((i - 1 + N) % N) * N + j][0] * norm_factor;
                    double phi_yp1 = phi_real_loop[i * N + ((j + 1) % N)][0] * norm_factor;
                    double phi_ym1 = phi_real_loop[i * N + ((j - 1 + N) % N)][0] * norm_factor;
                    
                    double acc_x = (phi_xp1 - phi_xm1) / (2.0 * BOX_RES) * G_over_a;
                    double acc_y = (phi_yp1 - phi_ym1) / (2.0 * BOX_RES) * G_over_a;
                    acc_grid[i * N + j] << acc_x, acc_y;
                }
            }
            
            VectorOfVec2D acc(TOTAL_PARTICLES);
            #pragma omp parallel for
            for (int i = 0; i < TOTAL_PARTICLES; ++i) {
                acc[i] = interp_2d(acc_grid, x_grid[i]);
            }
            
            double dt_da = da_dt(a);
            #pragma omp parallel for
            for(int i = 0; i < TOTAL_PARTICLES; ++i) {
                acc[i] *= -1.0 / dt_da;
            }
            return acc;
        };

        // 1. First half kick
        VectorOfVec2D momentum_update = calculate_momentum_update(positions, time);
        #pragma omp parallel for
        for (int i = 0; i < TOTAL_PARTICLES; ++i) {
            momenta[i] += (DT / 2.0) * momentum_update[i];
        }

        // 2. Full drift
        double dpos_factor = DT / (time * time * da_dt(time));
        #pragma omp parallel for
        for (int i = 0; i < TOTAL_PARTICLES; ++i) {
            positions[i] += momenta[i] * dpos_factor;

            // Apply periodic boundary conditions using Eigen's fmod-like functions
            positions[i] = positions[i].unaryExpr([&](double val){ return fmod(val, L); });
            positions[i] = positions[i].unaryExpr([&](double val){ return val < 0 ? val + L : val; });
        }

        // 3. Second half kick (at new time t + dt)
        time += DT;
        momentum_update = calculate_momentum_update(positions, time);
        #pragma omp parallel for
        for (int i = 0; i < TOTAL_PARTICLES; ++i) {
            momenta[i] += (DT / 2.0) * momentum_update[i];
        }

        save_data(positions, momenta, time);

        step_count++;
        printf("Step %d: Simulation time a = %.4f\n", step_count, time);
    }
    
    fftw_destroy_plan(p_fwd_loop);
    fftw_destroy_plan(p_bwd_loop);
    fftw_free(delta_grid);
    fftw_free(delta_f);
    fftw_free(phi_f_loop);
    fftw_free(phi_real_loop);

    std::cout << "Simulation finished." << std::endl;
    return 0;
}
