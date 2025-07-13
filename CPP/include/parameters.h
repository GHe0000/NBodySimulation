#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <vector>
#include <Eigen/Dense>
#include <fftw3.h>

// ———————————— 模拟参数 ————————————
// 模拟参数
const int NUM_THREADS = 8;      // 并行线程数

// 粒子网格参数
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
const double A_INIT = 0.02;     // 初始尺度因子
const double A_FINAL = 4.0;     // 终止尺度因子i
const double DT = 0.02;         // 时间步长 (以尺度因子 a 为单位)

// 使用 Eigen 库定义二维向量及其动态数组
using Vec2D = Eigen::Vector2d;
// 使用 Eigen 的对齐分配器来确保 SSE 向量化指令可以安全使用
using VectorOfVec2D = std::vector<Vec2D, Eigen::aligned_allocator<Vec2D>>;

#endif
