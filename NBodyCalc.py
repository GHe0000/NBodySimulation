#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
这是一个 N-Body 模拟的单文件实现，它整合了计算和绘图功能。
此脚本旨在复现原始多文件、基于类的代码库的核心功能，但结构更简单，不使用类。

执行流程:
1.  设置模拟参数 (宇宙学模型、盒子大小、时间步等)。
2.  生成初始条件：
    a. 在傅里叶空间中根据指定的功率谱生成一个高斯随机场。
    b. 应用泽尔多维奇近似，根据该场计算粒子的初始位移和速度。
3.  运行 N-Body 模拟：
    a. 使用“蛙跳法”积分器随时间演化系统。
    b. 在每个时间步中：
        i.   使用 CIC 方法将粒子质量分配到网格上。
        ii.  在傅里叶空间中求解泊松方程，得到引力势。
        iii. 从势场计算引力，并插值到每个粒子上。
        iv.  更新粒子的位置和动量。
    c. 将每个时间步的粒子位置和动量数据保存到 `data/` 目录。
4.  绘制结果：
    a. 模拟结束后，加载指定时间点的数据。
    b. 创建初始网格的三角剖分。
    c. 计算每个三角形的面积，以此来表示局部密度。
    d. 使用 Matplotlib 的 `tripcolor` 函数绘制相空间图，并保存为图像文件。
"""

import os
import numpy as np
from scipy.integrate import quad
from functools import partial
import numba

from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib import colors
# --- 1. 参数设置 ---
# 对应 `nbody.py` 主程序中的参数定义
N = 256          # 粒子网格的一边大小
L = 50.0         # 模拟盒子的物理尺寸 (Mpc/h)
BOX_RES = L / N  # 盒子分辨率
DIM = 2          # 维度

# 宇宙学参数 (对应 `nbody.py` 中的 EdS 模型)
H0 = 68.0
OmegaM = 0.31
OmegaL = 0.69
# H0 = 70.0
# OmegaM = 1.0
# OmegaL = 0.0
OmegaK = 1 - OmegaM - OmegaL
G_const = 3./2 * OmegaM * H0**2

# 模拟时间参数
A_INIT = 0.02    # 初始尺度因子
A_FINAL = 4.0    # 终止尺度因子
DT = 0.02        # 时间步长

# 初始条件参数
POWER_LAW_N = -0.5 # 功率谱指数
SCALE_SIGMA = 0.2  # 平滑尺度
FIELD_AMPLITUDE = 10.0 # 场振幅
SEED = 4           # 随机种子

# --- 辅助函数和物理计算 ---

def _wave_number(shape, L):
    """
    计算傅里叶空间的波数 k。
    对应 `cft.py` 中的 `_wave_number` 和 `Box.K` 的部分功能。
    """
    N = shape[0]
    k_indices = np.indices(shape)
    k_indices = np.where(k_indices > N / 2, k_indices - N, k_indices)
    return k_indices * 2 * np.pi / L

def _k_pow(k, n):
    """
    安全地计算 k 的 n 次方，避免除以零的错误。
    对应 `cft.py` 中的 `_K_pow`。
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        # k 是一个包含 kx, ky 的数组，先计算模长
        k_magnitude_sq = (k**2).sum(axis=0)
        k_magnitude = np.sqrt(k_magnitude_sq)
        # 只有当 k_magnitude 不为零时才计算
        result = np.where(k_magnitude == 0, 0, k_magnitude**n)
    return result

# --- 2. 初始条件生成 ---

def generate_initial_potential(shape, L, power_law_n, scale_sigma, seed):
    """
    生成高斯随机引力势场 phi。
    这个函数整合了 `cft.py` 中的多个功能:
    - `Box`: shape 和 L 参数的作用
    - `Power_law`, `Scale`, `Cutoff`: 用于构建功率谱
    - `Potential`: 转换到势场
    - `garfield`: 生成最终的随机场
    """
    # a. 计算波数
    K = _wave_number(shape, L)
    k_magnitude_sq = (K**2).sum(axis=0)
    k_max = shape[0] * np.pi / L

    # b. 定义功率谱 P(k)
    # 对应 `cft.Power_law`, `cft.Scale`, `cft.Cutoff`
    pk = _k_pow(K, power_law_n) # Power_law

    # 对应 `cft.Scale` 滤波器
    for i in range(DIM):
        pk *= np.exp(scale_sigma**2 / (L/shape[0])**2 * (np.cos(K[i] * (L/shape[0])) - 1))

    pk *= np.where(k_magnitude_sq <= k_max**2, 1, 0) # Cutoff

    # c. 生成随机场
    # 对应 `cft.garfield`
    if seed is not None:
        np.random.seed(seed)

    white_noise = np.random.normal(0, 1, shape)
    white_noise_f = np.fft.fftn(white_noise)

    # 将功率谱应用到白噪声上
    field_f = white_noise_f * np.sqrt(pk)

    # d. 转换为势场 (乘以 1/k^2)
    # 对应 `cft.Potential`
    with np.errstate(divide='ignore', invalid='ignore'):
        potential_kernel = -_k_pow(K, -2) # a = -k/k^2

    phi_f = field_f * potential_kernel

    # e. 返回空间场
    return np.fft.ifftn(phi_f).real

def zeldovich_approximation(phi, a_init, shape, L):
    """
    应用泽尔多维奇近似来计算初始位移和速度。
    对应 `nbody.py` 中的 `Zeldovich` 类。
    """
    res = L / shape[0]

    # 计算位移场 u (phi 的负梯度)
    # 对应 `nbody.Zeldovich.u`
    u_x = -gradient_2nd_order(phi, 0) / res
    u_y = -gradient_2nd_order(phi, 1) / res
    u = np.array([u_x, u_y])

    # 初始粒子位置在一个均匀网格上
    # 对应 `nbody.a2r` 和 `np.indices`
    grid_coords = np.indices(shape) * res

    # 应用位移
    initial_pos = grid_coords + a_init * u
    # 将 [dim, N, N] -> [N*N, dim]
    initial_pos = initial_pos.transpose([1, 2, 0]).reshape([shape[0] * shape[1], DIM])

    # 初始动量 P = a * u
    initial_mom = a_init * u
    # 将 [dim, N, N] -> [N*N, dim]
    initial_mom = initial_mom.transpose([1, 2, 0]).reshape([shape[0] * shape[1], DIM])

    return initial_pos, initial_mom

# --- 3. N-Body 模拟核心函数 ---

@numba.jit
def md_cic_2d(shape, pos, tgt):
    """
    Cloud-in-Cell 质量分配。
    直接从 `nbody.py` 中提取，使用 numba 加速。
    """
    for i in range(len(pos)):
        idx0, idx1 = int(np.floor(pos[i,0])), int(np.floor(pos[i,1]))
        f0, f1     = pos[i,0] - idx0, pos[i,1] - idx1
        tgt[idx0 % shape[0], idx1 % shape[1]] += (1 - f0) * (1 - f1)
        tgt[(idx0 + 1) % shape[0], idx1 % shape[1]] += f0 * (1 - f1)
        tgt[idx0 % shape[0], (idx1 + 1) % shape[1]] += (1 - f0) * f1
        tgt[(idx0 + 1) % shape[0], (idx1 + 1) % shape[1]] += f0 * f1

def interp_2d(data, x):
    """
    二维双线性插值。
    对应 `nbody.py` 中的 `Interp2D` 类。
    """
    shape = data.shape
    X1 = np.floor(x).astype(int) % shape
    X2 = np.ceil(x).astype(int) % shape
    xm = x % 1.0
    xn = 1.0 - xm

    f1 = data[X1[:,0], X1[:,1]]
    f2 = data[X2[:,0], X1[:,1]]
    f3 = data[X1[:,0], X2[:,1]]
    f4 = data[X2[:,0], X2[:,1]]

    return  f1 * xn[:,0] * xn[:,1] + \
            f2 * xm[:,0] * xn[:,1] + \
            f3 * xn[:,0] * xm[:,1] + \
            f4 * xm[:,0] * xm[:,1]

def gradient_2nd_order(F, i):
    """
    二阶精度梯度计算。
    直接从 `nbody.py` 中提取。
    """
    return (1./12 * np.roll(F,  2, axis=i) - 2./3  * np.roll(F,  1, axis=i) \
          + 2./3  * np.roll(F, -1, axis=i) - 1./12 * np.roll(F, -2, axis=i))

def da_dt(a, H0, OmegaM, OmegaL, OmegaK):
    """
    计算宇宙尺度因子 a 的时间导数。
    对应 `nbody.py` `Cosmology.da`。
    """
    return H0 * a * np.sqrt(OmegaL + OmegaM * a**-3 + OmegaK * a**-2)

def calculate_momentum_update(pos, a, shape, L, G):
    """
    计算动量更新（即引力）。
    这部分对应 `nbody.py` `PoissonVlasov.momentumEquation`。
    """
    res = L / shape[0]

    # a. 质量分配
    x_grid = pos / res
    delta = np.zeros(shape, dtype='f8')
    md_cic_2d(shape, x_grid, delta)
    delta /= delta.mean() # 归一化粒子数密度为1
    delta -= 1.0 # 计算密度扰动

    # b. 求解泊松方程
    delta_f = np.fft.fftn(delta)
    K = _wave_number(shape, L)
    with np.errstate(divide='ignore', invalid='ignore'):
        potential_kernel = -_k_pow(K, -2)
    phi_f = delta_f * potential_kernel

    # c. 计算引力势 phi，并应用物理常数
    phi = np.fft.ifftn(phi_f).real * G / a

    # d. 计算引力加速度
    acc_x_grid = gradient_2nd_order(phi, 0) / res
    acc_y_grid = gradient_2nd_order(phi, 1) / res

    # e. 插值得到每个粒子的加速度
    acc_x = interp_2d(acc_x_grid, x_grid)
    acc_y = interp_2d(acc_y_grid, x_grid)
    acc = np.c_[acc_x, acc_y]

    # f. 返回动量更新项
    return -acc / da_dt(a, H0, OmegaM, OmegaL, OmegaK)

def calculate_position_update(mom, a):
    """
    计算位置更新。
    对应 `nbody.py` `PoissonVlasov.positionEquation`。
    """
    return mom / (a**2 * da_dt(a, H0, OmegaM, OmegaL, OmegaK))

def run_simulation(initial_pos, initial_mom, a_init, a_final, dt):
    """
    主模拟循环。
    对应 `nbody.py` 的 `iterate_step` 和 `leap_frog` 的逻辑。
    """
    # 初始化状态
    time = a_init
    position = initial_pos.copy()
    momentum = initial_mom.copy()

    # 确保数据目录存在
    if not os.path.exists('data'):
        os.makedirs('data')

    # 保存初始状态
    fn = f'data/x.{int(round(time*1000)):05d}.npy'
    with open(fn, 'wb') as f:
        np.save(f, position)
        np.save(f, momentum)

    # 蛙跳法积分循环
    step_count = 0
    while time < a_final:
        # Leap-frog: kick-drift-kick
        # 1. First half kick
        momentum_update = calculate_momentum_update(position, time, (N, N), L, G_const)
        momentum += dt/2 * momentum_update

        # 2. Full drift
        position_update = calculate_position_update(momentum, time)
        position += dt * position_update

        # 3. Second half kick (at new time t + dt)
        time += dt
        momentum_update = calculate_momentum_update(position, time, (N, N), L, G_const)
        momentum += dt/2 * momentum_update

        # 保存数据
        fn = f'data/x.{int(round(time*1000)):05d}.npy'
        with open(fn, 'wb') as f:
            np.save(f, position)
            np.save(f, momentum)

        step_count += 1
        print(f"Step {step_count}: Simulation time a = {time:.4f}")

    print("Simulation finished.")

# --- 4. 绘图函数 ---

def box_triangles(shape):
    """
    为初始网格创建三角剖分。
    直接从 `nbody/phase_plot.py` 提取。
    """
    size = shape[0] * shape[1]
    idx = np.arange(size, dtype=int).reshape(shape)

    x0 = idx[:-1,:-1]
    x1 = idx[:-1,1:]
    x2 = idx[1:,:-1]
    x3 = idx[1:,1:]

    upper_triangles = np.array([x0, x1, x2]).transpose([1,2,0]).reshape([-1,3])
    lower_triangles = np.array([x3, x2, x1]).transpose([1,2,0]).reshape([-1,3])

    return np.r_[upper_triangles, lower_triangles]

def triangle_area(x, y, t):
    """
    计算三角形面积。
    直接从 `nbody/phase_plot.py` 提取。
    """
    return (x[t[:,0]] * y[t[:,1]] + x[t[:,1]] * y[t[:,2]] + x[t[:,2]] * y[t[:,0]] \
          - x[t[:,1]] * y[t[:,0]] - x[t[:,2]] * y[t[:,1]] - x[t[:,0]] * y[t[:,2]]) / 2

def plot_for_time(shape, res, triangles, time, bbox, ax):
    """
    加载数据并为指定时间绘制相空间图。
    直接从 `nbody/phase_plot.py` 提取并稍作修改。
    """
    fn = f'data/x.{int(round(time*1000)):05d}.npy'
    try:
        with open(fn, "rb") as f:
            x_pos = np.load(f)
            p_mom = np.load(f) # 动量在此次绘图中未使用
    except FileNotFoundError:
        print(f"Error: Data file not found: {fn}")
        return

    # 面积反比于密度
    area = abs(triangle_area(x_pos[:,0], x_pos[:,1], triangles)) / res**2

    # 为避免 log(0) 错误，并且为了更好的可视化，我们对 area 进行排序
    # 面积小的（密度大的）后画，这样能覆盖面积大的（密度小的）
    sorting = np.argsort(area)[::-1]

    # 使用 tripcolor 绘图
    # 颜色表示密度的对数 (1/area)
    ax.tripcolor(x_pos[:,0], x_pos[:,1], triangles[sorting], np.log(1./area[sorting]),
                  alpha=0.5, vmin=-2, vmax=3, cmap='viridis') # 使用 viridis 以获得更清晰的对比
    ax.set_xlim(*bbox[0])
    ax.set_ylim(*bbox[1])
    ax.set_aspect('equal')
    ax.set_facecolor('black')

# def create_final_plot():
#     """
#     创建并保存最终的拼合图。
#     对应 `nbody/phase_plot.py` 的主程序部分。
#     """
#     rcParams["font.family"] = "serif"
#     shape = (N, N)
#     res = L / N
#
#     print("Creating triangles for plotting...")
#     triangles = box_triangles(shape)
#
#     fig, axs = plt.subplots(2, 3, figsize=(15, 10), facecolor='white')
#     fig.suptitle('N-Body Simulation', fontsize=16)
#
#     plot_times = [0.02, 0.2, 1.0]
#
#     # 全局视图
#     bbox_full = [(0, L), (0, L)]
#     for i, t in enumerate(plot_times):
#         print(f"Plotting full view for a = {t}...")
#         plot_for_time(shape, res, triangles, t, bbox=bbox_full, ax=axs[0, i])
#         axs[0, i].set_title(f"a = {t}")
#
#     # 缩放视图
#     bbox_zoom = [(15, 30), (5, 20)]
#     for i, t in enumerate(plot_times):
#         print(f"Plotting zoomed view for a = {t}...")
#         plot_for_time(shape, res, triangles, t, bbox=bbox_zoom, ax=axs[1, i])
#         axs[1, i].set_title(f"Zoomed in, a = {t}")
#
#     # 移除多余的坐标轴标签，使图像更整洁
#     for ax in axs.flat:
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_xticks([])
#         ax.set_yticks([])
#
#     fig.tight_layout(rect=[0, 0.03, 1, 0.95])
#
#     output_filename = 'nbody_phase_plot.png'
#     fig.savefig(output_filename, dpi=300)
#     print(f"Final plot saved as {output_filename}")
#

def plot_density_for_time(shape, L, time, bbox, ax):
    """
    加载数据并为指定时间绘制密度图。
    """
    fn = f'data/x.{int(round(time*1000)):05d}.npy'
    try:
        with open(fn, "rb") as f:
            x_pos = np.load(f)
    except FileNotFoundError:
        print(f"Error: Data file not found: {fn}")
        ax.text(0.5, 0.5, f"Data not found for a={time}", ha='center', va='center', color='white')
        ax.set_facecolor('black')
        return

    # 创建一个用于绘图的网格，可以比模拟网格更精细以获得更平滑的图像
    plot_shape = (shape[0] * 2, shape[1] * 2)
    density_grid = np.zeros(plot_shape, dtype=np.float32)

    # 将粒子位置从物理单位转换为网格单位
    res = L / plot_shape[0]
    x_grid = x_pos / res

    # 将粒子质量分配到网格上
    md_cic_2d(plot_shape, x_grid, density_grid)

    # 使用 imshow 绘制密度场
    # 使用对数色阶来更好地展示结构
    im = ax.imshow(density_grid.T, origin='lower', cmap='viridis',
                   extent=[0, L, 0, L],
                   norm=colors.LogNorm(vmin=0.1, vmax=density_grid.max()))

    ax.set_xlim(*bbox[0])
    ax.set_ylim(*bbox[1])
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    return im

def create_final_density_plot():
    """
    创建并保存最终的拼合密度图。
    """
    rcParams["font.family"] = "serif"
    shape = (N, N)

    fig, axs = plt.subplots(2, 3, figsize=(15, 10), facecolor='white')
    fig.suptitle('N-Body Simulation Density Evolution', fontsize=16)

    plot_times = [0.5, 1.0, 2.0]

    # 全局视图
    bbox_full = [(0, L), (0, L)]
    for i, t in enumerate(plot_times):
        print(f"Plotting full density view for a = {t}...")
        plot_density_for_time(shape, L, t, bbox=bbox_full, ax=axs[0, i])
        axs[0, i].set_title(f"a = {t}")

    # 缩放视图
    bbox_zoom = [(15, 30), (5, 20)]
    for i, t in enumerate(plot_times):
        print(f"Plotting zoomed density view for a = {t}...")
        plot_density_for_time(shape, L, t, bbox=bbox_zoom, ax=axs[1, i])
        axs[1, i].set_title(f"Zoomed in, a = {t}")

    for ax in axs.flat:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_filename = 'nbody_density_plot.png'
    fig.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"Final density plot saved as {output_filename}")

if __name__ == "__main__":
    # --- 执行流程 ---

    # 1. 生成初始条件
    print("Generating initial conditions...")
    initial_shape = (N, N)
    phi = generate_initial_potential(initial_shape, L, POWER_LAW_N, SCALE_SIGMA, SEED)
    phi *= FIELD_AMPLITUDE # 应用振幅

    initial_position, initial_momentum = zeldovich_approximation(phi, A_INIT, initial_shape, L)
    print("Initial conditions generated.")

    # 2. 运行模拟
    print("Starting N-body simulation...")
    run_simulation(initial_position, initial_momentum, A_INIT, A_FINAL, DT)

    # 3. 绘制结果
    print("\nStarting to plot results...")
    create_final_density_plot()
    # create_final_plot()
