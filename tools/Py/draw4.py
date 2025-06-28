#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
本脚本用于将 N-Body 模拟产生的一系列数据文件渲染成一个动画。

此版本通过高斯模糊处理密度场，以平滑的区域平均密度（热力图）来展示宇宙结构的形成。

使用前请确保：
1. `data` 文件夹与本脚本在同一目录。
2. 您的系统中已经安装了 `ffmpeg` 和 `scipy`。
"""

import os
import glob
import numpy as np
import numba
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import colors
from matplotlib import rcParams
from scipy.ndimage import gaussian_filter # 导入高斯滤波器

# --- 基本参数 (需要与模拟脚本中的设置保持一致) ---
N = 256  # 粒子网格的一边大小
L = 50.0  # 模拟盒子的物理尺寸 (Mpc/h)
GAUSSIAN_SIGMA = 0.6 # 高斯模糊的强度，可以调整此值

# --- 核心绘图函数 ---

@numba.jit
def md_cic_2d(shape, pos, tgt):
    """
    Cloud-in-Cell 质量分配。
    用于将粒子位置转换为网格密度。
    """
    # 清空目标数组
    tgt.fill(0)
    for i in range(len(pos)):
        idx0, idx1 = int(np.floor(pos[i, 0])), int(np.floor(pos[i, 1]))
        f0, f1 = pos[i, 0] - idx0, pos[i, 1] - idx1
        tgt[idx0 % shape[0], idx1 % shape[1]] += (1 - f0) * (1 - f1)
        tgt[(idx0 + 1) % shape[0], idx1 % shape[1]] += f0 * (1 - f1)
        tgt[idx0 % shape[0], (idx1 + 1) % shape[1]] += (1 - f0) * f1
        tgt[(idx0 + 1) % shape[0], (idx1 + 1) % shape[1]] += f0 * f1


# --- 动画制作主程序 ---

if __name__ == "__main__":
    rcParams["font.family"] = "serif"

    # 1. 查找所有数据文件
    data_dir = 'data'
    if not os.path.isdir(data_dir):
        print(f"错误: 未找到 '{data_dir}' 目录。")
        print("请先运行模拟脚本生成数据。")
        exit()

    files = sorted(glob.glob(os.path.join(data_dir, '*.npy')))

    if not files:
        print(f"错误: '{data_dir}' 目录中没有找到 .npy 数据文件。")
        exit()

    print(f"找到了 {len(files)} 个数据文件，将开始渲染动画...")

    # 2. 设置 Matplotlib 动画
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    ax.set_xticks([])
    ax.set_yticks([])

    # 创建一个用于绘图的网格
    plot_shape = (N * 2, N * 2)
    density_grid = np.zeros(plot_shape, dtype=np.float32)
    res = L / plot_shape[0]

    # 初始化图像对象 (imshow)
    im = ax.imshow(density_grid.T, origin='lower', cmap='inferno', # 使用 inferno 色彩映射
                   extent=[0, L, 0, L],
                   norm=colors.LogNorm(vmin=0.2, vmax=100)) # vmin/vmax可能需要根据数据调整

    # 添加一个动态文本来显示时间
    time_text = ax.text(0.02, 0.95, '', color='white', transform=ax.transAxes, fontsize=14)

    # 3. 定义动画更新函数
    def update(frame):
        """
        这个函数会在每一帧被调用。
        `frame` 是当前帧的索引。
        """
        filepath = files[frame]

        # 从文件名中提取时间 (尺度因子 a)
        time_ms = int(os.path.basename(filepath).split('.')[1])
        current_time = time_ms / 1000.0

        # 加载粒子数据
        with open(filepath, "rb") as f:
            # 文件包含位置和动量，我们只需要第一个数组（位置）
            x_pos = np.load(f)

        # 计算密度
        x_grid = x_pos / res
        md_cic_2d(plot_shape, x_grid, density_grid)

        # 对密度场应用高斯模糊以获得区域平均效果
        smoothed_density = gaussian_filter(density_grid, sigma=GAUSSIAN_SIGMA)

        # 更新图像数据和文本
        im.set_data(smoothed_density.T)
        time_text.set_text(f'a = {current_time:.2f}')

        print(f'正在渲染第 {frame + 1}/{len(files)} 帧...')
        return im, time_text

    # 4. 创建并保存动画
    anim = animation.FuncAnimation(fig, update, frames=len(files), blit=True)

    # 保存为 MP4 文件
    output_filename = 'nbody_evolution_smoothed_heatmap.mp4'
    try:
        anim.save(output_filename, writer='ffmpeg', fps=15, dpi=150,
                  progress_callback=lambda i, n: print(f'正在保存视频: {i+1}/{n}'))
        print(f"\n动画已成功保存为 '{output_filename}'")
    except FileNotFoundError:
        print("\n错误: 未找到 `ffmpeg`。")
        print("请确保您已安装 ffmpeg 并将其添加到了系统路径中。")
    except Exception as e:
        print(f"\n保存动画时发生错误: {e}")
