#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
本脚本用于将 N-Body 模拟产生的一系列数据文件渲染成一个动画。

它会查找 'data/' 目录下的所有 '.npy' 文件，
然后逐帧绘制密度图，并最终将它们合成为一个 MP4 视频文件。

使用前请确保：
1. `data` 文件夹与本脚本在同一目录。
2. 您的系统中已经安装了 `ffmpeg`。
"""

import os
import glob
import numpy as np
import numba
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import colors
from matplotlib import rcParams

# --- 基本参数 (需要与模拟脚本中的设置保持一致) ---
N = 256  # 粒子网格的一边大小
L = 50.0  # 模拟盒子的物理尺寸 (Mpc/h)

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

    # 获取所有 .npy 文件并按文件名排序
    # 文件名中的数字代表时间步，排序确保了动画的时间顺序正确
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

    # 创建一个用于绘图的网格，可以比模拟网格更精细以获得更平滑的图像
    plot_shape = (N * 2, N * 2)
    density_grid = np.zeros(plot_shape, dtype=np.float32)

    # 预加载第一帧以设置色阶范围
    with open(files[0], "rb") as f:
        x_pos_init = np.load(f)
    res = L / plot_shape[0]
    x_grid_init = x_pos_init / res
    md_cic_2d(plot_shape, x_grid_init, density_grid)

    # 初始化图像对象
    im = ax.imshow(density_grid.T, origin='lower', cmap='viridis',
                   extent=[0, L, 0, L],
                   norm=colors.LogNorm(vmin=0.1, vmax=density_grid.max() * 0.5)) # 适当调整vmax以获得更好效果

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
        # 例如 'data/x.02000.npy' -> 2000 -> 2.0
        time_ms = int(os.path.basename(filepath).split('.')[1])
        current_time = time_ms / 1000.0

        # 加载粒子数据
        with open(filepath, "rb") as f:
            x_pos = np.load(f)

        # 计算密度
        x_grid = x_pos / res
        md_cic_2d(plot_shape, x_grid, density_grid)

        # 更新图像数据和文本
        im.set_data(density_grid.T)
        time_text.set_text(f'a = {current_time:.2f}')

        print(f'正在渲染第 {frame + 1}/{len(files)} 帧...')
        return im, time_text

    # 4. 创建并保存动画
    # `FuncAnimation` 会调用 `update` 函数来生成每一帧
    anim = animation.FuncAnimation(fig, update, frames=len(files), blit=True)

    # 保存为 MP4 文件
    output_filename = 'nbody_evolution.mp4'
    try:
        anim.save(output_filename, writer='ffmpeg', fps=15, dpi=150,
                  progress_callback=lambda i, n: print(f'正在保存视频: {i+1}/{n}'))
        print(f"\n动画已成功保存为 '{output_filename}'")
    except FileNotFoundError:
        print("\n错误: 未找到 `ffmpeg`。")
        print("请确保您已安装 ffmpeg 并将其添加到了系统路径中。")
    except Exception as e:
        print(f"\n保存动画时发生错误: {e}")
