
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
本脚本用于将 N-Body 模拟产生的一系列数据文件渲染成一个动画。

此版本使用三角剖分样式进行渲染，以动态展示相空间演化。

使用前请确保：
1. `data` 文件夹与本脚本在同一目录。
2. 您的系统中已经安装了 `ffmpeg`。
"""

import os
import glob
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import rcParams

# --- 基本参数 (需要与模拟脚本中的设置保持一致) ---
N = 256  # 粒子网格的一边大小
L = 50.0  # 模拟盒子的物理尺寸 (Mpc/h)

# --- 核心绘图函数 ---

def box_triangles(shape):
    """
    为初始网格创建三角剖分。
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
    """
    return (x[t[:,0]] * y[t[:,1]] + x[t[:,1]] * y[t[:,2]] + x[t[:,2]] * y[t[:,0]] \
          - x[t[:,1]] * y[t[:,0]] - x[t[:,2]] * y[t[:,1]] - x[t[:,0]] * y[t[:,2]]) / 2


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
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_xticks([])
    ax.set_yticks([])

    # 预先计算三角剖分
    triangles = box_triangles((N, N))
    res = L / N

    # 添加一个动态文本来显示时间
    time_text = ax.text(0.02, 0.95, '', color='white', transform=ax.transAxes, fontsize=14)

    # 初始化函数 (对于 `blit=False` 不是严格必需的，但良好实践)
    def init():
        time_text.set_text('')
        return ax.patches + [time_text] # 返回一个可迭代的对象

    # 3. 定义动画更新函数
    def update(frame):
        """
        这个函数会在每一帧被调用。
        `frame` 是当前帧的索引。
        """
        ax.clear() # 因为 tripcolor 会创建新的对象，所以每帧都清空
        ax.set_facecolor('black')
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.set_xticks([])
        ax.set_yticks([])

        filepath = files[frame]

        # 从文件名中提取时间 (尺度因子 a)
        time_ms = int(os.path.basename(filepath).split('.')[1])
        current_time = time_ms / 1000.0

        # 加载粒子数据
        with open(filepath, "rb") as f:
            x_pos = np.load(f)

        # 计算每个三角形的面积作为密度的代理
        area = abs(triangle_area(x_pos[:,0], x_pos[:,1], triangles)) / res**2
        sorting = np.argsort(area)[::-1]

        # 绘制三角剖分图
        ax.tripcolor(x_pos[:,0], x_pos[:,1], triangles[sorting], np.log(1./area[sorting]),
                      alpha=0.5, vmin=-2, vmax=3, cmap='viridis')

        # 更新时间文本
        time_text = ax.text(0.02, 0.95, f'a = {current_time:.2f}', color='white', transform=ax.transAxes, fontsize=14)

        print(f'正在渲染第 {frame + 1}/{len(files)} 帧...')
        # 因为我们没有使用 blit，所以不需要返回任何东西
        return []

    # 4. 创建并保存动画
    # 注意：对于 tripcolor 这种每帧都重绘的图，blit=False 通常更简单可靠
    anim = animation.FuncAnimation(fig, update, frames=len(files), init_func=init, blit=False)

    # 保存为 MP4 文件
    output_filename = 'nbody_evolution_triangulation.mp4'
    try:
        anim.save(output_filename, writer='ffmpeg', fps=15, dpi=150,
                  progress_callback=lambda i, n: print(f'正在保存视频: {i+1}/{n}'))
        print(f"\n动画已成功保存为 '{output_filename}'")
    except FileNotFoundError:
        print("\n错误: 未找到 `ffmpeg`。")
        print("请确保您已安装 ffmpeg 并将其添加到了系统路径中。")
    except Exception as e:
        print(f"\n保存动画时发生错误: {e}")

