import os
import glob
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import rcParams

# --- 基本参数 (需要与模拟脚本中的设置保持一致) ---
N = 256  # 粒子网格的一边大小
L = 50.0  # 模拟盒子的物理尺寸 (Mpc/h)

def box_triangles(shape):
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
    return (x[t[:,0]] * y[t[:,1]] + x[t[:,1]] * y[t[:,2]] + x[t[:,2]] * y[t[:,0]] \
          - x[t[:,1]] * y[t[:,0]] - x[t[:,2]] * y[t[:,1]] - x[t[:,0]] * y[t[:,2]]) / 2


if __name__ == "__main__":
    rcParams["font.family"] = "serif"

    data_dir = 'data'
    if not os.path.isdir(data_dir):
        print(f"'{data_dir}' not found.")
        exit()

    files = sorted(glob.glob(os.path.join(data_dir, '*.npy')))

    print(f"Find {len(files)} files.")

    fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_xticks([])
    ax.set_yticks([])

    triangles = box_triangles((N, N))
    res = L / N

    time_text = ax.text(0.02, 0.95, '', color='white', transform=ax.transAxes, fontsize=14)

    def init():
        time_text.set_text('')
        return ax.patches + [time_text] # 返回一个可迭代的对象

    # 3. 定义动画更新函数
    def update(frame):
        ax.clear() # 因为 tripcolor 会创建新的对象，所以每帧都清空
        ax.set_facecolor('black')
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.set_xticks([])
        ax.set_yticks([])

        filepath = files[frame]

        time_ms = int(os.path.basename(filepath).split('.')[0])
        current_time = time_ms / 1000.0

        with open(filepath, "rb") as f:
            x_pos = np.load(f)

        area = abs(triangle_area(x_pos[:,0], x_pos[:,1], triangles)) / res**2
        sorting = np.argsort(area)[::-1]

        ax.tripcolor(x_pos[:,0], x_pos[:,1], triangles[sorting], np.log(1./area[sorting]),
                      alpha=0.5, vmin=-2, vmax=3, cmap='viridis')

        time_text = ax.text(0.02, 0.95, f'a = {current_time:.2f}', color='white', transform=ax.transAxes, fontsize=14)

        print(f'Gen: {frame + 1}/{len(files)}')
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(files), init_func=init, blit=False)

    output_filename = 'nbody.mp4'
    try:
        anim.save(output_filename, writer='ffmpeg', fps=15, dpi=300,
                  progress_callback=lambda i, n: print(f'Save: {i+1}/{n}'))
        print(f"\nSave in file: '{output_filename}'")
    except Exception as e:
        print(f"\nSave error: {e}")

