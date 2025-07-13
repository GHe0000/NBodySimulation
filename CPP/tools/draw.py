import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams

# --- 模拟参数设置 ---
# 请确保和 C++ 项目中的 parameters.h 一致
N = 256         # 粒子网格的一边大小
L = 50.0        # 模拟盒子的物理尺寸 (Mpc/h)

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

def plot_for_time(shape, res, triangles, time, bbox, ax):
    N_total = shape[0] * shape[1]
    fn = f'../build/data/{int(round(time*1000)):05d}.bin'
    
    try:
        raw_data = np.fromfile(fn, dtype=np.float64)
        pos_count = N_total * 2
        x_pos = raw_data[0 : pos_count].reshape((N_total, 2))
        p_mom = raw_data[pos_count :].reshape((N_total, 2))

    except FileNotFoundError:
        print(f"File not found: {fn}")
        return
    except ValueError:
        print(f"Open {fn} failed.")
        return

    area = abs(triangle_area(x_pos[:,0], x_pos[:,1], triangles)) / res**2

    sorting = np.argsort(area)[::-1]

    ax.tripcolor(x_pos[:,0], x_pos[:,1], triangles[sorting], np.log(1./area[sorting]),
                  alpha=0.5, vmin=-2, vmax=3, cmap='viridis') # 使用 viridis 以获得更清晰的对比
    ax.set_xlim(*bbox[0])
    ax.set_ylim(*bbox[1])
    ax.set_aspect('equal')
    ax.set_facecolor('black')


if __name__ == '__main__':
    rcParams["font.family"] = "serif"
    shape = (N, N)
    res = L / N

    triangles = box_triangles(shape)

    fig, axs = plt.subplots(2, 3, figsize=(15, 10), facecolor='white')
    fig.suptitle('N-Body Simulation', fontsize=16)

    plot_times = [0.02, 0.2, 1.0]

    # 全局视图
    bbox_full = [(0, L), (0, L)]
    for i, t in enumerate(plot_times):
        plot_for_time(shape, res, triangles, t, bbox=bbox_full, ax=axs[0, i])
        axs[0, i].set_title(f"a = {t}")

    # 局部放大视图
    bbox_zoom = [(15, 30), (5, 20)]
    for i, t in enumerate(plot_times):
        plot_for_time(shape, res, triangles, t, bbox=bbox_zoom, ax=axs[1, i])
        axs[1, i].set_title(f"Zoomed in, a = {t}")

    # 清理坐标轴
    for ax in axs.flat:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_filename = 'nbody_plot.png'
    fig.savefig(output_filename, dpi=150)
    print(f"Saved plot as {output_filename}")
    plt.show()
