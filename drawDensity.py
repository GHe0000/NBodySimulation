import numpy as np
import numba
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib import colors

# --- 模拟参数设置 ---
# 请确保和计算时所用的参数一致
N = 256          # 粒子网格的一边大小
L = 50.0         # 模拟盒子的物理尺寸 (Mpc/h)

@numba.jit
def cic(shape, pos, tgt):
    for i in range(len(pos)):
        idx0, idx1 = int(np.floor(pos[i,0])), int(np.floor(pos[i,1]))
        f0, f1     = pos[i,0] - idx0, pos[i,1] - idx1
        tgt[idx0 % shape[0], idx1 % shape[1]] += (1 - f0) * (1 - f1)
        tgt[(idx0 + 1) % shape[0], idx1 % shape[1]] += f0 * (1 - f1)
        tgt[idx0 % shape[0], (idx1 + 1) % shape[1]] += (1 - f0) * f1
        tgt[(idx0 + 1) % shape[0], (idx1 + 1) % shape[1]] += f0 * f1

def plot_density_for_time(shape, L, time, bbox, ax):
    fn = f'data/{int(round(time*1000)):05d}.npy'
    try:
        with open(fn, "rb") as f:
            x_pos = np.load(f)
    except FileNotFoundError:
        print(f"Error: Data file not found: {fn}")
        ax.text(0.5, 0.5, f"Data not found for a={time}", ha='center', va='center', color='white')
        ax.set_facecolor('black')
        return

    plot_shape = (shape[0] * 2, shape[1] * 2)
    density_grid = np.zeros(plot_shape, dtype=np.float32)

    res = L / plot_shape[0]
    x_grid = x_pos / res

    cic(plot_shape, x_grid, density_grid)

    im = ax.imshow(density_grid.T, origin='lower', cmap='viridis',
                   extent=[0, L, 0, L],
                   norm=colors.LogNorm(vmin=0.1, vmax=density_grid.max()))

    ax.set_xlim(*bbox[0])
    ax.set_ylim(*bbox[1])
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    return im

if __name__ == '__main__':
    rcParams["font.family"] = "serif"
    shape = (N, N)

    fig, axs = plt.subplots(2, 3, figsize=(15, 10), facecolor='white')
    fig.suptitle('N-Body Simulation', fontsize=16)

    plot_times = [0.02, 0.2, 1.0]

    bbox_full = [(0, L), (0, L)]
    for i, t in enumerate(plot_times):
        plot_density_for_time(shape, L, t, bbox=bbox_full, ax=axs[0, i])
        axs[0, i].set_title(f"a = {t}")

    bbox_zoom = [(15, 30), (5, 20)]
    for i, t in enumerate(plot_times):
        plot_density_for_time(shape, L, t, bbox=bbox_zoom, ax=axs[1, i])
        axs[1, i].set_title(f"Zoomed in, a = {t}")

    for ax in axs.flat:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_filename = 'nbody_density_plot.png'
    fig.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_filename}")
    plt.show()
