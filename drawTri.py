import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams

# --- 模拟参数设置 ---
# 请确保和 C++ 项目中的 parameters.h 一致
N = 256         # 粒子网格的一边大小
L = 50.0        # 模拟盒子的物理尺寸 (Mpc/h)

def box_triangles(shape):
    """根据网格形状生成用于绘图的三角剖分索引。"""
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
    """计算每个三角形的面积，用于估计密度。"""
    return (x[t[:,0]] * y[t[:,1]] + x[t[:,1]] * y[t[:,2]] + x[t[:,2]] * y[t[:,0]] \
          - x[t[:,1]] * y[t[:,0]] - x[t[:,2]] * y[t[:,1]] - x[t[:,0]] * y[t[:,2]]) / 2

def plot_for_time(shape, res, triangles, time, bbox, ax):
    """为给定的时间点读取数据并绘图。"""
    N_total = shape[0] * shape[1]
    
    # --- 主要修改部分 ---
    # 修改文件名以匹配 C++ 的输出 (例如 'data/x.00020.bin')
    fn = f'../build/data/{int(round(time*1000)):05d}.bin'
    
    try:
        # 使用 np.fromfile 读取原始二进制数据。数据类型为 double，对应 np.float64
        raw_data = np.fromfile(fn, dtype=np.float64)
        
        # C++ 先写入 N_total * 2 个 double 作为位置数据
        pos_count = N_total * 2
        x_pos = raw_data[0 : pos_count].reshape((N_total, 2))
        
        # 接着写入 N_total * 2 个 double 作为动量数据
        p_mom = raw_data[pos_count :].reshape((N_total, 2))

    except FileNotFoundError:
        print(f"错误: 未找到数据文件: {fn}")
        return
    except ValueError:
        print(f"错误: 文件 {fn} 的大小不正确或已损坏。请确保 C++ 和 Python 中的粒子数 N 匹配。")
        return
    # --- 修改结束 ---

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


if __name__ == '__main__':
    rcParams["font.family"] = "serif"
    shape = (N, N)
    res = L / N

    triangles = box_triangles(shape)

    fig, axs = plt.subplots(2, 3, figsize=(15, 10), facecolor='white')
    fig.suptitle('N-Body Simulation', fontsize=16)

    # 选择要绘制的时间点
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
    print(f"绘图已保存为: {output_filename}")
    plt.show()
