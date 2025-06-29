import os
import numpy as np
import numba

# --- 模拟参数设置 ---
N = 256          # 粒子网格的一边大小
L = 50.0         # 模拟盒子的物理尺寸 (Mpc/h)
BOX_RES = L / N  # 盒子分辨率
DIM = 2          # 维度

H0 = 68.0
OmegaM = 0.31
OmegaL = 0.69
OmegaK = 1 - OmegaM - OmegaL # OmegaK = 0.0
G_const = 3./2 * OmegaM * H0**2

A_INIT = 0.02    # 初始尺度因子
A_FINAL = 1.0    # 终止尺度因子
DT = 0.005        # 时间步长

# NOTE: 模拟中的功率谱和现实宇宙中的不同，这里用了
#       一个及其简单的基于幂律分布的功率谱
POWER_LAW_N = -0.5 # 功率谱指数
SCALE_SIGMA = 0.2  # 平滑尺度
FIELD_AMPLITUDE = 10.0 # 场振幅
SEED = 4


def wave_number(shape, L):
    N = shape[0]
    k_indices = np.indices(shape)
    k_indices = np.where(k_indices > N / 2, k_indices - N, k_indices)
    return k_indices * 2 * np.pi / L

def k_pow(k, n):
    with np.errstate(divide='ignore', invalid='ignore'):
        # k 是一个包含 kx, ky 的数组，先计算模长
        k_magnitude_sq = (k**2).sum(axis=0)
        k_magnitude = np.sqrt(k_magnitude_sq)
        result = np.where(k_magnitude == 0, 0, k_magnitude**n)
    return result

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
    K = wave_number(shape, L)
    k_magnitude_sq = (K**2).sum(axis=0)
    k_max = shape[0] * np.pi / L

    # b. 定义功率谱 P(k)
    # 对应 `cft.Power_law`, `cft.Scale`, `cft.Cutoff`
    pk = k_pow(K, power_law_n) # Power_law

    for i in range(DIM):
        pk *= np.exp(scale_sigma**2 / (L/shape[0])**2 * (np.cos(K[i] * (L/shape[0])) - 1))

    pk *= np.where(k_magnitude_sq <= k_max**2, 1, 0) # Cutoff

    if seed is not None:
        np.random.seed(seed)

    white_noise = np.random.normal(0, 1, shape)
    white_noise_f = np.fft.fftn(white_noise)

    field_f = white_noise_f * np.sqrt(pk)

    # with np.errstate(divide='ignore', invalid='ignore'):
    #     potential_kernel = -k_pow(K, -2)
    potential_kernel = -k_pow(K, -2)

    phi_f = field_f * potential_kernel

    return np.fft.ifftn(phi_f).real

def zeldovich(phi, a_init, shape, L):
    res = L / shape[0]

    u_x = -gradient_2nd_order(phi, 0) / res
    u_y = -gradient_2nd_order(phi, 1) / res
    u = np.array([u_x, u_y])

    grid_coords = np.indices(shape) * res

    initial_pos = grid_coords + a_init * u
    initial_pos = initial_pos.transpose([1, 2, 0]).reshape([shape[0] * shape[1], DIM])

    initial_mom = a_init * u
    initial_mom = initial_mom.transpose([1, 2, 0]).reshape([shape[0] * shape[1], DIM])

    return initial_pos, initial_mom


@numba.jit
def cic(shape, pos, tgt):
    for i in range(len(pos)):
        idx0, idx1 = int(np.floor(pos[i,0])), int(np.floor(pos[i,1]))
        f0, f1     = pos[i,0] - idx0, pos[i,1] - idx1
        tgt[idx0 % shape[0], idx1 % shape[1]] += (1 - f0) * (1 - f1)
        tgt[(idx0 + 1) % shape[0], idx1 % shape[1]] += f0 * (1 - f1)
        tgt[idx0 % shape[0], (idx1 + 1) % shape[1]] += (1 - f0) * f1
        tgt[(idx0 + 1) % shape[0], (idx1 + 1) % shape[1]] += f0 * f1

def interp(data, x):
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
    return (1./12 * np.roll(F,  2, axis=i) - 2./3  * np.roll(F,  1, axis=i) \
          + 2./3  * np.roll(F, -1, axis=i) - 1./12 * np.roll(F, -2, axis=i))

def da_dt(a, H0, OmegaM, OmegaL, OmegaK):
    return H0 * a * np.sqrt(OmegaL + OmegaM * a**-3 + OmegaK * a**-2)

def calculate_momentum_update(pos, a, shape, L, G):
    res = L / shape[0]

    x_grid = pos / res
    delta = np.zeros(shape, dtype='f8')
    cic(shape, x_grid, delta)

    # 计算扰动场
    delta /= delta.mean()
    delta -= 1.0

    delta_f = np.fft.fftn(delta)
    K = wave_number(shape, L)
    potential_kernel = -k_pow(K, -2)
    # with np.errstate(divide='ignore', invalid='ignore'):
    #     potential_kernel = -k_pow(K, -2)
    phi_f = delta_f * potential_kernel

    phi = np.fft.ifftn(phi_f).real * G / a

    acc_x_grid = gradient_2nd_order(phi, 0) / res
    acc_y_grid = gradient_2nd_order(phi, 1) / res

    acc_x = interp(acc_x_grid, x_grid)
    acc_y = interp(acc_y_grid, x_grid)
    acc = np.c_[acc_x, acc_y]

    return -acc / da_dt(a, H0, OmegaM, OmegaL, OmegaK)

def calculate_position_update(mom, a):
    return mom / (a**2 * da_dt(a, H0, OmegaM, OmegaL, OmegaK))


if __name__ == "__main__":
    print("Generating initial conditions...")
    initial_shape = (N, N)
    phi = generate_initial_potential(initial_shape, L, POWER_LAW_N, SCALE_SIGMA, SEED)
    phi *= FIELD_AMPLITUDE # 应用振幅

    initial_position, initial_momentum = zeldovich(phi, A_INIT, initial_shape, L)

    print("Starting N-body simulation...")

    time = A_INIT
    position = initial_position.copy()
    momentum = initial_momentum.copy()

    if not os.path.exists('data'):
        os.makedirs('data')

    fn = f'data/{int(round(time*1000)):05d}.npy'
    with open(fn, 'wb') as f:
        np.save(f, position)
        np.save(f, momentum)

    # 蛙跳法积分循环
    step_count = 0
    while time < A_FINAL:
        momentum_update = calculate_momentum_update(position, time, (N, N), L, G_const)
        momentum += DT/2 * momentum_update

        position_update = calculate_position_update(momentum, time)
        position += DT * position_update

        time += DT
        momentum_update = calculate_momentum_update(position, time, (N, N), L, G_const)
        momentum += DT/2 * momentum_update

        fn = f'data/{int(round(time*1000)):05d}.npy'
        with open(fn, 'wb') as f:
            np.save(f, position)
            np.save(f, momentum)

        step_count += 1
        print(f"Step {step_count}: a = {time:.4f}")
    print("Simulation finished.")
