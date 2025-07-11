import numpy as np
from numba import njit, prange
import time
import sys

# --- 模拟参数 ---
# 这些参数应与你的 Manim 脚本保持一致
N_PARTICLES = 5
G = 2
DAMPING_FACTOR = 0.999
TOTAL_DURATION = 30
SOFTENING_FACTOR = 0.01
POSITION_RANGE = 5.0
# 【新增】控制施加在稳定解上的扰动强度
PERTURBATION_STRENGTH = 0.1

# 模拟的时间步长 (dt) 和总步数
FRAME_RATE = 60
DT = 1.0 / FRAME_RATE
TOTAL_STEPS = int(TOTAL_DURATION * FRAME_RATE)

@njit(fastmath=True, parallel=True)
def run_simulation(positions, velocities, masses):
    """
    使用 Numba 加速的核心模拟函数。
    """
    pos = positions.copy()
    vel = velocities.copy()

    for _ in range(TOTAL_STEPS):
        forces = np.zeros_like(pos)
        for i in prange(N_PARTICLES):
            for j in range(N_PARTICLES):
                if i == j:
                    continue
                r_vec = pos[j] - pos[i]
                dist = np.sqrt(r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2) + SOFTENING_FACTOR
                force_magnitude = (G * masses[i] * masses[j]) / (dist * dist)
                force_vector = force_magnitude * (r_vec / dist)
                forces[i] += force_vector
        
        for i in prange(N_PARTICLES):
            acceleration = forces[i] / masses[i]
            vel[i] += acceleration * DT
            vel[i] *= DAMPING_FACTOR
            pos[i] += vel[i] * DT
    return pos

@njit(fastmath=True)
def calculate_max_distance(positions):
    """
    计算所有粒子对之间的最大距离。
    """
    max_dist_sq = 0.0
    for i in range(N_PARTICLES):
        for j in range(i + 1, N_PARTICLES):
            r_vec = positions[j] - positions[i]
            dist_sq = r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2
            if dist_sq > max_dist_sq:
                max_dist_sq = dist_sq
    return np.sqrt(max_dist_sq)

def find_best_seed():
    """
    主函数，循环搜索能产生最稳定系统的扰动。
    """
    best_seed = -1
    min_max_distance = np.inf
    seed_counter = 0

    print("开始搜索最佳初始条件种子...")
    print("方法: 对称稳定轨道 + 随机微扰")
    print("目标: 最小化粒子间的最大距离")
    print("按 Ctrl+C 停止搜索。")
    
    try:
        while True:
            current_seed = seed_counter
            np.random.seed(current_seed)

            # --- 【已修改】生成稳定且带有扰动的初始条件 ---

            # 1. 生成质量，让第一个粒子成为大质量的“太阳”
            masses = np.random.uniform(1.0, 3.0, (N_PARTICLES,)).astype(np.float64)
            masses[0] = np.random.uniform(20.0, 30.0) # 中心粒子质量远大于其他

            # 2. 设置一个理论上稳定的圆形轨道构型
            R = POSITION_RANGE * 0.6  # 轨道半径
            central_mass = masses[0]
            # 计算稳定轨道所需的速度 v = sqrt(G*M/R)
            orbital_v = np.sqrt(G * central_mass / R)
            
            initial_positions = np.zeros((N_PARTICLES, 3), dtype=np.float64)
            initial_velocities = np.zeros((N_PARTICLES, 3), dtype=np.float64)

            # 围绕中心粒子对称放置其余“行星”
            num_orbiting = N_PARTICLES - 1
            for i in range(num_orbiting):
                angle = 2 * np.pi * i / num_orbiting
                # 设置位置
                px = R * np.cos(angle)
                py = R * np.sin(angle)
                initial_positions[i+1] = [px, py, 0]
                # 设置实现圆周运动的切向速度
                vx = -orbital_v * np.sin(angle)
                vy = orbital_v * np.cos(angle)
                initial_velocities[i+1] = [vx, vy, 0]

            # 3. 对稳定构型施加随机微扰
            pos_perturb = np.random.uniform(-1, 1, (N_PARTICLES, 3)) * PERTURBATION_STRENGTH
            vel_perturb = np.random.uniform(-1, 1, (N_PARTICLES, 3)) * PERTURBATION_STRENGTH
            pos_perturb[:, 2] = 0  # 保持2D
            vel_perturb[:, 2] = 0

            initial_positions += pos_perturb
            initial_velocities += vel_perturb

            # 4. 【关键】在施加扰动后，重新将系统总动量归零，以防整体漂移
            total_mass = np.sum(masses)
            weighted_velocities = np.sum(initial_velocities * masses.reshape(-1, 1), axis=0)
            center_of_mass_velocity = weighted_velocities / total_mass
            initial_velocities -= center_of_mass_velocity
            
            # --- 初始条件生成结束 ---

            # 运行模拟并评估结果
            final_positions = run_simulation(initial_positions, initial_velocities, masses)
            final_distance = calculate_max_distance(final_positions)

            # 检查是否是更优的结果
            if final_distance < min_max_distance:
                min_max_distance = final_distance
                best_seed = current_seed
                sys.stdout.write(f"\r已搜索 {seed_counter + 1} 个种子 | 新的最优种子: {best_seed} | 最小最大距离: {min_max_distance:.4f}  ")
                sys.stdout.flush()

            seed_counter += 1

    except KeyboardInterrupt:
        print(f"\n\n搜索停止。")
        print(f"总共搜索了 {seed_counter} 个种子。")
        print(f"找到的最佳种子是: {best_seed}")
        print(f"对应的最小最大距离是: {min_max_distance:.4f}")

if __name__ == "__main__":
    # 第一次运行时 Numba 需要一些时间来编译，后续运行会很快
    find_best_seed()
