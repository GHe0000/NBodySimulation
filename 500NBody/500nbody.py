import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numba
from tqdm import tqdm

# --- 模拟参数 ---
N = 500
R = 1.0
C_SOFTENING = 0.05
DT = 0.001
HALO_MASS_RATIO = 2.5 # 修改此值来模拟有或无暗物质晕的情况
SIGMA_CONST = 0.4 # Toomre Q 常数
G = 1.0
PARTICLE_MASS = 1.0

@numba.njit(fastmath=True)
def calculate_accelerations(pos, mass, halo_mass, G, softening, disk_radius):
    n_particles = pos.shape[0]
    acc = np.zeros((n_particles, 3))

    # 粒子间引力加速度
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            r_vec = pos[j] - pos[i]
            dist_sq = r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2
            inv_r3 = (dist_sq + softening**2)**(-1.5)
            force_mag = G * mass[i] * mass[j] * inv_r3
            force_vec = force_mag * r_vec
            acc[i] += force_vec / mass[i]
            acc[j] -= force_vec / mass[j]

    # 暗物质晕引力加速度
    if halo_mass > 0:
        for i in range(n_particles):
            r_vec = pos[i]
            r_mag = np.sqrt(r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2)
            if r_mag == 0:
                continue
            
            if r_mag < disk_radius:
                scalar_acc = -(1.1**2 * halo_mass) / (disk_radius * (r_mag + 0.1 * disk_radius)**2)
            else:
                scalar_acc = - (halo_mass / r_mag**3)

            acc[i] += scalar_acc * r_vec
            
    return acc

def setup_initial_conditions(n, r_disk, m_particle, sigma_k):
    # --- 0. 计算质量并准备数组 ---
    disk_mass_total = n * m_particle
    halo_mass = disk_mass_total * HALO_MASS_RATIO
    pos = np.zeros((n, 3))
    vel = np.zeros((n, 3))
    mass = np.ones(n) * m_particle

    # --- 1. 设置初始位置 ---
    n_rings = n // 10
    n_segments = 10
    particles_per_ring = n_segments
    
    for i in range(n_rings):
        r_min = i * (r_disk / n_rings)
        r_max = (i + 1) * (r_disk / n_rings)
        for j in range(n_segments):
            phi_min = j * (2 * np.pi / n_segments)
            phi_max = (j + 1) * (2 * np.pi / n_segments)
            
            rand_r = np.sqrt(np.random.uniform(r_min**2, r_max**2))
            rand_phi = np.random.uniform(phi_min, phi_max)
            
            idx = i * particles_per_ring + j
            pos[idx, 0] = rand_r * np.cos(rand_phi)
            pos[idx, 1] = rand_r * np.sin(rand_phi)
            pos[idx, 2] = 0.0

    # --- 2. 设置基础旋转速度 ---
    # 加速度计算包含暗物质晕，以保证平衡正确
    initial_acc = calculate_accelerations(pos, mass, halo_mass, G, C_SOFTENING, r_disk)
    
    radii = np.sqrt(pos[:,0]**2 + pos[:,1]**2)
    ring_indices = (radii / (r_disk / n_rings)).astype(int)
    # FIX: 防止粒子在盘面最外边缘时索引越界
    ring_indices[ring_indices >= n_rings] = n_rings - 1
    
    v_circ_profile = np.zeros(n_rings)
    for i in range(n_rings):
        mask = (ring_indices == i)
        if not np.any(mask): continue
        
        r_dot_a = pos[mask, 0] * initial_acc[mask, 0] + pos[mask, 1] * initial_acc[mask, 1]
        
        # FIX: 防止因数值误差导致均值为负，从而开方失败产生NaN
        mean_val = np.mean(-r_dot_a)
        v_circ_profile[i] = np.sqrt(max(0, mean_val))

    for i in range(n):
        # FIX: 跳过中心粒子，防止除以零
        if radii[i] == 0: continue
        ring_idx = ring_indices[i]
        v_mag = v_circ_profile[ring_idx]
        vel[i, 0] = -v_mag * pos[i, 1] / radii[i]
        vel[i, 1] =  v_mag * pos[i, 0] / radii[i]

    # --- 3. 增加速度弥散 ---
    ring_radii_avg = np.array([(i + 0.5) * (r_disk / n_rings) for i in range(n_rings)])
    # FIX: 防止因速度为0导致log(0)错误
    safe_v_profile = np.where(v_circ_profile > 0, v_circ_profile, 1e-9)
    d_ln_v_d_ln_r = np.gradient(np.log(safe_v_profile), np.log(ring_radii_avg))
    
    sigma_r_vals, sigma_theta_vals, sigma_z_vals = np.zeros(n_rings), np.zeros(n_rings), np.zeros(n_rings)

    for i in range(n_rings):
        v = v_circ_profile[i]
        if v == 0: continue

        term_for_sqrt = 1 + d_ln_v_d_ln_r[i]
        if term_for_sqrt < 0:
            term_for_sqrt = 0
        nu_val = v * np.sqrt(term_for_sqrt)
        
        # FIX: 防止 nu_val 为零导致除以零
        if nu_val == 0:
            sigma_r_T = np.inf # 设为无穷大，这样 min() 会自动选择另一个值
        else:
            sigma_r_T = sigma_k * np.pi * G * (particles_per_ring * m_particle / (2 * np.pi * ring_radii_avg[i] * (r_disk/n_rings))) / nu_val
        
        sigma_r_S = 0.4 * v
        sigma_r = min(sigma_r_T, sigma_r_S)
        # sigma_r = sigma_k * np.pi * G * (particles_per_ring * m_particle / (2 * np.pi * ring_radii_avg[i] * (r_disk/n_rings))) / nu_val


        if i >= n_rings - 1:
            sigma_r /= 2

        # FIX: 再次检查 term_for_sqrt 防止除以零
        if term_for_sqrt == 0:
            sigma_theta = 0 # 如果轨道临界不稳定，切向弥散也设为0
        else:
            sigma_theta = sigma_r / (np.sqrt(2) * np.sqrt(term_for_sqrt))
        
        sigma_z = sigma_theta

        sigma_r_vals[i], sigma_theta_vals[i], sigma_z_vals[i] = sigma_r, sigma_theta, sigma_z

    vel_before_random = vel.copy()
    for i in range(n):
        if radii[i] == 0: continue
        r, phi = radii[i], np.arctan2(pos[i,1], pos[i,0])
        ring_idx = ring_indices[i]
        
        rand_vr = np.random.normal(0, sigma_r_vals[ring_idx])
        rand_vtheta = np.random.normal(0, sigma_theta_vals[ring_idx])
        rand_vz = np.random.normal(0, sigma_z_vals[ring_idx])

        vel[i, 0] += rand_vr * np.cos(phi) - rand_vtheta * np.sin(phi)
        vel[i, 1] += rand_vr * np.sin(phi) + rand_vtheta * np.cos(phi)
        vel[i, 2] += rand_vz
        
    for i in range(n_rings):
        mask = (ring_indices == i)
        if not np.any(mask): continue
        ke_before = 0.5 * np.sum(mass[mask, np.newaxis] * vel_before_random[mask]**2)
        ke_after = 0.5 * np.sum(mass[mask, np.newaxis] * vel[mask]**2)
        if ke_after == 0: continue
        scale_factor = np.sqrt(ke_before / ke_after)
        vel[mask] *= scale_factor
        
    v_outer = v_circ_profile[-1]
    orbital_period = 2 * np.pi * r_disk / v_outer if v_outer > 0 else 10.0
    
    return pos, vel, mass, orbital_period

def run_simulation(sim_time, dt, pos, vel, mass, halo_mass, G, softening, disk_radius):
    num_steps = int(sim_time / dt)
    positions_history = np.zeros((num_steps, pos.shape[0], 3))
    
    acc = calculate_accelerations(pos, mass, halo_mass, G, softening, disk_radius)

    for step in tqdm(range(num_steps), desc="Simulating"):
        vel += acc * (dt / 2.0)
        pos += vel * dt
        
        acc = calculate_accelerations(pos, mass, halo_mass, G, softening, disk_radius)
        
        vel += acc * (dt / 2.0)
        
        positions_history[step] = pos
    return positions_history

if __name__ == '__main__':
    pos_init, vel_init, mass, P_outer = setup_initial_conditions(N, R, PARTICLE_MASS, SIGMA_CONST)
    SIM_TIME = P_outer * 2.0 # 模拟两个外圈轨道周期
    disk_mass_total = N * PARTICLE_MASS
    halo_mass_total = disk_mass_total * HALO_MASS_RATIO
    
    positions_history = run_simulation(SIM_TIME, DT, pos_init, vel_init, mass, halo_mass_total, G, C_SOFTENING, R)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor('black')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    scatter = ax.scatter(positions_history[0, :, 0], positions_history[0, :, 1], s=5, c='white', alpha=0.8)
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, color='white', fontsize=12)

    def init_animation():
        lim = R * 1.5
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        return scatter, time_text

    def animate(frame):
        scatter.set_offsets(positions_history[frame, :, 0:2])
        current_time_tau = (frame * DT) / P_outer
        time_text.set_text(f'τ = {current_time_tau:.2f}')
        return scatter, time_text
    
    num_steps = int(SIM_TIME / DT)
    frame_step = max(1, int(num_steps / 400)) # 保证动画总帧数在400左右
    num_frames = positions_history.shape[0] // frame_step
    
    ani = FuncAnimation(fig,
                        lambda frame: animate(frame * frame_step),
                        frames=num_frames,
                        init_func=init_animation,
                        blit=True,
                        interval=30)

    # output_filename = f"Halo_{int(HALO_MASS_RATIO*10)}.mp4"
    #
    # try:
    #     ani.save(output_filename, writer='ffmpeg', fps=30, dpi=200, progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'))
    #     print(f"Animation saved to: {output_filename}")
    # except Exception as e:
    #     print(f"Error saving animation: {e}")

    plt.show()
