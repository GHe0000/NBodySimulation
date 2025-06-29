from manim import *
import numpy as np
import random

class NBodySymmetric(Scene):
    # --- 关键参数 ---
    # 将你在优化器中找到的最佳种子填在这里！
    BEST_SEED = 42  # <--- 你可以修改这里来观察不同扰动的结果

    # --- 模拟参数 ---
    N_PARTICLES = 5
    G = 2
    DAMPING_FACTOR = 0.999
    TOTAL_DURATION = 30
    SOFTENING_FACTOR = 0.1
    PERTURBATION_STRENGTH = 0.1
    ACCELERATION_SCALE = 0.15
    # 【新增】动画速度调节。值越小，动画越慢。
    SIMULATION_SPEED = 0.2

    def construct(self):
        """
        构建 Manim 场景和动画。
        """
        # --- 设置种子以复现优化结果 ---
        np.random.seed(self.BEST_SEED)

        self.masses = np.random.uniform(3.0, 5.0, (self.N_PARTICLES,))

        # 2. 设置一个对称的环形轨道构型
        R = 2.5  # 轨道半径
        orbital_v = 2.18  # 设置一个合理的初始切向速度
        
        initial_positions = np.zeros((self.N_PARTICLES, 3))
        initial_velocities = np.zeros((self.N_PARTICLES, 3))

        # 将所有粒子均匀放置在环上
        for i in range(self.N_PARTICLES):
            angle = 2 * np.pi * i / self.N_PARTICLES
            px, py = R * np.cos(angle), R * np.sin(angle)
            initial_positions[i] = [px, py, 0]
            vx, vy = -orbital_v * np.sin(angle), orbital_v * np.cos(angle)
            initial_velocities[i] = [vx, vy, 0]

        # 3. 对稳定构型施加随机微扰
        pos_perturb = np.random.uniform(-1, 1, (self.N_PARTICLES, 3)) * self.PERTURBATION_STRENGTH
        vel_perturb = np.random.uniform(-1, 1, (self.N_PARTICLES, 3)) * self.PERTURBATION_STRENGTH
        pos_perturb[:, 2] = 0
        vel_perturb[:, 2] = 0
        initial_positions += pos_perturb
        initial_velocities += vel_perturb

        # 4. 在施加扰动后，重新将系统总动量归零
        total_mass = np.sum(self.masses)
        weighted_velocities = np.sum(initial_velocities * self.masses.reshape(-1, 1), axis=0)
        center_of_mass_velocity = weighted_velocities / total_mass
        initial_velocities -= center_of_mass_velocity

        # --- 创建 Manim 对象 ---
        self.particles = VGroup(*[
            Dot(point=pos, radius=0.05 * mass**0.5).set_color(color)
            for pos, mass, color in zip(initial_positions, self.masses, [BLUE, GREEN, RED, YELLOW, PURPLE])
        ])

        for i, p in enumerate(self.particles):
            p.velocity = initial_velocities[i]
            p.acceleration = np.zeros(3)

        trails = VGroup(*[TracedPath(p.get_center, stroke_width=2, stroke_color=p.get_color()) for p in self.particles])
        
        acceleration_vectors = VGroup()
        for p in self.particles:
            arrow = Arrow(p.get_center(), p.get_center() + RIGHT * 1e-6, buff=p.get_radius(), stroke_width=3, max_tip_length_to_length_ratio=0.3)
            arrow.set_opacity(0)
            acceleration_vectors.add(arrow)

        # --- 核心更新逻辑 ---
        def update_system(group, dt):
            # 【已修改】通过缩放dt来减慢模拟速度
            sim_dt = dt * self.SIMULATION_SPEED

            positions = np.array([p.get_center() for p in group])
            forces = np.zeros_like(positions)

            for i in range(self.N_PARTICLES):
                for j in range(self.N_PARTICLES):
                    if i == j: continue
                    r_vec = positions[j] - positions[i]
                    dist = np.linalg.norm(r_vec) + self.SOFTENING_FACTOR
                    force_magnitude = self.G * self.masses[i] * self.masses[j] / (dist**2)
                    force_vector = force_magnitude * (r_vec / dist)
                    forces[i] += force_vector

            for i, p in enumerate(group):
                p.acceleration = forces[i] / self.masses[i]
                p.velocity += p.acceleration * sim_dt
                p.velocity *= self.DAMPING_FACTOR
                p.move_to(p.get_center() + p.velocity * sim_dt)
            
            for i, p in enumerate(group):
                vec = acceleration_vectors[i]
                accel_norm = np.linalg.norm(p.acceleration)
                if accel_norm > 1e-6:
                    vec.set_opacity(1)
                    vec.put_start_and_end_on(p.get_center(), p.get_center() + p.acceleration * self.ACCELERATION_SCALE)
                else:
                    vec.set_opacity(0)

        # --- 运行动画 ---
        self.particles.add_updater(update_system)
        self.add(trails, self.particles, acceleration_vectors)
        self.wait(self.TOTAL_DURATION)
        self.particles.remove_updater(update_system)

# 要运行此脚本，请在终端中使用以下命令：
# manim -pql your_script_name.py NBodySymmetric
