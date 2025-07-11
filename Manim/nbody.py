from manim import *
import numpy as np
import random

# CN_FONT = "思源黑体 CN" # Arch Linux
CN_FONT = "Source Han Sans CN" # Windows

class NBody(Scene):
    BEST_SEED = 42

    N_PARTICLES = 5
    G = 2
    DAMPING_FACTOR = 0.999
    TOTAL_DURATION = 30
    SOFTENING_FACTOR = 0.1
    PERTURBATION_STRENGTH = 0.1
    ACCELERATION_SCALE = 0.15
    SIMULATION_SPEED = 0.2

    MIN_ARROW_LENGTH = 0.2
    MAX_ARROW_LENGTH = 0.8

    def construct(self):
        # --- 设置种子以复现优化结果 ---
        np.random.seed(self.BEST_SEED)

        self.masses = np.random.uniform(3.0, 5.0, (self.N_PARTICLES,))
        particle_colors = [BLUE, GREEN, RED, YELLOW, PURPLE]

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
            for pos, mass, color in zip(initial_positions, self.masses, particle_colors)
        ])

        for i, p in enumerate(self.particles):
            p.velocity = initial_velocities[i]
            p.acceleration = np.zeros(3)

        trails = VGroup(*[TracedPath(p.get_center, stroke_width=2, stroke_color=p.get_color()) for p in self.particles])
        
        # 使用 Vector 代替 Arrow，以获得更好的箭头缩放效果
        acceleration_vectors = VGroup()
        for p in self.particles:
            # 初始化一个零长度的矢量
            vec = Vector([0, 0, 0], tip_length=0.15, color=p.get_color())
            acceleration_vectors.add(vec)

        # --- 核心更新逻辑 ---
        def update_system(group, dt):
            # 通过缩放dt来减慢模拟速度
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
            
            # 更新加速度矢量
            for i, p in enumerate(group):
                vec = acceleration_vectors[i]
                accel = p.acceleration
                accel_norm = np.linalg.norm(accel)
                
                if accel_norm > 1e-6:
                    # 根据加速度的大小（模长）来设置箭头的粗细
                    stroke_width = np.clip(2 + accel_norm * 1.5, 2, 12)
                    
                    # 【已修改】对箭头的长度进行限制
                    scaled_length = accel_norm * self.ACCELERATION_SCALE
                    clamped_length = np.clip(scaled_length, self.MIN_ARROW_LENGTH, self.MAX_ARROW_LENGTH)
                    direction = accel / accel_norm
                    
                    # 创建一个临时的“目标”矢量，它拥有所有我们期望的属性
                    target_vec = Vector(
                        direction * clamped_length, # 使用限制长度后的矢量
                        color=p.get_color(),
                        tip_length=0.15
                    )
                    target_vec.set_stroke(width=stroke_width)
                    target_vec.move_to(p.get_center() + target_vec.get_vector() / 2) # 将矢量中心对准粒子
                    
                    # 让场景中的矢量“变成”目标矢量的样子
                    vec.become(target_vec)
                    vec.set_opacity(0.8)

                else:
                    vec.set_opacity(0)

        self.particles.add_updater(update_system)
        self.add(trails, self.particles, acceleration_vectors)
        self.wait(self.TOTAL_DURATION)
        self.particles.remove_updater(update_system)
