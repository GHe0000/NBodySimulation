from manim import *
import numpy as np
import random

class NBody(Scene):
    # 粒子数量
    N_PARTICLES = 5
    # 万有引力常数（为可视化效果调整了大小）
    G = 50
    # 动画总时长（秒）
    TOTAL_DURATION = 30
    # 用于防止计算中出现除以零错误的软化因子
    SOFTENING_FACTOR = 0.1
    # 粒子初始位置的分布范围
    POSITION_RANGE = 4.0
    # 加速度箭头的缩放比例，以便在视觉上更清晰
    ACCELERATION_SCALE = 0.05

    def construct(self):
        """
        构建 Manim 场景和动画。
        """
        # --- 场景标题 ---
        # title = Text("5体引力模拟 (5-Body Gravitational Simulation)", font_size=36)
        # self.play(Write(title))
        # self.wait(1)
        # self.play(title.animate.to_edge(UP))

        # --- 初始化粒子 ---
        # 为每个粒子设置随机质量、初始位置和颜色
        self.masses = np.array([random.uniform(1.0, 5.0) for _ in range(self.N_PARTICLES)])
        # 确保总动量为零，防止整个系统漂移
        initial_positions = np.random.uniform(-self.POSITION_RANGE, self.POSITION_RANGE, (self.N_PARTICLES, 3))
        initial_positions[:, 2] = 0 # 确保在 XY 平面上
        
        initial_velocities = np.random.uniform(-1, 1, (self.N_PARTICLES, 3))
        initial_velocities[:, 2] = 0
        # 通过减去加权平均速度来使总动量为零
        total_mass = np.sum(self.masses)
        weighted_velocities = np.sum(initial_velocities * self.masses[:, np.newaxis], axis=0)
        center_of_mass_velocity = weighted_velocities / total_mass
        initial_velocities -= center_of_mass_velocity

        # 创建 Manim 的点对象来表示粒子
        self.particles = VGroup(*[
            Dot(point=pos, radius=0.05 * mass**0.5).set_color(color)
            for pos, mass, color in zip(initial_positions, self.masses, [BLUE, GREEN, RED, YELLOW, PURPLE])
        ])

        # 将物理属性（速度、加速度）存储在每个点对象中
        for i, p in enumerate(self.particles):
            p.velocity = initial_velocities[i]
            p.acceleration = np.zeros(3)

        # --- 创建轨迹和加速度箭头 ---
        # TracedPath 用于绘制粒子的轨迹
        trails = VGroup(*[
            TracedPath(p.get_center, stroke_width=2, stroke_color=p.get_color()) for p in self.particles
        ])
        # Arrow 用于可视化加速度
        acceleration_vectors = VGroup(*[
            Arrow(
                p.get_center(),
                p.get_center() + p.acceleration * self.ACCELERATION_SCALE,
                buff=p.get_radius(),
                stroke_width=3,
                max_tip_length_to_length_ratio=0.3
            ) for p in self.particles
        ])

        # --- 核心更新逻辑 ---
        def update_system(group, dt):
            """
            此函数在每一帧被调用，用于更新粒子的状态。

            参数:
                group (VGroup): 包含所有粒子的组。
                dt (float): 自上一帧以来的时间增量。
            """
            # 步骤 1: 计算每个粒子受到的引力
            positions = np.array([p.get_center() for p in group])
            forces = np.zeros_like(positions)

            for i in range(self.N_PARTICLES):
                for j in range(self.N_PARTICLES):
                    if i == j:
                        continue
                    
                    # 计算两个粒子之间的矢量差
                    r_vec = positions[j] - positions[i]
                    # 计算距离，并加入软化因子防止数值不稳定
                    dist = np.linalg.norm(r_vec) + self.SOFTENING_FACTOR
                    
                    # 计算引力大小 F = G * m1 * m2 / r^2
                    force_magnitude = self.G * self.masses[i] * self.masses[j] / dist**2
                    
                    # 将力分解到方向上
                    force_vector = force_magnitude * (r_vec / dist)
                    forces[i] += force_vector

            # 步骤 2: 更新加速度、速度和位置
            for i, p in enumerate(group):
                # a = F / m
                p.acceleration = forces[i] / self.masses[i]
                # v = v0 + a*t
                p.velocity += p.acceleration * dt
                # p = p0 + v*t
                new_position = p.get_center() + p.velocity * dt
                p.move_to(new_position)
            
            # 步骤 3: 更新加速度箭头的位置和方向
            for i, p in enumerate(group):
                vec = acceleration_vectors[i]
                vec.put_start_and_end_on(
                    p.get_center(),
                    p.get_center() + p.acceleration * self.ACCELERATION_SCALE
                )


        # --- 运行动画 ---
        # 将更新器添加到粒子组
        self.particles.add_updater(update_system)

        # 将所有对象添加到场景中
        self.add(trails, self.particles, acceleration_vectors)

        # 等待，让模拟运行
        self.wait(self.TOTAL_DURATION)
        
        # 移除更新器以停止动画
        self.particles.remove_updater(update_system)


# 要运行此脚本，请在终端中使用以下命令：
# manim -pql your_script_name.py NBodyGravitySimulation
