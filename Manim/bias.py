from manim import *
import numpy as np
import random

# CN_FONT = "思源黑体 CN" # Arch Linux 示例
CN_FONT = "Source Han Sans CN" # Windows 示例


config.font = CN_FONT
class Bias(Scene):
    def construct(self):
        # --------------------------------------------------------------------
        # 1. 场景设置：创建暗物质和观测星系
        # --------------------------------------------------------------------
        
        random.seed(114)
        np.random.seed(11)

        text_top_buff = 1.0

        # 定义左右两个区域的中心
        left_center = LEFT * 3.5
        right_center = RIGHT * 3.5
        
        # 更新标题文本
        left_title = Text("模拟的暗物质分布", font_size=28).to_edge(UP).shift(left_center * 1.1)
        right_title = Text("观测的星系分布（明显成团）", font_size=28).to_edge(UP).shift(right_center * 1.1)
        
        # 创建中间的分割线
        divider = DashedLine(UP * 3.5, DOWN * 3.5, color=GRAY)

        self.play(Write(left_title), Write(right_title), Create(divider))
        self.wait(1)

        # --- 左侧：创建暗物质点云 ---
        num_dm_points = 2000
        dm_points_coords = []
        
        clump_centers = [
            left_center + np.array([0, 1.8, 0]),
            left_center + np.array([-1.5, -1.5, 0]),
            left_center + np.array([1.5, -1.5, 0]),
        ]
        
        dm_sigma_val = 0.25
        dm_sigma = np.sqrt(dm_sigma_val)
        
        points_per_clump = int(num_dm_points / 8)
        for center in clump_centers:
            cov_matrix = [[dm_sigma_val, 0, 0], [0, dm_sigma_val, 0], [0, 0, 0]]
            dm_points_coords.extend(
                np.random.multivariate_normal(center, cov_matrix, size=points_per_clump)
            )
        
        num_background_points = num_dm_points - (len(clump_centers) * points_per_clump)
        dm_points_coords.extend(
            left_center + np.random.uniform(-3, 3, (num_background_points, 3)) * [1, 1, 0]
        )

        dark_matter_dots = VGroup(*[Dot(p, radius=0.02, color=BLUE_C) for p in dm_points_coords])

        num_galaxy_points = 20
        galaxy_points_coords = []
        
        galaxy_clump_centers = [
            right_center + np.array([0, 1.8, 0]),
            right_center + np.array([-1.5, -1.5, 0]),
            right_center + np.array([1.5, -1.5, 0]),
        ]
        
        points_per_galaxy_clump = [7, 7, 6]
        galaxy_sigma_val = 0.2

        for i, center in enumerate(galaxy_clump_centers):
            cov_matrix = [[galaxy_sigma_val, 0, 0], [0, galaxy_sigma_val, 0], [0, 0, 0]]
            galaxy_points_coords.extend(
                np.random.multivariate_normal(center, cov_matrix, size=points_per_galaxy_clump[i])
            )

        observed_galaxies = VGroup(*[Dot(p, radius=0.06, color=YELLOW) for p in galaxy_points_coords])

        self.play(FadeIn(dark_matter_dots, scale=0.5), FadeIn(observed_galaxies, scale=0.5))
        self.wait(2)

        # --------------------------------------------------------------------
        # 2. 无偏袒模拟
        # --------------------------------------------------------------------
        
        def create_text_with_bg(text, color=WHITE):
            text_obj = Text(text, font_size=22, color=color).to_edge(UP, buff=text_top_buff)
            bg_rect = SurroundingRectangle(
                text_obj,
                color=BLACK,
                fill_color=BLACK,
                fill_opacity=0.6,
                buff=0.1
            )
            return VGroup(bg_rect, text_obj)

        text_group_unbiased_1 = create_text_with_bg("无偏袒性的情况")
        text_group_unbiased_2 = create_text_with_bg("模拟的聚集程度与观测不符", color=RED)

        self.play(Write(text_group_unbiased_1))
        self.wait(1)

        unbiased_sample_indices = random.sample(range(len(dark_matter_dots)), num_galaxy_points)
        
        sources = VGroup(*[dark_matter_dots[i].copy() for i in unbiased_sample_indices])
        targets = VGroup(*[Dot(s.get_center(), radius=0.06, color=YELLOW) for s in sources])

        self.play(LaggedStart(*[
            ReplacementTransform(sources[i], targets[i]) for i in range(num_galaxy_points)
        ], lag_ratio=0.05))
        
        self.wait(1)
        
        self.play(ReplacementTransform(text_group_unbiased_1, text_group_unbiased_2))
        self.wait(3)

        self.play(
            FadeOut(targets),
            FadeOut(text_group_unbiased_2),
        )
        self.wait(1)

        # --------------------------------------------------------------------
        # 3. 偏袒性模拟
        # --------------------------------------------------------------------
        
        text_group_biased_1 = create_text_with_bg("有偏袒性的情况")
        text_group_biased_2 = create_text_with_bg("只在暗物质密度足够高的区域生成星系")

        self.play(Write(text_group_biased_1))
        self.wait(1)

        high_density_regions_dots = VGroup()
        for i, p in enumerate(dm_points_coords):
            for center in clump_centers:
                if np.linalg.norm(p - center) < 2 * dm_sigma:
                    high_density_regions_dots.add(dark_matter_dots[i])
                    break 

        self.play(ReplacementTransform(text_group_biased_1, text_group_biased_2))
        
        self.play(high_density_regions_dots.animate.set_color(YELLOW), run_time=0.7)
        self.wait(0.5)
        self.play(high_density_regions_dots.animate.set_color(BLUE_C), run_time=0.7)

        self.wait(1)
        
        biased_sample_indices = []
        for center, num_to_sample in zip(clump_centers, points_per_galaxy_clump):
            points_in_this_clump_indices = [
                i for i, p in enumerate(dm_points_coords)
                if np.linalg.norm(p - center) < 2 * dm_sigma
            ]
            
            if len(points_in_this_clump_indices) < num_to_sample:
                sampled_for_this_clump = points_in_this_clump_indices
            else:
                sampled_for_this_clump = random.sample(
                    points_in_this_clump_indices, num_to_sample
                )
            
            biased_sample_indices.extend(sampled_for_this_clump)

        biased_sources = VGroup(*[dark_matter_dots[i].copy() for i in biased_sample_indices])
        biased_targets = VGroup(*[Dot(s.get_center(), radius=0.06, color=YELLOW) for s in biased_sources])

        self.play(FadeOut(text_group_biased_2))

        self.play(LaggedStart(*[
            ReplacementTransform(biased_sources[i], biased_targets[i]) 
            for i in range(len(biased_sample_indices))
        ], lag_ratio=0.05))

        self.wait(1)

        final_text_group = create_text_with_bg("模拟结果与观测相似", color=GREEN)
        
        self.play(Write(final_text_group))
        self.wait(5)
