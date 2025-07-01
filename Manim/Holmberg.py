from manim import *
import random
import numpy as np

# CN_FONT = "思源黑体 CN" # Arch Linux Example
CN_FONT = "Source Han Sans CN" # Windows/Generic Example

class Holmberg(Scene):
    def construct(self):
        # -----------------------------------------------------------------
        # 1. Introduction and Setup
        # -----------------------------------------------------------------
        random.seed(114)
        num_bulbs_per_galaxy = 15
        galaxy_radius = 1.5
        galaxy_ring_radius = 2.0
        
        ANNOTATION_X_OFFSET = 3.0
        ANNOTATION_Y_OFFSET = 2.5

        galaxy_a_dots = self.create_galaxy(num_bulbs_per_galaxy, galaxy_radius).shift(LEFT * 3.5)
        galaxy_b_dots = galaxy_a_dots.copy().shift(RIGHT * 7.0) # Ensure symmetry
        
        galaxies = VGroup(galaxy_a_dots, galaxy_b_dots)
        galaxy_a_ids = {id(bulb) for bulb in galaxy_a_dots}
        galaxy_b_ids = {id(bulb) for bulb in galaxy_b_dots}

        circle_a = Circle(radius=galaxy_ring_radius, color=BLUE).move_to(galaxy_a_dots.get_center())
        circle_b = Circle(radius=galaxy_ring_radius, color=BLUE).move_to(galaxy_b_dots.get_center())
        galaxy_circles = VGroup(circle_a, circle_b)

        annotation_position = galaxy_a_dots.get_center() + RIGHT * ANNOTATION_X_OFFSET + UP * ANNOTATION_Y_OFFSET

        self.play(
            LaggedStartMap(GrowFromCenter, galaxies, lag_ratio=0.1),
            Create(galaxy_circles)
        )
        self.wait(1)

        # -----------------------------------------------------------------
        # 2. Select Test Bulb and Prepare for Measurement
        # -----------------------------------------------------------------
        
        test_bulb = galaxy_a_dots[7]
        other_galaxy_bulbs = galaxy_b_dots

        photocell_body = Rectangle(width=0.3, height=0.3, color=BLUE_D, fill_opacity=0.8).round_corners(0.1)
        photocell_indicator = Line(ORIGIN, RIGHT * 0.4, color=WHITE, stroke_width=3).move_to(photocell_body.get_right(), aligned_edge=LEFT)
        photocell_group = VGroup(photocell_body, photocell_indicator).move_to(test_bulb.get_center())
        
        principle_text = Text(
            "1. 将光电管放置于待测灯泡处",
            font=CN_FONT,
            font_size=24
        ).move_to(annotation_position) # Changed from .to_edge(DOWN)
        self.play(Write(principle_text))

        self.play(Indicate(test_bulb, color=YELLOW, scale_factor=2.5, run_time=1.5))
        self.play(FadeOut(test_bulb), FadeIn(photocell_group))
        self.wait(1)

        GRAVITATIONAL_CONSTANT = 0.5
        net_force_vector = np.array([0., 0., 0.])
        for source_bulb in other_galaxy_bulbs:
            direction_vec = source_bulb.get_center() - test_bulb.get_center()
            distance_sq = np.sum(direction_vec**2)
            force_magnitude = GRAVITATIONAL_CONSTANT / distance_sq
            force_vector = force_magnitude * (direction_vec / np.linalg.norm(direction_vec))
            net_force_vector += force_vector

        # -----------------------------------------------------------------
        # 3. Demonstrate Rotational Measurement
        # -----------------------------------------------------------------
        measurement_text = Text(
            "2. 旋转光电管，测量 ±x, ±y 四个方向光强",
            font=CN_FONT, font_size=24
        ).move_to(annotation_position) # Repositioned
        self.play(ReplacementTransform(principle_text, measurement_text))

        source_bulbs = list(galaxy_b_dots) # Start with all bulbs from the other galaxy
        test_bulb_x = test_bulb.get_center()[0]

        for bulb in galaxy_a_dots:
            if id(bulb) == id(test_bulb): # Don't draw a line from the bulb to itself
                continue
            if bulb.get_center()[0] > test_bulb_x:
                source_bulbs.append(bulb)

        measurement_lines = VGroup(*[
            Line(source.get_center(), test_bulb.get_center(), buff=0.1, stroke_width=1, color=YELLOW).set_opacity(0.7)
            for source in source_bulbs
        ])

        self.play(Create(measurement_lines))
        self.wait(1.5)
        self.play(FadeOut(measurement_lines))

        # Animate rotation with labels
        direction_label = Text("+x", font_size=24).next_to(photocell_indicator, RIGHT, buff=0.1)
        self.play(Write(direction_label))
        self.wait(0.5)

        rotations = [
            (90 * DEGREES, "+y", UP),
            (90 * DEGREES, "-x", LEFT),
            (90 * DEGREES, "-y", DOWN),
        ]
        
        for angle, text, direction in rotations:
            new_label = Text(text, font_size=24).next_to(photocell_body, direction, buff=0.3)
            self.play(
                Rotate(photocell_group, angle=angle, about_point=photocell_group.get_center()),
                ReplacementTransform(direction_label, new_label),
                run_time=0.7
            )
            direction_label = new_label
            self.wait(0.5)
        
        # Final rotation back to start
        self.play(
            Rotate(photocell_group, angle=90 * DEGREES, about_point=photocell_group.get_center()),
            FadeOut(direction_label),
            run_time=0.7
        )
        self.wait(1)

        # -----------------------------------------------------------------
        # 4. Calculate and Show Resultant Force
        # -----------------------------------------------------------------
        calculation_text = Text(
            "3. 根据光强计算合力",
            font=CN_FONT, font_size=24
        ).move_to(annotation_position) # Repositioned
        self.play(ReplacementTransform(measurement_text, calculation_text))

        force_display_scale = 15.0

        # Manual aesthetic adjustment
        net_force_vector[0] *= 0.5
        net_force_vector[1] *= 2
        
        scaled_force_for_arrow = net_force_vector * force_display_scale
        center_point = test_bulb.get_center()
        
        fx_vec = Arrow(center_point, center_point + np.array([scaled_force_for_arrow[0], 0, 0]), buff=0, color=GREEN, stroke_width=5)
        fy_vec = Arrow(center_point, center_point + np.array([0, scaled_force_for_arrow[1], 0]), buff=0, color=ORANGE, stroke_width=5)
        fx_label = MathTex(r"\vec{F}_x").next_to(fx_vec, fx_vec.get_vector()/np.linalg.norm(fx_vec.get_vector()), buff=0.1).set_color(GREEN)
        fy_label = MathTex(r"\vec{F}_y").next_to(fy_vec, fy_vec.get_vector()/np.linalg.norm(fy_vec.get_vector()), buff=0.1).set_color(ORANGE)

        self.play(GrowArrow(fx_vec), GrowArrow(fy_vec))
        self.play(Write(fx_label), Write(fy_label))
        self.wait(1.5)

        resultant_arrow = Arrow(center_point, center_point + scaled_force_for_arrow, buff=0, stroke_width=7, color=RED)
        resultant_label = MathTex(r"\vec{F}").next_to(resultant_arrow.get_end(), buff=0.1).set_color(RED)
        
        self.play(
            ReplacementTransform(VGroup(fx_vec, fy_vec), resultant_arrow),
            ReplacementTransform(VGroup(fx_label, fy_label), resultant_label)
        )
        self.wait(2)

        # -----------------------------------------------------------------
        # 5. Time Step
        # -----------------------------------------------------------------
        step_text = Text(
            "4. 根据合力计算下一时刻的位置",
            font=CN_FONT, font_size=24
        ).move_to(annotation_position) # Repositioned
        self.play(ReplacementTransform(calculation_text, step_text))
        
        self.play(FadeOut(photocell_group), FadeOut(resultant_arrow), FadeOut(resultant_label), FadeIn(test_bulb))

        dt_demo = 0.5
        displacement = 0.5 * net_force_vector * (dt_demo**2)
        displacement_vis_scale = 15.0
        
        self.play(test_bulb.animate.shift(displacement * displacement_vis_scale))
        self.wait(1)

        # -----------------------------------------------------------------
        # 6. Full Simulation
        # -----------------------------------------------------------------
        simulation_text = Text(
            "5. 对所有灯泡重复此过程",
            font=CN_FONT, font_size=24
        ).move_to(annotation_position) # Repositioned
        self.play(ReplacementTransform(step_text, simulation_text))
        self.wait(3)

    def create_bulb(self):
        bulb = Dot(radius=0.08, color=YELLOW_A)
        glow = Dot(radius=0.2, color=YELLOW_E, fill_opacity=0.1)
        return VGroup(glow, bulb)

    def create_galaxy(self, num_bulbs, base_radius):
        bulbs = VGroup()
        rings_config = [
            (1, 0.0),
            (6, 0.5),
            (8, 1.0),
        ]
        
        assert num_bulbs == sum(c[0] for c in rings_config)

        initial_angle_offset = 0
        for num_points, radius_multiplier in rings_config:
            ring_radius = base_radius * radius_multiplier
            if num_points == 1 and ring_radius == 0.0:
                bulbs.add(self.create_bulb().move_to(ORIGIN))
                continue
            
            for i in range(num_points):
                angle = (i / num_points) * TAU + initial_angle_offset
                pos = np.array([
                    ring_radius * np.cos(angle),
                    ring_radius * np.sin(angle),
                    0
                ])
                bulbs.add(self.create_bulb().move_to(pos))
            initial_angle_offset += PI / num_points

        return bulbs
