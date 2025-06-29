from manim import *

class Tree(Scene):
    def construct(self):
        np.random.seed(114)
        title = Text("基于树的算法可视化", font="思源黑体 CN").to_edge(UP)
        self.play(Write(title))
        self.wait(1)

        target_particle = Dot(point=LEFT * 5, radius=0.15, color=YELLOW)
        target_label = Text("目标粒子", font="思源黑体 CN", font_size=24).next_to(target_particle, DOWN)

        particle_cluster_center = RIGHT * 4
        particles = VGroup(*[
            Dot(
                point=particle_cluster_center + np.array([
                    np.random.normal(0, 0.7),
                    np.random.normal(0, 0.7),
                    0
                ]),
                radius=0.05,
                color=BLUE
            ) for _ in range(50)
        ])
        cluster_label = Text("遥远的粒子群", font="思源黑体 CN", font_size=24).next_to(particles, DOWN)

        self.play(
            FadeIn(target_particle, scale=0.5),
            Write(target_label),
            FadeIn(particles, lag_ratio=0.1),
            Write(cluster_label)
        )
        self.wait(2)

        explanation1 = Text("传统方法：计算每个粒子对目标的引力", font="思源黑体 CN", font_size=28).to_edge(DOWN)
        self.play(Write(explanation1))

        interaction_lines = VGroup(*[Line(target_particle.get_center(), p.get_center(), stroke_width=1, color=GRAY) for p in particles])
        self.play(Create(interaction_lines, lag_ratio=0.1), run_time=3)
        self.wait(1)

        self.play(FadeOut(interaction_lines), FadeOut(explanation1))

        # complexity_text = Text("计算量巨大!", font="思源黑体 CN", font_size=36, color=RED).move_to(ORIGIN)
        # self.play(FadeIn(complexity_text, scale=1.5))
        # self.wait(2)
        # self.play(FadeOut(complexity_text), FadeOut(interaction_lines), FadeOut(explanation1))
        # self.wait(1)

        explanation2 = Text("基于树的方法：对粒子进行分层，将远处粒子群视为一个整体", font="思源黑体 CN", font_size=28).to_edge(DOWN)
        self.play(Write(explanation2))

        bounding_box = SurroundingRectangle(particles, buff=0.2, color=GREEN, stroke_width=3)
        self.play(Create(bounding_box))
        self.wait(2)


        # explanation3 = Text("计算质心，并将其等效为一个“超级粒子”", font="思源黑体 CN", font_size=28).to_edge(DOWN)
        # self.play(Write(explanation3))

        center_of_mass_point = np.mean([p.get_center() for p in particles], axis=0)
        center_of_mass_particle = Dot(point=center_of_mass_point, radius=0.25, color=ORANGE)
        com_label = Text("等效粒子", font="思源黑体 CN", font_size=24).next_to(center_of_mass_particle, UP)
        
        self.play(FadeIn(center_of_mass_particle, scale=1.5))
        self.wait(1)

        self.play(
            ReplacementTransform(particles, center_of_mass_particle),
            FadeOut(bounding_box),
            FadeOut(cluster_label),
            Write(com_label)
        )
        self.wait(2)

        self.play(FadeOut(explanation2))
        # self.play(FadeOut(explanation3))
        explanation4 = Text("从而只需进行一次计算，大大减少计算量", font="思源黑体 CN", font_size=28).to_edge(DOWN)
        self.play(Write(explanation4))

        simplified_line = Line(target_particle.get_center(), center_of_mass_particle.get_center(),  stroke_width=4, color=YELLOW)
        self.play(Create(simplified_line), run_time=2)
        self.wait(1)
        
        # final_text = Text("计算量大大减少!", font="思源黑体 CN", font_size=36, color=GREEN).move_to(ORIGIN)
        # self.play(FadeIn(final_text, scale=1.5))
        # self.wait(3)

        self.play(
            FadeOut(explanation4),
            #FadeOut(final_text),
            FadeOut(simplified_line),
            FadeOut(center_of_mass_particle),
            FadeOut(com_label),
            FadeOut(target_particle),
            FadeOut(target_label),
            FadeOut(title)
        )
        self.wait(1)
