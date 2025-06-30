from manim import *

# CN_FONT = "思源黑体 CN" # Arch Linux 示例
CN_FONT = "Source Han Sans CN" # Windows 示例

math_template = TexTemplate(preamble=
r"""
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{mathrsfs}
"""
)

class PM(Scene):
    def construct(self):

        top_y = 2.0
        bottom_y = -1.0

        poisson_eq = MathTex(
            r"\nabla^2 \Phi(\vec{x}, t)", r"=", r"4\pi G \rho(\vec{x}, t)",
            tex_to_color_map={
                r"\Phi(\vec{x}, t)": GREEN,
                r"\rho(\vec{x}, t)": BLUE,
                r"\nabla^2": PINK,
            }
        ).move_to(LEFT * 3.5 + UP * top_y)

        poisson_eq_note = MarkupText(f"描述引力势的 Poisson 方程", font=CN_FONT, font_size=30).next_to(poisson_eq, UP)

        potential_phi = MathTex(
            r"\Phi(\vec{x}, t)",
            tex_to_color_map={
                r"\Phi(\vec{x}, t)": GREEN
            }
        ).move_to(RIGHT * 3.5 + UP * top_y)
        
        direct_arrow = Arrow(
            poisson_eq.get_right(), potential_phi.get_left(), buff=0.5, color=RED
        )
        direct_label = MarkupText(f"直接求解，<span foreground='{RED}'>慢!</span>", font=CN_FONT, font_size=30).next_to(direct_arrow, UP)
        complexity_direct = MathTex(r"\mathcal{O}(N^2)", color=ORANGE).next_to(direct_arrow, DOWN)

        fourier_eq = MathTex(
            r"-|\vec{k}|^2 \hat{\Phi}(\vec{k})", r"=", r"4\pi G \hat{\rho}(\vec{k})",
            tex_to_color_map={
                r"-|\vec{k}|^2": PINK,
                r"\hat{\Phi}(\vec{k})": GREEN,
                r"\hat{\rho}(\vec{k})": BLUE
            }
        ).move_to(LEFT * 3.5 + UP * bottom_y)

        solved_fourier_eq = MathTex(
            r"\hat{\Phi}(\vec{k})", r"=", r"- \frac{4\pi G}{|\vec{k}|^2} \hat{\rho}(\vec{k})",
            tex_to_color_map={
                r"-": PINK,
                r"|\vec{k}|^2": PINK,
                r"\hat{\Phi}(\vec{k})": GREEN,
                r"\hat{\rho}(\vec{k})": BLUE
            }
        ).move_to(RIGHT * 3.5 + UP * bottom_y)

        # --- 连接箭头与标签 ---
        arrow_down = Arrow(poisson_eq.get_bottom(), fourier_eq.get_top(), buff=0.2, color=GREEN)
        fourier_label = MathTex(r"\mathscr{F}",tex_template=math_template, color=YELLOW).next_to(arrow_down, LEFT)

        nable_note = MathTex(r"\nabla^2 \Leftrightarrow -|\vec{k}|^2", color=PINK).next_to(arrow_down, RIGHT)

        algebra_arrow = Arrow(fourier_eq.get_right(), solved_fourier_eq.get_left(), buff=0.5, color=GREEN)
        algebra_label = MarkupText(f"简单的代数方程", font=CN_FONT, font_size=30).next_to(algebra_arrow, UP)

        arrow_up = Arrow(solved_fourier_eq.get_top(), potential_phi.get_bottom(), buff=0.2, color=GREEN)
        inv_fourier_label = MathTex(r"\mathscr{F}^{-1}",tex_template=math_template, color=YELLOW).next_to(arrow_up, RIGHT)

        self.play(
            Write(poisson_eq),
            Write(poisson_eq_note)
        )
        self.wait(0.5)
        self.play(
            Create(direct_arrow),
            Write(direct_label),
            Write(complexity_direct)
        )
        self.play(Write(potential_phi))
        self.wait(2)

        self.play(Create(arrow_down), Write(fourier_label), Write(nable_note))
        self.play(TransformMatchingTex(poisson_eq.copy(), fourier_eq))
        self.play(
            Create(algebra_arrow),
            Write(algebra_label)
        )
        self.play(Write(solved_fourier_eq))

        self.play(
            Create(arrow_up),
            Write(inv_fourier_label)
        )
        self.wait(2)

        group_to_shift = Group(*self.mobjects)
        
        self.play(group_to_shift.animate.shift(UP * 1))
        self.wait(0.5)
        
        summary_formula = MathTex(
            r"\Phi = \mathscr{F}^{-1} \left[ - \frac{4\pi G}{|\vec{k}|^2} \mathscr{F}[\rho] \right]\quad\mathcal{O}(N \log N)",
            tex_template=math_template,
            tex_to_color_map={
                r"\Phi": GREEN,
                r"\mathscr{F}^{-1}": YELLOW,
                r"\mathscr{F}": YELLOW,
                r"\rho": BLUE,
                r"\mathcal{O}(N \log N)": ORANGE
            }
        ).next_to(group_to_shift, DOWN, buff=1.0) 

        summary_group = VGroup(summary_formula)
        surrounding_rect = SurroundingRectangle(summary_group, buff=0.3, corner_radius=0.2)

        self.play(
            Write(summary_formula),
            Create(surrounding_rect)
        )
        self.wait(3)
        final_group = Group(*self.mobjects)
        self.play(FadeOut(final_group))
        self.wait(1)
