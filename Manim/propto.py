from manim import *

# 设置中文显示
# 如果您在本地运行，请确保您已安装并配置了支持中文的字体，
# 例如 "Source Han Sans SC" 或 "思源黑体"。
# config.font = "Source Han Sans SC"

class Gravity(ThreeDScene):
    """
    一个独立的 3D Manim 场景，用于演示万有引力。
    - 一个小星球围绕一个大星球公转。
    - 一个箭头实时显示引力的方向。
    - 屏幕上显示标题和公式。
    - 相机缓慢旋转以增强3D效果。
    """
    def construct(self):
        # 1. 3D 场景和相机设置
        # ---------------------
        self.set_camera_orientation(phi=75 * DEGREES, theta=-60 * DEGREES)
        # 添加一个三维坐标轴作为参考
        axes = ThreeDAxes(x_range=[-5, 5], y_range=[-5, 5], z_range=[-3, 3])

        # 2. 创建标题和公式
        # -------------------
        # FIX: 将标题移动到顶部居中
        title = Text("万有引力", font_size=36).to_edge(UP, buff=0.5)
        formula = MathTex(r"F \propto \frac{1}{r^2}", font_size=50).to_edge(DOWN, buff=0.5)
        self.add_fixed_in_frame_mobjects(title, formula)

        # 3. 创建天体和轨道
        # -------------------
        center_planet = Sphere(center=ORIGIN, radius=0.6, color=YELLOW)
        center_planet.set_shade_in_3d(True)

        orbit_path = Circle(radius=2.5, color=BLUE_C)
        
        # 4. 创建行星和箭头的组合体
        # -------------------------
        # 先创建一个虚拟的小星球，用于定位
        small_planet = Sphere(radius=0.2, color=WHITE)
        small_planet.set_shade_in_3d(True)
        small_planet.move_to(orbit_path.point_from_proportion(0))
        
        # 创建一个指向中心的箭头
        force_arrow = Arrow3D(
            start=small_planet.get_center(),
            end=center_planet.get_center(),
            color=RED,
            resolution=8
        )
        
        # FIX: 缩短箭头，使其视觉效果更好
        # 以箭头的末端（中心星球）为缩放中心，将箭头缩放为原来的60%
        force_arrow.scale(0.5, about_point=force_arrow.get_start())
        
        # 将小星球和箭头组合成一个 VGroup，确保它们成为一个整体
        planet_system = VGroup(small_planet, force_arrow)

        # 5. 播放初始动画
        # -----------------
        self.play(
            FadeIn(axes),
            Write(title),
            Write(formula)
        )
        self.play(
            FadeIn(center_planet, scale=0.5),
            Create(orbit_path),
            FadeIn(planet_system) # 将组合体作为一个整体淡入
        )
        self.wait(1)

        # 6. 播放主循环动画
        # -------------------
        # 启动相机背景旋转
        self.begin_ambient_camera_rotation(rate=0.1)
        
        # 对“行星系统”这个整体进行旋转，而不是单独移动星球
        self.play(
            Rotate(
                planet_system,
                angle=2 * PI, # 旋转360度
                axis=OUT, # 绕Z轴旋转
                about_point=center_planet.get_center(), # 绕中心星球旋转
                rate_func=linear
            ),
            # FIX: 增加视频时长
            run_time=20
        )
        
        # 停止相机旋转并等待
        self.stop_ambient_camera_rotation()
        self.wait(2)

        # 7. 清理场景
        # -----------
        self.play(
            *[FadeOut(mob) for mob in self.mobjects]
        )
        self.wait()



class Light(ThreeDScene):

    def construct(self):
        # --- 1. 场景和相机设置 (Scene and Camera Setup) ---
        
        # 设置 3D 相机的视角
        # phi是俯仰角, theta是方位角
        # Set the 3D camera orientation (phi for elevation, theta for azimuth)
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        
        # 在画面中添加一个固定的标题
        # Add a title that stays fixed on the screen
        title = Text("光强", font_size=36).to_edge(UP, buff=0.5)
        self.add_fixed_in_frame_mobjects(title)

        # 在画面底部添加物理公式
        # Add the physics formula at the bottom of the screen
        formula = MathTex(r"I \propto \frac{1}{r^2}", font_size=50).to_edge(DOWN, buff=0.5)
        self.add_fixed_in_frame_mobjects(formula)
        formula.to_edge(DOWN, buff=0.8)

        # --- 2. 创建场景中的物体 (Create Objects in the Scene) ---

        # 创建中心的光源（一个不透明的黄色小球）
        # Create the central light source (a small, opaque yellow sphere)
        light_source = Sphere(
            center=ORIGIN,
            radius=0.1,
            resolution=(24, 24) # 提高球体的平滑度 (Increase resolution for a smoother sphere)
        ).set_color(YELLOW).set_opacity(0.9)

        # 创建一个 ValueTracker 来动态控制外层球体的半径
        # Create a ValueTracker to dynamically control the radius of the outer sphere
        radius_tracker = ValueTracker(1.0)

        # 定义一个与半径相关的透明度
        # k 是一个比例常数，确保初始透明度合适
        # Define an opacity that depends on the radius.
        # 'k' is a proportionality constant to make the initial opacity look good.
        opacity_k = 0.5

        # 创建外层的半透明球体
        # Create the outer, semi-transparent sphere
        light_shell = Sphere(
            radius=radius_tracker.get_value(),
            resolution=(32, 32) # 更高的分辨率以获得更好的视觉效果
        ).set_color(YELLOW).set_opacity(opacity_k / radius_tracker.get_value()**2)

        # --- 3. 添加 Updater 实现动态更新 (Add Updaters for Dynamic Animation) ---

        # 为外层球体添加一个 "updater" 函数
        # 这个函数会在每一帧被调用，根据 radius_tracker 的值来更新球体
        # Add an "updater" function to the light shell.
        # This function is called every frame to update the sphere based on the radius_tracker's value.
        light_shell.add_updater(
            lambda mob: mob.become(
                Sphere(
                    radius=radius_tracker.get_value(),
                    resolution=(32, 32)
                ).set_color(YELLOW).set_opacity(opacity_k / radius_tracker.get_value()**2)
            )
        )
        # 注意: .become() 方法会用一个全新的物体替换旧的物体。
        # 对于这个场景来说，它的性能足够好，并且代码更易于理解。
        # NOTE: The .become() method replaces the mobject with a new one.
        # For this scene, it's performant enough and makes the code easier to understand.

        # --- 4. 播放动画 (Play the Animation) ---

        # 将所有物体添加到场景中
        # Add all mobjects to the scene
        self.add(light_source, light_shell)
        self.wait(1) # 初始等待1秒

        # 播放动画：让半径从 1 增加到 4
        # As the radius increases, the opacity decreases according to 1/r^2
        self.play(
            radius_tracker.animate.set_value(4),
            run_time=8,
            rate_func=smooth # 使用线性速率函数，使变化看起来更均匀
        )
        self.wait(1)

        # 播放动画：让半径从 4 减小到 1
        # As the radius decreases, the opacity increases
        self.play(
            radius_tracker.animate.set_value(1),
            run_time=5,
            rate_func=linear
        )
        self.wait(2)

        # 结束时，可以添加一个旋转动画来更好地展示 3D 效果
        # At the end, add a rotation animation to better showcase the 3D effect
        self.move_camera(theta=360 * DEGREES, run_time=8, rate_func=linear)
        self.wait(2)

