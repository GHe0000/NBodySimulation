from manim import *
import csv
import io

SIMULATION_DATA_STRING = """
1; Aarseth ;1966;100;0
2; Peebles ;1970;300;0
3; Press & Schechter ;1974;1000;0
4; Miyoshi & Kihara ;1975;400;-0.1
5; White ;1976;700;0.3
6; Aarseth et al.;1979;4000;0
7; Terlevich ;1980;1000;0
8; Efstathiou & Eastwood ;1981;20000;0
9; White, Frenk & Davis ;1983;32768;0
10; Davis et al. ;1985;32768;0.15
11; Quinn et al.;1986;7000;0
12; Inagaki ;1986;3000;0
13; White et al.;1987;216000;0
14; Carlberg & Couchman ;1989;524288;0
15; Dubinski & Carlberg ;1991;300000;-0.05
16; Suto & Suginohara ;1991;2097152;-0.1
17; Warren et al.;1992;1000000;0
18; Couchman & Carlberg ;1992;2097152;0.25
19; Aarseth & Heggie ;1993;6000;0
20; Gelb & Bertschinger ;1994;2985984;0.01
21; Zurek et al.;1994;17154598;0
22; Spurzem et al.;1996;10000;0
23; Makino ;1996;32768;0
24; Jenkins et al. ;1998;16777216;0
25; Governato et al. ;1999;47000000;0
26; Colberg et al. ;2000;1000000000;0.1
27; Bode et al.;2001;134217728;0
28; Baumgardt & Makino ;2003;131028;0
29; Wambsganss et al.;2004;1073741824;-0.1
30; Portegies Zwart et al. ;2004;140000;-0.15
31; Springel et al.;2005;10077696000;0
"""

CN_FONT = "思源黑体 CN" # Arch Linux 示例
# CN_FONT = "Source Han Sans CN" # Windows 示例

config.font = CN_FONT

class Moore(Scene):
    def construct(self):
        data = []
        f = io.StringIO(SIMULATION_DATA_STRING)
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            if not row or not row[0].strip():
                continue
            try:
                author = row[1].strip()
                year = int(row[2].strip())
                n_particles = int(row[3].strip())
                offset = float(row[4].strip()) # 读取偏移量
                if year <= 2005:
                    # 清理作者姓名
                    if '(Millennium)' in author:
                        author = 'Springel et al.'
                    elif 'et al.' in author:
                        author = author.split('et al.')[0] + 'et al.'
                    elif len(author.split(',')) > 2:
                         author = author.split(',')[0] + ' et al.'
                    data.append({"author": author, "year": year, "n": n_particles, "offset": offset})
            except (ValueError, IndexError):
                continue

        data.sort(key=lambda x: x['year'])

        # 2. 设置坐标轴
        axes = Axes(
            x_range=[1965, 2006, 5],
            y_range=[2, 11, 1],
            x_length=12,
            y_length=6,
            axis_config={"color": BLUE},
            x_axis_config={
                "numbers_to_include": np.arange(1970, 2010, 5),
                "label_direction": DOWN,
            },
            y_axis_config={
                "scaling": LogBase(10),
                "numbers_to_include": np.arange(2, 12, 1),
            },
            tips=False,
        )

        # 手动旋转 X 轴的刻度标签
        for label in axes.x_axis.numbers:
            label.rotate(-PI / 4)

        # 将坐标轴整体移动，为标签留出更多空间
        axes.shift(UP * 0.4 + RIGHT * 0.6)

        x_label = axes.get_x_axis_label(
            Text("年份", font_size=24), 
            edge=DOWN, 
            direction=DOWN,
            buff=0.8
        )

        y_label = axes.get_y_axis_label(
            Text("粒子数 (N)", font_size=24).rotate(PI/2), 
            edge=LEFT, 
            direction=LEFT, 
            buff=0.8
        )
        title = Text("N-Body 模拟粒子数 (1966-2005)", font_size=30).to_edge(UP)
        
        self.play(Write(title))
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.wait(1)

        # 3. 创建并动画化柱状图
        animations = []
        bar_group = VGroup()

        for item in data:
            year = item['year']
            n_particles = item['n']
            author = item['author']
            
            min_x_val = axes.x_range[0]
            min_y_val = 10**axes.y_range[0]

            x_pos = axes.c2p(year, min_y_val)[0]
            y_pos = axes.c2p(min_x_val, n_particles)[1]
            y_bottom = axes.c2p(min_x_val, min_y_val)[1]

            bar = Rectangle(
                width=0.4,
                height=y_pos - y_bottom,
                fill_color=interpolate_color(YELLOW, RED, (np.log10(n_particles) - 2) / 9),
                fill_opacity=0.8,
                stroke_width=1,
                stroke_color=WHITE,
            )
            bar.move_to(np.array([x_pos, y_bottom, 0]), aligned_edge=DOWN)

            # 创建作者标签并应用手动设置的偏移量
            label = Text(author, font_size=12, color=WHITE).next_to(bar, UP, buff=0.1)
            label.shift(UP * item['offset']) # 应用垂直偏移
            
            bar_with_label = VGroup(bar, label)
            bar_group.add(bar_with_label)
            
            animations.append(GrowFromEdge(bar, DOWN))

        self.play(LaggedStart(*animations, lag_ratio=0.2), run_time=4)
        
        self.play(
            LaggedStart(
                *[Write(bar.submobjects[1]) for bar in bar_group],
                lag_ratio=0.1
            ),
            run_time=2
        )

        self.wait(3)
