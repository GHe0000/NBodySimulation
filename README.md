# NBodySimulation

[![wakatime](https://wakatime.com/badge/user/70908aa3-b2c6-4f44-a07f-7bd45f260e48/project/8d71157a-a2b7-40e5-b06b-88a9af1f501a.svg)](https://wakatime.com/badge/user/70908aa3-b2c6-4f44-a07f-7bd45f260e48/project/8d71157a-a2b7-40e5-b06b-88a9af1f501a)

一个简单的基于 PM 算法的二维 N-Body 宇宙学模拟.

> [!NOTE]
> 本仓库除了包含最主要的基于 PM 算法的 N-Body 宇宙学模拟代码外，还包含其他一些宇宙学数值模拟代码，这些代码被放在子文件中. 点进子文件夹即可查看相关说明.

> [!WARNING]
> 目前模拟中的功率谱直接采用了一个及其简单的基于幂律分布的功率谱，并未来自现实的观测数据，和现实宇宙中的不同. 因此模拟仅为了演示宇宙大尺度结构的形成过程.

算法原理（未编写完成）可以查看 `doc` 文件夹下的 `NBody宇宙学模拟.pdf`，或者在此 Blog 查看: [模拟算法介绍](https://ghe0000.pp.ua/%E6%95%B0%E5%80%BC%E8%AE%A1%E7%AE%97/NBody%E5%AE%87%E5%AE%99%E5%AD%A6%E6%A8%A1%E6%8B%9F/)

## 运行方法

1. 克隆或下载本仓库.
2. 安装 Python 环境，并安装以下库：
    - numpy
    - matplotlib
    - numba
3. 运行 `NBodyCalc.py` 进行计算，计算数据会自动保存在 `data` 文件夹下.
4. 运行 `drawTri.py` 进行结果绘制（需要先进行计算）
5. 运行 `genAnimation.py` 绘制动画（需要安装 ffmpeg 以导出视频）

> [!IMPORTANT]
> 运行前可以修改 `NBodyCalc.py` 中的模拟参数，但需要注意在 `drawTri.py` 和 `genAnimation.py` 中也要相应修改参数以匹配计算结果.
