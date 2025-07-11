# NBodySimulation

[![wakatime](https://wakatime.com/badge/user/70908aa3-b2c6-4f44-a07f-7bd45f260e48/project/8d71157a-a2b7-40e5-b06b-88a9af1f501a.svg)](https://wakatime.com/badge/user/70908aa3-b2c6-4f44-a07f-7bd45f260e48/project/8d71157a-a2b7-40e5-b06b-88a9af1f501a)

没有任何优化算法的直接计算的 N-body 运动模拟，这里尝试复现了 Jeremiah Ostriker 和 James Peebles 的利用早期 N 体模拟对鼓励星系的星系盘进行稳定性分析的研究.

> [!WARNING]
> 目前模拟中的初始条件可能和原论文中的有些不同，论文中的有些描述不确定是否和模拟的现象进行对应，但大体的现象是一致的.

## 运行方法

1. 克隆或下载本仓库.
2. 安装 Python 环境，并安装以下库：
    - numpy
    - matplotlib
    - numba
3. 运行 `nbody.py` 进行计算并绘图.

> [!IMPORTANT]
> 运行前可以修改 `nbody.py` 中的模拟参数
