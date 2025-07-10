# ""
# 用傅里叶参数画大象并生成摆鼻动画GIF
# 原始作者：Piotr A. Zolnierczuk, David Bailey
# 修改：Junjie Hu
# 中文化注释：DeepSeek Chat
# """

import matplotlib
matplotlib.use('TKAgg')  # 设置matplotlib后端
from matplotlib import animation
from numpy import append, cos, linspace, pi, sin, zeros
import matplotlib.pyplot as plt

# 大象参数（复数表示），最后一个参数控制眼睛
parameters = [0 - 20j, 30 + 10j, 30 + 20j, 20 - 50j, -35 + 13j]

def fourier(t, C):
    """傅里叶级数计算"""
    f = zeros(t.shape)
    for k in range(len(C)):
        f += C.real[k] * cos(k * t) + C.imag[k] * sin(k * t)
    return f

def elephant(t, p):
    """生成大象坐标"""
    npar = 6
    Cx = zeros((npar,), dtype='complex')
    Cy = zeros((npar,), dtype='complex')

    Cx[1] = p[0].real * 1j
    Cy[1] = p[3].imag + p[0].imag * 1j

    Cx[2] = p[1].real * 1j
    Cy[2] = p[1].imag * 1j

    Cx[3] = p[2].real
    Cy[3] = p[2].imag * 1j

    Cx[5] = p[3].real

    x = append(fourier(t, Cy), [p[4].imag])
    y = -append(fourier(t, Cx), [-p[4].imag])
    return x, y

def init_plot():
    """初始化绘图（大象身体）"""
    x, y = elephant(linspace(2 * pi + 0.9 * pi, 0.4 + 3.3 * pi, 1000), parameters)
    for ii in range(len(y) - 1):
        y[ii] -= sin(((x[ii] - x[0]) * pi / len(y))) * sin(float(0)) * parameters[4].real
    trunk.set_data(x, y)
    return trunk,

def move_trunk(i):
    """控制鼻子摆动"""
    x, y = elephant(linspace(2 * pi + 0.9 * pi, 0.4 + 3.3 * pi, 1000), parameters)
    for ii in range(len(y) - 1):
        y[ii] -= sin(((x[ii] - x[0]) * pi / len(y))) * sin(float(i)) * parameters[4].real
    trunk.set_data(x, y)
    return trunk,

# 创建图形和坐标轴
fig, ax = plt.subplots()

# 绘制大象身体（静态部分）
x, y = elephant(t=linspace(0.4 + 1.3 * pi, 2 * pi + 0.9 * pi, 1000), p=parameters)
plt.plot(x, y, 'b.')
plt.xlim([-75, 90])
plt.ylim([-70, 87])
plt.axis('off')

# 初始化鼻子（动态部分）
trunk, = ax.plot([], [], 'b.')

# 创建动画（减少到100帧以提高GIF生成速度）
ani = animation.FuncAnimation(
    fig=fig,
    func=move_trunk,
    frames=100,
    init_func=init_plot,
    interval=50,
    blit=False,
    repeat=True
)

ani.save('/Users/liboyang/Desktop/fishfish_fish.gif', writer='pillow', fps=30)
