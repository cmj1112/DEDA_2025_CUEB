import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 蚯蚓参数
parameters = [15 - 8j, 5 + 3j, 1 - 2j, 0 + 0j]


def fourier(t, C):
    f = np.zeros(t.shape)
    for k in range(len(C)):
        f += C.real[k] * np.cos(k * t) + C.imag[k] * np.sin(k * t)
    return f


def worm(t, p):
    npar = 4
    Cx = np.zeros((npar,), dtype='complex')
    Cy = np.zeros((npar,), dtype='complex')
    Cx[1] = p[0].real * 1j
    Cy[1] = p[0].imag * 1j
    Cx[2] = p[1].real * 1j
    Cy[2] = p[1].imag * 1j
    x = fourier(t, Cx)
    y = -fourier(t, Cy)
    return x, y


# 初始化图形
fig, ax = plt.subplots()
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
ax.axis('off')

# 蚯蚓身体和眼睛（初始化）
worm_line, = ax.plot([], [], 'g-', lw=3)
eye, = ax.plot([], [], 'ko', markersize=8)  # 黑色圆点作为眼睛


def init():
    worm_line.set_data([], [])
    eye.set_data([], [])
    return worm_line, eye


def update(frame):
    t = np.linspace(0, 2 * np.pi, 1000)

    # 动态参数（让蚯蚓蠕动）
    dynamic_params = [
        parameters[0] * (1 + 0.1 * np.sin(frame * 0.1)),
        parameters[1],
        parameters[2],
        parameters[3]
    ]

    x, y = worm(t, dynamic_params)
    worm_line.set_data(x, y)

    # 关键修改：眼睛位置 = 头部前10%的点 + 垂直偏移
    head_idx = int(0.1 * len(x))  # 取前10%的点作为头部
    eye_x = x[head_idx]
    eye_y = y[head_idx] + 1.5  # 向上偏移1.5单位
    eye.set_data([eye_x], [eye_y])  # 必须传入列表

    return worm_line, eye


# 创建动画
ani = FuncAnimation(
    fig, update, frames=50, init_func=init,
    blit=True, interval=100, repeat=True
)

plt.title("Wiggling Worm with Eye (Corrected Position)")
plt.show()
