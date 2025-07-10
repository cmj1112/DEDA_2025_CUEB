import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(100)
n = 1000  # 样本点数
msize = 5  # 标记大小（3D中需要更大些）
z_offset = 1.0  # 上下层之间的垂直间距

# === 生成原始数据（圆形分布） ===
x_orig = -np.log(np.random.uniform(low=0, high=1, size=n))
a = np.sqrt(2*x_orig)
phi = np.random.uniform(low=0, high=2 * np.pi, size=n)
x_circle = a * np.cos(phi)
y_circle = a * np.sin(phi)

# === 生成变换后数据（椭圆形分布） ===
A = [[3, 1], [1, 1]]
A_eig = np.linalg.eig(A)
E_val = A_eig[0]
Gamma = A_eig[1]
Lambda = np.diag(E_val)
Lambda12 = np.sqrt(Lambda)
A12 = np.dot(np.dot(Gamma, Lambda12), np.transpose(Gamma))
c = np.vstack([x_circle, y_circle])
tfxy = np.dot(A12, c)

# === 创建3D图形 ===
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# 设置坐标轴标签和标题
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('3D Rotating Distribution Comparison', fontsize=16)

# 设置一致的XY轴范围，便于比较
xy_padding = 1.0
xy_limits = [
    min(x_circle.min(), tfxy[0].min()) - xy_padding,
    max(x_circle.max(), tfxy[0].max()) + xy_padding
]
ax.set_xlim(xy_limits)
ax.set_ylim(xy_limits)
ax.set_zlim(-0.5, z_offset + 0.5)  # 包含两层分布的垂直范围

# === 创建两个分布层 ===
# 下层：原始圆形分布 (z=0)
scatter_circle = ax.scatter(
    x_circle,
    y_circle,
    np.zeros(n),           # Z坐标全为0
    c='blue',
    s=msize,
    alpha=0.7,
    depthshade=True,
    label='Original (Circular)'
)

# 上层：变换后椭圆分布 (z=z_offset)
scatter_ellipse = ax.scatter(
    tfxy[0],
    tfxy[1],
    np.full(n, z_offset),  # Z坐标全为z_offset
    c='red',
    s=msize,
    alpha=0.7,
    depthshade=True,
    label='Transformed (Elliptical)'
)

# 添加图例和网格
ax.legend(loc='upper right')
ax.grid(True)

# 添加半透明平面增强立体感
xx, yy = np.meshgrid(np.linspace(*xy_limits, 10), np.linspace(*xy_limits, 10))
ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.15, color='blue')
ax.plot_surface(xx, yy, np.full_like(xx, z_offset), alpha=0.15, color='red')

# === 创建旋转动画 ===
def update(frame):
    # 同时改变仰角(elev)和方位角(azim)
    elev = 15 + 10 * np.sin(frame * np.pi/45)  # 15°-25°之间摆动
    azim = frame % 360
    ax.view_init(elev=elev, azim=azim)
    return []

# 创建动画（每0.5度一帧，共720帧）
ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 0.5),
                    blit=True, interval=20)

# 保存为GIF
ani.save('3d_dual_distributions.gif', writer='pillow', fps=30, dpi=150)

plt.show()