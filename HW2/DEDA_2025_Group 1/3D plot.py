import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

# 设置全局字体和图像DPI - 确保PPT中显示清晰
plt.rcParams.update({
    'font.family': 'Arial',  # 使用PPT常用字体
    'font.size': 14,
    'figure.dpi': 150,
    'axes.titlesize': 18,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'savefig.facecolor': 'white'  # 保存为白色背景
})

# ============================= 生成原始数据 =============================
np.random.seed(100)
n = 800  # 数据点数量

# 生成标准正态分布 (底层z=0平面)
x = -np.log(np.random.uniform(low=0, high=1, size=n))
a = np.sqrt(2 * x)
phi = np.random.uniform(low=0, high=2 * np.pi, size=n)
x0 = a * np.cos(phi)
y0 = a * np.sin(phi)
z0 = np.zeros(n)

# 添加高斯噪声使分布更自然
x0 += np.random.normal(0, 0.1, n)
y0 += np.random.normal(0, 0.1, n)

# ============================= 计算协方差变换 =============================
# 目标协方差矩阵
A = np.array([[3, 1], [1, 1]])

# 特征分解计算矩阵平方根
A_eig = np.linalg.eig(A)
E_val = A_eig[0]
Gamma = A_eig[1]
Lambda = np.diag(E_val)
Lambda12 = np.sqrt(Lambda)
A12 = np.dot(np.dot(Gamma, Lambda12), np.transpose(Gamma))

# 应用变换 (顶层z=1平面)
orig_points = np.vstack((x0, y0))
trans_points = np.dot(A12, orig_points)
x1 = trans_points[0, :]
y1 = trans_points[1, :]
z1 = np.ones(n)

# ============================= 创建3D图形 =============================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制底层平面（原始分布）
scatter0 = ax.scatter(x0, y0, z0,
                      s=15,
                      c='royalblue',
                      alpha=0.8,
                      depthshade=True,
                      label='Standard Normal (Σ=I)')

# 绘制顶层平面（变换后分布）
scatter1 = ax.scatter(x1, y1, z1,
                      s=15,
                      c='crimson',
                      alpha=0.8,
                      depthshade=True,
                      label='Transformed (Σ=A)')

# 添加连接线显示对应关系（部分采样）
lines = []
for i in range(0, n, 15):
    line, = ax.plot([x0[i], x1[i]],
                    [y0[i], y1[i]],
                    [z0[i], z1[i]],
                    'k-', alpha=0.15, lw=0.8)
    lines.append(line)

# 设置图形属性 - 确保在PPT中清晰可见
ax.set_xlabel('X Axis', fontweight='bold', labelpad=10, fontsize=14)
ax.set_ylabel('Y Axis', fontweight='bold', labelpad=10, fontsize=14)
ax.set_zlabel('Z Plane', fontweight='bold', labelpad=10, fontsize=14)
ax.set_title('Box-Muller Transformations', pad=15, fontsize=18)
ax.legend(loc='upper right', framealpha=0.9, fontsize=12)
ax.grid(True, linestyle=':', alpha=0.4)

# 设置坐标轴范围
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.set_zlim([-0.5, 1.5])

# 设置视角初始位置
ax.view_init(elev=30, azim=45)


# ============================= 自动旋转动画 =============================
def update_angle(frame):
    """更新视角实现自动旋转"""
    # 计算平滑的旋转角度
    azim = frame % 360
    elev = 20 + 15 * np.sin(np.radians(frame * 2))  # 仰角在20-35度之间平滑变化

    ax.view_init(elev=elev, azim=azim)
    return fig,


# 创建动画 (每帧1度，共360帧)
ani = FuncAnimation(fig,
                    update_angle,
                    frames=np.arange(0, 360, 2),  # 每2度一帧，减少GIF大小
                    interval=50,  # 50ms/帧
                    blit=False)

# ============================= 保存为GIF =============================
print("正在生成GIF动画，请稍候...")
gif_path = '3d_rotation.gif'

# 使用PillowWriter保存为GIF
writer = PillowWriter(fps=15,  # 每秒15帧
                      metadata=dict(artist='Box-Muller Visualization'),
                      bitrate=1800)  # 控制文件大小

ani.save(gif_path, writer=writer, dpi=100)

print(f"GIF已保存至: {gif_path}")
print("提示: 您可以在PPT中直接插入此GIF文件")

# 显示成功消息但不显示图表（因为我们要保存为GIF）
plt.close(fig)  # 关闭图表以节省内存