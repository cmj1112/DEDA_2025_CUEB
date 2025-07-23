from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from sklearn import svm

# 生成数据
np.random.seed(42)
X, y = make_moons(n_samples=100, noise=0.1)

# 1. 原始二维空间的可视化
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='coolwarm', edgecolors='k')
plt.title("Original 2D Space - Moons Dataset")
plt.savefig('2D_moons.png', dpi=200, transparent=True)
plt.close()

# 2. 定义映射函数到三维空间 (特征空间映射)
def map_to_3D(X):
    """将2D数据映射到3D特征空间(多项式核函数的特征映射)"""
    x1 = X[:, 0]
    x2 = X[:, 1]
    # 映射ψ(x) = (x1², x2², √2x1x2)
    return np.c_[x1**2, x2**2, np.sqrt(2)*x1*x2]

# 映射数据到3D空间
X_3D = map_to_3D(X)

# 在特征空间中训练线性SVM
model = svm.SVC(kernel='linear', C=100)
model.fit(X_3D, y)

# 3. 创建3D可视化
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制数据点（不同类别不同颜色）
scatter = ax.scatter(
    X_3D[:, 0],
    X_3D[:, 1],
    X_3D[:, 2],
    c=y,
    s=50,
    cmap='coolwarm',
    edgecolors='k',
    depthshade=True
)

# 创建网格用于绘制决策平面
xx, yy = np.meshgrid(
    np.linspace(X_3D[:, 0].min()-1, X_3D[:, 0].max()+1, 30),
    np.linspace(X_3D[:, 1].min()-1, X_3D[:, 1].max()+1, 30)
)

# 计算决策边界平面方程
w = model.coef_[0]
b = model.intercept_[0]
z_func = lambda x, y: (-w[0]*x - w[1]*y - b) / w[2]

# 计算网格对应的z值
zz = z_func(xx, yy)

# 绘制决策平面
plane = ax.plot_surface(
    xx, yy, zz,
    alpha=0.5,
    color='lightblue',
    rstride=1,
    cstride=1
)

# 添加标题和标签
ax.set_title("3D Feature Space Mapping and Linear Separation", fontsize=14)
ax.set_xlabel('$X^2$', fontsize=12)
ax.set_ylabel('$Y^2$', fontsize=12)
ax.set_zlabel(r'$\sqrt{2}XY$', fontsize=12)
fig.colorbar(scatter, ax=ax, label='Class')

# 4. 旋转动画
def rotate(angle):
    """更新视角的函数"""
    ax.view_init(elev=30, azim=angle)
    return plane,  # 返回可迭代对象以支持动画

# 创建动画对象
ani = animation.FuncAnimation(
    fig,
    rotate,
    frames=np.arange(0, 360, 2),  # 0到360度，2度一帧
    interval=30,  # 帧间隔(ms)
    blit=True
)

# 保存GIF动画
ani.save('3d_svm_rotation.gif', writer='pillow', dpi=150, fps=20)

# 5. 交互式3D视图(可选)
"""
fig2 = plt.figure(figsize=(10, 8))
ax2 = fig2.add_subplot(111, projection='3d')
ax2.scatter(X_3D[:, 0], X_3D[:, 1], X_3D[:, 2], c=y, s=50, cmap='coolwarm', edgecolors='k')
ax2.plot_surface(xx, yy, zz, alpha=0.5, color='lightblue')
ax2.set_xlabel('$X^2$')
ax2.set_ylabel('$Y^2$')
ax2.set_zlabel('$\sqrt{2}XY$')
ax2.view_init(elev=25, azim=45)  # 设置初始视角
plt.tight_layout()
plt.savefig('3d_static_view.png', dpi=200, transparent=True)
plt.show()
"""