import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn import svm
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# 确保负号正确显示
plt.rcParams["axes.unicode_minus"] = False

# 生成月牙形数据
X, y = make_moons(n_samples=100, noise=0.2, random_state=42)

# 训练SVM模型
model = svm.SVC(C=10, kernel='rbf', gamma='scale')
model.fit(X, y)

# 创建网格数据
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 创建3D图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制决策边界
contour = ax.contourf(xx, yy, Z, alpha=0.5, cmap='coolwarm', levels=20)
ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')

# 绘制数据点
distance = model.decision_function(X)
scatter = ax.scatter(X[:, 0], X[:, 1], distance, c=y, cmap='coolwarm', 
                     edgecolors='k', s=60, alpha=0.8)

# 设置英文标签和标题
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Distance to Decision Boundary')
ax.set_title('SVM Rotation Plot (RBF Kernel)')

# 颜色条
cbar = fig.colorbar(contour, ax=ax, shrink=0.7, aspect=10)
cbar.set_label('Decision Function Value')

# 初始化视角
ax.view_init(elev=30, azim=45)

# 定义旋转函数
def update(frame):
    ax.view_init(elev=30, azim=frame)
    return scatter,

# 创建动画
ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), 
                    interval=100, blit=True)

# 保存动画（可选）
# ani.save('svm_rotation_plot.gif', writer='pillow', fps=10)

plt.tight_layout()
plt.show()
