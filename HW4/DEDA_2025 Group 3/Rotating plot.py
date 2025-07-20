import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 生成模拟数据
np.random.seed(42)
n = 50
x1 = np.random.uniform(-1, 1, n)
x2_red = np.random.uniform(0.2, 1, n)
x2_blue = np.random.uniform(-1, -0.2, n)

# 标签红点与蓝点
red = np.column_stack((x1, x2_red))
blue = np.column_stack((x1, x2_blue))

# 映射函数 ψ(x)
def psi(x):
    x1, x2 = x[:, 0], x[:, 1]
    return np.column_stack((x1**2, np.sqrt(2)*x1*x2, x2**2))

# 映射到3D特征空间
red_3d = psi(red)
blue_3d = psi(blue)

# 画图
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(red_3d[:, 0], red_3d[:, 1], red_3d[:, 2], color='red', label='Red Class')
ax.scatter(blue_3d[:, 0], blue_3d[:, 1], blue_3d[:, 2], color='blue', label='Blue Class')

ax.set_xlabel('z1 = x1^2')
ax.set_ylabel('z2 = √2·x1·x2')
ax.set_zlabel('z3 = x2^2')
ax.set_title('Rotation Plot: Mapping to Feature Space')
ax.view_init(elev=30, azim=45)  # 旋转角度
ax.legend()
plt.tight_layout()
plt.show()
