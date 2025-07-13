# 开发人员:xiaol
# 开发时间:2025/7/13 10:53
# 文件名称:HW2_codePDF_plot.PY
# 开发工具:PyCharm

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D

# Global parameter settings
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 14,
    'figure.dpi': 150,
    'axes.titlesize': 18,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'savefig.facecolor': 'white'
})

# =============== Generate 3D normal distribution point cloud ===============
np.random.seed(42)
n = 1000  # Number of points
mu = np.array([1, 2, -1])  # 3D mean
SIGMA = np.array([[3, 1, 1.5],
                  [1, 2, 0.5],
                  [1.5, 0.5, 2]])  # 3D covariance matrix

data = np.random.multivariate_normal(mu, SIGMA, size=n)  # shape: (n, 3)
x, y, z = data[:, 0], data[:, 1], data[:, 2]

# =============== Plot 3D point cloud ===============
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(x, y, z,
                c=z,           # Color varies with z
                cmap='viridis',
                s=20, alpha=0.8,
                edgecolors='w', linewidth=0.2)

ax.set_xlabel('X', fontweight='bold', labelpad=10, fontsize=14)
ax.set_ylabel('Y', fontweight='bold', labelpad=10, fontsize=14)
ax.set_zlabel('Z', fontweight='bold', labelpad=10, fontsize=14)
ax.set_title('3D Multivariate Normal Distribution', pad=15, fontsize=18)
ax.grid(True, linestyle=':', alpha=0.3)
ax.set_xlim(mu[0] - 4*np.sqrt(SIGMA[0,0]), mu[0] + 4*np.sqrt(SIGMA[0,0]))
ax.set_ylim(mu[1] - 4*np.sqrt(SIGMA[1,1]), mu[1] + 4*np.sqrt(SIGMA[1,1]))
ax.set_zlim(mu[2] - 4*np.sqrt(SIGMA[2,2]), mu[2] + 4*np.sqrt(SIGMA[2,2]))

ax.view_init(elev=25, azim=30)
cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1, label='Z Value')

# =============== 3D rotation animation ===============
def rotate(angle):
    ax.view_init(elev=25, azim=angle)
    return fig,

ani = FuncAnimation(fig, rotate, frames=np.arange(0, 360, 2), interval=50, blit=False)

# =============== Save GIF ===============
gif_path = '3d_normal.gif'
print("Generating 3D point cloud GIF animation...")
writer = PillowWriter(fps=15, metadata=dict(artist='3D Normal'), bitrate=1800)
ani.save(gif_path, writer=writer, dpi=100)
print(f"GIF saved to: {gif_path}\nCan be inserted into PPT or webpage for dynamic demonstration")

plt.close(fig)