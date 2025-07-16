# 开发人员:xiaol
# 开发时间:2025/7/13 11:01
# 文件名称:HW2_2.PY
# 开发工具:PyCharm

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.animation import FuncAnimation, PillowWriter

# Set parameters
mu = np.array([0, 0])
sigma = np.array([[1, 0.8], [0.8, 1]])

# Create grid and calculate PDF
x = y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))
rv = multivariate_normal(mu, sigma)
Z = rv.pdf(pos)

# Create figure and 3D axes
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot surface
surf = ax.plot_surface(X, Y, Z, cmap=cm.plasma,
                      rstride=2, cstride=2,
                      alpha=0.8, linewidth=0.1)

# Set figure properties
ax.view_init(elev=25, azim=45)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Density')
ax.set_title(f'Bivariate Normal Distribution\nμ={mu}, Σ={sigma.tolist()}')
fig.colorbar(surf, shrink=0.6, aspect=10)
plt.tight_layout()

# Animation function
def update(frame):
    ax.view_init(elev=25, azim=frame)
    return fig,

# Create and save animation
ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50)
ani.save('normal_distribution.gif', writer=PillowWriter(fps=15), dpi=100)

print("GIF animation saved as 'normal_distribution.gif'")
plt.close()