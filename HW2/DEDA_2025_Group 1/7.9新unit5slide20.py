import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 参数
mu = 0
sigma = 1.0
color = 'red'

x = np.linspace(-10, 10, 1000)
pdf = norm.pdf(x, mu, sigma)
cdf = norm.cdf(x, mu, sigma)

plt.figure(figsize=(8, 8))
plt.suptitle("Distribution Analysis _Normal Distribution", fontsize=24, color='red', y=0.98, weight='bold')

# PDF
plt.subplot(2, 1, 1)
plt.plot(x, pdf, color=color)
plt.title(r'Probability Density Function pdf for $\mu=0$ and $\sigma=1.0$', fontsize=14, weight='bold')
plt.xlabel('X', fontsize=12)
plt.ylabel('pdf', fontsize=12)
plt.xlim(-10, 10)
plt.ylim(0, 1.0)
plt.grid(alpha=0.3)

# CDF
plt.subplot(2, 1, 2)
plt.plot(x, cdf, color=color)
plt.title(r'Cumulative Density Function cdf for $\mu=0$ and $\sigma=1.0$', fontsize=14, weight='bold')
plt.xlabel('X', fontsize=12)
plt.ylabel('cdf', fontsize=12)
plt.xlim(-10, 10)
plt.ylim(0, 1.0)
plt.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.93])
plt.figtext(0.03, 0.01, "Digital Economy & Decision Analytics", fontsize=14, color='gray', ha='left')
plt.show()
