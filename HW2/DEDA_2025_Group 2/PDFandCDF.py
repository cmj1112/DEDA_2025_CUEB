import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import plotly.express as px
from ipywidgets import interact, interactive, fixed, interact_manual

# Set up figure size of plot
plt.rcParams['figure.figsize'] = (10, 8)  # length and width
plt.rcParams['figure.dpi'] = 120  # general box size

"""
We define the general environment of our plot: size, colour and style. 
"""


# Plot the cdf and pdf in one figure
def f(mu, sigma, colour):
    fig = plt.figure(figsize=(10, 8))
    fig.subplots_adjust(hspace=0.3)
    fig.patch.set_facecolor('#eeefef')
    plt.style.use('classic')

    # upper plot: pdf
    plt.subplot(2, 1, 1)  # (rows, columns, which one)
    x_axis = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 1000)
    plt.plot(x_axis, stats.norm.pdf(x_axis, mu, sigma), c=colour, linewidth=2)
    plt.xlabel('X')
    plt.ylabel('pdf')
    plt.ylim(0, 1)
    plt.xlim(-10, 10)
    plt.title(rf'Probability Density Function pdf for $\mu= {mu}$ and $\sigma= {round(sigma, 2)}$', fontweight="bold")
    # lower plot: cdf
    plt.subplot(2, 1, 2)
    plt.plot(x_axis, stats.norm.cdf(x_axis, mu, sigma), c=colour, linewidth=2)
    plt.xlabel('X')
    plt.ylabel('cdf')
    plt.ylim(0, 1)
    plt.xlim(-10, 10)
    plt.title(rf'Cumulative Density Function cdf for $\mu= {mu}$ and $\sigma= {round(sigma, 2)}$', fontweight="bold")


colours = ['red', 'green', 'blue']
interact(f, mu=(-10, 10, 1), sigma=(1, 1, 0.5), colour=colours)
plt.savefig("filename.png", facecolor='white')