import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

st.set_page_config(page_title="Normal Distribution Explorer", layout="centered")

st.markdown(
    "<h2 style='color: red;'>Distribution Analysis _Normal Distribution</h2>",
    unsafe_allow_html=True
)

# Sidebar controls
st.sidebar.header("Controls")
mu = st.sidebar.slider("μ (mean)", -5.0, 5.0, 0.0, 0.1)
sigma = st.sidebar.slider("σ (std dev)", 0.5, 5.0, 1.0, 0.1)
color = st.sidebar.selectbox("Curve Color", options=["red", "blue", "green", "orange", "black"], index=0)

x = np.linspace(-10, 10, 1000)
pdf = norm.pdf(x, mu, sigma)
cdf = norm.cdf(x, mu, sigma)

fig, axes = plt.subplots(2, 1, figsize=(8, 6))
# PDF
axes[0].plot(x, pdf, color=color, lw=2)
axes[0].set_title(f"Probability Density Function pdf for μ={mu} and σ={sigma}", fontsize=12, weight='bold')
axes[0].set_xlabel('X')
axes[0].set_ylabel('pdf')
axes[0].set_xlim(-10, 10)
axes[0].set_ylim(0, 1)
axes[0].grid(alpha=0.3)

# CDF
axes[1].plot(x, cdf, color=color, lw=2)
axes[1].set_title(f"Cumulative Density Function cdf for μ={mu} and σ={sigma}", fontsize=12, weight='bold')
axes[1].set_xlabel('X')
axes[1].set_ylabel('cdf')
axes[1].set_xlim(-10, 10)
axes[1].set_ylim(0, 1)
axes[1].grid(alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

st.markdown(
    "<div style='text-align: left; color: gray;'>Digital Economy & Decision Analytics</div>",
    unsafe_allow_html=True
)
