#!/usr/bin/env python3
"""
Figures for Chapter 10: Distance and Why We Square It
======================================================

Generates:
- fig10_pythagoras.png: The Pythagorean distance between two phenotypes
- fig10_variance_as_distance.png: Variance as mean squared distance from mean
- fig10_covariance_ellipse.png: The covariance matrix defines an ellipse

Author: Daniel Ortiz-Barrientos
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
from matplotlib.lines import Line2D

# Style settings for book figures
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Color palette
BLUE = '#2E86AB'
GREEN = '#06D6A0'
ORANGE = '#F18F01'
RED = '#E63946'
PURPLE = '#762A83'
GRAY = '#666666'

# =============================================================================
# Figure 1: Pythagorean Distance
# =============================================================================

print("Creating fig10_pythagoras.png...")

fig, ax = plt.subplots(figsize=(6, 5))

# Two points
xi, yi = 1.5, 1.0
xj, yj = 4.5, 3.5

# Calculate differences
dx = xj - xi
dy = yj - yi
distance = np.sqrt(dx**2 + dy**2)

# Draw axes
ax.axhline(0, color='lightgray', linewidth=0.5, zorder=0)
ax.axvline(0, color='lightgray', linewidth=0.5, zorder=0)

# Draw the right triangle
# Horizontal leg
ax.plot([xi, xj], [yi, yi], '--', color=GRAY, linewidth=1.5, zorder=1)
# Vertical leg
ax.plot([xj, xj], [yi, yj], '--', color=GRAY, linewidth=1.5, zorder=1)
# Hypotenuse (the distance)
ax.plot([xi, xj], [yi, yj], '-', color=BLUE, linewidth=2.5, zorder=2)

# Draw points
ax.plot(xi, yi, 'o', color=ORANGE, markersize=12, zorder=3)
ax.plot(xj, yj, 'o', color=ORANGE, markersize=12, zorder=3)

# Labels for points
ax.text(xi - 0.3, yi + 0.2, 'Individual $i$\n$(x_i, y_i)$', fontsize=10, ha='right')
ax.text(xj + 0.2, yj + 0.2, 'Individual $j$\n$(x_j, y_j)$', fontsize=10, ha='left')

# Labels for legs
ax.text((xi + xj)/2, yi - 0.35, r'$\Delta x = x_j - x_i$', fontsize=10, ha='center', color=GRAY)
ax.text(xj + 0.25, (yi + yj)/2, r'$\Delta y = y_j - y_i$', fontsize=10, ha='left', va='center', color=GRAY)

# Label for hypotenuse
mid_x, mid_y = (xi + xj)/2, (yi + yj)/2
ax.text(mid_x - 0.5, mid_y + 0.3, r'$d = \sqrt{(\Delta x)^2 + (\Delta y)^2}$', 
        fontsize=11, ha='center', color=BLUE, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=0.8))

# Right angle marker
from matplotlib.patches import Rectangle
right_angle_size = 0.2
right_angle = plt.Polygon([(xj - right_angle_size, yi), 
                            (xj - right_angle_size, yi + right_angle_size),
                            (xj, yi + right_angle_size)],
                           fill=False, edgecolor=GRAY, linewidth=1)
ax.add_patch(right_angle)

ax.set_xlim(0, 6)
ax.set_ylim(0, 5)
ax.set_xlabel('Trait 1 (e.g., body length)')
ax.set_ylabel('Trait 2 (e.g., wing span)')
ax.set_aspect('equal')

plt.savefig('fig10_pythagoras.png', dpi=300, facecolor='white')
print("  Saved: fig10_pythagoras.png")
plt.close()

# =============================================================================
# Figure 2: Variance as Mean Squared Distance
# =============================================================================

print("Creating fig10_variance_as_distance.png...")

fig, ax = plt.subplots(figsize=(7, 5))

# Generate sample data
np.random.seed(42)
n = 12
x = np.random.normal(3, 1.2, n)
y = np.random.normal(3, 1.2, n)

# Compute mean
mean_x, mean_y = np.mean(x), np.mean(y)

# Draw axes
ax.axhline(0, color='lightgray', linewidth=0.5, zorder=0)
ax.axvline(0, color='lightgray', linewidth=0.5, zorder=0)

# Draw arrows from mean to each point
for i in range(n):
    ax.annotate('', xy=(x[i], y[i]), xytext=(mean_x, mean_y),
                arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.5, alpha=0.6))

# Draw points
ax.scatter(x, y, s=80, c=ORANGE, zorder=3, edgecolor='white', linewidth=1)

# Draw mean
ax.plot(mean_x, mean_y, 's', color=RED, markersize=14, zorder=4, 
        markeredgecolor='white', markeredgewidth=2)
ax.text(mean_x + 0.25, mean_y + 0.25, r'Mean $(\bar{x}, \bar{y})$', fontsize=11, 
        color=RED, fontweight='bold')

# Add annotation
ax.text(0.5, 5.5, r'Variance $= \frac{1}{n-1}\sum_i \|\mathbf{d}_i\|^2$', 
        fontsize=12, ha='left',
        bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor=ORANGE, alpha=0.9))

ax.text(0.5, 4.8, r'Each arrow $\mathbf{d}_i$ connects mean to individual $i$',
        fontsize=10, ha='left', color=GRAY)

ax.set_xlim(0, 6)
ax.set_ylim(0, 6)
ax.set_xlabel('Trait 1')
ax.set_ylabel('Trait 2')
ax.set_aspect('equal')

plt.savefig('fig10_variance_as_distance.png', dpi=300, facecolor='white')
print("  Saved: fig10_variance_as_distance.png")
plt.close()

# =============================================================================
# Figure 3: Covariance Ellipse
# =============================================================================

print("Creating fig10_covariance_ellipse.png...")

fig, ax = plt.subplots(figsize=(7, 6))

# Generate correlated data
np.random.seed(123)
n = 50
mean = [4, 5]
cov = [[2.5, 2.0], [2.0, 2.5]]  # Positive correlation

data = np.random.multivariate_normal(mean, cov, n)
x, y = data[:, 0], data[:, 1]

# Compute sample covariance
sample_cov = np.cov(x, y)
sample_mean = [np.mean(x), np.mean(y)]

# Eigendecomposition for ellipse
eigenvalues, eigenvectors = np.linalg.eigh(sample_cov)

# Draw axes
ax.axhline(0, color='lightgray', linewidth=0.5, zorder=0)
ax.axvline(0, color='lightgray', linewidth=0.5, zorder=0)

# Draw data points
ax.scatter(x, y, s=40, c=BLUE, alpha=0.6, zorder=2, edgecolor='white', linewidth=0.5)

# Draw mean
ax.plot(sample_mean[0], sample_mean[1], 's', color=RED, markersize=12, zorder=4,
        markeredgecolor='white', markeredgewidth=2)

# Draw covariance ellipse (1 and 2 standard deviations)
for n_std, alpha in [(1, 0.3), (2, 0.15)]:
    # Ellipse parameters
    width = 2 * n_std * np.sqrt(eigenvalues[1])
    height = 2 * n_std * np.sqrt(eigenvalues[0])
    angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
    
    ellipse = Ellipse(sample_mean, width, height, angle=angle,
                      fill=True, facecolor=GREEN, alpha=alpha,
                      edgecolor=GREEN if n_std == 1 else 'none', linewidth=2)
    ax.add_patch(ellipse)

# Draw eigenvector axes
for i, (ev, col) in enumerate(zip([eigenvalues[1], eigenvalues[0]], [PURPLE, ORANGE])):
    vec = eigenvectors[:, 1-i] * np.sqrt(ev) * 1.5
    ax.annotate('', xy=(sample_mean[0] + vec[0], sample_mean[1] + vec[1]),
                xytext=sample_mean,
                arrowprops=dict(arrowstyle='->', color=col, lw=2))
    ax.annotate('', xy=(sample_mean[0] - vec[0], sample_mean[1] - vec[1]),
                xytext=sample_mean,
                arrowprops=dict(arrowstyle='->', color=col, lw=2))

# Labels
ax.text(sample_mean[0] + 0.3, sample_mean[1] - 0.5, 'Mean', fontsize=10, color=RED)

# Annotation box - use simple text since matplotlib mathtext doesn't support pmatrix
ax.text(0.5, 8.5, 'Covariance matrix defines the ellipse:', fontsize=11)
ax.text(0.5, 7.6, r'$\mathbf{S}$ = [variance$_1$, covariance; covariance, variance$_2$]', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor=BLUE, alpha=0.9))

ax.text(7, 2, 'Eigenvectors\n= axis directions', fontsize=9, ha='center', color=PURPLE)
ax.text(7, 1, 'Eigenvalues\n= axis lengths²', fontsize=9, ha='center', color=ORANGE)

ax.set_xlim(0, 9)
ax.set_ylim(0, 9)
ax.set_xlabel('Trait 1')
ax.set_ylabel('Trait 2')
ax.set_aspect('equal')

plt.savefig('fig10_covariance_ellipse.png', dpi=300, facecolor='white')
print("  Saved: fig10_covariance_ellipse.png")
plt.close()

# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*60)
print("Chapter 10 Figures Complete")
print("="*60)
print("""
Generated figures:
  1. fig10_pythagoras.png
     - Two phenotypes in trait space
     - Right triangle showing Δx, Δy, and distance d
     
  2. fig10_variance_as_distance.png  
     - Sample of individuals around a mean
     - Arrows showing deviations
     - Variance as average squared arrow length
     
  3. fig10_covariance_ellipse.png
     - Correlated bivariate data
     - Covariance ellipse at 1 and 2 SD
     - Eigenvectors showing principal axes
""")
