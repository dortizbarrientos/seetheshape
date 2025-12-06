#!/usr/bin/env python3
"""
Figures for Chapter 31: The Fitness Surface and Gamma
=======================================================

Generates:
- fig31_fitness_surface.png: 3D fitness surface with contours
- fig31_curvature_types.png: Stabilising, flat, disruptive curvature
- fig31_gaussian_surface.png: Gaussian fitness surface with elliptical contours

Author: Daniel Ortiz-Barrientos & Claude
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Style settings for book figures
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
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
LIGHT_BLUE = '#a8d4e6'
LIGHT_GREEN = '#a8f0d4'
TEAL = '#20B2AA'

# =============================================================================
# Figure 1: Fitness Surface (3D)
# =============================================================================

print("Creating fig31_fitness_surface.png...")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create meshgrid
x = np.linspace(-2, 2, 50)
y = np.linspace(-2, 2, 50)
X, Y = np.meshgrid(x, y)

# Gaussian fitness surface with asymmetric width
# w(z) = exp(-0.5 * z' * omega^{-1} * z)
# omega = [[1.5, 0.3], [0.3, 0.8]] -> wider along one axis
omega = np.array([[1.5, 0.3], [0.3, 0.8]])
omega_inv = np.linalg.inv(omega)

# Compute fitness at each point
W = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        z = np.array([X[i, j], Y[i, j]])
        W[i, j] = np.exp(-0.5 * z @ omega_inv @ z)

# Plot surface
surf = ax.plot_surface(X, Y, W, cmap=cm.viridis, alpha=0.8, 
                        linewidth=0, antialiased=True)

# Add contour lines on the bottom
ax.contour(X, Y, W, zdir='z', offset=0, levels=10, cmap=cm.viridis, alpha=0.7)

# Mark the optimum
ax.scatter([0], [0], [1], color=RED, s=100, zorder=5)
ax.text(0.1, 0.1, 1.05, 'Optimum', fontsize=10, color=RED)

ax.set_xlabel('Trait 1')
ax.set_ylabel('Trait 2')
ax.set_zlabel('Fitness')
ax.set_title('Fitness Surface over Trait Space\n(Elliptical contours indicate asymmetric selection)',
             fontsize=12, fontweight='bold')

# Adjust viewing angle
ax.view_init(elev=25, azim=45)

plt.savefig('fig31_fitness_surface.png', dpi=300, facecolor='white')
print("  Saved: fig31_fitness_surface.png")
plt.close()

# =============================================================================
# Figure 2: Curvature Types (1D examples)
# =============================================================================

print("Creating fig31_curvature_types.png...")

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

z = np.linspace(-2, 2, 100)

# Left: Stabilising selection (negative curvature)
ax = axes[0]
w_stab = np.exp(-0.5 * z**2)
ax.plot(z, w_stab, '-', color=BLUE, linewidth=3)
ax.fill_between(z, 0, w_stab, color=LIGHT_BLUE, alpha=0.3)

# Mark the peak
ax.scatter([0], [1], color=RED, s=100, zorder=5)
ax.annotate('', xy=(-0.5, 0.95), xytext=(-1.2, 0.7),
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=2))
ax.annotate('', xy=(0.5, 0.95), xytext=(1.2, 0.7),
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=2))
ax.text(0, 0.5, 'Selection pushes\ntoward optimum', fontsize=9, ha='center',
        color=GREEN)

ax.set_xlim(-2.5, 2.5)
ax.set_ylim(0, 1.15)
ax.set_xlabel('Phenotype')
ax.set_ylabel('Fitness')
ax.set_title('Stabilising Selection\n($\\gamma < 0$, peak)', fontsize=11, fontweight='bold')
ax.axhline(0, color='lightgray', linewidth=0.5)

# Centre: No curvature (directional only)
ax = axes[1]
w_dir = 0.3 + 0.2 * z  # Linear
w_dir = np.clip(w_dir, 0, 1)
ax.plot(z, w_dir, '-', color=BLUE, linewidth=3)
ax.fill_between(z, 0, w_dir, color=LIGHT_BLUE, alpha=0.3)

# Arrow showing direction
ax.annotate('', xy=(1.5, 0.6), xytext=(-0.5, 0.2),
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=2.5))
ax.text(0.5, 0.25, 'Selection pushes\nin one direction', fontsize=9, ha='center',
        color=GREEN)

ax.set_xlim(-2.5, 2.5)
ax.set_ylim(0, 1.15)
ax.set_xlabel('Phenotype')
ax.set_ylabel('Fitness')
ax.set_title('Directional Selection\n($\\gamma = 0$, slope)', fontsize=11, fontweight='bold')
ax.axhline(0, color='lightgray', linewidth=0.5)

# Right: Disruptive selection (positive curvature)
ax = axes[2]
w_disr = 0.3 + 0.3 * z**2
w_disr = np.clip(w_disr, 0, 1)
ax.plot(z, w_disr, '-', color=BLUE, linewidth=3)
ax.fill_between(z, 0, w_disr, color=LIGHT_BLUE, alpha=0.3)

# Mark the valley
ax.scatter([0], [0.3], color=RED, s=100, zorder=5)
ax.annotate('', xy=(-1.5, 0.95), xytext=(-0.3, 0.35),
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=2))
ax.annotate('', xy=(1.5, 0.95), xytext=(0.3, 0.35),
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=2))
ax.text(0, 0.6, 'Selection pushes\naway from centre', fontsize=9, ha='center',
        color=GREEN)

ax.set_xlim(-2.5, 2.5)
ax.set_ylim(0, 1.15)
ax.set_xlabel('Phenotype')
ax.set_ylabel('Fitness')
ax.set_title('Disruptive Selection\n($\\gamma > 0$, valley)', fontsize=11, fontweight='bold')
ax.axhline(0, color='lightgray', linewidth=0.5)

plt.tight_layout()
plt.savefig('fig31_curvature_types.png', dpi=300, facecolor='white')
print("  Saved: fig31_curvature_types.png")
plt.close()

# =============================================================================
# Figure 3: Gaussian Fitness Surface with Contours (2D view)
# =============================================================================

print("Creating fig31_gaussian_surface.png...")

fig, ax = plt.subplots(figsize=(8, 7))

# Gaussian fitness surface
# Optimum at (0.5, 0.3)
theta = np.array([0.5, 0.3])

# Width matrix (asymmetric)
omega = np.array([[1.2, 0.4], [0.4, 0.6]])
omega_inv = np.linalg.inv(omega)

# Eigendecomposition for ellipse orientation
eigenvalues, eigenvectors = np.linalg.eigh(omega)
angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))

# Create contour plot
x = np.linspace(-2, 3, 100)
y = np.linspace(-2, 2.5, 100)
X, Y = np.meshgrid(x, y)

W = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        z = np.array([X[i, j], Y[i, j]]) - theta
        W[i, j] = np.exp(-0.5 * z @ omega_inv @ z)

# Plot contours
levels = [0.1, 0.3, 0.5, 0.7, 0.9]
contour = ax.contour(X, Y, W, levels=levels, colors=[BLUE], linewidths=1.5)
ax.clabel(contour, inline=True, fontsize=8, fmt='%.1f')

# Fill the peak region
contourf = ax.contourf(X, Y, W, levels=[0.8, 1.0], colors=[LIGHT_GREEN], alpha=0.4)

# Mark optimum
ax.scatter([theta[0]], [theta[1]], color=RED, s=150, zorder=5, 
           edgecolor='white', linewidth=2)
ax.text(theta[0] + 0.15, theta[1] + 0.15, r'Optimum $\boldsymbol{\theta}$', 
        fontsize=11, color=RED, fontweight='bold')

# Draw eigenvectors of omega (principal axes of the fitness peak)
for i, (ev, color) in enumerate(zip([eigenvalues[1], eigenvalues[0]], [ORANGE, PURPLE])):
    vec = eigenvectors[:, 1-i] * np.sqrt(ev) * 1.5
    ax.annotate('', xy=(theta[0] + vec[0], theta[1] + vec[1]),
                xytext=theta,
                arrowprops=dict(arrowstyle='->', color=color, lw=2))
    ax.annotate('', xy=(theta[0] - vec[0], theta[1] - vec[1]),
                xytext=theta,
                arrowprops=dict(arrowstyle='->', color=color, lw=2))

# Selection gradient at a point away from optimum
z_point = np.array([-0.8, -0.5])
beta = omega_inv @ (theta - z_point)  # Selection gradient
beta_scale = 0.3  # Scale for visibility

ax.scatter([z_point[0]], [z_point[1]], color=TEAL, s=100, zorder=5,
           marker='s', edgecolor='white', linewidth=2)
ax.annotate('', xy=(z_point[0] + beta[0]*beta_scale, z_point[1] + beta[1]*beta_scale),
            xytext=z_point,
            arrowprops=dict(arrowstyle='->', color=TEAL, lw=2.5))
ax.text(z_point[0] - 0.3, z_point[1] - 0.25, 
        r'$\boldsymbol{\beta}$ (selection' + '\ngradient)', 
        fontsize=10, color=TEAL, ha='right')

ax.axhline(0, color='lightgray', linewidth=0.5, zorder=0)
ax.axvline(0, color='lightgray', linewidth=0.5, zorder=0)

ax.set_xlim(-2, 3)
ax.set_ylim(-2, 2.5)
ax.set_aspect('equal')
ax.set_xlabel('Trait 1')
ax.set_ylabel('Trait 2')
ax.set_title('Gaussian Fitness Surface\nContours show equal fitness; arrows show $\\boldsymbol{\\omega}$ axes',
             fontsize=12, fontweight='bold')

# Explanation box
explanation = (
    "Gaussian fitness function:\n"
    r"$w(\mathbf{z}) = \exp(-\frac{1}{2}(\mathbf{z}-\boldsymbol{\theta})^\top"
    r"\boldsymbol{\omega}^{-1}(\mathbf{z}-\boldsymbol{\theta}))$" + "\n\n"
    r"$\boldsymbol{\gamma} = -\boldsymbol{\omega}^{-1}$"
)
ax.text(0.98, 0.02, explanation, transform=ax.transAxes, fontsize=9, 
        ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor=BLUE, alpha=0.9))

plt.savefig('fig31_gaussian_surface.png', dpi=300, facecolor='white')
print("  Saved: fig31_gaussian_surface.png")
plt.close()

# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*60)
print("Chapter 31 Figures Complete")
print("="*60)
print("""
Generated figures:
  1. fig31_fitness_surface.png
     - 3D fitness surface with peak at origin
     - Contour lines projected on base plane
     - Shows elliptical (asymmetric) fitness peak
     
  2. fig31_curvature_types.png
     - Three panels showing 1D fitness functions
     - Left: Stabilising (peak, γ < 0)
     - Centre: Directional (slope, γ = 0)
     - Right: Disruptive (valley, γ > 0)
     
  3. fig31_gaussian_surface.png
     - 2D contour plot of Gaussian fitness surface
     - Optimum marked, width matrix axes shown
     - Selection gradient β shown at a point
""")
