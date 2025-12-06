#!/usr/bin/env python3
"""
Figures for Chapter 20: Diagonalisation and Natural Axes
==========================================================

Generates:
- fig20_eigenvectors.png: Matrix action on eigenvectors vs other vectors
- fig20_ellipse_axes.png: Covariance ellipse with eigenvector axes
- fig20_gmax.png: G matrix with g_max direction highlighted

Author: Daniel Ortiz-Barrientos
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

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
LIGHT_BLUE = '#a8d4e6'
LIGHT_GREEN = '#a8f0d4'
TEAL = '#20B2AA'

# =============================================================================
# Figure 1: Eigenvectors - Matrix acts as pure scaling
# =============================================================================

print("Creating fig20_eigenvectors.png...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

# Matrix from the chapter example
A = np.array([[3, 1], [1, 3]])
eigenvalues, eigenvectors = np.linalg.eigh(A)

# Sort by decreasing eigenvalue
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

v1 = eigenvectors[:, 0]  # eigenvalue 4
v2 = eigenvectors[:, 1]  # eigenvalue 2

# Left panel: Show eigenvectors - matrix stretches without rotating
ax = axes[0]

# Draw coordinate axes
ax.axhline(0, color='lightgray', linewidth=0.5, zorder=0)
ax.axvline(0, color='lightgray', linewidth=0.5, zorder=0)

# Draw unit circle for reference
theta = np.linspace(0, 2*np.pi, 100)
ax.plot(np.cos(theta), np.sin(theta), ':', color=GRAY, linewidth=1, alpha=0.5)

# Draw eigenvector v1 and its image
scale = 0.8
ax.annotate('', xy=v1*scale, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color=ORANGE, lw=2.5))
ax.annotate('', xy=A @ (v1*scale), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color=ORANGE, lw=2.5, linestyle='--'))

# Labels for v1
ax.text(v1[0]*scale + 0.15, v1[1]*scale + 0.15, r'$\mathbf{v}_1$', 
        fontsize=12, color=ORANGE, fontweight='bold')
ax.text((A @ (v1*scale))[0] + 0.15, (A @ (v1*scale))[1] + 0.15, 
        r'$A\mathbf{v}_1 = 4\mathbf{v}_1$', 
        fontsize=11, color=ORANGE)

# Draw eigenvector v2 and its image
ax.annotate('', xy=v2*scale, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color=BLUE, lw=2.5))
ax.annotate('', xy=A @ (v2*scale), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color=BLUE, lw=2.5, linestyle='--'))

# Labels for v2
ax.text(v2[0]*scale - 0.4, v2[1]*scale + 0.1, r'$\mathbf{v}_2$', 
        fontsize=12, color=BLUE, fontweight='bold')
ax.text((A @ (v2*scale))[0] - 0.5, (A @ (v2*scale))[1] - 0.3, 
        r'$A\mathbf{v}_2 = 2\mathbf{v}_2$', 
        fontsize=11, color=BLUE)

ax.set_xlim(-4, 4)
ax.set_ylim(-3, 3.5)
ax.set_aspect('equal')
ax.set_xlabel('Trait 1')
ax.set_ylabel('Trait 2')
ax.set_title('Eigenvectors: Matrix stretches\nwithout changing direction', 
             fontsize=12, fontweight='bold')

# Legend
legend_elements = [
    Line2D([0], [0], color=ORANGE, linewidth=2.5, label=r'$\mathbf{v}_1$ (eigenvalue 4)'),
    Line2D([0], [0], color=BLUE, linewidth=2.5, label=r'$\mathbf{v}_2$ (eigenvalue 2)'),
    Line2D([0], [0], color=GRAY, linewidth=2, linestyle='--', label='After transformation'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

# Right panel: Non-eigenvector gets rotated
ax = axes[1]

ax.axhline(0, color='lightgray', linewidth=0.5, zorder=0)
ax.axvline(0, color='lightgray', linewidth=0.5, zorder=0)

# A non-eigenvector
w = np.array([1, 0])  # along trait 1 axis
w_norm = w / np.linalg.norm(w) * scale

# Draw w and its image
ax.annotate('', xy=w_norm, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color=PURPLE, lw=2.5))
Aw = A @ w_norm
ax.annotate('', xy=Aw, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color=PURPLE, lw=2.5, linestyle='--'))

# Draw arc to show rotation
from matplotlib.patches import Arc
angle_w = np.degrees(np.arctan2(w_norm[1], w_norm[0]))
angle_Aw = np.degrees(np.arctan2(Aw[1], Aw[0]))
arc = Arc((0, 0), 1.2, 1.2, angle=0, theta1=angle_w, theta2=angle_Aw,
          color=RED, linewidth=1.5, linestyle='-')
ax.add_patch(arc)
ax.text(0.7, 0.3, 'rotated!', fontsize=10, color=RED, style='italic')

# Labels
ax.text(w_norm[0] + 0.1, w_norm[1] - 0.2, r'$\mathbf{w}$', 
        fontsize=12, color=PURPLE, fontweight='bold')
ax.text(Aw[0] + 0.1, Aw[1] + 0.1, r'$A\mathbf{w}$', 
        fontsize=11, color=PURPLE)

# Also show eigenvectors faintly for reference
ax.annotate('', xy=v1*1.5, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1, alpha=0.3))
ax.annotate('', xy=v2*1.5, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color=BLUE, lw=1, alpha=0.3))

ax.set_xlim(-4, 4)
ax.set_ylim(-3, 3.5)
ax.set_aspect('equal')
ax.set_xlabel('Trait 1')
ax.set_ylabel('Trait 2')
ax.set_title('Non-eigenvector: Matrix\nboth stretches AND rotates', 
             fontsize=12, fontweight='bold')

# Annotation
ax.text(0.02, 0.02, 
        'Eigenvectors (faint) are the\nonly directions that stay fixed', 
        transform=ax.transAxes, fontsize=9, va='bottom',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig('fig20_eigenvectors.png', dpi=300, facecolor='white')
print("  Saved: fig20_eigenvectors.png")
plt.close()

# =============================================================================
# Figure 2: Covariance Ellipse with Eigenvector Axes
# =============================================================================

print("Creating fig20_ellipse_axes.png...")

fig, ax = plt.subplots(figsize=(8, 7))

# Covariance matrix with correlation
rho = 0.8
Sigma = np.array([[1, rho], [rho, 1]])
eigenvalues, eigenvectors = np.linalg.eigh(Sigma)

# Sort by decreasing eigenvalue
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

# Generate data
np.random.seed(42)
n = 300
data = np.random.multivariate_normal([0, 0], Sigma, n)

ax.scatter(data[:, 0], data[:, 1], s=15, c=LIGHT_BLUE, alpha=0.4, zorder=1)

# Draw covariance ellipse (1 SD and 2 SD)
for n_std, alpha in [(1, 0.3), (2, 0.15)]:
    width = 2 * n_std * np.sqrt(eigenvalues[0])
    height = 2 * n_std * np.sqrt(eigenvalues[1])
    ellipse = Ellipse((0, 0), width, height, angle=angle,
                      fill=True, facecolor=GREEN, alpha=alpha,
                      edgecolor=GREEN, linewidth=2)
    ax.add_patch(ellipse)

# Draw eigenvector axes through the ellipse
for i, (ev, lam, color, label) in enumerate([
    (eigenvectors[:, 0], eigenvalues[0], ORANGE, r'$\mathbf{v}_1$'),
    (eigenvectors[:, 1], eigenvalues[1], BLUE, r'$\mathbf{v}_2$')
]):
    # Draw axis line
    scale = 2.5
    ax.plot([-ev[0]*scale, ev[0]*scale], [-ev[1]*scale, ev[1]*scale],
            '-', color=color, linewidth=2.5, zorder=3)
    
    # Arrow at end
    ax.annotate('', xy=ev*2.2, xytext=ev*1.8,
                arrowprops=dict(arrowstyle='->', color=color, lw=2))
    
    # Label
    ax.text(ev[0]*2.4, ev[1]*2.4, f'{label}\n$\\lambda_{i+1}={lam:.2f}$', 
            fontsize=10, color=color, ha='center', fontweight='bold')

# Mark the semi-axis lengths
ax.plot([0, eigenvectors[0, 0]*np.sqrt(eigenvalues[0])],
        [0, eigenvectors[1, 0]*np.sqrt(eigenvalues[0])],
        '-', color=RED, linewidth=3, zorder=4)
ax.text(0.6, 0.8, r'$\sqrt{\lambda_1}$', fontsize=11, color=RED, fontweight='bold')

ax.plot([0, eigenvectors[0, 1]*np.sqrt(eigenvalues[1])],
        [0, eigenvectors[1, 1]*np.sqrt(eigenvalues[1])],
        '-', color=PURPLE, linewidth=3, zorder=4)
ax.text(-0.5, 0.2, r'$\sqrt{\lambda_2}$', fontsize=11, color=PURPLE, fontweight='bold')

# Origin
ax.scatter([0], [0], s=80, c='black', marker='+', zorder=5, linewidths=2)

ax.axhline(0, color='lightgray', linewidth=0.5, zorder=0)
ax.axvline(0, color='lightgray', linewidth=0.5, zorder=0)

ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.set_xlabel('Trait 1')
ax.set_ylabel('Trait 2')
ax.set_title('Covariance Ellipse: Eigenvectors are axes,\neigenvalues are variances along axes', 
             fontsize=12, fontweight='bold')

# Summary box
summary = (
    f"Eigenvalues:\n"
    f"  $\\lambda_1$ = {eigenvalues[0]:.2f} (major axis)\n"
    f"  $\\lambda_2$ = {eigenvalues[1]:.2f} (minor axis)\n\n"
    f"Ratio: {eigenvalues[0]/eigenvalues[1]:.1f}:1"
)
ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor=GREEN, alpha=0.9))

plt.savefig('fig20_ellipse_axes.png', dpi=300, facecolor='white')
print("  Saved: fig20_ellipse_axes.png")
plt.close()

# =============================================================================
# Figure 3: G matrix and g_max
# =============================================================================

print("Creating fig20_gmax.png...")

fig, ax = plt.subplots(figsize=(8, 7))

# A G matrix with strong constraint (highly eccentric)
G = np.array([[1.5, 1.2], [1.2, 1.0]])
eigenvalues_G, eigenvectors_G = np.linalg.eigh(G)

# Sort by decreasing eigenvalue
idx = np.argsort(eigenvalues_G)[::-1]
eigenvalues_G = eigenvalues_G[idx]
eigenvectors_G = eigenvectors_G[:, idx]

gmax = eigenvectors_G[:, 0]
gmin = eigenvectors_G[:, 1]

angle_G = np.degrees(np.arctan2(gmax[1], gmax[0]))

# Generate "genetic" data
np.random.seed(123)
n = 250
data_G = np.random.multivariate_normal([0, 0], G, n)

ax.scatter(data_G[:, 0], data_G[:, 1], s=20, c=LIGHT_GREEN, alpha=0.5, zorder=1,
           label='Genetic variation')

# Draw G ellipse
for n_std in [1, 2]:
    width = 2 * n_std * np.sqrt(eigenvalues_G[0])
    height = 2 * n_std * np.sqrt(eigenvalues_G[1])
    ellipse = Ellipse((0, 0), width, height, angle=angle_G,
                      fill=False, edgecolor=TEAL, linewidth=2.5)
    ax.add_patch(ellipse)

# Draw g_max prominently
scale = 3
ax.annotate('', xy=gmax*scale, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color=ORANGE, lw=4))
ax.annotate('', xy=-gmax*scale, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color=ORANGE, lw=4))

ax.text(gmax[0]*scale + 0.2, gmax[1]*scale + 0.2, 
        r'$\mathbf{g}_{max}$' + '\n"Line of least\nresistance"', 
        fontsize=11, color=ORANGE, fontweight='bold')

# Draw g_min (perpendicular)
ax.annotate('', xy=gmin*scale*0.5, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color=PURPLE, lw=2, alpha=0.7))
ax.annotate('', xy=-gmin*scale*0.5, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color=PURPLE, lw=2, alpha=0.7))

ax.text(gmin[0]*1.2 - 0.3, gmin[1]*1.2, 
        r'$\mathbf{g}_{min}$' + '\n"Constrained\ndirection"', 
        fontsize=10, color=PURPLE, ha='right')

# Show a selection gradient and the response
beta = np.array([0.3, 1.0])  # Selection mostly on trait 2
beta = beta / np.linalg.norm(beta)  # Normalize
response = G @ beta * 1.5  # Scale for visibility

# Selection gradient
ax.annotate('', xy=beta*2, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color=RED, lw=2.5, linestyle='--'))
ax.text(beta[0]*2 + 0.15, beta[1]*2 + 0.15, 
        r'$\boldsymbol{\beta}$ (selection)', 
        fontsize=10, color=RED)

# Response
ax.annotate('', xy=response, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color=BLUE, lw=2.5))
ax.text(response[0] + 0.15, response[1] - 0.3, 
        r'$G\boldsymbol{\beta}$ (response)', 
        fontsize=10, color=BLUE)

# Origin
ax.scatter([0], [0], s=100, c='black', marker='+', zorder=5, linewidths=2)

ax.axhline(0, color='lightgray', linewidth=0.5, zorder=0)
ax.axvline(0, color='lightgray', linewidth=0.5, zorder=0)

ax.set_xlim(-4, 4)
ax.set_ylim(-3.5, 3.5)
ax.set_aspect('equal')
ax.set_xlabel('Trait 1')
ax.set_ylabel('Trait 2')
ax.set_title('The G Matrix: Genetic variance channels evolutionary response', 
             fontsize=12, fontweight='bold')

# Summary box
summary = (
    f"Eigenvalues of G:\n"
    f"  $\\lambda_1$ = {eigenvalues_G[0]:.2f}\n"
    f"  $\\lambda_2$ = {eigenvalues_G[1]:.2f}\n\n"
    f"Ratio: {eigenvalues_G[0]/eigenvalues_G[1]:.0f}:1\n"
    f"(strong constraint)"
)
ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor=TEAL, alpha=0.9))

# Annotation about deflection
ax.text(0.98, 0.02, 
        'Selection (red) aims one way,\n'
        'but response (blue) is deflected\n'
        r'toward $\mathbf{g}_{max}$', 
        transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.savefig('fig20_gmax.png', dpi=300, facecolor='white')
print("  Saved: fig20_gmax.png")
plt.close()

# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*60)
print("Chapter 20 Figures Complete")
print("="*60)
print("""
Generated figures:
  1. fig20_eigenvectors.png
     - Left: Eigenvectors stretched without rotation
     - Right: Non-eigenvector gets both stretched AND rotated
     
  2. fig20_ellipse_axes.png
     - Covariance ellipse with eigenvector axes marked
     - Eigenvalues shown as semi-axis lengths squared
     - Shows λ₁ = 1.8, λ₂ = 0.2 for ρ = 0.8 correlation
     
  3. fig20_gmax.png
     - G matrix with g_max (line of least resistance)
     - Shows selection gradient β and deflected response Gβ
     - Demonstrates how G channels evolutionary response
""")
