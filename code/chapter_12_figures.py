#!/usr/bin/env python3
"""
Figures for Chapter 12: Covariance and Mahalanobis Distance
============================================================

Generates:
- fig12_inverse_geometry.png: How the inverse reshapes distance contours
- fig12_worked_example.png: Three points with different Mahalanobis distances

Author: Daniel Ortiz-Barrientos & Claude
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
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
LIGHT_BLUE = '#a8d4e6'
LIGHT_GREEN = '#a8f0d4'

# =============================================================================
# Figure 1: Geometry of the Inverse
# =============================================================================

print("Creating fig12_inverse_geometry.png...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

# Covariance matrix: different variances, no correlation
# Sigma = [[4, 0], [0, 1]]
sigma1_sq = 4  # variance of trait 1
sigma2_sq = 1  # variance of trait 2
sigma1 = np.sqrt(sigma1_sq)
sigma2 = np.sqrt(sigma2_sq)

# Left panel: The covariance ellipse (data distribution)
ax = axes[0]

# Generate data
np.random.seed(42)
n = 200
data = np.random.multivariate_normal([0, 0], [[sigma1_sq, 0], [0, sigma2_sq]], n)

ax.scatter(data[:, 0], data[:, 1], s=20, c=LIGHT_BLUE, alpha=0.5, zorder=1)

# Draw covariance ellipses (1 and 2 SD)
for n_std in [1, 2]:
    ellipse = Ellipse((0, 0), 2*n_std*sigma1, 2*n_std*sigma2, angle=0,
                      fill=False, edgecolor=BLUE, linewidth=2, 
                      linestyle='-' if n_std == 1 else '--')
    ax.add_patch(ellipse)

# Mark the standard deviations on axes
ax.plot([sigma1, sigma1], [-0.3, 0.3], '-', color=ORANGE, linewidth=2)
ax.plot([-0.3, 0.3], [sigma2, sigma2], '-', color=ORANGE, linewidth=2)
ax.text(sigma1 + 0.15, -0.5, r'$\sigma_1 = 2$', fontsize=10, color=ORANGE)
ax.text(0.4, sigma2 + 0.15, r'$\sigma_2 = 1$', fontsize=10, color=ORANGE)

# Origin
ax.scatter([0], [0], s=80, c='black', marker='+', zorder=4, linewidths=2)

ax.set_xlabel('Trait 1')
ax.set_ylabel('Trait 2')
ax.set_xlim(-5, 5)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.axhline(0, color='lightgray', linewidth=0.5, zorder=0)
ax.axvline(0, color='lightgray', linewidth=0.5, zorder=0)
ax.set_title('Data and Covariance Ellipse\n(Trait 1 has more variance)', 
             fontsize=12, fontweight='bold')

# Annotation
ax.text(0.02, 0.98, r'$\Sigma$: var$_1$=4, var$_2$=1, $\rho$=0',
        transform=ax.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor=BLUE, alpha=0.9))

# Right panel: Mahalanobis distance contours
ax = axes[1]

# The Mahalanobis distance contours are ellipses in original space
# but they match the data shape
# d²_Mah = z'Σ⁻¹z = z₁²/4 + z₂²/1 = constant

# Show some points at equal Euclidean distance but different Mahalanobis
ax.scatter(data[:, 0], data[:, 1], s=20, c=LIGHT_BLUE, alpha=0.3, zorder=1)

# Draw Mahalanobis distance contours (d_Mah = 1, 2)
# These are ellipses where z₁²/4 + z₂² = d²
for d_mah in [1, 2]:
    # Semi-axes: a = d*σ₁, b = d*σ₂
    ellipse = Ellipse((0, 0), 2*d_mah*sigma1, 2*d_mah*sigma2, angle=0,
                      fill=False, edgecolor=GREEN, linewidth=2.5,
                      linestyle='-' if d_mah == 1 else '--')
    ax.add_patch(ellipse)
    ax.text(d_mah*sigma1 + 0.2, 0.2, f'$d_{{Mah}}={d_mah}$', 
            fontsize=10, color=GREEN, fontweight='bold')

# Show two points: same Euclidean distance, different Mahalanobis
euc_dist = 2
# Point along trait 1 (high variance direction)
p1 = np.array([euc_dist, 0])
# Point along trait 2 (low variance direction)  
p2 = np.array([0, euc_dist])

ax.scatter([p1[0]], [p1[1]], s=150, c=ORANGE, marker='o', zorder=5,
           edgecolor='white', linewidth=2)
ax.scatter([p2[0]], [p2[1]], s=150, c=RED, marker='o', zorder=5,
           edgecolor='white', linewidth=2)

# Compute Mahalanobis distances
d_mah_p1 = np.sqrt(p1[0]**2/sigma1_sq + p1[1]**2/sigma2_sq)
d_mah_p2 = np.sqrt(p2[0]**2/sigma1_sq + p2[1]**2/sigma2_sq)

ax.text(p1[0] + 0.2, p1[1] + 0.3, 
        f'Eucl: {euc_dist}\nMah: {d_mah_p1:.1f}', 
        fontsize=9, color=ORANGE)
ax.text(p2[0] + 0.3, p2[1], 
        f'Eucl: {euc_dist}\nMah: {d_mah_p2:.1f}', 
        fontsize=9, color=RED)

# Draw Euclidean circle for reference
circle = Circle((0, 0), euc_dist, fill=False, edgecolor=GRAY,
                linewidth=1.5, linestyle=':', alpha=0.7)
ax.add_patch(circle)

# Origin
ax.scatter([0], [0], s=80, c='black', marker='+', zorder=4, linewidths=2)

ax.set_xlabel('Trait 1')
ax.set_ylabel('Trait 2')
ax.set_xlim(-5, 5)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.axhline(0, color='lightgray', linewidth=0.5, zorder=0)
ax.axvline(0, color='lightgray', linewidth=0.5, zorder=0)
ax.set_title('Mahalanobis Distance Contours\n(ellipses match data shape)', 
             fontsize=12, fontweight='bold')

# Annotation
ax.text(0.02, 0.98, r'$\Sigma^{-1}$: 1/4 along trait 1, 1 along trait 2',
        transform=ax.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor=GREEN, alpha=0.9))

# Legend
legend_elements = [
    Line2D([0], [0], color=GREEN, linewidth=2, label='Mahalanobis contours'),
    Line2D([0], [0], color=GRAY, linewidth=1.5, linestyle=':', label='Euclidean circle'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig('fig12_inverse_geometry.png', dpi=300, facecolor='white')
print("  Saved: fig12_inverse_geometry.png")
plt.close()

# =============================================================================
# Figure 2: Worked Example
# =============================================================================

print("Creating fig12_worked_example.png...")

fig, ax = plt.subplots(figsize=(8, 7))

# Covariance matrix with correlation
rho = 0.8
Sigma = np.array([[1, rho], [rho, 1]])
Sigma_inv = np.linalg.inv(Sigma)

# Eigendecomposition for ellipse
eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))

# Generate data for background
np.random.seed(123)
n = 300
data = np.random.multivariate_normal([0, 0], Sigma, n)
ax.scatter(data[:, 0], data[:, 1], s=15, c=LIGHT_BLUE, alpha=0.4, zorder=1)

# Draw covariance ellipse (1 SD)
width = 2 * np.sqrt(eigenvalues[1])
height = 2 * np.sqrt(eigenvalues[0])
ellipse = Ellipse((0, 0), width, height, angle=angle,
                  fill=False, edgecolor=BLUE, linewidth=2, alpha=0.8)
ax.add_patch(ellipse)

# Three points at Euclidean distance 1
sqrt2 = np.sqrt(2)
points = {
    'E': (1/sqrt2, 1/sqrt2, 'Along correlation', ORANGE),
    'F': (1/sqrt2, -1/sqrt2, 'Against correlation', RED),
    'G': (1.0, 0.0, 'Trait 1 only', PURPLE),
}

# Plot points and compute Mahalanobis distances
results = []
for name, (x, y, desc, color) in points.items():
    z = np.array([x, y])
    d_mah_sq = z @ Sigma_inv @ z
    d_mah = np.sqrt(d_mah_sq)
    results.append((name, x, y, 1.0, d_mah, desc))
    
    ax.scatter([x], [y], s=200, c=color, marker='o', zorder=5,
               edgecolor='white', linewidth=2)

# Draw Euclidean circle
circle = Circle((0, 0), 1.0, fill=False, edgecolor=GRAY,
                linewidth=2, linestyle='--', alpha=0.7)
ax.add_patch(circle)

# Labels for points
ax.text(0.707 + 0.15, 0.707 + 0.15, f'E\n$d_{{Mah}}$ = {results[0][4]:.2f}', 
        fontsize=10, color=ORANGE, fontweight='bold')
ax.text(0.707 + 0.15, -0.707 - 0.1, f'F\n$d_{{Mah}}$ = {results[1][4]:.2f}', 
        fontsize=10, color=RED, fontweight='bold', va='top')
ax.text(1.0 + 0.15, 0.15, f'G\n$d_{{Mah}}$ = {results[2][4]:.2f}', 
        fontsize=10, color=PURPLE, fontweight='bold')

# Draw lines from origin to each point
for name, (x, y, desc, color) in points.items():
    ax.plot([0, x], [0, y], '--', color=color, linewidth=1.5, alpha=0.6)

# Origin
ax.scatter([0], [0], s=100, c='black', marker='+', zorder=4, linewidths=2)
ax.text(0.1, -0.15, 'Mean', fontsize=10)

ax.set_xlabel('Trait 1')
ax.set_ylabel('Trait 2')
ax.set_xlim(-2, 2.5)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.axhline(0, color='lightgray', linewidth=0.5, zorder=0)
ax.axvline(0, color='lightgray', linewidth=0.5, zorder=0)

# Title
ax.set_title('Three Points at Euclidean Distance 1\nwith Different Mahalanobis Distances',
             fontsize=12, fontweight='bold')

# Table in corner
table_text = (
    "Point  Direction           Eucl  Mah\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    f"  E    Along correlation   1.0   {results[0][4]:.2f}\n"
    f"  F    Against correlation 1.0   {results[1][4]:.2f}\n"
    f"  G    Trait 1 only        1.0   {results[2][4]:.2f}"
)
ax.text(0.02, 0.02, table_text, transform=ax.transAxes, fontsize=9,
        fontfamily='monospace', va='bottom',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95))

# Covariance matrix annotation
ax.text(0.98, 0.98, r'$\rho = 0.8$', transform=ax.transAxes, fontsize=11, 
        ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor=BLUE, alpha=0.9))

# Legend
legend_elements = [
    Line2D([0], [0], color=GRAY, linewidth=2, linestyle='--', 
           label='Euclidean circle ($d_{Euc}=1$)'),
    Line2D([0], [0], color=BLUE, linewidth=2, label='Covariance ellipse'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

plt.savefig('fig12_worked_example.png', dpi=300, facecolor='white')
print("  Saved: fig12_worked_example.png")
plt.close()

# =============================================================================
# Figure 3: Whitening Transformation (Bonus)
# =============================================================================

print("Creating fig12_whitening.png...")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Covariance matrix
rho = 0.7
Sigma = np.array([[1.5, rho*np.sqrt(1.5*0.8)], [rho*np.sqrt(1.5*0.8), 0.8]])
eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))

# Whitening matrix (Sigma^{-1/2})
Sigma_inv_sqrt = eigenvectors @ np.diag(1/np.sqrt(eigenvalues)) @ eigenvectors.T

# Generate data
np.random.seed(456)
n = 200
data = np.random.multivariate_normal([0, 0], Sigma, n)

# Whitened data
data_white = (Sigma_inv_sqrt @ data.T).T

# Panel 1: Original data
ax = axes[0]
ax.scatter(data[:, 0], data[:, 1], s=20, c=BLUE, alpha=0.5)
ellipse = Ellipse((0, 0), 2*np.sqrt(eigenvalues[1]), 2*np.sqrt(eigenvalues[0]), 
                  angle=angle, fill=False, edgecolor=ORANGE, linewidth=2)
ax.add_patch(ellipse)
ax.scatter([0], [0], s=80, c='black', marker='+', zorder=4, linewidths=2)
ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.axhline(0, color='lightgray', linewidth=0.5, zorder=0)
ax.axvline(0, color='lightgray', linewidth=0.5, zorder=0)
ax.set_xlabel('Trait 1')
ax.set_ylabel('Trait 2')
ax.set_title('Original Space\n(correlated, unequal variances)', fontsize=11, fontweight='bold')

# Panel 2: The transformation
ax = axes[1]
ax.axis('off')

transform_text = (
    "THE WHITENING\nTRANSFORMATION\n\n"
    "w = S^(-1/2) (z - mean)\n\n"
    "This transformation:\n"
    "1. Centres at origin\n"
    "2. Rotates to principal axes\n"
    "3. Rescales each axis\n\n"
    "Result: spherical data\n"
    "with identity covariance"
)
ax.text(0.5, 0.5, transform_text, transform=ax.transAxes, fontsize=11,
        ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor=ORANGE,
                  linewidth=2, alpha=0.95))

# Arrow
ax.annotate('', xy=(0.95, 0.5), xytext=(0.05, 0.5),
            arrowprops=dict(arrowstyle='->', color=PURPLE, lw=3))

ax.set_title('Transformation', fontsize=11, fontweight='bold')

# Panel 3: Whitened data
ax = axes[2]
ax.scatter(data_white[:, 0], data_white[:, 1], s=20, c=GREEN, alpha=0.5)
circle = Circle((0, 0), 1, fill=False, edgecolor=ORANGE, linewidth=2)
ax.add_patch(circle)
ax.scatter([0], [0], s=80, c='black', marker='+', zorder=4, linewidths=2)
ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.axhline(0, color='lightgray', linewidth=0.5, zorder=0)
ax.axvline(0, color='lightgray', linewidth=0.5, zorder=0)
ax.set_xlabel('Whitened axis 1')
ax.set_ylabel('Whitened axis 2')
ax.set_title('Whitened Space\n(spherical, identity covariance)', fontsize=11, fontweight='bold')

# Add note
ax.text(0.98, 0.02, 'Euclidean distance\nhere = Mahalanobis\ndistance in original', 
        transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()
plt.savefig('fig12_whitening.png', dpi=300, facecolor='white')
print("  Saved: fig12_whitening.png")
plt.close()

# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*60)
print("Chapter 12 Figures Complete")
print("="*60)
print("""
Generated figures:
  1. fig12_inverse_geometry.png
     - Left: Covariance ellipse showing unequal variances
     - Right: Mahalanobis contours (ellipses) vs Euclidean circle
     - Two points: same Euclidean distance, different Mahalanobis
     
  2. fig12_worked_example.png
     - Three points E, F, G at Euclidean distance 1
     - E (along correlation): Mah = 0.75
     - F (against correlation): Mah = 2.24
     - G (trait 1 only): Mah = 1.67
     - Table showing the comparison
     
  3. fig12_whitening.png
     - Original space → whitening transformation → whitened space
     - Shows how whitening makes ellipse circular
     - Euclidean in whitened = Mahalanobis in original
""")
