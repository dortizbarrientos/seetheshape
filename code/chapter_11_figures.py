#!/usr/bin/env python3
"""
Figures for Chapter 11: When Euclidean Distance Fails
======================================================

Generates:
- fig11_scale_problem.png: Beetles A and B showing scale dependence
- fig11_correlation_problem.png: Beetles C and D with correlation structure
- fig11_probability_contours.png: Probability ellipses vs Euclidean circle
- fig11_geometric_preview.png: Side-by-side circles vs ellipses

Author: Daniel Ortiz-Barrientos & Claude
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
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

# =============================================================================
# Figure 1: The Scale Problem
# =============================================================================

print("Creating fig11_scale_problem.png...")

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# Left panel: Original units
ax = axes[0]

# Reference point (mean)
ref_elytra = 12  # mm
ref_mass = 450   # mg

# Beetle A: +2mm in elytra
a_elytra = 14
a_mass = 450

# Beetle B: +80mg in mass
b_elytra = 12
b_mass = 530

# Plot in original units (but we'll scale for visualization)
# Scale mass to be comparable for plotting
mass_scale = 0.025  # 1 mg = 0.025 units for plotting

ax.scatter([ref_elytra], [ref_mass * mass_scale], s=150, c=GRAY, 
           marker='s', zorder=3, label='Reference (mean)')
ax.scatter([a_elytra], [a_mass * mass_scale], s=150, c=ORANGE, 
           marker='o', zorder=3, label='Beetle A')
ax.scatter([b_elytra], [b_mass * mass_scale], s=150, c=BLUE, 
           marker='o', zorder=3, label='Beetle B')

# Draw distance lines
ax.plot([ref_elytra, a_elytra], [ref_mass * mass_scale, a_mass * mass_scale], 
        '--', color=ORANGE, linewidth=2, alpha=0.7)
ax.plot([ref_elytra, b_elytra], [ref_mass * mass_scale, b_mass * mass_scale], 
        '--', color=BLUE, linewidth=2, alpha=0.7)

# Labels
ax.text(a_elytra + 0.3, a_mass * mass_scale, 'A\n+2 mm', fontsize=10, color=ORANGE)
ax.text(b_elytra + 0.3, b_mass * mass_scale, 'B\n+80 mg', fontsize=10, color=BLUE)
ax.text(ref_elytra - 0.3, ref_mass * mass_scale, 'Mean', fontsize=10, 
        ha='right', color=GRAY)

ax.set_xlabel('Elytra length (mm)')
ax.set_ylabel('Body mass (scaled)')
ax.set_title('Original measurements', fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)

# Add annotation about Euclidean distances
ax.text(0.05, 0.95, 'Euclidean distances:\nA: 2 units\nB: 80 units', 
        transform=ax.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

# Right panel: Standardised
ax = axes[1]

# Standardised values (both are 1 SD from mean)
# A: (14-12)/2 = 1 SD in elytra, 0 in mass
# B: 0 in elytra, (530-450)/80 = 1 SD in mass

ax.scatter([0], [0], s=150, c=GRAY, marker='s', zorder=3, label='Mean (origin)')
ax.scatter([1], [0], s=150, c=ORANGE, marker='o', zorder=3, label='Beetle A')
ax.scatter([0], [1], s=150, c=BLUE, marker='o', zorder=3, label='Beetle B')

# Draw distance lines
ax.plot([0, 1], [0, 0], '--', color=ORANGE, linewidth=2, alpha=0.7)
ax.plot([0, 0], [0, 1], '--', color=BLUE, linewidth=2, alpha=0.7)

# Unit circle to show equal distance
theta = np.linspace(0, 2*np.pi, 100)
ax.plot(np.cos(theta), np.sin(theta), ':', color=GRAY, linewidth=1.5, alpha=0.5)

# Labels  
ax.text(1.15, 0, 'A\n(1 SD)', fontsize=10, color=ORANGE, va='center')
ax.text(0.1, 1.1, 'B\n(1 SD)', fontsize=10, color=BLUE)

ax.set_xlabel('Elytra length (standardised)')
ax.set_ylabel('Body mass (standardised)')
ax.set_title('Standardised (SD = 1)', fontsize=12, fontweight='bold')
ax.set_xlim(-1.5, 1.8)
ax.set_ylim(-1.5, 1.8)
ax.set_aspect('equal')
ax.legend(loc='lower right', fontsize=9)
ax.axhline(0, color='lightgray', linewidth=0.5, zorder=0)
ax.axvline(0, color='lightgray', linewidth=0.5, zorder=0)

# Add annotation
ax.text(0.05, 0.95, 'After standardising:\nBoth are 1 SD from mean\nEqually unusual!', 
        transform=ax.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round', facecolor=LIGHT_GREEN, alpha=0.9))

plt.tight_layout()
plt.savefig('fig11_scale_problem.png', dpi=300, facecolor='white')
print("  Saved: fig11_scale_problem.png")
plt.close()

# =============================================================================
# Figure 2: The Correlation Problem
# =============================================================================

print("Creating fig11_correlation_problem.png...")

fig, ax = plt.subplots(figsize=(8, 7))

# Generate correlated data
np.random.seed(42)
n = 200
mean = [0, 0]
rho = 0.8
cov = [[1, rho], [rho, 1]]
data = np.random.multivariate_normal(mean, cov, n)

# Plot data cloud
ax.scatter(data[:, 0], data[:, 1], s=20, c=LIGHT_BLUE, alpha=0.6, zorder=1)

# Compute and plot covariance ellipse
eigenvalues, eigenvectors = np.linalg.eigh(cov)
angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))

for n_std, alpha in [(1, 0.3), (2, 0.15)]:
    width = 2 * n_std * np.sqrt(eigenvalues[1])
    height = 2 * n_std * np.sqrt(eigenvalues[0])
    ellipse = Ellipse((0, 0), width, height, angle=angle,
                      fill=False, edgecolor=GREEN, linewidth=2, alpha=0.8)
    ax.add_patch(ellipse)

# Point C: along correlation (both positive)
# Point D: against correlation (one positive, one negative)
# Both at Euclidean distance 1.5 from origin

dist = 1.5
# Along major axis (45 degrees, roughly)
c_x = dist * np.cos(np.radians(45))
c_y = dist * np.sin(np.radians(45))

# Along minor axis (135 degrees, roughly) 
d_x = dist * np.cos(np.radians(135))
d_y = dist * np.sin(np.radians(135))

# Plot points C and D
ax.scatter([c_x], [c_y], s=200, c=ORANGE, marker='o', zorder=5, 
           edgecolor='white', linewidth=2)
ax.scatter([d_x], [d_y], s=200, c=RED, marker='o', zorder=5,
           edgecolor='white', linewidth=2)

# Labels
ax.text(c_x + 0.2, c_y + 0.2, 'C\n"Big beetle"\n(common)', fontsize=10, 
        color=ORANGE, fontweight='bold')
ax.text(d_x - 0.2, d_y + 0.2, 'D\n"Odd proportions"\n(rare)', fontsize=10, 
        color=RED, ha='right', fontweight='bold')

# Draw Euclidean circle through C and D
circle = Circle((0, 0), dist, fill=False, edgecolor=GRAY, 
                linewidth=2, linestyle='--', zorder=2)
ax.add_patch(circle)

# Draw lines from origin to C and D
ax.plot([0, c_x], [0, c_y], '--', color=ORANGE, linewidth=1.5, alpha=0.7)
ax.plot([0, d_x], [0, d_y], '--', color=RED, linewidth=1.5, alpha=0.7)

# Origin
ax.scatter([0], [0], s=100, c='black', marker='+', zorder=4, linewidths=2)
ax.text(0.15, -0.2, 'Mean', fontsize=10)

# Axes
ax.axhline(0, color='lightgray', linewidth=0.5, zorder=0)
ax.axvline(0, color='lightgray', linewidth=0.5, zorder=0)

ax.set_xlabel('Trait 1 (standardised)')
ax.set_ylabel('Trait 2 (standardised)')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')

# Legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=LIGHT_BLUE, 
           markersize=8, label='Population'),
    Line2D([0], [0], color=GREEN, linewidth=2, label='Covariance ellipse'),
    Line2D([0], [0], color=GRAY, linewidth=2, linestyle='--', 
           label='Euclidean circle'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

# Annotation
ax.text(0.02, 0.98, 'Same Euclidean distance\nfrom mean, but...\n\n'
        'C is common (along correlation)\n'
        'D is rare (against correlation)', 
        transform=ax.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.savefig('fig11_correlation_problem.png', dpi=300, facecolor='white')
print("  Saved: fig11_correlation_problem.png")
plt.close()

# =============================================================================
# Figure 3: Probability Contours
# =============================================================================

print("Creating fig11_probability_contours.png...")

fig, ax = plt.subplots(figsize=(8, 7))

# Covariance structure
rho = 0.8
cov = np.array([[1, rho], [rho, 1]])
cov_inv = np.linalg.inv(cov)

# Eigendecomposition for ellipse
eigenvalues, eigenvectors = np.linalg.eigh(cov)
angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))

# Draw probability contours (ellipses)
# For bivariate normal, contours of equal density are ellipses where
# (z - mu)' Sigma^-1 (z - mu) = c
# The values c = 2.30, 6.18, 11.83 correspond to 68%, 95%, 99% probability

contour_levels = [
    (2.30, '68%', 0.4),
    (6.18, '95%', 0.25),
    (11.83, '99%', 0.1),
]

for c, label, alpha in contour_levels:
    # For (z' Sigma^-1 z) = c, the semi-axes are sqrt(c * lambda_i)
    width = 2 * np.sqrt(c * eigenvalues[1])
    height = 2 * np.sqrt(c * eigenvalues[0])
    ellipse = Ellipse((0, 0), width, height, angle=angle,
                      fill=True, facecolor=GREEN, alpha=alpha,
                      edgecolor=GREEN, linewidth=2)
    ax.add_patch(ellipse)
    # Label the contour
    label_x = np.sqrt(c * eigenvalues[1]) * 0.7
    label_y = np.sqrt(c * eigenvalues[1]) * 0.7
    ax.text(label_x + 0.3, label_y + 0.3, label, fontsize=9, color=GREEN,
            fontweight='bold')

# Draw Euclidean circle
circle_radius = 2.0
circle = Circle((0, 0), circle_radius, fill=False, edgecolor=GRAY,
                linewidth=2.5, linestyle='--', zorder=3)
ax.add_patch(circle)

# Mark specific points on the circle to show density difference
angles_deg = [45, 135]  # Along major and minor axes
for ang, color, label, offset in [(45, ORANGE, 'Common\n(high density)', (0.3, 0.3)),
                                   (135, RED, 'Rare\n(low density)', (-0.3, 0.3))]:
    x = circle_radius * np.cos(np.radians(ang))
    y = circle_radius * np.sin(np.radians(ang))
    ax.scatter([x], [y], s=150, c=color, zorder=5, edgecolor='white', linewidth=2)
    ax.text(x + offset[0], y + offset[1], label, fontsize=9, color=color,
            ha='center', fontweight='bold')

# Origin
ax.scatter([0], [0], s=80, c='black', marker='+', zorder=4, linewidths=2)

# Axes
ax.axhline(0, color='lightgray', linewidth=0.5, zorder=0)
ax.axvline(0, color='lightgray', linewidth=0.5, zorder=0)

ax.set_xlabel('Trait 1 (standardised)')
ax.set_ylabel('Trait 2 (standardised)')
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_aspect('equal')

# Annotation
ax.text(0.02, 0.98, 
        'Ellipses = equal probability density\n'
        'Dashed circle = equal Euclidean distance\n\n'
        'Points on the same circle have\n'
        'very different probabilities!',
        transform=ax.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.savefig('fig11_probability_contours.png', dpi=300, facecolor='white')
print("  Saved: fig11_probability_contours.png")
plt.close()

# =============================================================================
# Figure 4: Geometric Preview (Circles vs Ellipses)
# =============================================================================

print("Creating fig11_geometric_preview.png...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

# Generate same correlated data for both panels
np.random.seed(123)
n = 150
rho = 0.75
cov = [[1, rho], [rho, 1]]
data = np.random.multivariate_normal([0, 0], cov, n)

eigenvalues, eigenvectors = np.linalg.eigh(cov)
angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))

# Left panel: Euclidean distance (circles)
ax = axes[0]
ax.scatter(data[:, 0], data[:, 1], s=25, c=LIGHT_BLUE, alpha=0.5, zorder=1)

# Draw concentric circles
for r in [0.5, 1.0, 1.5, 2.0]:
    circle = Circle((0, 0), r, fill=False, edgecolor=GRAY, 
                    linewidth=1.5, linestyle='-', alpha=0.7)
    ax.add_patch(circle)

ax.scatter([0], [0], s=100, c='black', marker='+', zorder=4, linewidths=2)

ax.set_xlabel('Trait 1')
ax.set_ylabel('Trait 2')
ax.set_title('Euclidean Distance\n(circles ignore data shape)', fontsize=12, fontweight='bold')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.axhline(0, color='lightgray', linewidth=0.5, zorder=0)
ax.axvline(0, color='lightgray', linewidth=0.5, zorder=0)

# Highlight problematic points
ax.scatter([1.2], [1.2], s=120, c=ORANGE, zorder=5, edgecolor='white', linewidth=2)
ax.scatter([-1.2], [1.2], s=120, c=RED, zorder=5, edgecolor='white', linewidth=2)
ax.text(1.4, 1.4, 'Same\ndistance', fontsize=9, color=GRAY)

# Right panel: Mahalanobis distance (ellipses)
ax = axes[1]
ax.scatter(data[:, 0], data[:, 1], s=25, c=LIGHT_BLUE, alpha=0.5, zorder=1)

# Draw concentric ellipses matching covariance
for scale in [0.5, 1.0, 1.5, 2.0]:
    width = 2 * scale * np.sqrt(eigenvalues[1])
    height = 2 * scale * np.sqrt(eigenvalues[0])
    ellipse = Ellipse((0, 0), width, height, angle=angle,
                      fill=False, edgecolor=GREEN, linewidth=1.5, alpha=0.7)
    ax.add_patch(ellipse)

ax.scatter([0], [0], s=100, c='black', marker='+', zorder=4, linewidths=2)

ax.set_xlabel('Trait 1')
ax.set_ylabel('Trait 2')
ax.set_title('Mahalanobis Distance\n(ellipses match data shape)', fontsize=12, fontweight='bold')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.axhline(0, color='lightgray', linewidth=0.5, zorder=0)
ax.axvline(0, color='lightgray', linewidth=0.5, zorder=0)

# Highlight the same points - now at different distances
ax.scatter([1.2], [1.2], s=120, c=ORANGE, zorder=5, edgecolor='white', linewidth=2)
ax.scatter([-1.2], [1.2], s=120, c=RED, zorder=5, edgecolor='white', linewidth=2)
ax.text(1.4, 1.0, 'Closer\n(common)', fontsize=9, color=ORANGE)
ax.text(-1.0, 1.5, 'Farther\n(rare)', fontsize=9, color=RED, ha='right')

plt.tight_layout()
plt.savefig('fig11_geometric_preview.png', dpi=300, facecolor='white')
print("  Saved: fig11_geometric_preview.png")
plt.close()

# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*60)
print("Chapter 11 Figures Complete")
print("="*60)
print("""
Generated figures:
  1. fig11_scale_problem.png
     - Beetles A and B: same standardised deviation, different Euclidean distance
     - Left: original units; Right: standardised
     
  2. fig11_correlation_problem.png
     - Correlated data cloud with covariance ellipse
     - Points C (along correlation) and D (against) at same Euclidean distance
     
  3. fig11_probability_contours.png
     - Probability density contours (ellipses) vs Euclidean circle
     - Same circle, very different densities
     
  4. fig11_geometric_preview.png
     - Side-by-side: circles (Euclidean) vs ellipses (Mahalanobis)
     - Preview of the solution
""")
