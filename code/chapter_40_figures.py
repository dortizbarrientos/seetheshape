#!/usr/bin/env python3
"""
Figures for Chapter 40: Worked Examples
=========================================

Generates:
- fig40_example1_two_trait.png: Two-trait G matrix analysis
- fig40_example2_four_trait.png: Four-trait G* analysis
- fig40_example3_selection.png: Selection with G and gamma

Author: Daniel Ortiz-Barrientos & Claude
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

# Style settings
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
# Figure 1: Example 1 - Two-trait analysis
# =============================================================================

print("Creating fig40_example1_two_trait.png...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

# Matrices from Example 1
G = np.array([[25, 15], [15, 36]])
P = np.array([[50, 20], [20, 60]])

# Eigendecompose G
eig_G = np.linalg.eigh(G)
idx_G = np.argsort(eig_G[0])[::-1]
eigenvalues_G = eig_G[0][idx_G]
eigenvectors_G = eig_G[1][:, idx_G]
g_max = eigenvectors_G[:, 0]
angle_G = np.degrees(np.arctan2(g_max[1], g_max[0]))

# Eigendecompose P
eig_P = np.linalg.eigh(P)
idx_P = np.argsort(eig_P[0])[::-1]
eigenvalues_P = eig_P[0][idx_P]
eigenvectors_P = eig_P[1][:, idx_P]
p_max = eigenvectors_P[:, 0]
angle_P = np.degrees(np.arctan2(p_max[1], p_max[0]))

# Left panel: G and P ellipses
ax = axes[0]

# P ellipse (1 SD)
width_P = 2 * np.sqrt(eigenvalues_P[0])
height_P = 2 * np.sqrt(eigenvalues_P[1])
ellipse_P = Ellipse((0, 0), width_P, height_P, angle=angle_P,
                    fill=True, facecolor=LIGHT_BLUE, alpha=0.3,
                    edgecolor=BLUE, linewidth=2.5, label='P ellipse')
ax.add_patch(ellipse_P)

# G ellipse (1 SD)
width_G = 2 * np.sqrt(eigenvalues_G[0])
height_G = 2 * np.sqrt(eigenvalues_G[1])
ellipse_G = Ellipse((0, 0), width_G, height_G, angle=angle_G,
                    fill=True, facecolor=LIGHT_GREEN, alpha=0.4,
                    edgecolor=GREEN, linewidth=2.5, label='G ellipse')
ax.add_patch(ellipse_G)

# Draw g_max
scale = 10
ax.annotate('', xy=g_max*scale, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color=ORANGE, lw=2.5))
ax.text(g_max[0]*scale + 0.5, g_max[1]*scale + 0.5, 
        r'$\mathbf{g}_{max}$' + f'\n$\\lambda = {eigenvalues_G[0]:.1f}$', 
        fontsize=10, color=ORANGE, fontweight='bold')

# Draw specific directions with h² values
directions = [
    ((1, 0), 'Trait 1 only', 0.50),
    ((0, 1), 'Trait 2 only', 0.60),
]

for (d, label, h2) in directions:
    d = np.array(d) / np.linalg.norm(d)
    ax.plot([0, d[0]*8], [0, d[1]*8], '--', color=GRAY, linewidth=1.5, alpha=0.7)
    ax.text(d[0]*8.5, d[1]*8.5, f'{label}\n$h^2={h2:.2f}$', 
            fontsize=9, color=GRAY, ha='center')

ax.axhline(0, color='lightgray', linewidth=0.5, zorder=0)
ax.axvline(0, color='lightgray', linewidth=0.5, zorder=0)
ax.scatter([0], [0], s=60, c='black', marker='+', zorder=4, linewidths=2)

ax.set_xlim(-12, 12)
ax.set_ylim(-10, 10)
ax.set_aspect('equal')
ax.set_xlabel('Flowering time (genetic units)')
ax.set_ylabel('Height (genetic units)')
ax.set_title('G inside P: Two-Trait Analysis', fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)

# Right panel: Directional heritabilities
ax = axes[1]

# Compute G*
V_P = eigenvectors_P
Lambda_P_inv_sqrt = np.diag(1/np.sqrt(eigenvalues_P))
P_inv_sqrt = V_P @ Lambda_P_inv_sqrt @ V_P.T
G_star = P_inv_sqrt @ G @ P_inv_sqrt

# Eigenvalues of G* are directional h²
eig_Gstar = np.linalg.eigh(G_star)
h2_max = max(eig_Gstar[0])
h2_min = min(eig_Gstar[0])

# Bar chart of heritabilities
directions_labels = ['Trait 1\nonly', 'Trait 2\nonly', r'$\mathbf{g}_{max}$', 
                     r'$\mathbf{g}_{min}$', 'Max $h^2$\ndir.', 'Min $h^2$\ndir.']
h2_values = [0.50, 0.60, 0.616, 0.420, h2_max, h2_min]
colors_bars = [GRAY, GRAY, ORANGE, PURPLE, GREEN, RED]

bars = ax.bar(range(len(h2_values)), h2_values, color=colors_bars, alpha=0.7,
              edgecolor='black', linewidth=1)

for i, (v, c) in enumerate(zip(h2_values, colors_bars)):
    ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=10, fontweight='bold')

ax.axhline(0.5, color='lightgray', linestyle='--', linewidth=1)
ax.text(5.5, 0.51, 'Mean', fontsize=9, color='gray')

ax.set_xticks(range(len(h2_values)))
ax.set_xticklabels(directions_labels, fontsize=9)
ax.set_ylabel('Directional Heritability')
ax.set_ylim(0, 0.8)
ax.set_title('Heritability Varies by Direction', fontsize=12, fontweight='bold')

# Annotation
ax.text(0.98, 0.98, 
        f'Range: {h2_min:.2f} to {h2_max:.2f}\n'
        f'Difference: {h2_max - h2_min:.2f}',
        transform=ax.transAxes, fontsize=10, ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig('fig40_example1_two_trait.png', dpi=300, facecolor='white')
print("  Saved: fig40_example1_two_trait.png")
plt.close()

# =============================================================================
# Figure 2: Example 2 - Four-trait analysis
# =============================================================================

print("Creating fig40_example2_four_trait.png...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

# Four-trait matrices
G_4 = np.array([
    [0.80, 0.45, 0.20, 0.15],
    [0.45, 0.60, 0.25, 0.18],
    [0.20, 0.25, 0.35, 0.28],
    [0.15, 0.18, 0.28, 0.30]
])

P_4 = np.array([
    [1.20, 0.55, 0.30, 0.22],
    [0.55, 0.95, 0.35, 0.25],
    [0.30, 0.35, 0.55, 0.40],
    [0.22, 0.25, 0.40, 0.50]
])

# Compute G*
eig_P4 = np.linalg.eigh(P_4)
V_P4 = eig_P4[1]
P4_inv_sqrt = V_P4 @ np.diag(1/np.sqrt(eig_P4[0])) @ V_P4.T
G_star_4 = P4_inv_sqrt @ G_4 @ P4_inv_sqrt

eig_Gstar4 = np.linalg.eigh(G_star_4)
h2_eigenvalues = np.sort(eig_Gstar4[0])[::-1]

# Left panel: Eigenvalue spectrum
ax = axes[0]

pc_nums = np.arange(1, 5)
bars = ax.bar(pc_nums, h2_eigenvalues, color=[GREEN, TEAL, BLUE, PURPLE], 
              alpha=0.7, edgecolor='black', linewidth=1)

for i, v in enumerate(h2_eigenvalues):
    ax.text(i+1, v + 0.02, f'{v:.2f}', ha='center', fontsize=11, fontweight='bold')

ax.axhline(np.mean(h2_eigenvalues), color=ORANGE, linestyle='--', linewidth=2)
ax.text(4.5, np.mean(h2_eigenvalues) + 0.02, f'Mean = {np.mean(h2_eigenvalues):.2f}', 
        fontsize=10, color=ORANGE)

ax.set_xlabel('Principal Axis')
ax.set_ylabel('Directional Heritability (eigenvalue of G*)')
ax.set_xticks(pc_nums)
ax.set_xticklabels(['Axis 1\n(Size)', 'Axis 2', 'Axis 3', 'Axis 4\n(Shape)'])
ax.set_ylim(0, 0.95)
ax.set_title('Eigenvalues of G*\nDirectional Heritabilities Along Principal Axes',
             fontsize=12, fontweight='bold')

# Right panel: Loading interpretation
ax = axes[1]

# Get eigenvectors of G*
eigenvectors_Gstar4 = eig_Gstar4[1]
idx_sort = np.argsort(eig_Gstar4[0])[::-1]
eigenvectors_sorted = eigenvectors_Gstar4[:, idx_sort]

# Plot loadings as heatmap
trait_names = ['Wing', 'Tarsus', 'Bill depth', 'Bill width']
im = ax.imshow(eigenvectors_sorted.T, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

for i in range(4):
    for j in range(4):
        val = eigenvectors_sorted[j, i]
        color = 'white' if abs(val) > 0.4 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=10, color=color)

ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(trait_names, fontsize=9)
ax.set_yticklabels([f'Axis {i+1}\n($h^2$={h2_eigenvalues[i]:.2f})' for i in range(4)], 
                  fontsize=9)
ax.set_xlabel('Trait')
ax.set_title('Eigenvector Loadings\nWhat each axis represents', fontsize=12, fontweight='bold')

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Loading', fontsize=10)

plt.tight_layout()
plt.savefig('fig40_example2_four_trait.png', dpi=300, facecolor='white')
print("  Saved: fig40_example2_four_trait.png")
plt.close()

# =============================================================================
# Figure 3: Example 3 - Selection analysis
# =============================================================================

print("Creating fig40_example3_selection.png...")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Matrices from Example 3
G_sel = np.array([[0.45, 0.30], [0.30, 0.35]])
gamma = np.array([[-0.15, 0.08], [0.08, -0.10]])
beta = np.array([0.18, 0.12])

# Eigendecompose G
eig_G_sel = np.linalg.eigh(G_sel)
idx = np.argsort(eig_G_sel[0])[::-1]
eigenvalues_G_sel = eig_G_sel[0][idx]
eigenvectors_G_sel = eig_G_sel[1][:, idx]
g_max_sel = eigenvectors_G_sel[:, 0]
angle_G_sel = np.degrees(np.arctan2(g_max_sel[1], g_max_sel[0]))

# Eigendecompose gamma
eig_gamma = np.linalg.eigh(gamma)
idx_g = np.argsort(np.abs(eig_gamma[0]))  # Sort by absolute value
eigenvalues_gamma = eig_gamma[0][idx_g]
eigenvectors_gamma = eig_gamma[1][:, idx_g]

# Left panel: G ellipse with selection
ax = axes[0]

# G ellipse
width_G_sel = 2 * 2 * np.sqrt(eigenvalues_G_sel[0])
height_G_sel = 2 * 2 * np.sqrt(eigenvalues_G_sel[1])
ellipse_G_sel = Ellipse((0, 0), width_G_sel, height_G_sel, angle=angle_G_sel,
                        fill=True, facecolor=LIGHT_GREEN, alpha=0.4,
                        edgecolor=GREEN, linewidth=2.5, label='G ellipse')
ax.add_patch(ellipse_G_sel)

# Draw g_max
ax.annotate('', xy=g_max_sel*1.8, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color=ORANGE, lw=2.5))
ax.text(g_max_sel[0]*1.9 + 0.1, g_max_sel[1]*1.9, r'$\mathbf{g}_{max}$', 
        fontsize=10, color=ORANGE, fontweight='bold')

# Selection gradient
beta_scale = 5
ax.annotate('', xy=beta*beta_scale, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color=RED, lw=3))
ax.text(beta[0]*beta_scale + 0.1, beta[1]*beta_scale + 0.1, 
        r'$\boldsymbol{\beta}$', fontsize=11, color=RED, fontweight='bold')

# Response
response = G_sel @ beta
response_scale = 5
ax.annotate('', xy=response*response_scale, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color=BLUE, lw=3))
ax.text(response[0]*response_scale - 0.15, response[1]*response_scale + 0.15, 
        r'$\Delta\bar{\mathbf{z}}$', fontsize=11, color=BLUE, fontweight='bold')

ax.axhline(0, color='lightgray', linewidth=0.5, zorder=0)
ax.axvline(0, color='lightgray', linewidth=0.5, zorder=0)
ax.scatter([0], [0], s=60, c='black', marker='+', zorder=4, linewidths=2)

ax.set_xlim(-2, 2)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.set_xlabel('Trait 1')
ax.set_ylabel('Trait 2')
ax.set_title('G Matrix and Selection Response', fontsize=11, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)

# Middle panel: Gamma (fitness curvature)
ax = axes[1]

# Gamma eigenvectors and eigenvalues
gamma_v1 = eigenvectors_gamma[:, 0]  # Weak stabilising
gamma_v2 = eigenvectors_gamma[:, 1]  # Strong stabilising

# Draw fitness contours (schematic)
# The gamma matrix defines curvature, so contours are ellipses
# Negative eigenvalues = stabilising = peak
# Draw as level curves of quadratic

theta_range = np.linspace(0, 2*np.pi, 100)
for level in [0.5, 1.0, 1.5]:
    # Ellipse scaled inversely by sqrt(|eigenvalue|)
    # Larger |eigenvalue| = narrower in that direction
    a = level / np.sqrt(np.abs(eigenvalues_gamma[1]))  # Strong direction (narrow)
    b = level / np.sqrt(np.abs(eigenvalues_gamma[0]))  # Weak direction (wide)
    angle_gamma = np.arctan2(gamma_v2[1], gamma_v2[0])
    x = a * np.cos(theta_range) * np.cos(angle_gamma) - b * np.sin(theta_range) * np.sin(angle_gamma)
    y = a * np.cos(theta_range) * np.sin(angle_gamma) + b * np.sin(theta_range) * np.cos(angle_gamma)
    alpha = 0.8 - 0.2 * level
    ax.plot(x, y, '-', color=PURPLE, linewidth=1.5, alpha=alpha)

# Mark eigenvector directions
ax.annotate('', xy=gamma_v1*2, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=2))
ax.text(gamma_v1[0]*2.2, gamma_v1[1]*2.2, 
        f'Weak stab.\n$\\lambda = {eigenvalues_gamma[0]:.2f}$', 
        fontsize=9, color=GREEN, ha='center')

ax.annotate('', xy=gamma_v2*1.2, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color=RED, lw=2))
ax.text(gamma_v2[0]*1.4, gamma_v2[1]*1.4 - 0.2, 
        f'Strong stab.\n$\\lambda = {eigenvalues_gamma[1]:.2f}$', 
        fontsize=9, color=RED, ha='center')

ax.axhline(0, color='lightgray', linewidth=0.5, zorder=0)
ax.axvline(0, color='lightgray', linewidth=0.5, zorder=0)
ax.scatter([0], [0], s=80, c=PURPLE, marker='*', zorder=5, label='Optimum')

ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.set_xlabel('Trait 1')
ax.set_ylabel('Trait 2')
ax.set_title(r'Fitness Surface ($\boldsymbol{\gamma}$)', fontsize=11, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)

# Right panel: Alignment summary
ax = axes[2]
ax.axis('off')

# Compute angle between g_max and gamma eigenvectors
angle_g_max = np.arctan2(g_max_sel[1], g_max_sel[0])
angle_gamma_weak = np.arctan2(gamma_v1[1], gamma_v1[0])
angle_between = np.abs(angle_g_max - angle_gamma_weak)
if angle_between > np.pi/2:
    angle_between = np.pi - angle_between

summary_text = """
ALIGNMENT ANALYSIS

G matrix (genetic variation):
  • λ₁ = 0.68 (along g_max)
  • λ₂ = 0.12 (perpendicular)
  • Ratio: 5.7:1

γ matrix (fitness curvature):
  • λ₁ = -0.05 (weak stabilising)
  • λ₂ = -0.20 (strong stabilising)
  • Ratio: 1:4

ALIGNMENT:
  g_max aligns with weak stabilising
  direction (both ≈ 45°)

CONSEQUENCE:
  ✓ Evolution CAN proceed along g_max
  ✓ Fitness ridge allows movement
  ✗ Perpendicular constrained by:
    - Low genetic variance (0.12)
    - Strong stabilising selection (-0.20)

PREDICTION:
  Population will evolve along the
  size axis; shape variation eroded.
"""

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
        fontfamily='monospace', va='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95))

ax.set_title('G-γ Alignment Summary', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('fig40_example3_selection.png', dpi=300, facecolor='white')
print("  Saved: fig40_example3_selection.png")
plt.close()

# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*60)
print("Chapter 40 Figures Complete")
print("="*60)
print("""
Generated figures:
  1. fig40_example1_two_trait.png
     - Left: G inside P ellipses for flowering time and height
     - Right: Bar chart of directional heritabilities
     
  2. fig40_example2_four_trait.png
     - Left: Eigenvalue spectrum of G* (dir. heritabilities)
     - Right: Eigenvector loadings heatmap
     
  3. fig40_example3_selection.png
     - Left: G ellipse with selection and response
     - Middle: Fitness surface from gamma
     - Right: Alignment analysis summary
""")
