#!/usr/bin/env python3
"""
Figures for Chapter 21: Whitening and the P-Sphere
====================================================

Generates:
- fig21_sphere_vs_psphere.png: Euclidean sphere vs P-sphere
- fig21_constraint_trap.png: Constraint trap in whitened space
- fig21_g_inside_p.png: G ellipse inside P ellipse, before and after whitening

Author: Daniel Ortiz-Barrientos
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from matplotlib.lines import Line2D
from scipy.linalg import sqrtm

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
# Figure 1: Euclidean Sphere vs P-Sphere
# =============================================================================

print("Creating fig21_sphere_vs_psphere.png...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

# Phenotypic covariance matrix
rho = 0.7
P = np.array([[1.5, rho*np.sqrt(1.5*0.8)], [rho*np.sqrt(1.5*0.8), 0.8]])
eigenvalues_P, eigenvectors_P = np.linalg.eigh(P)
angle_P = np.degrees(np.arctan2(eigenvectors_P[1, 1], eigenvectors_P[0, 1]))

# Left panel: Euclidean unit sphere (circle)
ax = axes[0]

# Draw unit circle
theta = np.linspace(0, 2*np.pi, 100)
ax.plot(np.cos(theta), np.sin(theta), '-', color=BLUE, linewidth=2.5,
        label='Euclidean unit sphere')

# Sample some "uniform" directions on Euclidean sphere
np.random.seed(42)
n_samples = 12
angles = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
for ang in angles:
    x, y = np.cos(ang), np.sin(ang)
    ax.plot([0, x], [0, y], '-', color=GRAY, linewidth=0.8, alpha=0.5)
    ax.scatter([x], [y], s=50, c=ORANGE, zorder=3)

# Draw P-ellipse for reference (where phenotypic variance = 1)
# P-sphere: beta' P beta = 1
# This is an ellipse with semi-axes 1/sqrt(eigenvalues)
width_P = 2 / np.sqrt(eigenvalues_P[1])
height_P = 2 / np.sqrt(eigenvalues_P[0])
ellipse_P = Ellipse((0, 0), width_P, height_P, angle=angle_P,
                    fill=False, edgecolor=GREEN, linewidth=2, linestyle='--',
                    label='P-sphere (unit pheno. var.)')
ax.add_patch(ellipse_P)

ax.axhline(0, color='lightgray', linewidth=0.5, zorder=0)
ax.axvline(0, color='lightgray', linewidth=0.5, zorder=0)
ax.scatter([0], [0], s=60, c='black', marker='+', zorder=4, linewidths=2)

ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.set_xlabel('Trait 1')
ax.set_ylabel('Trait 2')
ax.set_title('Euclidean Unit Sphere\n(ignores phenotypic structure)', 
             fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)

# Annotation
ax.text(0.02, 0.98, 
        '"Uniform" here is NOT\nuniform w.r.t. phenotype', 
        transform=ax.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

# Right panel: P-sphere
ax = axes[1]

# Draw P-sphere (ellipse in original coordinates)
ellipse_P2 = Ellipse((0, 0), width_P, height_P, angle=angle_P,
                     fill=False, edgecolor=GREEN, linewidth=2.5,
                     label='P-sphere')
ax.add_patch(ellipse_P2)

# Sample uniform directions on P-sphere
# In whitened space, sample from unit circle, then transform back
P_sqrt = eigenvectors_P @ np.diag(np.sqrt(eigenvalues_P)) @ eigenvectors_P.T
P_inv_sqrt = eigenvectors_P @ np.diag(1/np.sqrt(eigenvalues_P)) @ eigenvectors_P.T

for ang in angles:
    # Unit vector in whitened space
    w = np.array([np.cos(ang), np.sin(ang)])
    # Transform to original space (on P-sphere)
    beta = P_inv_sqrt @ w
    # Normalize to be on P-sphere (should already be there)
    beta = beta / np.sqrt(beta @ P @ beta)
    ax.plot([0, beta[0]], [0, beta[1]], '-', color=GRAY, linewidth=0.8, alpha=0.5)
    ax.scatter([beta[0]], [beta[1]], s=50, c=ORANGE, zorder=3)

# Draw Euclidean circle for reference
circle = Circle((0, 0), 1, fill=False, edgecolor=BLUE, linewidth=1.5, 
                linestyle=':', alpha=0.5, label='Euclidean sphere')
ax.add_patch(circle)

ax.axhline(0, color='lightgray', linewidth=0.5, zorder=0)
ax.axvline(0, color='lightgray', linewidth=0.5, zorder=0)
ax.scatter([0], [0], s=60, c='black', marker='+', zorder=4, linewidths=2)

ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.set_xlabel('Trait 1')
ax.set_ylabel('Trait 2')
ax.set_title('P-Sphere\n(uniform w.r.t. phenotypic variance)', 
             fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)

# Annotation
ax.text(0.02, 0.98, 
        'All points have equal\nphenotypic variance', 
        transform=ax.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round', facecolor=LIGHT_GREEN, alpha=0.9))

plt.tight_layout()
plt.savefig('fig21_sphere_vs_psphere.png', dpi=300, facecolor='white')
print("  Saved: fig21_sphere_vs_psphere.png")
plt.close()

# =============================================================================
# Figure 2: Constraint Trap
# =============================================================================

print("Creating fig21_constraint_trap.png...")

fig, ax = plt.subplots(figsize=(8, 7))

# In whitened space, P becomes the unit circle
# G* is an ellipse inside it

# G* eigenvalues (directional heritabilities)
h2_max = 0.85  # Maximum directional heritability
h2_min = 0.15  # Minimum directional heritability (the constraint trap)

# G* is diagonal in its own eigenbasis
# Let's rotate it 30 degrees for visual interest
rotation_angle = 30
theta_rad = np.radians(rotation_angle)
R = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
              [np.sin(theta_rad), np.cos(theta_rad)]])

# Draw P-sphere (unit circle in whitened space)
theta = np.linspace(0, 2*np.pi, 100)
ax.plot(np.cos(theta), np.sin(theta), '-', color=BLUE, linewidth=2.5,
        label='P-sphere (unit circle)')
ax.fill(np.cos(theta), np.sin(theta), color=LIGHT_BLUE, alpha=0.15)

# Draw G* ellipse
g_star_ellipse = Ellipse((0, 0), 2*np.sqrt(h2_max), 2*np.sqrt(h2_min), 
                         angle=rotation_angle,
                         fill=True, facecolor=LIGHT_GREEN, alpha=0.4,
                         edgecolor=GREEN, linewidth=2.5, label='G* ellipse')
ax.add_patch(g_star_ellipse)

# Mark the high h² direction (g*_max)
g_max_dir = R @ np.array([1, 0])
ax.annotate('', xy=g_max_dir*1.3, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color=ORANGE, lw=3))
ax.text(g_max_dir[0]*1.4, g_max_dir[1]*1.4 + 0.15, 
        f'High $h^2$ direction\n$h^2 = {h2_max:.2f}$', 
        fontsize=10, color=ORANGE, ha='center', fontweight='bold')

# Mark the low h² direction (constraint trap)
g_min_dir = R @ np.array([0, 1])
ax.annotate('', xy=g_min_dir*1.3, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color=RED, lw=3))
ax.text(g_min_dir[0]*1.4 - 0.2, g_min_dir[1]*1.4, 
        f'CONSTRAINT TRAP\n$h^2 = {h2_min:.2f}$', 
        fontsize=10, color=RED, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor=RED, alpha=0.9))

# Show the gap between G* and P-sphere in the trap direction
trap_on_gsphere = g_min_dir * np.sqrt(h2_min)
trap_on_psphere = g_min_dir * 1.0
ax.annotate('', xy=trap_on_psphere, xytext=trap_on_gsphere,
            arrowprops=dict(arrowstyle='<->', color=PURPLE, lw=2))
ax.text(-0.1, 0.65, 'Gap = low $h^2$', fontsize=9, color=PURPLE, ha='right')

ax.axhline(0, color='lightgray', linewidth=0.5, zorder=0)
ax.axvline(0, color='lightgray', linewidth=0.5, zorder=0)
ax.scatter([0], [0], s=60, c='black', marker='+', zorder=4, linewidths=2)

ax.set_xlim(-1.8, 1.8)
ax.set_ylim(-1.5, 1.8)
ax.set_aspect('equal')
ax.set_xlabel('Whitened trait 1')
ax.set_ylabel('Whitened trait 2')
ax.set_title('Constraint Trap: G* thin relative to P-sphere', 
             fontsize=12, fontweight='bold')

# Legend
ax.legend(loc='lower right', fontsize=9)

# Explanation box
explanation = (
    "In whitened space:\n"
    "• P-sphere = unit circle\n"
    "• G* ellipse sits inside\n"
    "• Where G* is thin → low $h^2$\n"
    "• Plenty of pheno. variance,\n"
    "  but little genetic variance"
)
ax.text(0.02, 0.02, explanation, transform=ax.transAxes, fontsize=9, va='bottom',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95))

plt.savefig('fig21_constraint_trap.png', dpi=300, facecolor='white')
print("  Saved: fig21_constraint_trap.png")
plt.close()

# =============================================================================
# Figure 3: G inside P (before and after whitening)
# =============================================================================

print("Creating fig21_g_inside_p.png...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

# Define G and P matrices
P = np.array([[1.2, 0.6], [0.6, 0.8]])
G = np.array([[0.8, 0.5], [0.5, 0.4]])  # G < P (h² < 1 in all directions)

# Eigendecompose both
eig_P = np.linalg.eigh(P)
eig_G = np.linalg.eigh(G)

angle_P = np.degrees(np.arctan2(eig_P[1][1, 1], eig_P[1][0, 1]))
angle_G = np.degrees(np.arctan2(eig_G[1][1, 1], eig_G[1][0, 1]))

# Left panel: Original coordinates
ax = axes[0]

# Generate phenotypic and genetic data
np.random.seed(789)
n = 150
data_P = np.random.multivariate_normal([0, 0], P, n)
data_G = np.random.multivariate_normal([0, 0], G, n)

ax.scatter(data_P[:, 0], data_P[:, 1], s=15, c=LIGHT_BLUE, alpha=0.3, 
           zorder=1, label='Phenotypic')

# Draw P ellipse (1 SD)
ellipse_P = Ellipse((0, 0), 2*np.sqrt(eig_P[0][1]), 2*np.sqrt(eig_P[0][0]), 
                    angle=angle_P,
                    fill=False, edgecolor=BLUE, linewidth=2.5, label='P ellipse')
ax.add_patch(ellipse_P)

# Draw G ellipse (1 SD)
ellipse_G = Ellipse((0, 0), 2*np.sqrt(eig_G[0][1]), 2*np.sqrt(eig_G[0][0]), 
                    angle=angle_G,
                    fill=False, edgecolor=GREEN, linewidth=2.5, label='G ellipse')
ax.add_patch(ellipse_G)

ax.axhline(0, color='lightgray', linewidth=0.5, zorder=0)
ax.axvline(0, color='lightgray', linewidth=0.5, zorder=0)
ax.scatter([0], [0], s=60, c='black', marker='+', zorder=4, linewidths=2)

ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.set_xlabel('Trait 1')
ax.set_ylabel('Trait 2')
ax.set_title('Original Coordinates\n(both G and P are ellipses)', 
             fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)

# Right panel: Whitened coordinates
ax = axes[1]

# Compute P^{-1/2}
P_inv_sqrt = eig_P[1] @ np.diag(1/np.sqrt(eig_P[0])) @ eig_P[1].T

# Compute G*
G_star = P_inv_sqrt @ G @ P_inv_sqrt

# Eigendecompose G*
eig_Gstar = np.linalg.eigh(G_star)
angle_Gstar = np.degrees(np.arctan2(eig_Gstar[1][1, 1], eig_Gstar[1][0, 1]))

# Whiten the phenotypic data
data_P_white = (P_inv_sqrt @ data_P.T).T

ax.scatter(data_P_white[:, 0], data_P_white[:, 1], s=15, c=LIGHT_BLUE, alpha=0.3, 
           zorder=1, label='Whitened data')

# P becomes unit circle
circle_P = Circle((0, 0), 1, fill=False, edgecolor=BLUE, linewidth=2.5,
                  label='P* = unit circle')
ax.add_patch(circle_P)

# G* ellipse
ellipse_Gstar = Ellipse((0, 0), 2*np.sqrt(eig_Gstar[0][1]), 2*np.sqrt(eig_Gstar[0][0]), 
                        angle=angle_Gstar,
                        fill=False, edgecolor=GREEN, linewidth=2.5, label='G* ellipse')
ax.add_patch(ellipse_Gstar)

# Mark eigenvalues of G* (directional heritabilities)
h2_vals = sorted(eig_Gstar[0], reverse=True)
ax.text(0.98, 0.98, 
        f"Eigenvalues of G*\n(directional $h^2$):\n"
        f"  $\\lambda_1^*$ = {h2_vals[0]:.3f}\n"
        f"  $\\lambda_2^*$ = {h2_vals[1]:.3f}", 
        transform=ax.transAxes, fontsize=10, va='top', ha='right',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor=GREEN, alpha=0.9))

ax.axhline(0, color='lightgray', linewidth=0.5, zorder=0)
ax.axvline(0, color='lightgray', linewidth=0.5, zorder=0)
ax.scatter([0], [0], s=60, c='black', marker='+', zorder=4, linewidths=2)

ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.set_xlabel('Whitened axis 1')
ax.set_ylabel('Whitened axis 2')
ax.set_title('Whitened Coordinates\n(P = circle, G* reveals constraint)', 
             fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig('fig21_g_inside_p.png', dpi=300, facecolor='white')
print("  Saved: fig21_g_inside_p.png")
plt.close()

# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*60)
print("Chapter 21 Figures Complete")
print("="*60)
print("""
Generated figures:
  1. fig21_sphere_vs_psphere.png
     - Left: Euclidean unit sphere (ignores phenotypic structure)
     - Right: P-sphere (uniform w.r.t. phenotypic variance)
     - Shows same directions sampled on both
     
  2. fig21_constraint_trap.png
     - P-sphere as unit circle (whitened space)
     - G* ellipse inside, showing high h² and low h² directions
     - Constraint trap where G* is thin relative to P-sphere
     
  3. fig21_g_inside_p.png
     - Left: Original coordinates, G and P both ellipses
     - Right: Whitened coordinates, P = unit circle
     - Eigenvalues of G* shown as directional heritabilities
""")
