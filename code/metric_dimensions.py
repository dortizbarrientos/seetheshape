#!/usr/bin/env python3
"""
Building the Metric: From 1D to Many Dimensions
================================================

A pedagogical progression that builds geometric intuition by starting
with the simplest possible case (1D) and systematically adding complexity.

1D: A metric is just a number (variance)
2D: A metric is an ellipse  
3D: A metric is an ellipsoid
pD: A metric is a hyperellipsoid (but the math is identical!)

Author: Daniel Ortiz-Barrientos & Claude
Date: December 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, FancyArrowPatch, Rectangle
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patheffects as pe

# =============================================================================
# COLOR PALETTE
# =============================================================================
PAL = {
    'euclidean': '#4A4A4A',
    'G': '#2E86AB',
    'P': '#B2182B',
    'E': '#F4A582',
    'Gstar': '#762A83',
    'beta': '#F18F01',
    'high': '#1B7837',
    'low': '#E63946',
    'neutral': '#878787',
    'dim1': '#2E86AB',
    'dim2': '#E63946',
    'dim3': '#1B7837',
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'figure.facecolor': 'white',
    'savefig.dpi': 300,
})

# =============================================================================
# FIGURE 1: THE 1D CASE - WHERE IT ALL BEGINS
# =============================================================================

print("Creating Figure 1: The 1D Case...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# -----------------------------------------------------------------------------
# Panel A: In 1D, the "unit ball" is an interval
# -----------------------------------------------------------------------------
ax = axes[0, 0]

# Draw number line
ax.axhline(0, color='black', linewidth=1)
ax.plot([-2, 2], [0, 0], 'k-', linewidth=0.5)

# Euclidean unit ball: |x| ≤ 1
ax.plot([-1, 1], [0.3, 0.3], color=PAL['euclidean'], linewidth=8, solid_capstyle='butt')
ax.plot(-1, 0.3, 'o', color=PAL['euclidean'], markersize=10)
ax.plot(1, 0.3, 'o', color=PAL['euclidean'], markersize=10)
ax.text(0, 0.45, 'Euclidean: |x| ≤ 1', ha='center', fontsize=10, color=PAL['euclidean'])

# Weighted unit ball with σ² = 0.5 (more spread)
sigma_sq = 0.5
ax.plot([-1/np.sqrt(sigma_sq), 1/np.sqrt(sigma_sq)], [-0.3, -0.3], 
       color=PAL['G'], linewidth=8, solid_capstyle='butt')
ax.plot(-1/np.sqrt(sigma_sq), -0.3, 'o', color=PAL['G'], markersize=10)
ax.plot(1/np.sqrt(sigma_sq), -0.3, 'o', color=PAL['G'], markersize=10)
ax.text(0, -0.45, f'σ² = {sigma_sq}: |x| ≤ 1/√σ² = {1/np.sqrt(sigma_sq):.2f}', 
       ha='center', fontsize=10, color=PAL['G'])

# Mark the origin
ax.plot(0, 0, 'ko', markersize=8)

ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-0.8, 0.8)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('A. The "unit ball" in 1D is an interval\nSmaller σ² → wider interval', fontsize=11)

# -----------------------------------------------------------------------------
# Panel B: The metric in 1D
# -----------------------------------------------------------------------------
ax = axes[0, 1]
ax.axis('off')

text = """
IN ONE DIMENSION

The "metric" is just a single number: σ²

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Euclidean length:     ||x||² = x²

Weighted length:      ||x||²_σ = σ² · x²

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The "covariance matrix" is just σ² (a 1×1 matrix!)

The "eigenvalue" is just σ² itself.

There's only ONE direction (+x or -x), so
there's no directional dependence.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Heritability: h² = σ²_G / σ²_P

Just one number. No geometry yet!
"""

ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=10,
       ha='center', va='center', fontfamily='monospace',
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
ax.set_title('B. The 1D metric is trivial', fontsize=11)

# -----------------------------------------------------------------------------
# Panel C: Visualizing G and P in 1D
# -----------------------------------------------------------------------------
ax = axes[0, 2]

# Number line
ax.axhline(0, color='black', linewidth=1)

# G interval (smaller variance = wider reach per unit of squared length)
sigma_G = 0.6
sigma_P = 1.0

# For x'Gx = 1, we need x² = 1/G, so x = ±1/√G
# But in 1D, G = σ²_G, so unit ball is |x| ≤ √(1/σ²_G) = 1/σ_G... wait
# Actually for variance, larger σ² means MORE spread, so the distribution is wider
# Let me think about this more carefully.

# For the MVN in 1D: f(x) ∝ exp(-x²/(2σ²))
# The "unit" contour at 1 SD is |x| = σ
# So larger variance → wider interval

ax.fill_between([-sigma_P, sigma_P], -0.1, 0.1, color=PAL['P'], alpha=0.3, label='P (phenotypic)')
ax.fill_between([-sigma_G, sigma_G], -0.05, 0.05, color=PAL['G'], alpha=0.5, label='G (genetic)')

ax.plot([-sigma_P, -sigma_P], [-0.15, 0.15], color=PAL['P'], linewidth=2)
ax.plot([sigma_P, sigma_P], [-0.15, 0.15], color=PAL['P'], linewidth=2)
ax.plot([-sigma_G, -sigma_G], [-0.15, 0.15], color=PAL['G'], linewidth=2)
ax.plot([sigma_G, sigma_G], [-0.15, 0.15], color=PAL['G'], linewidth=2)

ax.text(0, 0.25, f'h² = σ²_G / σ²_P = {sigma_G**2:.2f}/{sigma_P**2:.2f} = {sigma_G**2/sigma_P**2:.2f}',
       ha='center', fontsize=11, fontweight='bold')

ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-0.4, 0.4)
ax.legend(loc='lower center', fontsize=9)
ax.axis('off')
ax.set_title('C. G and P in 1D: just two intervals\nh² = ratio of squared lengths', fontsize=11)

# -----------------------------------------------------------------------------
# Panel D: The transition to 2D
# -----------------------------------------------------------------------------
ax = axes[1, 0]

# Show 1D "interval" morphing conceptually to 2D ellipse
# Left side: 1D
ax.axhline(0.5, color='black', linewidth=0.5, xmin=0.05, xmax=0.4)
ax.plot([0.1, 0.35], [0.5, 0.5], color=PAL['G'], linewidth=6, solid_capstyle='butt')
ax.text(0.225, 0.65, '1D: interval', ha='center', fontsize=10, transform=ax.transAxes)

# Arrow
ax.annotate('', xy=(0.6, 0.5), xytext=(0.4, 0.5),
           arrowprops=dict(arrowstyle='->', lw=2, color='black'),
           transform=ax.transAxes)
ax.text(0.5, 0.55, 'add a\ndimension', ha='center', fontsize=9, transform=ax.transAxes)

# Right side: 2D ellipse
ellipse = Ellipse((0.75, 0.5), 0.25, 0.15, angle=30,
                  fill=True, facecolor=PAL['G'], alpha=0.5,
                  edgecolor=PAL['G'], linewidth=2, transform=ax.transAxes)
ax.add_patch(ellipse)
ax.text(0.75, 0.7, '2D: ellipse', ha='center', fontsize=10, transform=ax.transAxes)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('D. Adding dimensions: intervals become ellipses', fontsize=11)

# -----------------------------------------------------------------------------
# Panel E: Why 2D is different
# -----------------------------------------------------------------------------
ax = axes[1, 1]
ax.axis('off')

text = """
THE JUMP TO 2D

In 1D:
  • One variance σ²
  • One direction (just + or -)
  • h² is a single number

In 2D:
  • Covariance MATRIX (2×2)
  • Infinitely many directions!
  • h²(β) depends on direction

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The key new ingredient: COVARIANCE

If traits are correlated, the ellipse
is TILTED, not axis-aligned.

Now there are "good" and "bad" directions
for selection!
"""

ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=10,
       ha='center', va='center', fontfamily='monospace',
       bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))
ax.set_title('E. 2D introduces direction-dependence', fontsize=11)

# -----------------------------------------------------------------------------
# Panel F: The eigenvalue interpretation
# -----------------------------------------------------------------------------
ax = axes[1, 2]

# 1D: single eigenvalue
ax.text(0.2, 0.8, '1D:', fontsize=12, fontweight='bold', transform=ax.transAxes)
ax.text(0.2, 0.7, 'λ = σ²\n(the only eigenvalue)', fontsize=10, transform=ax.transAxes)

# 2D: two eigenvalues
ax.text(0.2, 0.5, '2D:', fontsize=12, fontweight='bold', transform=ax.transAxes)
ax.text(0.2, 0.35, 'λ₁, λ₂\n(variance along each\nprincipal axis)', fontsize=10, transform=ax.transAxes)

# General
ax.text(0.2, 0.15, 'pD:', fontsize=12, fontweight='bold', transform=ax.transAxes)
ax.text(0.2, 0.0, 'λ₁, λ₂, ..., λₚ\n(p eigenvalues)', fontsize=10, transform=ax.transAxes)

ax.set_xlim(0, 1)
ax.set_ylim(-0.1, 1)
ax.axis('off')
ax.set_title('F. Eigenvalues: from 1 to many', fontsize=11)

plt.suptitle('LEVEL 1: The One-Dimensional Case',
            fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('metric_1D.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved: metric_1D.png")
plt.close()

# =============================================================================
# FIGURE 2: THE 2D CASE - GEOMETRY EMERGES
# =============================================================================

print("Creating Figure 2: The 2D Case...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Define example matrices
G = np.array([[0.6, 0.2],
              [0.2, 0.4]])
P = G + np.array([[0.4, 0.1], [0.1, 0.5]])

# -----------------------------------------------------------------------------
# Panel A: The Euclidean unit ball is a circle
# -----------------------------------------------------------------------------
ax = axes[0, 0]

circle = Circle((0, 0), 1, fill=False, edgecolor=PAL['euclidean'], linewidth=3)
ax.add_patch(circle)

# Draw axes
ax.annotate('', xy=(1.3, 0), xytext=(0, 0),
           arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax.annotate('', xy=(0, 1.3), xytext=(0, 0),
           arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax.text(1.35, 0, 'x₁', fontsize=11)
ax.text(0, 1.35, 'x₂', fontsize=11)

ax.text(0, -1.5, '||x||² = x₁² + x₂² = 1\nAll directions equivalent', 
       ha='center', fontsize=10)

ax.set_xlim(-1.8, 1.8)
ax.set_ylim(-1.8, 1.8)
ax.set_aspect('equal')
ax.axhline(0, color=PAL['neutral'], linewidth=0.3)
ax.axvline(0, color=PAL['neutral'], linewidth=0.3)
ax.set_title('A. Euclidean: the circle\n(Identity matrix)', fontsize=11)

# -----------------------------------------------------------------------------
# Panel B: A diagonal matrix gives axis-aligned ellipse
# -----------------------------------------------------------------------------
ax = axes[0, 1]

D = np.diag([0.5, 1.5])
eigvals_D = [0.5, 1.5]

# Ellipse semi-axes are sqrt(eigenvalue)
width = 2 * np.sqrt(eigvals_D[1])
height = 2 * np.sqrt(eigvals_D[0])

ellipse = Ellipse((0, 0), width, height, angle=0,
                  fill=True, facecolor=PAL['dim1'], alpha=0.3,
                  edgecolor=PAL['dim1'], linewidth=3)
ax.add_patch(ellipse)

# Mark the eigenvalues
ax.annotate('', xy=(np.sqrt(eigvals_D[1]), 0), xytext=(0, 0),
           arrowprops=dict(arrowstyle='->', color=PAL['high'], lw=2))
ax.text(np.sqrt(eigvals_D[1])+0.1, 0.1, f'√λ₁={np.sqrt(eigvals_D[1]):.2f}', 
       fontsize=9, color=PAL['high'])

ax.annotate('', xy=(0, np.sqrt(eigvals_D[0])), xytext=(0, 0),
           arrowprops=dict(arrowstyle='->', color=PAL['low'], lw=2))
ax.text(0.1, np.sqrt(eigvals_D[0])+0.1, f'√λ₂={np.sqrt(eigvals_D[0]):.2f}', 
       fontsize=9, color=PAL['low'])

ax.text(0, -1.5, f'Diagonal: λ₁={eigvals_D[1]}, λ₂={eigvals_D[0]}\nEllipse aligned with axes', 
       ha='center', fontsize=10)

ax.set_xlim(-1.8, 1.8)
ax.set_ylim(-1.8, 1.8)
ax.set_aspect('equal')
ax.axhline(0, color=PAL['neutral'], linewidth=0.3)
ax.axvline(0, color=PAL['neutral'], linewidth=0.3)
ax.set_title('B. Diagonal matrix: axis-aligned ellipse\n(no covariance)', fontsize=11)

# -----------------------------------------------------------------------------
# Panel C: General covariance matrix - tilted ellipse
# -----------------------------------------------------------------------------
ax = axes[0, 2]

Sigma = np.array([[1.0, 0.6], [0.6, 0.8]])
eigvals, eigvecs = np.linalg.eigh(Sigma)

width = 2 * np.sqrt(eigvals[1])
height = 2 * np.sqrt(eigvals[0])
angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))

ellipse = Ellipse((0, 0), width, height, angle=angle,
                  fill=True, facecolor=PAL['Gstar'], alpha=0.3,
                  edgecolor=PAL['Gstar'], linewidth=3)
ax.add_patch(ellipse)

# Draw eigenvectors
for i in range(2):
    v = eigvecs[:, i] * np.sqrt(eigvals[i])
    color = PAL['high'] if i == 1 else PAL['low']
    ax.annotate('', xy=v, xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', color=color, lw=2))
    ax.text(v[0]*1.2, v[1]*1.2, f'v{i+1}\nλ={eigvals[i]:.2f}', 
           fontsize=9, ha='center', color=color)

ax.text(0, -1.5, 'Covariance ≠ 0: ellipse is tilted\nEigenvectors give the tilt', 
       ha='center', fontsize=10)

ax.set_xlim(-1.8, 1.8)
ax.set_ylim(-1.8, 1.8)
ax.set_aspect('equal')
ax.axhline(0, color=PAL['neutral'], linewidth=0.3)
ax.axvline(0, color=PAL['neutral'], linewidth=0.3)
ax.set_title('C. General covariance: tilted ellipse\n(covariance rotates)', fontsize=11)

# -----------------------------------------------------------------------------
# Panel D: G and P ellipses
# -----------------------------------------------------------------------------
ax = axes[1, 0]

for M, color, label, ls in [(P, PAL['P'], 'P', '--'), (G, PAL['G'], 'G', '-')]:
    ev, evec = np.linalg.eigh(M)
    w = 2 * np.sqrt(ev[1])
    h = 2 * np.sqrt(ev[0])
    ang = np.degrees(np.arctan2(evec[1, 1], evec[0, 1]))
    
    fill = (M is G)
    ellipse = Ellipse((0, 0), w, h, angle=ang,
                     fill=fill, facecolor=color if fill else 'none',
                     alpha=0.3 if fill else 1.0,
                     edgecolor=color, linewidth=2.5, linestyle=ls,
                     label=label)
    ax.add_patch(ellipse)

ax.legend(loc='lower right', fontsize=10)
ax.set_xlim(-1.8, 1.8)
ax.set_ylim(-1.8, 1.8)
ax.set_aspect('equal')
ax.axhline(0, color=PAL['neutral'], linewidth=0.3)
ax.axvline(0, color=PAL['neutral'], linewidth=0.3)
ax.set_title('D. Two rulers: G and P\nP always contains G (since P = G + E)', fontsize=11)

# -----------------------------------------------------------------------------
# Panel E: h²(β) polar plot
# -----------------------------------------------------------------------------
ax = axes[1, 1]

# Compute h² for all directions
thetas = np.linspace(0, 2*np.pi, 360)
h2_vals = []
for theta in thetas:
    beta = np.array([np.cos(theta), np.sin(theta)])
    h2_vals.append((beta @ G @ beta) / (beta @ P @ beta))
h2_vals = np.array(h2_vals)

# Plot as polar in Cartesian
r = 0.3 + 0.7 * h2_vals
x = r * np.cos(thetas)
y = r * np.sin(thetas)

points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = LineCollection(segments, cmap='RdYlGn', 
                   norm=plt.Normalize(h2_vals.min(), h2_vals.max()),
                   linewidth=4)
lc.set_array(h2_vals[:-1])
ax.add_collection(lc)

# Mark max and min
max_idx = np.argmax(h2_vals)
min_idx = np.argmin(h2_vals)
ax.plot(x[max_idx], y[max_idx], 'o', color=PAL['high'], markersize=12,
       markeredgecolor='white', markeredgewidth=2)
ax.plot(x[min_idx], y[min_idx], 'o', color=PAL['low'], markersize=12,
       markeredgecolor='white', markeredgewidth=2)

ax.text(0, -1.4, f'h² ranges from {h2_vals.min():.2f} to {h2_vals.max():.2f}\nDirection matters!', 
       ha='center', fontsize=10)

ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.axhline(0, color=PAL['neutral'], linewidth=0.3)
ax.axvline(0, color=PAL['neutral'], linewidth=0.3)
ax.set_title('E. h²(β) varies with direction\n(radius = heritability)', fontsize=11)

# Colorbar
sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(h2_vals.min(), h2_vals.max()))
plt.colorbar(sm, ax=ax, label='h²(β)', shrink=0.8)

# -----------------------------------------------------------------------------
# Panel F: Summary statistics
# -----------------------------------------------------------------------------
ax = axes[1, 2]
ax.axis('off')

# Compute G*
eigvals_P, eigvecs_P = np.linalg.eigh(P)
P_inv_sqrt = eigvecs_P @ np.diag(1/np.sqrt(eigvals_P)) @ eigvecs_P.T
Gstar = P_inv_sqrt @ G @ P_inv_sqrt
eigvals_Gstar = np.linalg.eigvalsh(Gstar)

mean_h2 = np.mean(eigvals_Gstar)
V_rel = np.var(eigvals_Gstar) / mean_h2**2
CV_h2 = np.sqrt(2 / 4 * V_rel)

text = f"""
2D SUMMARY STATISTICS

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

G* eigenvalues: [{eigvals_Gstar[0]:.3f}, {eigvals_Gstar[1]:.3f}]

Mean h²:  {mean_h2:.3f}
Min h²:   {eigvals_Gstar.min():.3f}
Max h²:   {eigvals_Gstar.max():.3f}

V_rel:    {V_rel:.3f}
CV(h²):   {CV_h2:.3f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

With 2 eigenvalues, we already see
directional variation in h².

The ratio λ_max/λ_min = {eigvals_Gstar.max()/eigvals_Gstar.min():.2f}
tells us the constraint strength.
"""

ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=10,
       ha='center', va='center', fontfamily='monospace',
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
ax.set_title('F. 2D gives us eigenvalue spread', fontsize=11)

plt.suptitle('LEVEL 2: The Two-Dimensional Case — Geometry Emerges',
            fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('metric_2D.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved: metric_2D.png")
plt.close()

# =============================================================================
# FIGURE 3: THE 3D CASE - FULL ELLIPSOID
# =============================================================================

print("Creating Figure 3: The 3D Case...")

fig = plt.figure(figsize=(16, 10))

# Define 3D matrices
G_3d = np.array([[0.6, 0.2, 0.1],
                 [0.2, 0.4, 0.15],
                 [0.1, 0.15, 0.3]])

P_3d = G_3d + np.array([[0.4, 0.1, 0.05],
                        [0.1, 0.5, 0.1],
                        [0.05, 0.1, 0.4]])

# -----------------------------------------------------------------------------
# Panel A: 3D ellipsoid
# -----------------------------------------------------------------------------
ax = fig.add_subplot(2, 3, 1, projection='3d')

# Generate ellipsoid surface
eigvals_G, eigvecs_G = np.linalg.eigh(G_3d)

# Parametric surface
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 30)
U, V = np.meshgrid(u, v)

# Unit sphere
X = np.sin(V) * np.cos(U)
Y = np.sin(V) * np.sin(U)
Z = np.cos(V)

# Transform by sqrt(eigenvalues) along eigenvectors
# This creates the ellipsoid x'G⁻¹x = 1
for i in range(len(u)):
    for j in range(len(v)):
        point = np.array([X[j,i], Y[j,i], Z[j,i]])
        # Scale by sqrt(eigenvalues) in eigenvector basis
        point_eig = eigvecs_G.T @ point
        point_eig = point_eig * np.sqrt(eigvals_G)
        point_transformed = eigvecs_G @ point_eig
        X[j,i], Y[j,i], Z[j,i] = point_transformed

ax.plot_surface(X, Y, Z, alpha=0.4, color=PAL['G'], edgecolor='none')

# Draw principal axes
for i in range(3):
    v = eigvecs_G[:, i] * np.sqrt(eigvals_G[i]) * 1.2
    ax.plot([0, v[0]], [0, v[1]], [0, v[2]], 
           color=['r', 'g', 'b'][i], linewidth=2)
    ax.text(v[0]*1.1, v[1]*1.1, v[2]*1.1, f'λ{i+1}={eigvals_G[i]:.2f}', fontsize=8)

ax.set_xlabel('Trait 1')
ax.set_ylabel('Trait 2')
ax.set_zlabel('Trait 3')
ax.set_title('A. The G ellipsoid in 3D\n(3 eigenvalues = 3 axes)')

# -----------------------------------------------------------------------------
# Panel B: Three 2D slices
# -----------------------------------------------------------------------------
ax = fig.add_subplot(2, 3, 2)

# Show the three 2D slices (projections)
slice_pairs = [(0, 1), (0, 2), (1, 2)]
slice_labels = ['Traits 1-2', 'Traits 1-3', 'Traits 2-3']
offsets = [(0, 0.6), (0.7, 0), (0.7, 0.6)]

for (i, j), label, (ox, oy) in zip(slice_pairs, slice_labels, [(0, 0.5), (0, 0), (0.5, 0)]):
    # Extract 2x2 submatrix
    G_slice = G_3d[np.ix_([i, j], [i, j])]
    
    ev, evec = np.linalg.eigh(G_slice)
    w = 2 * np.sqrt(ev[1]) * 0.4
    h = 2 * np.sqrt(ev[0]) * 0.4
    ang = np.degrees(np.arctan2(evec[1, 1], evec[0, 1]))
    
    ellipse = Ellipse((ox + 0.25, oy + 0.25), w, h, angle=ang,
                     fill=True, facecolor=PAL['G'], alpha=0.4,
                     edgecolor=PAL['G'], linewidth=2, transform=ax.transAxes)
    ax.add_patch(ellipse)
    ax.text(ox + 0.25, oy + 0.02, label, fontsize=9, ha='center', 
           transform=ax.transAxes)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('B. 2D slices through the 3D ellipsoid\n(each slice is a 2D projection)', fontsize=11)

# -----------------------------------------------------------------------------
# Panel C: h²(β) on the sphere
# -----------------------------------------------------------------------------
ax = fig.add_subplot(2, 3, 3, projection='3d')

# Sample directions on sphere and color by h²
n_points = 1000
np.random.seed(42)
betas_3d = np.random.randn(n_points, 3)
betas_3d = betas_3d / np.linalg.norm(betas_3d, axis=1, keepdims=True)

h2_vals_3d = []
for beta in betas_3d:
    h2 = (beta @ G_3d @ beta) / (beta @ P_3d @ beta)
    h2_vals_3d.append(h2)
h2_vals_3d = np.array(h2_vals_3d)

# Plot as colored points on unit sphere
sc = ax.scatter(betas_3d[:, 0], betas_3d[:, 1], betas_3d[:, 2],
               c=h2_vals_3d, cmap='RdYlGn', s=10, alpha=0.7)
plt.colorbar(sc, ax=ax, label='h²(β)', shrink=0.6)

ax.set_xlabel('β₁')
ax.set_ylabel('β₂')
ax.set_zlabel('β₃')
ax.set_title(f'C. h²(β) on the unit sphere\nRange: [{h2_vals_3d.min():.2f}, {h2_vals_3d.max():.2f}]')

# -----------------------------------------------------------------------------
# Panel D: Distribution of h²
# -----------------------------------------------------------------------------
ax = fig.add_subplot(2, 3, 4)

ax.hist(h2_vals_3d, bins=40, density=True, color=PAL['Gstar'], 
       alpha=0.7, edgecolor='white')
ax.axvline(np.mean(h2_vals_3d), color='black', linestyle='--', linewidth=2,
          label=f'Mean = {np.mean(h2_vals_3d):.3f}')
ax.axvline(np.min(h2_vals_3d), color=PAL['low'], linestyle=':', linewidth=2,
          label=f'Min = {np.min(h2_vals_3d):.3f}')
ax.axvline(np.max(h2_vals_3d), color=PAL['high'], linestyle=':', linewidth=2,
          label=f'Max = {np.max(h2_vals_3d):.3f}')

ax.set_xlabel('h²(β)')
ax.set_ylabel('Density')
ax.set_title('D. Distribution of h² across directions')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# -----------------------------------------------------------------------------
# Panel E: Eigenvalue spectrum
# -----------------------------------------------------------------------------
ax = fig.add_subplot(2, 3, 5)

# Compute G*
eigvals_P_3d, eigvecs_P_3d = np.linalg.eigh(P_3d)
P_inv_sqrt_3d = eigvecs_P_3d @ np.diag(1/np.sqrt(eigvals_P_3d)) @ eigvecs_P_3d.T
Gstar_3d = P_inv_sqrt_3d @ G_3d @ P_inv_sqrt_3d
eigvals_Gstar_3d = np.linalg.eigvalsh(Gstar_3d)

colors = [PAL['low'], PAL['beta'], PAL['high']]
bars = ax.bar(range(3), sorted(eigvals_Gstar_3d), color=colors, 
             edgecolor='black', linewidth=1.5)
ax.set_xticks(range(3))
ax.set_xticklabels(['λ*₁\n(min h²)', 'λ*₂', 'λ*₃\n(max h²)'])
ax.set_ylabel('Eigenvalue of G* = h² along axis')
ax.set_title('E. Three eigenvalues of G*\n= h² along three principal directions')

for i, v in enumerate(sorted(eigvals_Gstar_3d)):
    ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')

ax.set_ylim(0, 1)
ax.grid(alpha=0.3, axis='y')

# -----------------------------------------------------------------------------
# Panel F: Summary
# -----------------------------------------------------------------------------
ax = fig.add_subplot(2, 3, 6)
ax.axis('off')

mean_h2_3d = np.mean(eigvals_Gstar_3d)
V_rel_3d = np.var(eigvals_Gstar_3d) / mean_h2_3d**2
CV_h2_3d = np.sqrt(2 / 5 * V_rel_3d)  # p=3, so p+2=5

text = f"""
3D SUMMARY

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

G* eigenvalues:
  λ*₁ = {sorted(eigvals_Gstar_3d)[0]:.3f} (min h²)
  λ*₂ = {sorted(eigvals_Gstar_3d)[1]:.3f}
  λ*₃ = {sorted(eigvals_Gstar_3d)[2]:.3f} (max h²)

Mean h²:  {mean_h2_3d:.3f}
V_rel:    {V_rel_3d:.3f}
CV(h²):   {CV_h2_3d:.3f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

In 3D:
• The ellipsoid has 3 principal axes
• h² varies over the unit sphere
• CV(h²) = √(2V_rel/(p+2))
        = √(2×{V_rel_3d:.3f}/5)
        = {CV_h2_3d:.3f}
"""

ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=10,
       ha='center', va='center', fontfamily='monospace',
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
ax.set_title('F. The pattern continues...', fontsize=11)

plt.suptitle('LEVEL 3: The Three-Dimensional Case — Full Ellipsoid',
            fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('metric_3D.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved: metric_3D.png")
plt.close()

# =============================================================================
# FIGURE 4: THE GENERAL CASE - HYPERELLIPSOIDS
# =============================================================================

print("Creating Figure 4: The General Case (Many Dimensions)...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# -----------------------------------------------------------------------------
# Panel A: The progression
# -----------------------------------------------------------------------------
ax = axes[0, 0]
ax.axis('off')

progression = """
THE DIMENSIONAL LADDER

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1D:  Interval   │  1 eigenvalue
    ─────────   │  No direction dependence

2D:  Ellipse    │  2 eigenvalues
    ⬭           │  h² varies around circle

3D:  Ellipsoid  │  3 eigenvalues
    🥚          │  h² varies over sphere

pD:  Hyperellipsoid  │  p eigenvalues
    [...]            │  h² varies over (p-1)-sphere

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The geometry is always the same:
• p eigenvalues = p principal axes
• Length² in direction β = Σᵢ λᵢ(β·vᵢ)²
• h² = G-length² / P-length²
"""

ax.text(0.5, 0.5, progression, transform=ax.transAxes, fontsize=10,
       ha='center', va='center', fontfamily='monospace',
       bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))
ax.set_title('A. The dimensional ladder', fontsize=11)

# -----------------------------------------------------------------------------
# Panel B: Eigenvalue spectrum grows
# -----------------------------------------------------------------------------
ax = axes[0, 1]

# Show eigenvalue spectra for different p
np.random.seed(42)

for p, color, offset in [(2, PAL['dim1'], 0), (5, PAL['dim2'], 3), (10, PAL['dim3'], 9)]:
    # Generate random G* with some spread
    eigvals = np.linspace(0.3, 0.8, p) + np.random.uniform(-0.05, 0.05, p)
    eigvals = np.sort(eigvals)
    
    x_positions = np.arange(p) + offset
    ax.bar(x_positions, eigvals, color=color, alpha=0.7, 
          edgecolor='black', linewidth=0.5, label=f'p = {p}')

ax.set_xlabel('Eigenvalue index')
ax.set_ylabel('λ* (= h² along that axis)')
ax.set_title('B. More dimensions = more eigenvalues\n(each is an h² value)', fontsize=11)
ax.legend()
ax.set_ylim(0, 1)
ax.grid(alpha=0.3, axis='y')

# -----------------------------------------------------------------------------
# Panel C: The CV formula
# -----------------------------------------------------------------------------
ax = axes[0, 2]

# Plot CV²(h²) = (2/(p+2)) × V_rel for different p
V_rel_range = np.linspace(0, 1, 100)

for p, color in [(2, PAL['dim1']), (5, PAL['dim2']), (10, PAL['dim3']), (20, PAL['Gstar'])]:
    CV_squared = (2 / (p + 2)) * V_rel_range
    CV = np.sqrt(CV_squared)
    ax.plot(V_rel_range, CV, color=color, linewidth=2, label=f'p = {p}')

ax.set_xlabel('V_rel (eigenvalue dispersion)')
ax.set_ylabel('CV(h²)')
ax.set_title('C. The universal formula\nCV²(h²) = 2V_rel / (p+2)', fontsize=11)
ax.legend(title='Dimensions')
ax.grid(alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 0.7)

# -----------------------------------------------------------------------------
# Panel D: The curse of dimensionality for alignment
# -----------------------------------------------------------------------------
ax = axes[1, 0]

from scipy import special

dimensions = [2, 3, 5, 10, 20, 50]
angles = np.linspace(0, 90, 100)

for p in dimensions:
    probs = []
    for theta_deg in angles:
        theta = np.radians(theta_deg)
        prob = special.betainc((p-1)/2, 0.5, np.sin(theta)**2)
        probs.append(prob)
    ax.plot(angles, probs, linewidth=2, label=f'p = {p}')

ax.set_xlabel('Angular tolerance from g_max (degrees)')
ax.set_ylabel('Probability β falls within tolerance')
ax.set_title('D. High dimensions: hard to find g_max\n(curse of dimensionality)', fontsize=11)
ax.legend(title='Dimensions', fontsize=8)
ax.grid(alpha=0.3)
ax.set_xlim(0, 90)
ax.set_ylim(0, 1)

# Annotation
ax.annotate('In 20D, only 0.1% within 30° of g_max!',
           xy=(30, 0.001), xytext=(50, 0.3),
           fontsize=9, arrowprops=dict(arrowstyle='->', color='black'))

# -----------------------------------------------------------------------------
# Panel E: What we can't visualize
# -----------------------------------------------------------------------------
ax = axes[1, 1]
ax.axis('off')

text = """
IN HIGH DIMENSIONS

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

We CAN'T visualize:
  • The hyperellipsoid itself
  • The (p-1)-sphere of directions
  • The h²(β) "landscape"

But we CAN compute:
  • All p eigenvalues of G*
  • Mean, min, max h²
  • V_rel and CV(h²)
  • h²(β) for any specific β

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The algebra handles what
our eyes cannot see.

But the INTUITION remains:
  • Eigenvalues = stretches
  • Eigenvectors = axes
  • h²(β) = ratio of lengths

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=10,
       ha='center', va='center', fontfamily='monospace',
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
ax.set_title('E. Beyond visualization', fontsize=11)

# -----------------------------------------------------------------------------
# Panel F: The complete picture
# -----------------------------------------------------------------------------
ax = axes[1, 2]
ax.axis('off')

summary = """
THE UNIFIED VIEW

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

In ANY dimension p:

1. G and P define hyperellipsoids

2. Eigenvalues λᵢ = length² along axis i

3. β'Mβ = Σᵢ λᵢ(β·vᵢ)²
        = weighted average of eigenvalues

4. h²(β) = β'Gβ / β'Pβ
        = ratio of two hyperellipsoid lengths

5. G* eigenvalues = h² along principal axes

6. V_rel measures eigenvalue spread

7. CV(h²) = √(2V_rel/(p+2))

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The formulas are IDENTICAL in every
dimension. Only the number of
eigenvalues changes!
"""

ax.text(0.5, 0.5, summary, transform=ax.transAxes, fontsize=10,
       ha='center', va='center', fontfamily='monospace',
       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3,
                edgecolor=PAL['high'], linewidth=2))
ax.set_title('F. The unified framework', fontsize=11)

plt.suptitle('LEVEL 4: The General Case — Hyperellipsoids in p Dimensions',
            fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('metric_pD.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved: metric_pD.png")
plt.close()

# =============================================================================
# FIGURE 5: ONE-PAGE SUMMARY
# =============================================================================

print("Creating Figure 5: One-Page Summary...")

fig = plt.figure(figsize=(16, 12))

# Big summary panel
ax = fig.add_subplot(111)
ax.axis('off')

summary = """
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                                                                             ┃
┃                         F R O M   1 D   T O   M A N Y   D I M E N S I O N S                                 ┃
┃                                                                                                             ┃
┃                              The Geometry of Covariance Matrices                                            ┃
┃                                                                                                             ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃                                                                                                             ┃
┃  DIMENSION        UNIT BALL          EIGENVALUES        h²(β)                  VISUALIZATION               ┃
┃  ─────────────────────────────────────────────────────────────────────────────────────────────────────────  ┃
┃                                                                                                             ┃
┃     1D            Interval           1 number           Single value           ◄────────►                   ┃
┃                   [-σ, σ]            λ = σ²             (no direction)         (just a line segment)        ┃
┃                                                                                                             ┃
┃     2D            Ellipse            2 numbers          Varies around          ⬭                            ┃
┃                                      λ₁, λ₂             the circle             (we can draw this)           ┃
┃                                                                                                             ┃
┃     3D            Ellipsoid          3 numbers          Varies over            🥚                           ┃
┃                                      λ₁, λ₂, λ₃         the sphere             (we can plot this)           ┃
┃                                                                                                             ┃
┃     pD            Hyperellipsoid     p numbers          Varies over            [...]                        ┃
┃                                      λ₁, ..., λₚ        the (p-1)-sphere       (we compute this)            ┃
┃                                                                                                             ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃                                                                                                             ┃
┃  THE INVARIANT STRUCTURE (SAME IN EVERY DIMENSION)                                                          ┃
┃  ────────────────────────────────────────────────────────────────────────────────────────────────────────── ┃
┃                                                                                                             ┃
┃     1. A covariance matrix Σ defines a METRIC (a way to measure length)                                     ┃
┃                                                                                                             ┃
┃     2. The EIGENVALUES λᵢ are the squared lengths along the principal axes                                  ┃
┃                                                                                                             ┃
┃     3. The EIGENVECTORS vᵢ are the directions of the principal axes                                         ┃
┃                                                                                                             ┃
┃     4. For ANY direction β:  β'Σβ = Σᵢ λᵢ (β·vᵢ)²  = weighted average of eigenvalues                        ┃
┃                                                                                                             ┃
┃     5. Directional heritability:  h²(β) = β'Gβ / β'Pβ  = ratio of two metrics                               ┃
┃                                                                                                             ┃
┃     6. The eigenvalues of G* are the h² values along principal axes                                         ┃
┃                                                                                                             ┃
┃     7. V_rel = Var(λ*) / Mean(λ*)²  measures how different the axes are                                     ┃
┃                                                                                                             ┃
┃     8. CV(h²) = √(2 V_rel / (p+2))  is the coefficient of variation of h²                                   ┃
┃                                                                                                             ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃                                                                                                             ┃
┃  THE PEDAGOGICAL INSIGHT                                                                                    ┃
┃  ──────────────────────────────────────────────────────────────────────────────────────────────────────────  ┃
┃                                                                                                             ┃
┃     • Start with 1D to see that variance IS a metric (a single stretch factor)                              ┃
┃                                                                                                             ┃
┃     • Move to 2D to see that direction matters (the ellipse is not a circle)                                ┃
┃                                                                                                             ┃
┃     • Move to 3D to see it's still the same geometry (now with 3 axes)                                      ┃
┃                                                                                                             ┃
┃     • Generalize to pD: the algebra handles dimensions we cannot visualize                                  ┃
┃                                                                                                             ┃
┃     • The INTUITION built in 2D and 3D TRANSFERS to any dimension!                                          ┃
┃                                                                                                             ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

                    "The geometry is always the same. Only the number of axes changes."
"""

ax.text(0.5, 0.5, summary, transform=ax.transAxes, fontsize=11,
       ha='center', va='center', fontfamily='monospace')

plt.savefig('metric_summary.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved: metric_summary.png")
plt.close()

print("\n" + "=" * 70)
print("ALL FIGURES COMPLETE!")
print("=" * 70)
print("""
Files created:
  • metric_1D.png  - The one-dimensional case (variance as metric)
  • metric_2D.png  - The two-dimensional case (ellipse geometry)
  • metric_3D.png  - The three-dimensional case (ellipsoid)
  • metric_pD.png  - The general case (hyperellipsoids)
  • metric_summary.png - One-page summary
""")
