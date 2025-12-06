#!/usr/bin/env python3
"""
Geometry First: A Visual Primer on Quantitative Genetics
=========================================================

This script creates a single-page visual summary that teaches
quantitative genetics starting from geometric intuition, not algebra.

The core message: Everything is about measuring length with different rulers.

Author: Daniel Ortiz-Barrientos & Claude
Date: December 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, FancyArrowPatch, Rectangle
from matplotlib.collections import LineCollection
import matplotlib.patheffects as pe

# =============================================================================
# COLOR PALETTE
# =============================================================================
PAL = {
    'euclidean': '#4A4A4A',
    'G': '#2E86AB',
    'P': '#B2182B',
    'Gstar': '#762A83',
    'beta': '#F18F01',
    'high': '#1B7837',
    'low': '#E63946',
    'neutral': '#878787',
    'background': '#FAFAFA',
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'figure.facecolor': 'white',
    'savefig.dpi': 300,
})

# =============================================================================
# THE GEOMETRY-FIRST SUMMARY
# =============================================================================

fig = plt.figure(figsize=(16, 20))

# Create a grid for the panels
gs = fig.add_gridspec(5, 3, hspace=0.35, wspace=0.25,
                     height_ratios=[1, 1, 1, 1, 0.8])

# =============================================================================
# ROW 1: The Fundamental Insight
# =============================================================================

# Title panel
ax_title = fig.add_subplot(gs[0, :])
ax_title.axis('off')

title_text = """
G E O M E T R Y   F I R S T
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A covariance matrix is a RULER — a way to measure length in different directions.
Everything in quantitative genetics is about comparing measurements from different rulers.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

ax_title.text(0.5, 0.5, title_text, transform=ax_title.transAxes,
             fontsize=14, ha='center', va='center', fontfamily='monospace')

# =============================================================================
# ROW 2: From Identity to Ellipse
# =============================================================================

# Panel 1: The Euclidean ruler (identity matrix)
ax = fig.add_subplot(gs[1, 0])

# Unit circle
circle = Circle((0, 0), 1, fill=False, edgecolor=PAL['euclidean'], 
               linewidth=3, linestyle='-')
ax.add_patch(circle)

# Draw unit vectors
for angle, label in [(0, 'e₁'), (90, 'e₂')]:
    rad = np.radians(angle)
    ax.annotate('', xy=(np.cos(rad)*0.9, np.sin(rad)*0.9), xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', color=PAL['euclidean'], lw=2))

ax.text(0, -1.5, 'I = Identity Matrix\n"The fair ruler"\nAll directions equal', 
       ha='center', fontsize=10)

ax.set_xlim(-1.8, 1.8)
ax.set_ylim(-1.8, 1.8)
ax.set_aspect('equal')
ax.axhline(0, color=PAL['neutral'], linewidth=0.3)
ax.axvline(0, color=PAL['neutral'], linewidth=0.3)
ax.set_title('1. Start: The Circle\n||x||² = x′Ix = x₁² + x₂²', fontsize=11)

# Panel 2: Stretching (diagonal matrix)
ax = fig.add_subplot(gs[1, 1])

# Stretched ellipse (diagonal matrix with different eigenvalues)
D = np.diag([0.5, 1.5])
eigvals = [0.5, 1.5]
width = 2 * np.sqrt(eigvals[1])
height = 2 * np.sqrt(eigvals[0])

ellipse = Ellipse((0, 0), width, height, angle=0,
                  fill=False, edgecolor=PAL['G'], linewidth=3)
ax.add_patch(ellipse)

# Show the stretch
ax.annotate('', xy=(0, np.sqrt(eigvals[0])), xytext=(0, 0),
           arrowprops=dict(arrowstyle='->', color=PAL['low'], lw=2))
ax.annotate('', xy=(np.sqrt(eigvals[1]), 0), xytext=(0, 0),
           arrowprops=dict(arrowstyle='->', color=PAL['high'], lw=2))

ax.text(np.sqrt(eigvals[1])+0.1, 0.1, f'√λ₁={np.sqrt(eigvals[1]):.2f}', 
       fontsize=9, color=PAL['high'])
ax.text(0.1, np.sqrt(eigvals[0])+0.1, f'√λ₂={np.sqrt(eigvals[0]):.2f}', 
       fontsize=9, color=PAL['low'])

ax.text(0, -1.5, 'D = Diagonal Matrix\n"Stretches along axes"\nEigenvalues = stretch²', 
       ha='center', fontsize=10)

ax.set_xlim(-1.8, 1.8)
ax.set_ylim(-1.8, 1.8)
ax.set_aspect('equal')
ax.axhline(0, color=PAL['neutral'], linewidth=0.3)
ax.axvline(0, color=PAL['neutral'], linewidth=0.3)
ax.set_title('2. Stretch: Eigenvalues\nDifferent lengths along axes', fontsize=11)

# Panel 3: Rotating (general covariance matrix)
ax = fig.add_subplot(gs[1, 2])

# Rotated ellipse
Sigma = np.array([[1.0, 0.6], [0.6, 0.8]])
eigvals, eigvecs = np.linalg.eigh(Sigma)
width = 2 * np.sqrt(eigvals[1])
height = 2 * np.sqrt(eigvals[0])
angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))

ellipse = Ellipse((0, 0), width, height, angle=angle,
                  fill=True, facecolor=PAL['G'], alpha=0.3,
                  edgecolor=PAL['G'], linewidth=3)
ax.add_patch(ellipse)

# Show eigenvectors
for i in range(2):
    v = eigvecs[:, i] * np.sqrt(eigvals[i])
    color = PAL['high'] if i == 1 else PAL['low']
    ax.annotate('', xy=v, xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', color=color, lw=2))

ax.text(0, -1.5, 'Σ = Covariance Matrix\n"Stretch + Rotate"\nEigenvectors = new axes', 
       ha='center', fontsize=10)

ax.set_xlim(-1.8, 1.8)
ax.set_ylim(-1.8, 1.8)
ax.set_aspect('equal')
ax.axhline(0, color=PAL['neutral'], linewidth=0.3)
ax.axvline(0, color=PAL['neutral'], linewidth=0.3)
ax.set_title('3. Rotate: Eigenvectors\nCovariance tilts the ellipse', fontsize=11)

# =============================================================================
# ROW 3: What Eigenvalues Mean
# =============================================================================

# Panel 4: Eigenvalues as directional variances
ax = fig.add_subplot(gs[2, 0])

# Same ellipse
ellipse = Ellipse((0, 0), width, height, angle=angle,
                  fill=True, facecolor=PAL['G'], alpha=0.2,
                  edgecolor=PAL['G'], linewidth=2)
ax.add_patch(ellipse)

# Show that eigenvalues = variance along eigenvectors
for i in range(2):
    v = eigvecs[:, i]
    lam = eigvals[i]
    color = PAL['high'] if i == 1 else PAL['low']
    
    # Draw direction
    ax.plot([0, v[0]*1.5], [0, v[1]*1.5], '-', color=color, linewidth=1.5)
    
    # Mark the variance
    ax.plot(v[0]*np.sqrt(lam), v[1]*np.sqrt(lam), 'o', color=color, 
           markersize=10, markeredgecolor='white', markeredgewidth=2)
    
    # Label
    ax.text(v[0]*np.sqrt(lam)*1.3, v[1]*np.sqrt(lam)*1.3, 
           f'λ={lam:.2f}', fontsize=9, color=color, fontweight='bold')

ax.text(0, -1.5, 'Eigenvalue λᵢ = variance\nalong eigenvector vᵢ\n(v\'Σv = λ)', 
       ha='center', fontsize=10)

ax.set_xlim(-1.8, 1.8)
ax.set_ylim(-1.8, 1.8)
ax.set_aspect('equal')
ax.axhline(0, color=PAL['neutral'], linewidth=0.3)
ax.axvline(0, color=PAL['neutral'], linewidth=0.3)
ax.set_title('4. λ = Length² Along Axis\nEigenvalues ARE directional variances', fontsize=11)

# Panel 5: Any direction interpolates
ax = fig.add_subplot(gs[2, 1])

# Ellipse
ellipse = Ellipse((0, 0), width, height, angle=angle,
                  fill=True, facecolor=PAL['G'], alpha=0.2,
                  edgecolor=PAL['G'], linewidth=2)
ax.add_patch(ellipse)

# Show a general direction
beta = np.array([0.8, 0.5])
beta = beta / np.linalg.norm(beta)

ax.annotate('', xy=beta*1.3, xytext=(0, 0),
           arrowprops=dict(arrowstyle='->', color=PAL['beta'], lw=3))
ax.text(beta[0]*1.4, beta[1]*1.4, 'β', fontsize=12, color=PAL['beta'], fontweight='bold')

# Compute β'Σβ
var_beta = beta @ Sigma @ beta
ax.text(0.5, 0.2, f'β\'Σβ = {var_beta:.2f}', fontsize=10, color=PAL['beta'],
       transform=ax.transAxes)

# Show projections onto eigenvectors
for i in range(2):
    v = eigvecs[:, i]
    proj = np.dot(beta, v)
    color = PAL['high'] if i == 1 else PAL['low']
    ax.plot([0, v[0]*np.abs(proj)], [0, v[1]*np.abs(proj)], '--', 
           color=color, linewidth=1.5, alpha=0.5)

ax.text(0, -1.5, 'β\'Σβ = Σᵢ λᵢ(β·vᵢ)²\n"Weighted average of λ\'s"\nWeights = alignment²', 
       ha='center', fontsize=10)

ax.set_xlim(-1.8, 1.8)
ax.set_ylim(-1.8, 1.8)
ax.set_aspect('equal')
ax.axhline(0, color=PAL['neutral'], linewidth=0.3)
ax.axvline(0, color=PAL['neutral'], linewidth=0.3)
ax.set_title('5. Any Direction Interpolates\nβ\'Σβ blends eigenvalues by alignment', fontsize=11)

# Panel 6: V_rel measures eccentricity
ax = fig.add_subplot(gs[2, 2])

# Show two ellipses: one circular, one elongated
# Circular
Sigma_sphere = np.eye(2) * 0.7
circle = Circle((0, 0.9), np.sqrt(0.7), fill=False, 
               edgecolor=PAL['high'], linewidth=2.5)
ax.add_patch(circle)
ax.text(0, 0.9, 'V_rel ≈ 0\n"Sphere"', ha='center', va='center', fontsize=9)

# Elongated
Sigma_elong = np.diag([0.2, 1.2])
ellipse_elong = Ellipse((0, -0.9), 2*np.sqrt(1.2), 2*np.sqrt(0.2), angle=0,
                        fill=False, edgecolor=PAL['low'], linewidth=2.5)
ax.add_patch(ellipse_elong)
ax.text(0, -0.9, 'V_rel >> 0\n"Cigar"', ha='center', va='center', fontsize=9)

ax.text(0, -2.0, 'V_rel = Var(λ)/Mean(λ)²\n"How different are the stretches?"', 
       ha='center', fontsize=10)

ax.set_xlim(-2, 2)
ax.set_ylim(-2.3, 2)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('6. V_rel = Eccentricity\nHow much does length vary with direction?', fontsize=11)

# =============================================================================
# ROW 4: Comparing Two Rulers
# =============================================================================

# Panel 7: Two ellipses (G and P)
ax = fig.add_subplot(gs[3, 0])

G = np.array([[0.6, 0.2], [0.2, 0.4]])
P = G + np.array([[0.4, 0.1], [0.1, 0.5]])

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

ax.legend(loc='lower right', fontsize=9)
ax.text(0, -1.6, 'G = genetic ruler\nP = phenotypic ruler\n(P = G + E, always bigger)', 
       ha='center', fontsize=10)

ax.set_xlim(-1.8, 1.8)
ax.set_ylim(-1.8, 1.8)
ax.set_aspect('equal')
ax.axhline(0, color=PAL['neutral'], linewidth=0.3)
ax.axvline(0, color=PAL['neutral'], linewidth=0.3)
ax.set_title('7. Two Rulers\nG (genetic) and P (phenotypic)', fontsize=11)

# Panel 8: h² as ratio
ax = fig.add_subplot(gs[3, 1])

# Same ellipses
for M, color, ls in [(P, PAL['P'], '--'), (G, PAL['G'], '-')]:
    ev, evec = np.linalg.eigh(M)
    w = 2 * np.sqrt(ev[1])
    h = 2 * np.sqrt(ev[0])
    ang = np.degrees(np.arctan2(evec[1, 1], evec[0, 1]))
    
    fill = (M is G)
    ellipse = Ellipse((0, 0), w, h, angle=ang,
                     fill=fill, facecolor=color if fill else 'none',
                     alpha=0.2 if fill else 1.0,
                     edgecolor=color, linewidth=2, linestyle=ls)
    ax.add_patch(ellipse)

# Show a direction
beta = np.array([0.7, 0.7])
beta = beta / np.linalg.norm(beta)

ax.annotate('', xy=beta*1.4, xytext=(0, 0),
           arrowprops=dict(arrowstyle='->', color=PAL['beta'], lw=3))

# Compute h²
h2 = (beta @ G @ beta) / (beta @ P @ beta)

ax.text(0.5, 0.85, f'h²(β) = {h2:.2f}', fontsize=12, color=PAL['beta'],
       transform=ax.transAxes, fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

ax.text(0, -1.6, 'h²(β) = β\'Gβ / β\'Pβ\n"G-length² / P-length²"\nCompare the two rulers!', 
       ha='center', fontsize=10)

ax.set_xlim(-1.8, 1.8)
ax.set_ylim(-1.8, 1.8)
ax.set_aspect('equal')
ax.axhline(0, color=PAL['neutral'], linewidth=0.3)
ax.axvline(0, color=PAL['neutral'], linewidth=0.3)
ax.set_title('8. h²(β) = Ruler Ratio\nGenetic length² ÷ Phenotypic length²', fontsize=11)

# Panel 9: P-whitening
ax = fig.add_subplot(gs[3, 2])

# After whitening: P = I (circle), G* is the constraint
eigvals_P, eigvecs_P = np.linalg.eigh(P)
P_inv_sqrt = eigvecs_P @ np.diag(1/np.sqrt(eigvals_P)) @ eigvecs_P.T
Gstar = P_inv_sqrt @ G @ P_inv_sqrt

# P becomes unit circle
circle = Circle((0, 0), 1, fill=False, edgecolor=PAL['P'], 
               linewidth=2.5, linestyle='--', label='P* = I')
ax.add_patch(circle)

# G* ellipse
ev_Gstar, evec_Gstar = np.linalg.eigh(Gstar)
w = 2 * np.sqrt(ev_Gstar[1])
h = 2 * np.sqrt(ev_Gstar[0])
ang = np.degrees(np.arctan2(evec_Gstar[1, 1], evec_Gstar[0, 1]))

ellipse_Gstar = Ellipse((0, 0), w, h, angle=ang,
                        fill=True, facecolor=PAL['Gstar'], alpha=0.3,
                        edgecolor=PAL['Gstar'], linewidth=2.5, label='G*')
ax.add_patch(ellipse_Gstar)

# Label eigenvalues
for i in range(2):
    v = evec_Gstar[:, i]
    lam = ev_Gstar[i]
    ax.text(v[0]*1.3, v[1]*1.3, f'λ*={lam:.2f}\n=h² along axis', 
           fontsize=8, ha='center', va='center',
           color=PAL['high'] if i == 1 else PAL['low'])

ax.legend(loc='lower right', fontsize=9)
ax.text(0, -1.6, 'G* = P⁻¹ᐟ² G P⁻¹ᐟ²\nP becomes unit circle\nG* eigenvalues = h² values!', 
       ha='center', fontsize=10)

ax.set_xlim(-1.8, 1.8)
ax.set_ylim(-1.8, 1.8)
ax.set_aspect('equal')
ax.axhline(0, color=PAL['neutral'], linewidth=0.3)
ax.axvline(0, color=PAL['neutral'], linewidth=0.3)
ax.set_title('9. P-Whitening\nMake P the "fair ruler", G* shows constraint', fontsize=11)

# =============================================================================
# ROW 5: The Punchline
# =============================================================================

ax_summary = fig.add_subplot(gs[4, :])
ax_summary.axis('off')

summary_text = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE GEOMETRIC VOCABULARY OF QUANTITATIVE GENETICS

    Matrix Σ         →  A ruler (way to measure length)
    Eigenvalue λᵢ    →  How much the ruler stretches axis i
    Eigenvector vᵢ   →  The direction of axis i
    β′Σβ             →  Length² of β measured by this ruler
    V_rel            →  How "unfair" the ruler is (eccentric vs round)
    CV(h²)           →  How much h² varies with direction = √(2Vrel/(p+2))

    G matrix         →  The genetic ruler
    P matrix         →  The phenotypic ruler
    h²(β)            →  Ratio of ruler measurements in direction β
    G*               →  G expressed in "P-units" (P-whitened)
    λ*(G*)           →  Directional h² along principal axes

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"Once you see it as geometry, you never unsee it."
"""

ax_summary.text(0.5, 0.5, summary_text, transform=ax_summary.transAxes,
               fontsize=11, ha='center', va='center', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9,
                        edgecolor=PAL['G'], linewidth=2))

plt.savefig('geometry_first_summary.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved: geometry_first_summary.png")
plt.close()

# =============================================================================
# THE HISTORICAL TRAGEDY
# =============================================================================

print("""
═══════════════════════════════════════════════════════════════════════════════
WHY ISN'T IT TAUGHT THIS WAY?
═══════════════════════════════════════════════════════════════════════════════

The tragedy of linear algebra education:

1. HISTORICAL: Linear algebra was developed for solving systems of equations.
   Matrices were "arrays of numbers" before they were "transformations."
   
2. COMPUTATIONAL: Algebraic rules (row reduction, determinants) are easier
   to test on exams than geometric intuition.
   
3. ABSTRACT: Mathematicians prize abstraction. "A matrix is just a linear
   map" is true but hides the geometric soul.
   
4. DISCIPLINARY SILOS: Geometry belongs to geometers, algebra to algebraists.
   The unification (geometric algebra) is rarely taught.

THE RESULT:
   Students learn to multiply matrices without knowing they're composing
   transformations. They compute eigenvalues without knowing they're finding
   the natural axes of a deformation. They invert matrices without knowing
   they're reversing a transformation.

FOR QUANTITATIVE GENETICS:
   Students learn G = VA, P = G + E, h² = VA/VP as formulas.
   They compute eigendecompositions as "a technique."
   They never see that G and P are TWO DIFFERENT WAYS OF MEASURING LENGTH
   and h² asks: "how much of the P-length is explained by the G-length?"

THE FIX:
   Start with circles and ellipses.
   Show that matrices DEFORM space.
   Reveal eigenvalues as "how much deformation along each axis."
   THEN introduce the algebraic machinery as a way to COMPUTE these things.

   Geometry first. Algebra as servant.

═══════════════════════════════════════════════════════════════════════════════
""")
