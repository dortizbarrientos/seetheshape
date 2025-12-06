#!/usr/bin/env python3
"""
Publication-Ready Pedagogical Figures: Introduction to Directional Heritability
================================================================================

A visual journey from first principles to constraint traps, designed for
presentations and teaching. Each figure builds on the previous, creating
a complete conceptual framework.

Figure 1: What is Directional Heritability?
Figure 2: The Geometry of Quadratic Forms  
Figure 3: P-Whitening and the G* Matrix
Figure 4: From Eigenvalues to CV(h²)
Figure 5: Evolvability vs Directional Heritability
Figure 6: Constraint Traps Revealed
Figure 7: The Complete Picture

Author: Daniel Ortiz-Barrientos
Date: December 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, Circle, Wedge
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe
from scipy import stats
import os

# =============================================================================
# STYLE CONFIGURATION
# =============================================================================

# Color palette - sophisticated, publication-ready
COLORS = {
    'G_ellipse': '#2166AC',       # Deep blue for G
    'P_ellipse': '#B2182B',       # Deep red for P  
    'Gstar': '#762A83',           # Purple for G*
    'selection': '#D6604D',       # Coral for β
    'response': '#1B7837',        # Forest green for response
    'high_h2': '#1B7837',         # Green for high heritability
    'low_h2': '#B2182B',          # Red for low heritability
    'neutral': '#878787',         # Gray
    'accent': '#F4A582',          # Peach accent
    'background': '#FAFAFA',      # Off-white background
    'text': '#2D2D2D',            # Dark gray text
}

# Create custom colormaps
h2_cmap = LinearSegmentedColormap.from_list(
    'h2_cmap',
    ['#B2182B', '#F4A582', '#FDDBC7', '#D1E5F0', '#67A9CF', '#2166AC']
)

# Figure settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.linewidth': 1.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
})


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def make_ellipse_points(center, width, height, angle, n_points=100):
    """Generate points along an ellipse."""
    t = np.linspace(0, 2*np.pi, n_points)
    cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    x = center[0] + width/2 * np.cos(t) * cos_a - height/2 * np.sin(t) * sin_a
    y = center[1] + width/2 * np.cos(t) * sin_a + height/2 * np.sin(t) * cos_a
    return x, y


def get_ellipse_params(matrix):
    """Get ellipse parameters (width, height, angle) from 2x2 matrix."""
    eigvals, eigvecs = np.linalg.eigh(matrix)
    # Sort by eigenvalue (largest first)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    width = 2 * np.sqrt(eigvals[0])
    height = 2 * np.sqrt(eigvals[1])
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    
    return width, height, angle, eigvals, eigvecs


def directional_h2(beta, G, P):
    """Compute directional heritability."""
    return float(beta @ G @ beta) / float(beta @ P @ beta)


def evolvability(beta, G):
    """Compute evolvability."""
    return float(beta @ G @ beta) / float(beta @ beta)


def add_text_box(ax, text, x, y, fontsize=11, ha='center', va='center',
                 bbox_color='white', alpha=0.9):
    """Add text with a nice background box."""
    ax.text(x, y, text, transform=ax.transAxes, fontsize=fontsize,
            ha=ha, va=va, 
            bbox=dict(boxstyle='round,pad=0.4', facecolor=bbox_color, 
                     alpha=alpha, edgecolor='none'))


# =============================================================================
# FIGURE 1: WHAT IS DIRECTIONAL HERITABILITY?
# =============================================================================

def figure_1_what_is_dirh2(save_path=None):
    """
    Figure 1: Introduction to directional heritability.
    
    Shows that heritability depends on which direction you measure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Create example G matrix (anisotropic)
    G = np.array([[1.0, 0.4], [0.4, 0.3]])
    P = np.array([[1.5, 0.4], [0.4, 0.8]])
    
    # Panel A: The concept - different directions, different h²
    ax = axes[0]
    
    # Draw coordinate system
    ax.axhline(0, color=COLORS['neutral'], linewidth=0.5, zorder=0)
    ax.axvline(0, color=COLORS['neutral'], linewidth=0.5, zorder=0)
    
    # Draw G ellipse
    w, h, angle, _, _ = get_ellipse_params(G)
    ellipse_G = Ellipse((0, 0), w, h, angle=angle, 
                        fill=False, edgecolor=COLORS['G_ellipse'], 
                        linewidth=3, label='G (genetic)', linestyle='-')
    ax.add_patch(ellipse_G)
    
    # Draw P ellipse
    w, h, angle, _, _ = get_ellipse_params(P)
    ellipse_P = Ellipse((0, 0), w, h, angle=angle,
                        fill=False, edgecolor=COLORS['P_ellipse'],
                        linewidth=3, label='P (phenotypic)', linestyle='--')
    ax.add_patch(ellipse_P)
    
    # Show different selection directions with their h²
    angles = [0, 45, 90, 135]
    for theta in angles:
        rad = np.radians(theta)
        beta = np.array([np.cos(rad), np.sin(rad)])
        h2 = directional_h2(beta, G, P)
        
        # Arrow length proportional to h²
        length = 0.8 + 0.8 * h2
        color = plt.cm.RdYlGn(h2)  # Color by h²
        
        ax.annotate('', xy=(length*np.cos(rad), length*np.sin(rad)),
                   xytext=(0, 0),
                   arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
        
        # Label with h² value
        label_r = length + 0.25
        ax.text(label_r*np.cos(rad), label_r*np.sin(rad), 
               f'h²={h2:.2f}', fontsize=10, ha='center', va='center',
               color=color, fontweight='bold')
    
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlabel('Trait 1')
    ax.set_ylabel('Trait 2')
    ax.set_title('A. Heritability depends on direction')
    
    # Panel B: The formula
    ax = axes[1]
    ax.axis('off')
    
    # Main formula
    ax.text(0.5, 0.75, r'$h^2(\boldsymbol{\beta}) = \frac{\boldsymbol{\beta}^\mathsf{T} \mathbf{G} \boldsymbol{\beta}}{\boldsymbol{\beta}^\mathsf{T} \mathbf{P} \boldsymbol{\beta}}$',
           fontsize=28, ha='center', va='center', transform=ax.transAxes)
    
    # Explanation
    explanation = (
        "Directional heritability measures\n"
        "the proportion of phenotypic variance\n"
        "in direction β that is genetic.\n\n"
        "• G = genetic covariance matrix\n"
        "• P = phenotypic covariance matrix\n"
        "• β = selection gradient (direction)"
    )
    ax.text(0.5, 0.35, explanation, fontsize=12, ha='center', va='center',
           transform=ax.transAxes, linespacing=1.5)
    
    ax.set_title('B. The definition')
    
    # Panel C: Polar plot of h² around the circle
    ax = axes[2]
    
    # Compute h² for all directions
    n_angles = 180
    thetas = np.linspace(0, 2*np.pi, n_angles)
    h2_values = []
    
    for theta in thetas:
        beta = np.array([np.cos(theta), np.sin(theta)])
        h2_values.append(directional_h2(beta, G, P))
    
    h2_values = np.array(h2_values)
    
    # Create polar-style plot in Cartesian coordinates
    # Scale radius by h²
    r = 0.3 + 0.7 * h2_values  # Scale to visible range
    x = r * np.cos(thetas)
    y = r * np.sin(thetas)
    
    # Color by h²
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    norm = plt.Normalize(h2_values.min(), h2_values.max())
    lc = LineCollection(segments, cmap='RdYlGn', norm=norm, linewidth=4)
    lc.set_array(h2_values[:-1])
    ax.add_collection(lc)
    
    # Add reference circle for mean h²
    mean_h2 = np.mean(h2_values)
    circle = Circle((0, 0), 0.3 + 0.7*mean_h2, fill=False, 
                   color=COLORS['neutral'], linestyle=':', linewidth=1.5,
                   label=f'Mean h² = {mean_h2:.2f}')
    ax.add_patch(circle)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, label='h²(β)')
    
    # Mark max and min
    max_idx = np.argmax(h2_values)
    min_idx = np.argmin(h2_values)
    
    ax.plot(x[max_idx], y[max_idx], 'o', color=COLORS['high_h2'], 
           markersize=12, markeredgecolor='white', markeredgewidth=2,
           label=f'Max: {h2_values[max_idx]:.2f}')
    ax.plot(x[min_idx], y[min_idx], 'o', color=COLORS['low_h2'],
           markersize=12, markeredgecolor='white', markeredgewidth=2,
           label=f'Min: {h2_values[min_idx]:.2f}')
    
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlabel('Trait 1 direction')
    ax.set_ylabel('Trait 2 direction')
    ax.set_title('C. h² varies continuously with direction')
    ax.axhline(0, color=COLORS['neutral'], linewidth=0.5, zorder=0)
    ax.axvline(0, color=COLORS['neutral'], linewidth=0.5, zorder=0)
    
    plt.suptitle('Figure 1: What is Directional Heritability?', 
                fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# FIGURE 2: THE GEOMETRY OF QUADRATIC FORMS
# =============================================================================

def figure_2_quadratic_forms(save_path=None):
    """
    Figure 2: Understanding h² as a ratio of quadratic forms.
    
    Shows how G and P define ellipsoids, and h² is their ratio.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Example matrices
    G = np.array([[1.0, 0.3], [0.3, 0.4]])
    P = np.array([[1.5, 0.3], [0.3, 0.9]])
    
    # Panel A: The numerator - β'Gβ
    ax = axes[0]
    
    # Draw G ellipse (level set of β'Gβ = 1)
    w, h, angle, eigvals_G, eigvecs_G = get_ellipse_params(G)
    
    # Draw multiple level sets
    for level in [0.5, 1.0, 1.5, 2.0]:
        scale = np.sqrt(level)
        ellipse = Ellipse((0, 0), w*scale, h*scale, angle=angle,
                         fill=False, edgecolor=COLORS['G_ellipse'],
                         linewidth=2, alpha=0.3 + 0.2*level)
        ax.add_patch(ellipse)
    
    # Highlight the unit ellipse
    ellipse_unit = Ellipse((0, 0), w, h, angle=angle,
                          fill=False, edgecolor=COLORS['G_ellipse'],
                          linewidth=3, label=r"$\boldsymbol{\beta}'\mathbf{G}\boldsymbol{\beta} = 1$")
    ax.add_patch(ellipse_unit)
    
    # Draw eigenvectors
    for i in range(2):
        ev = eigvecs_G[:, i] * np.sqrt(eigvals_G[i]) * 1.2
        ax.annotate('', xy=ev, xytext=(0, 0),
                   arrowprops=dict(arrowstyle='->', color=COLORS['G_ellipse'], lw=2))
        ax.annotate('', xy=-ev, xytext=(0, 0),
                   arrowprops=dict(arrowstyle='->', color=COLORS['G_ellipse'], lw=2))
    
    # Show a test vector
    beta = np.array([0.8, 0.6])
    beta = beta / np.linalg.norm(beta) * 1.3
    ax.annotate('', xy=beta, xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', color=COLORS['selection'], lw=3))
    
    qf_G = beta @ G @ beta
    ax.text(beta[0]+0.15, beta[1]+0.1, f'β\'Gβ = {qf_G:.2f}', 
           fontsize=11, color=COLORS['selection'], fontweight='bold')
    
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlabel('Trait 1')
    ax.set_ylabel('Trait 2')
    ax.set_title(r'A. Numerator: $\boldsymbol{\beta}^\mathsf{T}\mathbf{G}\boldsymbol{\beta}$ (genetic variance)')
    ax.axhline(0, color=COLORS['neutral'], linewidth=0.5, zorder=0)
    ax.axvline(0, color=COLORS['neutral'], linewidth=0.5, zorder=0)
    
    # Panel B: The denominator - β'Pβ
    ax = axes[1]
    
    w, h, angle, eigvals_P, eigvecs_P = get_ellipse_params(P)
    
    # Draw multiple level sets
    for level in [0.5, 1.0, 1.5, 2.0]:
        scale = np.sqrt(level)
        ellipse = Ellipse((0, 0), w*scale, h*scale, angle=angle,
                         fill=False, edgecolor=COLORS['P_ellipse'],
                         linewidth=2, alpha=0.3 + 0.2*level)
        ax.add_patch(ellipse)
    
    # Highlight the unit ellipse
    ellipse_unit = Ellipse((0, 0), w, h, angle=angle,
                          fill=False, edgecolor=COLORS['P_ellipse'],
                          linewidth=3, linestyle='--',
                          label=r"$\boldsymbol{\beta}'\mathbf{P}\boldsymbol{\beta} = 1$")
    ax.add_patch(ellipse_unit)
    
    # Same test vector
    ax.annotate('', xy=beta, xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', color=COLORS['selection'], lw=3))
    
    qf_P = beta @ P @ beta
    ax.text(beta[0]+0.15, beta[1]+0.1, f'β\'Pβ = {qf_P:.2f}', 
           fontsize=11, color=COLORS['selection'], fontweight='bold')
    
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlabel('Trait 1')
    ax.set_ylabel('Trait 2')
    ax.set_title(r'B. Denominator: $\boldsymbol{\beta}^\mathsf{T}\mathbf{P}\boldsymbol{\beta}$ (phenotypic variance)')
    ax.axhline(0, color=COLORS['neutral'], linewidth=0.5, zorder=0)
    ax.axvline(0, color=COLORS['neutral'], linewidth=0.5, zorder=0)
    
    # Panel C: The ratio
    ax = axes[2]
    
    # Draw both ellipses overlaid
    w_G, h_G, angle_G, _, _ = get_ellipse_params(G)
    w_P, h_P, angle_P, _, _ = get_ellipse_params(P)
    
    ellipse_G = Ellipse((0, 0), w_G, h_G, angle=angle_G,
                        fill=True, facecolor=COLORS['G_ellipse'], alpha=0.3,
                        edgecolor=COLORS['G_ellipse'], linewidth=3,
                        label='G')
    ellipse_P = Ellipse((0, 0), w_P, h_P, angle=angle_P,
                        fill=False, edgecolor=COLORS['P_ellipse'],
                        linewidth=3, linestyle='--', label='P')
    ax.add_patch(ellipse_G)
    ax.add_patch(ellipse_P)
    
    # Show the same vector
    ax.annotate('', xy=beta, xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', color=COLORS['selection'], lw=3))
    
    h2 = qf_G / qf_P
    
    # Result box
    result_text = (f"h²(β) = {qf_G:.2f} / {qf_P:.2f}\n"
                  f"     = {h2:.2f}")
    ax.text(0.95, 0.95, result_text, transform=ax.transAxes,
           fontsize=14, ha='right', va='top', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                    edgecolor=COLORS['selection']))
    
    # Add interpretation
    ax.text(0.5, 0.05, 
           f'{h2*100:.0f}% of phenotypic variance\nin this direction is genetic',
           transform=ax.transAxes, ha='center', va='bottom',
           fontsize=11, style='italic')
    
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlabel('Trait 1')
    ax.set_ylabel('Trait 2')
    ax.set_title(r'C. The ratio: $h^2(\boldsymbol{\beta})$')
    ax.axhline(0, color=COLORS['neutral'], linewidth=0.5, zorder=0)
    ax.axvline(0, color=COLORS['neutral'], linewidth=0.5, zorder=0)
    
    plt.suptitle('Figure 2: Directional Heritability as a Ratio of Quadratic Forms',
                fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# FIGURE 3: P-WHITENING AND G*
# =============================================================================

def figure_3_whitening(save_path=None):
    """
    Figure 3: The P-whitening transformation reveals constraint structure.
    
    Shows how transforming to G* makes the geometry interpretable.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Example with correlated G and P
    G = np.array([[1.0, 0.5], [0.5, 0.4]])
    P = np.array([[1.8, 0.6], [0.6, 1.0]])
    
    # Compute G*
    eigvals_P, eigvecs_P = np.linalg.eigh(P)
    P_inv_sqrt = eigvecs_P @ np.diag(1/np.sqrt(eigvals_P)) @ eigvecs_P.T
    Gstar = P_inv_sqrt @ G @ P_inv_sqrt
    
    # Panel A: Original space
    ax = axes[0]
    
    w_G, h_G, angle_G, _, _ = get_ellipse_params(G)
    w_P, h_P, angle_P, _, _ = get_ellipse_params(P)
    
    # Draw both ellipses
    ellipse_G = Ellipse((0, 0), w_G, h_G, angle=angle_G,
                        fill=True, facecolor=COLORS['G_ellipse'], alpha=0.3,
                        edgecolor=COLORS['G_ellipse'], linewidth=3, label='G')
    ellipse_P = Ellipse((0, 0), w_P, h_P, angle=angle_P,
                        fill=False, edgecolor=COLORS['P_ellipse'],
                        linewidth=3, linestyle='--', label='P')
    ax.add_patch(ellipse_P)
    ax.add_patch(ellipse_G)
    
    # Show a selection direction
    beta = np.array([0.6, 0.8])
    beta = beta / np.linalg.norm(beta)
    ax.annotate('', xy=beta*1.5, xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', color=COLORS['selection'], lw=3))
    ax.text(beta[0]*1.6, beta[1]*1.6, 'β', fontsize=14, 
           color=COLORS['selection'], fontweight='bold')
    
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1.8, 1.8)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlabel('Trait 1')
    ax.set_ylabel('Trait 2')
    ax.set_title('A. Original trait space')
    ax.axhline(0, color=COLORS['neutral'], linewidth=0.5, zorder=0)
    ax.axvline(0, color=COLORS['neutral'], linewidth=0.5, zorder=0)
    
    # Add info box
    ax.text(0.95, 0.05, 'G and P have\ndifferent shapes\nand orientations',
           transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Panel B: Transformation arrow
    ax = axes[1]
    ax.axis('off')
    
    # Big transformation arrow
    ax.annotate('', xy=(0.85, 0.5), xytext=(0.15, 0.5),
               arrowprops=dict(arrowstyle='->', color=COLORS['Gstar'], 
                              lw=4, mutation_scale=30),
               transform=ax.transAxes)
    
    # Transformation formula
    ax.text(0.5, 0.72, r'$\mathbf{G}^* = \mathbf{P}^{-1/2} \mathbf{G} \mathbf{P}^{-1/2}$',
           fontsize=20, ha='center', va='center', transform=ax.transAxes,
           color=COLORS['Gstar'], fontweight='bold')
    
    ax.text(0.5, 0.5, 'P-whitening\ntransformation',
           fontsize=14, ha='center', va='center', transform=ax.transAxes)
    
    # Explanation
    explanation = (
        "After whitening:\n"
        "• P becomes the identity (a circle)\n"
        "• G* captures all constraint information\n"
        "• Eigenvalues of G* = directional h²\n"
        "  along principal axes"
    )
    ax.text(0.5, 0.2, explanation, fontsize=11, ha='center', va='center',
           transform=ax.transAxes, linespacing=1.5,
           bbox=dict(boxstyle='round', facecolor=COLORS['background'], alpha=0.8))
    
    ax.set_title('B. The transformation')
    
    # Panel C: Whitened space
    ax = axes[2]
    
    # P becomes identity (circle)
    circle_P = Circle((0, 0), 1, fill=False, edgecolor=COLORS['P_ellipse'],
                      linewidth=3, linestyle='--', label='P* = I (circle)')
    ax.add_patch(circle_P)
    
    # G* ellipse
    w_Gstar, h_Gstar, angle_Gstar, eigvals_Gstar, eigvecs_Gstar = get_ellipse_params(Gstar)
    
    ellipse_Gstar = Ellipse((0, 0), w_Gstar, h_Gstar, angle=angle_Gstar,
                           fill=True, facecolor=COLORS['Gstar'], alpha=0.3,
                           edgecolor=COLORS['Gstar'], linewidth=3, label='G*')
    ax.add_patch(ellipse_Gstar)
    
    # Draw eigenvectors with eigenvalue labels
    colors = [COLORS['high_h2'], COLORS['low_h2']]
    for i in range(2):
        ev = eigvecs_Gstar[:, 1-i] * 1.3  # Reverse order (largest first)
        eigval = eigvals_Gstar[1-i]
        ax.annotate('', xy=ev, xytext=(0, 0),
                   arrowprops=dict(arrowstyle='->', color=colors[i], lw=3))
        ax.text(ev[0]*1.15, ev[1]*1.15, f'λ*={eigval:.2f}\n(h² along axis)', 
               fontsize=9, ha='center', va='center', color=colors[i],
               fontweight='bold')
    
    # Transform and show the selection direction
    beta_star = P_inv_sqrt @ beta
    beta_star = beta_star / np.linalg.norm(beta_star)
    ax.annotate('', xy=beta_star*1.3, xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', color=COLORS['selection'], lw=3))
    ax.text(beta_star[0]*1.4, beta_star[1]*1.4, 'β*', fontsize=14,
           color=COLORS['selection'], fontweight='bold')
    
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlabel('Whitened Trait 1')
    ax.set_ylabel('Whitened Trait 2')
    ax.set_title('C. P-whitened space')
    ax.axhline(0, color=COLORS['neutral'], linewidth=0.5, zorder=0)
    ax.axvline(0, color=COLORS['neutral'], linewidth=0.5, zorder=0)
    
    # Add key insight
    ax.text(0.5, 0.05,
           'Directional heritability is\nencoded in G* eigenvalues',
           transform=ax.transAxes, ha='center', va='bottom',
           fontsize=11, style='italic')
    
    plt.suptitle('Figure 3: Whitening P Reveals the Geometry of Constraints',
                fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# FIGURE 4: EIGENVALUES AND CV(h²)
# =============================================================================

def figure_4_eigenvalues_and_cv(save_path=None):
    """
    Figure 4: How eigenvalue dispersion controls the spread of directional heritability.
    
    Shows the theoretical and simulated relationship between eigenvalue dispersion
    and CV(h²).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: Theoretical relationship (schematic)
    ax = axes[0]
    # Simple schematic curve: CV(h²) vs V_rel(G*)
    x = np.linspace(0, 0.99, 200)
    # Use Watanabe / Ortiz-Barrientos relationship: CV^2 ≈ (2/(p+2)) * V_rel
    p_vals = [3, 5, 10]
    colors = ['#1b9e77', '#d95f02', '#7570b3']
    for p, c in zip(p_vals, colors):
        cv2 = 2/(p+2) * x
        cv = np.sqrt(cv2)
        ax.plot(x, cv, color=c, label=f'p={p}')
    
    ax.set_xlabel(r'Relative eigenvalue variance $V_{\mathrm{rel}}(\mathbf{G}^*)$')
    ax.set_ylabel(r'Coefficient of variation $\mathrm{CV}(h^2)$')
    ax.set_title('A. Theory: eigenvalue dispersion shapes CV(h²)')
    ax.legend(title='Dimension', fontsize=9)
    ax.grid(alpha=0.2)
    
    # Panel B: Simulation example
    ax = axes[1]
    
    np.random.seed(42)
    
    def random_spd_matrix(p, v_rel_target):
        """
        Generate a random SPD matrix with approximate relative eigenvalue variance v_rel_target.
        We use a simple construction: one large eigenvalue and (p-1) smaller ones.
        """
        # Choose eigenvalues
        # Let lambda_1 = a, others = b
        # Then V_rel = Var(lambda) / mean(lambda)^2
        # With p-1 equal small eigenvalues.
        a = 1.0 + 2.0 * v_rel_target
        b = 1.0 - 0.5 * v_rel_target
        lambdas = np.array([a] + [b]*(p-1))
        # Random orthogonal matrix
        Q, _ = np.linalg.qr(np.random.randn(p, p))
        return Q @ np.diag(lambdas) @ Q.T
    
    def simulate_cv_h2(p, n_points=1000):
        """Simulate CV(h²) for random SPD matrix and random directions."""
        v_rel_values = np.linspace(0, 0.9, 10)
        cv_values = []
        
        for v_rel in v_rel_values:
            Gstar = random_spd_matrix(p, v_rel)
            P = np.eye(p)
            hs = []
            for _ in range(n_points):
                beta = np.random.randn(p)
                beta /= np.linalg.norm(beta)
                hs.append(directional_h2(beta, Gstar, P))
            hs = np.array(hs)
            cv = np.std(hs) / np.mean(hs)
            cv_values.append(cv)
        
        return v_rel_values, np.array(cv_values)
    
    p = 5
    v_rel_sim, cv_sim = simulate_cv_h2(p=p, n_points=2000)
    
    ax.plot(v_rel_sim, cv_sim, 'o-', color=COLORS['response'],
           label=f'Simulation (p={p})')
    
    # Add rough theoretical line for that p
    cv2_theory = 2/(p+2) * v_rel_sim
    cv_theory = np.sqrt(cv2_theory)
    ax.plot(v_rel_sim, cv_theory, '--', color=COLORS['neutral'],
           label='Theory (approx.)')
    
    ax.set_xlabel(r'Relative eigenvalue variance $V_{\mathrm{rel}}(\mathbf{G}^*)$')
    ax.set_ylabel(r'$\mathrm{CV}(h^2)$ from simulations')
    ax.set_title('B. Simulated CV(h²) vs eigenvalue dispersion')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    
    plt.suptitle('Figure 4: Eigenvalue Structure Controls the Spread of Directional Heritability',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# FIGURE 5: EVOLVABILITY VS DIRECTIONAL HERITABILITY
# =============================================================================

def figure_5_evolvability_vs_h2(save_path=None):
    """
    Figure 5: Compare directional evolvability and directional heritability.
    
    Shows how these metrics can diverge and what each tells you.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Construct example G and P with some means
    p = 3
    G = np.array([[0.8, 0.2, 0.1],
                  [0.2, 0.5, 0.15],
                  [0.1, 0.15, 0.3]])
    P = np.array([[1.0, 0.3, 0.2],
                  [0.3, 0.9, 0.25],
                  [0.2, 0.25, 0.7]])
    zbar = np.array([10.0, 4.0, 2.0])
    
    # Sample random directions and compute metrics
    n = 2000
    betas = np.random.randn(n, p)
    betas = betas / np.linalg.norm(betas, axis=1, keepdims=True)
    
    h2_values = []
    e_values = []
    edir_values = []
    for beta in betas:
        h2_values.append(directional_h2(beta, G, P))
        e_values.append(evolvability(beta, G))
        # mean-standardized evolvability along direction (Hansen-Houle)
        proj_mean = beta @ zbar
        if np.isclose(proj_mean, 0.0):
            edir_values.append(np.nan)
        else:
            edir_values.append((beta @ G @ beta) / (proj_mean**2))
    
    h2_values = np.array(h2_values)
    e_values = np.array(e_values)
    edir_values = np.array(edir_values)
    
    # Panel A: h² vs directional evolvability (raw)
    ax = axes[0]
    ax.scatter(e_values, h2_values, alpha=0.4, s=10,
               color=COLORS['response'])
    ax.set_xlabel(r'Raw evolvability $e(\boldsymbol{\beta}) = \boldsymbol{\beta}^\mathsf{T}\mathbf{G}\boldsymbol{\beta}$')
    ax.set_ylabel(r'Directional heritability $h^2(\boldsymbol{\beta})$')
    ax.set_title('A. h² vs raw evolvability')
    ax.grid(alpha=0.2)
    
    # Panel B: mean-standardized directional evolvability vs h²
    ax = axes[1]
    mask = ~np.isnan(edir_values)
    ax.scatter(edir_values[mask], h2_values[mask], alpha=0.4, s=10,
               color=COLORS['selection'])
    ax.set_xlabel(r'Mean-standardized $e_d(\boldsymbol{\beta}) = \frac{\boldsymbol{\beta}^\mathsf{T}\mathbf{G}\boldsymbol{\beta}}{(\boldsymbol{\beta}^\mathsf{T}\bar{\mathbf{z}})^2}$')
    ax.set_ylabel(r'$h^2(\boldsymbol{\beta})$')
    ax.set_title('B. h² vs mean-standardized evolvability')
    ax.grid(alpha=0.2)
    
    # Panel C: Distribution comparison
    ax = axes[2]
    bins = 30
    ax.hist(h2_values, bins=bins, alpha=0.5, density=True,
            label=r'$h^2(\boldsymbol{\beta})$', color=COLORS['high_h2'])
    ax.hist(e_values, bins=bins, alpha=0.5, density=True,
            label=r'$e(\boldsymbol{\beta})$', color=COLORS['selection'])
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title('C. Distributions across directions')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    
    plt.suptitle('Figure 5: Comparing Directional Heritability and Evolvability',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# FIGURE 6: CONSTRAINT TRAPS
# =============================================================================

def figure_6_constraint_traps(save_path=None):
    """
    Figure 6: Visualizing constraint traps with G and P misalignment.
    
    Shows how small phase offsets between G and P create low-h² regions.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 2D example: misaligned G and P
    angle_G = np.radians(0)
    angle_P = np.radians(30)
    
    # Eigenvalues
    g1, g2 = 1.0, 0.2
    p1, p2 = 1.0, 0.7
    
    # Build rotation matrices
    RG = np.array([[np.cos(angle_G), -np.sin(angle_G)],
                   [np.sin(angle_G),  np.cos(angle_G)]])
    RP = np.array([[np.cos(angle_P), -np.sin(angle_P)],
                   [np.sin(angle_P),  np.cos(angle_P)]])
    
    # Construct G and P from eigenvalues and rotations
    G = RG @ np.diag([g1, g2]) @ RG.T
    P = RP @ np.diag([p1, p2]) @ RP.T
    
    # Panel A: e(β) vs p(β) with iso-h² lines
    ax = axes[0]
    n = 2000
    thetas = np.random.uniform(0, 2*np.pi, n)
    betas = np.column_stack([np.cos(thetas), np.sin(thetas)])
    
    e_vals = np.einsum('ij,ij->i', betas @ G, betas)
    p_vals = np.einsum('ij,ij->i', betas @ P, betas)
    h2_vals = e_vals / p_vals
    
    sc = ax.scatter(e_vals, p_vals, c=h2_vals, cmap='viridis',
                    s=15, alpha=0.6)
    ax.set_xlabel(r'Genetic variance $G(\boldsymbol{\beta})$')
    ax.set_ylabel(r'Phenotypic variance $P(\boldsymbol{\beta})$')
    ax.set_title('A. G vs P with iso-h² structure')
    cbar = plt.colorbar(sc, ax=ax, label=r'$h^2(\boldsymbol{\beta})$')
    ax.grid(alpha=0.2)
    
    # Panel B: Respondability vs h²
    ax = axes[1]
    responses = betas @ G  # Gβ
    respond = np.linalg.norm(responses, axis=1)
    ax.scatter(respond, h2_vals, s=15, alpha=0.6, color=COLORS['selection'])
    ax.set_xlabel(r'Respondability $\|\mathbf{G}\boldsymbol{\beta}\|$')
    ax.set_ylabel(r'$h^2(\boldsymbol{\beta})$')
    ax.set_title('B. Large response, low heritability')
    ax.grid(alpha=0.2)
    
    # Mark low-h² region
    low_mask = h2_vals < np.percentile(h2_vals, 10)
    ax.scatter(respond[low_mask], h2_vals[low_mask], s=15,
               color=COLORS['low_h2'], alpha=0.7,
               label='Low-h² trap')
    ax.legend(fontsize=9)
    
    # Panel C: Phase diagram e(θ), p(θ), h²(θ)
    ax = axes[2]
    thetas = np.linspace(0, 2*np.pi, 360)
    betas = np.column_stack([np.cos(thetas), np.sin(thetas)])
    
    e_vals = np.einsum('ij,ij->i', betas @ G, betas)
    p_vals = np.einsum('ij,ij->i', betas @ P, betas)
    h2_vals = e_vals / p_vals
    
    e_scaled = (e_vals - e_vals.min()) / (e_vals.max() - e_vals.min())
    p_scaled = (p_vals - p_vals.min()) / (p_vals.max() - p_vals.min())
    h_scaled = (h2_vals - h2_vals.min()) / (h2_vals.max() - h2_vals.min())
    
    ax.plot(thetas, e_scaled, label='e(β) (scaled)', color=COLORS['G_ellipse'])
    ax.plot(thetas, p_scaled, label='p(β) (scaled)', color=COLORS['P_ellipse'])
    ax.plot(thetas, h_scaled, label='h²(β) (scaled)', color=COLORS['response'])
    
    ax.set_xlabel(r'Angle $\theta$')
    ax.set_ylabel('Scaled value')
    ax.set_title('C. Phase offsets create heritability traps')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    
    plt.suptitle('Figure 6: Constraint Traps from Misaligned G and P',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# FIGURE 7: SUMMARY PANEL
# =============================================================================

def figure_7_summary(save_path=None):
    """
    Figure 7: Summary panel combining key visuals.
    
    A multi-panel figure suitable for talk/overview: ellipses, whitening,
    and distributions together.
    """
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 3)
    
    # Panel A: G and P ellipses
    axA = fig.add_subplot(gs[0, 0])
    G = np.array([[1.0, 0.4], [0.4, 0.3]])
    P = np.array([[1.5, 0.4], [0.4, 0.8]])
    w_G, h_G, angle_G, _, _ = get_ellipse_params(G)
    w_P, h_P, angle_P, _, _ = get_ellipse_params(P)
    ellipse_G = Ellipse((0, 0), w_G, h_G, angle=angle_G,
                        fill=True, facecolor=COLORS['G_ellipse'], alpha=0.3,
                        edgecolor=COLORS['G_ellipse'], linewidth=3)
    ellipse_P = Ellipse((0, 0), w_P, h_P, angle=angle_P,
                        fill=False, edgecolor=COLORS['P_ellipse'],
                        linewidth=3, linestyle='--')
    axA.add_patch(ellipse_P)
    axA.add_patch(ellipse_G)
    axA.set_xlim(-2, 2)
    axA.set_ylim(-1.5, 1.5)
    axA.set_aspect('equal')
    axA.set_title('A. G and P ellipses')
    axA.axhline(0, color=COLORS['neutral'], linewidth=0.5, zorder=0)
    axA.axvline(0, color=COLORS['neutral'], linewidth=0.5, zorder=0)
    
    # Panel B: h² polar plot
    axB = fig.add_subplot(gs[0, 1])
    thetas = np.linspace(0, 2*np.pi, 180)
    h2_vals = []
    for th in thetas:
        beta = np.array([np.cos(th), np.sin(th)])
        h2_vals.append(directional_h2(beta, G, P))
    h2_vals = np.array(h2_vals)
    r = 0.3 + 0.7*h2_vals
    x = r*np.cos(thetas)
    y = r*np.sin(thetas)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(h2_vals.min(), h2_vals.max())
    lc = LineCollection(segments, cmap='RdYlGn', norm=norm, linewidth=3)
    lc.set_array(h2_vals[:-1])
    axB.add_collection(lc)
    axB.set_xlim(-1.3, 1.3)
    axB.set_ylim(-1.3, 1.3)
    axB.set_aspect('equal')
    axB.set_title('B. h² field on the circle')
    axB.axhline(0, color=COLORS['neutral'], linewidth=0.5, zorder=0)
    axB.axvline(0, color=COLORS['neutral'], linewidth=0.5, zorder=0)
    
    # Panel C: Whitening schematic
    axC = fig.add_subplot(gs[0, 2])
    axC.axis('off')
    axC.text(0.5, 0.7, r'$\mathbf{G}^* = \mathbf{P}^{-1/2}\mathbf{G}\mathbf{P}^{-1/2}$',
             fontsize=18, ha='center', va='center', transform=axC.transAxes)
    axC.text(0.5, 0.45,
             'Whitening puts P on the\nunit sphere and leaves\nG* carrying constraints',
             fontsize=12, ha='center', va='center', transform=axC.transAxes)
    
    # Panel D: CV(h²) vs V_rel schematic (reuse from Fig 4A)
    axD = fig.add_subplot(gs[1, :2])
    x = np.linspace(0, 0.99, 200)
    p_vals = [3, 5, 10]
    colors = ['#1b9e77', '#d95f02', '#7570b3']
    for p, c in zip(p_vals, colors):
        cv2 = 2/(p+2) * x
        cv = np.sqrt(cv2)
        axD.plot(x, cv, color=c, label=f'p={p}')
    axD.set_xlabel(r'$V_{\mathrm{rel}}(\mathbf{G}^*)$')
    axD.set_ylabel(r'$\mathrm{CV}(h^2)$')
    axD.set_title('D. Eigenvalue dispersion controls heritability spread')
    axD.legend(fontsize=9)
    axD.grid(alpha=0.2)
    
    # Panel E: Histogram of h²
    axE = fig.add_subplot(gs[1, 2])
    axE.hist(h2_vals, bins=20, color=COLORS['response'], alpha=0.7, density=True)
    axE.set_xlabel(r'$h^2(\boldsymbol{\beta})$')
    axE.set_ylabel('Density')
    axE.set_title('E. Distribution of h² across directions')
    axE.grid(alpha=0.2)
    
    # Panel F: Respondability vs h²
    axF = fig.add_subplot(gs[2, :])
    betas = np.random.randn(1000, 2)
    betas /= np.linalg.norm(betas, axis=1, keepdims=True)
    responses = betas @ G
    respond = np.linalg.norm(responses, axis=1)
    h2_vals2 = np.array([directional_h2(b, G, P) for b in betas])
    axF.scatter(respond, h2_vals2, alpha=0.4, s=10,
                color=COLORS['selection'])
    axF.set_xlabel(r'Respondability $\|\mathbf{G}\boldsymbol{\beta}\|$')
    axF.set_ylabel(r'$h^2(\boldsymbol{\beta})$')
    axF.set_title('F. Large response need not mean high heritability')
    axF.grid(alpha=0.2)
    
    plt.suptitle('Figure 7: Summary of the Geometry of Directional Heritability',
                fontsize=18, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# MAIN DRIVER
# =============================================================================

def save_all_figures(output_dir="figures_pedagogical"):
    """
    Generate and save all figures as PDFs in the specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig1 = figure_1_what_is_dirh2(save_path=os.path.join(output_dir, "figure1_directional_h2.pdf"))
    plt.close(fig1)
    
    fig2 = figure_2_quadratic_forms(save_path=os.path.join(output_dir, "figure2_quadratic_forms.pdf"))
    plt.close(fig2)
    
    fig3 = figure_3_whitening(save_path=os.path.join(output_dir, "figure3_whitening.pdf"))
    plt.close(fig3)
    
    fig4 = figure_4_eigenvalues_and_cv(save_path=os.path.join(output_dir, "figure4_eigenvalues_cv.pdf"))
    plt.close(fig4)
    
    fig5 = figure_5_evolvability_vs_h2(save_path=os.path.join(output_dir, "figure5_evolvability_vs_h2.pdf"))
    plt.close(fig5)
    
    fig6 = figure_6_constraint_traps(save_path=os.path.join(output_dir, "figure6_constraint_traps.pdf"))
    plt.close(fig6)
    
    fig7 = figure_7_summary(save_path=os.path.join(output_dir, "figure7_summary.pdf"))
    plt.close(fig7)
    
    print(f"All pedagogical figures saved to: {output_dir}")


if __name__ == "__main__":
    save_all_figures()
