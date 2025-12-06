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
h2_cmap = LinearSegmentedColormap.from_list('h2_cmap', 
    ['#B2182B', '#F4A582', '#FDDBC7', '#D1E5F0', '#67A9CF', '#2166AC'])

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
    ax.text(0.95, 0.05, 'G* shape reveals\nconstraint structure!',
           transform=ax.transAxes, ha='right', va='bottom', fontsize=11,
           fontweight='bold', color=COLORS['Gstar'],
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Figure 3: P-Whitening Reveals the Constraint Structure',
                fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# FIGURE 4: FROM EIGENVALUES TO CV(h²)
# =============================================================================

def figure_4_eigenvalues_to_cv(save_path=None):
    """
    Figure 4: How G* eigenvalues determine CV(h²).
    
    Shows the connection between anisotropy and variance in heritability.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel A: Low anisotropy
    ax = axes[0]
    
    # Nearly isotropic G*
    eigvals_low = np.array([0.55, 0.45])
    G_low = np.diag(eigvals_low)
    
    # Draw circle (P* = I)
    circle = Circle((0, 0), 1, fill=False, edgecolor=COLORS['P_ellipse'],
                   linewidth=2, linestyle='--', alpha=0.5)
    ax.add_patch(circle)
    
    # Draw G* ellipse
    ellipse = Ellipse((0, 0), 2*np.sqrt(eigvals_low[0]), 2*np.sqrt(eigvals_low[1]),
                     fill=True, facecolor=COLORS['Gstar'], alpha=0.4,
                     edgecolor=COLORS['Gstar'], linewidth=3)
    ax.add_patch(ellipse)
    
    # Draw h² around the circle (polar plot)
    n_angles = 100
    thetas = np.linspace(0, 2*np.pi, n_angles)
    h2_values = []
    for theta in thetas:
        beta = np.array([np.cos(theta), np.sin(theta)])
        h2_values.append(beta @ G_low @ beta)  # Since P* = I
    h2_values = np.array(h2_values)
    
    r = 0.3 + h2_values
    x = r * np.cos(thetas)
    y = r * np.sin(thetas)
    ax.plot(x, y, color=COLORS['response'], linewidth=2.5)
    ax.fill(x, y, color=COLORS['response'], alpha=0.2)
    
    # Statistics
    cv = np.std(h2_values) / np.mean(h2_values)
    vrel = np.var(eigvals_low) / np.mean(eigvals_low)**2
    
    ax.text(0.5, 0.95, f'CV(h²) = {cv:.3f}', transform=ax.transAxes,
           fontsize=14, ha='center', va='top', fontweight='bold',
           color=COLORS['response'])
    ax.text(0.5, 0.05, f'h² range: {h2_values.min():.2f} – {h2_values.max():.2f}',
           transform=ax.transAxes, fontsize=11, ha='center', va='bottom')
    
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.set_title('A. Low anisotropy: small CV')
    ax.axis('off')
    
    # Panel B: High anisotropy
    ax = axes[1]
    
    # Highly anisotropic G*
    eigvals_high = np.array([0.8, 0.2])
    G_high = np.diag(eigvals_high)
    
    # Draw circle (P* = I)
    circle = Circle((0, 0), 1, fill=False, edgecolor=COLORS['P_ellipse'],
                   linewidth=2, linestyle='--', alpha=0.5)
    ax.add_patch(circle)
    
    # Draw G* ellipse
    ellipse = Ellipse((0, 0), 2*np.sqrt(eigvals_high[0]), 2*np.sqrt(eigvals_high[1]),
                     fill=True, facecolor=COLORS['Gstar'], alpha=0.4,
                     edgecolor=COLORS['Gstar'], linewidth=3)
    ax.add_patch(ellipse)
    
    # Draw h² around the circle
    h2_values_high = []
    for theta in thetas:
        beta = np.array([np.cos(theta), np.sin(theta)])
        h2_values_high.append(beta @ G_high @ beta)
    h2_values_high = np.array(h2_values_high)
    
    r = 0.3 + h2_values_high
    x = r * np.cos(thetas)
    y = r * np.sin(thetas)
    ax.plot(x, y, color=COLORS['response'], linewidth=2.5)
    ax.fill(x, y, color=COLORS['response'], alpha=0.2)
    
    # Mark high and low directions
    ax.annotate('', xy=(1.15, 0), xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', color=COLORS['high_h2'], lw=3))
    ax.text(1.25, 0, 'HIGH\nh²', fontsize=10, ha='left', va='center',
           color=COLORS['high_h2'], fontweight='bold')
    
    ax.annotate('', xy=(0, 0.55), xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', color=COLORS['low_h2'], lw=3))
    ax.text(0, 0.7, 'LOW h²', fontsize=10, ha='center', va='bottom',
           color=COLORS['low_h2'], fontweight='bold')
    
    cv_high = np.std(h2_values_high) / np.mean(h2_values_high)
    
    ax.text(0.5, 0.95, f'CV(h²) = {cv_high:.3f}', transform=ax.transAxes,
           fontsize=14, ha='center', va='top', fontweight='bold',
           color=COLORS['response'])
    ax.text(0.5, 0.05, f'h² range: {h2_values_high.min():.2f} – {h2_values_high.max():.2f}',
           transform=ax.transAxes, fontsize=11, ha='center', va='bottom')
    
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.set_title('B. High anisotropy: large CV')
    ax.axis('off')
    
    # Panel C: The formula connecting them
    ax = axes[2]
    ax.axis('off')
    
    # Main formula
    ax.text(0.5, 0.8, r'$\mathrm{CV}^2(h^2) = \frac{2 \cdot V_\mathrm{rel}}{p + 2}$',
           fontsize=24, ha='center', va='center', transform=ax.transAxes)
    
    # Define Vrel
    ax.text(0.5, 0.6, r'where $V_\mathrm{rel} = \frac{\mathrm{Var}(\lambda^*)}{\mathrm{E}(\lambda^*)^2}$',
           fontsize=16, ha='center', va='center', transform=ax.transAxes)
    
    # Explanation box
    explanation = (
        "The Key Insight:\n\n"
        "• Vrel measures G* anisotropy\n"
        "  (relative spread of eigenvalues)\n\n"
        "• More anisotropy → larger CV(h²)\n"
        "  → more variation in heritability\n"
        "  → greater potential for constraint\n\n"
        "• p = number of traits"
    )
    ax.text(0.5, 0.25, explanation, fontsize=12, ha='center', va='center',
           transform=ax.transAxes, linespacing=1.4,
           bbox=dict(boxstyle='round', facecolor=COLORS['background'], 
                    edgecolor=COLORS['Gstar'], linewidth=2))
    
    ax.set_title('C. The CV(h²) – Vrel connection')
    
    plt.suptitle('Figure 4: From Eigenvalues to CV(h²)',
                fontsize=18, fontweight='bold', y=1.02)
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
    Figure 5: The critical distinction between evolvability and heritability.
    
    Shows how a direction can have high evolvability but low heritability.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Create G and P where one direction has high G-variance but also high E-variance
    # This creates a "constraint trap"
    G = np.array([[1.2, 0], [0, 0.3]])   # More G-variance in direction 1
    E = np.array([[1.8, 0], [0, 0.1]])   # Much more E-variance in direction 1 too
    P = G + E
    
    # Panel A: Evolvability view (G only)
    ax = axes[0]
    
    w_G, h_G, angle_G, _, _ = get_ellipse_params(G)
    
    # Draw G ellipse
    ellipse_G = Ellipse((0, 0), w_G, h_G, angle=angle_G,
                        fill=True, facecolor=COLORS['G_ellipse'], alpha=0.3,
                        edgecolor=COLORS['G_ellipse'], linewidth=3)
    ax.add_patch(ellipse_G)
    
    # Show evolvability for different directions
    thetas = np.linspace(0, 2*np.pi, 100)
    e_values = []
    for theta in thetas:
        beta = np.array([np.cos(theta), np.sin(theta)])
        e_values.append(evolvability(beta, G))
    e_values = np.array(e_values)
    
    # Polar-style plot
    r = 0.3 + 0.8 * (e_values / e_values.max())
    x = r * np.cos(thetas)
    y = r * np.sin(thetas)
    ax.plot(x, y, color=COLORS['G_ellipse'], linewidth=2.5)
    ax.fill(x, y, color=COLORS['G_ellipse'], alpha=0.15)
    
    # Mark the high-evolvability direction
    ax.annotate('', xy=(1.3, 0), xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', color=COLORS['selection'], lw=3))
    ax.text(1.4, 0.15, f'e(β) = {evolvability(np.array([1,0]), G):.2f}', 
           fontsize=11, color=COLORS['selection'], fontweight='bold')
    ax.text(1.4, -0.15, 'HIGH genetic\nvariance!', fontsize=10, 
           color=COLORS['high_h2'], ha='left')
    
    ax.set_xlim(-1.5, 1.8)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.set_title(r'A. Evolvability: $e(\boldsymbol{\beta}) = \frac{\boldsymbol{\beta}^\mathsf{T}\mathbf{G}\boldsymbol{\beta}}{|\boldsymbol{\beta}|^2}$')
    ax.axis('off')
    
    # Add formula box
    ax.text(0.5, 0.05, 'Evolvability = genetic variance\nin selection direction',
           transform=ax.transAxes, ha='center', va='bottom', fontsize=10,
           style='italic')
    
    # Panel B: Heritability view (G/P)
    ax = axes[1]
    
    # Draw P ellipse (includes E)
    w_P, h_P, angle_P, _, _ = get_ellipse_params(P)
    ellipse_P = Ellipse((0, 0), w_P, h_P, angle=angle_P,
                        fill=False, edgecolor=COLORS['P_ellipse'],
                        linewidth=3, linestyle='--')
    ax.add_patch(ellipse_P)
    
    # Draw G ellipse inside
    ellipse_G = Ellipse((0, 0), w_G, h_G, angle=angle_G,
                        fill=True, facecolor=COLORS['G_ellipse'], alpha=0.3,
                        edgecolor=COLORS['G_ellipse'], linewidth=3)
    ax.add_patch(ellipse_G)
    
    # Show h² for different directions
    h2_values = []
    for theta in thetas:
        beta = np.array([np.cos(theta), np.sin(theta)])
        h2_values.append(directional_h2(beta, G, P))
    h2_values = np.array(h2_values)
    
    # Polar-style plot - colored by h²
    r = 0.3 + 0.8 * h2_values
    x = r * np.cos(thetas)
    y = r * np.sin(thetas)
    
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(h2_values.min(), h2_values.max())
    lc = LineCollection(segments, cmap='RdYlGn', norm=norm, linewidth=3)
    lc.set_array(h2_values[:-1])
    ax.add_collection(lc)
    
    # Mark the same direction - now low h²!
    h2_trap = directional_h2(np.array([1, 0]), G, P)
    ax.annotate('', xy=(1.3, 0), xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', color=COLORS['selection'], lw=3))
    ax.text(1.4, 0.15, f'h²(β) = {h2_trap:.2f}', 
           fontsize=11, color=COLORS['selection'], fontweight='bold')
    ax.text(1.4, -0.15, 'LOW heritability!', fontsize=10,
           color=COLORS['low_h2'], ha='left', fontweight='bold')
    
    ax.set_xlim(-1.5, 1.8)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.set_title(r'B. Dir. Heritability: $h^2(\boldsymbol{\beta}) = \frac{\boldsymbol{\beta}^\mathsf{T}\mathbf{G}\boldsymbol{\beta}}{\boldsymbol{\beta}^\mathsf{T}\mathbf{P}\boldsymbol{\beta}}$')
    ax.axis('off')
    
    ax.text(0.5, 0.05, 'Heritability = fraction of variance\nthat is genetic',
           transform=ax.transAxes, ha='center', va='bottom', fontsize=10,
           style='italic')
    
    # Panel C: The constraint trap
    ax = axes[2]
    ax.axis('off')
    
    # Create a visual showing the trap
    ax.text(0.5, 0.92, 'THE CONSTRAINT TRAP', fontsize=16, ha='center', 
           fontweight='bold', transform=ax.transAxes, color=COLORS['low_h2'])
    
    # Draw a schematic
    # High evolvability
    ax.add_patch(plt.Rectangle((0.1, 0.55), 0.35, 0.25, 
                               facecolor=COLORS['G_ellipse'], alpha=0.3,
                               edgecolor=COLORS['G_ellipse'], linewidth=2,
                               transform=ax.transAxes))
    ax.text(0.275, 0.75, 'High e(β)\n(genetic variance)', fontsize=11,
           ha='center', va='center', transform=ax.transAxes)
    
    # Plus high environmental
    ax.text(0.5, 0.675, '+', fontsize=24, ha='center', va='center',
           transform=ax.transAxes, fontweight='bold')
    
    ax.add_patch(plt.Rectangle((0.55, 0.55), 0.35, 0.25,
                               facecolor=COLORS['P_ellipse'], alpha=0.3,
                               edgecolor=COLORS['P_ellipse'], linewidth=2,
                               transform=ax.transAxes))
    ax.text(0.725, 0.75, 'High E-variance\n(environmental)', fontsize=11,
           ha='center', va='center', transform=ax.transAxes)
    
    # Equals low h²
    ax.annotate('', xy=(0.5, 0.35), xytext=(0.5, 0.52),
               arrowprops=dict(arrowstyle='->', color=COLORS['text'], lw=3),
               transform=ax.transAxes)
    
    ax.add_patch(plt.Rectangle((0.25, 0.15), 0.5, 0.18,
                               facecolor=COLORS['low_h2'], alpha=0.3,
                               edgecolor=COLORS['low_h2'], linewidth=3,
                               transform=ax.transAxes))
    ax.text(0.5, 0.24, 'LOW h²(β)\nResponse swamped by noise!', fontsize=12,
           ha='center', va='center', transform=ax.transAxes,
           fontweight='bold', color=COLORS['low_h2'])
    
    # Key insight
    ax.text(0.5, 0.02, 
           'Evolvability alone can miss constraint traps.\nDirectional heritability reveals them.',
           fontsize=11, ha='center', va='bottom', transform=ax.transAxes,
           style='italic',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    ax.set_title('C. The hidden trap')
    
    plt.suptitle('Figure 5: Evolvability vs Directional Heritability',
                fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# FIGURE 6: CONSTRAINT TRAPS REVEALED
# =============================================================================

def figure_6_constraint_traps(save_path=None):
    """
    Figure 6: How directional heritability reveals constraint traps.
    
    Shows the selection-response dynamics when hitting a low-h² direction.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Create anisotropic G and P
    G = np.array([[0.8, 0.3], [0.3, 0.2]])
    P = np.array([[1.2, 0.3], [0.3, 0.6]])
    
    # Get eigenvectors of G* for reference
    eigvals_P, eigvecs_P = np.linalg.eigh(P)
    P_inv_sqrt = eigvecs_P @ np.diag(1/np.sqrt(eigvals_P)) @ eigvecs_P.T
    Gstar = P_inv_sqrt @ G @ P_inv_sqrt
    eigvals_Gstar, eigvecs_Gstar = np.linalg.eigh(Gstar)
    
    # Panel A: High-h² selection
    ax = axes[0]
    
    # Draw ellipses
    w_G, h_G, angle_G, _, _ = get_ellipse_params(G)
    w_P, h_P, angle_P, _, _ = get_ellipse_params(P)
    
    ellipse_G = Ellipse((0, 0), w_G, h_G, angle=angle_G,
                        fill=True, facecolor=COLORS['G_ellipse'], alpha=0.3,
                        edgecolor=COLORS['G_ellipse'], linewidth=2)
    ellipse_P = Ellipse((0, 0), w_P, h_P, angle=angle_P,
                        fill=False, edgecolor=COLORS['P_ellipse'],
                        linewidth=2, linestyle='--')
    ax.add_patch(ellipse_P)
    ax.add_patch(ellipse_G)
    
    # High-h² selection direction (along g_max)
    # Find the direction with highest h²
    best_theta = None
    best_h2 = 0
    for theta in np.linspace(0, np.pi, 100):
        beta = np.array([np.cos(theta), np.sin(theta)])
        h2 = directional_h2(beta, G, P)
        if h2 > best_h2:
            best_h2 = h2
            best_theta = theta
    
    beta_good = np.array([np.cos(best_theta), np.sin(best_theta)])
    
    # Selection arrow
    ax.annotate('', xy=beta_good*1.5, xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', color=COLORS['selection'], lw=3))
    
    # Response arrow
    response = G @ beta_good
    response = response / np.linalg.norm(response) * 1.3
    ax.annotate('', xy=response, xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', color=COLORS['response'], lw=3))
    
    # Calculate angle
    cos_angle = np.dot(beta_good, response/np.linalg.norm(response))
    angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
    
    ax.text(0.5, 0.95, f'h²(β) = {best_h2:.2f}', transform=ax.transAxes,
           fontsize=14, ha='center', va='top', fontweight='bold',
           color=COLORS['high_h2'])
    ax.text(0.5, 0.05, f'Deflection: {angle_deg:.0f}°', transform=ax.transAxes,
           fontsize=12, ha='center', va='bottom')
    
    # Labels
    ax.text(beta_good[0]*1.6, beta_good[1]*1.6, 'β', fontsize=14,
           color=COLORS['selection'], fontweight='bold')
    ax.text(response[0]*1.1, response[1]*1.1, 'Gβ', fontsize=14,
           color=COLORS['response'], fontweight='bold')
    
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title('A. Selection along high-h² direction')
    ax.axis('off')
    
    # Panel B: Low-h² selection (constraint trap!)
    ax = axes[1]
    
    ellipse_G = Ellipse((0, 0), w_G, h_G, angle=angle_G,
                        fill=True, facecolor=COLORS['G_ellipse'], alpha=0.3,
                        edgecolor=COLORS['G_ellipse'], linewidth=2)
    ellipse_P = Ellipse((0, 0), w_P, h_P, angle=angle_P,
                        fill=False, edgecolor=COLORS['P_ellipse'],
                        linewidth=2, linestyle='--')
    ax.add_patch(ellipse_P)
    ax.add_patch(ellipse_G)
    
    # Low-h² selection direction (perpendicular to best)
    beta_bad = np.array([-beta_good[1], beta_good[0]])  # Perpendicular
    h2_bad = directional_h2(beta_bad, G, P)
    
    # Selection arrow
    ax.annotate('', xy=beta_bad*1.5, xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', color=COLORS['selection'], lw=3))
    
    # Response arrow (will be deflected!)
    response = G @ beta_bad
    response_norm = response / np.linalg.norm(response) * 0.8  # Shorter due to low h²
    ax.annotate('', xy=response_norm, xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', color=COLORS['response'], lw=3))
    
    # Calculate angle
    cos_angle = np.dot(beta_bad, response/np.linalg.norm(response))
    angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
    
    ax.text(0.5, 0.95, f'h²(β) = {h2_bad:.2f}', transform=ax.transAxes,
           fontsize=14, ha='center', va='top', fontweight='bold',
           color=COLORS['low_h2'])
    ax.text(0.5, 0.05, f'Deflection: {angle_deg:.0f}°', transform=ax.transAxes,
           fontsize=12, ha='center', va='bottom')
    
    # Danger zone highlight
    wedge = Wedge((0, 0), 1.0, 
                  np.degrees(np.arctan2(beta_bad[1], beta_bad[0])) - 30,
                  np.degrees(np.arctan2(beta_bad[1], beta_bad[0])) + 30,
                  facecolor=COLORS['low_h2'], alpha=0.15)
    ax.add_patch(wedge)
    
    ax.text(beta_bad[0]*1.6, beta_bad[1]*1.6, 'β', fontsize=14,
           color=COLORS['selection'], fontweight='bold')
    ax.text(response_norm[0]*1.2, response_norm[1]*1.2, 'Gβ\n(weak!)', fontsize=12,
           color=COLORS['response'], fontweight='bold', ha='center')
    
    # Add "TRAP" label
    ax.text(beta_bad[0]*0.6, beta_bad[1]*0.6, 'CONSTRAINT\nTRAP', fontsize=10,
           ha='center', va='center', color=COLORS['low_h2'], fontweight='bold',
           rotation=np.degrees(np.arctan2(beta_bad[1], beta_bad[0])))
    
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title('B. Selection into constraint trap')
    ax.axis('off')
    
    # Panel C: Summary schematic
    ax = axes[2]
    ax.axis('off')
    
    ax.text(0.5, 0.95, 'Detecting Constraint Traps', fontsize=16,
           ha='center', va='top', fontweight='bold', transform=ax.transAxes)
    
    # Comparison table
    table_data = [
        ['', 'High h²(β)', 'Low h²(β)'],
        ['Genetic variance', '✓ Present', '✓ Present*'],
        ['Response magnitude', 'Strong', 'Weak'],
        ['Response direction', 'Aligned', 'Deflected'],
        ['Evolutionary outcome', 'Adaptive', 'Constrained']
    ]
    
    cell_colors = [
        ['white', COLORS['high_h2'], COLORS['low_h2']],
        ['white', '#d4edda', '#f8d7da'],
        ['white', '#d4edda', '#f8d7da'],
        ['white', '#d4edda', '#f8d7da'],
        ['white', '#d4edda', '#f8d7da']
    ]
    
    table = ax.table(cellText=table_data, cellColours=cell_colors,
                    loc='center', cellLoc='center',
                    bbox=[0.05, 0.25, 0.9, 0.55])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    
    # Make header row bold
    for i in range(3):
        table[(0, i)].set_text_props(fontweight='bold')
        table[(0, i)].set_facecolor('#e0e0e0')
    
    # Footnote
    ax.text(0.5, 0.15, '*Evolvability may be high, but response is swamped by environmental variance',
           fontsize=10, ha='center', va='top', transform=ax.transAxes, style='italic')
    
    ax.text(0.5, 0.05, 
           'CV(h²) summarizes how much constraint potential exists across all directions',
           fontsize=11, ha='center', va='bottom', transform=ax.transAxes,
           fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    ax.set_title('C. The diagnostic')
    
    plt.suptitle('Figure 6: Constraint Traps Revealed',
                fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# FIGURE 7: THE COMPLETE PICTURE
# =============================================================================

def figure_7_complete_picture(save_path=None):
    """
    Figure 7: The complete conceptual framework.
    
    Ties everything together in a single visual summary.
    """
    fig = plt.figure(figsize=(16, 10))
    
    # Create a complex grid
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3,
                         height_ratios=[1, 1, 0.8])
    
    # --- Top row: The transformation pipeline ---
    
    # Box 1: G and P matrices
    ax = fig.add_subplot(gs[0, 0])
    
    G = np.array([[0.8, 0.3], [0.3, 0.3]])
    P = np.array([[1.2, 0.3], [0.3, 0.7]])
    
    w_G, h_G, angle_G, _, _ = get_ellipse_params(G)
    w_P, h_P, angle_P, _, _ = get_ellipse_params(P)
    
    ellipse_G = Ellipse((0, 0), w_G*0.8, h_G*0.8, angle=angle_G,
                        fill=True, facecolor=COLORS['G_ellipse'], alpha=0.4,
                        edgecolor=COLORS['G_ellipse'], linewidth=2, label='G')
    ellipse_P = Ellipse((0, 0), w_P*0.8, h_P*0.8, angle=angle_P,
                        fill=False, edgecolor=COLORS['P_ellipse'],
                        linewidth=2, linestyle='--', label='P')
    ax.add_patch(ellipse_P)
    ax.add_patch(ellipse_G)
    
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title('1. Raw matrices', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Arrow
    ax_arrow1 = fig.add_subplot(gs[0, 1])
    ax_arrow1.axis('off')
    ax_arrow1.annotate('', xy=(0.9, 0.5), xytext=(0.1, 0.5),
                      arrowprops=dict(arrowstyle='->', lw=3, color=COLORS['Gstar']),
                      transform=ax_arrow1.transAxes)
    ax_arrow1.text(0.5, 0.65, 'P-whiten', fontsize=11, ha='center',
                  transform=ax_arrow1.transAxes, fontweight='bold')
    ax_arrow1.text(0.5, 0.35, r'$\mathbf{G}^* = \mathbf{P}^{-1/2}\mathbf{G}\mathbf{P}^{-1/2}$',
                  fontsize=10, ha='center', transform=ax_arrow1.transAxes)
    
    # Box 2: G* matrix
    ax = fig.add_subplot(gs[0, 2])
    
    eigvals_P, eigvecs_P = np.linalg.eigh(P)
    P_inv_sqrt = eigvecs_P @ np.diag(1/np.sqrt(eigvals_P)) @ eigvecs_P.T
    Gstar = P_inv_sqrt @ G @ P_inv_sqrt
    
    w_Gs, h_Gs, angle_Gs, eigvals_Gs, _ = get_ellipse_params(Gstar)
    
    circle = Circle((0, 0), 0.8, fill=False, edgecolor=COLORS['P_ellipse'],
                   linewidth=2, linestyle='--', label='P* = I')
    ellipse_Gs = Ellipse((0, 0), w_Gs*0.8, h_Gs*0.8, angle=angle_Gs,
                        fill=True, facecolor=COLORS['Gstar'], alpha=0.4,
                        edgecolor=COLORS['Gstar'], linewidth=2, label='G*')
    ax.add_patch(circle)
    ax.add_patch(ellipse_Gs)
    
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title('2. Whitened space', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Arrow to eigenvalues
    ax_arrow2 = fig.add_subplot(gs[0, 3])
    ax_arrow2.axis('off')
    ax_arrow2.annotate('', xy=(0.9, 0.5), xytext=(0.1, 0.5),
                      arrowprops=dict(arrowstyle='->', lw=3, color=COLORS['response']),
                      transform=ax_arrow2.transAxes)
    ax_arrow2.text(0.5, 0.65, 'Eigendecompose', fontsize=11, ha='center',
                  transform=ax_arrow2.transAxes, fontweight='bold')
    ax_arrow2.text(0.5, 0.3, f'λ* = [{eigvals_Gs[1]:.2f}, {eigvals_Gs[0]:.2f}]',
                  fontsize=10, ha='center', transform=ax_arrow2.transAxes)
    
    # --- Middle row: Key quantities ---
    
    # CV(h²) formula
    ax = fig.add_subplot(gs[1, 0:2])
    ax.axis('off')
    
    # Calculate actual values
    mean_eigval = np.mean(eigvals_Gs)
    var_eigval = np.var(eigvals_Gs)
    vrel = var_eigval / mean_eigval**2
    p = 2
    cv_h2 = np.sqrt(2 * vrel / (p + 2))
    
    ax.text(0.5, 0.85, '3. Key Summary Statistics', fontsize=14, ha='center',
           fontweight='bold', transform=ax.transAxes)
    
    formula_text = (
        f"Mean h² = mean(λ*) = {mean_eigval:.3f}\n\n"
        f"Vrel = Var(λ*)/mean(λ*)² = {vrel:.3f}\n\n"
        f"CV(h²) = √(2·Vrel/(p+2)) = {cv_h2:.3f}"
    )
    ax.text(0.5, 0.4, formula_text, fontsize=13, ha='center', va='center',
           transform=ax.transAxes, fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor=COLORS['background'], 
                    edgecolor=COLORS['Gstar'], linewidth=2))
    
    # Interpretation
    ax = fig.add_subplot(gs[1, 2:4])
    ax.axis('off')
    
    ax.text(0.5, 0.85, '4. Biological Interpretation', fontsize=14, ha='center',
           fontweight='bold', transform=ax.transAxes)
    
    interpretation = (
        f"CV(h²) = {cv_h2:.3f} tells us:\n\n"
        "• How much heritability varies by direction\n"
        "• Higher CV → more potential for constraint traps\n"
        "• Selection response depends on WHERE you push\n\n"
        f"In this example:\n"
        f"  Best direction: h² = {eigvals_Gs.max():.2f}\n"
        f"  Worst direction: h² = {eigvals_Gs.min():.2f}"
    )
    ax.text(0.5, 0.35, interpretation, fontsize=11, ha='center', va='center',
           transform=ax.transAxes, linespacing=1.4,
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # --- Bottom row: The punchline ---
    
    ax = fig.add_subplot(gs[2, :])
    ax.axis('off')
    
    ax.text(0.5, 0.9, 'THE KEY INSIGHT', fontsize=16, ha='center',
           fontweight='bold', transform=ax.transAxes,
           color=COLORS['Gstar'])
    
    punchline = (
        "CV(h²) measures the CAPACITY for constraint.\n"
        "The selection gradient β determines whether that constraint is REALIZED.\n\n"
        "Directional heritability h²(β) reveals constraint traps that evolvability alone cannot detect."
    )
    ax.text(0.5, 0.4, punchline, fontsize=14, ha='center', va='center',
           transform=ax.transAxes, linespacing=1.6,
           bbox=dict(boxstyle='round', facecolor='white', 
                    edgecolor=COLORS['Gstar'], linewidth=3))
    
    plt.suptitle('Figure 7: The Complete Framework',
                fontsize=20, fontweight='bold', y=0.98)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# MAIN: Generate all figures
# =============================================================================

if __name__ == "__main__":
    output_dir = os.getcwd()
    print(f"Generating pedagogical figures in: {output_dir}\n")
    
    print("Figure 1: What is Directional Heritability?")
    figure_1_what_is_dirh2(os.path.join(output_dir, 'pedagogical_fig1_what_is_dirh2.png'))
    plt.close()
    
    print("\nFigure 2: Quadratic Forms")
    figure_2_quadratic_forms(os.path.join(output_dir, 'pedagogical_fig2_quadratic_forms.png'))
    plt.close()
    
    print("\nFigure 3: P-Whitening")
    figure_3_whitening(os.path.join(output_dir, 'pedagogical_fig3_whitening.png'))
    plt.close()
    
    print("\nFigure 4: Eigenvalues to CV")
    figure_4_eigenvalues_to_cv(os.path.join(output_dir, 'pedagogical_fig4_eigenvalues_cv.png'))
    plt.close()
    
    print("\nFigure 5: Evolvability vs Heritability")
    figure_5_evolvability_vs_h2(os.path.join(output_dir, 'pedagogical_fig5_evolvability_vs_h2.png'))
    plt.close()
    
    print("\nFigure 6: Constraint Traps")
    figure_6_constraint_traps(os.path.join(output_dir, 'pedagogical_fig6_constraint_traps.png'))
    plt.close()
    
    print("\nFigure 7: Complete Picture")
    figure_7_complete_picture(os.path.join(output_dir, 'pedagogical_fig7_complete_picture.png'))
    plt.close()
    
    print("\n" + "="*60)
    print("ALL FIGURES GENERATED SUCCESSFULLY!")
    print("="*60)
    print("""
    Figure 1: What is directional heritability?
             - Introduction to the concept
             - Shows h² depends on direction
    
    Figure 2: Quadratic forms
             - The mathematical foundation
             - Numerator (G) and denominator (P)
    
    Figure 3: P-whitening transformation
             - How G* is constructed
             - Why it reveals constraint structure
    
    Figure 4: From eigenvalues to CV(h²)
             - The anisotropy-CV connection
             - Visual comparison low vs high CV
    
    Figure 5: Evolvability vs directional heritability  
             - The critical distinction
             - Why evolvability can miss traps
    
    Figure 6: Constraint traps revealed
             - High-h² vs low-h² selection
             - The diagnostic table
    
    Figure 7: The complete framework
             - Pipeline from G,P to CV(h²)
             - The key insight summary
    """)
