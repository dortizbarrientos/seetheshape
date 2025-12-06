#!/usr/bin/env python3
"""
Empiricist-Focused Figures: A Practical Guide to Directional Heritability
==========================================================================

These figures are designed for empiricists who want to:
1. Understand what data they need
2. Compute CV(h²) for their system
3. Interpret results for practical decisions
4. Identify constraint traps in breeding or evolution

Uses the manuscript color palette for consistency.

Author: Daniel Ortiz-Barrientos
Date: December 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.patches import ArrowStyle, ConnectionPatch
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe
from scipy import stats
import os

# =============================================================================
# MANUSCRIPT COLOR PALETTE
# =============================================================================

PAL = {
    # Primary matrix colors
    'G': '#2E86AB',           # Deep blue (genetic)
    'P': '#06D6A0',           # Teal (phenotypic)
    'G_fill': '#2E86AB40',    # With transparency
    'P_fill': '#06D6A020',
    
    # Selection and response
    'beta': '#F18F01',        # Warm orange
    'response': '#C73E1D',    # Warm red
    
    # Constraint visualization
    'constraint': '#F18F01',
    'trap': '#C73E1D',
    
    # Polar plots
    'aligned': '#2E86AB',
    'misaligned': '#A23B72',
    
    # Neutral/structural
    'axis': '#0B3C5D',
    'eigenvec': '#0B3C5D',
    'grid': '#D9D9D9',
    'text': '#1A2332',
    
    # Scenario colors
    'Highway': '#2E86AB',
    'Flexible': '#06D6A0',
    'Natural': '#67B7D1',
    'Trap': '#A23B72',
    'Walling': '#0B3C5D',
    'DeadEnd': '#C73E1D',
    
    # Heatmap
    'heatmap_low': '#F7F7F7',
    'heatmap_mid': '#67B7D1',
    'heatmap_high': '#0B3C5D',
    
    # Additional
    'success': '#06D6A0',
    'warning': '#F18F01',
    'danger': '#C73E1D',
    'neutral': '#D9D9D9',
    'white': '#FFFFFF',
}

def pal_alpha(color_name, alpha):
    """Add transparency to a palette color."""
    hex_color = PAL.get(color_name, PAL['G'])
    rgb = tuple(int(hex_color.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
    return (*rgb, alpha)

# Figure settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
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

def get_ellipse_params(matrix):
    """Get ellipse parameters from 2x2 matrix."""
    eigvals, eigvecs = np.linalg.eigh(matrix)
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


def compute_cv_h2(G, P):
    """Compute CV(h²) from G and P matrices."""
    # P-whiten
    eigvals_P, eigvecs_P = np.linalg.eigh(P)
    P_inv_sqrt = eigvecs_P @ np.diag(1/np.sqrt(eigvals_P)) @ eigvecs_P.T
    Gstar = P_inv_sqrt @ G @ P_inv_sqrt
    
    # Get eigenvalues of G*
    eigvals_Gstar = np.linalg.eigvalsh(Gstar)
    
    # Compute statistics
    mean_h2 = np.mean(eigvals_Gstar)
    var_h2 = np.var(eigvals_Gstar)
    vrel = var_h2 / (mean_h2 ** 2)
    p = G.shape[0]
    cv_h2 = np.sqrt(2 * vrel / (p + 2))
    
    return {
        'cv_h2': cv_h2,
        'mean_h2': mean_h2,
        'vrel': vrel,
        'eigvals': np.sort(eigvals_Gstar),
        'min_h2': eigvals_Gstar.min(),
        'max_h2': eigvals_Gstar.max(),
        'Gstar': Gstar
    }


# =============================================================================
# FIGURE E1: WHAT DATA DO YOU NEED?
# =============================================================================

def figure_E1_what_data(save_path=None):
    """
    Figure E1: What data do you need to compute directional heritability?
    
    Shows the required inputs in a practical, accessible way.
    """
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3,
                         height_ratios=[1.2, 1])
    
    # --- Panel A: The G matrix ---
    ax = fig.add_subplot(gs[0, 0])
    
    # Draw a stylized matrix
    G_example = np.array([[0.42, 0.15, 0.08],
                          [0.15, 0.28, 0.12],
                          [0.08, 0.12, 0.19]])
    
    im = ax.imshow(G_example, cmap='Blues', aspect='equal', vmin=0, vmax=0.5)
    
    # Add values
    for i in range(3):
        for j in range(3):
            color = 'white' if G_example[i,j] > 0.25 else PAL['text']
            ax.text(j, i, f'{G_example[i,j]:.2f}', ha='center', va='center',
                   fontsize=12, fontweight='bold', color=color)
    
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(['Trait 1', 'Trait 2', 'Trait 3'], fontsize=10)
    ax.set_yticklabels(['Trait 1', 'Trait 2', 'Trait 3'], fontsize=10)
    ax.set_title('A. Genetic Covariance Matrix (G)', color=PAL['G'])
    
    # Add annotation
    ax.text(0.5, -0.25, 'From breeding design,\npedigree, or genomics',
           transform=ax.transAxes, ha='center', fontsize=10, style='italic')
    
    # --- Panel B: The P matrix ---
    ax = fig.add_subplot(gs[0, 1])
    
    P_example = np.array([[0.85, 0.18, 0.10],
                          [0.18, 0.62, 0.15],
                          [0.10, 0.15, 0.48]])
    
    im = ax.imshow(P_example, cmap='Greens', aspect='equal', vmin=0, vmax=1.0)
    
    for i in range(3):
        for j in range(3):
            color = 'white' if P_example[i,j] > 0.5 else PAL['text']
            ax.text(j, i, f'{P_example[i,j]:.2f}', ha='center', va='center',
                   fontsize=12, fontweight='bold', color=color)
    
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(['Trait 1', 'Trait 2', 'Trait 3'], fontsize=10)
    ax.set_yticklabels(['Trait 1', 'Trait 2', 'Trait 3'], fontsize=10)
    ax.set_title('B. Phenotypic Covariance Matrix (P)', color=PAL['P'])
    
    ax.text(0.5, -0.25, 'From phenotypic\nmeasurements',
           transform=ax.transAxes, ha='center', fontsize=10, style='italic')
    
    # --- Panel C: Optional - Selection target ---
    ax = fig.add_subplot(gs[0, 2])
    ax.axis('off')
    
    ax.text(0.5, 0.85, 'C. Selection Target (optional)', fontsize=14,
           ha='center', fontweight='bold', transform=ax.transAxes,
           color=PAL['beta'])
    
    # Draw a vector
    ax.annotate('', xy=(0.8, 0.5), xytext=(0.2, 0.5),
               arrowprops=dict(arrowstyle='->', lw=3, color=PAL['beta']),
               transform=ax.transAxes)
    
    ax.text(0.5, 0.38, 'β = [0.5, 0.3, −0.2]', fontsize=14, ha='center',
           transform=ax.transAxes, fontfamily='monospace', color=PAL['beta'])
    
    ax.text(0.5, 0.15, 'Direction of desired\nimprovement or\nselection gradient',
           transform=ax.transAxes, ha='center', fontsize=10, style='italic')
    
    # --- Bottom row: What you get ---
    ax = fig.add_subplot(gs[1, :])
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, 'What You Can Compute', fontsize=16, ha='center',
           fontweight='bold', transform=ax.transAxes, color=PAL['axis'])
    
    # Three output boxes
    box_width = 0.28
    box_height = 0.55
    box_y = 0.15
    
    # Box 1: CV(h²)
    rect1 = FancyBboxPatch((0.04, box_y), box_width, box_height,
                           boxstyle="round,pad=0.02", 
                           facecolor=pal_alpha('G', 0.15),
                           edgecolor=PAL['G'], linewidth=2,
                           transform=ax.transAxes)
    ax.add_patch(rect1)
    ax.text(0.04 + box_width/2, box_y + box_height - 0.08, 'CV(h²)',
           fontsize=14, ha='center', fontweight='bold', transform=ax.transAxes)
    ax.text(0.04 + box_width/2, box_y + 0.25,
           'Overall constraint\npotential\n\nHow much does h²\nvary with direction?',
           fontsize=10, ha='center', va='center', transform=ax.transAxes)
    
    # Box 2: h² range
    rect2 = FancyBboxPatch((0.36, box_y), box_width, box_height,
                           boxstyle="round,pad=0.02",
                           facecolor=pal_alpha('P', 0.15),
                           edgecolor=PAL['P'], linewidth=2,
                           transform=ax.transAxes)
    ax.add_patch(rect2)
    ax.text(0.36 + box_width/2, box_y + box_height - 0.08, 'h² Range',
           fontsize=14, ha='center', fontweight='bold', transform=ax.transAxes)
    ax.text(0.36 + box_width/2, box_y + 0.25,
           'Best & worst directions\n\nMin h² to Max h²\nalong eigenvectors',
           fontsize=10, ha='center', va='center', transform=ax.transAxes)
    
    # Box 3: h²(β)
    rect3 = FancyBboxPatch((0.68, box_y), box_width, box_height,
                           boxstyle="round,pad=0.02",
                           facecolor=pal_alpha('beta', 0.15),
                           edgecolor=PAL['beta'], linewidth=2,
                           transform=ax.transAxes)
    ax.add_patch(rect3)
    ax.text(0.68 + box_width/2, box_y + box_height - 0.08, 'h²(β)',
           fontsize=14, ha='center', fontweight='bold', transform=ax.transAxes)
    ax.text(0.68 + box_width/2, box_y + 0.25,
           'Target-specific h²\n\nIs your selection\ntarget in a trap?',
           fontsize=10, ha='center', va='center', transform=ax.transAxes)
    
    # Arrows from top to bottom
    ax.annotate('', xy=(0.18, 0.78), xytext=(0.18, 0.95),
               arrowprops=dict(arrowstyle='->', lw=2, color=PAL['grid']),
               transform=ax.transAxes)
    ax.annotate('', xy=(0.5, 0.78), xytext=(0.5, 0.95),
               arrowprops=dict(arrowstyle='->', lw=2, color=PAL['grid']),
               transform=ax.transAxes)
    ax.annotate('', xy=(0.82, 0.78), xytext=(0.82, 0.95),
               arrowprops=dict(arrowstyle='->', lw=2, color=PAL['grid']),
               transform=ax.transAxes)
    
    ax.text(0.18, 0.86, 'G + P', fontsize=11, ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.86, 'G + P', fontsize=11, ha='center', transform=ax.transAxes)
    ax.text(0.82, 0.86, 'G + P + β', fontsize=11, ha='center', transform=ax.transAxes)
    
    plt.suptitle('Figure E1: What Data Do You Need?', fontsize=18, 
                fontweight='bold', y=0.98)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# FIGURE E2: STEP-BY-STEP CALCULATION
# =============================================================================

def figure_E2_worked_example(save_path=None):
    """
    Figure E2: A worked example computing CV(h²).
    
    Shows the calculation step-by-step with real numbers.
    """
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 4, hspace=0.4, wspace=0.25,
                         height_ratios=[1, 1.2])
    
    # Example data (2D for clarity)
    G = np.array([[0.48, 0.18], [0.18, 0.22]])
    P = np.array([[0.92, 0.20], [0.20, 0.58]])
    
    # Compute everything
    results = compute_cv_h2(G, P)
    
    # --- Step 1: Your matrices ---
    ax = fig.add_subplot(gs[0, 0])
    ax.axis('off')
    
    ax.text(0.5, 0.95, 'Step 1: Your Matrices', fontsize=13,
           ha='center', fontweight='bold', transform=ax.transAxes,
           color=PAL['axis'])
    
    # G matrix
    ax.text(0.5, 0.75, 'G =', fontsize=14, ha='right', transform=ax.transAxes,
           color=PAL['G'], fontweight='bold')
    g_text = f'[{G[0,0]:.2f}  {G[0,1]:.2f}]\n[{G[1,0]:.2f}  {G[1,1]:.2f}]'
    ax.text(0.55, 0.75, g_text, fontsize=12, ha='left', va='center',
           transform=ax.transAxes, fontfamily='monospace', color=PAL['G'])
    
    # P matrix
    ax.text(0.5, 0.45, 'P =', fontsize=14, ha='right', transform=ax.transAxes,
           color=PAL['P'], fontweight='bold')
    p_text = f'[{P[0,0]:.2f}  {P[0,1]:.2f}]\n[{P[1,0]:.2f}  {P[1,1]:.2f}]'
    ax.text(0.55, 0.45, p_text, fontsize=12, ha='left', va='center',
           transform=ax.transAxes, fontfamily='monospace', color=PAL['P'])
    
    # Visual
    ax_vis = ax.inset_axes([0.15, 0.02, 0.7, 0.35])
    w_G, h_G, angle_G, _, _ = get_ellipse_params(G)
    w_P, h_P, angle_P, _, _ = get_ellipse_params(P)
    
    ellipse_G = Ellipse((0, 0), w_G*0.8, h_G*0.8, angle=angle_G,
                        fill=True, facecolor=pal_alpha('G', 0.3),
                        edgecolor=PAL['G'], linewidth=2)
    ellipse_P = Ellipse((0, 0), w_P*0.8, h_P*0.8, angle=angle_P,
                        fill=False, edgecolor=PAL['P'], linewidth=2, linestyle='--')
    ax_vis.add_patch(ellipse_P)
    ax_vis.add_patch(ellipse_G)
    ax_vis.set_xlim(-1.2, 1.2)
    ax_vis.set_ylim(-0.9, 0.9)
    ax_vis.set_aspect('equal')
    ax_vis.axis('off')
    
    # --- Step 2: P-whiten ---
    ax = fig.add_subplot(gs[0, 1])
    ax.axis('off')
    
    ax.text(0.5, 0.95, 'Step 2: P-Whiten', fontsize=13,
           ha='center', fontweight='bold', transform=ax.transAxes,
           color=PAL['axis'])
    
    ax.text(0.5, 0.72, 'G* = P⁻¹/² G P⁻¹/²', fontsize=13, ha='center',
           transform=ax.transAxes, fontfamily='monospace')
    
    Gstar = results['Gstar']
    gstar_text = f'G* = [{Gstar[0,0]:.2f}  {Gstar[0,1]:.2f}]\n     [{Gstar[1,0]:.2f}  {Gstar[1,1]:.2f}]'
    ax.text(0.5, 0.5, gstar_text, fontsize=11, ha='center',
           transform=ax.transAxes, fontfamily='monospace',
           color=PAL['misaligned'])
    
    # Visual - G* with unit circle
    ax_vis = ax.inset_axes([0.15, 0.02, 0.7, 0.4])
    w_Gs, h_Gs, angle_Gs, _, _ = get_ellipse_params(Gstar)
    
    circle = Circle((0, 0), 0.8, fill=False, edgecolor=PAL['P'],
                   linewidth=2, linestyle='--')
    ellipse_Gs = Ellipse((0, 0), w_Gs*0.8, h_Gs*0.8, angle=angle_Gs,
                        fill=True, facecolor=pal_alpha('misaligned', 0.3),
                        edgecolor=PAL['misaligned'], linewidth=2)
    ax_vis.add_patch(circle)
    ax_vis.add_patch(ellipse_Gs)
    ax_vis.set_xlim(-1.1, 1.1)
    ax_vis.set_ylim(-1.1, 1.1)
    ax_vis.set_aspect('equal')
    ax_vis.axis('off')
    ax_vis.text(0.95, 0.1, 'P* = I', fontsize=9, color=PAL['P'],
               transform=ax_vis.transAxes, ha='right')
    
    # --- Step 3: Eigenvalues ---
    ax = fig.add_subplot(gs[0, 2])
    ax.axis('off')
    
    ax.text(0.5, 0.95, 'Step 3: Eigenvalues', fontsize=13,
           ha='center', fontweight='bold', transform=ax.transAxes,
           color=PAL['axis'])
    
    eigvals = results['eigvals']
    ax.text(0.5, 0.7, 'Eigenvalues of G*:', fontsize=11, ha='center',
           transform=ax.transAxes)
    ax.text(0.5, 0.55, f'λ₁* = {eigvals[0]:.3f}  (min h²)', fontsize=12,
           ha='center', transform=ax.transAxes, color=PAL['trap'],
           fontfamily='monospace')
    ax.text(0.5, 0.42, f'λ₂* = {eigvals[1]:.3f}  (max h²)', fontsize=12,
           ha='center', transform=ax.transAxes, color=PAL['Highway'],
           fontfamily='monospace')
    
    # Bar chart
    ax_bar = ax.inset_axes([0.2, 0.02, 0.6, 0.35])
    bars = ax_bar.bar([0, 1], eigvals, color=[PAL['trap'], PAL['Highway']],
                      edgecolor='black', linewidth=1, width=0.6)
    ax_bar.set_xticks([0, 1])
    ax_bar.set_xticklabels(['λ₁*', 'λ₂*'], fontsize=10)
    ax_bar.set_ylabel('h²', fontsize=10)
    ax_bar.set_ylim(0, 0.8)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    
    # --- Step 4: Compute CV ---
    ax = fig.add_subplot(gs[0, 3])
    ax.axis('off')
    
    ax.text(0.5, 0.95, 'Step 4: Compute CV', fontsize=13,
           ha='center', fontweight='bold', transform=ax.transAxes,
           color=PAL['axis'])
    
    mean_h2 = results['mean_h2']
    vrel = results['vrel']
    cv_h2 = results['cv_h2']
    
    calcs = (
        f"Mean(λ*) = {mean_h2:.3f}\n\n"
        f"Vrel = Var(λ*)/Mean² = {vrel:.3f}\n\n"
        f"CV(h²) = √(2·Vrel/(p+2))\n"
        f"       = √(2×{vrel:.3f}/4)\n"
        f"       = {cv_h2:.3f}"
    )
    ax.text(0.5, 0.5, calcs, fontsize=11, ha='center', va='center',
           transform=ax.transAxes, fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor=PAL['heatmap_low'],
                    edgecolor=PAL['axis'], linewidth=2))
    
    # --- Bottom: Interpretation ---
    ax = fig.add_subplot(gs[1, :])
    ax.axis('off')
    
    ax.text(0.5, 0.95, 'Step 5: Interpret Your Result', fontsize=14,
           ha='center', fontweight='bold', transform=ax.transAxes,
           color=PAL['axis'])
    
    # Result summary
    summary = (
        f"Your CV(h²) = {cv_h2:.2f}\n\n"
        f"Heritability ranges from {eigvals[0]:.2f} (worst direction) to {eigvals[1]:.2f} (best direction)\n"
        f"This is a {cv_h2*100:.0f}% coefficient of variation"
    )
    ax.text(0.5, 0.72, summary, fontsize=13, ha='center',
           transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor=pal_alpha('G', 0.1),
                    edgecolor=PAL['G'], linewidth=2))
    
    # Interpretation scale
    scale_y = 0.35
    
    # Draw scale bar
    ax.annotate('', xy=(0.85, scale_y), xytext=(0.15, scale_y),
               arrowprops=dict(arrowstyle='-', lw=3, color=PAL['grid']),
               transform=ax.transAxes)
    
    # Add tick marks and labels
    ticks = [0, 0.1, 0.2, 0.3, 0.4]
    tick_x = [0.15 + t * 1.75 for t in ticks]
    
    for x, t in zip(tick_x, ticks):
        ax.plot([x, x], [scale_y - 0.02, scale_y + 0.02], 
               color=PAL['text'], lw=2, transform=ax.transAxes)
        ax.text(x, scale_y - 0.06, f'{t:.1f}', ha='center', fontsize=10,
               transform=ax.transAxes)
    
    ax.text(0.5, scale_y - 0.12, 'CV(h²)', ha='center', fontsize=12,
           fontweight='bold', transform=ax.transAxes)
    
    # Color zones
    zone_height = 0.08
    # Low (green)
    rect_low = Rectangle((0.15, scale_y + 0.04), 0.175, zone_height,
                         facecolor=PAL['success'], alpha=0.5, transform=ax.transAxes)
    ax.add_patch(rect_low)
    ax.text(0.24, scale_y + 0.04 + zone_height/2, 'Low\nConstraint', 
           ha='center', va='center', fontsize=9, transform=ax.transAxes)
    
    # Medium (yellow)
    rect_med = Rectangle((0.325, scale_y + 0.04), 0.175, zone_height,
                         facecolor=PAL['warning'], alpha=0.5, transform=ax.transAxes)
    ax.add_patch(rect_med)
    ax.text(0.41, scale_y + 0.04 + zone_height/2, 'Moderate', 
           ha='center', va='center', fontsize=9, transform=ax.transAxes)
    
    # High (red)
    rect_high = Rectangle((0.5, scale_y + 0.04), 0.35, zone_height,
                          facecolor=PAL['danger'], alpha=0.5, transform=ax.transAxes)
    ax.add_patch(rect_high)
    ax.text(0.67, scale_y + 0.04 + zone_height/2, 'High Constraint Potential', 
           ha='center', va='center', fontsize=9, transform=ax.transAxes)
    
    # Mark the current value
    current_x = 0.15 + cv_h2 * 1.75
    ax.plot(current_x, scale_y, 'v', markersize=15, color=PAL['beta'],
           transform=ax.transAxes, zorder=10)
    ax.text(current_x, scale_y + 0.16, f'YOU\n({cv_h2:.2f})', ha='center',
           fontsize=10, fontweight='bold', color=PAL['beta'],
           transform=ax.transAxes)
    
    plt.suptitle('Figure E2: A Worked Example', fontsize=18,
                fontweight='bold', y=0.98)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# FIGURE E3: INTERPRETING YOUR CV VALUE
# =============================================================================

def figure_E3_interpretation(save_path=None):
    """
    Figure E3: What does your CV(h²) value mean?
    
    Shows three scenarios with practical implications.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    # Three scenarios
    scenarios = [
        {'name': 'Low Constraint', 'cv': 0.08, 'eigvals': [0.48, 0.52],
         'color': PAL['success'], 'interpretation': 
         'All directions have\nsimilar heritability.\n\nSelection works well\nregardless of target.'},
        {'name': 'Moderate Constraint', 'cv': 0.20, 'eigvals': [0.35, 0.65],
         'color': PAL['warning'], 'interpretation':
         'Some directions are\nbetter than others.\n\nConsider your target\ncarefully.'},
        {'name': 'High Constraint', 'cv': 0.35, 'eigvals': [0.20, 0.80],
         'color': PAL['danger'], 'interpretation':
         'Strong directional\ndependence!\n\nSelection target\ncritically important.'},
    ]
    
    for ax, scenario in zip(axes, scenarios):
        # Create G* with these eigenvalues
        eigvals = np.array(scenario['eigvals'])
        Gstar = np.diag(eigvals)
        
        # Draw polar plot of h²
        thetas = np.linspace(0, 2*np.pi, 100)
        h2_values = []
        for theta in thetas:
            beta = np.array([np.cos(theta), np.sin(theta)])
            h2_values.append(beta @ Gstar @ beta)
        h2_values = np.array(h2_values)
        
        # Scale for visibility
        r = 0.3 + 0.6 * h2_values
        x = r * np.cos(thetas)
        y = r * np.sin(thetas)
        
        # Fill the area
        ax.fill(x, y, color=scenario['color'], alpha=0.3)
        ax.plot(x, y, color=scenario['color'], linewidth=3)
        
        # Reference circle (mean h²)
        mean_h2 = np.mean(eigvals)
        r_mean = 0.3 + 0.6 * mean_h2
        circle = Circle((0, 0), r_mean, fill=False, color=PAL['grid'],
                        linewidth=2, linestyle=':')
        ax.add_patch(circle)
        
        # Mark high and low directions
        ax.plot([0, 0.3 + 0.6*eigvals[1]], [0, 0], 'o-', 
               color=PAL['Highway'], linewidth=2, markersize=8)
        ax.plot([0, 0], [0, 0.3 + 0.6*eigvals[0]], 'o-',
               color=PAL['trap'], linewidth=2, markersize=8)
        
        ax.text(0.3 + 0.6*eigvals[1] + 0.1, 0.05, f'h²={eigvals[1]:.2f}',
               fontsize=10, color=PAL['Highway'])
        ax.text(0.05, 0.3 + 0.6*eigvals[0] + 0.1, f'h²={eigvals[0]:.2f}',
               fontsize=10, color=PAL['trap'])
        
        # Title
        ax.set_title(f"{scenario['name']}\nCV(h²) = {scenario['cv']:.2f}",
                    fontsize=14, fontweight='bold', color=scenario['color'])
        
        # Interpretation box
        ax.text(0.5, -0.15, scenario['interpretation'],
               transform=ax.transAxes, ha='center', va='top', fontsize=11,
               bbox=dict(boxstyle='round', facecolor='white', 
                        edgecolor=scenario['color'], linewidth=2))
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
    
    plt.suptitle('Figure E3: Interpreting Your CV(h²) Value', fontsize=18,
                fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# FIGURE E4: IS YOUR TARGET IN A TRAP?
# =============================================================================

def figure_E4_target_evaluation(save_path=None):
    """
    Figure E4: Evaluating a specific selection target.
    
    Shows how to check if your breeding goal is in a constraint trap.
    """
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)
    
    # Example system
    G = np.array([[0.6, 0.25], [0.25, 0.3]])
    P = np.array([[1.0, 0.25], [0.25, 0.7]])
    
    results = compute_cv_h2(G, P)
    
    # Two selection targets
    beta_good = np.array([1, 0.3])  # Along high-h² direction
    beta_good = beta_good / np.linalg.norm(beta_good)
    
    beta_bad = np.array([0.2, 1])   # Along low-h² direction
    beta_bad = beta_bad / np.linalg.norm(beta_bad)
    
    h2_good = directional_h2(beta_good, G, P)
    h2_bad = directional_h2(beta_bad, G, P)
    
    # --- Panel A: Good target ---
    ax = fig.add_subplot(gs[0, 0])
    
    w_G, h_G, angle_G, _, _ = get_ellipse_params(G)
    w_P, h_P, angle_P, _, _ = get_ellipse_params(P)
    
    ellipse_G = Ellipse((0, 0), w_G, h_G, angle=angle_G,
                        fill=True, facecolor=pal_alpha('G', 0.3),
                        edgecolor=PAL['G'], linewidth=2)
    ellipse_P = Ellipse((0, 0), w_P, h_P, angle=angle_P,
                        fill=False, edgecolor=PAL['P'], linewidth=2, linestyle='--')
    ax.add_patch(ellipse_P)
    ax.add_patch(ellipse_G)
    
    # Selection arrow
    ax.annotate('', xy=beta_good*1.3, xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', lw=3, color=PAL['beta']))
    
    # Response arrow
    response = G @ beta_good
    response = response / np.linalg.norm(response) * 1.1
    ax.annotate('', xy=response, xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', lw=3, color=PAL['response']))
    
    ax.text(beta_good[0]*1.4, beta_good[1]*1.4, 'β', fontsize=14,
           color=PAL['beta'], fontweight='bold')
    ax.text(response[0]*1.15, response[1]*1.15, 'Gβ', fontsize=14,
           color=PAL['response'], fontweight='bold')
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Result box
    ax.text(0.5, -0.08, f'h²(β) = {h2_good:.2f}', transform=ax.transAxes,
           ha='center', fontsize=16, fontweight='bold', color=PAL['success'],
           bbox=dict(boxstyle='round', facecolor=pal_alpha('success', 0.2),
                    edgecolor=PAL['success'], linewidth=2))
    
    ax.set_title('A. Good Target: High h²(β)', fontsize=14, fontweight='bold',
                color=PAL['success'])
    
    # --- Panel B: Bad target (trap!) ---
    ax = fig.add_subplot(gs[0, 1])
    
    ellipse_G = Ellipse((0, 0), w_G, h_G, angle=angle_G,
                        fill=True, facecolor=pal_alpha('G', 0.3),
                        edgecolor=PAL['G'], linewidth=2)
    ellipse_P = Ellipse((0, 0), w_P, h_P, angle=angle_P,
                        fill=False, edgecolor=PAL['P'], linewidth=2, linestyle='--')
    ax.add_patch(ellipse_P)
    ax.add_patch(ellipse_G)
    
    # Selection arrow
    ax.annotate('', xy=beta_bad*1.3, xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', lw=3, color=PAL['beta']))
    
    # Response arrow (shorter!)
    response = G @ beta_bad
    response = response / np.linalg.norm(response) * 0.7
    ax.annotate('', xy=response, xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', lw=3, color=PAL['response']))
    
    ax.text(beta_bad[0]*1.4, beta_bad[1]*1.4, 'β', fontsize=14,
           color=PAL['beta'], fontweight='bold')
    ax.text(response[0]*1.2, response[1]*1.2, 'Gβ\n(weak!)', fontsize=12,
           color=PAL['response'], fontweight='bold', ha='center')
    
    # Danger zone shading
    from matplotlib.patches import Wedge
    wedge = Wedge((0, 0), 1.3, 60, 100, facecolor=PAL['trap'], alpha=0.15)
    ax.add_patch(wedge)
    ax.text(0.2, 1.0, 'TRAP\nZONE', fontsize=10, color=PAL['trap'],
           fontweight='bold', ha='center')
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')
    
    ax.text(0.5, -0.08, f'h²(β) = {h2_bad:.2f}', transform=ax.transAxes,
           ha='center', fontsize=16, fontweight='bold', color=PAL['trap'],
           bbox=dict(boxstyle='round', facecolor=pal_alpha('trap', 0.2),
                    edgecolor=PAL['trap'], linewidth=2))
    
    ax.set_title('B. Bad Target: Low h²(β) — CONSTRAINT TRAP!', fontsize=14,
                fontweight='bold', color=PAL['trap'])
    
    # --- Panel C: Decision framework ---
    ax = fig.add_subplot(gs[1, :])
    ax.axis('off')
    
    ax.text(0.5, 0.95, 'C. Decision Framework: Is Your Target in a Trap?',
           fontsize=14, ha='center', fontweight='bold', transform=ax.transAxes)
    
    # Flowchart
    # Box 1: Compute h²(β)
    box1 = FancyBboxPatch((0.05, 0.55), 0.2, 0.25,
                          boxstyle="round,pad=0.02",
                          facecolor=pal_alpha('G', 0.2),
                          edgecolor=PAL['G'], linewidth=2,
                          transform=ax.transAxes)
    ax.add_patch(box1)
    ax.text(0.15, 0.675, 'Compute\nh²(β)', fontsize=12, ha='center', va='center',
           fontweight='bold', transform=ax.transAxes)
    
    # Arrow
    ax.annotate('', xy=(0.3, 0.675), xytext=(0.25, 0.675),
               arrowprops=dict(arrowstyle='->', lw=2, color=PAL['text']),
               transform=ax.transAxes)
    
    # Diamond: Compare to mean
    diamond_x = [0.4, 0.475, 0.55, 0.475, 0.4]
    diamond_y = [0.675, 0.85, 0.675, 0.5, 0.675]
    ax.fill(diamond_x, diamond_y, facecolor=pal_alpha('warning', 0.2),
           edgecolor=PAL['warning'], linewidth=2, transform=ax.transAxes)
    ax.text(0.475, 0.675, 'h²(β) >\nmean h²?', fontsize=10, ha='center', va='center',
           transform=ax.transAxes)
    
    # Yes path
    ax.annotate('', xy=(0.7, 0.85), xytext=(0.55, 0.76),
               arrowprops=dict(arrowstyle='->', lw=2, color=PAL['success']),
               transform=ax.transAxes)
    ax.text(0.6, 0.82, 'YES', fontsize=10, color=PAL['success'], fontweight='bold',
           transform=ax.transAxes)
    
    # Good outcome box
    box_good = FancyBboxPatch((0.65, 0.7), 0.3, 0.25,
                              boxstyle="round,pad=0.02",
                              facecolor=pal_alpha('success', 0.3),
                              edgecolor=PAL['success'], linewidth=2,
                              transform=ax.transAxes)
    ax.add_patch(box_good)
    ax.text(0.8, 0.825, '✓ PROCEED', fontsize=12, ha='center', va='center',
           fontweight='bold', color=PAL['success'], transform=ax.transAxes)
    ax.text(0.8, 0.75, 'Target is favorable', fontsize=10, ha='center',
           transform=ax.transAxes)
    
    # No path
    ax.annotate('', xy=(0.7, 0.45), xytext=(0.55, 0.59),
               arrowprops=dict(arrowstyle='->', lw=2, color=PAL['trap']),
               transform=ax.transAxes)
    ax.text(0.6, 0.48, 'NO', fontsize=10, color=PAL['trap'], fontweight='bold',
           transform=ax.transAxes)
    
    # Bad outcome box
    box_bad = FancyBboxPatch((0.65, 0.2), 0.3, 0.3,
                             boxstyle="round,pad=0.02",
                             facecolor=pal_alpha('trap', 0.2),
                             edgecolor=PAL['trap'], linewidth=2,
                             transform=ax.transAxes)
    ax.add_patch(box_bad)
    ax.text(0.8, 0.42, '⚠ CAUTION', fontsize=12, ha='center', va='center',
           fontweight='bold', color=PAL['trap'], transform=ax.transAxes)
    ax.text(0.8, 0.32, 'Consider:\n• Redefine target\n• Accept slower progress\n• Multi-generation plan',
           fontsize=9, ha='center', transform=ax.transAxes)
    
    # Key insight
    ax.text(0.15, 0.15, 
           f'For this example:\n  Mean h² = {results["mean_h2"]:.2f}\n  h²(good target) = {h2_good:.2f} ✓\n  h²(bad target) = {h2_bad:.2f} ✗',
           fontsize=11, transform=ax.transAxes, fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor=PAL['heatmap_low'],
                    edgecolor=PAL['grid'], linewidth=1))
    
    plt.suptitle('Figure E4: Is Your Selection Target in a Constraint Trap?',
                fontsize=18, fontweight='bold', y=0.98)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# FIGURE E5: COMPARING POPULATIONS
# =============================================================================

def figure_E5_comparing_populations(save_path=None):
    """
    Figure E5: Comparing constraint across populations or environments.
    
    Shows how to use CV(h²) for comparative analysis.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    
    # Create three populations with different constraint levels
    np.random.seed(42)
    
    populations = [
        {'name': 'Population A\n(Lab strain)', 
         'G': np.array([[0.5, 0.2], [0.2, 0.45]]),
         'P': np.array([[0.7, 0.22], [0.22, 0.6]])},
        {'name': 'Population B\n(Field site 1)',
         'G': np.array([[0.6, 0.15], [0.15, 0.25]]),
         'P': np.array([[1.0, 0.18], [0.18, 0.65]])},
        {'name': 'Population C\n(Field site 2)',
         'G': np.array([[0.7, 0.1], [0.1, 0.15]]),
         'P': np.array([[1.3, 0.12], [0.12, 0.8]])},
    ]
    
    # Compute CV for each
    for pop in populations:
        results = compute_cv_h2(pop['G'], pop['P'])
        pop['cv'] = results['cv_h2']
        pop['mean_h2'] = results['mean_h2']
        pop['min_h2'] = results['min_h2']
        pop['max_h2'] = results['max_h2']
    
    # --- Each population as a polar plot ---
    for ax, pop in zip(axes, populations):
        G = pop['G']
        P = pop['P']
        
        # Compute h² for all directions
        thetas = np.linspace(0, 2*np.pi, 100)
        h2_values = []
        for theta in thetas:
            beta = np.array([np.cos(theta), np.sin(theta)])
            h2_values.append(directional_h2(beta, G, P))
        h2_values = np.array(h2_values)
        
        # Plot
        r = 0.2 + 0.7 * h2_values
        x = r * np.cos(thetas)
        y = r * np.sin(thetas)
        
        # Color based on CV
        if pop['cv'] < 0.15:
            color = PAL['success']
        elif pop['cv'] < 0.25:
            color = PAL['warning']
        else:
            color = PAL['trap']
        
        ax.fill(x, y, color=color, alpha=0.3)
        ax.plot(x, y, color=color, linewidth=3)
        
        # Reference circle
        mean_r = 0.2 + 0.7 * pop['mean_h2']
        circle = Circle((0, 0), mean_r, fill=False, color=PAL['grid'],
                        linewidth=2, linestyle=':')
        ax.add_patch(circle)
        
        # Title with stats
        ax.set_title(f"{pop['name']}", fontsize=13, fontweight='bold')
        
        # Stats box
        stats_text = (f"CV(h²) = {pop['cv']:.2f}\n"
                     f"Mean h² = {pop['mean_h2']:.2f}\n"
                     f"Range: {pop['min_h2']:.2f} – {pop['max_h2']:.2f}")
        ax.text(0.5, -0.12, stats_text, transform=ax.transAxes,
               ha='center', fontsize=10, fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='white',
                        edgecolor=color, linewidth=2))
        
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
        ax.axis('off')
    
    # Add comparison arrow and text below
    fig.text(0.5, 0.02, 
            '← Lower constraint                                         Higher constraint →\n'
            'Selection target matters less          Selection target matters more',
            ha='center', fontsize=11, style='italic')
    
    plt.suptitle('Figure E5: Comparing Constraint Across Populations',
                fontsize=18, fontweight='bold', y=1.0)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# FIGURE E6: SUMMARY DECISION GUIDE
# =============================================================================

def figure_E6_decision_guide(save_path=None):
    """
    Figure E6: A practical decision guide for empiricists.
    
    Summary flowchart for using directional heritability in practice.
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.97, 'Practical Decision Guide', fontsize=20,
           ha='center', fontweight='bold', transform=ax.transAxes,
           color=PAL['axis'])
    
    # --- STEP 1: Estimate matrices ---
    box1 = FancyBboxPatch((0.35, 0.82), 0.3, 0.1,
                          boxstyle="round,pad=0.015",
                          facecolor=pal_alpha('G', 0.2),
                          edgecolor=PAL['G'], linewidth=2.5,
                          transform=ax.transAxes)
    ax.add_patch(box1)
    ax.text(0.5, 0.87, '1. Estimate G and P matrices', fontsize=13,
           ha='center', va='center', fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.835, 'breeding design, genomics, phenotyping', fontsize=10,
           ha='center', style='italic', transform=ax.transAxes)
    
    # Arrow down
    ax.annotate('', xy=(0.5, 0.75), xytext=(0.5, 0.82),
               arrowprops=dict(arrowstyle='->', lw=2.5, color=PAL['text']),
               transform=ax.transAxes)
    
    # --- STEP 2: Compute CV ---
    box2 = FancyBboxPatch((0.35, 0.65), 0.3, 0.1,
                          boxstyle="round,pad=0.015",
                          facecolor=pal_alpha('P', 0.2),
                          edgecolor=PAL['P'], linewidth=2.5,
                          transform=ax.transAxes)
    ax.add_patch(box2)
    ax.text(0.5, 0.7, '2. Compute CV(h²)', fontsize=13,
           ha='center', va='center', fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.665, 'overall constraint potential', fontsize=10,
           ha='center', style='italic', transform=ax.transAxes)
    
    # Arrow down
    ax.annotate('', xy=(0.5, 0.58), xytext=(0.5, 0.65),
               arrowprops=dict(arrowstyle='->', lw=2.5, color=PAL['text']),
               transform=ax.transAxes)
    
    # --- STEP 3: Decision diamond ---
    diamond_x = np.array([0.5, 0.6, 0.5, 0.4, 0.5])
    diamond_y = np.array([0.58, 0.5, 0.42, 0.5, 0.58])
    ax.fill(diamond_x, diamond_y, facecolor=pal_alpha('warning', 0.3),
           edgecolor=PAL['warning'], linewidth=2.5, transform=ax.transAxes)
    ax.text(0.5, 0.5, 'CV(h²)\nhigh?', fontsize=12, ha='center', va='center',
           fontweight='bold', transform=ax.transAxes)
    
    # --- LOW CV PATH (left) ---
    ax.annotate('', xy=(0.22, 0.5), xytext=(0.4, 0.5),
               arrowprops=dict(arrowstyle='->', lw=2.5, color=PAL['success']),
               transform=ax.transAxes)
    ax.text(0.31, 0.52, 'LOW', fontsize=11, color=PAL['success'],
           fontweight='bold', transform=ax.transAxes)
    
    box_low = FancyBboxPatch((0.02, 0.35), 0.2, 0.3,
                             boxstyle="round,pad=0.015",
                             facecolor=pal_alpha('success', 0.2),
                             edgecolor=PAL['success'], linewidth=2.5,
                             transform=ax.transAxes)
    ax.add_patch(box_low)
    ax.text(0.12, 0.6, 'Low Constraint', fontsize=12, ha='center',
           fontweight='bold', color=PAL['success'], transform=ax.transAxes)
    low_text = ('• All directions similar\n'
                '• Target choice less critical\n'
                '• Standard breeding/\n  selection works well\n'
                '• Rapid progress expected')
    ax.text(0.12, 0.47, low_text, fontsize=9, ha='center', va='center',
           transform=ax.transAxes)
    
    # --- HIGH CV PATH (right) ---
    ax.annotate('', xy=(0.78, 0.5), xytext=(0.6, 0.5),
               arrowprops=dict(arrowstyle='->', lw=2.5, color=PAL['trap']),
               transform=ax.transAxes)
    ax.text(0.66, 0.52, 'HIGH', fontsize=11, color=PAL['trap'],
           fontweight='bold', transform=ax.transAxes)
    
    box_high = FancyBboxPatch((0.78, 0.35), 0.2, 0.3,
                              boxstyle="round,pad=0.015",
                              facecolor=pal_alpha('trap', 0.15),
                              edgecolor=PAL['trap'], linewidth=2.5,
                              transform=ax.transAxes)
    ax.add_patch(box_high)
    ax.text(0.88, 0.6, 'High Constraint', fontsize=12, ha='center',
           fontweight='bold', color=PAL['trap'], transform=ax.transAxes)
    high_text = ('• Directions differ greatly\n'
                 '• Must evaluate target!\n'
                 '• Compute h²(β) for your\n  specific goal\n'
                 '• Consider alternatives')
    ax.text(0.88, 0.47, high_text, fontsize=9, ha='center', va='center',
           transform=ax.transAxes)
    
    # Arrow from high CV to target evaluation
    ax.annotate('', xy=(0.88, 0.28), xytext=(0.88, 0.35),
               arrowprops=dict(arrowstyle='->', lw=2.5, color=PAL['trap']),
               transform=ax.transAxes)
    
    # --- STEP 4: Target evaluation (for high CV only) ---
    box_target = FancyBboxPatch((0.73, 0.12), 0.25, 0.16,
                                boxstyle="round,pad=0.015",
                                facecolor=pal_alpha('beta', 0.2),
                                edgecolor=PAL['beta'], linewidth=2.5,
                                transform=ax.transAxes)
    ax.add_patch(box_target)
    ax.text(0.855, 0.24, '3. Evaluate h²(β)', fontsize=12, ha='center',
           fontweight='bold', transform=ax.transAxes)
    ax.text(0.855, 0.17, 'for your specific\nselection target', fontsize=10,
           ha='center', style='italic', transform=ax.transAxes)
    
    # Outcomes from target evaluation
    ax.annotate('', xy=(0.55, 0.08), xytext=(0.73, 0.15),
               arrowprops=dict(arrowstyle='->', lw=2, color=PAL['text']),
               transform=ax.transAxes)
    
    # Final outcomes
    ax.text(0.35, 0.08, 'h²(β) > mean h²:', fontsize=10, ha='right',
           transform=ax.transAxes, fontweight='bold')
    ax.text(0.36, 0.08, 'Proceed with confidence', fontsize=10, ha='left',
           transform=ax.transAxes, color=PAL['success'])
    
    ax.text(0.35, 0.03, 'h²(β) < mean h²:', fontsize=10, ha='right',
           transform=ax.transAxes, fontweight='bold')
    ax.text(0.36, 0.03, 'Reconsider target or expect slower progress', fontsize=10,
           ha='left', transform=ax.transAxes, color=PAL['trap'])
    
    # Key insight box at bottom
    insight_box = FancyBboxPatch((0.15, -0.08), 0.7, 0.1,
                                 boxstyle="round,pad=0.02",
                                 facecolor=PAL['heatmap_low'],
                                 edgecolor=PAL['axis'], linewidth=2,
                                 transform=ax.transAxes)
    ax.add_patch(insight_box)
    ax.text(0.5, -0.03,
           'KEY: CV(h²) tells you WHETHER direction matters. h²(β) tells you IF your direction is good.',
           fontsize=11, ha='center', va='center', fontweight='bold',
           transform=ax.transAxes)
    
    plt.suptitle('Figure E6: Decision Guide for Empiricists', fontsize=18,
                fontweight='bold', y=0.99)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# MAIN: Generate all empiricist figures
# =============================================================================

if __name__ == "__main__":
    output_dir = os.getcwd()
    print(f"Generating empiricist figures in: {output_dir}\n")
    
    print("Figure E1: What Data Do You Need?")
    figure_E1_what_data(os.path.join(output_dir, 'empiricist_fig1_what_data.png'))
    plt.close()
    
    print("\nFigure E2: Worked Example")
    figure_E2_worked_example(os.path.join(output_dir, 'empiricist_fig2_worked_example.png'))
    plt.close()
    
    print("\nFigure E3: Interpreting CV")
    figure_E3_interpretation(os.path.join(output_dir, 'empiricist_fig3_interpretation.png'))
    plt.close()
    
    print("\nFigure E4: Target Evaluation")
    figure_E4_target_evaluation(os.path.join(output_dir, 'empiricist_fig4_target_evaluation.png'))
    plt.close()
    
    print("\nFigure E5: Comparing Populations")
    figure_E5_comparing_populations(os.path.join(output_dir, 'empiricist_fig5_comparing_pops.png'))
    plt.close()
    
    print("\nFigure E6: Decision Guide")
    figure_E6_decision_guide(os.path.join(output_dir, 'empiricist_fig6_decision_guide.png'))
    plt.close()
    
    print("\n" + "="*60)
    print("ALL EMPIRICIST FIGURES GENERATED!")
    print("="*60)
    print("""
    Figure E1: What data do you need?
              - G and P matrices, optionally selection target
              - What outputs you can compute
    
    Figure E2: Worked example
              - Step-by-step calculation with real numbers
              - Interpretation scale
    
    Figure E3: Interpreting your CV value
              - Low, moderate, high constraint scenarios
              - Practical implications
    
    Figure E4: Is your target in a trap?
              - Good vs bad target comparison
              - Decision flowchart
    
    Figure E5: Comparing populations
              - Side-by-side polar plots
              - Comparative statistics
    
    Figure E6: Decision guide
              - Complete practical flowchart
              - When to worry, what to do
    """)
