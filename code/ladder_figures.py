#!/usr/bin/env python3
"""
Building Intuition: From One Trait to Many
==========================================

A step-by-step ladder from univariate heritability to directional heritability.

Level 1: One trait → one heritability (a simple ratio)
Level 2: Two traits → two heritabilities (one per trait)  
Level 3: Trait combinations → heritability depends on the mix
Level 4: The full picture → ellipses show all directions at once

Author: Daniel Ortiz-Barrientos
Date: December 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib.patches import ConnectionPatch
from numpy.linalg import eigh
import os

# =============================================================================
# MANUSCRIPT COLOR PALETTE
# =============================================================================

PAL = {
    'G': '#2E86AB',           # Deep blue (genetic)
    'P': '#06D6A0',           # Teal (phenotypic)
    'E': '#F4A582',           # Peach (environmental)
    'beta': '#F18F01',        # Warm orange (selection)
    'response': '#C73E1D',    # Warm red
    'axis': '#0B3C5D',        # Dark navy
    'grid': '#D9D9D9',
    'text': '#1A2332',
    'white': '#FFFFFF',
    'success': '#06D6A0',
    'warning': '#F18F01', 
    'danger': '#C73E1D',
}

def pal_alpha(color_name, alpha):
    hex_color = PAL.get(color_name, PAL['G'])
    rgb = tuple(int(hex_color.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
    return (*rgb, alpha)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'axes.linewidth': 1.5,
    'figure.facecolor': 'white',
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


# =============================================================================
# FIGURE L1: ONE TRAIT - THE FOUNDATION
# =============================================================================

def figure_L1_one_trait(save_path=None):
    """
    Level 1: Single trait heritability.
    
    h² = genetic variance / total variance
    
    This is the foundation everyone knows.
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    # Example: Yield in a crop
    V_G = 45   # Genetic variance
    V_E = 55   # Environmental variance  
    V_P = V_G + V_E  # Total phenotypic variance
    h2 = V_G / V_P
    
    # =========================================================================
    # Panel A: The Setup
    # =========================================================================
    ax = axes[0]
    ax.axis('off')
    
    ax.text(0.5, 0.92, 'A. One Trait: Yield', fontsize=14, fontweight='bold',
           ha='center', transform=ax.transAxes)
    
    # Draw a number line representing variance
    line_y = 0.5
    ax.plot([0.1, 0.9], [line_y, line_y], color=PAL['axis'], linewidth=3,
           transform=ax.transAxes)
    
    # Mark zero and total
    ax.plot([0.1, 0.1], [line_y - 0.03, line_y + 0.03], color=PAL['axis'], 
           linewidth=3, transform=ax.transAxes)
    ax.plot([0.9, 0.9], [line_y - 0.03, line_y + 0.03], color=PAL['axis'],
           linewidth=3, transform=ax.transAxes)
    
    ax.text(0.1, line_y - 0.08, '0', fontsize=12, ha='center', transform=ax.transAxes)
    ax.text(0.9, line_y - 0.08, f'{V_P}', fontsize=12, ha='center', transform=ax.transAxes)
    ax.text(0.5, line_y - 0.15, 'Total Phenotypic Variance (V_P)', fontsize=11,
           ha='center', transform=ax.transAxes)
    
    # Shade genetic portion
    g_end = 0.1 + 0.8 * (V_G / V_P)
    rect_g = Rectangle((0.1, line_y + 0.02), g_end - 0.1, 0.12,
                       facecolor=PAL['G'], alpha=0.7, transform=ax.transAxes)
    ax.add_patch(rect_g)
    ax.text((0.1 + g_end)/2, line_y + 0.08, f'Genetic\n(V_G = {V_G})', 
           fontsize=10, ha='center', va='center', color='white', fontweight='bold',
           transform=ax.transAxes)
    
    # Shade environmental portion
    rect_e = Rectangle((g_end, line_y + 0.02), 0.9 - g_end, 0.12,
                       facecolor=PAL['E'], alpha=0.7, transform=ax.transAxes)
    ax.add_patch(rect_e)
    ax.text((g_end + 0.9)/2, line_y + 0.08, f'Environmental\n(V_E = {V_E})',
           fontsize=10, ha='center', va='center', color=PAL['text'], fontweight='bold',
           transform=ax.transAxes)
    
    # Formula
    ax.text(0.5, 0.22, f'h² = V_G / V_P = {V_G} / {V_P} = {h2:.2f}',
           fontsize=14, ha='center', transform=ax.transAxes,
           fontfamily='monospace', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightyellow', 
                    edgecolor=PAL['beta'], linewidth=2))
    
    # =========================================================================
    # Panel B: Bar Chart View
    # =========================================================================
    ax = axes[1]
    
    # Stacked bar
    ax.bar(['Yield'], [V_G], color=PAL['G'], edgecolor='black', linewidth=1.5,
          label='Genetic (V_G)')
    ax.bar(['Yield'], [V_E], bottom=[V_G], color=PAL['E'], edgecolor='black', 
          linewidth=1.5, label='Environmental (V_E)')
    
    # Add h² annotation
    ax.annotate('', xy=(0.35, V_G), xytext=(0.35, V_P),
               arrowprops=dict(arrowstyle='<->', lw=2, color=PAL['text']))
    ax.text(0.45, (V_G + V_P)/2, f'h² = {h2:.0%}', fontsize=14, fontweight='bold',
           va='center')
    
    ax.set_ylabel('Variance', fontsize=12)
    ax.set_ylim(0, 120)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title('B. Variance Components', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # =========================================================================
    # Panel C: What It Means
    # =========================================================================
    ax = axes[2]
    ax.axis('off')
    
    ax.text(0.5, 0.92, 'C. What This Means', fontsize=14, fontweight='bold',
           ha='center', transform=ax.transAxes)
    
    meaning = (
        f"For YIELD alone:\n\n"
        f"• {h2:.0%} of the variation you\n"
        f"  observe is due to genes\n\n"
        f"• {1-h2:.0%} is environmental noise\n"
        f"  (weather, soil, management)\n\n"
        f"• If you select the best plants,\n"
        f"  offspring will be {h2:.0%} of the\n"
        f"  way toward the parents"
    )
    ax.text(0.5, 0.48, meaning, fontsize=12, ha='center', va='center',
           transform=ax.transAxes, linespacing=1.5,
           bbox=dict(boxstyle='round', facecolor=pal_alpha('G', 0.1),
                    edgecolor=PAL['G'], linewidth=2))
    
    ax.text(0.5, 0.05, 'This is the h² you learned in intro genetics!',
           fontsize=11, ha='center', transform=ax.transAxes, style='italic')
    
    plt.suptitle('Level 1: One Trait = One Heritability',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# FIGURE L2: TWO TRAITS - EACH HAS ITS OWN h²
# =============================================================================

def figure_L2_two_traits(save_path=None):
    """
    Level 2: Two traits, each with its own heritability.
    
    Now we have two separate h² values.
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    # Two traits with different heritabilities
    traits = {
        'Yield': {'V_G': 45, 'V_E': 55},           # h² = 0.45
        'Drought\nTolerance': {'V_G': 25, 'V_E': 75}  # h² = 0.25
    }
    
    # =========================================================================
    # Panel A: Two Number Lines
    # =========================================================================
    ax = axes[0]
    ax.axis('off')
    
    ax.text(0.5, 0.95, 'A. Two Traits, Two Lines', fontsize=14, fontweight='bold',
           ha='center', transform=ax.transAxes)
    
    for i, (trait, variances) in enumerate(traits.items()):
        V_G = variances['V_G']
        V_E = variances['V_E']
        V_P = V_G + V_E
        h2 = V_G / V_P
        
        line_y = 0.7 - i * 0.35
        
        # Draw line
        ax.plot([0.1, 0.9], [line_y, line_y], color=PAL['axis'], linewidth=2,
               transform=ax.transAxes)
        
        # Trait label
        ax.text(0.05, line_y, trait, fontsize=11, ha='right', va='center',
               transform=ax.transAxes, fontweight='bold')
        
        # Shade genetic portion
        g_end = 0.1 + 0.8 * (V_G / V_P)
        rect_g = Rectangle((0.1, line_y - 0.06), g_end - 0.1, 0.12,
                           facecolor=PAL['G'], alpha=0.7, transform=ax.transAxes)
        ax.add_patch(rect_g)
        
        # Shade environmental portion
        rect_e = Rectangle((g_end, line_y - 0.06), 0.9 - g_end, 0.12,
                           facecolor=PAL['E'], alpha=0.7, transform=ax.transAxes)
        ax.add_patch(rect_e)
        
        # h² label
        ax.text(0.95, line_y, f'h² = {h2:.2f}', fontsize=12, ha='left', va='center',
               transform=ax.transAxes, fontweight='bold',
               color=PAL['G'] if h2 > 0.35 else PAL['danger'])
    
    # Legend
    ax.text(0.3, 0.15, '■ Genetic', fontsize=10, color=PAL['G'], 
           transform=ax.transAxes, fontweight='bold')
    ax.text(0.6, 0.15, '■ Environmental', fontsize=10, color=PAL['E'],
           transform=ax.transAxes, fontweight='bold')
    
    # =========================================================================
    # Panel B: Side-by-side Bars
    # =========================================================================
    ax = axes[1]
    
    x = np.arange(2)
    width = 0.6
    
    V_Gs = [traits['Yield']['V_G'], traits['Drought\nTolerance']['V_G']]
    V_Es = [traits['Yield']['V_E'], traits['Drought\nTolerance']['V_E']]
    h2s = [V_Gs[i]/(V_Gs[i]+V_Es[i]) for i in range(2)]
    
    ax.bar(x, V_Gs, width, color=PAL['G'], edgecolor='black', linewidth=1.5,
          label='Genetic')
    ax.bar(x, V_Es, width, bottom=V_Gs, color=PAL['E'], edgecolor='black',
          linewidth=1.5, label='Environmental')
    
    # Add h² labels
    for i in range(2):
        ax.text(i, V_Gs[i] + V_Es[i] + 3, f'h² = {h2s[i]:.2f}', 
               ha='center', fontsize=12, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Yield', 'Drought\nTolerance'], fontsize=11)
    ax.set_ylabel('Variance', fontsize=12)
    ax.set_ylim(0, 130)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title('B. Different Traits, Different h²', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # =========================================================================
    # Panel C: The Question
    # =========================================================================
    ax = axes[2]
    ax.axis('off')
    
    ax.text(0.5, 0.92, 'C. So Far, So Good...', fontsize=14, fontweight='bold',
           ha='center', transform=ax.transAxes)
    
    text1 = (
        "Each trait has its own h²:\n\n"
        "• Yield: h² = 0.45\n"
        "  (moderately heritable)\n\n"
        "• Drought Tolerance: h² = 0.25\n"
        "  (less heritable)"
    )
    ax.text(0.5, 0.62, text1, fontsize=11, ha='center', va='center',
           transform=ax.transAxes, linespacing=1.4,
           bbox=dict(boxstyle='round', facecolor='white',
                    edgecolor=PAL['grid'], linewidth=2))
    
    # The key question
    question = (
        "BUT WHAT IF...\n\n"
        "You want to improve\n"
        "BOTH traits at once?\n\n"
        "What's the h² for a\n"
        "COMBINATION of traits?"
    )
    ax.text(0.5, 0.18, question, fontsize=12, ha='center', va='center',
           transform=ax.transAxes, linespacing=1.4, fontweight='bold',
           color=PAL['beta'],
           bbox=dict(boxstyle='round', facecolor=pal_alpha('beta', 0.15),
                    edgecolor=PAL['beta'], linewidth=2))
    
    plt.suptitle('Level 2: Two Traits = Two Heritabilities',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# FIGURE L3: COMBINATIONS MATTER
# =============================================================================

def figure_L3_combinations(save_path=None):
    """
    Level 3: When you combine traits, the heritability depends on the mix.
    
    This is the key insight - still using simple bar/line representations.
    """
    
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.25)
    
    # Trait variances
    V_G1, V_E1 = 45, 55   # Yield: h² = 0.45
    V_G2, V_E2 = 25, 75   # Drought: h² = 0.25
    
    # Genetic covariance (positive - good genes for yield also help drought)
    cov_G = 15
    cov_E = 10
    
    # =========================================================================
    # Panel A: Recap - Individual Traits
    # =========================================================================
    ax = fig.add_subplot(gs[0, 0])
    
    x = np.arange(2)
    width = 0.6
    
    ax.bar(x, [V_G1, V_G2], width, color=PAL['G'], edgecolor='black', linewidth=1.5)
    ax.bar(x, [V_E1, V_E2], width, bottom=[V_G1, V_G2], color=PAL['E'], 
          edgecolor='black', linewidth=1.5)
    
    h2_1 = V_G1 / (V_G1 + V_E1)
    h2_2 = V_G2 / (V_G2 + V_E2)
    
    ax.text(0, V_G1 + V_E1 + 5, f'h² = {h2_1:.2f}', ha='center', fontsize=11, fontweight='bold')
    ax.text(1, V_G2 + V_E2 + 5, f'h² = {h2_2:.2f}', ha='center', fontsize=11, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Yield', 'Drought Tol.'], fontsize=10)
    ax.set_ylabel('Variance', fontsize=11)
    ax.set_ylim(0, 130)
    ax.set_title('A. Individual Trait h²', fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # =========================================================================
    # Panel B: 50-50 Combination
    # =========================================================================
    ax = fig.add_subplot(gs[0, 1])
    
    # For a 50-50 mix (weights w1=w2=0.5), the combined variance is:
    # V_G(combo) = w1²V_G1 + w2²V_G2 + 2*w1*w2*cov_G
    # V_P(combo) = w1²V_P1 + w2²V_P2 + 2*w1*w2*cov_P
    
    w1, w2 = 0.5, 0.5
    V_G_combo = w1**2 * V_G1 + w2**2 * V_G2 + 2*w1*w2*cov_G
    V_E_combo = w1**2 * V_E1 + w2**2 * V_E2 + 2*w1*w2*cov_E
    V_P_combo = V_G_combo + V_E_combo
    h2_combo = V_G_combo / V_P_combo
    
    ax.bar(['50% Yield +\n50% Drought'], [V_G_combo], color=PAL['G'], 
          edgecolor='black', linewidth=1.5, width=0.5)
    ax.bar(['50% Yield +\n50% Drought'], [V_E_combo], bottom=[V_G_combo], 
          color=PAL['E'], edgecolor='black', linewidth=1.5, width=0.5)
    
    ax.text(0, V_P_combo + 2, f'h² = {h2_combo:.2f}', ha='center', 
           fontsize=12, fontweight='bold', color=PAL['warning'])
    
    ax.set_ylabel('Variance', fontsize=11)
    ax.set_ylim(0, 60)
    ax.set_title('B. Equal Mix (50-50)', fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # =========================================================================
    # Panel C: Different Mixes
    # =========================================================================
    ax = fig.add_subplot(gs[0, 2])
    
    # Try different weightings
    mixes = [
        (1.0, 0.0, 'All Yield'),
        (0.8, 0.2, '80-20'),
        (0.5, 0.5, '50-50'),
        (0.2, 0.8, '20-80'),
        (0.0, 1.0, 'All Drought'),
    ]
    
    h2_values = []
    labels = []
    for w1, w2, label in mixes:
        # Normalize weights to unit vector for fair comparison
        norm = np.sqrt(w1**2 + w2**2)
        w1n, w2n = w1/norm, w2/norm
        
        V_G_mix = w1n**2 * V_G1 + w2n**2 * V_G2 + 2*w1n*w2n*cov_G
        V_E_mix = w1n**2 * V_E1 + w2n**2 * V_E2 + 2*w1n*w2n*cov_E
        h2_mix = V_G_mix / (V_G_mix + V_E_mix)
        h2_values.append(h2_mix)
        labels.append(label)
    
    colors = [PAL['G'], PAL['warning'], PAL['warning'], PAL['warning'], PAL['danger']]
    bars = ax.bar(range(len(mixes)), h2_values, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_xticks(range(len(mixes)))
    ax.set_xticklabels(labels, fontsize=9, rotation=15)
    ax.set_ylabel('Heritability h²', fontsize=11)
    ax.set_ylim(0, 0.55)
    ax.axhline(h2_1, color=PAL['G'], linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axhline(h2_2, color=PAL['danger'], linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(4.3, h2_1, 'Yield h²', fontsize=9, va='center', color=PAL['G'])
    ax.text(4.3, h2_2, 'Drought h²', fontsize=9, va='center', color=PAL['danger'])
    
    ax.set_title('C. h² Changes with the Mix!', fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # =========================================================================
    # Panel D: The Insight (Text)
    # =========================================================================
    ax = fig.add_subplot(gs[1, 0])
    ax.axis('off')
    
    ax.text(0.5, 0.95, 'D. The Key Insight', fontsize=14, fontweight='bold',
           ha='center', transform=ax.transAxes)
    
    insight = (
        "When you select for a\n"
        "COMBINATION of traits,\n"
        "the heritability is NOT\n"
        "simply the average!\n\n"
        "It depends on:\n"
        "• How much of each trait\n"
        "• How traits covary genetically\n"
        "• How traits covary environmentally"
    )
    ax.text(0.5, 0.45, insight, fontsize=12, ha='center', va='center',
           transform=ax.transAxes, linespacing=1.5,
           bbox=dict(boxstyle='round', facecolor=pal_alpha('beta', 0.1),
                    edgecolor=PAL['beta'], linewidth=2))
    
    # =========================================================================
    # Panel E: Visual - The Mixing
    # =========================================================================
    ax = fig.add_subplot(gs[1, 1])
    ax.axis('off')
    
    ax.text(0.5, 0.95, 'E. What "Direction" Means', fontsize=14, fontweight='bold',
           ha='center', transform=ax.transAxes)
    
    # Draw a simple 2D space
    ax.annotate('', xy=(0.9, 0.35), xytext=(0.15, 0.35),
               arrowprops=dict(arrowstyle='->', lw=2, color=PAL['axis']),
               transform=ax.transAxes)
    ax.annotate('', xy=(0.15, 0.9), xytext=(0.15, 0.35),
               arrowprops=dict(arrowstyle='->', lw=2, color=PAL['axis']),
               transform=ax.transAxes)
    
    ax.text(0.9, 0.28, 'Yield', fontsize=11, ha='center', transform=ax.transAxes)
    ax.text(0.08, 0.9, 'Drought\nTol.', fontsize=11, ha='center', transform=ax.transAxes)
    
    # Show different direction arrows
    # All yield
    ax.annotate('', xy=(0.7, 0.35), xytext=(0.15, 0.35),
               arrowprops=dict(arrowstyle='->', lw=3, color=PAL['G']),
               transform=ax.transAxes)
    ax.text(0.72, 0.38, 'h²=0.45', fontsize=9, color=PAL['G'], transform=ax.transAxes)
    
    # All drought
    ax.annotate('', xy=(0.15, 0.75), xytext=(0.15, 0.35),
               arrowprops=dict(arrowstyle='->', lw=3, color=PAL['danger']),
               transform=ax.transAxes)
    ax.text(0.18, 0.75, 'h²=0.25', fontsize=9, color=PAL['danger'], transform=ax.transAxes)
    
    # Diagonal (combination)
    ax.annotate('', xy=(0.6, 0.7), xytext=(0.15, 0.35),
               arrowprops=dict(arrowstyle='->', lw=3, color=PAL['warning']),
               transform=ax.transAxes)
    ax.text(0.62, 0.68, f'h²={h2_combo:.2f}', fontsize=9, color=PAL['warning'], 
           transform=ax.transAxes)
    
    ax.text(0.5, 0.12, 'Each arrow is a "direction"\n= a trait combination',
           fontsize=11, ha='center', transform=ax.transAxes, style='italic')
    
    # =========================================================================
    # Panel F: The Punchline
    # =========================================================================
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    
    ax.text(0.5, 0.95, 'F. Why This Matters', fontsize=14, fontweight='bold',
           ha='center', transform=ax.transAxes)
    
    punchline = (
        "YOUR SELECTION TARGET\n"
        "is a direction in trait space.\n\n"
        "Different directions have\n"
        "DIFFERENT heritabilities!\n\n"
        "You might aim for a combination\n"
        "that looks good phenotypically\n"
        "but has LOW heritability.\n\n"
        "That's a CONSTRAINT TRAP."
    )
    ax.text(0.5, 0.48, punchline, fontsize=12, ha='center', va='center',
           transform=ax.transAxes, linespacing=1.4,
           bbox=dict(boxstyle='round', facecolor='lightyellow',
                    edgecolor=PAL['danger'], linewidth=2))
    
    plt.suptitle('Level 3: Trait Combinations Have Their Own Heritability',
                fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# FIGURE L4: THE ELLIPSE EMERGES
# =============================================================================

def figure_L4_ellipse_emerges(save_path=None):
    """
    Level 4: Now we see why ellipses appear.
    
    The ellipse shows ALL possible directions at once.
    """
    
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25,
                         height_ratios=[1, 1.2])
    
    # Create G and P matrices from our example
    G = np.array([[45, 15],    # Yield genetic variance, covariance
                  [15, 25]])   # Covariance, Drought genetic variance
    
    E = np.array([[55, 10],
                  [10, 75]])
    
    P = G + E
    
    # =========================================================================
    # Panel A: Many Directions, Many h² Values
    # =========================================================================
    ax = fig.add_subplot(gs[0, 0:2])
    
    # Compute h² for many directions
    angles = np.linspace(0, 180, 37)  # Every 5 degrees
    h2_values = []
    
    for angle in angles:
        rad = np.radians(angle)
        beta = np.array([np.cos(rad), np.sin(rad)])
        V_G = beta @ G @ beta
        V_P = beta @ P @ beta
        h2_values.append(V_G / V_P)
    
    # Bar chart
    colors = [PAL['G'] if h > 0.35 else (PAL['warning'] if h > 0.28 else PAL['danger']) 
             for h in h2_values]
    ax.bar(angles, h2_values, width=4.5, color=colors, edgecolor='none')
    
    ax.set_xlabel('Direction (degrees from Yield axis)', fontsize=11)
    ax.set_ylabel('Heritability h²', fontsize=11)
    ax.set_xlim(-5, 185)
    ax.set_ylim(0, 0.55)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xticklabels(['0°\n(Yield)', '45°', '90°\n(Drought)', '135°', '180°'])
    
    # Mark extremes
    max_idx = np.argmax(h2_values)
    min_idx = np.argmin(h2_values)
    ax.annotate(f'Max: {h2_values[max_idx]:.2f}', 
               xy=(angles[max_idx], h2_values[max_idx]),
               xytext=(angles[max_idx]+20, h2_values[max_idx]+0.05),
               arrowprops=dict(arrowstyle='->', color=PAL['G']),
               fontsize=10, color=PAL['G'], fontweight='bold')
    ax.annotate(f'Min: {h2_values[min_idx]:.2f}',
               xy=(angles[min_idx], h2_values[min_idx]),
               xytext=(angles[min_idx]-30, h2_values[min_idx]+0.08),
               arrowprops=dict(arrowstyle='->', color=PAL['danger']),
               fontsize=10, color=PAL['danger'], fontweight='bold')
    
    ax.set_title('A. Heritability Varies Continuously with Direction', 
                fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # =========================================================================
    # Panel B: Why Ellipses?
    # =========================================================================
    ax = fig.add_subplot(gs[0, 2])
    ax.axis('off')
    
    ax.text(0.5, 0.95, 'B. Why Ellipses?', fontsize=14, fontweight='bold',
           ha='center', transform=ax.transAxes)
    
    explanation = (
        "With 2 traits, every possible\n"
        "direction is a point on a circle.\n\n"
        "The VARIANCE in each direction\n"
        "creates an ellipse:\n\n"
        "• High variance directions\n"
        "  extend FAR from center\n\n"
        "• Low variance directions\n"
        "  stay CLOSE to center\n\n"
        "One ellipse for G (genetic)\n"
        "One ellipse for P (total)"
    )
    ax.text(0.5, 0.45, explanation, fontsize=11, ha='center', va='center',
           transform=ax.transAxes, linespacing=1.4,
           bbox=dict(boxstyle='round', facecolor='white',
                    edgecolor=PAL['axis'], linewidth=2))
    
    # =========================================================================
    # Panel C: The Two Ellipses
    # =========================================================================
    ax = fig.add_subplot(gs[1, 0])
    
    # Compute ellipse contours
    def get_ellipse(M, n=200):
        eigvals, eigvecs = eigh(M)
        radii = np.sqrt(1.0 / eigvals)  # For x'Mx = 1
        angles = np.linspace(0, 2*np.pi, n)
        circle = np.array([np.cos(angles), np.sin(angles)])
        scaled = np.diag(radii) @ circle
        rotated = eigvecs @ scaled
        return rotated[0], rotated[1]
    
    # Scale matrices for visibility
    scale = 0.01
    x_G, y_G = get_ellipse(G * scale)
    x_P, y_P = get_ellipse(P * scale)
    
    ax.fill(x_G, y_G, color=PAL['G'], alpha=0.4, label='G (genetic)')
    ax.plot(x_G, y_G, color=PAL['G'], linewidth=3)
    ax.plot(x_P, y_P, color=PAL['P'], linewidth=3, linestyle='--', label='P (total)')
    
    ax.axhline(0, color=PAL['grid'], linewidth=1, zorder=0)
    ax.axvline(0, color=PAL['grid'], linewidth=1, zorder=0)
    ax.set_xlim(-18, 18)
    ax.set_ylim(-16, 16)
    ax.set_aspect('equal')
    ax.set_xlabel('Yield direction', fontsize=11)
    ax.set_ylabel('Drought direction', fontsize=11)
    ax.legend(loc='lower left', fontsize=10)
    ax.set_title('C. G and P as Ellipses', fontsize=13, fontweight='bold')
    
    # =========================================================================
    # Panel D: h² is the Ratio
    # =========================================================================
    ax = fig.add_subplot(gs[1, 1])
    
    # Same ellipses with a direction highlighted
    ax.fill(x_G, y_G, color=PAL['G'], alpha=0.3)
    ax.plot(x_G, y_G, color=PAL['G'], linewidth=2)
    ax.plot(x_P, y_P, color=PAL['P'], linewidth=2, linestyle='--')
    
    # Show a specific direction
    angle = 25  # degrees
    rad = np.radians(angle)
    beta = np.array([np.cos(rad), np.sin(rad)])
    
    V_G_dir = beta @ G @ beta
    V_P_dir = beta @ P @ beta
    h2_dir = V_G_dir / V_P_dir
    
    # Scale for where direction intersects ellipses
    r_G = 1 / np.sqrt(V_G_dir * scale)
    r_P = 1 / np.sqrt(V_P_dir * scale)
    
    # Draw direction and mark intersections
    ax.annotate('', xy=beta * 15, xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', lw=2.5, color=PAL['beta']))
    ax.plot(beta[0] * r_G, beta[1] * r_G, 'o', color=PAL['G'], markersize=10,
           markeredgecolor='white', markeredgewidth=2, zorder=5)
    ax.plot(beta[0] * r_P, beta[1] * r_P, 's', color=PAL['P'], markersize=10,
           markeredgecolor='white', markeredgewidth=2, zorder=5)
    
    # Annotate
    ax.text(10, 12, f'In this direction:\nh² = {h2_dir:.2f}', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor=PAL['beta']))
    
    ax.axhline(0, color=PAL['grid'], linewidth=1, zorder=0)
    ax.axvline(0, color=PAL['grid'], linewidth=1, zorder=0)
    ax.set_xlim(-18, 18)
    ax.set_ylim(-16, 16)
    ax.set_aspect('equal')
    ax.set_xlabel('Yield direction', fontsize=11)
    ax.set_ylabel('Drought direction', fontsize=11)
    ax.set_title('D. h²(β) = How G Fills P', fontsize=13, fontweight='bold')
    
    # =========================================================================
    # Panel E: The Complete Picture
    # =========================================================================
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    
    ax.text(0.5, 0.95, 'E. The Complete Picture', fontsize=14, fontweight='bold',
           ha='center', transform=ax.transAxes)
    
    summary = (
        "The ELLIPSE VIEW shows\n"
        "everything at once:\n\n"
        "• G ellipse = genetic variance\n"
        "  in each direction\n\n"
        "• P ellipse = total variance\n"
        "  in each direction\n\n"
        "• h²(direction) = how much\n"
        "  G fills P in that direction\n\n"
        "• CV(h²) = how much h² varies\n"
        "  across directions"
    )
    ax.text(0.5, 0.5, summary, fontsize=11, ha='center', va='center',
           transform=ax.transAxes, linespacing=1.4,
           bbox=dict(boxstyle='round', facecolor=pal_alpha('G', 0.1),
                    edgecolor=PAL['G'], linewidth=2))
    
    ax.text(0.5, 0.05, 'This is directional heritability!',
           fontsize=12, ha='center', transform=ax.transAxes, 
           fontweight='bold', color=PAL['beta'])
    
    plt.suptitle('Level 4: The Ellipses Show All Directions at Once',
                fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# FIGURE L5: ONE-PAGE SUMMARY
# =============================================================================

def figure_L5_summary(save_path=None):
    """
    One-page summary of the entire ladder.
    """
    
    fig = plt.figure(figsize=(14, 10))
    
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.97, 'From One Trait to Many: The Ladder of Understanding',
           fontsize=18, fontweight='bold', ha='center', transform=ax.transAxes)
    
    # Levels
    levels = [
        {
            'y': 0.82,
            'title': 'Level 1: One Trait',
            'content': 'h² = V_G / V_P\nOne trait → One number',
            'color': PAL['G'],
            'example': 'Yield h² = 0.45'
        },
        {
            'y': 0.64,
            'title': 'Level 2: Two Traits',
            'content': 'Each trait has its own h²\nTwo traits → Two numbers',
            'color': PAL['P'],
            'example': 'Yield h² = 0.45\nDrought h² = 0.25'
        },
        {
            'y': 0.46,
            'title': 'Level 3: Combinations',
            'content': 'Trait mixes have their own h²\nMix depends on weights AND covariances',
            'color': PAL['warning'],
            'example': '50-50 mix h² = 0.35\n(not the average!)'
        },
        {
            'y': 0.28,
            'title': 'Level 4: All Directions',
            'content': 'Ellipses show variance in every direction\nh²(direction) = G-extent / P-extent',
            'color': PAL['beta'],
            'example': 'h² ranges from 0.25 to 0.45\nCV(h²) captures this spread'
        },
    ]
    
    for level in levels:
        # Level box
        box = FancyBboxPatch((0.05, level['y'] - 0.07), 0.55, 0.14,
                            boxstyle="round,pad=0.02",
                            facecolor=pal_alpha(list(PAL.keys())[list(PAL.values()).index(level['color'])], 0.15),
                            edgecolor=level['color'], linewidth=2,
                            transform=ax.transAxes)
        ax.add_patch(box)
        
        ax.text(0.07, level['y'] + 0.05, level['title'], fontsize=13, 
               fontweight='bold', color=level['color'], transform=ax.transAxes)
        ax.text(0.07, level['y'] - 0.02, level['content'], fontsize=10,
               transform=ax.transAxes, va='top', linespacing=1.3)
        
        # Example box
        ax.text(0.75, level['y'], level['example'], fontsize=10,
               ha='center', va='center', transform=ax.transAxes,
               fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='white', 
                        edgecolor=level['color'], linewidth=1.5))
        
        # Arrow down (except last)
        if level['y'] > 0.3:
            ax.annotate('', xy=(0.3, level['y'] - 0.1), xytext=(0.3, level['y'] - 0.07),
                       arrowprops=dict(arrowstyle='->', lw=2, color=PAL['text']),
                       transform=ax.transAxes)
    
    # Bottom message
    ax.text(0.5, 0.08, 
           'THE TAKEAWAY: Heritability depends on direction.\n'
           'CV(h²) tells you how much. h²(β) tells you if your target is in a trap.',
           fontsize=12, ha='center', transform=ax.transAxes, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightyellow',
                    edgecolor=PAL['axis'], linewidth=2))
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    output_dir = os.getcwd()
    print(f"Generating ladder figures in: {output_dir}\n")
    
    print("Figure L1: One Trait")
    figure_L1_one_trait(os.path.join(output_dir, 'ladder_L1_one_trait.png'))
    plt.close()
    
    print("\nFigure L2: Two Traits")
    figure_L2_two_traits(os.path.join(output_dir, 'ladder_L2_two_traits.png'))
    plt.close()
    
    print("\nFigure L3: Combinations")
    figure_L3_combinations(os.path.join(output_dir, 'ladder_L3_combinations.png'))
    plt.close()
    
    print("\nFigure L4: Ellipse Emerges")
    figure_L4_ellipse_emerges(os.path.join(output_dir, 'ladder_L4_ellipse.png'))
    plt.close()
    
    print("\nFigure L5: Summary")
    figure_L5_summary(os.path.join(output_dir, 'ladder_L5_summary.png'))
    plt.close()
    
    print("\n" + "=" * 60)
    print("LADDER FIGURES COMPLETE!")
    print("=" * 60)
    print("""
    L1: One trait - the h² everyone knows
    L2: Two traits - each with its own h²
    L3: Combinations - h² depends on the mix
    L4: Ellipses - showing all directions at once
    L5: Summary - the complete ladder on one page
    """)
