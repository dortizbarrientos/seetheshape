#!/usr/bin/env python3
"""
==============================================================================
CHAPTER 12: WORKED EXAMPLES - COMPLETE ANALYSES
==============================================================================

Seeing the Shape: A Geometric Introduction to Multivariate Quantitative Genetics
Code Companion - Chapter 12

Author: Daniel Ortiz-Barrientos
School of the Environment, The University of Queensland

------------------------------------------------------------------------------
THE CAPSTONE CHAPTER
------------------------------------------------------------------------------

This chapter brings together everything we have learned. We work through
complete analyses from raw data to biological interpretation, showing each
step explicitly.

The goal is not just to demonstrate techniques, but to illustrate the thought
process: when to use each tool, how to check assumptions, and how to connect
mathematical results to biological meaning.

THREE WORKED EXAMPLES:
---------------------
1. Two-trait G matrix analysis - computing directional heritabilities by hand
2. Four-trait G-P comparison with P-whitening
3. Selection analysis combining β and γ with the G matrix

Each example follows the same arc:
    Data → Check assumptions → Eigendecompose → Interpret biologically

------------------------------------------------------------------------------
THE GEOMETRIC PHILOSOPHY
------------------------------------------------------------------------------

Throughout this chapter, remember:

    • The G matrix is potential - it describes what evolution CAN do
    • The γ matrix is actuality - it describes what selection WANTS
    • Their interaction IS evolution

    "The shape of the ellipsoid and the direction of the arrow—these two
     things, together, determine what will happen."

------------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, sqrtm, inv
from typing import Tuple, Dict, List, Optional, NamedTuple
from dataclasses import dataclass
import warnings

# Set random seed for reproducibility
np.random.seed(42)

# Configure matplotlib
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.figsize': (12, 10),
    'figure.dpi': 100
})


# =============================================================================
# CORE UTILITY FUNCTIONS
# =============================================================================
"""
These functions implement the fundamental operations from earlier chapters.
They form the building blocks for all analyses.
"""


def eigendecompose(A: np.ndarray, 
                   sort_descending: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Eigendecomposition of a symmetric matrix.
    
    For symmetric A: A = V Λ V^T
    
    Parameters
    ----------
    A : np.ndarray
        Symmetric matrix to decompose
    
    sort_descending : bool
        If True (default), return eigenvalues in descending order
    
    Returns
    -------
    eigenvalues : np.ndarray
        Eigenvalues (λ₁, λ₂, ..., λₚ)
    
    eigenvectors : np.ndarray
        Orthonormal eigenvectors as columns (v₁, v₂, ..., vₚ)
    
    Notes
    -----
    Uses scipy.linalg.eigh which is optimised for symmetric matrices
    and guarantees real eigenvalues and orthonormal eigenvectors.
    """
    eigenvalues, eigenvectors = eigh(A)
    
    if sort_descending:
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
    
    return eigenvalues, eigenvectors


def compute_P_inv_sqrt(P: np.ndarray) -> np.ndarray:
    """
    Compute P^{-1/2} for whitening transformation.
    
    This is the matrix that transforms phenotypic covariance to identity:
        P^{-1/2} P P^{-1/2} = I
    
    Parameters
    ----------
    P : np.ndarray
        Phenotypic covariance matrix (must be positive definite)
    
    Returns
    -------
    np.ndarray
        The matrix P^{-1/2}
    
    Mathematical Details
    --------------------
    Given eigendecomposition P = V_P Λ_P V_P^T:
        P^{-1/2} = V_P Λ_P^{-1/2} V_P^T
    
    where Λ_P^{-1/2} has diagonal entries 1/√λᵢ
    """
    eigenvalues, eigenvectors = eigendecompose(P)
    
    # Check positive definiteness
    if np.any(eigenvalues <= 0):
        raise ValueError("P must be positive definite (all eigenvalues > 0)")
    
    # Λ^{-1/2}
    Lambda_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues))
    
    # P^{-1/2} = V Λ^{-1/2} V^T
    P_inv_sqrt = eigenvectors @ Lambda_inv_sqrt @ eigenvectors.T
    
    return P_inv_sqrt


def compute_G_star(G: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Compute the P-whitened genetic matrix G* = P^{-1/2} G P^{-1/2}.
    
    In whitened space:
        • P becomes the identity matrix (P* = I)
        • G becomes G*, whose eigenvalues are directional heritabilities
    
    Parameters
    ----------
    G : np.ndarray
        Additive genetic covariance matrix
    
    P : np.ndarray
        Phenotypic covariance matrix
    
    Returns
    -------
    np.ndarray
        G* = P^{-1/2} G P^{-1/2}
    
    Key Insight
    -----------
    The eigenvalues of G* are the maximum and minimum directional
    heritabilities. The eigenvectors are the directions that achieve
    these extremes.
    """
    P_inv_sqrt = compute_P_inv_sqrt(P)
    G_star = P_inv_sqrt @ G @ P_inv_sqrt
    
    return G_star


def directional_heritability(direction: np.ndarray,
                              G: np.ndarray,
                              P: np.ndarray) -> float:
    """
    Compute heritability in a given direction.
    
    h²(β) = β^T G β / β^T P β
    
    Parameters
    ----------
    direction : np.ndarray
        Direction vector β (need not be unit length)
    
    G : np.ndarray
        Genetic covariance matrix
    
    P : np.ndarray
        Phenotypic covariance matrix
    
    Returns
    -------
    float
        Directional heritability h²(β)
    
    Biological Interpretation
    -------------------------
    This measures what fraction of phenotypic variance in direction β
    is genetic. If h²(β) = 0.6, then 60% of variance in that direction
    is heritable.
    """
    numerator = direction @ G @ direction      # Genetic variance
    denominator = direction @ P @ direction    # Phenotypic variance
    
    return numerator / denominator


def selection_response(G: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Compute evolutionary response using the breeder's equation.
    
    Δz̄ = G β
    
    Parameters
    ----------
    G : np.ndarray
        Genetic covariance matrix
    
    beta : np.ndarray
        Selection gradient
    
    Returns
    -------
    np.ndarray
        Response vector Δz̄
    
    Biological Interpretation
    -------------------------
    β points where selection wants to go.
    G determines where genetic variation allows movement.
    The response Δz̄ = Gβ is the compromise.
    
    If G is elongated, the response is deflected toward g_max
    regardless of where β points.
    """
    return G @ beta


def evolvability(direction: np.ndarray, G: np.ndarray) -> float:
    """
    Compute evolvability in a given direction.
    
    e(β) = β^T G β / ||β||²
    
    For a unit vector, this is simply β^T G β, the genetic variance
    in that direction.
    
    Parameters
    ----------
    direction : np.ndarray
        Direction of selection
    
    G : np.ndarray
        Genetic covariance matrix
    
    Returns
    -------
    float
        Evolvability (genetic variance in direction β)
    """
    beta_unit = direction / np.linalg.norm(direction)
    return beta_unit @ G @ beta_unit


# =============================================================================
# EXAMPLE 1: TWO-TRAIT G MATRIX ANALYSIS
# =============================================================================
"""
This example demonstrates:
• Eigendecomposition by hand (2×2 case)
• Computing directional heritabilities
• Finding extreme heritabilities via G*
• Interpreting the genetic ellipse geometry
"""


def example_1_two_trait_analysis():
    """
    Complete analysis of a two-trait G matrix.
    
    From the book: A plant breeding program estimates G and P for
    flowering time and plant height.
    """
    print("=" * 70)
    print("EXAMPLE 1: TWO-TRAIT G MATRIX ANALYSIS")
    print("=" * 70)
    print("""
    A plant breeding program has estimated additive genetic and phenotypic
    covariance matrices for:
        • Flowering time (days)
        • Plant height (cm)
    
    GOALS:
    1. Find the principal axes of genetic variation (g_max, g_min)
    2. Compute heritability in several directions
    3. Identify directions of maximum and minimum heritability
    """)
    
    # ----- Define the matrices -----
    G = np.array([
        [25, 15],
        [15, 36]
    ])
    
    P = np.array([
        [50, 20],
        [20, 60]
    ])
    
    trait_names = ['Flowering time', 'Plant height']
    
    print("\n" + "-" * 70)
    print("THE DATA")
    print("-" * 70)
    print("\nGenetic covariance matrix G:")
    print(G)
    print("\nPhenotypic covariance matrix P:")
    print(P)
    
    # ----- Step 1: Eigendecompose G -----
    print("\n" + "-" * 70)
    print("STEP 1: EIGENDECOMPOSE G")
    print("-" * 70)
    print("""
    We solve the characteristic equation det(G - λI) = 0:
    
    | 25-λ   15  |
    |  15   36-λ | = (25-λ)(36-λ) - 225 = 0
    
    Expanding: λ² - 61λ + (900 - 225) = λ² - 61λ + 675 = 0
    """)
    
    # Solve characteristic equation
    a, b, c = 1, -61, 675
    discriminant = b**2 - 4*a*c
    lambda_1 = (-b + np.sqrt(discriminant)) / (2*a)
    lambda_2 = (-b - np.sqrt(discriminant)) / (2*a)
    
    print(f"Using quadratic formula:")
    print(f"  λ = (61 ± √{discriminant:.2f}) / 2")
    print(f"  λ₁ = {lambda_1:.2f}  (genetic variance along g_max)")
    print(f"  λ₂ = {lambda_2:.2f}  (genetic variance along g_min)")
    
    # Find eigenvectors
    eigenvalues_G, eigenvectors_G = eigendecompose(G)
    g_max = eigenvectors_G[:, 0]
    g_min = eigenvectors_G[:, 1]
    
    print(f"\nEigenvectors:")
    print(f"  g_max = [{g_max[0]:.3f}, {g_max[1]:.3f}]")
    print(f"  g_min = [{g_min[0]:.3f}, {g_min[1]:.3f}]")
    
    # Verify orthogonality
    print(f"\nVerification:")
    print(f"  g_max · g_min = {np.dot(g_max, g_min):.6f} (should be ≈ 0)")
    print(f"  ||g_max|| = {np.linalg.norm(g_max):.4f} (should be 1)")
    
    print("""
    INTERPRETATION:
    ---------------
    g_max = [0.573, 0.820] points toward "tall and late-flowering"
    
    Plants with high breeding values tend to be BOTH tall AND late-flowering.
    The genetic variance along this axis is λ₁ = 46.48.
    
    g_min points perpendicular: "tall but early" or "short but late"
    The genetic variance here is only λ₂ = 14.52 (about 1/3 of g_max).
    """)
    
    # ----- Step 2: Univariate heritabilities -----
    print("-" * 70)
    print("STEP 2: UNIVARIATE HERITABILITIES")
    print("-" * 70)
    
    h2_time = G[0, 0] / P[0, 0]
    h2_height = G[1, 1] / P[1, 1]
    
    print(f"""
    Standard univariate heritabilities:
    
    h²(time)   = G₁₁/P₁₁ = {G[0,0]}/{P[0,0]} = {h2_time:.2f}
    h²(height) = G₂₂/P₂₂ = {G[1,1]}/{P[1,1]} = {h2_height:.2f}
    
    Both traits have moderate heritability (~50-60%).
    But this misses the DIRECTIONAL story...
    """)
    
    # ----- Step 3: Directional heritabilities -----
    print("-" * 70)
    print("STEP 3: DIRECTIONAL HERITABILITIES")
    print("-" * 70)
    
    # Along g_max
    h2_gmax = directional_heritability(g_max, G, P)
    genetic_var_gmax = g_max @ G @ g_max
    pheno_var_gmax = g_max @ P @ g_max
    
    print(f"""
    Along g_max = [{g_max[0]:.3f}, {g_max[1]:.3f}]:
    
    Genetic variance:    β^T G β = {genetic_var_gmax:.2f}
    Phenotypic variance: β^T P β = {pheno_var_gmax:.2f}
    
    h²(g_max) = {genetic_var_gmax:.2f} / {pheno_var_gmax:.2f} = {h2_gmax:.3f}
    """)
    
    # Along g_min
    h2_gmin = directional_heritability(g_min, G, P)
    genetic_var_gmin = g_min @ G @ g_min
    pheno_var_gmin = g_min @ P @ g_min
    
    print(f"""
    Along g_min = [{g_min[0]:.3f}, {g_min[1]:.3f}]:
    
    Genetic variance:    β^T G β = {genetic_var_gmin:.2f}
    Phenotypic variance: β^T P β = {pheno_var_gmin:.2f}
    
    h²(g_min) = {genetic_var_gmin:.2f} / {pheno_var_gmin:.2f} = {h2_gmin:.3f}
    """)
    
    # Along individual trait axes
    h2_time_only = directional_heritability(np.array([1, 0]), G, P)
    h2_height_only = directional_heritability(np.array([0, 1]), G, P)
    
    print(f"""
    Along individual trait axes:
    
    h²(time only)   = {h2_time_only:.3f}
    h²(height only) = {h2_height_only:.3f}
    """)
    
    # ----- Step 4: Find extreme heritabilities via G* -----
    print("-" * 70)
    print("STEP 4: EXTREME HERITABILITIES VIA G*")
    print("-" * 70)
    print("""
    To find the TRUE maximum and minimum heritabilities across ALL directions,
    we compute G* = P^{-1/2} G P^{-1/2} and find its eigenvalues.
    
    In whitened space:
        • P becomes the identity (unit sphere)
        • G* is an ellipse whose axes lengths ARE directional heritabilities
    """)
    
    # Compute G*
    G_star = compute_G_star(G, P)
    eigenvalues_Gstar, eigenvectors_Gstar = eigendecompose(G_star)
    
    print(f"\nG* = P^{{-1/2}} G P^{{-1/2}}:")
    print(G_star.round(4))
    
    print(f"\nEigenvalues of G* (these ARE directional heritabilities):")
    print(f"  λ*₁ = {eigenvalues_Gstar[0]:.3f} = h²_max")
    print(f"  λ*₂ = {eigenvalues_Gstar[1]:.3f} = h²_min")
    
    print(f"""
    KEY INSIGHT:
    ------------
    The eigenvalues of G* give the EXTREME directional heritabilities:
    
    • Maximum h² = {eigenvalues_Gstar[0]:.3f} (62% genetic in best direction)
    • Minimum h² = {eigenvalues_Gstar[1]:.3f} (42% genetic in worst direction)
    
    The range is {eigenvalues_Gstar[0] - eigenvalues_Gstar[1]:.3f}
    That's a {(eigenvalues_Gstar[0] - eigenvalues_Gstar[1])/eigenvalues_Gstar[1]*100:.0f}% 
    increase from worst to best direction!
    """)
    
    # ----- Summary table -----
    print("-" * 70)
    print("SUMMARY TABLE")
    print("-" * 70)
    print(f"""
    ┌────────────────────────┬─────────────────┬─────────────┐
    │ Direction              │ Genetic Var     │ h²          │
    ├────────────────────────┼─────────────────┼─────────────┤
    │ Flowering time only    │ {G[0,0]:>10.1f}     │ {h2_time_only:>10.2f}  │
    │ Height only            │ {G[1,1]:>10.1f}     │ {h2_height_only:>10.2f}  │
    │ g_max                  │ {genetic_var_gmax:>10.1f}     │ {h2_gmax:>10.2f}  │
    │ g_min                  │ {genetic_var_gmin:>10.1f}     │ {h2_gmin:>10.2f}  │
    │ Max h² direction       │       —         │ {eigenvalues_Gstar[0]:>10.2f}  │
    │ Min h² direction       │       —         │ {eigenvalues_Gstar[1]:>10.2f}  │
    └────────────────────────┴─────────────────┴─────────────┘
    
    NOTE: The direction of maximum GENETIC variance (g_max) is CLOSE to
    but not identical to the direction of maximum HERITABILITY.
    
    • g_max maximises β^T G β
    • max h² direction maximises β^T G β / β^T P β
    
    These coincide only when G and P have the same eigenvectors.
    """)
    
    # ----- Visualisation -----
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel A: G and P ellipses
    ax = axes[0, 0]
    theta = np.linspace(0, 2*np.pi, 100)
    
    # Unit circle
    circle = np.array([np.cos(theta), np.sin(theta)])
    
    # Transform to G ellipse
    eigenvalues_G, eigenvectors_G = eigendecompose(G)
    G_sqrt = eigenvectors_G @ np.diag(np.sqrt(eigenvalues_G)) @ eigenvectors_G.T
    G_ellipse = G_sqrt @ circle
    
    # Transform to P ellipse  
    eigenvalues_P, eigenvectors_P = eigendecompose(P)
    P_sqrt = eigenvectors_P @ np.diag(np.sqrt(eigenvalues_P)) @ eigenvectors_P.T
    P_ellipse = P_sqrt @ circle
    
    ax.plot(P_ellipse[0], P_ellipse[1], 'b-', linewidth=2, label='P (phenotypic)')
    ax.plot(G_ellipse[0], G_ellipse[1], 'r-', linewidth=2, label='G (genetic)')
    
    # Plot g_max
    scale = 8
    ax.arrow(0, 0, g_max[0]*scale, g_max[1]*scale, head_width=0.5, 
             head_length=0.3, fc='red', ec='red')
    ax.text(g_max[0]*scale*1.1, g_max[1]*scale*1.1, '$g_{max}$', fontsize=12, color='red')
    
    ax.set_xlabel(trait_names[0])
    ax.set_ylabel(trait_names[1])
    ax.set_title('(A) G and P Ellipses in Original Space')
    ax.legend()
    ax.set_aspect('equal')
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    
    # Panel B: Whitened space (P-sphere)
    ax = axes[0, 1]
    
    # P-sphere is unit circle
    ax.plot(np.cos(theta), np.sin(theta), 'b-', linewidth=2, label='P-sphere (unit circle)')
    
    # G* ellipse
    eigenvalues_Gstar, eigenvectors_Gstar = eigendecompose(G_star)
    Gstar_sqrt = eigenvectors_Gstar @ np.diag(np.sqrt(eigenvalues_Gstar)) @ eigenvectors_Gstar.T
    Gstar_ellipse = Gstar_sqrt @ circle
    
    ax.plot(Gstar_ellipse[0], Gstar_ellipse[1], 'r-', linewidth=2, label='G* ellipse')
    ax.fill(Gstar_ellipse[0], Gstar_ellipse[1], color='red', alpha=0.2)
    
    # Mark heritabilities
    ax.annotate(f'$h^2_{{max}} = {eigenvalues_Gstar[0]:.2f}$', 
                xy=(eigenvectors_Gstar[0, 0]*np.sqrt(eigenvalues_Gstar[0]),
                    eigenvectors_Gstar[1, 0]*np.sqrt(eigenvalues_Gstar[0])),
                xytext=(0.6, 0.8), fontsize=10, 
                arrowprops=dict(arrowstyle='->', color='red'))
    
    ax.set_xlabel('Whitened dimension 1')
    ax.set_ylabel('Whitened dimension 2')
    ax.set_title('(B) Whitened Space: P-sphere and G*')
    ax.legend()
    ax.set_aspect('equal')
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    
    # Panel C: Heritability as function of direction
    ax = axes[1, 0]
    
    angles = np.linspace(0, 2*np.pi, 361)
    h2_values = []
    
    for angle in angles:
        direction = np.array([np.cos(angle), np.sin(angle)])
        h2 = directional_heritability(direction, G, P)
        h2_values.append(h2)
    
    ax.plot(np.degrees(angles), h2_values, 'k-', linewidth=2)
    ax.axhline(y=eigenvalues_Gstar[0], color='green', linestyle='--', 
               label=f'Max h² = {eigenvalues_Gstar[0]:.3f}')
    ax.axhline(y=eigenvalues_Gstar[1], color='red', linestyle='--',
               label=f'Min h² = {eigenvalues_Gstar[1]:.3f}')
    ax.axhline(y=np.mean(eigenvalues_Gstar), color='blue', linestyle=':',
               label=f'Mean h² = {np.mean(eigenvalues_Gstar):.3f}')
    
    ax.set_xlabel('Direction (degrees from trait 1 axis)')
    ax.set_ylabel('Directional heritability h²(β)')
    ax.set_title('(C) Heritability Varies with Direction')
    ax.legend()
    ax.set_xlim(0, 360)
    ax.set_ylim(0.3, 0.7)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.grid(True, alpha=0.3)
    
    # Panel D: Response to selection
    ax = axes[1, 1]
    
    # Different selection directions
    selection_directions = [
        (np.array([1, 0]), 'Select on time only'),
        (np.array([0, 1]), 'Select on height only'),
        (np.array([1, 1])/np.sqrt(2), 'Select on both equally'),
    ]
    
    colors = ['#e41a1c', '#377eb8', '#4daf4a']
    
    for i, (beta, label) in enumerate(selection_directions):
        response = selection_response(G, beta)
        
        # Plot selection direction
        ax.arrow(0, 0, beta[0]*3, beta[1]*3, head_width=0.2, head_length=0.15,
                fc='none', ec=colors[i], linestyle='--', linewidth=1.5)
        
        # Plot response direction
        ax.arrow(0, 0, response[0], response[1], head_width=0.2, head_length=0.15,
                fc=colors[i], ec=colors[i], linewidth=2, label=label)
    
    # Plot g_max
    ax.arrow(0, 0, g_max[0]*6, g_max[1]*6, head_width=0.2, head_length=0.15,
            fc='none', ec='black', linestyle=':', linewidth=2, label='$g_{max}$')
    
    ax.set_xlabel(trait_names[0])
    ax.set_ylabel(trait_names[1])
    ax.set_title('(D) Selection (dashed) vs Response (solid)')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_aspect('equal')
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.set_xlim(-1, 6)
    ax.set_ylim(-1, 6)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/ch12_example1_two_trait.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nFigure saved to: /home/claude/ch12_example1_two_trait.png")
    
    print("\n" + "-" * 70)
    print("BIOLOGICAL CONCLUSIONS")
    print("-" * 70)
    print(f"""
    1. GENETIC CONSTRAINT:
       The G matrix is elongated along the "tall + late" diagonal.
       Genetic variation is 3× higher along g_max than g_min.
       Evolution is EASIER along g_max, HARDER perpendicular to it.
    
    2. HERITABILITY VARIATION:
       Heritability ranges from {eigenvalues_Gstar[1]:.0%} to {eigenvalues_Gstar[0]:.0%}
       depending on direction. A breeder selecting along g_max gets
       {eigenvalues_Gstar[0]/eigenvalues_Gstar[1]*100:.0f}% more "bang for buck" 
       than one selecting perpendicular.
    
    3. DEFLECTION OF RESPONSE:
       Selection in ANY direction produces a response deflected toward g_max.
       Even if you select only on height, flowering time will increase too
       (due to positive genetic correlation).
    
    4. IMPLICATION FOR BREEDING:
       To break the correlation (get tall early-flowering plants),
       you need to select AGAINST the correlation—in the g_min direction.
       This is genetically difficult (low variance there).
    """)
    
    return {
        'G': G, 'P': P, 'G_star': G_star,
        'eigenvalues_G': eigenvalues_G, 'eigenvectors_G': eigenvectors_G,
        'eigenvalues_Gstar': eigenvalues_Gstar, 'eigenvectors_Gstar': eigenvectors_Gstar
    }


# =============================================================================
# EXAMPLE 2: FOUR-TRAIT G-P COMPARISON WITH WHITENING
# =============================================================================
"""
This example demonstrates:
• Full P-whitening procedure
• Distribution of directional heritability
• Identifying constraint traps
• The coefficient of variation of h²
"""


def example_2_four_trait_gp():
    """
    Four-trait analysis comparing G and P with P-whitening.
    
    From the book: Morphological traits in a passerine bird population.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: FOUR-TRAIT G-P COMPARISON")
    print("=" * 70)
    print("""
    A study of passerine birds estimates G and P for four traits:
        • Wing length
        • Tarsus length  
        • Bill depth
        • Bill width
    
    GOALS:
    1. Check positive definiteness and consistency
    2. Compute G* and its eigenvalues (directional heritabilities)
    3. Identify constraint traps
    4. Quantify heritability variation across directions
    """)
    
    # ----- Define the matrices -----
    G = np.array([
        [0.80, 0.45, 0.20, 0.15],
        [0.45, 0.60, 0.25, 0.18],
        [0.20, 0.25, 0.35, 0.28],
        [0.15, 0.18, 0.28, 0.30]
    ])
    
    P = np.array([
        [1.20, 0.55, 0.30, 0.22],
        [0.55, 0.95, 0.35, 0.25],
        [0.30, 0.35, 0.55, 0.40],
        [0.22, 0.25, 0.40, 0.50]
    ])
    
    trait_names = ['Wing', 'Tarsus', 'Bill depth', 'Bill width']
    
    print("\n" + "-" * 70)
    print("STEP 1: BASIC CHECKS")
    print("-" * 70)
    
    # Check positive definiteness
    eigenvalues_G, _ = eigendecompose(G)
    eigenvalues_P, _ = eigendecompose(P)
    
    print("\nEigenvalues of G:", eigenvalues_G.round(3))
    print("Eigenvalues of P:", eigenvalues_P.round(3))
    
    print(f"\nPositive definite checks:")
    print(f"  All G eigenvalues > 0: {np.all(eigenvalues_G > 0)}")
    print(f"  All P eigenvalues > 0: {np.all(eigenvalues_P > 0)}")
    
    # Check G_ii ≤ P_ii
    print("\nVariance consistency check (G_ii ≤ P_ii):")
    for i, name in enumerate(trait_names):
        status = "✓" if G[i,i] <= P[i,i] else "✗"
        print(f"  {name}: G={G[i,i]:.2f} ≤ P={P[i,i]:.2f} {status}")
    
    # ----- Step 2: Univariate heritabilities -----
    print("\n" + "-" * 70)
    print("STEP 2: UNIVARIATE HERITABILITIES")
    print("-" * 70)
    
    print(f"\n{'Trait':<15} {'G_ii':>8} {'P_ii':>8} {'h²':>8}")
    print("-" * 45)
    for i, name in enumerate(trait_names):
        h2 = G[i,i] / P[i,i]
        print(f"{name:<15} {G[i,i]:>8.2f} {P[i,i]:>8.2f} {h2:>8.2f}")
    
    print("""
    All traits have similar univariate heritabilities (60-67%).
    But this masks important DIRECTIONAL variation...
    """)
    
    # ----- Step 3: Compute G* -----
    print("-" * 70)
    print("STEP 3: COMPUTE G* AND EIGENSTRUCTURE")
    print("-" * 70)
    
    G_star = compute_G_star(G, P)
    eigenvalues_Gstar, eigenvectors_Gstar = eigendecompose(G_star)
    
    print("\nG* = P^{-1/2} G P^{-1/2}:")
    print(G_star.round(4))
    
    print("\nEigenvalues of G* (= directional heritabilities):")
    for i, (eig, vec) in enumerate(zip(eigenvalues_Gstar, eigenvectors_Gstar.T)):
        print(f"  λ*_{i+1} = {eig:.3f}")
    
    # ----- Step 4: Interpret heritability distribution -----
    print("\n" + "-" * 70)
    print("STEP 4: HERITABILITY DISTRIBUTION")
    print("-" * 70)
    
    h2_max = eigenvalues_Gstar[0]
    h2_min = eigenvalues_Gstar[-1]
    h2_mean = np.mean(eigenvalues_Gstar)
    h2_range = h2_max - h2_min
    
    # CV formula from Chapter 13
    p = len(eigenvalues_Gstar)
    var_lambda = np.var(eigenvalues_Gstar, ddof=0)  # Population variance
    cv_lambda = np.sqrt(var_lambda) / h2_mean
    cv_h2 = np.sqrt(2 / (p + 2)) * cv_lambda
    
    # Constraint severity
    severity = 1 - h2_min / h2_mean
    
    print(f"""
    HERITABILITY STATISTICS:
    
    Maximum h²:  {h2_max:.3f}  ({h2_max*100:.0f}% genetic in best direction)
    Minimum h²:  {h2_min:.3f}  ({h2_min*100:.0f}% genetic in worst direction)
    Mean h²:     {h2_mean:.3f}
    Range:       {h2_range:.3f}
    
    Coefficient of variation of h²: CV(h²) = {cv_h2:.3f} ({cv_h2*100:.0f}%)
    Constraint severity: 1 - h²_min/h²_mean = {severity:.3f}
    
    INTERPRETATION:
    ---------------
    • Mean heritability is moderate (54%)
    • But it ranges from 35% to 71% depending on direction!
    • CV of 16% indicates substantial constraint heterogeneity
    • Severity of 0.34 means worst direction is 34% below average
    """)
    
    # ----- Step 5: Identify constraint traps -----
    print("-" * 70)
    print("STEP 5: IDENTIFY CONSTRAINT TRAPS")
    print("-" * 70)
    
    # Direction of minimum heritability
    v_min_h2 = eigenvectors_Gstar[:, -1]  # Last eigenvector
    v_max_h2 = eigenvectors_Gstar[:, 0]   # First eigenvector
    
    print("\nDirection of MAXIMUM heritability (h² = {:.3f}):".format(h2_max))
    print("  Loadings: ", end="")
    for name, loading in zip(trait_names, v_max_h2):
        print(f"{name}={loading:+.2f}  ", end="")
    
    print("\n\nDirection of MINIMUM heritability (h² = {:.3f}):".format(h2_min))
    print("  Loadings: ", end="")
    for name, loading in zip(trait_names, v_min_h2):
        print(f"{name}={loading:+.2f}  ", end="")
    
    print(f"""
    
    CONSTRAINT TRAP ANALYSIS:
    -------------------------
    The HIGH-h² direction loads heavily on wing ({v_max_h2[0]:+.2f}) and 
    tarsus ({v_max_h2[1]:+.2f}) — overall body SIZE.
    Selection for larger/smaller birds has h² = {h2_max:.0%}.
    
    The LOW-h² direction contrasts bill traits:
    bill depth ({v_min_h2[2]:+.2f}) vs bill width ({v_min_h2[3]:+.2f})
    Selection for bill SHAPE (deep vs wide) has only h² = {h2_min:.0%}.
    
    BIOLOGICAL MEANING:
    Body size variation is 71% genetic — easy to breed for.
    Bill shape variation is only 35% genetic — harder to change.
    Bill shape is a CONSTRAINT TRAP.
    """)
    
    # ----- Visualisation -----
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel A: Eigenvalue comparison
    ax = axes[0, 0]
    
    x = np.arange(p)
    width = 0.35
    
    ax.bar(x - width/2, eigenvalues_G, width, label='G eigenvalues', color='red', alpha=0.7)
    ax.bar(x + width/2, eigenvalues_P, width, label='P eigenvalues', color='blue', alpha=0.7)
    
    ax.set_xlabel('Eigenvalue rank')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('(A) G vs P Eigenvalues')
    ax.set_xticks(x)
    ax.set_xticklabels(['λ₁', 'λ₂', 'λ₃', 'λ₄'])
    ax.legend()
    
    # Panel B: G* eigenvalues (directional heritabilities)
    ax = axes[0, 1]
    
    colors = ['forestgreen' if eig > h2_mean else 'crimson' for eig in eigenvalues_Gstar]
    bars = ax.bar(x, eigenvalues_Gstar, color=colors, alpha=0.7, edgecolor='black')
    
    ax.axhline(y=h2_mean, color='blue', linestyle='--', linewidth=2, 
               label=f'Mean h² = {h2_mean:.2f}')
    ax.axhline(y=h2_max, color='green', linestyle=':', alpha=0.7)
    ax.axhline(y=h2_min, color='red', linestyle=':', alpha=0.7)
    
    ax.set_xlabel('G* eigenvalue rank')
    ax.set_ylabel('Directional heritability h²')
    ax.set_title('(B) Eigenvalues of G* = Directional Heritabilities')
    ax.set_xticks(x)
    ax.set_xticklabels(['λ*₁', 'λ*₂', 'λ*₃', 'λ*₄'])
    ax.legend()
    ax.set_ylim(0, 0.85)
    
    # Add text labels on bars
    for i, (bar, eig) in enumerate(zip(bars, eigenvalues_Gstar)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{eig:.2f}', ha='center', fontsize=10)
    
    # Panel C: Eigenvector loadings
    ax = axes[1, 0]
    
    im = ax.imshow(eigenvectors_Gstar.T, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    ax.set_xticks(range(p))
    ax.set_xticklabels(trait_names, rotation=45, ha='right')
    ax.set_yticks(range(p))
    ax.set_yticklabels([f'h² = {eig:.2f}' for eig in eigenvalues_Gstar])
    ax.set_xlabel('Trait')
    ax.set_ylabel('Heritability axis')
    ax.set_title('(C) Eigenvector Loadings (rows = h² directions)')
    
    plt.colorbar(im, ax=ax, shrink=0.8, label='Loading')
    
    # Add text values
    for i in range(p):
        for j in range(p):
            val = eigenvectors_Gstar[j, i]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                   color=color, fontsize=9)
    
    # Panel D: Covariance matrices heatmaps
    ax = axes[1, 1]
    
    # Convert to correlations for visualization
    def cov_to_corr(cov):
        D = np.sqrt(np.diag(cov))
        return cov / np.outer(D, D)
    
    corr_G = cov_to_corr(G)
    corr_P = cov_to_corr(P)
    
    # Plot G correlations in lower triangle, P in upper
    combined = np.triu(corr_P, k=1) + np.tril(corr_G)
    
    im = ax.imshow(combined, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(p))
    ax.set_xticklabels(trait_names, rotation=45, ha='right')
    ax.set_yticks(range(p))
    ax.set_yticklabels(trait_names)
    ax.set_title('(D) Correlations: G (lower) vs P (upper)')
    
    plt.colorbar(im, ax=ax, shrink=0.8, label='Correlation')
    
    # Add text and diagonal line
    for i in range(p):
        for j in range(p):
            val = combined[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                   color=color, fontsize=9)
    
    plt.tight_layout()
    plt.savefig('/home/claude/ch12_example2_four_trait.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nFigure saved to: /home/claude/ch12_example2_four_trait.png")
    
    print("\n" + "-" * 70)
    print("BIOLOGICAL CONCLUSIONS")
    print("-" * 70)
    print(f"""
    1. UNIVARIATE VS DIRECTIONAL VIEW:
       All four traits look similar in univariate analysis (~63% h²).
       But directional h² ranges from {h2_min:.0%} to {h2_max:.0%}!
       The univariate view is MISLEADING.
    
    2. SIZE VS SHAPE:
       Body size (wing + tarsus) has HIGH heritability ({h2_max:.0%}).
       Bill shape (depth vs width contrast) has LOW heritability ({h2_min:.0%}).
       
       Evolutionary response to selection depends strongly on whether
       selection targets size or shape.
    
    3. CONSTRAINT HETEROGENEITY:
       CV(h²) = {cv_h2:.0%} indicates substantial variation.
       Some selection directions will respond 2× better than others.
    
    4. BREEDING IMPLICATIONS:
       A breeding program for larger birds: EASY (h² = {h2_max:.0%})
       A breeding program for bill proportions: HARD (h² = {h2_min:.0%})
       
       The bill shape is a "constraint trap": plenty of phenotypic
       variation exists, but most is environmental, not genetic.
    """)
    
    return {
        'G': G, 'P': P, 'G_star': G_star,
        'eigenvalues_Gstar': eigenvalues_Gstar,
        'eigenvectors_Gstar': eigenvectors_Gstar,
        'cv_h2': cv_h2,
        'h2_max': h2_max, 'h2_min': h2_min
    }


# =============================================================================
# EXAMPLE 3: SELECTION ANALYSIS WITH G AND γ
# =============================================================================
"""
This example demonstrates:
• Interpreting the selection gradient β
• Interpreting the curvature matrix γ
• Predicting evolutionary response
• Analysing G-γ alignment
"""


def example_3_selection_analysis():
    """
    Complete selection analysis combining β, γ, and G.
    
    From the book: Survival selection on two morphological traits in birds.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: SELECTION ANALYSIS WITH G AND γ")
    print("=" * 70)
    print("""
    A study measures survival in relation to two standardised traits.
    We have estimates of:
        • β (selection gradient) - direction of steepest fitness increase
        • γ (curvature matrix) - shape of the fitness surface
        • G (genetic covariance) - available genetic variation
    
    GOALS:
    1. Interpret directional selection (β)
    2. Interpret stabilising/disruptive selection (γ)
    3. Predict evolutionary response (Δz̄ = Gβ)
    4. Analyse G-γ alignment and its implications
    """)
    
    # ----- Define the data -----
    # Selection gradient
    beta = np.array([0.18, 0.12])
    
    # Curvature matrix (quadratic selection gradient)
    gamma = np.array([
        [-0.15, 0.08],
        [0.08, -0.10]
    ])
    
    # Genetic covariance matrix
    G = np.array([
        [0.45, 0.30],
        [0.30, 0.35]
    ])
    
    trait_names = ['Trait 1', 'Trait 2']
    
    print("\n" + "-" * 70)
    print("THE DATA")
    print("-" * 70)
    print("\nSelection gradient β:")
    print(f"  β = [{beta[0]:.2f}, {beta[1]:.2f}]")
    
    print("\nQuadratic selection gradient γ:")
    print(gamma)
    
    print("\nGenetic covariance matrix G:")
    print(G)
    
    # ----- Step 1: Interpret β -----
    print("\n" + "-" * 70)
    print("STEP 1: INTERPRET DIRECTIONAL SELECTION (β)")
    print("-" * 70)
    
    beta_magnitude = np.linalg.norm(beta)
    beta_direction = beta / beta_magnitude
    
    print(f"""
    β = [{beta[0]:.2f}, {beta[1]:.2f}]
    
    Magnitude: ||β|| = {beta_magnitude:.3f}
    Direction: β/||β|| = [{beta_direction[0]:.3f}, {beta_direction[1]:.3f}]
    
    INTERPRETATION:
    ---------------
    • Both β elements are POSITIVE → selection favours INCREASES in both traits
    • β₁ = {beta[0]:.2f} > β₂ = {beta[1]:.2f} → trait 1 under stronger selection
    • The selection direction is ~{np.degrees(np.arctan2(beta[1], beta[0])):.0f}° 
      from the trait 1 axis
    """)
    
    # ----- Step 2: Interpret γ -----
    print("-" * 70)
    print("STEP 2: INTERPRET CURVATURE (γ)")
    print("-" * 70)
    
    eigenvalues_gamma, eigenvectors_gamma = eigendecompose(gamma, sort_descending=False)
    # Note: sort ascending for gamma (most negative = strongest stabilising)
    eigenvalues_gamma = eigenvalues_gamma[::-1]
    eigenvectors_gamma = eigenvectors_gamma[:, ::-1]
    
    print(f"""
    γ matrix:
    [{gamma[0,0]:+.2f}  {gamma[0,1]:+.2f}]
    [{gamma[1,0]:+.2f}  {gamma[1,1]:+.2f}]
    
    Diagonal elements (univariate curvature):
    • γ₁₁ = {gamma[0,0]:+.2f} → {"STABILISING" if gamma[0,0] < 0 else "DISRUPTIVE"} selection on trait 1
    • γ₂₂ = {gamma[1,1]:+.2f} → {"STABILISING" if gamma[1,1] < 0 else "DISRUPTIVE"} selection on trait 2
    
    Off-diagonal element (correlational selection):
    • γ₁₂ = {gamma[0,1]:+.2f} → {"POSITIVE" if gamma[0,1] > 0 else "NEGATIVE"} trait combinations favoured
    
    Eigendecomposition of γ:
    • λ₁ = {eigenvalues_gamma[0]:+.3f} (weak stabilising)
    • λ₂ = {eigenvalues_gamma[1]:+.3f} (strong stabilising)
    """)
    
    v_weak = eigenvectors_gamma[:, 0]
    v_strong = eigenvectors_gamma[:, 1]
    
    print(f"""
    INTERPRETATION:
    ---------------
    Both eigenvalues are NEGATIVE → overall STABILISING selection.
    
    Weak stabilising axis: [{v_weak[0]:+.3f}, {v_weak[1]:+.3f}]
        |λ| = {abs(eigenvalues_gamma[0]):.3f} — selection pressure is WEAK
        This is roughly the "both traits high" or "both low" direction.
    
    Strong stabilising axis: [{v_strong[0]:+.3f}, {v_strong[1]:+.3f}]
        |λ| = {abs(eigenvalues_gamma[1]):.3f} — selection pressure is STRONG
        This is the "one high, one low" direction (trait contrast).
    
    The fitness surface is like a RIDGE:
    • Wide along the "both high/low" direction (can vary freely)
    • Narrow along the "contrast" direction (strongly penalised)
    """)
    
    # ----- Step 3: Predict response -----
    print("-" * 70)
    print("STEP 3: PREDICT EVOLUTIONARY RESPONSE")
    print("-" * 70)
    
    response = selection_response(G, beta)
    response_magnitude = np.linalg.norm(response)
    response_direction = response / response_magnitude
    
    print(f"""
    Lande equation: Δz̄ = Gβ
    
    G = [{G[0,0]:.2f}  {G[0,1]:.2f}]      β = [{beta[0]:.2f}]
        [{G[1,0]:.2f}  {G[1,1]:.2f}]          [{beta[1]:.2f}]
    
    Δz̄ = G β = [{response[0]:.3f}]
               [{response[1]:.3f}]
    
    Selection direction:   [{beta_direction[0]:.3f}, {beta_direction[1]:.3f}]
    Response direction:    [{response_direction[0]:.3f}, {response_direction[1]:.3f}]
    """)
    
    # Angle between selection and response
    cos_angle = np.dot(beta_direction, response_direction)
    angle_degrees = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
    
    print(f"""
    DEFLECTION ANALYSIS:
    --------------------
    Angle between selection and response: {angle_degrees:.1f}°
    
    The response is {"ALIGNED" if angle_degrees < 15 else "DEFLECTED"} 
    {"" if angle_degrees < 15 else f"by {angle_degrees:.0f}° toward g_max"}
    
    Reason: The genetic correlation (G₁₂ = {G[0,1]:.2f}) causes trait 1 and 
    trait 2 to respond together. Selection on trait 2 "drags" trait 1 along.
    """)
    
    # ----- Step 4: G-γ alignment -----
    print("-" * 70)
    print("STEP 4: G-γ ALIGNMENT ANALYSIS")
    print("-" * 70)
    
    eigenvalues_G, eigenvectors_G = eigendecompose(G)
    g_max = eigenvectors_G[:, 0]
    g_min = eigenvectors_G[:, 1]
    
    # Compute alignment between g_max and γ eigenvectors
    alignment_gmax_weak = abs(np.dot(g_max, v_weak))
    alignment_gmax_strong = abs(np.dot(g_max, v_strong))
    
    print(f"""
    G matrix eigenstructure:
    • g_max = [{g_max[0]:.3f}, {g_max[1]:.3f}], λ = {eigenvalues_G[0]:.3f}
    • g_min = [{g_min[0]:.3f}, {g_min[1]:.3f}], λ = {eigenvalues_G[1]:.3f}
    
    γ matrix eigenstructure:
    • v_weak = [{v_weak[0]:.3f}, {v_weak[1]:.3f}], λ = {eigenvalues_gamma[0]:+.3f}
    • v_strong = [{v_strong[0]:.3f}, {v_strong[1]:.3f}], λ = {eigenvalues_gamma[1]:+.3f}
    
    ALIGNMENT (absolute dot products):
    • |g_max · v_weak|   = {alignment_gmax_weak:.3f}
    • |g_max · v_strong| = {alignment_gmax_strong:.3f}
    """)
    
    print(f"""
    INTERPRETATION:
    ---------------
    g_max (direction of maximum genetic variance) aligns with
    v_weak (direction of WEAK stabilising selection).
    
    This is FAVOURABLE:
    • The population CAN vary along g_max (high genetic variance: {eigenvalues_G[0]:.2f})
    • Selection ALLOWS variation there (weak curvature: |λ| = {abs(eigenvalues_gamma[0]):.3f})
    • Evolution proceeds along the "fitness ridge"
    
    Conversely:
    • g_min (low genetic variance: {eigenvalues_G[1]:.2f}) aligns with
    • v_strong (strong stabilising: |λ| = {abs(eigenvalues_gamma[1]):.3f})
    • The population CANNOT vary where selection is STRONGEST
    
    This alignment is likely NOT coincidental:
    Mutation-selection balance erodes variance in directions of strong
    stabilising selection, while preserving variance along fitness ridges.
    """)
    
    # ----- Visualisation -----
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel A: Selection vectors
    ax = axes[0, 0]
    
    # Plot G ellipse
    theta = np.linspace(0, 2*np.pi, 100)
    circle = np.array([np.cos(theta), np.sin(theta)])
    G_sqrt = eigenvectors_G @ np.diag(np.sqrt(eigenvalues_G)) @ eigenvectors_G.T
    G_ellipse = G_sqrt @ circle
    
    ax.plot(G_ellipse[0], G_ellipse[1], 'r-', linewidth=2, alpha=0.5, label='G ellipse')
    ax.fill(G_ellipse[0], G_ellipse[1], color='red', alpha=0.1)
    
    # Plot β (selection)
    ax.arrow(0, 0, beta[0]*2, beta[1]*2, head_width=0.05, head_length=0.03,
            fc='blue', ec='blue', linewidth=2)
    ax.text(beta[0]*2.2, beta[1]*2.2, 'β', fontsize=14, color='blue', fontweight='bold')
    
    # Plot response
    ax.arrow(0, 0, response[0], response[1], head_width=0.05, head_length=0.03,
            fc='green', ec='green', linewidth=2)
    ax.text(response[0]*1.1, response[1]*1.1, 'Δz̄', fontsize=14, color='green', fontweight='bold')
    
    # Plot g_max
    ax.arrow(0, 0, g_max[0]*0.8, g_max[1]*0.8, head_width=0.05, head_length=0.03,
            fc='red', ec='red', linewidth=2, linestyle='--')
    ax.text(g_max[0]*0.9, g_max[1]*0.9, '$g_{max}$', fontsize=12, color='red')
    
    ax.set_xlabel('Trait 1')
    ax.set_ylabel('Trait 2')
    ax.set_title('(A) Selection (β) vs Response (Δz̄)')
    ax.legend(loc='upper left')
    ax.set_aspect('equal')
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    
    # Panel B: Fitness surface
    ax = axes[0, 1]
    
    # Create grid
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    
    # Compute fitness (quadratic approximation)
    W = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            z = np.array([X[i,j], Y[i,j]])
            W[i,j] = beta @ z + 0.5 * z @ gamma @ z
    
    contour = ax.contour(X, Y, W, levels=15, cmap='RdYlGn')
    ax.clabel(contour, inline=True, fontsize=8)
    
    # Plot γ eigenvectors
    ax.arrow(0, 0, v_weak[0]*1.5, v_weak[1]*1.5, head_width=0.1, head_length=0.05,
            fc='orange', ec='orange', linewidth=2)
    ax.text(v_weak[0]*1.7, v_weak[1]*1.7, 'weak', fontsize=10, color='orange')
    
    ax.arrow(0, 0, v_strong[0]*1.5, v_strong[1]*1.5, head_width=0.1, head_length=0.05,
            fc='purple', ec='purple', linewidth=2)
    ax.text(v_strong[0]*1.7, v_strong[1]*1.7, 'strong', fontsize=10, color='purple')
    
    ax.set_xlabel('Trait 1')
    ax.set_ylabel('Trait 2')
    ax.set_title('(B) Fitness Surface (contours) and γ Eigenvectors')
    ax.set_aspect('equal')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    
    # Panel C: G and γ eigenvector comparison
    ax = axes[1, 0]
    
    vectors = [
        (g_max, eigenvalues_G[0], 'g_max', 'red', '-'),
        (g_min, eigenvalues_G[1], 'g_min', 'red', '--'),
        (v_weak, eigenvalues_gamma[0], 'γ_weak', 'blue', '-'),
        (v_strong, eigenvalues_gamma[1], 'γ_strong', 'blue', '--'),
    ]
    
    for vec, eig, name, color, ls in vectors:
        ax.arrow(0, 0, vec[0], vec[1], head_width=0.05, head_length=0.03,
                fc=color, ec=color, linewidth=2, linestyle=ls, label=f'{name} (λ={eig:+.2f})')
    
    ax.set_xlabel('Trait 1')
    ax.set_ylabel('Trait 2')
    ax.set_title('(C) G eigenvectors (red) vs γ eigenvectors (blue)')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_aspect('equal')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    
    # Panel D: Summary bar chart
    ax = axes[1, 1]
    
    categories = ['G eigenvalues\n(genetic variance)', 'γ eigenvalues\n(curvature)']
    x = np.array([0, 1.5])
    width = 0.35
    
    # G eigenvalues
    ax.bar(x[0] - width/2, eigenvalues_G[0], width, color='red', alpha=0.7, label='λ₁ (max)')
    ax.bar(x[0] + width/2, eigenvalues_G[1], width, color='red', alpha=0.4, label='λ₂ (min)')
    
    # γ eigenvalues (show absolute values)
    ax.bar(x[1] - width/2, abs(eigenvalues_gamma[0]), width, color='blue', alpha=0.7)
    ax.bar(x[1] + width/2, abs(eigenvalues_gamma[1]), width, color='blue', alpha=0.4)
    
    # Add text labels
    ax.text(x[0] - width/2, eigenvalues_G[0] + 0.02, f'{eigenvalues_G[0]:.2f}', ha='center', fontsize=10)
    ax.text(x[0] + width/2, eigenvalues_G[1] + 0.02, f'{eigenvalues_G[1]:.2f}', ha='center', fontsize=10)
    ax.text(x[1] - width/2, abs(eigenvalues_gamma[0]) + 0.02, f'{eigenvalues_gamma[0]:+.2f}', ha='center', fontsize=10)
    ax.text(x[1] + width/2, abs(eigenvalues_gamma[1]) + 0.02, f'{eigenvalues_gamma[1]:+.2f}', ha='center', fontsize=10)
    
    ax.set_ylabel('Eigenvalue magnitude')
    ax.set_title('(D) Eigenvalue Summary')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.axhline(y=0, color='gray', linewidth=0.5)
    
    # Add alignment annotation
    ax.annotate(f'g_max aligns with weak γ axis\n(correlation = {alignment_gmax_weak:.2f})',
               xy=(0.5, 0.3), fontsize=10, ha='center',
               transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('/home/claude/ch12_example3_selection.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nFigure saved to: /home/claude/ch12_example3_selection.png")
    
    print("\n" + "-" * 70)
    print("BIOLOGICAL CONCLUSIONS")
    print("-" * 70)
    print(f"""
    1. DIRECTIONAL SELECTION:
       Both traits are favoured to increase (β > 0).
       Trait 1 is under slightly stronger selection.
    
    2. STABILISING SELECTION:
       The fitness surface has an elongated peak (ridge).
       • Weak stabilising along "both high/low" direction
       • Strong stabilising along "contrast" direction
       
       This means: overall size can vary, but proportions are constrained.
    
    3. RESPONSE PREDICTION:
       Selection targets direction [{beta_direction[0]:.2f}, {beta_direction[1]:.2f}]
       Response is in direction [{response_direction[0]:.2f}, {response_direction[1]:.2f}]
       Deflection = {angle_degrees:.1f}° toward g_max
       
       The genetic correlation deflects the response.
    
    4. G-γ ALIGNMENT:
       Maximum genetic variance (g_max) aligns with weak stabilising (v_weak).
       
       This is evolutionarily favourable:
       • Evolution can proceed along the fitness ridge
       • The population is not fighting strong curvature
       • Constraint is ALIGNED with selection tolerance
       
       This pattern is expected under mutation-selection balance:
       selection erodes variance where curvature is strong.
    
    5. LONG-TERM PREDICTION:
       • Size variation will be maintained (high G, low |γ|)
       • Proportion variation will be eroded (low G, high |γ|)
       • The population will evolve along the ridge, not across it
    """)
    
    return {
        'beta': beta, 'gamma': gamma, 'G': G,
        'response': response,
        'eigenvalues_G': eigenvalues_G, 'eigenvectors_G': eigenvectors_G,
        'eigenvalues_gamma': eigenvalues_gamma, 'eigenvectors_gamma': eigenvectors_gamma
    }


# =============================================================================
# COMPUTATIONAL TOOLS SECTION
# =============================================================================


def print_computational_summary():
    """
    Print a summary of the key computational functions.
    """
    print("\n" + "=" * 70)
    print("COMPUTATIONAL TOOLS SUMMARY")
    print("=" * 70)
    print("""
    KEY FUNCTIONS IN THIS MODULE:
    
    ┌────────────────────────────┬────────────────────────────────────────┐
    │ Function                   │ Purpose                                │
    ├────────────────────────────┼────────────────────────────────────────┤
    │ eigendecompose(A)          │ A = V Λ V^T for symmetric A           │
    │ compute_P_inv_sqrt(P)      │ P^{-1/2} for whitening                │
    │ compute_G_star(G, P)       │ G* = P^{-1/2} G P^{-1/2}              │
    │ directional_heritability() │ h²(β) = β'Gβ / β'Pβ                   │
    │ selection_response(G, β)   │ Δz̄ = Gβ (breeder's equation)         │
    │ evolvability(β, G)         │ e(β) = β'Gβ (genetic variance)        │
    └────────────────────────────┴────────────────────────────────────────┘
    
    EXAMPLE USAGE:
    
    >>> G = np.array([[0.5, 0.3], [0.3, 0.4]])
    >>> P = np.array([[1.0, 0.4], [0.4, 0.8]])
    >>> 
    >>> # Compute G* and its eigenvalues
    >>> G_star = compute_G_star(G, P)
    >>> eigenvalues, eigenvectors = eigendecompose(G_star)
    >>> print(f"Max h² = {eigenvalues[0]:.3f}")
    >>> print(f"Min h² = {eigenvalues[-1]:.3f}")
    >>> 
    >>> # Compute heritability in a specific direction
    >>> beta = np.array([1, 1]) / np.sqrt(2)
    >>> h2 = directional_heritability(beta, G, P)
    >>> print(f"h² along β: {h2:.3f}")
    >>> 
    >>> # Predict response to selection
    >>> selection = np.array([0.2, 0.1])
    >>> response = selection_response(G, selection)
    >>> print(f"Response: {response}")
    """)


# =============================================================================
# MAIN EXECUTION
# =============================================================================


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║  CHAPTER 12: WORKED EXAMPLES - COMPLETE ANALYSES                    ║
    ║  Seeing the Shape - Code Companion                                   ║
    ╚══════════════════════════════════════════════════════════════════════╝
    
    This chapter brings together everything from the book:
    
    "The shape of the ellipsoid and the direction of the arrow—these two
     things, together, determine what will happen. The G matrix is potential;
     selection is actuality. Their interaction is evolution."
    
    Running three complete worked examples...
    """)
    
    # Run all examples
    result1 = example_1_two_trait_analysis()
    result2 = example_2_four_trait_gp()
    result3 = example_3_selection_analysis()
    
    # Print computational summary
    print_computational_summary()
    
    print("\n" + "=" * 70)
    print("CHAPTER 12 COMPLETE")
    print("=" * 70)
    print("""
    THE GEOMETRIC PERSPECTIVE - A FINAL WORD:
    
    These examples illustrate that matrices are not just tables of numbers.
    They are SHAPES that constrain and channel evolution:
    
    • G is an ellipsoid showing where evolution CAN go
    • γ is a paraboloid showing where selection WANTS to go  
    • Their alignment determines whether evolution is fast or slow,
      direct or deflected
    
    By visualising these objects and understanding their eigenstructure,
    we gain insight into evolutionary potential and constraint that would
    be invisible from univariate analyses alone.
    
    "Symmetric matrices describe shapes. The algebra is a precise language
     for those shapes. Whenever the symbols become opaque, the right move
     is to go back to the picture and draw the ellipse."
    """)
    
    print("\nOutput files created:")
    print("  • /home/claude/ch12_example1_two_trait.png")
    print("  • /home/claude/ch12_example2_four_trait.png")
    print("  • /home/claude/ch12_example3_selection.png")
