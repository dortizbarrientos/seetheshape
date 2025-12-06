#!/usr/bin/env python3
"""
================================================================================
SEEING THE SHAPE: A Geometric Introduction to Multivariate Quantitative Genetics
================================================================================
Chapter 8: Whitening and the P-sphere
================================================================================

Key Insight:
-----------
    To compare GENETIC variation (G) and PHENOTYPIC variation (P) fairly, we 
    need a common reference frame. The whitening transformation provides this:
    
        P^{-1/2} transforms P → I (the identity)
        
    In this "whitened" space:
    
        - The P-sphere becomes the ordinary unit sphere
        - G* = P^{-1/2} G P^{-1/2} reveals DIRECTIONAL HERITABILITY
        - Eigenvalues of G* ARE the extreme heritabilities
        - Low-eigenvalue directions of G* are CONSTRAINT TRAPS
    
    This unifies the geometry of G and P into a single picture.

Sections in this file:
---------------------
    8.1  The problem: comparing G and P
    8.2  The naive approach and its problem
    8.3  The P-sphere: uniform with respect to phenotype
    8.4  The whitening transformation
    8.5  A remarkable fact: eigenvalues of G* are directional heritabilities
    8.6  The distribution of directional heritability
    8.7  Constraint traps
    8.8  Visualising G inside P
    8.9  Connection to the breeder's equation
    8.10 Computing G* in practice
    8.11 Why whitening matters

Author: Code companion to Ortiz-Barrientos (2025)
License: CC BY-NC-SA 4.0
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
from matplotlib.lines import Line2D
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# Publication-quality defaults
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


# =============================================================================
# SECTION 8.1: THE PROBLEM—COMPARING G AND P
# =============================================================================
"""
The Fundamental Problem:
------------------------
In quantitative genetics, we have TWO fundamental matrices:

    G = additive genetic covariance matrix (heritable variation)
    P = phenotypic covariance matrix (total observed variation)

For a SINGLE trait, heritability is simple:
    
    h² = V_G / V_P

But with MULTIPLE traits, G and P are MATRICES. How do we generalize h²?

The answer: DIRECTIONAL heritability
    
    h²(β) = β^T G β / β^T P β
    
    = "fraction of phenotypic variance that is genetic in direction β"

The challenge: h²(β) varies with direction! Some directions have high 
heritability, others have low. How do we sample directions "fairly"?
"""

def demonstrate_problem():
    """
    Show why comparing G and P is not straightforward.
    """
    print("\n" + "="*70)
    print("SECTION 8.1: The Problem—Comparing G and P")
    print("="*70)
    
    # Example G and P matrices
    G = np.array([[0.45, 0.30],
                  [0.30, 0.35]])
    
    P = np.array([[0.90, 0.40],
                  [0.40, 0.70]])
    
    print("\nGenetic covariance matrix G:")
    print(G)
    print("\nPhenotypic covariance matrix P:")
    print(P)
    
    # Univariate heritabilities
    h2_trait1 = G[0, 0] / P[0, 0]
    h2_trait2 = G[1, 1] / P[1, 1]
    
    print(f"\nUnivariate heritabilities:")
    print(f"  Trait 1: h² = {G[0,0]:.2f} / {P[0,0]:.2f} = {h2_trait1:.3f}")
    print(f"  Trait 2: h² = {G[1,1]:.2f} / {P[1,1]:.2f} = {h2_trait2:.3f}")
    
    # Directional heritability function
    def h2_direction(beta, G, P):
        """Heritability in direction beta."""
        beta = beta / np.linalg.norm(beta)
        return (beta @ G @ beta) / (beta @ P @ beta)
    
    # Try several directions
    print("\n" + "-"*50)
    print("Directional heritabilities in various directions:")
    print("-"*50)
    
    directions = [
        ("Trait 1 only", np.array([1, 0])),
        ("Trait 2 only", np.array([0, 1])),
        ("Both equal", np.array([1, 1])),
        ("Opposite", np.array([1, -1])),
        ("60° from trait 1", np.array([np.cos(np.pi/3), np.sin(np.pi/3)])),
    ]
    
    for name, beta in directions:
        h2 = h2_direction(beta, G, P)
        print(f"  {name:20s}: h²(β) = {h2:.3f}")
    
    print("\n" + "-"*50)
    print("THE PROBLEM:")
    print("-"*50)
    print("""
    Heritability VARIES with direction!
    
    We need a principled way to:
    1. Define "all directions" fairly (respecting phenotypic variance)
    2. Find the maximum and minimum heritabilities
    3. Identify which directions are constraint traps
    
    The answer: the P-sphere and whitening transformation.
    """)
    
    return G, P


# =============================================================================
# SECTION 8.2: THE NAIVE APPROACH AND ITS PROBLEM
# =============================================================================
"""
The Naive Approach:
-------------------
Sample directions uniformly from the unit sphere: all β with ||β|| = 1.

The Problem:
------------
"Uniform on the unit sphere" is ambiguous when traits have different scales!

Consider:
    - Body mass in kg
    - Wing length in mm

A "uniform" sample emphasizes directions dominated by the larger numerical 
scale. This is meaningless biologically.

Even after standardizing each trait to variance 1, if traits are CORRELATED,
sampling uniformly from the Euclidean unit sphere still ignores the 
covariance structure.
"""

def demonstrate_naive_problem():
    """
    Show why uniform sampling on the Euclidean sphere is problematic.
    """
    print("\n" + "="*70)
    print("SECTION 8.2: The Naive Approach and Its Problem")
    print("="*70)
    
    # P matrix with different variances
    P = np.array([[100, 30],
                  [30, 4]])
    
    print("\nPhenotypic covariance matrix P:")
    print(P)
    print(f"\n  Var(trait 1) = {P[0,0]} (large scale, e.g., mass in grams)")
    print(f"  Var(trait 2) = {P[1,1]} (small scale, e.g., color score)")
    
    # Sample "uniform" directions from Euclidean unit sphere
    np.random.seed(42)
    n_samples = 1000
    angles = np.random.uniform(0, 2*np.pi, n_samples)
    euclidean_directions = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Compute phenotypic variance in each direction
    pheno_variances = np.array([d @ P @ d for d in euclidean_directions])
    
    print(f"\n" + "-"*50)
    print("'Uniform' directions on Euclidean unit sphere:")
    print("-"*50)
    print(f"  Phenotypic variance ranges from {pheno_variances.min():.1f} to {pheno_variances.max():.1f}")
    print(f"  Mean: {pheno_variances.mean():.1f}, Std: {pheno_variances.std():.1f}")
    
    print("""
    The problem: These directions are NOT equivalent phenotypically!
    
    A direction pointing toward trait 1 has HUGE phenotypic variance (≈100)
    A direction pointing toward trait 2 has SMALL phenotypic variance (≈4)
    
    We're mixing apples and oranges. We need directions that are 
    EQUAL in phenotypic variance—that's the P-sphere.
    """)


# =============================================================================
# SECTION 8.3: THE P-SPHERE—UNIFORM WITH RESPECT TO PHENOTYPE
# =============================================================================
"""
The P-sphere:
-------------
Instead of the Euclidean unit sphere {β : β^T β = 1}, define:

    P-sphere = {β : β^T P β = 1}

Points on the P-sphere all have UNIT PHENOTYPIC VARIANCE.

This is the natural normalization: we ask "per unit of phenotypic variance, 
how much is genetic?"

In original coordinates, the P-sphere is an ellipse (matching P's shape).
After whitening, it becomes the ordinary unit sphere.
"""

def demonstrate_p_sphere():
    """
    Show the P-sphere and how it differs from the Euclidean sphere.
    """
    print("\n" + "="*70)
    print("SECTION 8.3: The P-sphere—Uniform with Respect to Phenotype")
    print("="*70)
    
    P = np.array([[1.0, 0.6],
                  [0.6, 1.0]])
    
    print("\nPhenotypic covariance matrix P:")
    print(P)
    
    # Generate points on Euclidean unit sphere
    theta = np.linspace(0, 2*np.pi, 360)
    euclidean_sphere = np.column_stack([np.cos(theta), np.sin(theta)])
    
    # Generate points on P-sphere: β^T P β = 1
    # These satisfy β = P^{-1/2} u where u is on unit sphere
    eigenvalues, eigenvectors = np.linalg.eigh(P)
    P_inv_sqrt = eigenvectors @ np.diag(1.0/np.sqrt(eigenvalues)) @ eigenvectors.T
    P_sqrt = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
    
    # P-sphere points: (P^{1/2})^{-1} × unit circle = P^{-1/2} × unit circle
    p_sphere = (P_inv_sqrt @ euclidean_sphere.T).T
    
    # Verify all P-sphere points have unit P-variance
    p_variances = np.array([d @ P @ d for d in p_sphere])
    print(f"\nP-sphere verification:")
    print(f"  All points have β^T P β = 1? Max deviation: {np.max(np.abs(p_variances - 1)):.2e} ✓")
    
    # Compare Euclidean and P-sphere at specific angles
    print("\n" + "-"*50)
    print("Comparison at specific directions:")
    print("-"*50)
    
    for angle_deg in [0, 45, 90]:
        angle = np.radians(angle_deg)
        
        # Euclidean unit vector
        u = np.array([np.cos(angle), np.sin(angle)])
        euclidean_var = u @ P @ u
        
        # P-sphere point in same direction
        p_point = P_inv_sqrt @ u
        p_point = p_point / np.linalg.norm(p_point)  # Normalize to P-sphere
        # Actually, P-sphere point is the Euclidean point scaled to have P-var=1
        scale = 1.0 / np.sqrt(euclidean_var)
        p_point = u * scale
        p_var = p_point @ P @ p_point
        
        print(f"\n  Direction {angle_deg}°:")
        print(f"    Euclidean unit: {u}, P-variance = {euclidean_var:.3f}")
        print(f"    P-sphere point: {p_point}, P-variance = {p_var:.3f}")
    
    print("\n" + "-"*50)
    print("KEY INSIGHT:")
    print("-"*50)
    print("""
    The P-sphere ensures we compare directions that have EQUAL 
    phenotypic variance. This is the fair comparison:
    
        "Per unit of phenotypic variance, how much is genetic?"
    
    In the original coordinate system, the P-sphere is an ellipse.
    After whitening, it becomes the ordinary unit circle/sphere.
    """)
    
    return P, euclidean_sphere, p_sphere


def visualize_p_sphere():
    """
    Create a figure comparing the Euclidean sphere and P-sphere.
    """
    print("\n  Generating P-sphere figure...")
    
    P = np.array([[1.0, 0.6],
                  [0.6, 1.0]])
    
    eigenvalues, eigenvectors = np.linalg.eigh(P)
    P_inv_sqrt = eigenvectors @ np.diag(1.0/np.sqrt(eigenvalues)) @ eigenvectors.T
    
    # Generate shapes
    theta = np.linspace(0, 2*np.pi, 360)
    euclidean_circle = np.column_stack([np.cos(theta), np.sin(theta)])
    p_sphere = (P_inv_sqrt @ euclidean_circle.T).T
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    
    # Panel (a): Original space
    ax = axes[0]
    ax.set_title('(a) Original Trait Space\n' + 
                 r'Euclidean circle $\neq$ P-sphere', fontsize=12, fontweight='bold')
    
    # Draw Euclidean unit circle
    ax.plot(euclidean_circle[:, 0], euclidean_circle[:, 1], 
            '-', color='steelblue', lw=2, label='Euclidean unit circle')
    
    # Draw P-sphere (ellipse)
    ax.plot(p_sphere[:, 0], p_sphere[:, 1], 
            '-', color='forestgreen', lw=2.5, label=r'P-sphere: $\beta^\top\mathbf{P}\beta = 1$')
    
    # Mark points with same angle but different radii
    for angle_deg, marker, color in [(45, 'o', 'red'), (0, 's', 'orange')]:
        angle = np.radians(angle_deg)
        
        # Euclidean point
        u = np.array([np.cos(angle), np.sin(angle)])
        ax.plot(u[0], u[1], marker, color=color, markersize=10, 
                markeredgecolor='white', markeredgewidth=1.5)
        
        # P-sphere point in same direction
        var = u @ P @ u
        p_point = u / np.sqrt(var)
        ax.plot(p_point[0], p_point[1], marker, color=color, markersize=10,
                markeredgecolor='white', markeredgewidth=1.5)
        
        # Connect with dashed line
        ax.plot([u[0], p_point[0]], [u[1], p_point[1]], '--', color=color, lw=1.5, alpha=0.7)
    
    ax.set_xlabel('Trait 1', fontsize=12)
    ax.set_ylabel('Trait 2', fontsize=12)
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', lw=0.5)
    ax.axvline(x=0, color='k', lw=0.5)
    
    # Panel (b): Whitened space
    ax = axes[1]
    ax.set_title('(b) Whitened Space\n' + 
                 r'P-sphere = unit circle', fontsize=12, fontweight='bold')
    
    # In whitened space, P-sphere IS the unit circle
    ax.plot(euclidean_circle[:, 0], euclidean_circle[:, 1], 
            '-', color='forestgreen', lw=2.5, 
            label=r'P-sphere = unit circle')
    
    # The original Euclidean circle becomes an ellipse
    P_sqrt = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
    euclidean_whitened = (P_sqrt @ euclidean_circle.T).T
    ax.plot(euclidean_whitened[:, 0], euclidean_whitened[:, 1], 
            '-', color='steelblue', lw=2, alpha=0.7,
            label='Original Euclidean circle')
    
    ax.set_xlabel('Whitened trait 1', fontsize=12)
    ax.set_ylabel('Whitened trait 2', fontsize=12)
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', lw=0.5)
    ax.axvline(x=0, color='k', lw=0.5)
    
    # Annotations
    ax.text(0, -1.5, 'After whitening:\nUniform on P-sphere =\nUniform on unit circle',
            ha='center', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.suptitle('The P-sphere Ensures Fair Comparison of Directions',
                 fontsize=13, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig('fig_08_03_p_sphere.png', dpi=300, bbox_inches='tight')
    plt.savefig('fig_08_03_p_sphere.pdf', bbox_inches='tight')
    print("  Saved: fig_08_03_p_sphere.png/pdf")
    plt.close()


# =============================================================================
# SECTION 8.4: THE WHITENING TRANSFORMATION
# =============================================================================
"""
The Whitening Transformation:
-----------------------------
Define P^{-1/2} using eigendecomposition:

    P = V_P Λ_P V_P^T
    P^{-1/2} = V_P Λ_P^{-1/2} V_P^T

where Λ_P^{-1/2} has diagonal entries 1/√λᵢ.

Apply this to both G and P:

    P* = P^{-1/2} P P^{-1/2} = I  (the identity!)
    G* = P^{-1/2} G P^{-1/2}      (the whitened genetic matrix)

In whitened coordinates:
    - P becomes the identity (spherical)
    - The P-sphere becomes the ordinary unit sphere
    - Uniform sampling is now trivial
    - G* reveals the HERITABILITY structure
"""

def demonstrate_whitening():
    """
    Show the whitening transformation step by step.
    """
    print("\n" + "="*70)
    print("SECTION 8.4: The Whitening Transformation")
    print("="*70)
    
    # Define G and P
    G = np.array([[0.45, 0.30],
                  [0.30, 0.35]])
    
    P = np.array([[0.90, 0.40],
                  [0.40, 0.70]])
    
    print("\nOriginal matrices:")
    print("\nG (genetic covariance):")
    print(G)
    print("\nP (phenotypic covariance):")
    print(P)
    
    # Step 1: Eigendecompose P
    print("\n" + "-"*50)
    print("STEP 1: Eigendecompose P = V_P Λ_P V_P^T")
    print("-"*50)
    
    eigenvalues_P, V_P = np.linalg.eigh(P)
    idx = np.argsort(eigenvalues_P)[::-1]
    eigenvalues_P = eigenvalues_P[idx]
    V_P = V_P[:, idx]
    Lambda_P = np.diag(eigenvalues_P)
    
    print(f"\nEigenvalues of P: {eigenvalues_P}")
    print(f"\nEigenvectors V_P:")
    print(V_P)
    
    # Verify reconstruction
    P_reconstructed = V_P @ Lambda_P @ V_P.T
    print(f"\nReconstruction check: max|P - V_P Λ_P V_P^T| = {np.max(np.abs(P - P_reconstructed)):.2e} ✓")
    
    # Step 2: Compute P^{-1/2}
    print("\n" + "-"*50)
    print("STEP 2: Compute P^{-1/2} = V_P Λ_P^{-1/2} V_P^T")
    print("-"*50)
    
    Lambda_P_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues_P))
    P_inv_sqrt = V_P @ Lambda_P_inv_sqrt @ V_P.T
    
    print(f"\nΛ_P^{{-1/2}} = diag(1/√λ₁, 1/√λ₂) = diag({1/np.sqrt(eigenvalues_P[0]):.4f}, {1/np.sqrt(eigenvalues_P[1]):.4f})")
    print(f"\nP^{{-1/2}}:")
    print(np.round(P_inv_sqrt, 4))
    
    # Verify: P^{-1/2} P P^{-1/2} = I
    P_star = P_inv_sqrt @ P @ P_inv_sqrt
    print(f"\nVerification: P^{{-1/2}} P P^{{-1/2}} = ?")
    print(np.round(P_star, 10))
    print(f"Is this the identity? {np.allclose(P_star, np.eye(2))} ✓")
    
    # Step 3: Compute G*
    print("\n" + "-"*50)
    print("STEP 3: Compute G* = P^{-1/2} G P^{-1/2}")
    print("-"*50)
    
    G_star = P_inv_sqrt @ G @ P_inv_sqrt
    
    print(f"\nG* (whitened genetic matrix):")
    print(np.round(G_star, 4))
    
    # Eigendecompose G*
    eigenvalues_Gstar, eigenvectors_Gstar = np.linalg.eigh(G_star)
    idx = np.argsort(eigenvalues_Gstar)[::-1]
    eigenvalues_Gstar = eigenvalues_Gstar[idx]
    eigenvectors_Gstar = eigenvectors_Gstar[:, idx]
    
    print(f"\nEigenvalues of G*: {eigenvalues_Gstar}")
    print("(These are the DIRECTIONAL HERITABILITIES along principal axes!)")
    
    return G, P, G_star, P_inv_sqrt


# =============================================================================
# SECTION 8.5: A REMARKABLE FACT—EIGENVALUES OF G* ARE DIRECTIONAL HERITABILITIES
# =============================================================================
"""
The Key Result:
---------------
In whitened coordinates, for a unit vector β* (on the ordinary sphere):

    h²(β*) = (β*)^T G* β* / (β*)^T I β*
           = (β*)^T G* β*

This is just a QUADRATIC FORM in G*!

From Chapter 7, we know:
    - MAXIMUM value = largest eigenvalue of G* = h²_max
    - MINIMUM value = smallest eigenvalue of G* = h²_min
    - Extreme directions = eigenvectors of G*

The eigenvalues of G* ARE the extreme directional heritabilities!
"""

def demonstrate_remarkable_fact():
    """
    Show that eigenvalues of G* are the extreme directional heritabilities.
    """
    print("\n" + "="*70)
    print("SECTION 8.5: Eigenvalues of G* ARE Directional Heritabilities!")
    print("="*70)
    
    # Define matrices
    G = np.array([[0.45, 0.30],
                  [0.30, 0.35]])
    
    P = np.array([[0.90, 0.40],
                  [0.40, 0.70]])
    
    # Compute P^{-1/2} and G*
    eigenvalues_P, V_P = np.linalg.eigh(P)
    P_inv_sqrt = V_P @ np.diag(1.0/np.sqrt(eigenvalues_P)) @ V_P.T
    G_star = P_inv_sqrt @ G @ P_inv_sqrt
    
    # Eigendecompose G*
    eigenvalues_Gstar, eigenvectors_Gstar = np.linalg.eigh(G_star)
    idx = np.argsort(eigenvalues_Gstar)[::-1]
    eigenvalues_Gstar = eigenvalues_Gstar[idx]
    eigenvectors_Gstar = eigenvectors_Gstar[:, idx]
    
    h2_max = eigenvalues_Gstar[0]
    h2_min = eigenvalues_Gstar[1]
    
    print(f"\nG* eigenvalues (= directional heritabilities):")
    print(f"  h²_max = {h2_max:.4f}")
    print(f"  h²_min = {h2_min:.4f}")
    
    print(f"\nG* eigenvectors (= directions of extreme h²):")
    print(f"  v*₁ (max h²): {eigenvectors_Gstar[:, 0]}")
    print(f"  v*₂ (min h²): {eigenvectors_Gstar[:, 1]}")
    
    # Verify by computing h² directly
    print("\n" + "-"*50)
    print("VERIFICATION: Compute h²(β) for eigenvector directions")
    print("-"*50)
    
    def h2_original(beta, G, P):
        """Heritability in original coordinates."""
        beta = beta / np.linalg.norm(beta)
        return (beta @ G @ beta) / (beta @ P @ beta)
    
    # Transform eigenvectors back to original coordinates
    P_sqrt = V_P @ np.diag(np.sqrt(eigenvalues_P)) @ V_P.T
    
    for i, (h2_expected, v_star) in enumerate(zip(eigenvalues_Gstar, eigenvectors_Gstar.T)):
        # v in original coords = P^{1/2} v*
        v_original = P_sqrt @ v_star
        v_original = v_original / np.linalg.norm(v_original)
        
        h2_computed = h2_original(v_original, G, P)
        
        print(f"\n  Direction {i+1}:")
        print(f"    Eigenvector in whitened space: {v_star}")
        print(f"    Eigenvector in original space: {v_original}")
        print(f"    Expected h² (eigenvalue): {h2_expected:.4f}")
        print(f"    Computed h² (β^TGβ/β^TPβ): {h2_computed:.4f}")
        print(f"    Match? {np.isclose(h2_expected, h2_computed)} ✓")
    
    # Sample random directions and verify bounds
    print("\n" + "-"*50)
    print("VERIFICATION: Sample random directions (should all be in bounds)")
    print("-"*50)
    
    n_samples = 1000
    angles = np.random.uniform(0, 2*np.pi, n_samples)
    
    h2_values = []
    for angle in angles:
        # Random unit vector in whitened space
        beta_star = np.array([np.cos(angle), np.sin(angle)])
        # h² = β*^T G* β* (since denominator = 1 in whitened space)
        h2 = beta_star @ G_star @ beta_star
        h2_values.append(h2)
    
    h2_values = np.array(h2_values)
    
    print(f"\n  Sampled {n_samples} random directions:")
    print(f"    Min h²: {h2_values.min():.4f} (should be ≥ {h2_min:.4f})")
    print(f"    Max h²: {h2_values.max():.4f} (should be ≤ {h2_max:.4f})")
    print(f"    Mean h²: {h2_values.mean():.4f}")
    print(f"\n  All values in bounds? {all(h2_values >= h2_min - 1e-10) and all(h2_values <= h2_max + 1e-10)} ✓")
    
    print("\n" + "="*70)
    print("KEY INSIGHT: To find extreme heritabilities, just diagonalize G*!")
    print("             The eigenvalues ARE the extreme h² values.")
    print("="*70)
    
    return G_star, eigenvalues_Gstar, eigenvectors_Gstar


# =============================================================================
# SECTION 8.6: THE DISTRIBUTION OF DIRECTIONAL HERITABILITY
# =============================================================================
"""
The Distribution of h² Across Directions:
-----------------------------------------
For random directions sampled uniformly from the P-sphere (= unit sphere in 
whitened space), the variance of h² is:

    Var[h²(β)] = (2 / (p+2)) × Var(λ*)

where Var(λ*) is the variance of the eigenvalues of G*, and p is the 
number of traits.

The coefficient of variation of directional heritability is:

    CV[h²] = sqrt(2 / (p+2)) × CV(λ*)

Two factors control heritability variation:
    1. Eigenvalue dispersion: How different are the extreme h² values?
    2. Dimensionality: More traits → more "averaging" → less variation
"""

def demonstrate_h2_distribution():
    """
    Explore how directional heritability varies across directions.
    """
    print("\n" + "="*70)
    print("SECTION 8.6: The Distribution of Directional Heritability")
    print("="*70)
    
    # Define matrices
    G = np.array([[0.45, 0.30],
                  [0.30, 0.35]])
    
    P = np.array([[0.90, 0.40],
                  [0.40, 0.70]])
    
    # Compute G*
    eigenvalues_P, V_P = np.linalg.eigh(P)
    P_inv_sqrt = V_P @ np.diag(1.0/np.sqrt(eigenvalues_P)) @ V_P.T
    G_star = P_inv_sqrt @ G @ P_inv_sqrt
    
    eigenvalues_Gstar, _ = np.linalg.eigh(G_star)
    eigenvalues_Gstar = np.sort(eigenvalues_Gstar)[::-1]
    
    h2_max = eigenvalues_Gstar[0]
    h2_min = eigenvalues_Gstar[1]
    mean_h2 = np.mean(eigenvalues_Gstar)
    
    print(f"\nG* eigenvalues (directional heritabilities):")
    print(f"  h²_max = {h2_max:.4f}")
    print(f"  h²_min = {h2_min:.4f}")
    print(f"  Mean = {mean_h2:.4f}")
    
    # Theoretical CV
    p = 2
    var_lambda = np.var(eigenvalues_Gstar, ddof=0)  # Population variance
    cv_lambda = np.std(eigenvalues_Gstar, ddof=0) / np.mean(eigenvalues_Gstar)
    cv_h2_theory = np.sqrt(2 / (p + 2)) * cv_lambda
    
    print(f"\nTheoretical formulas (p = {p} traits):")
    print(f"  Var(λ*) = {var_lambda:.6f}")
    print(f"  CV(λ*) = {cv_lambda:.4f}")
    print(f"  CV(h²) = sqrt(2/(p+2)) × CV(λ*) = sqrt(2/4) × {cv_lambda:.4f} = {cv_h2_theory:.4f}")
    
    # Empirical verification via sampling
    print("\n" + "-"*50)
    print("Empirical verification via Monte Carlo sampling:")
    print("-"*50)
    
    n_samples = 10000
    angles = np.random.uniform(0, 2*np.pi, n_samples)
    
    h2_values = []
    for angle in angles:
        beta_star = np.array([np.cos(angle), np.sin(angle)])
        h2 = beta_star @ G_star @ beta_star
        h2_values.append(h2)
    
    h2_values = np.array(h2_values)
    
    mean_h2_empirical = np.mean(h2_values)
    std_h2_empirical = np.std(h2_values, ddof=1)
    cv_h2_empirical = std_h2_empirical / mean_h2_empirical
    
    print(f"\n  Sampled {n_samples} random directions:")
    print(f"    Empirical mean h² = {mean_h2_empirical:.4f} (theory: {mean_h2:.4f})")
    print(f"    Empirical std h² = {std_h2_empirical:.4f}")
    print(f"    Empirical CV(h²) = {cv_h2_empirical:.4f} (theory: {cv_h2_theory:.4f})")
    
    print("\n" + "-"*50)
    print("INTERPRETATION:")
    print("-"*50)
    print(f"""
    The heritability ranges from {h2_min:.2f} to {h2_max:.2f} depending on direction.
    
    CV(h²) = {cv_h2_empirical:.2f} means heritability varies by about 
    {cv_h2_empirical*100:.0f}% around the mean (relative to the mean).
    
    If eigenvalues were equal: CV(h²) = 0 (same h² in all directions)
    If one eigenvalue dominates: CV(h²) → large
    
    More traits (larger p) → more averaging → smaller CV(h²)
    """)
    
    return h2_values, mean_h2, cv_h2_empirical


def visualize_h2_distribution():
    """
    Create a comprehensive figure showing the distribution of directional h².
    """
    print("\n  Generating h² distribution figure...")
    
    # Define matrices
    G = np.array([[0.45, 0.30],
                  [0.30, 0.35]])
    
    P = np.array([[0.90, 0.40],
                  [0.40, 0.70]])
    
    # Compute G*
    eigenvalues_P, V_P = np.linalg.eigh(P)
    P_inv_sqrt = V_P @ np.diag(1.0/np.sqrt(eigenvalues_P)) @ V_P.T
    P_sqrt = V_P @ np.diag(np.sqrt(eigenvalues_P)) @ V_P.T
    G_star = P_inv_sqrt @ G @ P_inv_sqrt
    
    eigenvalues_Gstar, eigenvectors_Gstar = np.linalg.eigh(G_star)
    idx = np.argsort(eigenvalues_Gstar)[::-1]
    eigenvalues_Gstar = eigenvalues_Gstar[idx]
    eigenvectors_Gstar = eigenvectors_Gstar[:, idx]
    
    h2_max = eigenvalues_Gstar[0]
    h2_min = eigenvalues_Gstar[1]
    mean_h2 = np.mean(eigenvalues_Gstar)
    
    # Sample h² values
    n_samples = 10000
    angles = np.random.uniform(0, 2*np.pi, n_samples)
    h2_values = np.array([np.array([np.cos(a), np.sin(a)]) @ G_star @ np.array([np.cos(a), np.sin(a)]) 
                          for a in angles])
    
    # Create figure with 3 panels
    fig = plt.figure(figsize=(14, 5))
    
    # Colors
    color_P = '#28965A'      # Green for P
    color_G = '#2E86AB'      # Blue for G/G*
    color_h2 = '#A23B72'     # Magenta for h²
    color_max = '#E63946'    # Red for max
    color_min = '#1D3557'    # Dark blue for min
    
    # Panel (a): G* ellipse inside P-sphere
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title('(a) G* ellipse inside P-sphere\n(whitened space)', fontsize=11, fontweight='bold')
    
    theta = np.linspace(0, 2*np.pi, 100)
    
    # P-sphere (unit circle in whitened space)
    ax1.plot(np.cos(theta), np.sin(theta), '-', color=color_P, lw=2.5, label='P-sphere (unit circle)')
    
    # G* ellipse
    # Points on G* ellipse: G*^{1/2} × unit circle
    eigenvalues_Gstar_plot, V_Gstar = np.linalg.eigh(G_star)
    G_star_sqrt = V_Gstar @ np.diag(np.sqrt(eigenvalues_Gstar_plot)) @ V_Gstar.T
    g_star_ellipse = (G_star_sqrt @ np.column_stack([np.cos(theta), np.sin(theta)]).T).T
    ax1.plot(g_star_ellipse[:, 0], g_star_ellipse[:, 1], '-', color=color_G, lw=2.5, label='G* ellipse')
    
    # Draw eigenvector directions
    scale = 1.1
    ax1.annotate('', xy=eigenvectors_Gstar[:, 0]*scale, xytext=-eigenvectors_Gstar[:, 0]*scale,
                arrowprops=dict(arrowstyle='<->', color=color_max, lw=2))
    ax1.annotate('', xy=eigenvectors_Gstar[:, 1]*scale, xytext=-eigenvectors_Gstar[:, 1]*scale,
                arrowprops=dict(arrowstyle='<->', color=color_min, lw=2))
    
    ax1.text(eigenvectors_Gstar[0, 0]*scale + 0.1, eigenvectors_Gstar[1, 0]*scale + 0.05,
             rf'$h^2_{{max}} = {h2_max:.2f}$', fontsize=10, color=color_max)
    ax1.text(eigenvectors_Gstar[0, 1]*scale + 0.1, eigenvectors_Gstar[1, 1]*scale - 0.1,
             rf'$h^2_{{min}} = {h2_min:.2f}$', fontsize=10, color=color_min)
    
    ax1.set_xlabel('Whitened trait 1', fontsize=11)
    ax1.set_ylabel('Whitened trait 2', fontsize=11)
    ax1.set_xlim(-1.4, 1.4)
    ax1.set_ylim(-1.4, 1.4)
    ax1.set_aspect('equal')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', lw=0.5)
    ax1.axvline(x=0, color='k', lw=0.5)
    
    # Panel (b): Histogram of h² values
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title(f'(b) Distribution of h²\n({n_samples:,} random directions)', 
                  fontsize=11, fontweight='bold')
    
    # Histogram
    n_bins = 40
    counts, bins, patches = ax2.hist(h2_values, bins=n_bins, density=True, 
                                      alpha=0.7, color=color_h2, edgecolor='white')
    
    # Mark eigenvalues
    ax2.axvline(h2_max, color=color_max, lw=2.5, ls='--', label=rf'$h^2_{{max}} = {h2_max:.3f}$')
    ax2.axvline(h2_min, color=color_min, lw=2.5, ls='--', label=rf'$h^2_{{min}} = {h2_min:.3f}$')
    ax2.axvline(mean_h2, color='black', lw=2, ls=':', label=rf'Mean $h^2 = {mean_h2:.3f}$')
    
    # Shade constraint trap zone
    trap_threshold = mean_h2 - np.std(h2_values)
    ax2.axvspan(h2_min, trap_threshold, alpha=0.2, color='gray', label='Constraint trap zone')
    
    ax2.set_xlabel(r'Directional heritability $h^2(\beta)$', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.set_xlim(h2_min - 0.05, h2_max + 0.05)
    
    # Panel (c): h² as function of angle
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_title('(c) Heritability vs direction\n(180° periodicity)', fontsize=11, fontweight='bold')
    
    # Compute h² for all angles
    fine_angles = np.linspace(0, 2*np.pi, 360)
    h2_by_angle = np.array([np.array([np.cos(a), np.sin(a)]) @ G_star @ np.array([np.cos(a), np.sin(a)]) 
                            for a in fine_angles])
    
    ax3.fill_between(np.degrees(fine_angles), h2_min, h2_by_angle, alpha=0.3, color=color_h2)
    ax3.plot(np.degrees(fine_angles), h2_by_angle, '-', color=color_h2, lw=2)
    
    # Mark bounds
    ax3.axhline(h2_max, color=color_max, lw=2, ls='--')
    ax3.axhline(h2_min, color=color_min, lw=2, ls='--')
    ax3.axhline(mean_h2, color='black', lw=1.5, ls=':')
    
    # Mark eigenvector angles
    angle_max = np.degrees(np.arctan2(eigenvectors_Gstar[1, 0], eigenvectors_Gstar[0, 0]))
    angle_min = np.degrees(np.arctan2(eigenvectors_Gstar[1, 1], eigenvectors_Gstar[0, 1]))
    
    for angle, color in [(angle_max, color_max), (angle_max + 180, color_max),
                         (angle_min, color_min), (angle_min + 180, color_min)]:
        if 0 <= angle <= 360:
            ax3.axvline(angle, color=color, lw=1.5, ls=':', alpha=0.7)
    
    ax3.set_xlabel('Direction angle θ (degrees)', fontsize=11)
    ax3.set_ylabel(r'$h^2(\theta)$', fontsize=11)
    ax3.set_xlim(0, 360)
    ax3.set_ylim(h2_min - 0.05, h2_max + 0.05)
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle('Distribution of Directional Heritability Across the P-sphere',
                 fontsize=13, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig('fig_08_06_h2_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig('fig_08_06_h2_distribution.pdf', bbox_inches='tight')
    print("  Saved: fig_08_06_h2_distribution.png/pdf")
    plt.close()


# =============================================================================
# SECTION 8.7: CONSTRAINT TRAPS
# =============================================================================
"""
Constraint Traps:
-----------------
A CONSTRAINT TRAP is a direction where:
    - Phenotypic variance is NORMAL (we're on the P-sphere)
    - Genetic variance is LOW → Heritability is low
    
These are directions near the eigenvectors of G* with SMALL eigenvalues.

The danger: A breeder or natural selection might target a direction with 
plenty of phenotypic variation, expecting a good response. But if that 
direction is a constraint trap, response will be disappointing—most of 
the variation is environmental, not genetic!

Constraint Severity:
    Severity = 1 - (h²_min / mean h²)
    
    = 0: No constraint (all directions equally heritable)
    → 1: Severe constraint (worst direction has near-zero h²)
"""

def demonstrate_constraint_traps():
    """
    Identify and characterize constraint traps.
    """
    print("\n" + "="*70)
    print("SECTION 8.7: Constraint Traps")
    print("="*70)
    
    # Create a G matrix with a clear constraint
    # High variance in "size" direction, low in "shape" direction
    G = np.array([[0.60, 0.50],
                  [0.50, 0.55]])
    
    P = np.array([[1.00, 0.60],
                  [0.60, 0.90]])
    
    print("\nGenetic covariance matrix G:")
    print(G)
    print("\nPhenotypic covariance matrix P:")
    print(P)
    
    # Compute G*
    eigenvalues_P, V_P = np.linalg.eigh(P)
    P_inv_sqrt = V_P @ np.diag(1.0/np.sqrt(eigenvalues_P)) @ V_P.T
    P_sqrt = V_P @ np.diag(np.sqrt(eigenvalues_P)) @ V_P.T
    G_star = P_inv_sqrt @ G @ P_inv_sqrt
    
    eigenvalues_Gstar, eigenvectors_Gstar = np.linalg.eigh(G_star)
    idx = np.argsort(eigenvalues_Gstar)[::-1]
    eigenvalues_Gstar = eigenvalues_Gstar[idx]
    eigenvectors_Gstar = eigenvectors_Gstar[:, idx]
    
    h2_max = eigenvalues_Gstar[0]
    h2_min = eigenvalues_Gstar[1]
    mean_h2 = np.mean(eigenvalues_Gstar)
    
    print(f"\nG* eigenvalues:")
    print(f"  h²_max = {h2_max:.4f}")
    print(f"  h²_min = {h2_min:.4f}")
    print(f"  Mean h² = {mean_h2:.4f}")
    
    # Constraint severity
    constraint_severity = 1 - h2_min / mean_h2
    
    print(f"\nConstraint severity = 1 - h²_min/mean = {constraint_severity:.3f}")
    
    # Interpret the constraint trap direction
    print("\n" + "-"*50)
    print("CONSTRAINT TRAP ANALYSIS:")
    print("-"*50)
    
    # Transform eigenvectors back to original space
    v_trap_whitened = eigenvectors_Gstar[:, 1]  # Min h² direction
    v_trap_original = P_sqrt @ v_trap_whitened
    v_trap_original = v_trap_original / np.linalg.norm(v_trap_original)
    
    v_free_whitened = eigenvectors_Gstar[:, 0]  # Max h² direction
    v_free_original = P_sqrt @ v_free_whitened
    v_free_original = v_free_original / np.linalg.norm(v_free_original)
    
    print(f"\n  CONSTRAINT TRAP direction (h² = {h2_min:.3f}):")
    print(f"    In whitened space: {v_trap_whitened}")
    print(f"    In original space: {v_trap_original}")
    angle_trap = np.degrees(np.arctan2(v_trap_original[1], v_trap_original[0]))
    print(f"    Angle: {angle_trap:.1f}° from trait 1")
    
    print(f"\n  FREE EVOLUTION direction (h² = {h2_max:.3f}):")
    print(f"    In whitened space: {v_free_whitened}")
    print(f"    In original space: {v_free_original}")
    angle_free = np.degrees(np.arctan2(v_free_original[1], v_free_original[0]))
    print(f"    Angle: {angle_free:.1f}° from trait 1")
    
    # Biological interpretation
    print("\n" + "-"*50)
    print("BIOLOGICAL INTERPRETATION:")
    print("-"*50)
    print(f"""
    The "free evolution" direction ({angle_free:.0f}°) has h² = {h2_max:.2f}:
        Selection here produces strong response
        This direction likely represents "overall size"
    
    The "constraint trap" direction ({angle_trap:.0f}°) has h² = {h2_min:.2f}:
        Selection here produces weak response
        Lots of phenotypic variance, but mostly environmental!
        This direction likely represents "shape" (trait contrast)
    
    Constraint severity = {constraint_severity:.2f}:
        The worst direction has h² that is {(1-constraint_severity)*100:.0f}% 
        of the average heritability.
        
    WARNING: A breeding program targeting trait contrast would be 
    surprised by the poor response!
    """)
    
    return G, P, eigenvalues_Gstar, eigenvectors_Gstar


# =============================================================================
# SECTION 8.8: VISUALISING G INSIDE P
# =============================================================================
"""
The Core Visualization:
-----------------------
Plot the G ellipse and P ellipse together (same center):

    - P ellipse: Shows where PHENOTYPIC variation extends
    - G ellipse: Shows where GENETIC variation extends
    
Directions where G is thin relative to P → LOW heritability
Directions where G nearly fills P → HIGH heritability

After whitening:
    - P becomes the unit circle
    - G* ellipse sits inside it
    - The gap between them shows where heritability is low
"""

def visualize_g_inside_p():
    """
    Create a two-panel figure showing G inside P in original and whitened space.
    """
    print("\n  Generating G inside P figure...")
    
    # Define matrices
    G = np.array([[0.60, 0.50],
                  [0.50, 0.55]])
    
    P = np.array([[1.00, 0.60],
                  [0.60, 0.90]])
    
    # Eigendecompositions
    eigenvalues_P, V_P = np.linalg.eigh(P)
    eigenvalues_G, V_G = np.linalg.eigh(G)
    
    # G*
    P_inv_sqrt = V_P @ np.diag(1.0/np.sqrt(eigenvalues_P)) @ V_P.T
    P_sqrt = V_P @ np.diag(np.sqrt(eigenvalues_P)) @ V_P.T
    G_star = P_inv_sqrt @ G @ P_inv_sqrt
    
    eigenvalues_Gstar, eigenvectors_Gstar = np.linalg.eigh(G_star)
    idx = np.argsort(eigenvalues_Gstar)[::-1]
    eigenvalues_Gstar = eigenvalues_Gstar[idx]
    eigenvectors_Gstar = eigenvectors_Gstar[:, idx]
    
    h2_max = eigenvalues_Gstar[0]
    h2_min = eigenvalues_Gstar[1]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    theta = np.linspace(0, 2*np.pi, 100)
    unit_circle = np.column_stack([np.cos(theta), np.sin(theta)])
    
    # Colors
    color_P = '#28965A'
    color_G = '#2E86AB'
    color_max = '#E63946'
    color_min = '#1D3557'
    
    # Helper to draw ellipse from covariance matrix
    def cov_ellipse_points(cov, n_std=1):
        eigvals, eigvecs = np.linalg.eigh(cov)
        sqrt_cov = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        return (sqrt_cov @ unit_circle.T).T * n_std
    
    # Panel (a): Original space
    ax = axes[0]
    ax.set_title('(a) Original Trait Space\nG and P ellipses (1 SD)', fontsize=12, fontweight='bold')
    
    # P ellipse
    P_ellipse = cov_ellipse_points(P, 1)
    ax.plot(P_ellipse[:, 0], P_ellipse[:, 1], '-', color=color_P, lw=2.5, label='P ellipse')
    ax.fill(P_ellipse[:, 0], P_ellipse[:, 1], color=color_P, alpha=0.1)
    
    # G ellipse
    G_ellipse = cov_ellipse_points(G, 1)
    ax.plot(G_ellipse[:, 0], G_ellipse[:, 1], '-', color=color_G, lw=2.5, label='G ellipse')
    ax.fill(G_ellipse[:, 0], G_ellipse[:, 1], color=color_G, alpha=0.2)
    
    # Draw eigenvectors of G (g_max direction)
    g_max = V_G[:, np.argmax(eigenvalues_G)]
    g_min = V_G[:, np.argmin(eigenvalues_G)]
    
    ax.annotate('', xy=g_max*1.2, xytext=-g_max*1.2,
                arrowprops=dict(arrowstyle='<->', color=color_max, lw=2, linestyle='--'))
    ax.text(g_max[0]*1.25, g_max[1]*1.25, r'$\mathbf{g}_{max}$', fontsize=11, color=color_max)
    
    ax.set_xlabel('Trait 1', fontsize=11)
    ax.set_ylabel('Trait 2', fontsize=11)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', lw=0.5)
    ax.axvline(x=0, color='k', lw=0.5)
    
    # Panel (b): Whitened space
    ax = axes[1]
    ax.set_title('(b) Whitened Space\nP-sphere and G* ellipse', fontsize=12, fontweight='bold')
    
    # P-sphere (unit circle)
    ax.plot(np.cos(theta), np.sin(theta), '-', color=color_P, lw=2.5, label='P-sphere (unit circle)')
    ax.fill(np.cos(theta), np.sin(theta), color=color_P, alpha=0.1)
    
    # G* ellipse
    Gstar_ellipse = cov_ellipse_points(G_star, 1)
    ax.plot(Gstar_ellipse[:, 0], Gstar_ellipse[:, 1], '-', color=color_G, lw=2.5, label='G* ellipse')
    ax.fill(Gstar_ellipse[:, 0], Gstar_ellipse[:, 1], color=color_G, alpha=0.2)
    
    # Draw eigenvector directions with h² labels
    scale = 1.15
    ax.annotate('', xy=eigenvectors_Gstar[:, 0]*scale, xytext=-eigenvectors_Gstar[:, 0]*scale,
                arrowprops=dict(arrowstyle='<->', color=color_max, lw=2))
    ax.annotate('', xy=eigenvectors_Gstar[:, 1]*scale, xytext=-eigenvectors_Gstar[:, 1]*scale,
                arrowprops=dict(arrowstyle='<->', color=color_min, lw=2))
    
    ax.text(eigenvectors_Gstar[0, 0]*scale + 0.08, eigenvectors_Gstar[1, 0]*scale + 0.08,
            rf'$h^2 = {h2_max:.2f}$', fontsize=10, color=color_max, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(eigenvectors_Gstar[0, 1]*scale + 0.12, eigenvectors_Gstar[1, 1]*scale - 0.08,
            rf'$h^2 = {h2_min:.2f}$', fontsize=10, color=color_min, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Mark the "gap" (constraint trap)
    ax.annotate('', xy=(0.9, 0), xytext=(np.sqrt(h2_min), 0),
                arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
    ax.text(0.75, 0.1, 'Gap =\nconstraint', fontsize=9, ha='center', color='gray')
    
    ax.set_xlabel('Whitened trait 1', fontsize=11)
    ax.set_ylabel('Whitened trait 2', fontsize=11)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', lw=0.5)
    ax.axvline(x=0, color='k', lw=0.5)
    
    plt.suptitle('Visualizing G Inside P: Where Heritability Is High or Low',
                 fontsize=13, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig('fig_08_08_g_inside_p.png', dpi=300, bbox_inches='tight')
    plt.savefig('fig_08_08_g_inside_p.pdf', bbox_inches='tight')
    print("  Saved: fig_08_08_g_inside_p.png/pdf")
    plt.close()


# =============================================================================
# SECTION 8.9: CONNECTION TO THE BREEDER'S EQUATION
# =============================================================================
"""
The Multivariate Breeder's Equation:
------------------------------------
    Δz̄ = G P^{-1} S = G β

where S is the selection differential and β = P^{-1} S is the selection gradient.

In whitened coordinates:
    Δz̄* = G* β*

The response depends on:
    1. Direction of selection β*
    2. Shape of G* (where genetic variance exists)

If β* points toward a high-h² direction → STRONG response
If β* points toward a constraint trap → WEAK response
"""

def demonstrate_breeders_equation():
    """
    Show how whitening clarifies the breeder's equation.
    """
    print("\n" + "="*70)
    print("SECTION 8.9: Connection to the Breeder's Equation")
    print("="*70)
    
    # Define matrices
    G = np.array([[0.60, 0.50],
                  [0.50, 0.55]])
    
    P = np.array([[1.00, 0.60],
                  [0.60, 0.90]])
    
    # Compute G*
    eigenvalues_P, V_P = np.linalg.eigh(P)
    P_inv_sqrt = V_P @ np.diag(1.0/np.sqrt(eigenvalues_P)) @ V_P.T
    P_sqrt = V_P @ np.diag(np.sqrt(eigenvalues_P)) @ V_P.T
    G_star = P_inv_sqrt @ G @ P_inv_sqrt
    
    eigenvalues_Gstar, eigenvectors_Gstar = np.linalg.eigh(G_star)
    idx = np.argsort(eigenvalues_Gstar)[::-1]
    eigenvalues_Gstar = eigenvalues_Gstar[idx]
    eigenvectors_Gstar = eigenvectors_Gstar[:, idx]
    
    print("\nMatrices:")
    print(f"G:\n{G}")
    print(f"\nP:\n{P}")
    print(f"\nG*:\n{np.round(G_star, 4)}")
    
    print(f"\nG* eigenvalues: {eigenvalues_Gstar} (directional h²)")
    
    # Consider different selection gradients
    print("\n" + "-"*50)
    print("Selection response depends on direction:")
    print("-"*50)
    
    # Selection scenarios
    scenarios = [
        ("Select for trait 1 only", np.array([1, 0])),
        ("Select for trait 2 only", np.array([0, 1])),
        ("Select for 'size' (both traits)", np.array([1, 1])),
        ("Select for 'shape' (trait contrast)", np.array([1, -1])),
        ("Select along g_max", eigenvectors_Gstar[:, 0]),
        ("Select along constraint trap", eigenvectors_Gstar[:, 1]),
    ]
    
    for name, beta in scenarios:
        beta = beta / np.linalg.norm(beta)
        
        # Response in original coordinates
        response = G @ beta
        response_magnitude = np.linalg.norm(response)
        
        # Response in whitened coordinates
        beta_star = P_inv_sqrt @ beta
        beta_star = beta_star / np.linalg.norm(beta_star)
        response_star = G_star @ beta_star
        
        # Heritability in this direction
        h2 = (beta @ G @ beta) / (beta @ P @ beta)
        
        print(f"\n  {name}:")
        print(f"    β (original) = {beta}")
        print(f"    Δz̄ = Gβ = {np.round(response, 4)}")
        print(f"    |Δz̄| = {response_magnitude:.4f}")
        print(f"    h²(β) = {h2:.4f}")
    
    print("\n" + "-"*50)
    print("KEY INSIGHT:")
    print("-"*50)
    print("""
    The breeder's equation Δz̄ = Gβ shows that:
    
    1. Response DIRECTION is determined by G acting on β
       (Response is deflected toward g_max)
    
    2. Response MAGNITUDE depends on h²(β)
       (Strong response if selection aligns with high-h² direction)
    
    In whitened space, Δz̄* = G* β* makes this transparent:
       G* directly encodes heritability structure
       Selection along eigenvectors of G* has predictable response
    """)


# =============================================================================
# SECTION 8.10: COMPUTING G* IN PRACTICE
# =============================================================================
"""
Computing G* from Estimates of G and P:
---------------------------------------
1. Eigendecompose P: P = V_P Λ_P V_P^T
2. Compute P^{-1/2} = V_P Λ_P^{-1/2} V_P^T
3. Form G* = P^{-1/2} G P^{-1/2}
4. Eigendecompose G* to get directional heritabilities
"""

def compute_g_star(G, P, verbose=True):
    """
    Compute the P-whitened genetic matrix G* and its eigenstructure.
    
    Parameters
    ----------
    G : ndarray
        Additive genetic covariance matrix (p × p)
    P : ndarray
        Phenotypic covariance matrix (p × p)
    verbose : bool
        Print intermediate results
    
    Returns
    -------
    dict with:
        'G_star': The whitened genetic matrix
        'eigenvalues': Directional heritabilities (sorted descending)
        'eigenvectors': Corresponding directions (in whitened space)
        'eigenvectors_original': Directions transformed back to original space
        'P_inv_sqrt': The whitening matrix
        'h2_max', 'h2_min', 'mean_h2': Summary statistics
        'cv_h2': Coefficient of variation of directional h²
        'constraint_severity': 1 - h2_min/mean_h2
    """
    p = G.shape[0]
    
    if verbose:
        print("\n" + "-"*50)
        print("COMPUTING G* = P^{-1/2} G P^{-1/2}")
        print("-"*50)
    
    # Step 1: Eigendecompose P
    eigenvalues_P, V_P = np.linalg.eigh(P)
    
    # Check positive definiteness
    if np.any(eigenvalues_P <= 0):
        print("  WARNING: P has non-positive eigenvalues!")
        print(f"  Eigenvalues of P: {eigenvalues_P}")
    
    # Step 2: Compute P^{-1/2}
    Lambda_P_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues_P))
    P_inv_sqrt = V_P @ Lambda_P_inv_sqrt @ V_P.T
    P_sqrt = V_P @ np.diag(np.sqrt(eigenvalues_P)) @ V_P.T
    
    # Step 3: Compute G*
    G_star = P_inv_sqrt @ G @ P_inv_sqrt
    
    # Step 4: Eigendecompose G*
    eigenvalues_Gstar, eigenvectors_Gstar = np.linalg.eigh(G_star)
    
    # Sort descending
    idx = np.argsort(eigenvalues_Gstar)[::-1]
    eigenvalues_Gstar = eigenvalues_Gstar[idx]
    eigenvectors_Gstar = eigenvectors_Gstar[:, idx]
    
    # Transform eigenvectors back to original space
    eigenvectors_original = np.zeros_like(eigenvectors_Gstar)
    for i in range(p):
        v = P_sqrt @ eigenvectors_Gstar[:, i]
        eigenvectors_original[:, i] = v / np.linalg.norm(v)
    
    # Summary statistics
    h2_max = eigenvalues_Gstar[0]
    h2_min = eigenvalues_Gstar[-1]
    mean_h2 = np.mean(eigenvalues_Gstar)
    
    cv_lambda = np.std(eigenvalues_Gstar, ddof=0) / mean_h2
    cv_h2 = np.sqrt(2 / (p + 2)) * cv_lambda
    
    constraint_severity = 1 - h2_min / mean_h2
    
    if verbose:
        print(f"\n  G* (whitened genetic matrix):")
        print(np.round(G_star, 4))
        print(f"\n  Eigenvalues of G* (directional heritabilities):")
        for i, h2 in enumerate(eigenvalues_Gstar):
            print(f"    λ*_{i+1} = {h2:.4f}")
        print(f"\n  Summary:")
        print(f"    h²_max = {h2_max:.4f}")
        print(f"    h²_min = {h2_min:.4f}")
        print(f"    Mean h² = {mean_h2:.4f}")
        print(f"    CV(h²) = {cv_h2:.4f}")
        print(f"    Constraint severity = {constraint_severity:.4f}")
    
    return {
        'G_star': G_star,
        'eigenvalues': eigenvalues_Gstar,
        'eigenvectors': eigenvectors_Gstar,
        'eigenvectors_original': eigenvectors_original,
        'P_inv_sqrt': P_inv_sqrt,
        'h2_max': h2_max,
        'h2_min': h2_min,
        'mean_h2': mean_h2,
        'cv_h2': cv_h2,
        'constraint_severity': constraint_severity,
    }


# =============================================================================
# SECTION 8.11: WHY WHITENING MATTERS
# =============================================================================
"""
Why Whitening Matters:
----------------------
1. FAIR COMPARISON: Whitening by P ensures directions are equivalent 
   phenotypically. Without it, we mix apples and oranges.

2. SIMPLIFIED SAMPLING: In whitened space, uniform on unit sphere = 
   uniform on P-sphere. Easy to simulate.

3. DIRECT INTERPRETATION: Eigenvalues of G* ARE directional heritabilities.
   No additional calculation needed.

4. CONSTRAINT DETECTION: The gap between G* and unit sphere directly 
   shows where heritability is low.

5. GENERALIZATION: h²(β) = β^T G* β for unit vectors generalizes the 
   univariate h² = V_G/V_P to any direction.
"""

def demonstrate_why_whitening_matters():
    """
    Summarize the key benefits of whitening.
    """
    print("\n" + "="*70)
    print("SECTION 8.11: Why Whitening Matters")
    print("="*70)
    
    print("""
    The whitening transformation P^{-1/2} serves as the multivariate 
    generalization of "dividing by the standard deviation."
    
    In univariate analysis:
        h² = V_G / V_P
        z-score = (x - μ) / σ
    
    In multivariate analysis:
        G* = P^{-1/2} G P^{-1/2}
        Whitened data: z* = P^{-1/2} (x - μ)
    
    Why this works:
    
    1. FAIR COMPARISON
       Without whitening, directions have different phenotypic variances.
       Comparing h² across directions mixes different "units."
       Whitening makes all directions have unit phenotypic variance.
    
    2. SIMPLIFIED GEOMETRY
       The P-sphere (an ellipsoid) becomes the unit sphere.
       Uniform sampling becomes trivial.
       
    3. EIGENVALUES = HERITABILITIES
       The eigenvalues of G* ARE the extreme directional heritabilities.
       No need to compute β^T G β / β^T P β for each direction.
    
    4. CONSTRAINT VISUALIZATION
       The gap between G* ellipse and unit sphere shows constraint.
       Directions where G* is thin have low heritability.
    
    5. CONNECTION TO STATISTICS
       In whitened space, Mahalanobis distance = Euclidean distance.
       PCA of G* gives principal axes of genetic variation relative to P.
    """)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("CHAPTER 8: WHITENING AND THE P-SPHERE")
    print("="*70)
    print("""
    This chapter develops the whitening transformation—the key tool for
    comparing genetic (G) and phenotypic (P) variation fairly.
    
    Key result: The eigenvalues of G* = P^{-1/2} G P^{-1/2} are the 
    extreme directional heritabilities.
    """)
    
    # Section 8.1: The problem
    G, P = demonstrate_problem()
    
    # Section 8.2: Naive approach
    demonstrate_naive_problem()
    
    # Section 8.3: P-sphere
    demonstrate_p_sphere()
    visualize_p_sphere()
    
    # Section 8.4: Whitening
    demonstrate_whitening()
    
    # Section 8.5: Remarkable fact
    demonstrate_remarkable_fact()
    
    # Section 8.6: Distribution of h²
    demonstrate_h2_distribution()
    visualize_h2_distribution()
    
    # Section 8.7: Constraint traps
    demonstrate_constraint_traps()
    
    # Section 8.8: G inside P
    visualize_g_inside_p()
    
    # Section 8.9: Breeder's equation
    demonstrate_breeders_equation()
    
    # Section 8.10: Computing G* in practice
    print("\n" + "="*70)
    print("SECTION 8.10: Computing G* in Practice")
    print("="*70)
    
    # Example with a larger matrix
    G_example = np.array([[0.45, 0.30, 0.15],
                          [0.30, 0.50, 0.25],
                          [0.15, 0.25, 0.35]])
    
    P_example = np.array([[0.90, 0.40, 0.20],
                          [0.40, 0.85, 0.35],
                          [0.20, 0.35, 0.70]])
    
    results = compute_g_star(G_example, P_example)
    
    # Section 8.11: Why whitening matters
    demonstrate_why_whitening_matters()
    
    print("\n" + "="*70)
    print("CHAPTER 8 COMPLETE")
    print("="*70)
    print("""
    KEY TAKEAWAYS:
    
    1. Directional heritability h²(β) = β^T G β / β^T P β varies with direction
    2. The P-sphere {β : β^T P β = 1} is the natural space for comparison
    3. Whitening by P^{-1/2} transforms P → I and G → G*
    4. Eigenvalues of G* ARE the extreme directional heritabilities
    5. Constraint traps are low-eigenvalue directions of G*
    6. The gap between G* and the P-sphere reveals constraint structure
    7. The breeder's equation becomes Δz̄* = G* β* in whitened space
    
    This framework unifies G and P into a single geometric picture,
    revealing which directions are evolvable and which are constrained.
    """)


# =============================================================================
# EXERCISES
# =============================================================================
"""
EXERCISES FOR CHAPTER 8
=======================

EXERCISE 8.1: P-sphere by Hand
------------------------------
For P = [[4, 2], [2, 3]]:
(a) Find the eigenvalues and eigenvectors of P
(b) Compute P^{-1/2}
(c) Verify that P^{-1/2} P P^{-1/2} = I
(d) Sketch the P-sphere in original coordinates (it's an ellipse)

EXERCISE 8.2: Computing G*
--------------------------
Given:
    G = [[0.6, 0.3], [0.3, 0.4]]
    P = [[1.0, 0.4], [0.4, 0.8]]

(a) Compute G* = P^{-1/2} G P^{-1/2}
(b) Find the eigenvalues of G* (these are h²_max and h²_min)
(c) Compare to the univariate heritabilities G[i,i]/P[i,i]
(d) Are the univariate heritabilities always between h²_min and h²_max?

EXERCISE 8.3: Directional Heritability
--------------------------------------
Using the matrices from Exercise 8.2:
(a) Compute h²(β) for β = (1, 0)
(b) Compute h²(β) for β = (0, 1)
(c) Compute h²(β) for β = (1, 1)/√2
(d) Verify that h²(eigenvector of G*) = eigenvalue of G*

EXERCISE 8.4: Constraint Severity
---------------------------------
Three populations have G* eigenvalues:
    Population A: (0.7, 0.6)
    Population B: (0.9, 0.2)
    Population C: (0.5, 0.5)

(a) Compute constraint severity = 1 - h²_min/mean for each
(b) Compute CV(h²) for each (use the formula from Section 8.6)
(c) Which population has the most uniform heritability?
(d) Which population has the strongest constraint traps?

EXERCISE 8.5: Breeding Program Design
--------------------------------------
A breeding program has:
    G = [[1.0, 0.8], [0.8, 1.0]]  (high genetic correlation)
    P = [[2.0, 1.0], [1.0, 2.0]]

The breeder wants to improve trait 2 while keeping trait 1 constant.
This requires selection in the direction β ∝ (0, 1).

(a) Compute G*
(b) What is h²(β) for β = (0, 1)?
(c) Is this a constraint trap? Compare to h²_max and h²_min.
(d) Predict the response Δz̄ = Gβ for unit selection intensity
(e) Why can't the breeder achieve the desired goal (improve 2, keep 1 constant)?

EXERCISE 8.6: Monte Carlo Verification
--------------------------------------
Write code to:
(a) Generate 10,000 random directions from the P-sphere
(b) Compute h²(β) for each direction
(c) Verify that the empirical mean ≈ trace(G*)/p = mean eigenvalue
(d) Verify that all values fall between h²_min and h²_max
(e) Compare the empirical CV to the theoretical formula
"""
