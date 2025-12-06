#!/usr/bin/env python3
"""
================================================================================
SEEING THE SHAPE: A Geometric Introduction to Multivariate Quantitative Genetics
================================================================================
Chapter 7: Diagonalisation and Natural Axes
================================================================================

Key Insight:
-----------
    Eigenvalues and eigenvectors answer a geometric question: in which directions
    does a symmetric matrix act as PURE STRETCHING with no rotation?
    
    These "natural axes" are the principal axes of the ellipse defined by the 
    matrix. Once found, they reveal:
    
        - The DIRECTIONS of maximum and minimum variance
        - The AMOUNTS of variance along each direction
        - A coordinate system where the matrix becomes diagonal (simple!)
    
    For covariance matrices, eigendecomposition IS the ellipse—eigenvectors 
    point along the axes, eigenvalues are the squared semi-axis lengths.

Sections in this file:
---------------------
    7.1  The question that leads to eigenvalues
    7.2  A concrete example (by hand)
    7.3  The spectral theorem: why symmetric matrices are special
    7.4  Diagonalisation: the matrix factorisation A = VΛV^T
    7.5  Geometric interpretation: the ellipse revealed
    7.6  Why diagonalisation simplifies everything
    7.7  The trace and determinant as summaries
    7.8  Variance in any direction: the quadratic form revisited
    7.9  Principal Component Analysis (PCA) in one paragraph
    7.10 Computing eigenvalues and eigenvectors in practice
    7.11 A biological example: the G matrix
    7.12 Positive definiteness and what eigenvalues tell us
    7.13 The condition number: how "ill-behaved" is the matrix?

Author: Code companion to Ortiz-Barrientos (2025)
License: CC BY-NC-SA 4.0
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

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
# SECTION 7.1: THE QUESTION THAT LEADS TO EIGENVALUES
# =============================================================================
"""
The Central Question:
--------------------
Given a symmetric matrix A, most vectors v get ROTATED when transformed:
    
    v → Av   (Av points in a different direction than v)

But some special vectors ONLY get stretched (or compressed):
    
    Av = λv   (Av points in the SAME direction as v, just scaled by λ)

These special vectors are EIGENVECTORS. The scaling factors are EIGENVALUES.

Biological Meaning:
------------------
For a covariance matrix Σ:
    - Eigenvectors are the natural axes of variation
    - Eigenvalues are the variances along those axes
    
For the G matrix:
    - The first eigenvector (g_max) is the direction of maximum genetic variance
    - This is the "line of least evolutionary resistance"
"""

def demonstrate_eigenvector_concept():
    """
    Show that eigenvectors are special directions that don't rotate.
    
    For most vectors v, Av points in a different direction.
    For eigenvectors, Av points in the SAME direction (just stretched).
    """
    print("\n" + "="*70)
    print("SECTION 7.1: The Question That Leads to Eigenvalues")
    print("="*70)
    
    # Define a symmetric matrix
    A = np.array([[3, 1],
                  [1, 3]])
    
    print("\nMatrix A:")
    print(A)
    
    # Test several vectors
    test_vectors = [
        np.array([1, 0]),      # Standard basis vector e1
        np.array([0, 1]),      # Standard basis vector e2
        np.array([1, 1]),      # Diagonal direction (this IS an eigenvector!)
        np.array([1, -1]),     # Anti-diagonal (also an eigenvector!)
        np.array([2, 1]),      # Arbitrary vector
    ]
    
    print("\nTesting whether Av is parallel to v:")
    print("-" * 60)
    
    for v in test_vectors:
        Av = A @ v
        
        # Check if Av is parallel to v (cross product = 0 in 2D)
        # Cross product in 2D: v1 * Av2 - v2 * Av1
        cross = v[0] * Av[1] - v[1] * Av[0]
        is_parallel = abs(cross) < 1e-10
        
        # If parallel, compute the eigenvalue (ratio)
        if is_parallel:
            # Find non-zero component to compute ratio
            if abs(v[0]) > 1e-10:
                eigenvalue = Av[0] / v[0]
            else:
                eigenvalue = Av[1] / v[1]
            status = f"✓ EIGENVECTOR! λ = {eigenvalue:.1f}"
        else:
            # Compute angle change
            angle_v = np.arctan2(v[1], v[0])
            angle_Av = np.arctan2(Av[1], Av[0])
            angle_change = np.degrees(angle_Av - angle_v)
            status = f"✗ Rotated by {angle_change:.1f}°"
        
        print(f"  v = {v} → Av = {Av}  {status}")
    
    print("\n" + "="*70)
    print("KEY INSIGHT: Eigenvectors are directions where A acts as pure scaling.")
    print("             Finding them reveals the 'natural axes' of the matrix.")
    print("="*70)


# =============================================================================
# SECTION 7.2: A CONCRETE EXAMPLE (BY HAND)
# =============================================================================
"""
Finding Eigenvalues and Eigenvectors by Hand:
---------------------------------------------
For A, we seek λ and non-zero v such that Av = λv.

Rearranging: (A - λI)v = 0

For non-zero v to exist, det(A - λI) = 0  ← The characteristic equation

This gives us λ values. Then we substitute back to find v.
"""

def eigendecomposition_by_hand():
    """
    Walk through the eigendecomposition of a 2×2 matrix step by step.
    
    This mirrors the worked example in Section 7.2 of the book.
    """
    print("\n" + "="*70)
    print("SECTION 7.2: Finding Eigenvalues and Eigenvectors by Hand")
    print("="*70)
    
    # The matrix from the book
    A = np.array([[3, 1],
                  [1, 3]], dtype=float)
    
    print("\nMatrix A:")
    print(A)
    
    # Step 1: Form the characteristic equation
    print("\n" + "-"*50)
    print("STEP 1: Form the characteristic equation det(A - λI) = 0")
    print("-"*50)
    print("""
    A - λI = [3-λ   1  ]
             [ 1   3-λ ]
    
    det(A - λI) = (3-λ)(3-λ) - (1)(1)
                = (3-λ)² - 1
                = λ² - 6λ + 9 - 1
                = λ² - 6λ + 8
                = (λ-4)(λ-2)
                = 0
    
    Therefore: λ₁ = 4, λ₂ = 2
    """)
    
    # Verify with numpy
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    eigenvalues = eigenvalues[::-1]  # Sort descending
    eigenvectors = eigenvectors[:, ::-1]
    
    print(f"  Verification (numpy): λ = {eigenvalues}")
    
    # Step 2: Find eigenvectors for each eigenvalue
    print("\n" + "-"*50)
    print("STEP 2: Find eigenvectors for each eigenvalue")
    print("-"*50)
    
    # For λ₁ = 4
    print("""
    For λ₁ = 4:
    -----------
    (A - 4I)v = 0
    
    [-1   1 ] [v₁]   [0]
    [ 1  -1 ] [v₂] = [0]
    
    From first row: -v₁ + v₂ = 0  →  v₂ = v₁
    
    So v = t[1, 1]ᵀ for any t ≠ 0
    
    Normalized: v₁ = [1/√2, 1/√2]ᵀ ≈ [0.707, 0.707]ᵀ
    """)
    
    # For λ₂ = 2
    print("""
    For λ₂ = 2:
    -----------
    (A - 2I)v = 0
    
    [1   1 ] [v₁]   [0]
    [1   1 ] [v₂] = [0]
    
    From first row: v₁ + v₂ = 0  →  v₂ = -v₁
    
    So v = t[1, -1]ᵀ for any t ≠ 0
    
    Normalized: v₂ = [1/√2, -1/√2]ᵀ ≈ [0.707, -0.707]ᵀ
    """)
    
    print(f"  Verification (numpy):")
    print(f"    v₁ = {eigenvectors[:, 0]}")
    print(f"    v₂ = {eigenvectors[:, 1]}")
    
    # Step 3: Verify orthogonality
    print("\n" + "-"*50)
    print("STEP 3: Verify orthogonality")
    print("-"*50)
    
    v1 = eigenvectors[:, 0]
    v2 = eigenvectors[:, 1]
    dot_product = np.dot(v1, v2)
    
    print(f"  v₁ · v₂ = {dot_product:.10f}")
    print(f"  (Should be 0 for orthogonal vectors)")
    print(f"  ✓ Eigenvectors are PERPENDICULAR!")
    
    # Step 4: Verify Av = λv
    print("\n" + "-"*50)
    print("STEP 4: Verify Av = λv")
    print("-"*50)
    
    for i, (lam, v) in enumerate(zip(eigenvalues, eigenvectors.T)):
        Av = A @ v
        lam_v = lam * v
        diff = np.max(np.abs(Av - lam_v))
        print(f"  λ_{i+1} = {lam:.1f}: Av = {Av}, λv = {lam_v}")
        print(f"       Max difference: {diff:.2e} ✓")
    
    return eigenvalues, eigenvectors


# =============================================================================
# SECTION 7.3: THE SPECTRAL THEOREM
# =============================================================================
"""
The Spectral Theorem for Symmetric Matrices:
--------------------------------------------
Every symmetric matrix has THREE special properties:

    1. REAL EIGENVALUES: All eigenvalues are real numbers (never complex)
    
    2. ORTHOGONAL EIGENVECTORS: Eigenvectors for different eigenvalues are 
       perpendicular
    
    3. COMPLETE SET: There are p eigenvectors forming an orthonormal basis

Why This Matters:
-----------------
    - Covariance matrices are ALWAYS symmetric (Cov(X,Y) = Cov(Y,X))
    - This guarantees their eigenvalues are real and eigenvectors are orthogonal
    - The matrix describes pure stretching along perpendicular axes
    - No rotation, no shear, no reflection—just stretching!
"""

def demonstrate_spectral_theorem():
    """
    Demonstrate the three properties guaranteed by the spectral theorem.
    """
    print("\n" + "="*70)
    print("SECTION 7.3: The Spectral Theorem—Why Symmetric Matrices Are Special")
    print("="*70)
    
    # Create a symmetric matrix (covariance matrix)
    Sigma = np.array([[4, 2, 1],
                      [2, 5, 2],
                      [1, 2, 3]])
    
    print("\nSymmetric matrix Σ:")
    print(Sigma)
    print(f"\nIs Σ symmetric? {np.allclose(Sigma, Sigma.T)}")
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
    
    # Property 1: Real eigenvalues
    print("\n" + "-"*50)
    print("Property 1: ALL EIGENVALUES ARE REAL")
    print("-"*50)
    print(f"  Eigenvalues: {eigenvalues}")
    print(f"  All real? ✓")
    
    # Property 2: Orthogonal eigenvectors
    print("\n" + "-"*50)
    print("Property 2: EIGENVECTORS ARE ORTHOGONAL")
    print("-"*50)
    print("  Eigenvector matrix V:")
    print(eigenvectors)
    
    # Check V^T V = I
    VtV = eigenvectors.T @ eigenvectors
    print("\n  V^T V (should be identity):")
    print(np.round(VtV, 10))
    print(f"\n  Is V^T V = I? {np.allclose(VtV, np.eye(3))} ✓")
    
    # Property 3: Complete set (basis)
    print("\n" + "-"*50)
    print("Property 3: EIGENVECTORS FORM A COMPLETE BASIS")
    print("-"*50)
    print(f"  Number of traits (p): 3")
    print(f"  Number of eigenvectors: {eigenvectors.shape[1]}")
    print(f"  They span all of R³ ✓")
    
    # Biological implication
    print("\n" + "="*70)
    print("BIOLOGICAL IMPLICATION:")
    print("  Covariance matrices describe ELLIPSOIDS, not parallelograms.")
    print("  The axes are perpendicular—this is not a choice, it's guaranteed!")
    print("="*70)


# =============================================================================
# SECTION 7.4: DIAGONALISATION—THE MATRIX FACTORISATION
# =============================================================================
"""
The Eigendecomposition:
-----------------------
Every symmetric matrix A can be written as:

    A = V Λ V^T

where:
    V = matrix of eigenvectors (as columns)
    Λ = diagonal matrix of eigenvalues
    V^T = transpose of V

Geometric Interpretation (Rotate–Stretch–Rotate):
-------------------------------------------------
    1. V^T rotates from original axes to eigenvector axes
    2. Λ stretches along each eigenvector axis
    3. V rotates back to original axes

This is WHY symmetric matrices correspond to ellipses!
"""

def demonstrate_diagonalisation():
    """
    Show the eigendecomposition A = VΛV^T and verify it.
    """
    print("\n" + "="*70)
    print("SECTION 7.4: Diagonalisation—The Matrix Factorisation")
    print("="*70)
    
    # The matrix from Section 7.2
    A = np.array([[3, 1],
                  [1, 3]], dtype=float)
    
    print("\nOriginal matrix A:")
    print(A)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    V = eigenvectors
    Lambda = np.diag(eigenvalues)
    
    print("\n" + "-"*50)
    print("Eigendecomposition: A = V Λ V^T")
    print("-"*50)
    
    print("\nV (eigenvector matrix):")
    print(np.round(V, 4))
    
    print("\nΛ (diagonal eigenvalue matrix):")
    print(Lambda)
    
    print("\nV^T:")
    print(np.round(V.T, 4))
    
    # Verify A = V Λ V^T
    print("\n" + "-"*50)
    print("Verification: V Λ V^T = ?")
    print("-"*50)
    
    reconstructed = V @ Lambda @ V.T
    print("\nV Λ V^T =")
    print(np.round(reconstructed, 10))
    
    print(f"\nMatches A? {np.allclose(A, reconstructed)} ✓")
    
    # Show the rotate-stretch-rotate interpretation
    print("\n" + "-"*50)
    print("GEOMETRIC INTERPRETATION: Rotate–Stretch–Rotate")
    print("-"*50)
    print("""
    To apply A to a vector v:
    
    1. V^T (rotate): Transform from original coords to eigenvector coords
       The unit vectors e₁, e₂ become the eigenvectors v₁, v₂
    
    2. Λ (stretch): Scale by λ₁ along first axis, λ₂ along second
       Here: stretch by 4 along v₁ = (1,1)/√2
             stretch by 2 along v₂ = (1,-1)/√2
    
    3. V (rotate back): Transform back to original coordinates
    
    The unit circle → ellipse with axes along eigenvectors
    """)
    
    return V, Lambda


def visualize_rotate_stretch_rotate():
    """
    Create a 4-panel figure showing the rotate-stretch-rotate interpretation.
    
    This is Figure 7.1 in the book (referenced in the code).
    """
    print("\n  Generating rotate-stretch-rotate figure...")
    
    A = np.array([[3, 1],
                  [1, 3]], dtype=float)
    
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    V = eigenvectors
    Lambda = np.diag(eigenvalues)
    
    # Generate points on unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    circle = np.array([np.cos(theta), np.sin(theta)])
    
    # Apply transformations step by step
    step0 = circle                    # Original circle
    step1 = V.T @ circle              # After V^T (rotate to eigenvector axes)
    step2 = Lambda @ step1            # After Λ (stretch)
    step3 = V @ step2                 # After V (rotate back)
    
    # Also compute direct: A @ circle
    direct = A @ circle
    
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    
    colors = {
        'circle': '#1f77b4',
        'ellipse': '#d62728',
        'evec1': '#2ca02c',
        'evec2': '#9467bd',
    }
    
    # Panel 1: Original circle with eigenvectors
    ax = axes[0]
    ax.plot(step0[0], step0[1], '-', color=colors['circle'], lw=2)
    ax.set_title('(a) Original\nUnit circle', fontsize=11)
    
    # Draw standard basis
    ax.annotate('', xy=(1.2, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.annotate('', xy=(0, 1.2), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.text(1.3, -0.1, r'$\mathbf{e}_1$', fontsize=10, color='gray')
    ax.text(0.1, 1.25, r'$\mathbf{e}_2$', fontsize=10, color='gray')
    
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', lw=0.5)
    ax.axvline(x=0, color='k', lw=0.5)
    
    # Panel 2: After V^T (rotate to eigenvector coordinates)
    ax = axes[1]
    ax.plot(step1[0], step1[1], '-', color=colors['circle'], lw=2)
    ax.set_title(r'(b) After $\mathbf{V}^\top$' + '\nRotate to eigenvector axes', fontsize=11)
    
    # Now standard axes ARE the eigenvector directions
    ax.annotate('', xy=(1.2, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=colors['evec1'], lw=1.5))
    ax.annotate('', xy=(0, 1.2), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=colors['evec2'], lw=1.5))
    ax.text(1.3, -0.15, r'$\mathbf{v}_1$ axis', fontsize=9, color=colors['evec1'])
    ax.text(0.1, 1.25, r'$\mathbf{v}_2$ axis', fontsize=9, color=colors['evec2'])
    
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', lw=0.5)
    ax.axvline(x=0, color='k', lw=0.5)
    
    # Panel 3: After Λ (stretch)
    ax = axes[2]
    ax.plot(step2[0], step2[1], '-', color=colors['ellipse'], lw=2)
    ax.set_title(r'(c) After $\mathbf{\Lambda}$' + f'\nStretch by λ₁={eigenvalues[0]:.0f}, λ₂={eigenvalues[1]:.0f}', 
                 fontsize=11)
    
    # Show stretched axes
    ax.annotate('', xy=(eigenvalues[0]**0.5 * 1.2, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=colors['evec1'], lw=1.5))
    ax.annotate('', xy=(0, eigenvalues[1]**0.5 * 1.2), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=colors['evec2'], lw=1.5))
    
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', lw=0.5)
    ax.axvline(x=0, color='k', lw=0.5)
    
    # Panel 4: After V (rotate back) = final ellipse
    ax = axes[3]
    ax.plot(step3[0], step3[1], '-', color=colors['ellipse'], lw=2, label='Ellipse = A × circle')
    ax.set_title(r'(d) After $\mathbf{V}$' + '\nRotate back: final ellipse', fontsize=11)
    
    # Show eigenvectors as axes of the ellipse
    scale = np.sqrt(eigenvalues[0]) * 1.2
    ax.annotate('', xy=V[:, 0]*scale, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=colors['evec1'], lw=2))
    ax.annotate('', xy=-V[:, 0]*scale, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=colors['evec1'], lw=2, linestyle='--'))
    
    scale2 = np.sqrt(eigenvalues[1]) * 1.2
    ax.annotate('', xy=V[:, 1]*scale2, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=colors['evec2'], lw=2))
    ax.annotate('', xy=-V[:, 1]*scale2, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=colors['evec2'], lw=2, linestyle='--'))
    
    ax.text(V[0, 0]*scale + 0.15, V[1, 0]*scale + 0.15, r'$\mathbf{v}_1$', 
            fontsize=10, color=colors['evec1'], fontweight='bold')
    ax.text(V[0, 1]*scale2 + 0.15, V[1, 1]*scale2 - 0.25, r'$\mathbf{v}_2$', 
            fontsize=10, color=colors['evec2'], fontweight='bold')
    
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', lw=0.5)
    ax.axvline(x=0, color='k', lw=0.5)
    
    # Add arrows between panels
    for i in range(3):
        fig.text(0.23 + i*0.225, 0.5, '→', fontsize=24, ha='center', va='center',
                 fontweight='bold', color='gray')
    
    plt.suptitle(r'Eigendecomposition: $\mathbf{A} = \mathbf{V}\mathbf{\Lambda}\mathbf{V}^\top$ means Rotate–Stretch–Rotate',
                 fontsize=13, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig('fig_07_04_rotate_stretch_rotate.png', dpi=300, bbox_inches='tight')
    plt.savefig('fig_07_04_rotate_stretch_rotate.pdf', bbox_inches='tight')
    print("  Saved: fig_07_04_rotate_stretch_rotate.png/pdf")
    plt.close()


# =============================================================================
# SECTION 7.5: GEOMETRIC INTERPRETATION—THE ELLIPSE REVEALED
# =============================================================================
"""
Covariance Matrices ARE Ellipses:
---------------------------------
If Σ is a covariance matrix with eigendecomposition Σ = VΛV^T, then:

    - Eigenvectors v₁, v₂, ..., vₚ are the PRINCIPAL AXES of the ellipse
    - Eigenvalues λ₁, λ₂, ..., λₚ are the VARIANCES along those axes
    - Semi-axis LENGTHS are √λ₁, √λ₂, ..., √λₚ

The ellipse defined by points satisfying:
    
    (z - μ)^T Σ^{-1} (z - μ) = c
    
has axes along eigenvectors with lengths proportional to √λᵢ.
"""

def demonstrate_ellipse_geometry():
    """
    Show how eigenvalues and eigenvectors define the shape of an ellipse.
    """
    print("\n" + "="*70)
    print("SECTION 7.5: Geometric Interpretation—The Ellipse Revealed")
    print("="*70)
    
    # A covariance matrix with correlation
    Sigma = np.array([[1.0, 0.8],
                      [0.8, 1.0]])
    
    print("\nCovariance matrix Σ:")
    print(Sigma)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    print(f"\nEigenvalues: λ₁ = {eigenvalues[0]:.2f}, λ₂ = {eigenvalues[1]:.2f}")
    print(f"Ratio λ₁/λ₂ = {eigenvalues[0]/eigenvalues[1]:.1f}")
    print(f"  → The major axis is {np.sqrt(eigenvalues[0]/eigenvalues[1]):.1f}× longer than minor")
    
    print(f"\nEigenvectors:")
    print(f"  v₁ = {eigenvectors[:, 0]} (direction of max variance)")
    print(f"  v₂ = {eigenvectors[:, 1]} (direction of min variance)")
    
    # Interpret the eigenvector directions
    angle1 = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    angle2 = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
    
    print(f"\nAngles from horizontal:")
    print(f"  v₁: {angle1:.1f}° (45° means 'both traits increase together')")
    print(f"  v₂: {angle2:.1f}° (perpendicular: 'traits in opposition')")
    
    print("\n" + "-"*50)
    print("INTERPRETATION:")
    print("-"*50)
    print(f"""
    The covariance matrix describes an ELLIPSE:
    
    - MAJOR AXIS: Along v₁ = (0.71, 0.71)
      Points at 45° → both traits high or both low together
      Variance = λ₁ = {eigenvalues[0]:.2f}
      
    - MINOR AXIS: Along v₂ = (0.71, -0.71)
      Points at 135° → one trait high, other low
      Variance = λ₂ = {eigenvalues[1]:.2f}
    
    The ellipse is elongated because variance along the positive diagonal 
    (both traits together) is {eigenvalues[0]/eigenvalues[1]:.0f}× larger than 
    variance along the negative diagonal (traits in opposition).
    """)
    
    return Sigma, eigenvalues, eigenvectors


def visualize_covariance_ellipse():
    """
    Create a publication-quality figure showing eigenvalues/eigenvectors
    defining the covariance ellipse.
    """
    print("\n  Generating covariance ellipse figure...")
    
    Sigma = np.array([[1.0, 0.8],
                      [0.8, 1.0]])
    
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Generate data
    np.random.seed(42)
    n = 200
    data = np.random.multivariate_normal([0, 0], Sigma, n)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Plot data points
    ax.scatter(data[:, 0], data[:, 1], alpha=0.3, s=30, c='steelblue', 
               label='Data points')
    
    # Draw covariance ellipse (1, 2, 3 standard deviations)
    for n_std, alpha in [(1, 0.3), (2, 0.2), (3, 0.1)]:
        # Compute ellipse parameters
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width = 2 * n_std * np.sqrt(eigenvalues[0])
        height = 2 * n_std * np.sqrt(eigenvalues[1])
        
        ellipse = Ellipse((0, 0), width, height, angle=angle,
                         fill=False, edgecolor='firebrick', lw=2, 
                         linestyle='-' if n_std == 1 else '--', alpha=1-alpha*1.5)
        ax.add_patch(ellipse)
    
    # Draw eigenvectors
    scale = 2.5
    
    # v₁ (major axis)
    ax.annotate('', xy=eigenvectors[:, 0]*scale, xytext=-eigenvectors[:, 0]*scale,
                arrowprops=dict(arrowstyle='<->', color='forestgreen', lw=2.5))
    ax.text(eigenvectors[0, 0]*scale + 0.15, eigenvectors[1, 0]*scale + 0.1,
            rf'$\mathbf{{v}}_1$: $\lambda_1 = {eigenvalues[0]:.2f}$',
            fontsize=11, color='forestgreen', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # v₂ (minor axis)
    ax.annotate('', xy=eigenvectors[:, 1]*scale*0.5, xytext=-eigenvectors[:, 1]*scale*0.5,
                arrowprops=dict(arrowstyle='<->', color='darkorange', lw=2.5))
    ax.text(eigenvectors[0, 1]*scale*0.5 + 0.2, eigenvectors[1, 1]*scale*0.5 - 0.1,
            rf'$\mathbf{{v}}_2$: $\lambda_2 = {eigenvalues[1]:.2f}$',
            fontsize=11, color='darkorange', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Labels
    ax.set_xlabel('Trait 1', fontsize=12)
    ax.set_ylabel('Trait 2', fontsize=12)
    ax.set_title('Covariance Matrix as Ellipse\n' + 
                 r'Eigenvectors = axes, Eigenvalues = variances along axes',
                 fontsize=13, fontweight='bold')
    
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', lw=0.5)
    ax.axvline(x=0, color='k', lw=0.5)
    
    # Add legend for ellipse contours
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='firebrick', lw=2, label='1σ ellipse'),
        Line2D([0], [0], color='firebrick', lw=2, linestyle='--', alpha=0.7, label='2σ, 3σ ellipses'),
        Line2D([0], [0], color='forestgreen', lw=2.5, label=r'$\mathbf{v}_1$ (major axis)'),
        Line2D([0], [0], color='darkorange', lw=2.5, label=r'$\mathbf{v}_2$ (minor axis)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('fig_07_05_covariance_ellipse.png', dpi=300, bbox_inches='tight')
    plt.savefig('fig_07_05_covariance_ellipse.pdf', bbox_inches='tight')
    print("  Saved: fig_07_05_covariance_ellipse.png/pdf")
    plt.close()


# =============================================================================
# SECTION 7.6: WHY DIAGONALISATION SIMPLIFIES EVERYTHING
# =============================================================================
"""
In the Eigenvector Coordinate System:
-------------------------------------
Once we diagonalise Σ = VΛV^T, everything becomes simple:

    - INVERSE: Σ^{-1} = V Λ^{-1} V^T  (just invert the diagonal)
    - SQUARE ROOT: Σ^{1/2} = V Λ^{1/2} V^T  (just sqrt the diagonal)
    - POWERS: Σ^k = V Λ^k V^T
    - DETERMINANT: det(Σ) = λ₁ × λ₂ × ... × λₚ  (product of eigenvalues)
    - TRACE: tr(Σ) = λ₁ + λ₂ + ... + λₚ  (sum of eigenvalues)

This is the power of diagonalisation: hard matrix operations become easy 
scalar operations on eigenvalues.
"""

def demonstrate_diagonalisation_power():
    """
    Show how diagonalisation simplifies matrix operations.
    """
    print("\n" + "="*70)
    print("SECTION 7.6: Why Diagonalisation Simplifies Everything")
    print("="*70)
    
    Sigma = np.array([[4, 2],
                      [2, 5]])
    
    print("\nCovariance matrix Σ:")
    print(Sigma)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
    V = eigenvectors
    Lambda = np.diag(eigenvalues)
    
    print(f"\nEigenvalues: λ₁ = {eigenvalues[0]:.4f}, λ₂ = {eigenvalues[1]:.4f}")
    
    # Operation 1: Inverse
    print("\n" + "-"*50)
    print("Operation 1: INVERSE")
    print("-"*50)
    
    Lambda_inv = np.diag(1.0 / eigenvalues)
    Sigma_inv_via_eigen = V @ Lambda_inv @ V.T
    Sigma_inv_direct = np.linalg.inv(Sigma)
    
    print("  Via eigendecomposition: Σ^{-1} = V Λ^{-1} V^T")
    print(f"  Λ^{{-1}} = diag(1/λ₁, 1/λ₂) = diag({1/eigenvalues[0]:.4f}, {1/eigenvalues[1]:.4f})")
    print("\n  Result:")
    print(np.round(Sigma_inv_via_eigen, 6))
    print(f"\n  Matches numpy inverse? {np.allclose(Sigma_inv_via_eigen, Sigma_inv_direct)} ✓")
    
    # Operation 2: Square root
    print("\n" + "-"*50)
    print("Operation 2: SQUARE ROOT")
    print("-"*50)
    
    Lambda_sqrt = np.diag(np.sqrt(eigenvalues))
    Sigma_sqrt = V @ Lambda_sqrt @ V.T
    
    print("  Via eigendecomposition: Σ^{1/2} = V Λ^{1/2} V^T")
    print(f"  Λ^{{1/2}} = diag(√λ₁, √λ₂) = diag({np.sqrt(eigenvalues[0]):.4f}, {np.sqrt(eigenvalues[1]):.4f})")
    print("\n  Σ^{1/2}:")
    print(np.round(Sigma_sqrt, 6))
    
    # Verify: (Σ^{1/2})^2 = Σ
    print("\n  Verification: (Σ^{1/2})² = ?")
    print(np.round(Sigma_sqrt @ Sigma_sqrt, 6))
    print(f"  Matches Σ? {np.allclose(Sigma_sqrt @ Sigma_sqrt, Sigma)} ✓")
    
    # Operation 3: Determinant
    print("\n" + "-"*50)
    print("Operation 3: DETERMINANT")
    print("-"*50)
    
    det_via_eigen = np.prod(eigenvalues)
    det_direct = np.linalg.det(Sigma)
    
    print(f"  Via eigenvalues: det(Σ) = λ₁ × λ₂ = {eigenvalues[0]:.4f} × {eigenvalues[1]:.4f} = {det_via_eigen:.4f}")
    print(f"  Via numpy:       det(Σ) = {det_direct:.4f}")
    print(f"  Match? {np.isclose(det_via_eigen, det_direct)} ✓")
    
    # Operation 4: Trace
    print("\n" + "-"*50)
    print("Operation 4: TRACE (total variance)")
    print("-"*50)
    
    trace_via_eigen = np.sum(eigenvalues)
    trace_direct = np.trace(Sigma)
    trace_via_diag = Sigma[0, 0] + Sigma[1, 1]
    
    print(f"  Via eigenvalues: tr(Σ) = λ₁ + λ₂ = {eigenvalues[0]:.4f} + {eigenvalues[1]:.4f} = {trace_via_eigen:.4f}")
    print(f"  Via diagonal:    tr(Σ) = Σ₁₁ + Σ₂₂ = {Sigma[0,0]} + {Sigma[1,1]} = {trace_via_diag}")
    print(f"  Match? {np.isclose(trace_via_eigen, trace_direct)} ✓")
    
    print("\n" + "="*70)
    print("KEY INSIGHT: Diagonalisation converts matrix operations to scalar operations.")
    print("             This is why eigenvectors are so powerful!")
    print("="*70)


# =============================================================================
# SECTION 7.7: THE TRACE AND DETERMINANT AS SUMMARIES
# =============================================================================
"""
Two Key Summaries of the Eigenvalue Spectrum:
---------------------------------------------

TRACE = sum of eigenvalues = sum of diagonal elements
    tr(Σ) = λ₁ + λ₂ + ... + λₚ = Σ₁₁ + Σ₂₂ + ... + Σₚₚ
    
    Interpretation: TOTAL VARIANCE across all traits
    Geometric: Overall "size" of the ellipse
    
DETERMINANT = product of eigenvalues
    det(Σ) = λ₁ × λ₂ × ... × λₚ
    
    Interpretation: GENERALIZED VARIANCE (accounts for correlations)
    Geometric: Squared volume of the ellipsoid
    
    If det = 0: At least one eigenvalue is 0 → matrix is SINGULAR
               The data lie in a lower-dimensional subspace
"""

def demonstrate_trace_and_determinant():
    """
    Show trace and determinant as summaries of eigenvalue spectrum.
    """
    print("\n" + "="*70)
    print("SECTION 7.7: The Trace and Determinant as Summaries")
    print("="*70)
    
    # Create different covariance matrices with same trace but different determinants
    matrices = {
        'Spherical': np.array([[3, 0], [0, 3]]),
        'Eccentric': np.array([[5.5, 0], [0, 0.5]]),
        'Correlated': np.array([[4, 2], [2, 2]]),
    }
    
    print("\nThree matrices with same trace but different determinants:")
    print("-" * 60)
    
    for name, Sigma in matrices.items():
        eigenvalues = np.linalg.eigvalsh(Sigma)
        trace = np.trace(Sigma)
        det = np.linalg.det(Sigma)
        
        print(f"\n{name}:")
        print(Sigma)
        print(f"  Eigenvalues: {eigenvalues}")
        print(f"  Trace = {trace:.2f} (total variance)")
        print(f"  Determinant = {det:.2f} (generalized variance)")
        print(f"  √det = {np.sqrt(det):.2f} (proportional to ellipse 'area')")
    
    print("\n" + "-"*60)
    print("INTERPRETATION:")
    print("-"*60)
    print("""
    All three matrices have trace = 6 (same total variance).
    But their determinants differ:
    
    - Spherical (det=9): Variance spread equally in all directions
      → Circular, largest possible area for given trace
      
    - Eccentric (det=2.75): Variance concentrated in one direction
      → Thin ellipse, much smaller area
      
    - Correlated (det=4): Positive correlation
      → Tilted ellipse, intermediate area
    
    The determinant captures shape information that trace misses.
    Two populations can have equal total variance but very different
    constraint structures—determinant reveals this.
    """)


# =============================================================================
# SECTION 7.8: VARIANCE IN ANY DIRECTION—THE QUADRATIC FORM REVISITED
# =============================================================================
"""
Variance in Direction β:
------------------------
For a unit vector β, the variance of the population in that direction is:

    σ²(β) = β^T Σ β

This is a QUADRATIC FORM. Using eigendecomposition Σ = VΛV^T:

    σ²(β) = β^T V Λ V^T β
          = (V^T β)^T Λ (V^T β)
          = Σᵢ λᵢ cᵢ²

where cᵢ = β · vᵢ (projection of β onto eigenvector vᵢ).

The variance is a WEIGHTED AVERAGE of eigenvalues, weighted by squared 
projections onto eigenvectors.

Bounds:
    λ_min ≤ σ²(β) ≤ λ_max

Variance is maximized when β = v₁, minimized when β = vₚ.
"""

def demonstrate_variance_in_direction():
    """
    Show how variance varies with direction and is bounded by eigenvalues.
    """
    print("\n" + "="*70)
    print("SECTION 7.8: Variance in Any Direction—The Quadratic Form Revisited")
    print("="*70)
    
    Sigma = np.array([[1.0, 0.8],
                      [0.8, 1.0]])
    
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    print("\nCovariance matrix Σ:")
    print(Sigma)
    print(f"\nEigenvalues: λ₁ = {eigenvalues[0]:.2f}, λ₂ = {eigenvalues[1]:.2f}")
    print(f"Eigenvectors:")
    print(f"  v₁ = {eigenvectors[:, 0]}")
    print(f"  v₂ = {eigenvectors[:, 1]}")
    
    print("\n" + "-"*50)
    print("Variance σ²(β) = β^T Σ β for various directions:")
    print("-"*50)
    
    # Test various directions
    test_directions = [
        ('Trait 1 only', np.array([1, 0])),
        ('Trait 2 only', np.array([0, 1])),
        ('Both equal', np.array([1, 1]) / np.sqrt(2)),
        ('Opposite', np.array([1, -1]) / np.sqrt(2)),
        ('v₁ (eigenvector 1)', eigenvectors[:, 0]),
        ('v₂ (eigenvector 2)', eigenvectors[:, 1]),
    ]
    
    for name, beta in test_directions:
        # Ensure unit vector
        beta = beta / np.linalg.norm(beta)
        
        # Compute variance
        var = beta @ Sigma @ beta
        
        # Compute as weighted average of eigenvalues
        c = eigenvectors.T @ beta  # projections onto eigenvectors
        var_weighted = np.sum(eigenvalues * c**2)
        
        print(f"\n  {name}:")
        print(f"    β = {beta}")
        print(f"    σ²(β) = β^T Σ β = {var:.4f}")
        print(f"    Projections: c₁ = {c[0]:.3f}, c₂ = {c[1]:.3f}")
        print(f"    Check: λ₁c₁² + λ₂c₂² = {eigenvalues[0]*c[0]**2:.4f} + {eigenvalues[1]*c[1]**2:.4f} = {var_weighted:.4f}")
    
    print("\n" + "-"*50)
    print("KEY INSIGHT:")
    print("-"*50)
    print(f"""
    Variance ranges from λ₂ = {eigenvalues[1]:.2f} to λ₁ = {eigenvalues[0]:.2f}
    
    - MAXIMUM variance ({eigenvalues[0]:.2f}) along eigenvector v₁
    - MINIMUM variance ({eigenvalues[1]:.2f}) along eigenvector v₂
    - Trait axes ({Sigma[0,0]:.2f}, {Sigma[1,1]:.2f}) are intermediate
    
    The eigenvalues BOUND the variance in all directions!
    """)


def visualize_variance_by_direction():
    """
    Create a figure showing how variance varies with direction.
    """
    print("\n  Generating variance-by-direction figure...")
    
    Sigma = np.array([[1.0, 0.8],
                      [0.8, 1.0]])
    
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Compute variance for all angles
    angles = np.linspace(0, 2*np.pi, 360)
    variances = []
    
    for theta in angles:
        beta = np.array([np.cos(theta), np.sin(theta)])
        var = beta @ Sigma @ beta
        variances.append(var)
    
    variances = np.array(variances)
    
    # Create figure with two panels
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel 1: Variance vs angle (Cartesian)
    ax = axes[0]
    ax.fill_between(np.degrees(angles), eigenvalues[1], variances, alpha=0.3, color='steelblue')
    ax.plot(np.degrees(angles), variances, '-', color='steelblue', lw=2, label=r'$\sigma^2(\theta)$')
    
    # Mark eigenvalue bounds
    ax.axhline(eigenvalues[0], color='forestgreen', lw=2, ls='--', 
               label=f'$\\lambda_1 = {eigenvalues[0]:.2f}$ (max)')
    ax.axhline(eigenvalues[1], color='darkorange', lw=2, ls='--', 
               label=f'$\\lambda_2 = {eigenvalues[1]:.2f}$ (min)')
    
    # Mark eigenvector directions
    for i, (lam, evec, color) in enumerate(zip(eigenvalues, eigenvectors.T, ['forestgreen', 'darkorange'])):
        angle_deg = np.degrees(np.arctan2(evec[1], evec[0]))
        ax.axvline(angle_deg, color=color, lw=1.5, ls=':', alpha=0.7)
        ax.axvline(angle_deg + 180, color=color, lw=1.5, ls=':', alpha=0.7)
    
    ax.set_xlabel('Direction angle θ (degrees)', fontsize=12)
    ax.set_ylabel(r'Variance $\sigma^2(\theta) = \beta^\top \Sigma \beta$', fontsize=12)
    ax.set_title('(a) Variance as a Function of Direction', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 2)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Polar plot (variance as radius)
    ax = axes[1]
    
    # The "variance ellipse" in polar coordinates
    ax.plot(variances * np.cos(angles), variances * np.sin(angles), 
            '-', color='steelblue', lw=2, label='Variance by direction')
    
    # Mark eigenvalue bounds as circles
    theta_circle = np.linspace(0, 2*np.pi, 100)
    ax.plot(eigenvalues[0] * np.cos(theta_circle), eigenvalues[0] * np.sin(theta_circle),
            '--', color='forestgreen', lw=1.5, label=f'$\\lambda_1 = {eigenvalues[0]:.2f}$')
    ax.plot(eigenvalues[1] * np.cos(theta_circle), eigenvalues[1] * np.sin(theta_circle),
            '--', color='darkorange', lw=1.5, label=f'$\\lambda_2 = {eigenvalues[1]:.2f}$')
    
    # Draw eigenvectors
    scale = 2
    ax.annotate('', xy=eigenvectors[:, 0]*scale, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='forestgreen', lw=2))
    ax.annotate('', xy=eigenvectors[:, 1]*scale, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='darkorange', lw=2))
    ax.text(eigenvectors[0, 0]*scale + 0.1, eigenvectors[1, 0]*scale + 0.1,
            r'$\mathbf{v}_1$', fontsize=11, color='forestgreen', fontweight='bold')
    ax.text(eigenvectors[0, 1]*scale + 0.15, eigenvectors[1, 1]*scale - 0.1,
            r'$\mathbf{v}_2$', fontsize=11, color='darkorange', fontweight='bold')
    
    ax.set_xlabel('Trait 1 direction', fontsize=12)
    ax.set_ylabel('Trait 2 direction', fontsize=12)
    ax.set_title('(b) Variance "Ellipse" (Polar View)\n' + 
                 r'Distance from origin = $\sigma^2(\theta)$', fontsize=12, fontweight='bold')
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', lw=0.5)
    ax.axvline(x=0, color='k', lw=0.5)
    
    plt.suptitle('Variance in Any Direction is Bounded by Eigenvalues',
                 fontsize=13, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig('fig_07_08_variance_by_direction.png', dpi=300, bbox_inches='tight')
    plt.savefig('fig_07_08_variance_by_direction.pdf', bbox_inches='tight')
    print("  Saved: fig_07_08_variance_by_direction.png/pdf")
    plt.close()


# =============================================================================
# SECTION 7.9: PRINCIPAL COMPONENT ANALYSIS (PCA) IN ONE PARAGRAPH
# =============================================================================
"""
PCA in One Paragraph:
---------------------
Project the data onto the eigenvectors of the covariance matrix.

    - PC1 = projection onto v₁ → captures λ₁/(λ₁+λ₂+...+λₚ) of variance
    - PC2 = projection onto v₂ → captures next largest fraction
    - etc.

If λ₁ >> other eigenvalues, most variation is in one direction and the data 
are approximately one-dimensional.

That's it. PCA IS eigendecomposition of the covariance matrix.
"""

def demonstrate_pca():
    """
    Show PCA as eigendecomposition of the covariance matrix.
    """
    print("\n" + "="*70)
    print("SECTION 7.9: Principal Component Analysis (PCA) in One Paragraph")
    print("="*70)
    
    # Generate correlated data
    np.random.seed(42)
    n = 200
    Sigma = np.array([[3, 2],
                      [2, 2]])
    data = np.random.multivariate_normal([0, 0], Sigma, n)
    
    # Compute sample covariance
    sample_cov = np.cov(data.T)
    
    print("\nSample covariance matrix:")
    print(np.round(sample_cov, 3))
    
    # Eigendecomposition = PCA
    eigenvalues, eigenvectors = np.linalg.eigh(sample_cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    print(f"\nEigenvalues (variances of PCs):")
    print(f"  λ₁ = {eigenvalues[0]:.3f}")
    print(f"  λ₂ = {eigenvalues[1]:.3f}")
    
    # Proportion of variance explained
    total_var = np.sum(eigenvalues)
    prop_var = eigenvalues / total_var
    
    print(f"\nProportion of variance explained:")
    print(f"  PC1: {prop_var[0]*100:.1f}%")
    print(f"  PC2: {prop_var[1]*100:.1f}%")
    
    print(f"\nEigenvectors (PC loadings):")
    print(f"  v₁ (PC1 direction) = {eigenvectors[:, 0]}")
    print(f"  v₂ (PC2 direction) = {eigenvectors[:, 1]}")
    
    # Project data onto PCs
    pc_scores = data @ eigenvectors
    
    # Verify variance of projections
    pc1_var = np.var(pc_scores[:, 0], ddof=1)
    pc2_var = np.var(pc_scores[:, 1], ddof=1)
    
    print(f"\nVariance of PC scores (should match eigenvalues):")
    print(f"  Var(PC1) = {pc1_var:.3f} (λ₁ = {eigenvalues[0]:.3f})")
    print(f"  Var(PC2) = {pc2_var:.3f} (λ₂ = {eigenvalues[1]:.3f})")
    
    print("\n" + "="*70)
    print("THAT'S IT! PCA = eigendecomposition of the covariance matrix.")
    print("           Eigenvectors are directions; eigenvalues are variances.")
    print("="*70)


# =============================================================================
# SECTION 7.11: A BIOLOGICAL EXAMPLE—THE G MATRIX
# =============================================================================
"""
The G Matrix:
-------------
The additive genetic covariance matrix G describes heritable variation.

    - Diagonal elements: Genetic variances of individual traits
    - Off-diagonal elements: Genetic covariances between traits

Its eigendecomposition reveals:
    
    - g_max (first eigenvector): Direction of MAXIMUM genetic variance
                                 = "Line of least evolutionary resistance"
    
    - Eigenvalue ratios: How "eccentric" is the genetic ellipsoid?
                        Large ratio → evolution strongly channeled
    
    - Effective dimensionality: How many independent directions of variation?
                               Low → strong genetic constraint
"""

def demonstrate_g_matrix():
    """
    Analyze a G matrix and interpret its eigenstructure biologically.
    """
    print("\n" + "="*70)
    print("SECTION 7.11: A Biological Example—The G Matrix")
    print("="*70)
    
    # A realistic G matrix for two traits (e.g., body size and shape)
    G = np.array([[0.45, 0.30],
                  [0.30, 0.35]])
    
    print("\nGenetic covariance matrix G:")
    print(G)
    
    # Compute genetic correlation
    r_G = G[0, 1] / np.sqrt(G[0, 0] * G[1, 1])
    print(f"\nGenetic correlation: r_G = {r_G:.3f}")
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(G)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    g_max = eigenvectors[:, 0]
    g_min = eigenvectors[:, 1]
    
    print(f"\nEigenvalues (genetic variances along principal axes):")
    print(f"  λ₁ = {eigenvalues[0]:.4f} (along g_max)")
    print(f"  λ₂ = {eigenvalues[1]:.4f} (along g_min)")
    print(f"\nRatio λ₁/λ₂ = {eigenvalues[0]/eigenvalues[1]:.1f}")
    
    print(f"\nEigenvectors:")
    print(f"  g_max = {g_max} (direction of max genetic variance)")
    print(f"  g_min = {g_min} (direction of min genetic variance)")
    
    # Interpret g_max direction
    angle = np.degrees(np.arctan2(g_max[1], g_max[0]))
    print(f"\ng_max angle: {angle:.1f}° from trait 1 axis")
    
    # Effective dimensionality
    n_eff = np.sum(eigenvalues)**2 / np.sum(eigenvalues**2)
    print(f"\nEffective dimensionality: n_eff = {n_eff:.2f}")
    print(f"  (Range: 1 to {len(eigenvalues)}; lower = more constrained)")
    
    print("\n" + "-"*50)
    print("BIOLOGICAL INTERPRETATION:")
    print("-"*50)
    print(f"""
    The G matrix is a genetic ellipse:
    
    1. g_max points at {angle:.0f}° → Both traits tend to increase together genetically
       Selection along this axis produces strong response
       
    2. Ratio λ₁/λ₂ = {eigenvalues[0]/eigenvalues[1]:.1f} → Ellipse is eccentric
       Genetic variance concentrated in one direction
       
    3. n_eff = {n_eff:.2f} → Moderate constraint
       Not all directions are equally evolvable
       
    4. Genetic correlation r_G = {r_G:.2f} → Traits constrained to evolve together
       Selecting for one trait changes the other
    """)
    
    return G, eigenvalues, eigenvectors


# =============================================================================
# SECTION 7.12: POSITIVE DEFINITENESS AND WHAT EIGENVALUES TELL US
# =============================================================================
"""
Positive Definite Matrices:
---------------------------
A matrix is POSITIVE DEFINITE if ALL eigenvalues are strictly positive.

For covariance matrices:
    - Positive definite → Every direction has positive variance
                       → Matrix can be inverted
                       → Ellipse is proper (not degenerate)
    
    - Positive semi-definite → Some eigenvalues may be 0
                            → Some direction has zero variance
                            → Data lie in lower-dimensional subspace
                            → Matrix is SINGULAR (cannot invert)

In practice:
    Estimated covariance matrices may have small or negative eigenvalues
    due to sampling error. This causes numerical problems!
"""

def demonstrate_positive_definiteness():
    """
    Show the importance of positive definiteness and how to check it.
    """
    print("\n" + "="*70)
    print("SECTION 7.12: Positive Definiteness and What Eigenvalues Tell Us")
    print("="*70)
    
    # Three examples
    matrices = {
        'Positive definite': np.array([[4, 2], [2, 5]]),
        'Positive semi-definite': np.array([[4, 2], [2, 1]]),
        'Indefinite': np.array([[4, 5], [5, 4]]),
    }
    
    for name, M in matrices.items():
        print(f"\n{'-'*50}")
        print(f"{name} matrix:")
        print(M)
        
        eigenvalues = np.linalg.eigvalsh(M)
        det = np.linalg.det(M)
        
        print(f"\n  Eigenvalues: {eigenvalues}")
        print(f"  Determinant: {det:.2f}")
        print(f"  All λ > 0? {all(eigenvalues > 0)}")
        
        if all(eigenvalues > 0):
            print("  ✓ Can be inverted, Mahalanobis distance defined")
        elif all(eigenvalues >= 0):
            print("  ⚠ Singular! Data in lower-dimensional subspace")
        else:
            print("  ✗ Not a valid covariance matrix (has negative variance)")
    
    print("\n" + "-"*50)
    print("PRACTICAL IMPLICATION:")
    print("-"*50)
    print("""
    When estimating G or P from data:
    
    1. Small eigenvalues may become negative due to sampling error
    2. This makes the matrix non-invertible
    3. Regularization or shrinkage may be needed
    4. Small eigenvalues are always poorly estimated
    
    Rule of thumb: Don't trust eigenvalues < 0.1 × (largest eigenvalue)
    """)


# =============================================================================
# SECTION 7.13: THE CONDITION NUMBER
# =============================================================================
"""
The Condition Number:
---------------------
The condition number measures how "ill-behaved" a matrix is:

    κ = λ_max / λ_min

Large condition number indicates:
    - Ellipse is highly elongated
    - Matrix is nearly singular
    - Numerical operations may be unstable
    - Small estimation errors cause large downstream errors

In evolutionary terms:
    Large κ → Evolution strongly channeled along g_max
            → Response perpendicular to g_max nearly impossible
"""

def demonstrate_condition_number():
    """
    Show how condition number relates to matrix behavior.
    """
    print("\n" + "="*70)
    print("SECTION 7.13: The Condition Number—How 'Ill-Behaved' Is the Matrix?")
    print("="*70)
    
    # Create matrices with different condition numbers
    matrices = {
        'Spherical (κ=1)': np.array([[2, 0], [0, 2]]),
        'Moderate (κ=5)': np.array([[5, 0], [0, 1]]),
        'Eccentric (κ=100)': np.array([[100, 0], [0, 1]]),
        'Correlated': np.array([[1.0, 0.95], [0.95, 1.0]]),
    }
    
    for name, M in matrices.items():
        eigenvalues = np.linalg.eigvalsh(M)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        condition = eigenvalues[0] / eigenvalues[1]
        
        print(f"\n{name}:")
        print(f"  Eigenvalues: λ₁ = {eigenvalues[0]:.4f}, λ₂ = {eigenvalues[1]:.4f}")
        print(f"  Condition number κ = λ_max/λ_min = {condition:.1f}")
        
        # Interpretation
        if condition < 10:
            print("  → Well-conditioned: stable numerical operations")
        elif condition < 100:
            print("  → Moderate conditioning: some numerical care needed")
        else:
            print("  → Ill-conditioned: numerical instability likely!")
    
    print("\n" + "-"*50)
    print("BIOLOGICAL INTERPRETATION:")
    print("-"*50)
    print("""
    For a G matrix, high condition number means:
    
    1. Evolution is STRONGLY CHANNELED along g_max
    2. Response perpendicular to g_max is nearly BLOCKED
    3. The population has REDUCED EFFECTIVE DIMENSIONALITY
    4. Estimation of the matrix requires LARGE SAMPLE SIZES
    
    Many empirical G matrices have condition numbers > 10,
    indicating substantial evolutionary constraint.
    """)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("CHAPTER 7: DIAGONALISATION AND NATURAL AXES")
    print("="*70)
    print("""
    This chapter develops the central mathematical tool of multivariate
    analysis: eigendecomposition. By finding the natural axes of a symmetric
    matrix, we can understand its behavior completely.
    
    For covariance matrices, eigenvectors ARE the ellipse axes, and
    eigenvalues ARE the variances along those axes.
    """)
    
    # Section 7.1: The question
    demonstrate_eigenvector_concept()
    
    # Section 7.2: By-hand example
    eigendecomposition_by_hand()
    
    # Section 7.3: Spectral theorem
    demonstrate_spectral_theorem()
    
    # Section 7.4: Diagonalisation
    demonstrate_diagonalisation()
    visualize_rotate_stretch_rotate()
    
    # Section 7.5: Ellipse geometry
    demonstrate_ellipse_geometry()
    visualize_covariance_ellipse()
    
    # Section 7.6: Why diagonalisation simplifies everything
    demonstrate_diagonalisation_power()
    
    # Section 7.7: Trace and determinant
    demonstrate_trace_and_determinant()
    
    # Section 7.8: Variance in any direction
    demonstrate_variance_in_direction()
    visualize_variance_by_direction()
    
    # Section 7.9: PCA
    demonstrate_pca()
    
    # Section 7.11: G matrix
    demonstrate_g_matrix()
    
    # Section 7.12: Positive definiteness
    demonstrate_positive_definiteness()
    
    # Section 7.13: Condition number
    demonstrate_condition_number()
    
    print("\n" + "="*70)
    print("CHAPTER 7 COMPLETE")
    print("="*70)
    print("""
    KEY TAKEAWAYS:
    
    1. Eigenvectors are directions where a matrix acts as pure stretching
    2. Eigenvalues are the stretching factors
    3. A = VΛV^T means: rotate → stretch → rotate back
    4. Covariance matrices ARE ellipses (eigenvectors = axes, eigenvalues = variances)
    5. Variance in any direction is bounded: λ_min ≤ β^TΣβ ≤ λ_max
    6. PCA = eigendecomposition of the covariance matrix
    7. The G matrix eigenvectors reveal lines of least evolutionary resistance
    8. Condition number measures how eccentric/constrained the matrix is
    
    Next: Chapter 8 uses these tools to compare G and P via whitening.
    """)


# =============================================================================
# EXERCISES
# =============================================================================
"""
EXERCISES FOR CHAPTER 7
=======================

EXERCISE 7.1: Eigendecomposition by Hand
----------------------------------------
For the matrix A = [[2, 1], [1, 2]]:
(a) Find the eigenvalues using the characteristic equation
(b) Find the corresponding eigenvectors
(c) Verify that eigenvectors are orthogonal
(d) Verify A = VΛV^T
(e) Interpret the eigenvectors geometrically

EXERCISE 7.2: Variance Bounds
-----------------------------
For the covariance matrix Σ = [[5, 2], [2, 2]]:
(a) Find eigenvalues and eigenvectors
(b) Compute variance in the direction (1, 0)
(c) Compute variance in the direction (0, 1)
(d) Compute variance in the direction (1, 1)/√2
(e) Verify that variance along eigenvector directions equals eigenvalues

EXERCISE 7.3: PCA Interpretation
--------------------------------
A biologist measures wing length (mm) and wing width (mm) on 100 birds.
The sample covariance matrix is:
    S = [[25, 15], [15, 16]]

(a) Perform PCA (eigendecomposition)
(b) What proportion of variance does PC1 explain?
(c) Interpret PC1 biologically (what does it represent?)
(d) What proportion would PC1 explain if the traits were uncorrelated?

EXERCISE 7.4: G Matrix Analysis
-------------------------------
Consider the genetic covariance matrix:
    G = [[0.8, -0.3], [-0.3, 0.5]]

(a) What is the genetic correlation between traits?
(b) Find g_max (direction of maximum genetic variance)
(c) Calculate the condition number
(d) What does the negative correlation imply for selection response?

EXERCISE 7.5: Condition Number and Constraint
----------------------------------------------
Three populations have G matrices:
    Population A: G = [[1, 0], [0, 1]]     (κ = 1)
    Population B: G = [[1, 0.9], [0.9, 1]] (κ = 19)
    Population C: G = [[1, 0], [0, 0.01]]  (κ = 100)

(a) Calculate the condition number for each (verify given values)
(b) Which population can evolve most freely in any direction?
(c) Which is most constrained?
(d) For population C, what happens if selection targets trait 2?

EXERCISE 7.6: Singular Matrices
-------------------------------
Consider the matrix M = [[4, 2], [2, 1]].
(a) Compute its eigenvalues
(b) Why is this matrix singular?
(c) What does this imply about the data that produced it?
(d) Can you compute Mahalanobis distance using this matrix? Why or why not?
"""
