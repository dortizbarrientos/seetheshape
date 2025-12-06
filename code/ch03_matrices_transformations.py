#!/usr/bin/env python3
"""
================================================================================
CHAPTER 3: Matrices as Machines That Move Vectors
================================================================================
Book: "Seeing the Shape" by Daniel Ortiz-Barrientos

The central insight of this chapter:

    A MATRIX IS NOT JUST A TABLE OF NUMBERS—IT IS A MACHINE THAT TRANSFORMS SPACE.

When you see a matrix, ask: "What does this DO to vectors?"
  - Where do the coordinate axes go?
  - What happens to circles? (They become ellipses!)
  - What directions remain fixed? (These are eigenvectors—Chapter 7)

Covariance matrices, genetic matrices (G), and selection matrices (γ) are all
transformations. Understanding what they DO is the key to multivariate genetics.

Sections covered:
    §3.1 A motivating example: scaling traits differently
    §3.2 Matrix–vector multiplication
    §3.3 What happens to the unit vectors?
    §3.4 Geometric vocabulary: stretch, rotate, shear
    §3.5 Linearity: the defining property
    §3.6-3.7 Composition, identity, and inverse
    §3.8 Symmetric matrices: the special and important case
    §3.9 Preview: the quadratic form
    §3.10 A biological example: the G matrix as a transformation

================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from typing import Tuple, List, Callable

# -----------------------------------------------------------------------------
# SECTION 3.2: Matrix-Vector Multiplication
# -----------------------------------------------------------------------------
#
# When a matrix A acts on a vector v, it produces a new vector w = Av.
# 
# Each entry of w is a DOT PRODUCT: row i of A dotted with v.
#
#   w_i = Σ_j A_ij * v_j
#
# This is a LINEAR COMBINATION of the input components.

def apply_matrix(A: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Apply matrix A to vector v, returning Av.
    
    This is the fundamental operation: the matrix transforms the vector.
    
    Parameters
    ----------
    A : np.ndarray
        A (p × p) matrix representing a linear transformation.
    v : np.ndarray
        A vector of length p.
    
    Returns
    -------
    np.ndarray
        The transformed vector Av.
    
    Example
    -------
    >>> A = np.array([[2, 0], [0, 3]])  # Stretch x by 2, y by 3
    >>> v = np.array([1, 1])
    >>> apply_matrix(A, v)
    array([2, 3])
    """
    return A @ v


# -----------------------------------------------------------------------------
# SECTION 3.3: What Happens to the Unit Vectors?
# -----------------------------------------------------------------------------
#
# THE KEY INSIGHT: The columns of A are the images of the unit vectors!
#
#   A @ e_1 = first column of A
#   A @ e_2 = second column of A
#   ...
#
# If you know where the coordinate axes go, you know EVERYTHING about the
# transformation, because every vector is a combination of the axes.

def visualize_transformation(
    A: np.ndarray,
    title: str = "Matrix Transformation",
    show_unit_circle: bool = True,
    ax: plt.Axes = None
) -> plt.Axes:
    """
    Visualize how a 2×2 matrix transforms the plane.
    
    Shows:
    1. Where the unit vectors e₁ = (1,0) and e₂ = (0,1) go
    2. What happens to the unit circle (it becomes an ellipse!)
    
    This is the most important visualization for understanding matrices.
    
    Parameters
    ----------
    A : np.ndarray
        A (2 × 2) matrix.
    title : str
        Plot title.
    show_unit_circle : bool
        If True, show the unit circle and its image.
    ax : matplotlib Axes, optional
        Axes to plot on.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Unit vectors
    e1 = np.array([1, 0])
    e2 = np.array([0, 1])
    
    # Transformed unit vectors (= columns of A)
    Ae1 = A @ e1  # First column of A
    Ae2 = A @ e2  # Second column of A
    
    # Draw original unit vectors (dashed)
    ax.annotate('', xy=e1, xytext=[0, 0],
                arrowprops=dict(arrowstyle='->', color='blue', lw=2, 
                               linestyle='--', alpha=0.5))
    ax.annotate('', xy=e2, xytext=[0, 0],
                arrowprops=dict(arrowstyle='->', color='red', lw=2,
                               linestyle='--', alpha=0.5))
    
    # Draw transformed unit vectors (solid)
    ax.annotate('', xy=Ae1, xytext=[0, 0],
                arrowprops=dict(arrowstyle='->', color='blue', lw=2.5))
    ax.annotate('', xy=Ae2, xytext=[0, 0],
                arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
    
    # Labels
    ax.text(e1[0] + 0.1, e1[1] - 0.2, '$\\mathbf{e}_1$', fontsize=12, color='blue', alpha=0.5)
    ax.text(e2[0] - 0.3, e2[1] + 0.1, '$\\mathbf{e}_2$', fontsize=12, color='red', alpha=0.5)
    ax.text(Ae1[0] + 0.1, Ae1[1] - 0.2, '$A\\mathbf{e}_1$', fontsize=12, color='blue')
    ax.text(Ae2[0] - 0.4, Ae2[1] + 0.1, '$A\\mathbf{e}_2$', fontsize=12, color='red')
    
    # Unit circle and its image
    if show_unit_circle:
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = np.cos(theta)
        circle_y = np.sin(theta)
        
        # Original circle
        ax.plot(circle_x, circle_y, 'k--', alpha=0.4, label='Unit circle')
        
        # Transformed circle (becomes ellipse for most matrices)
        circle_points = np.column_stack([circle_x, circle_y])
        transformed = (A @ circle_points.T).T
        ax.plot(transformed[:, 0], transformed[:, 1], 'k-', lw=2, 
                label='Image of unit circle')
    
    # Formatting
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    
    # Set axis limits
    all_points = np.array([[0, 0], e1, e2, Ae1, Ae2])
    margin = 0.5
    ax.set_xlim(all_points[:, 0].min() - margin, all_points[:, 0].max() + margin)
    ax.set_ylim(all_points[:, 1].min() - margin, all_points[:, 1].max() + margin)
    
    return ax


# -----------------------------------------------------------------------------
# SECTION 3.4: Geometric Vocabulary — Stretch, Rotate, Shear
# -----------------------------------------------------------------------------
#
# Different matrices produce different geometric effects:
#
# 1. SCALING (diagonal matrix): Stretch each axis independently
# 2. ROTATION: Spin the plane, preserve lengths and angles
# 3. SHEAR: Slide in one direction proportional to the other coordinate
# 4. REFLECTION: Flip across an axis

def scaling_matrix(sx: float, sy: float) -> np.ndarray:
    """
    Create a scaling (stretching) matrix.
    
    Stretches x by factor sx, y by factor sy.
    
    Properties:
    - Diagonal matrix (off-diagonals are zero)
    - Circle → Ellipse aligned with axes
    - If sx = sy, this is uniform scaling (homothety)
    
    Example: Trait 1 in mm, trait 2 in cm → convert both to mm
    """
    return np.array([[sx, 0],
                     [0, sy]])


def rotation_matrix(theta: float) -> np.ndarray:
    """
    Create a rotation matrix for angle theta (in radians).
    
    Rotates counterclockwise by theta.
    
    Properties:
    - Preserves lengths: ||Rv|| = ||v||
    - Preserves angles: angle(Ru, Rv) = angle(u, v)
    - Circle → Circle (shape preserved)
    - det(R) = 1, R^T R = I (orthogonal matrix)
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s, c]])


def shear_matrix(k: float, direction: str = 'horizontal') -> np.ndarray:
    """
    Create a shear matrix.
    
    Shear slides the plane: points are displaced horizontally (or vertically)
    in proportion to their vertical (or horizontal) coordinate.
    
    Properties:
    - Square → Parallelogram
    - Does NOT preserve lengths
    - det = 1 (preserves area)
    """
    if direction == 'horizontal':
        return np.array([[1, k],
                         [0, 1]])
    else:
        return np.array([[1, 0],
                         [k, 1]])


def reflection_matrix(axis: str = 'y') -> np.ndarray:
    """
    Create a reflection matrix.
    
    Reflects across the x-axis or y-axis.
    
    Properties:
    - det = -1 (reverses orientation)
    - Two reflections = identity
    """
    if axis == 'y':
        return np.array([[-1, 0],
                         [0, 1]])
    else:
        return np.array([[1, 0],
                         [0, -1]])


def demonstrate_transformations():
    """
    Visualize the four basic types of linear transformations.
    """
    print("\n" + "=" * 60)
    print("THE FOUR BASIC TRANSFORMATIONS")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # 1. Scaling
    A_scale = scaling_matrix(2, 0.5)
    visualize_transformation(A_scale, "(a) Scaling: stretch x by 2, y by 0.5", ax=axes[0, 0])
    print(f"\nScaling matrix:\n{A_scale}")
    print("  → Diagonal entries are the stretch factors")
    
    # 2. Rotation
    theta = np.pi / 4  # 45 degrees
    A_rot = rotation_matrix(theta)
    visualize_transformation(A_rot, "(b) Rotation: 45° counterclockwise", ax=axes[0, 1])
    print(f"\nRotation matrix (45°):\n{A_rot.round(4)}")
    print("  → Orthogonal: preserves lengths and angles")
    
    # 3. Shear
    A_shear = shear_matrix(0.5, 'horizontal')
    visualize_transformation(A_shear, "(c) Shear: horizontal, k = 0.5", ax=axes[1, 0])
    print(f"\nShear matrix:\n{A_shear}")
    print("  → Parallelograms, not ellipses")
    
    # 4. Symmetric matrix (covariance-like)
    A_sym = np.array([[2, 1],
                      [1, 1.5]])
    visualize_transformation(A_sym, "(d) Symmetric: covariance-like", ax=axes[1, 1])
    print(f"\nSymmetric matrix:\n{A_sym}")
    print("  → Ellipse aligned with eigenvectors!")
    
    plt.tight_layout()
    plt.savefig('ch03_four_transformations.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Saved: ch03_four_transformations.png")


# -----------------------------------------------------------------------------
# SECTION 3.5: Linearity — The Defining Property
# -----------------------------------------------------------------------------
#
# Matrix transformations are LINEAR:
#   1. A(u + v) = Au + Av      (additivity)
#   2. A(cv) = c(Av)           (homogeneity)
#
# This is why they're so useful: complex transformations decompose into
# simple operations on basis vectors.

def verify_linearity(A: np.ndarray, u: np.ndarray, v: np.ndarray, c: float):
    """
    Verify the two linearity properties of matrix transformations.
    """
    print("\n--- Verifying Linearity ---")
    
    # Property 1: Additivity
    lhs1 = A @ (u + v)
    rhs1 = (A @ u) + (A @ v)
    print(f"\nAdditivity: A(u + v) = Au + Av")
    print(f"  A(u + v) = {lhs1}")
    print(f"  Au + Av  = {rhs1}")
    print(f"  Equal? {np.allclose(lhs1, rhs1)}")
    
    # Property 2: Homogeneity
    lhs2 = A @ (c * u)
    rhs2 = c * (A @ u)
    print(f"\nHomogeneity: A(cu) = c(Au)")
    print(f"  A({c}u) = {lhs2}")
    print(f"  {c}(Au) = {rhs2}")
    print(f"  Equal? {np.allclose(lhs2, rhs2)}")


# -----------------------------------------------------------------------------
# SECTION 3.8: Symmetric Matrices — The Special and Important Case
# -----------------------------------------------------------------------------
#
# A matrix is SYMMETRIC if A = A^T (i.e., a_ij = a_ji).
#
# COVARIANCE MATRICES ARE ALWAYS SYMMETRIC (by construction).
# So are genetic matrices (G) and quadratic selection matrices (γ).
#
# Symmetric matrices have remarkable properties:
#   1. All eigenvalues are REAL
#   2. Eigenvectors are ORTHOGONAL
#   3. They can be DIAGONALIZED by a rotation (no shear!)
#
# → SYMMETRIC MATRICES DESCRIBE ELLIPSES, NOT PARALLELOGRAMS.

def is_symmetric(A: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Check if a matrix is symmetric: A = A^T.
    
    Covariance matrices are symmetric because Cov(X, Y) = Cov(Y, X).
    """
    return np.allclose(A, A.T, atol=tol)


def make_symmetric(A: np.ndarray) -> np.ndarray:
    """
    Symmetrize a matrix: (A + A^T) / 2.
    
    Useful for correcting numerical errors in covariance estimates.
    """
    return (A + A.T) / 2


# -----------------------------------------------------------------------------
# SECTION 3.9: The Quadratic Form
# -----------------------------------------------------------------------------
#
# Given a symmetric matrix A and vector v, the QUADRATIC FORM is:
#
#   q(v) = v^T A v
#
# This is a SCALAR that measures how v interacts with the shape defined by A.
#
# Applications:
#   - Variance of a trait combination: a^T Σ a
#   - Mahalanobis distance: (z - μ)^T Σ^{-1} (z - μ)
#   - Quadratic selection: z^T γ z

def quadratic_form(A: np.ndarray, v: np.ndarray) -> float:
    """
    Compute the quadratic form v^T A v.
    
    For a 2×2 symmetric matrix A = [[a, b], [b, c]] and v = [v1, v2]:
    
        v^T A v = a*v1² + 2b*v1*v2 + c*v2²
    
    This is a weighted sum of squared components and cross-products.
    
    Parameters
    ----------
    A : np.ndarray
        A symmetric matrix (typically a covariance matrix).
    v : np.ndarray
        A vector.
    
    Returns
    -------
    float
        The value v^T A v.
    
    Interpretation
    --------------
    If A is a covariance matrix Σ and v is a direction (unit vector),
    then v^T Σ v is the variance in that direction!
    """
    return float(v @ A @ v)


def visualize_quadratic_form(A: np.ndarray, ax: plt.Axes = None) -> plt.Axes:
    """
    Visualize the quadratic form as a surface over the plane.
    
    The level curves (where v^T A v = constant) are ellipses!
    These are the same ellipses that the matrix defines.
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Create grid
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    
    # Compute quadratic form at each point
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            v = np.array([X[i, j], Y[i, j]])
            Z[i, j] = quadratic_form(A, v)
    
    # Plot surface
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    
    # Add contour projection on bottom
    ax.contour(X, Y, Z, zdir='z', offset=0, cmap='viridis', levels=10)
    
    ax.set_xlabel('$v_1$')
    ax.set_ylabel('$v_2$')
    ax.set_zlabel('$v^T A v$')
    ax.set_title('Quadratic Form $\\mathbf{v}^T A \\mathbf{v}$', fontsize=14, fontweight='bold')
    
    return ax


# -----------------------------------------------------------------------------
# SECTION 3.10: The G Matrix as a Transformation
# -----------------------------------------------------------------------------
#
# THE BREEDER'S EQUATION: Δz̄ = G β
#
# G is not just storing numbers—it's TRANSFORMING the selection gradient β
# into the evolutionary response Δz̄.
#
# If G is "spherical" (identity matrix): response = selection direction
# If G has genetic correlations: response is DEFLECTED toward g_max

def breeders_equation(G: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Apply the multivariate breeder's equation: Δz̄ = Gβ
    
    This is the fundamental equation of quantitative genetics.
    
    Parameters
    ----------
    G : np.ndarray
        The additive genetic covariance matrix.
    beta : np.ndarray
        The selection gradient (direction of steepest fitness increase).
    
    Returns
    -------
    np.ndarray
        The expected response to selection Δz̄.
    
    Biological Interpretation
    -------------------------
    - β tells you where selection WANTS to go
    - G tells you what evolution CAN do
    - Δz̄ is the COMPROMISE between desire and constraint
    
    If G has strong genetic correlations, Δz̄ will be deflected toward
    the direction of maximum genetic variance (g_max).
    """
    return G @ beta


def demonstrate_breeders_equation():
    """
    Demonstrate how the G matrix deflects selection response.
    """
    print("\n" + "=" * 60)
    print("THE MULTIVARIATE BREEDER'S EQUATION: Δz̄ = Gβ")
    print("=" * 60)
    
    # Case 1: No genetic correlation
    G_uncorrelated = np.array([[1.0, 0.0],
                               [0.0, 1.0]])
    
    # Case 2: Positive genetic correlation
    G_correlated = np.array([[1.0, 0.8],
                             [0.8, 1.0]])
    
    # Selection gradient: favor trait 2 only
    beta = np.array([0.0, 1.0])
    
    # Compute responses
    response_uncorr = breeders_equation(G_uncorrelated, beta)
    response_corr = breeders_equation(G_correlated, beta)
    
    print(f"\nSelection gradient β: {beta}")
    print("  → Selection favors ONLY trait 2")
    
    print(f"\nCase 1: G = I (no correlation)")
    print(f"  G = \n{G_uncorrelated}")
    print(f"  Response Δz̄ = {response_uncorr}")
    print("  → Response tracks selection perfectly")
    
    print(f"\nCase 2: G with positive correlation (r = 0.8)")
    print(f"  G = \n{G_correlated}")
    print(f"  Response Δz̄ = {response_corr}")
    print("  → Response INCLUDES trait 1, even though selection didn't favor it!")
    print("  → The genetic correlation 'drags' trait 1 along")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax, G, resp, title in [
        (axes[0], G_uncorrelated, response_uncorr, "(a) No correlation: G = I"),
        (axes[1], G_correlated, response_corr, "(b) Correlated: r = 0.8")
    ]:
        # Draw G ellipse
        eigenvalues, eigenvectors = np.linalg.eigh(G)
        angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
        ellipse = Ellipse((0, 0), 2*np.sqrt(eigenvalues[1]), 2*np.sqrt(eigenvalues[0]),
                         angle=angle, fill=False, color='green', linewidth=2,
                         label='G ellipse')
        ax.add_patch(ellipse)
        
        # Draw selection gradient
        ax.annotate('', xy=beta, xytext=[0, 0],
                    arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
        ax.text(beta[0] + 0.1, beta[1] - 0.15, '$\\beta$', fontsize=14, color='red')
        
        # Draw response
        ax.annotate('', xy=resp, xytext=[0, 0],
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2.5))
        ax.text(resp[0] + 0.1, resp[1] - 0.15, '$\\Delta\\bar{z}$', fontsize=14, color='blue')
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_aspect('equal')
        ax.axhline(0, color='gray', lw=0.5)
        ax.axvline(0, color='gray', lw=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Trait 1')
        ax.set_ylabel('Trait 2')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig('ch03_breeders_equation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Saved: ch03_breeders_equation.png")
    print("\nKey insight: Genetic correlations DEFLECT evolution from the")
    print("direction of selection toward the direction of maximum genetic variance.")


# -----------------------------------------------------------------------------
# MAIN DEMONSTRATION
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CHAPTER 3: Matrices as Machines That Move Vectors")
    print("Seeing the Shape — Code Companion")
    print("=" * 70)
    
    # Section 3.2-3.3: Matrix-vector multiplication
    print("\n" + "-" * 50)
    print("Section 3.2-3.3: Matrix as transformation")
    print("-" * 50)
    
    A = np.array([[2, 1],
                  [0, 1.5]])
    v = np.array([1, 1])
    
    print(f"Matrix A:\n{A}")
    print(f"\nVector v: {v}")
    print(f"Transformed Av: {A @ v}")
    print(f"\nColumns of A are images of unit vectors:")
    print(f"  A @ e₁ = {A @ np.array([1, 0])} (first column)")
    print(f"  A @ e₂ = {A @ np.array([0, 1])} (second column)")
    
    # Section 3.4: The four transformations
    print("\n" + "-" * 50)
    print("Section 3.4: Geometric transformations")
    print("-" * 50)
    demonstrate_transformations()
    
    # Section 3.5: Linearity
    print("\n" + "-" * 50)
    print("Section 3.5: Linearity")
    print("-" * 50)
    u = np.array([1, 2])
    v = np.array([3, -1])
    c = 2.5
    verify_linearity(A, u, v, c)
    
    # Section 3.8: Symmetric matrices
    print("\n" + "-" * 50)
    print("Section 3.8: Symmetric matrices")
    print("-" * 50)
    
    Sigma = np.array([[1.0, 0.6],
                      [0.6, 0.8]])
    print(f"\nCovariance matrix Σ:\n{Sigma}")
    print(f"Is symmetric? {is_symmetric(Sigma)}")
    print("\n→ Symmetric matrices describe ELLIPSES")
    print("→ Their eigenvectors are perpendicular")
    print("→ All covariance matrices are symmetric (by definition)")
    
    # Section 3.9: Quadratic form
    print("\n" + "-" * 50)
    print("Section 3.9: The quadratic form")
    print("-" * 50)
    
    direction = np.array([1, 0])  # Pure trait 1
    variance_in_direction = quadratic_form(Sigma, direction)
    print(f"\nDirection: {direction} (pure trait 1)")
    print(f"Variance in this direction: {variance_in_direction}")
    print(f"This equals Σ₁₁ = {Sigma[0, 0]}")
    
    direction2 = np.array([1, 1]) / np.sqrt(2)  # Diagonal
    variance2 = quadratic_form(Sigma, direction2)
    print(f"\nDirection: [1,1]/√2 (both traits equally)")
    print(f"Variance in this direction: {variance2:.4f}")
    
    # Section 3.10: Breeder's equation
    print("\n" + "-" * 50)
    print("Section 3.10: The breeder's equation")
    print("-" * 50)
    demonstrate_breeders_equation()
    
    print("\n" + "=" * 70)
    print("END OF CHAPTER 3")
    print("=" * 70)
    print("""
Key Takeaways:
  1. A matrix is a TRANSFORMATION that moves vectors
  2. The columns of A tell you where the unit vectors go
  3. Circles → Ellipses under general linear transformations
  4. SYMMETRIC matrices (like covariance) have perpendicular eigenvectors
  5. The quadratic form v^T A v gives variance in direction v
  6. The breeder's equation Δz̄ = Gβ is a matrix transformation!
  
Next: Chapter 4 connects distance to variance via Pythagoras.
""")


# =============================================================================
# EXERCISES
# =============================================================================
"""
EXERCISE 3.1: Transformation Identification
-------------------------------------------
For each matrix, describe what it does geometrically:
(a) [[3, 0], [0, 3]]
(b) [[0, -1], [1, 0]]
(c) [[1, 0.5], [0, 1]]
(d) [[0.6, 0.8], [0.8, -0.6]]

EXERCISE 3.2: The G Matrix as Transformation
--------------------------------------------
Given G = [[1.0, 0.6], [0.6, 0.5]] and selection gradient β = [0.5, 0.5]:
(a) Compute the response Δz̄ = Gβ
(b) What angle does β make with the x-axis?
(c) What angle does Δz̄ make? Is it deflected?
(d) Find the eigenvectors of G. Is the response deflected toward g_max?

EXERCISE 3.3: Quadratic Form and Variance
-----------------------------------------
Let Σ = [[4, 2], [2, 3]] be a covariance matrix.
(a) Compute the variance in direction (1, 0) — this should equal Σ₁₁
(b) Compute the variance in direction (0, 1) — this should equal Σ₂₂
(c) Compute the variance in direction (1, 1)/√2 — is it between (a) and (b)?
(d) What direction has the MAXIMUM variance? (Hint: eigenvector of Σ)

EXERCISE 3.4: Composition of Transformations
--------------------------------------------
Let R = rotation by 45° and S = scaling by (2, 1).
(a) Compute RS (rotate then scale). What does it do to the unit circle?
(b) Compute SR (scale then rotate). Is it the same as RS?
(c) Why does order matter?
"""
