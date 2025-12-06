#!/usr/bin/env python3
"""
================================================================================
CHAPTER 2: Vectors, Coordinates, and Angles
================================================================================
Book: "Seeing the Shape: A Geometric Introduction to Multivariate Quantitative 
       Genetics" by Daniel Ortiz-Barrientos

This chapter makes the algebra of arrows precise. We introduce:

  - COLUMN VECTORS: the standard notation for phenotypes and changes
  - THE DOT PRODUCT: the key operation that measures "how much two vectors 
    point in the same direction"
  - ANGLES AND ORTHOGONALITY: when vectors are perpendicular
  - PROJECTIONS: the "shadow" of one vector onto another

The Central Tool:
    
    The DOT PRODUCT  v · w = v₁w₁ + v₂w₂ + ... + vₚwₚ = v^T w
    
    This single operation unlocks:
      • Lengths:     ||v|| = √(v · v)
      • Angles:      cos θ = (v · w) / (||v|| ||w||)
      • Projections: component of v along w
      • Variance:    v^T Σ v  (the quadratic form—Chapter 3)

Why This Matters for Biology:
    
    Selection gradients, breeding values, and responses to selection are all
    vectors. The dot product lets us ask:
      • How strong is selection along a particular direction?
      • How much does the response align with the selection gradient?
      • What is the genetic variance in the direction of selection?

Sections covered:
    §2.1 Column vectors and coordinates
    §2.2 Unit vectors and decomposing changes
    §2.3 Lengths written in vector notation
    §2.4 The dot product and angles
    §2.5 Projections onto a direction
    §2.6 Distances between phenotypes

================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

# -----------------------------------------------------------------------------
# SECTION 2.1: Column Vectors and Coordinates
# -----------------------------------------------------------------------------
#
# A phenotype with p traits is written as a COLUMN VECTOR:
#
#       z = [ z₁ ]
#           [ z₂ ]
#           [ ⋮  ]
#           [ zₚ ]
#
# Each entry zᵢ is the value of trait i.
#
# Convention:
#   - Bold lowercase (z, x, β) for vectors
#   - Plain lowercase (z₁, x₂) for scalar components
#   - Bold uppercase (A, G, P) for matrices
#
# In NumPy, we typically use 1D arrays for vectors. When matrix operations
# require explicit column shape, we use reshape or np.newaxis.

def create_phenotype_vector(trait_values: List[float]) -> np.ndarray:
    """
    Create a phenotype vector from a list of trait measurements.
    
    This is conceptually simple but foundational: a phenotype is not just
    a list of numbers—it is a LOCATION in trait space, and we will treat
    it with all the tools of linear algebra.
    
    Parameters
    ----------
    trait_values : list of float
        Measurements for traits 1, 2, ..., p.
    
    Returns
    -------
    np.ndarray
        A 1D array representing the phenotype vector.
    
    Example
    -------
    >>> z = create_phenotype_vector([4.2, 3.1, 2.8])
    >>> print(z)
    [4.2 3.1 2.8]
    >>> print(f"This individual has {len(z)} traits")
    This individual has 3 traits
    """
    return np.array(trait_values, dtype=float)


def phenotype_difference(z_i: np.ndarray, z_j: np.ndarray) -> np.ndarray:
    """
    Compute the difference vector between two phenotypes.
    
    The difference Δz = z_j - z_i is the vector that takes you from
    individual i to individual j in trait space.
    
    This is the algebraic counterpart to the "arrow" from Chapter 1.
    
    Parameters
    ----------
    z_i, z_j : np.ndarray
        Phenotype vectors of two individuals.
    
    Returns
    -------
    np.ndarray
        The difference vector z_j - z_i.
    
    Biological Example
    ------------------
    If z_i = [10, 5] (body size, wing length) and z_j = [12, 6], then
    Δz = [2, 1] means: "To go from individual i to j, add 2 units to
    body size and 1 unit to wing length."
    """
    return z_j - z_i


# -----------------------------------------------------------------------------
# SECTION 2.2: Unit Vectors and Decomposing Changes
# -----------------------------------------------------------------------------
#
# In p dimensions, we have p special "unit vectors" that point along each axis:
#
#   e₁ = [1, 0, 0, ..., 0]^T   (one step along trait 1)
#   e₂ = [0, 1, 0, ..., 0]^T   (one step along trait 2)
#   ...
#   eₚ = [0, 0, 0, ..., 1]^T   (one step along trait p)
#
# ANY vector can be written as a combination of these unit vectors:
#
#   v = v₁e₁ + v₂e₂ + ... + vₚeₚ
#
# This is what the components v₁, v₂, ..., vₚ MEAN: they are the "amounts"
# of each unit vector needed to build v.

def unit_vector(i: int, p: int) -> np.ndarray:
    """
    Create the i-th standard unit vector in p dimensions.
    
    The unit vector eᵢ has 1 in position i and 0 elsewhere.
    It represents "one step along trait i, no change in other traits."
    
    Parameters
    ----------
    i : int
        Which axis (0-indexed: 0 for first trait, 1 for second, etc.)
    p : int
        Total number of dimensions (traits).
    
    Returns
    -------
    np.ndarray
        The unit vector eᵢ.
    
    Example
    -------
    >>> e1 = unit_vector(0, 3)  # First axis in 3D
    >>> print(e1)
    [1. 0. 0.]
    """
    e = np.zeros(p)
    e[i] = 1.0
    return e


def decompose_into_unit_vectors(v: np.ndarray) -> str:
    """
    Express a vector as a linear combination of unit vectors.
    
    This is pedagogical: it shows that the components of v are just
    the coefficients when we write v in terms of the standard basis.
    
    Parameters
    ----------
    v : np.ndarray
        Any vector.
    
    Returns
    -------
    str
        A string representation of v as a sum of unit vectors.
    
    Example
    -------
    >>> v = np.array([3, -2, 5])
    >>> print(decompose_into_unit_vectors(v))
    v = 3.0·e₁ + (-2.0)·e₂ + 5.0·e₃
    """
    # Subscript digits for pretty printing
    subscripts = "₀₁₂₃₄₅₆₇₈₉"
    
    def subscript(n):
        return ''.join(subscripts[int(d)] for d in str(n + 1))
    
    terms = [f"{v[i]}·e{subscript(i)}" if v[i] >= 0 else f"({v[i]})·e{subscript(i)}" 
             for i in range(len(v))]
    return "v = " + " + ".join(terms)


# -----------------------------------------------------------------------------
# SECTION 2.3: Lengths Written in Vector Notation
# -----------------------------------------------------------------------------
#
# The LENGTH (or NORM) of a vector is given by Pythagoras:
#
#   ||v|| = √(v₁² + v₂² + ... + vₚ²)
#
# The SQUARED LENGTH has a beautiful matrix form:
#
#   ||v||² = v₁² + v₂² + ... + vₚ² = v^T v
#
# Here v^T is the TRANSPOSE: the row vector [v₁, v₂, ..., vₚ].
#
# The product v^T v (row times column) gives a scalar: the sum of squares.
#
# This is our FIRST piece of matrix notation. The pattern "row × column = scalar"
# will appear again and again:
#   - Squared length: v^T v
#   - Dot product: u^T v  
#   - Quadratic form: v^T A v  (Chapter 3)
#   - Mahalanobis distance: (z-μ)^T Σ⁻¹ (z-μ)  (Chapter 6)

def vector_length(v: np.ndarray) -> float:
    """
    Compute the Euclidean length (norm) of a vector.
    
    ||v|| = √(v^T v) = √(Σ vᵢ²)
    
    This is the straight-line distance from the origin to the tip of v.
    In trait space, it measures the "magnitude" of a phenotypic change.
    
    Parameters
    ----------
    v : np.ndarray
        Any vector.
    
    Returns
    -------
    float
        The length ||v||.
    
    Note
    ----
    NumPy's np.linalg.norm(v) does the same thing but less transparently.
    We write it out to show the connection to v^T v.
    """
    return np.sqrt(v @ v)  # v @ v is Python's v^T v (dot product with itself)


def squared_length(v: np.ndarray) -> float:
    """
    Compute the squared length of a vector: ||v||² = v^T v.
    
    Why squared length matters:
    
    1. SIMPLER: No square root, just a sum of squares
    
    2. ADDITIVE: For independent components, variances add:
       Var(X + Y) = Var(X) + Var(Y)
       This only works with squared quantities!
    
    3. SMOOTH: The function f(v) = ||v||² is differentiable everywhere.
       The function f(v) = ||v|| has a "kink" at the origin.
       This matters for optimization (least squares).
    
    4. CONNECTS TO VARIANCE: Var(X) = E[||x - μ||²]
       Variance IS mean squared distance from the mean.
    
    Parameters
    ----------
    v : np.ndarray
        Any vector.
    
    Returns
    -------
    float
        The squared length ||v||².
    """
    return v @ v


def demonstrate_squared_length():
    """
    Show the connection between v^T v and the sum of squared components.
    """
    v = np.array([3, 4])
    
    print("\nDemonstrating squared length:")
    print(f"  v = {v}")
    print(f"  v^T v = {v[0]}² + {v[1]}² = {v[0]**2} + {v[1]**2} = {v @ v}")
    print(f"  ||v|| = √(v^T v) = √{v @ v} = {np.sqrt(v @ v)}")
    print(f"  (This is the 3-4-5 right triangle!)")


# -----------------------------------------------------------------------------
# SECTION 2.4: The Dot Product and Angles
# -----------------------------------------------------------------------------
#
# The DOT PRODUCT (or INNER PRODUCT) of two vectors u and v is:
#
#   u · v = u^T v = u₁v₁ + u₂v₂ + ... + uₚvₚ
#
# This simple sum encodes a profound geometric fact:
#
#   u · v = ||u|| ||v|| cos θ
#
# where θ is the angle between u and v.
#
# Consequences:
#   • u · v > 0  →  acute angle (vectors point "somewhat together")
#   • u · v = 0  →  right angle (vectors are ORTHOGONAL/perpendicular)
#   • u · v < 0  →  obtuse angle (vectors point "somewhat apart")
#
# The dot product answers: "How much do these vectors point in the same direction?"

def dot_product(u: np.ndarray, v: np.ndarray) -> float:
    """
    Compute the dot product (inner product) of two vectors.
    
    u · v = u^T v = Σ uᵢvᵢ
    
    Geometric meaning: u · v = ||u|| ||v|| cos θ
    
    Parameters
    ----------
    u, v : np.ndarray
        Two vectors of the same length.
    
    Returns
    -------
    float
        The dot product u · v.
    
    Biological Interpretation
    -------------------------
    If β is a selection gradient and Δz is an evolutionary response:
      • β · Δz > 0 means the response is "in the direction of" selection
      • β · Δz = 0 means the response is orthogonal to selection
      • β · Δz < 0 means the response opposes selection (rare, but possible
        with genetic constraints)
    """
    return u @ v


def angle_between_vectors(u: np.ndarray, v: np.ndarray) -> float:
    """
    Compute the angle (in radians) between two vectors.
    
    Uses: cos θ = (u · v) / (||u|| ||v||)
    
    Parameters
    ----------
    u, v : np.ndarray
        Two non-zero vectors.
    
    Returns
    -------
    float
        The angle θ in radians (between 0 and π).
    
    Example
    -------
    >>> u = np.array([1, 0])
    >>> v = np.array([1, 1])
    >>> theta = angle_between_vectors(u, v)
    >>> np.degrees(theta)  # Should be 45°
    45.0
    """
    cos_theta = dot_product(u, v) / (vector_length(u) * vector_length(v))
    # Clamp to [-1, 1] to handle numerical errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)


def are_orthogonal(u: np.ndarray, v: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Check if two vectors are orthogonal (perpendicular).
    
    Vectors are orthogonal if and only if u · v = 0.
    
    This is crucial for eigenvectors of symmetric matrices:
    eigenvectors corresponding to DIFFERENT eigenvalues are always orthogonal.
    (See Chapter 7: Diagonalisation)
    
    Parameters
    ----------
    u, v : np.ndarray
        Two vectors.
    tol : float
        Tolerance for numerical comparison to zero.
    
    Returns
    -------
    bool
        True if u ⊥ v (u perpendicular to v).
    """
    return np.abs(dot_product(u, v)) < tol


def plot_dot_product_geometry(u: np.ndarray, v: np.ndarray, ax: plt.Axes = None) -> plt.Axes:
    """
    Visualize the geometric meaning of the dot product.
    
    Shows:
    1. The two vectors u and v
    2. The angle θ between them
    3. The dot product value and its sign
    
    Parameters
    ----------
    u, v : np.ndarray
        Two 2D vectors.
    ax : matplotlib Axes, optional
        Axes to plot on.
    
    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    origin = np.array([0, 0])
    
    # Draw vectors
    ax.annotate('', xy=u, xytext=origin,
                arrowprops=dict(arrowstyle='->', color='blue', lw=2.5))
    ax.annotate('', xy=v, xytext=origin,
                arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
    
    # Labels
    ax.text(u[0] + 0.1, u[1] + 0.1, '$\\mathbf{u}$', fontsize=14, color='blue')
    ax.text(v[0] + 0.1, v[1] + 0.1, '$\\mathbf{v}$', fontsize=14, color='red')
    
    # Draw angle arc
    theta = angle_between_vectors(u, v)
    theta_deg = np.degrees(theta)
    
    # Arc from u-direction to v-direction
    angle_u = np.arctan2(u[1], u[0])
    angle_v = np.arctan2(v[1], v[0])
    
    arc_angles = np.linspace(angle_u, angle_v, 30)
    arc_radius = 0.3 * min(vector_length(u), vector_length(v))
    arc_x = arc_radius * np.cos(arc_angles)
    arc_y = arc_radius * np.sin(arc_angles)
    ax.plot(arc_x, arc_y, 'g-', lw=1.5)
    
    # Angle label
    mid_angle = (angle_u + angle_v) / 2
    label_radius = arc_radius + 0.15
    ax.text(label_radius * np.cos(mid_angle), label_radius * np.sin(mid_angle),
            f'$\\theta = {theta_deg:.1f}°$', fontsize=11, color='green')
    
    # Dot product info
    dp = dot_product(u, v)
    sign_word = "positive" if dp > 0 else ("zero" if dp == 0 else "negative")
    ax.text(0.02, 0.98, f'$\\mathbf{{u}} \\cdot \\mathbf{{v}} = {dp:.3f}$ ({sign_word})\n'
                        f'$\\cos\\theta = {np.cos(theta):.3f}$',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Formatting
    all_points = np.array([origin, u, v])
    margin = 0.5
    ax.set_xlim(all_points[:, 0].min() - margin, all_points[:, 0].max() + margin)
    ax.set_ylim(all_points[:, 1].min() - margin, all_points[:, 1].max() + margin)
    ax.set_aspect('equal')
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Trait 1', fontsize=12)
    ax.set_ylabel('Trait 2', fontsize=12)
    
    return ax


# -----------------------------------------------------------------------------
# SECTION 2.5: Projections onto a Direction
# -----------------------------------------------------------------------------
#
# Given a vector v and a direction û (a unit vector), how much of v points
# along that direction?
#
# The SCALAR PROJECTION is:
#
#   comp_û(v) = v · û = ||v|| cos θ
#
# This is the "shadow length" of v onto the line defined by û.
#
# The VECTOR PROJECTION is:
#
#   proj_û(v) = (v · û) û
#
# This is the actual shadow vector: it points along û with magnitude = shadow length.
#
# Why projections matter:
#   • If û is an eigenvector of G, then v · û tells us how much of v aligns with
#     that principal axis of genetic variation.
#   • The variance of a trait COMBINATION is v^T Σ v, which involves projections
#     onto eigenvectors. (Chapter 7)
#   • Selection along direction û has intensity proportional to β · û.

def normalize(v: np.ndarray) -> np.ndarray:
    """
    Convert a vector to a unit vector (length 1) pointing in the same direction.
    
    û = v / ||v||
    
    Unit vectors represent DIRECTIONS without magnitude.
    They are essential for:
      • Specifying a direction of selection
      • Eigenvectors (always normalized)
      • Computing angles (the formula cos θ = u·v / ||u|| ||v|| simplifies
        when both vectors are unit vectors)
    
    Parameters
    ----------
    v : np.ndarray
        Any non-zero vector.
    
    Returns
    -------
    np.ndarray
        The unit vector v / ||v||.
    
    Raises
    ------
    ValueError
        If v is the zero vector.
    """
    length = vector_length(v)
    if length < 1e-15:
        raise ValueError("Cannot normalize the zero vector")
    return v / length


def scalar_projection(v: np.ndarray, direction: np.ndarray) -> float:
    """
    Compute the scalar projection of v onto a direction.
    
    This is the "shadow length" of v when projected onto the line defined by `direction`.
    
    Formula: comp_d(v) = v · d̂ = (v · d) / ||d||
    
    If d is already a unit vector, this simplifies to v · d.
    
    Parameters
    ----------
    v : np.ndarray
        The vector to project.
    direction : np.ndarray
        The direction to project onto (need not be unit length).
    
    Returns
    -------
    float
        The scalar projection (can be negative if v points "away" from direction).
    
    Biological Example
    ------------------
    If v is a response to selection and direction is the selection gradient,
    the scalar projection measures how much of the response is "in the 
    direction of" selection.
    """
    d_hat = normalize(direction)
    return dot_product(v, d_hat)


def vector_projection(v: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """
    Compute the vector projection of v onto a direction.
    
    This is the "shadow" of v cast onto the line defined by `direction`.
    
    Formula: proj_d(v) = (v · d̂) d̂ = [(v · d) / ||d||²] d
    
    Properties:
      • proj_d(v) is parallel to d
      • ||proj_d(v)|| = |v · d̂| (absolute value of scalar projection)
      • v - proj_d(v) is perpendicular to d
    
    Parameters
    ----------
    v : np.ndarray
        The vector to project.
    direction : np.ndarray
        The direction to project onto.
    
    Returns
    -------
    np.ndarray
        The vector projection of v onto direction.
    """
    d_hat = normalize(direction)
    scalar_proj = dot_product(v, d_hat)
    return scalar_proj * d_hat


def orthogonal_complement(v: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """
    Compute the component of v perpendicular to a direction.
    
    v = proj_d(v) + perp_d(v)
    
    where perp_d(v) = v - proj_d(v) is orthogonal to d.
    
    This decomposes any vector into:
      • A component along the direction
      • A component perpendicular to it
    
    Parameters
    ----------
    v : np.ndarray
        The vector to decompose.
    direction : np.ndarray
        The reference direction.
    
    Returns
    -------
    np.ndarray
        The perpendicular component v - proj_d(v).
    """
    return v - vector_projection(v, direction)


def plot_projection(v: np.ndarray, direction: np.ndarray, ax: plt.Axes = None) -> plt.Axes:
    """
    Visualize the projection of v onto a direction.
    
    Shows:
    1. Original vector v
    2. Direction d (as a line through origin)
    3. Projection of v onto d
    4. Perpendicular component (dashed)
    
    Parameters
    ----------
    v : np.ndarray
        The vector to project (2D).
    direction : np.ndarray
        The direction to project onto (2D).
    ax : matplotlib Axes, optional
    
    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    origin = np.array([0, 0])
    
    # Compute projection and perpendicular component
    proj_v = vector_projection(v, direction)
    perp_v = orthogonal_complement(v, direction)
    d_hat = normalize(direction)
    
    # Draw direction line (extended)
    line_extent = max(vector_length(v), vector_length(direction)) * 1.2
    ax.plot([-line_extent * d_hat[0], line_extent * d_hat[0]],
            [-line_extent * d_hat[1], line_extent * d_hat[1]],
            'k-', lw=1, alpha=0.3, label='Direction d')
    
    # Draw original vector v
    ax.annotate('', xy=v, xytext=origin,
                arrowprops=dict(arrowstyle='->', color='blue', lw=2.5))
    ax.text(v[0] + 0.1, v[1] + 0.1, '$\\mathbf{v}$', fontsize=14, color='blue')
    
    # Draw projection
    ax.annotate('', xy=proj_v, xytext=origin,
                arrowprops=dict(arrowstyle='->', color='green', lw=2.5))
    ax.text(proj_v[0] - 0.1, proj_v[1] - 0.25, 
            '$\\mathrm{proj}_d(\\mathbf{v})$', fontsize=12, color='green')
    
    # Draw perpendicular component (from proj_v to v)
    ax.annotate('', xy=v, xytext=proj_v,
                arrowprops=dict(arrowstyle='->', color='red', lw=2, linestyle='--'))
    mid_perp = (proj_v + v) / 2
    ax.text(mid_perp[0] + 0.1, mid_perp[1], 
            '$\\mathbf{v} - \\mathrm{proj}$', fontsize=10, color='red')
    
    # Draw right angle marker
    perp_size = 0.1 * vector_length(v)
    if vector_length(proj_v) > 0.01:  # Only if projection is non-trivial
        # Right angle marker at the projection point
        perp_dir = normalize(perp_v) if vector_length(perp_v) > 1e-10 else np.array([0, 0])
        corner1 = proj_v + perp_size * perp_dir
        corner2 = proj_v + perp_size * d_hat
        corner3 = corner1 + perp_size * d_hat
        ax.plot([corner1[0], corner3[0], corner2[0]], 
                [corner1[1], corner3[1], corner2[1]], 'k-', lw=1)
    
    # Information box
    scalar_proj = scalar_projection(v, direction)
    ax.text(0.02, 0.98, 
            f'$\\mathbf{{v}} \\cdot \\hat{{d}} = {scalar_proj:.3f}$\n'
            f'$||\\mathrm{{proj}}_d(\\mathbf{{v}})|| = {vector_length(proj_v):.3f}$\n'
            f'$||\\perp|| = {vector_length(perp_v):.3f}$',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Verify Pythagorean theorem: ||v||² = ||proj||² + ||perp||²
    # (This is built into the geometry!)
    
    # Formatting
    all_points = np.array([origin, v, proj_v, direction])
    margin = 0.5
    ax.set_xlim(all_points[:, 0].min() - margin, all_points[:, 0].max() + margin)
    ax.set_ylim(all_points[:, 1].min() - margin, all_points[:, 1].max() + margin)
    ax.set_aspect('equal')
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Trait 1', fontsize=12)
    ax.set_ylabel('Trait 2', fontsize=12)
    
    return ax


# -----------------------------------------------------------------------------
# SECTION 2.6: Distances Between Phenotypes
# -----------------------------------------------------------------------------
#
# The distance between two phenotypes z_i and z_j is the LENGTH of their
# difference vector:
#
#   d(i, j) = ||z_j - z_i|| = √[(z_j - z_i)^T (z_j - z_i)]
#
# In matrix notation, squared distance is:
#
#   d²(i, j) = (z_j - z_i)^T (z_j - z_i)
#
# IMPORTANT PREVIEW:
#   This formula has the pattern: row × column.
#   Later (Chapter 6), we will INSERT A MATRIX in the middle:
#   
#   d²_Mahalanobis = (z_j - z_i)^T  Σ⁻¹  (z_j - z_i)
#
#   This changes everything. The matrix "reshapes" how we measure distance.

def euclidean_distance(z_i: np.ndarray, z_j: np.ndarray) -> float:
    """
    Compute the Euclidean distance between two phenotypes.
    
    d(i, j) = ||z_j - z_i|| = √[(z_j - z_i)^T (z_j - z_i)]
    
    This is straight-line distance in trait space.
    
    Parameters
    ----------
    z_i, z_j : np.ndarray
        Phenotype vectors of two individuals.
    
    Returns
    -------
    float
        The Euclidean distance d(i, j).
    
    WARNING
    -------
    Euclidean distance treats all traits equally. This is problematic when:
      • Traits are measured in different units (mm vs kg)
      • Traits have different variances
      • Traits are correlated
    
    Chapter 5 will show WHY this fails.
    Chapter 6 will show the FIX: Mahalanobis distance.
    """
    diff = z_j - z_i
    return vector_length(diff)


def squared_euclidean_distance(z_i: np.ndarray, z_j: np.ndarray) -> float:
    """
    Compute the squared Euclidean distance between two phenotypes.
    
    d²(i, j) = (z_j - z_i)^T (z_j - z_i) = Σ_k (z_jk - z_ik)²
    
    Why squared distance is often more useful:
    
    1. Connects to variance: Var(X) = E[d²(x, μ)]
    2. Computationally simpler (no square root)
    3. Leads naturally to least-squares methods
    4. The pattern (Δz)^T(Δz) generalizes to (Δz)^T M (Δz)
    
    Parameters
    ----------
    z_i, z_j : np.ndarray
        Phenotype vectors.
    
    Returns
    -------
    float
        The squared distance d²(i, j).
    """
    diff = z_j - z_i
    return squared_length(diff)


# -----------------------------------------------------------------------------
# SUMMARY VISUALIZATION: All Concepts Together
# -----------------------------------------------------------------------------

def demonstrate_all_concepts():
    """
    Create a comprehensive figure showing all concepts from Chapter 2.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    # Panel (a): Unit vectors and decomposition
    ax = axes[0, 0]
    v = np.array([3, 2])
    e1 = unit_vector(0, 2)
    e2 = unit_vector(1, 2)
    origin = np.array([0, 0])
    
    # Draw unit vectors
    ax.annotate('', xy=e1, xytext=origin,
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.annotate('', xy=e2, xytext=origin,
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.text(e1[0] + 0.1, e1[1] - 0.15, '$\\mathbf{e}_1$', fontsize=12, color='gray')
    ax.text(e2[0] - 0.25, e2[1] + 0.1, '$\\mathbf{e}_2$', fontsize=12, color='gray')
    
    # Draw v
    ax.annotate('', xy=v, xytext=origin,
                arrowprops=dict(arrowstyle='->', color='blue', lw=2.5))
    ax.text(v[0] + 0.1, v[1] + 0.1, f'$\\mathbf{{v}} = ({v[0]}, {v[1]})$', 
            fontsize=12, color='blue')
    
    # Show components
    ax.plot([v[0], v[0]], [0, v[1]], 'b--', alpha=0.5, lw=1.5)
    ax.plot([0, v[0]], [v[1], v[1]], 'b--', alpha=0.5, lw=1.5)
    ax.text(v[0] + 0.1, v[1]/2, f'$v_2 = {v[1]}$', fontsize=10, color='blue')
    ax.text(v[0]/2, v[1] + 0.1, f'$v_1 = {v[0]}$', fontsize=10, color='blue')
    
    ax.set_xlim(-0.5, 4)
    ax.set_ylim(-0.5, 3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Trait 1')
    ax.set_ylabel('Trait 2')
    ax.set_title('(a) Vector Components: $\\mathbf{v} = v_1\\mathbf{e}_1 + v_2\\mathbf{e}_2$',
                fontweight='bold')
    
    # Panel (b): Dot product and angles
    ax = axes[0, 1]
    u = np.array([2, 1])
    v = np.array([1, 2])
    plot_dot_product_geometry(u, v, ax=ax)
    ax.set_title('(b) Dot Product: $\\mathbf{u} \\cdot \\mathbf{v} = ||\\mathbf{u}|| \\, ||\\mathbf{v}|| \\cos\\theta$',
                fontweight='bold')
    
    # Panel (c): Projection
    ax = axes[1, 0]
    v = np.array([3, 2])
    d = np.array([2, 0.5])
    plot_projection(v, d, ax=ax)
    ax.set_title('(c) Projection: Decomposing $\\mathbf{v}$ Along a Direction',
                fontweight='bold')
    
    # Panel (d): Distance
    ax = axes[1, 1]
    z1 = np.array([1, 1])
    z2 = np.array([4, 3])
    
    ax.scatter(*z1, s=100, c='blue', zorder=5)
    ax.scatter(*z2, s=100, c='red', zorder=5)
    ax.text(z1[0] - 0.3, z1[1] + 0.15, '$\\mathbf{z}_1$', fontsize=12, color='blue')
    ax.text(z2[0] + 0.1, z2[1] + 0.1, '$\\mathbf{z}_2$', fontsize=12, color='red')
    
    # Difference vector
    ax.annotate('', xy=z2, xytext=z1,
                arrowprops=dict(arrowstyle='->', color='green', lw=2.5))
    
    # Show the right triangle
    ax.plot([z1[0], z2[0]], [z1[1], z1[1]], 'k--', alpha=0.5)
    ax.plot([z2[0], z2[0]], [z1[1], z2[1]], 'k--', alpha=0.5)
    
    diff = z2 - z1
    dist = euclidean_distance(z1, z2)
    mid = (z1 + z2) / 2
    
    ax.text(mid[0] - 0.5, mid[1] + 0.2, 
            f'$||\\mathbf{{z}}_2 - \\mathbf{{z}}_1|| = {dist:.2f}$',
            fontsize=12, color='green')
    ax.text((z1[0] + z2[0])/2, z1[1] - 0.25, f'$\\Delta_1 = {diff[0]}$', fontsize=10)
    ax.text(z2[0] + 0.1, (z1[1] + z2[1])/2, f'$\\Delta_2 = {diff[1]}$', fontsize=10)
    
    ax.text(0.02, 0.98, 
            f'$d^2 = \\Delta_1^2 + \\Delta_2^2 = {diff[0]**2} + {diff[1]**2} = {dist**2:.0f}$\n'
            f'$d = \\sqrt{{{dist**2:.0f}}} = {dist:.2f}$',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 4)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Trait 1')
    ax.set_ylabel('Trait 2')
    ax.set_title('(d) Distance: $d = ||\\mathbf{z}_2 - \\mathbf{z}_1||$ (Pythagoras)',
                fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('ch02_all_concepts.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Saved: ch02_all_concepts.png")


# -----------------------------------------------------------------------------
# BIOLOGICAL APPLICATION: Selection and Response Alignment
# -----------------------------------------------------------------------------

def selection_response_alignment(
    beta: np.ndarray, 
    delta_z: np.ndarray,
    verbose: bool = True
) -> dict:
    """
    Analyze the alignment between selection and evolutionary response.
    
    When selection gradient β and response Δz̄ are not parallel, the response
    is being DEFLECTED by genetic constraints (encoded in G).
    
    This function computes:
    1. The angle between β and Δz̄
    2. How much of the response is "in the direction of" selection
    3. How much is perpendicular (wasted effort, in a sense)
    
    Parameters
    ----------
    beta : np.ndarray
        Selection gradient (direction fitness wants to push).
    delta_z : np.ndarray
        Response to selection (how the mean actually moved).
    verbose : bool
        If True, print interpretation.
    
    Returns
    -------
    dict
        Contains: 'angle_rad', 'angle_deg', 'alignment' (cos θ),
                  'response_along_selection', 'response_perpendicular'
    """
    angle = angle_between_vectors(beta, delta_z)
    alignment = np.cos(angle)
    
    # Decompose response into selection-aligned and perpendicular components
    response_along = vector_projection(delta_z, beta)
    response_perp = orthogonal_complement(delta_z, beta)
    
    results = {
        'angle_rad': angle,
        'angle_deg': np.degrees(angle),
        'alignment': alignment,
        'response_along_selection': response_along,
        'response_perpendicular': response_perp,
        'fraction_along': vector_length(response_along) / vector_length(delta_z),
        'fraction_perp': vector_length(response_perp) / vector_length(delta_z)
    }
    
    if verbose:
        print("\nSelection-Response Alignment Analysis:")
        print("-" * 45)
        print(f"Selection gradient β:     {beta}")
        print(f"Response Δz̄:              {delta_z}")
        print(f"\nAngle between β and Δz̄:   {results['angle_deg']:.1f}°")
        print(f"Alignment (cos θ):         {results['alignment']:.3f}")
        print(f"\nResponse along selection:  {results['response_along_selection'].round(3)}")
        print(f"Response perpendicular:    {results['response_perpendicular'].round(3)}")
        print(f"\nFraction along:            {results['fraction_along']:.1%}")
        print(f"Fraction perpendicular:    {results['fraction_perp']:.1%}")
        
        if results['angle_deg'] < 15:
            print("\n→ Excellent alignment: response tracks selection closely")
        elif results['angle_deg'] < 45:
            print("\n→ Moderate deflection: genetic correlations are biasing response")
        else:
            print("\n→ Severe deflection: G matrix is strongly constraining evolution")
    
    return results


# -----------------------------------------------------------------------------
# MAIN DEMONSTRATION
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CHAPTER 2: Vectors, Coordinates, and Angles")
    print("Seeing the Shape — Code Companion")
    print("=" * 70)
    
    # Section 2.1: Column vectors
    print("\n" + "-" * 50)
    print("Section 2.1: Column vectors and coordinates")
    print("-" * 50)
    
    z = create_phenotype_vector([4.2, 3.1, 2.8])
    print(f"\nPhenotype vector z = {z}")
    print(f"Number of traits: {len(z)}")
    print(f"Trait 1 value: z[0] = {z[0]}")
    print(f"Trait 2 value: z[1] = {z[1]}")
    print(f"Trait 3 value: z[2] = {z[2]}")
    
    # Section 2.2: Unit vectors
    print("\n" + "-" * 50)
    print("Section 2.2: Unit vectors and decomposition")
    print("-" * 50)
    
    print("\nUnit vectors in 3D:")
    for i in range(3):
        print(f"  e{i+1} = {unit_vector(i, 3)}")
    
    v = np.array([3, -2, 5])
    print(f"\nFor v = {v}:")
    print(f"  {decompose_into_unit_vectors(v)}")
    print("  This just says: the components ARE the coefficients!")
    
    # Section 2.3: Lengths
    print("\n" + "-" * 50)
    print("Section 2.3: Lengths and squared lengths")
    print("-" * 50)
    
    demonstrate_squared_length()
    
    v = np.array([1, 2, 2])
    print(f"\nFor v = {v}:")
    print(f"  ||v||² = v^T v = {squared_length(v)}")
    print(f"  ||v|| = √(v^T v) = {vector_length(v)}")
    
    # Section 2.4: Dot product and angles
    print("\n" + "-" * 50)
    print("Section 2.4: The dot product and angles")
    print("-" * 50)
    
    u = np.array([1, 0])
    v1 = np.array([1, 1])
    v2 = np.array([0, 1])
    v3 = np.array([-1, 0])
    
    print(f"\nu = {u}")
    for v, name in [(v1, "(1,1)"), (v2, "(0,1)"), (v3, "(-1,0)")]:
        dp = dot_product(u, v)
        angle = np.degrees(angle_between_vectors(u, v))
        print(f"\nv = {name}:")
        print(f"  u · v = {dp}")
        print(f"  Angle = {angle:.1f}°")
        print(f"  Orthogonal? {are_orthogonal(u, v)}")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, v, title in [(axes[0], v1, "Acute angle"), 
                          (axes[1], v2, "Right angle"),
                          (axes[2], v3, "Obtuse angle")]:
        plot_dot_product_geometry(u, v, ax=ax)
        ax.set_title(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ch02_dot_product_angles.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n✓ Saved: ch02_dot_product_angles.png")
    
    # Section 2.5: Projections
    print("\n" + "-" * 50)
    print("Section 2.5: Projections onto a direction")
    print("-" * 50)
    
    v = np.array([3, 2])
    direction = np.array([1, 0])  # Project onto trait 1 axis
    
    print(f"\nProjecting v = {v} onto direction = {direction}")
    print(f"  Scalar projection (shadow length): {scalar_projection(v, direction):.3f}")
    print(f"  Vector projection: {vector_projection(v, direction)}")
    print(f"  Perpendicular component: {orthogonal_complement(v, direction)}")
    print(f"\n  Verify: proj + perp = v?")
    print(f"  {vector_projection(v, direction)} + {orthogonal_complement(v, direction)} = {v}")
    
    # Verify Pythagorean theorem
    proj = vector_projection(v, direction)
    perp = orthogonal_complement(v, direction)
    print(f"\n  Pythagorean check: ||v||² = ||proj||² + ||perp||²")
    print(f"  {squared_length(v)} = {squared_length(proj):.1f} + {squared_length(perp):.1f}")
    
    # Visualize projection
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_projection(v, np.array([2, 1]), ax=ax)
    ax.set_title('Projection of v onto direction d', fontsize=14, fontweight='bold')
    plt.savefig('ch02_projection.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n✓ Saved: ch02_projection.png")
    
    # Section 2.6: Distances
    print("\n" + "-" * 50)
    print("Section 2.6: Distances between phenotypes")
    print("-" * 50)
    
    z_i = np.array([2, 3])
    z_j = np.array([5, 7])
    
    print(f"\nTwo phenotypes:")
    print(f"  z_i = {z_i}")
    print(f"  z_j = {z_j}")
    print(f"\nDifference: z_j - z_i = {z_j - z_i}")
    print(f"Squared distance: {squared_euclidean_distance(z_i, z_j)}")
    print(f"Distance: {euclidean_distance(z_i, z_j):.4f}")
    
    # Show the calculation step by step
    diff = z_j - z_i
    print(f"\nStep by step:")
    print(f"  d² = (z_j - z_i)^T (z_j - z_i)")
    print(f"     = {diff}^T @ {diff}")
    print(f"     = {diff[0]}² + {diff[1]}²")
    print(f"     = {diff[0]**2} + {diff[1]**2}")
    print(f"     = {squared_euclidean_distance(z_i, z_j)}")
    print(f"  d  = √{squared_euclidean_distance(z_i, z_j)} = {euclidean_distance(z_i, z_j):.4f}")
    
    # Comprehensive figure
    print("\n" + "-" * 50)
    print("Creating comprehensive summary figure...")
    print("-" * 50)
    demonstrate_all_concepts()
    
    # Biological application
    print("\n" + "-" * 50)
    print("Biological Application: Selection-Response Alignment")
    print("-" * 50)
    
    # Simulate selection favoring trait 2, but response is deflected
    beta = np.array([0.0, 1.0])  # Selection on trait 2 only
    delta_z = np.array([0.4, 0.8])  # Response is deflected (includes trait 1 change)
    
    results = selection_response_alignment(beta, delta_z)
    
    # Visualize
    fig, ax = plt.subplots(figsize=(8, 8))
    origin = np.array([0, 0])
    
    ax.annotate('', xy=beta, xytext=origin,
                arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
    ax.text(beta[0] + 0.05, beta[1], '$\\beta$ (selection)', fontsize=12, color='red')
    
    ax.annotate('', xy=delta_z, xytext=origin,
                arrowprops=dict(arrowstyle='->', color='blue', lw=2.5))
    ax.text(delta_z[0] + 0.05, delta_z[1] - 0.1, '$\\Delta\\bar{z}$ (response)', 
            fontsize=12, color='blue')
    
    ax.annotate('', xy=results['response_along_selection'], xytext=origin,
                arrowprops=dict(arrowstyle='->', color='green', lw=2, alpha=0.7))
    ax.text(0.05, results['response_along_selection'][1] - 0.1, 
            'Along $\\beta$', fontsize=10, color='green')
    
    ax.set_xlim(-0.2, 0.8)
    ax.set_ylim(-0.2, 1.2)
    ax.set_aspect('equal')
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Trait 1')
    ax.set_ylabel('Trait 2')
    ax.set_title(f'Selection-Response Deflection: {results["angle_deg"]:.1f}°',
                fontsize=14, fontweight='bold')
    
    plt.savefig('ch02_selection_response.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n✓ Saved: ch02_selection_response.png")
    
    print("\n" + "=" * 70)
    print("END OF CHAPTER 2")
    print("=" * 70)
    print("""
Key Takeaways:
  1. A phenotype is a COLUMN VECTOR z = [z₁, z₂, ..., zₚ]^T
  2. Squared length is v^T v — the pattern "row × column" appears everywhere
  3. The DOT PRODUCT u·v = ||u|| ||v|| cos θ measures alignment
  4. Orthogonal vectors (u·v = 0) are perpendicular — crucial for eigenvectors
  5. PROJECTION decomposes a vector into parallel and perpendicular parts
  6. Euclidean distance is ||z_j - z_i|| = √[(z_j - z_i)^T (z_j - z_i)]
  
The Big Preview:
  The pattern (Δz)^T (Δz) for Euclidean distance will become
  (Δz)^T Σ⁻¹ (Δz) for Mahalanobis distance.
  
  The matrix Σ⁻¹ in the middle changes EVERYTHING. (Chapter 6)
  
Next: Chapter 3 introduces matrices as transformations.
""")


# =============================================================================
# EXERCISES
# =============================================================================
"""
EXERCISE 2.1: Vector Length and the 3-4-5 Triangle
--------------------------------------------------
(a) Compute the length of v = [3, 4] by hand using the Pythagorean theorem.
(b) Verify using ||v||² = v^T v.
(c) Find a 3D vector with integer components that has length 7.
    (Hint: 2² + 3² + 6² = ?)

EXERCISE 2.2: Dot Product Properties
------------------------------------
For u = [2, 1] and v = [1, -2]:
(a) Compute u · v.
(b) What is the angle between u and v?
(c) Are u and v orthogonal?
(d) Compute u · u and v · v. How do these relate to ||u|| and ||v||?

EXERCISE 2.3: Projection in Selection
-------------------------------------
A selection gradient is β = [0.3, 0.6] (both traits favored, trait 2 more strongly).
The actual response is Δz̄ = [0.5, 0.4].

(a) Compute the angle between β and Δz̄.
(b) How much of the response is "along" the selection direction?
(c) How much is perpendicular?
(d) Interpret biologically: why might the response not align with selection?

EXERCISE 2.4: Orthogonality of Eigenvectors
-------------------------------------------
Two eigenvectors of a symmetric matrix are v₁ = [1, 1]/√2 and v₂ = [1, -1]/√2.
(a) Verify that they are orthogonal.
(b) Verify that they are unit vectors.
(c) Express the vector w = [3, 1] as a linear combination of v₁ and v₂.
    (Hint: the coefficients are w · v₁ and w · v₂)
(d) Why is this decomposition useful? (Preview: PCA, Chapter 11)

EXERCISE 2.5: Distance in Different Units (Preview of Chapter 5)
----------------------------------------------------------------
Two beetles: A = [12 mm body, 5 g mass], B = [14 mm body, 6 g mass].
(a) Compute the Euclidean distance d(A, B).
(b) Now convert mass to mg: A' = [12, 5000], B' = [14, 6000].
    Compute d(A', B').
(c) The distances are wildly different! What went wrong?
(d) Preview: Mahalanobis distance will fix this. How might it work?

EXERCISE 2.6: The Angle Between g_max and Selection
---------------------------------------------------
Suppose the G matrix has first eigenvector (line of least resistance):
    g_max = [0.8, 0.6]

Selection favors only trait 1: β = [1, 0].

(a) What is the angle between g_max and β?
(b) Will the response Δz̄ = Gβ be closer to β or to g_max? Why?
(c) When would β and g_max be perfectly aligned?
"""
