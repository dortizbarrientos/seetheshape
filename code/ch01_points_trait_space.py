#!/usr/bin/env python3
"""
================================================================================
CHAPTER 1: Points and Trait Space
================================================================================
Book: "Seeing the Shape: A Geometric Introduction to Multivariate Quantitative 
       Genetics" by Daniel Ortiz-Barrientos

This script introduces the fundamental geometric perspective:
  - Traits as axes in a coordinate system
  - Individual phenotypes as POINTS in trait space
  - Differences between individuals as ARROWS (vectors)
  - The population mean as a natural reference point

Key Insight:
    A multivariate phenotype is not a list of numbers—it is a LOCATION in space.
    This simple change in perspective is the foundation of everything that follows.

Sections covered:
    §1.1 Traits as axes, individuals as points
    §1.2 Differences between individuals as arrows
    §1.3 The mean as natural reference
    §1.4 Adding and stretching arrows
    §1.5 From arrows to vectors
    §1.6 Distances and lengths of vectors

================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

# -----------------------------------------------------------------------------
# SECTION 1.1: Traits as Axes, Individuals as Points
# -----------------------------------------------------------------------------
# 
# The central conceptual move: think of each measured trait as an AXIS in a
# coordinate system. With p traits, we have a p-dimensional space.
#
# Each individual organism is then a POINT in this space, located at the
# coordinates given by its trait values.
#
# Example: If we measure body length (x) and wing span (y), then an individual
# with length 4.2 and wingspan 3.1 is the point (4.2, 3.1).

def plot_trait_space_2d(
    data: np.ndarray,
    trait_names: Tuple[str, str] = ("Trait 1", "Trait 2"),
    title: str = "Trait Space",
    show_mean: bool = True,
    ax: plt.Axes = None
) -> plt.Axes:
    """
    Plot individuals as points in a 2D trait space.
    
    This is the most fundamental visualization in multivariate analysis:
    each row of `data` becomes a point, revealing the SHAPE of variation.
    
    Parameters
    ----------
    data : np.ndarray
        An (n × 2) array where each row is an individual's phenotype.
    trait_names : tuple of str
        Labels for the x and y axes.
    title : str
        Plot title.
    show_mean : bool
        If True, mark the centroid (mean phenotype) with a red diamond.
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates a new figure.
    
    Returns
    -------
    ax : matplotlib Axes
        The axes with the plot.
    
    Example
    -------
    >>> data = np.array([[4.2, 3.1], [5.1, 3.8], [3.9, 2.9]])
    >>> plot_trait_space_2d(data, ("Body length", "Wing span"))
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot each individual as a point
    ax.scatter(data[:, 0], data[:, 1], s=60, alpha=0.7, 
               edgecolors='white', linewidths=0.5, label='Individuals')
    
    # Optionally show the mean (centroid)
    if show_mean:
        mean = np.mean(data, axis=0)
        ax.scatter(mean[0], mean[1], s=150, c='red', marker='D',
                   edgecolors='black', linewidths=1.5, label=f'Mean', zorder=5)
        ax.annotate(f'$\\bar{{z}} = ({mean[0]:.2f}, {mean[1]:.2f})$',
                    xy=mean, xytext=(10, 10), textcoords='offset points',
                    fontsize=10, color='red')
    
    ax.set_xlabel(trait_names[0], fontsize=12)
    ax.set_ylabel(trait_names[1], fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    return ax


# -----------------------------------------------------------------------------
# SECTION 1.2: Differences Between Individuals as Arrows
# -----------------------------------------------------------------------------
#
# The difference between two individuals i and j is the ARROW from i to j.
# 
# Algebraically: difference = z_j - z_i
# Geometrically: an arrow pointing from z_i to z_j
#
# This arrow has:
#   - A DIRECTION: which way you must go in trait space
#   - A LENGTH: how far you must go
#
# These arrows are the precursors to vectors.

def plot_difference_arrow(
    z_i: np.ndarray,
    z_j: np.ndarray,
    ax: plt.Axes,
    label_i: str = "i",
    label_j: str = "j",
    arrow_color: str = 'blue'
) -> None:
    """
    Draw an arrow from individual i to individual j, showing their difference.
    
    The difference z_j - z_i tells us:
      - "Start at z_i. Move (z_j[0] - z_i[0]) units in trait 1."
      - "Then move (z_j[1] - z_i[1]) units in trait 2."
      - "You arrive at z_j."
    
    Parameters
    ----------
    z_i, z_j : np.ndarray
        Phenotype vectors (1D arrays of length 2).
    ax : matplotlib Axes
        Axes to draw on.
    label_i, label_j : str
        Labels for the points.
    arrow_color : str
        Color of the difference arrow.
    """
    # Plot the two points
    ax.scatter(*z_i, s=100, c='black', zorder=5)
    ax.scatter(*z_j, s=100, c='black', zorder=5)
    
    # Label them
    ax.annotate(label_i, z_i, xytext=(5, 5), textcoords='offset points', fontsize=12)
    ax.annotate(label_j, z_j, xytext=(5, 5), textcoords='offset points', fontsize=12)
    
    # Draw the arrow from i to j
    ax.annotate('', xy=z_j, xytext=z_i,
                arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2))
    
    # Compute and display the difference
    diff = z_j - z_i
    midpoint = (z_i + z_j) / 2
    ax.text(midpoint[0], midpoint[1] + 0.2, 
            f'$\\Delta z = ({diff[0]:.1f}, {diff[1]:.1f})$',
            fontsize=10, ha='center', color=arrow_color)


# -----------------------------------------------------------------------------
# SECTION 1.3: The Mean as Natural Reference
# -----------------------------------------------------------------------------
#
# When summarizing a population, we need a reference point. The MEAN is the
# natural choice because:
#
#   1. Deviations from the mean sum to zero (by definition)
#   2. The mean minimizes the sum of squared distances
#   3. It represents the "center of mass" of the point cloud
#
# Once we fix the mean as reference, every individual is described by its
# DEVIATION from the mean: d_i = z_i - z̄

def compute_deviations(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the mean and individual deviations from the mean.
    
    The deviation d_i = z_i - z̄ is the arrow from the mean to individual i.
    
    Properties of deviations:
      - They always sum to zero: Σ d_i = 0 (this defines the mean!)
      - They are the building blocks of variance and covariance
    
    Parameters
    ----------
    data : np.ndarray
        An (n × p) array of phenotypes.
    
    Returns
    -------
    mean : np.ndarray
        The mean phenotype (length p).
    deviations : np.ndarray
        An (n × p) array of deviations from the mean.
    
    Example
    -------
    >>> data = np.array([[2, 3], [4, 5], [6, 7]])
    >>> mean, devs = compute_deviations(data)
    >>> print(mean)  # [4. 5.]
    >>> print(devs.sum(axis=0))  # [0. 0.] - deviations sum to zero!
    """
    mean = np.mean(data, axis=0)
    deviations = data - mean  # Broadcasting: subtracts mean from each row
    
    # Sanity check: deviations must sum to zero (up to floating point error)
    assert np.allclose(deviations.sum(axis=0), 0), "Deviations should sum to zero!"
    
    return mean, deviations


def plot_deviations_from_mean(
    data: np.ndarray,
    trait_names: Tuple[str, str] = ("Trait 1", "Trait 2"),
    ax: plt.Axes = None
) -> plt.Axes:
    """
    Visualize each individual's deviation from the mean as an arrow.
    
    This plot shows that variance is about the LENGTHS of these arrows:
    how far, on average, do individuals sit from the center?
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    mean, deviations = compute_deviations(data)
    n = len(data)
    
    # Plot the mean as the reference point
    ax.scatter(mean[0], mean[1], s=200, c='red', marker='D', 
               edgecolors='black', linewidths=2, zorder=10, label='Mean $\\bar{z}$')
    
    # Plot each individual and its deviation arrow
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n))
    for i, (point, dev) in enumerate(zip(data, deviations)):
        # Draw arrow from mean to point
        ax.annotate('', xy=point, xytext=mean,
                    arrowprops=dict(arrowstyle='->', color=colors[i], lw=1.5, alpha=0.7))
        # Plot the point
        ax.scatter(point[0], point[1], s=80, c=[colors[i]], 
                   edgecolors='white', linewidths=0.5, zorder=5)
    
    ax.set_xlabel(trait_names[0], fontsize=12)
    ax.set_ylabel(trait_names[1], fontsize=12)
    ax.set_title('Deviations from the Mean', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    return ax


# -----------------------------------------------------------------------------
# SECTION 1.4 & 1.5: Adding and Stretching Arrows → Vectors
# -----------------------------------------------------------------------------
#
# Arrows in trait space have two key properties:
#   1. They can be ADDED: join head-to-tail
#   2. They can be SCALED: stretch or shrink
#
# Any object with these properties is called a VECTOR.
# 
# The selection differential, the response to selection, and the selection
# gradient are all vectors. Evolutionary change IS vector arithmetic.

def add_vectors(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Add two vectors: the combined change from applying u then v.
    
    Geometrically: place v's tail at u's head. The sum is the arrow
    from u's tail to v's head.
    
    Biologically: if one generation changes by u and the next by v,
    the total change is u + v.
    """
    return u + v


def scale_vector(c: float, v: np.ndarray) -> np.ndarray:
    """
    Scale a vector by a constant factor.
    
    c > 1: stretch (larger change in the same direction)
    0 < c < 1: shrink (smaller change in the same direction)
    c < 0: reverse direction
    c = 0: no change at all (zero vector)
    """
    return c * v


def demonstrate_vector_operations():
    """
    Demonstrate vector addition and scaling with evolutionary interpretation.
    """
    print("=" * 60)
    print("VECTOR OPERATIONS IN TRAIT SPACE")
    print("=" * 60)
    
    # Example vectors
    u = np.array([2.0, 1.0])   # Change in generation 1
    v = np.array([1.0, 3.0])   # Change in generation 2
    
    print(f"\nVector u (gen 1 response): {u}")
    print(f"Vector v (gen 2 response): {v}")
    
    # Addition
    total = add_vectors(u, v)
    print(f"\nTotal change u + v: {total}")
    print("  → Two generations of response accumulate by vector addition")
    
    # Scaling
    doubled = scale_vector(2.0, u)
    halved = scale_vector(0.5, u)
    reversed_v = scale_vector(-1.0, v)
    
    print(f"\nDouble the selection intensity: 2u = {doubled}")
    print(f"Half the selection intensity: 0.5u = {halved}")
    print(f"Reverse selection direction: -v = {reversed_v}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel 1: Vector addition
    ax1 = axes[0]
    origin = np.array([0, 0])
    
    # Draw u from origin
    ax1.annotate('', xy=u, xytext=origin,
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax1.text(u[0]/2, u[1]/2 - 0.3, '$\\mathbf{u}$', fontsize=14, color='blue')
    
    # Draw v from tip of u
    ax1.annotate('', xy=u + v, xytext=u,
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax1.text(u[0] + v[0]/2 + 0.2, u[1] + v[1]/2, '$\\mathbf{v}$', fontsize=14, color='red')
    
    # Draw sum from origin
    ax1.annotate('', xy=u + v, xytext=origin,
                arrowprops=dict(arrowstyle='->', color='green', lw=2.5))
    ax1.text(total[0]/2 - 0.4, total[1]/2, '$\\mathbf{u+v}$', fontsize=14, color='green')
    
    ax1.set_xlim(-0.5, 5)
    ax1.set_ylim(-0.5, 5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Trait 1')
    ax1.set_ylabel('Trait 2')
    ax1.set_title('(a) Vector Addition: Head-to-Tail', fontweight='bold')
    
    # Panel 2: Scaling
    ax2 = axes[1]
    
    # Original vector
    ax2.annotate('', xy=u, xytext=origin,
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax2.text(u[0] + 0.1, u[1], '$\\mathbf{u}$', fontsize=14, color='blue')
    
    # Scaled versions
    ax2.annotate('', xy=doubled, xytext=origin,
                arrowprops=dict(arrowstyle='->', color='purple', lw=2, alpha=0.7))
    ax2.text(doubled[0] + 0.1, doubled[1], '$2\\mathbf{u}$', fontsize=14, color='purple')
    
    ax2.annotate('', xy=halved, xytext=origin,
                arrowprops=dict(arrowstyle='->', color='orange', lw=2, alpha=0.7))
    ax2.text(halved[0] + 0.1, halved[1] - 0.2, '$0.5\\mathbf{u}$', fontsize=14, color='orange')
    
    ax2.set_xlim(-0.5, 5)
    ax2.set_ylim(-0.5, 3)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Trait 1')
    ax2.set_ylabel('Trait 2')
    ax2.set_title('(b) Vector Scaling: Stretch and Shrink', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('ch01_vector_operations.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Figure saved: ch01_vector_operations.png")


# -----------------------------------------------------------------------------
# SECTION 1.6: Distances and Lengths of Vectors
# -----------------------------------------------------------------------------
#
# The LENGTH of a vector is given by the Pythagorean theorem:
#
#   ||v|| = sqrt(v₁² + v₂² + ... + vₚ²)
#
# The SQUARED LENGTH is even simpler:
#
#   ||v||² = v₁² + v₂² + ... + vₚ² = v^T v
#
# This sum of squared components is the foundation of variance.

def vector_length(v: np.ndarray) -> float:
    """
    Compute the Euclidean length (norm) of a vector.
    
    ||v|| = sqrt(Σ vᵢ²) = sqrt(v^T v)
    
    This is the straight-line distance from the origin to the tip of v.
    """
    return np.sqrt(np.sum(v**2))


def squared_length(v: np.ndarray) -> float:
    """
    Compute the squared length of a vector: ||v||² = v^T v.
    
    Why squared length matters:
      1. Computationally simpler (no square root)
      2. Connects directly to variance: Var(X) = E[||x - μ||²]
      3. The foundation of least-squares methods
    """
    return np.sum(v**2)


def euclidean_distance(z_i: np.ndarray, z_j: np.ndarray) -> float:
    """
    Euclidean distance between two phenotypes.
    
    d(i, j) = ||z_j - z_i|| = sqrt(Σ (z_jk - z_ik)²)
    
    This measures "how different" two individuals are in trait space.
    
    WARNING: This ignores correlations between traits! 
    (See Chapter 5 for why this is a problem.)
    """
    diff = z_j - z_i
    return vector_length(diff)


# -----------------------------------------------------------------------------
# MAIN DEMONSTRATION
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CHAPTER 1: Points and Trait Space")
    print("Seeing the Shape — Code Companion")
    print("=" * 70)
    
    # Generate example data: 20 individuals measured for 2 traits
    np.random.seed(42)
    n = 20
    
    # Create correlated data (positive covariance)
    # This simulates, e.g., body size traits that scale together
    mean_true = np.array([5.0, 4.0])
    cov_true = np.array([[1.0, 0.6],
                         [0.6, 0.8]])
    
    data = np.random.multivariate_normal(mean_true, cov_true, size=n)
    
    print(f"\nSimulated data: {n} individuals, 2 traits")
    print(f"True mean: {mean_true}")
    print(f"Sample mean: {data.mean(axis=0).round(3)}")
    
    # Demonstrate the core concepts
    print("\n" + "-" * 50)
    print("Section 1.1: Plotting trait space")
    print("-" * 50)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_trait_space_2d(data, ("Body length (mm)", "Wing span (mm)"), 
                        "Population in Trait Space", ax=ax)
    plt.savefig('ch01_trait_space.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Saved: ch01_trait_space.png")
    
    print("\n" + "-" * 50)
    print("Section 1.3: Deviations from the mean")
    print("-" * 50)
    
    mean, deviations = compute_deviations(data)
    print(f"Mean phenotype: {mean.round(3)}")
    print(f"Sum of deviations: {deviations.sum(axis=0).round(10)}")
    print("  → Deviations sum to zero (by definition of mean)")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_deviations_from_mean(data, ("Body length (mm)", "Wing span (mm)"), ax=ax)
    plt.savefig('ch01_deviations.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Saved: ch01_deviations.png")
    
    print("\n" + "-" * 50)
    print("Section 1.4-1.5: Vector operations")
    print("-" * 50)
    demonstrate_vector_operations()
    
    print("\n" + "-" * 50)
    print("Section 1.6: Distances and lengths")
    print("-" * 50)
    
    # Pick two individuals and compute their distance
    i, j = 0, 5
    z_i, z_j = data[i], data[j]
    dist = euclidean_distance(z_i, z_j)
    
    print(f"\nIndividual {i}: {z_i.round(3)}")
    print(f"Individual {j}: {z_j.round(3)}")
    print(f"Euclidean distance: {dist:.4f}")
    print(f"Difference vector: {(z_j - z_i).round(3)}")
    print(f"Length of difference: {vector_length(z_j - z_i):.4f}")
    
    # Verify the connection: distance = length of difference
    assert np.isclose(dist, vector_length(z_j - z_i))
    print("✓ Distance = length of difference vector (verified)")
    
    print("\n" + "=" * 70)
    print("END OF CHAPTER 1")
    print("=" * 70)
    print("""
Key Takeaways:
  1. A phenotype is a POINT in trait space
  2. Differences between phenotypes are VECTORS (arrows)
  3. The mean is the natural reference point; deviations sum to zero
  4. Vectors can be added (head-to-tail) and scaled
  5. Distance is the length of the difference vector
  
Next: Chapter 2 introduces the dot product and angles between vectors.
""")


# =============================================================================
# EXERCISES
# =============================================================================
"""
EXERCISE 1.1: Trait Space Dimensions
------------------------------------
A bird ecologist measures wing length, tarsus length, bill depth, and body mass.
(a) How many dimensions does this trait space have?
(b) Can you visualize this space directly?
(c) What strategies might help you understand patterns in 4D data?

EXERCISE 1.2: The Mean Minimizes Squared Distances
--------------------------------------------------
For data points x = [1, 3, 5, 7, 9]:
(a) Compute the mean x̄
(b) Compute the sum of squared distances to x̄: Σ(xᵢ - x̄)²
(c) Try a different reference point (e.g., 6). Is the sum larger?
(d) Prove that the mean minimizes this sum (hint: use calculus)

EXERCISE 1.3: Vector Arithmetic in Selection
---------------------------------------------
A population's mean phenotype is z̄ = (10, 8).
After selection, the mean of survivors is z̄* = (11, 8.5).
(a) What is the selection differential S = z̄* - z̄?
(b) If heritability is h² = 0.4 for both traits, what is the response R = h²S?
(c) What is the new mean after response?

EXERCISE 1.4: Distance in Different Units
------------------------------------------
Two beetles differ by 2mm in body length and 0.5g in mass.
(a) What is the Euclidean distance?
(b) Convert mass to mg. What is the new distance?
(c) Why is this a problem? (See Chapter 5 for the solution)
"""
