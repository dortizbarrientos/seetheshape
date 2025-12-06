/home/claude/ch04_distance_variance.py

#!/usr/bin/env python3
"""
================================================================================
CHAPTER 4: Distance and Why We Square It
================================================================================
Book: "Seeing the Shape: A Geometric Introduction to Multivariate Quantitative 
       Genetics" by Daniel Ortiz-Barrientos

This chapter answers a question that seems trivial but is actually profound:

    WHY DO WE SQUARE DISTANCES?

The answer is not "because that's how variance is defined." The answer reveals
deep connections between geometry, algebra, and calculus:

    1. GEOMETRY demands it: Pythagoras requires squares for straight-line distance
    2. ALGEBRA rewards it: Only squared quantities decompose additively
    3. CALCULUS prefers it: Squared functions are smooth and differentiable

The Central Insight:

    VARIANCE IS MEAN SQUARED DISTANCE FROM THE MEAN.
    
    Var(X) = E[(X - μ)²] = E[d²(X, μ)]

This is not a formula to memorize—it is a geometric fact. The variance measures
how spread out a population is by averaging the squared lengths of arrows from
the mean to each individual.

With multiple traits, this leads naturally to the COVARIANCE MATRIX: a single
object that captures variances (how each trait spreads) and covariances (how
traits spread together).

Sections covered:
    §4.1 Why distance matters
    §4.2 One trait: distance on a line
    §4.3 Two traits: the Pythagorean formula
    §4.4 Why do we square?
    §4.5 From individual distances to population spread
    §4.6 Variance in two traits: the covariance matrix appears
    §4.7 A worked example
    §4.8 The covariance matrix as a shape
    §4.9 What squared distance assumes

================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from typing import Tuple, List, Optional

# -----------------------------------------------------------------------------
# SECTION 4.1-4.2: Why Distance Matters / Distance on a Line
# -----------------------------------------------------------------------------
#
# Distance is fundamental to biology:
#   • How different are two species? → Distance in phenotype space
#   • How strong is selection? → Distance between survivors and non-survivors
#   • How heritable is a trait? → Distance between offspring and random individuals
#
# In ONE dimension, distance is simple:
#   d(x, y) = |y - x|
#
# This signed difference tells us magnitude AND direction.
# But for summarizing spread, we need an UNSIGNED measure.
# Two options: absolute value |x| or square x².

def signed_difference(x: float, y: float) -> float:
    """
    The signed difference between two values on a line.
    
    Positive if y > x, negative if y < x.
    This tells us DIRECTION as well as magnitude.
    """
    return y - x


def absolute_distance_1d(x: float, y: float) -> float:
    """
    Absolute distance on a line: |y - x|.
    
    This is a valid metric, but it has problems:
      • Not differentiable at zero (has a "kink")
      • Doesn't decompose nicely for sums of variables
    """
    return np.abs(y - x)


def squared_distance_1d(x: float, y: float) -> float:
    """
    Squared distance on a line: (y - x)².
    
    This is what we actually use, and for good reasons:
      • Differentiable everywhere (smooth)
      • Decomposes additively for independent variables
      • Connects to Pythagoras in higher dimensions
    """
    return (y - x) ** 2


# -----------------------------------------------------------------------------
# SECTION 4.3: Two Traits — The Pythagorean Formula
# -----------------------------------------------------------------------------
#
# In TWO dimensions, Pythagoras tells us the straight-line distance:
#
#   d = √[(Δx)² + (Δy)²]
#
# The SQUARED distance is simpler:
#
#   d² = (Δx)² + (Δy)² = (z_j - z_i)^T (z_j - z_i)
#
# This is not a human choice—it's what "straight line" MEANS in Euclidean space.

def euclidean_distance(z_i: np.ndarray, z_j: np.ndarray) -> float:
    """
    Euclidean (straight-line) distance between two phenotypes.
    
    d(i,j) = ||z_j - z_i|| = √[(z_j - z_i)^T (z_j - z_i)]
    
    This is the length of the arrow from z_i to z_j.
    """
    diff = z_j - z_i
    return np.sqrt(diff @ diff)


def squared_euclidean_distance(z_i: np.ndarray, z_j: np.ndarray) -> float:
    """
    Squared Euclidean distance: d² = (z_j - z_i)^T (z_j - z_i).
    
    Why squared distance is often better than distance:
    
    1. SIMPLER: No square root needed
    2. ADDITIVE: For independent components, variances add
    3. SMOOTH: Differentiable everywhere (no kink at zero)
    4. PYTHAGORAS: This IS the Pythagorean theorem
    """
    diff = z_j - z_i
    return diff @ diff


def demonstrate_pythagoras():
    """
    Visualize how Pythagoras gives us squared distance.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    z_i = np.array([1, 1])
    z_j = np.array([4, 5])
    diff = z_j - z_i
    
    # Plot points
    ax.scatter(*z_i, s=100, c='blue', zorder=5)
    ax.scatter(*z_j, s=100, c='red', zorder=5)
    ax.text(z_i[0] - 0.3, z_i[1], '$\\mathbf{z}_i$', fontsize=14, color='blue')
    ax.text(z_j[0] + 0.1, z_j[1], '$\\mathbf{z}_j$', fontsize=14, color='red')
    
    # Draw the right triangle
    # Horizontal leg
    ax.plot([z_i[0], z_j[0]], [z_i[1], z_i[1]], 'g-', lw=2, label=f'$\\Delta x = {diff[0]}$')
    # Vertical leg
    ax.plot([z_j[0], z_j[0]], [z_i[1], z_j[1]], 'purple', lw=2, label=f'$\\Delta y = {diff[1]}$')
    # Hypotenuse
    ax.plot([z_i[0], z_j[0]], [z_i[1], z_j[1]], 'k-', lw=2.5)
    
    # Right angle marker
    marker_size = 0.2
    ax.plot([z_j[0] - marker_size, z_j[0] - marker_size, z_j[0]], 
            [z_i[1], z_i[1] + marker_size, z_i[1] + marker_size], 'k-', lw=1)
    
    # Labels
    mid_x = (z_i[0] + z_j[0]) / 2
    mid_y = (z_i[1] + z_j[1]) / 2
    
    d_squared = squared_euclidean_distance(z_i, z_j)
    d = euclidean_distance(z_i, z_j)
    
    ax.text(mid_x - 0.8, mid_y + 0.3, 
            f'$d = \\sqrt{{\\Delta x^2 + \\Delta y^2}}$\n$= \\sqrt{{{diff[0]**2} + {diff[1]**2}}} = {d:.2f}$',
            fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Show squared quantities
    ax.text(0.02, 0.98, 
            f'Pythagorean Theorem:\n'
            f'$d^2 = (\\Delta x)^2 + (\\Delta y)^2$\n'
            f'$d^2 = {diff[0]}^2 + {diff[1]}^2 = {diff[0]**2} + {diff[1]**2} = {int(d_squared)}$\n'
            f'$d = \\sqrt{{{int(d_squared)}}} = {d:.2f}$',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    ax.set_xlabel('Trait 1', fontsize=12)
    ax.set_ylabel('Trait 2', fontsize=12)
    ax.set_title('Pythagoras: Distance Requires Squaring', fontsize=14, fontweight='bold')
    
    plt.savefig('ch04_pythagoras.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Saved: ch04_pythagoras.png")


# -----------------------------------------------------------------------------
# SECTION 4.4: Why Do We Square?
# -----------------------------------------------------------------------------
#
# Three deep reasons why squaring is not arbitrary:
#
# REASON 1: GEOMETRY DEMANDS IT
#   The Pythagorean theorem is a property of flat space. If you want
#   straight-line distance, you MUST add squares. Using |Δx| + |Δy|
#   gives "Manhattan distance" (walking along a grid), not straight-line.
#
# REASON 2: ALGEBRA REWARDS IT
#   For INDEPENDENT random variables X and Y:
#       Var(X + Y) = Var(X) + Var(Y)
#   This beautiful additivity ONLY works for squared quantities.
#   Mean absolute deviation does NOT decompose this way.
#
# REASON 3: CALCULUS PREFERS IT
#   The function f(x) = x² is smooth (differentiable everywhere).
#   The function f(x) = |x| has a kink at zero.
#   Least-squares optimization works because the objective is smooth.

def compare_absolute_vs_squared():
    """
    Demonstrate why squared deviations are preferred over absolute deviations.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel 1: The functions themselves
    ax = axes[0]
    x = np.linspace(-3, 3, 200)
    ax.plot(x, np.abs(x), 'b-', lw=2, label='$|x|$ (absolute value)')
    ax.plot(x, x**2, 'r-', lw=2, label='$x^2$ (square)')
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlabel('$x$ (deviation from mean)', fontsize=12)
    ax.set_ylabel('Contribution to spread measure', fontsize=12)
    ax.set_title('(a) Absolute vs. Squared', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Highlight the kink
    ax.annotate('Kink!\n(not differentiable)', xy=(0, 0), xytext=(0.8, 1.5),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='blue'),
                color='blue')
    
    # Panel 2: Derivatives
    ax = axes[1]
    x = np.linspace(-3, 3, 200)
    
    # Derivative of |x| is sign(x), undefined at 0
    deriv_abs = np.sign(x)
    deriv_abs[np.abs(x) < 0.01] = np.nan  # Undefined at 0
    
    # Derivative of x² is 2x, smooth everywhere
    deriv_sq = 2 * x
    
    ax.plot(x, deriv_abs, 'b-', lw=2, label="$d|x|/dx = \\mathrm{sign}(x)$")
    ax.plot(x, deriv_sq, 'r-', lw=2, label="$dx^2/dx = 2x$")
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.scatter([0], [0], c='blue', s=100, marker='o', facecolors='none', 
               linewidths=2, zorder=5)  # Open circle = undefined
    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('Derivative', fontsize=12)
    ax.set_title('(b) Derivatives: Smooth vs. Discontinuous', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-4, 4)
    
    # Panel 3: Optimization landscape
    ax = axes[2]
    
    # Data points
    data = np.array([1, 3, 4, 7, 10])
    
    # Compute sum of absolute deviations and sum of squared deviations
    # as a function of the reference point c
    c_values = np.linspace(0, 12, 200)
    sum_abs = np.array([np.sum(np.abs(data - c)) for c in c_values])
    sum_sq = np.array([np.sum((data - c)**2) for c in c_values])
    
    # Normalize for visualization
    sum_abs_norm = sum_abs / sum_abs.max() * 100
    sum_sq_norm = sum_sq / sum_sq.max() * 100
    
    ax.plot(c_values, sum_abs_norm, 'b-', lw=2, label='Sum of absolute deviations')
    ax.plot(c_values, sum_sq_norm, 'r-', lw=2, label='Sum of squared deviations')
    
    # Mark minima
    mean = np.mean(data)
    median = np.median(data)
    ax.axvline(mean, color='red', linestyle='--', alpha=0.7, label=f'Mean = {mean}')
    ax.axvline(median, color='blue', linestyle='--', alpha=0.7, label=f'Median = {median}')
    
    ax.set_xlabel('Reference point $c$', fontsize=12)
    ax.set_ylabel('Total "distance" (normalized)', fontsize=12)
    ax.set_title('(c) Minimizing: Mean vs. Median', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    ax.text(0.02, 0.98, 
            'Squared: minimized by MEAN\n'
            'Absolute: minimized by MEDIAN',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('ch04_why_square.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Saved: ch04_why_square.png")


def demonstrate_variance_additivity():
    """
    Show that variance (squared) is additive for independent variables,
    but mean absolute deviation is not.
    """
    print("\n" + "=" * 60)
    print("WHY SQUARING? Reason 2: Algebraic Additivity")
    print("=" * 60)
    
    np.random.seed(42)
    n = 10000
    
    # Two independent random variables
    X = np.random.normal(0, 2, n)  # Mean 0, SD 2
    Y = np.random.normal(0, 3, n)  # Mean 0, SD 3
    
    # Their sum
    Z = X + Y
    
    # Variances
    var_X = np.var(X, ddof=1)
    var_Y = np.var(Y, ddof=1)
    var_Z = np.var(Z, ddof=1)
    
    # Mean absolute deviations
    mad_X = np.mean(np.abs(X - np.mean(X)))
    mad_Y = np.mean(np.abs(Y - np.mean(Y)))
    mad_Z = np.mean(np.abs(Z - np.mean(Z)))
    
    print(f"\nX ~ N(0, 2²), Y ~ N(0, 3²), Z = X + Y")
    print(f"(X and Y are independent)")
    
    print(f"\n--- Variance (squared deviations) ---")
    print(f"Var(X) = {var_X:.3f}  (true: 4)")
    print(f"Var(Y) = {var_Y:.3f}  (true: 9)")
    print(f"Var(X) + Var(Y) = {var_X + var_Y:.3f}")
    print(f"Var(Z) = Var(X+Y) = {var_Z:.3f}  (true: 13)")
    print(f"✓ Var(X+Y) = Var(X) + Var(Y)  — ADDITIVITY HOLDS!")
    
    print(f"\n--- Mean Absolute Deviation ---")
    print(f"MAD(X) = {mad_X:.3f}")
    print(f"MAD(Y) = {mad_Y:.3f}")
    print(f"MAD(X) + MAD(Y) = {mad_X + mad_Y:.3f}")
    print(f"MAD(Z) = MAD(X+Y) = {mad_Z:.3f}")
    print(f"✗ MAD(X+Y) ≠ MAD(X) + MAD(Y)  — ADDITIVITY FAILS!")
    
    print(f"\n→ This is why ANOVA works, why we can partition variance into")
    print(f"  genetic and environmental components, why variances of")
    print(f"  independent effects add up. It's all because we SQUARE.")


# -----------------------------------------------------------------------------
# SECTION 4.5: From Individual Distances to Population Spread
# -----------------------------------------------------------------------------
#
# A population is a cloud of points in trait space. We want a single number
# that captures "how spread out" the cloud is.
#
# The natural choice: pick a reference point (the MEAN) and measure the
# average squared distance from each individual to that reference.
#
#   Variance = Mean Squared Distance from the Mean
#
#   Var(X) = (1/n) Σᵢ (xᵢ - x̄)² = (1/n) Σᵢ d²(xᵢ, x̄)
#
# This IS the definition of variance—but it's also a geometric fact about
# how we measure spread using squared distances.

def compute_variance_as_mean_squared_distance(data: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """
    Compute variance as mean squared distance from the mean.
    
    This function makes explicit that variance IS a distance measure.
    
    Parameters
    ----------
    data : np.ndarray
        1D array of observations.
    
    Returns
    -------
    mean : float
        The sample mean.
    variance : float
        The sample variance (using n-1 denominator).
    squared_distances : np.ndarray
        Array of squared distances from each point to the mean.
    """
    mean = np.mean(data)
    deviations = data - mean
    squared_distances = deviations ** 2  # d²(xᵢ, x̄)
    variance = np.sum(squared_distances) / (len(data) - 1)  # Bessel's correction
    
    return mean, variance, squared_distances


def visualize_variance_as_distance(data: np.ndarray, ax: plt.Axes = None) -> plt.Axes:
    """
    Visualize variance as the mean squared distance from the mean.
    
    Each observation is shown with an arrow to the mean, and the squared
    lengths of these arrows are averaged to give the variance.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    
    mean, variance, sq_dists = compute_variance_as_mean_squared_distance(data)
    n = len(data)
    
    # Plot number line
    ax.axhline(0, color='black', lw=1)
    
    # Plot mean
    ax.scatter([mean], [0], s=200, c='red', marker='D', zorder=5, label=f'Mean = {mean:.2f}')
    
    # Plot each observation and its arrow to the mean
    y_positions = np.linspace(-0.3, 0.3, n)  # Stagger vertically for visibility
    
    for i, (x, y_pos, sq_d) in enumerate(zip(data, y_positions, sq_dists)):
        ax.scatter([x], [y_pos], s=60, c='blue', zorder=4)
        ax.annotate('', xy=(mean, 0), xytext=(x, y_pos),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1, alpha=0.6))
        # Label squared distance
        mid_x = (x + mean) / 2
        ax.text(mid_x, y_pos + 0.08, f'{sq_d:.1f}', fontsize=8, ha='center', alpha=0.7)
    
    # Summary
    ax.text(0.02, 0.95, 
            f'$n = {n}$\n'
            f'$\\bar{{x}} = {mean:.2f}$\n'
            f'$\\sum d_i^2 = {np.sum(sq_dists):.2f}$\n'
            f'$s^2 = \\frac{{\\sum d_i^2}}{{n-1}} = {variance:.2f}$',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlim(data.min() - 1, data.max() + 1)
    ax.set_ylim(-0.6, 0.8)
    ax.set_xlabel('Trait value', fontsize=12)
    ax.set_title('Variance = Mean Squared Distance from the Mean', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    
    # Hide y-axis (this is a 1D visualization)
    ax.set_yticks([])
    
    return ax


# -----------------------------------------------------------------------------
# SECTION 4.6: Variance in Two Traits — The Covariance Matrix Appears
# -----------------------------------------------------------------------------
#
# With TWO traits, each individual's deviation from the mean is a VECTOR:
#
#   dᵢ = zᵢ - z̄ = [xᵢ - x̄]
#                  [yᵢ - ȳ]
#
# The squared distance from individual i to the mean is:
#
#   ||dᵢ||² = (xᵢ - x̄)² + (yᵢ - ȳ)²
#
# But this single number discards information about the SHAPE of the cloud.
# Is it elongated? In which direction?
#
# To capture shape, we compute not just ||dᵢ||² but the OUTER PRODUCT dᵢ dᵢᵀ:
#
#   dᵢ dᵢᵀ = [(xᵢ - x̄)²           (xᵢ - x̄)(yᵢ - ȳ)]
#            [(xᵢ - x̄)(yᵢ - ȳ)    (yᵢ - ȳ)²        ]
#
# Average these matrices over all individuals → COVARIANCE MATRIX:
#
#   S = (1/(n-1)) Σᵢ dᵢ dᵢᵀ = [Var(X)     Cov(X,Y)]
#                              [Cov(X,Y)   Var(Y) ]

def compute_covariance_matrix(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the sample covariance matrix as an average of outer products.
    
    This function shows explicitly how the covariance matrix arises from
    averaging dᵢ dᵢᵀ over all individuals.
    
    Parameters
    ----------
    data : np.ndarray
        An (n × p) array where each row is an individual's phenotype.
    
    Returns
    -------
    mean : np.ndarray
        The mean phenotype (length p).
    cov_matrix : np.ndarray
        The (p × p) sample covariance matrix.
    deviations : np.ndarray
        The (n × p) array of deviations from the mean.
    
    Mathematical Note
    -----------------
    The covariance matrix can be written as:
    
        S = (1/(n-1)) Σᵢ dᵢ dᵢᵀ = (1/(n-1)) Dᵀ D
    
    where D is the (n × p) matrix of deviations (one row per individual).
    """
    n, p = data.shape
    mean = np.mean(data, axis=0)
    deviations = data - mean  # (n × p) matrix D
    
    # Method 1: Explicit sum of outer products
    cov_matrix = np.zeros((p, p))
    for i in range(n):
        d_i = deviations[i]  # Deviation vector for individual i
        outer_product = np.outer(d_i, d_i)  # dᵢ dᵢᵀ
        cov_matrix += outer_product
    cov_matrix /= (n - 1)  # Bessel's correction
    
    # Method 2 (equivalent, more efficient): S = Dᵀ D / (n-1)
    # cov_matrix = (deviations.T @ deviations) / (n - 1)
    
    return mean, cov_matrix, deviations


def visualize_covariance_geometry(data: np.ndarray, ax: plt.Axes = None) -> plt.Axes:
    """
    Visualize the covariance matrix as a shape (ellipse) in trait space.
    
    Shows:
    1. The cloud of individuals
    2. Arrows from each individual to the mean
    3. The covariance ellipse (1 SD contour)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    mean, cov_matrix, deviations = compute_covariance_matrix(data)
    n = len(data)
    
    # Plot individuals
    ax.scatter(data[:, 0], data[:, 1], s=50, c='blue', alpha=0.6, label='Individuals')
    
    # Plot mean
    ax.scatter(*mean, s=200, c='red', marker='D', zorder=5, 
               label=f'Mean = ({mean[0]:.2f}, {mean[1]:.2f})')
    
    # Draw arrows from mean to each individual (deviation vectors)
    for i in range(min(n, 30)):  # Limit arrows for clarity
        ax.annotate('', xy=data[i], xytext=mean,
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.5, alpha=0.4))
    
    # Draw covariance ellipse
    # Eigendecomposition gives axes and orientation
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Ellipse parameters
    # Eigenvalues are variances along principal axes
    # Semi-axis lengths are √eigenvalue (standard deviations)
    width = 2 * np.sqrt(eigenvalues[1])  # 2 * SD for larger eigenvalue
    height = 2 * np.sqrt(eigenvalues[0])  # 2 * SD for smaller eigenvalue
    
    # Angle of rotation (eigenvector direction)
    angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
    
    # Draw ellipse
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                      fill=False, color='green', linewidth=2.5,
                      label='Covariance ellipse (1 SD)')
    ax.add_patch(ellipse)
    
    # Draw principal axes
    for i, (eigenvalue, eigenvector) in enumerate(zip(eigenvalues, eigenvectors.T)):
        # Scale by sqrt(eigenvalue) = SD
        endpoint = mean + np.sqrt(eigenvalue) * eigenvector
        ax.annotate('', xy=endpoint, xytext=mean,
                    arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2))
        ax.text(endpoint[0] + 0.1, endpoint[1] + 0.1, 
                f'$\\sqrt{{\\lambda_{i+1}}} = {np.sqrt(eigenvalue):.2f}$',
                fontsize=10, color='darkgreen')
    
    # Display covariance matrix
    ax.text(0.02, 0.98, 
            f'Covariance Matrix:\n'
            f'$S = \\begin{{bmatrix}} {cov_matrix[0,0]:.2f} & {cov_matrix[0,1]:.2f} \\\\ '
            f'{cov_matrix[1,0]:.2f} & {cov_matrix[1,1]:.2f} \\end{{bmatrix}}$\n\n'
            f'Eigenvalues: {eigenvalues[1]:.2f}, {eigenvalues[0]:.2f}\n'
            f'(variances along principal axes)',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('Trait 1', fontsize=12)
    ax.set_ylabel('Trait 2', fontsize=12)
    ax.set_title('The Covariance Matrix as a Shape', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    return ax


# -----------------------------------------------------------------------------
# SECTION 4.7: A Worked Example
# -----------------------------------------------------------------------------

def worked_example():
    """
    Complete worked example computing covariance matrix by hand.
    """
    print("\n" + "=" * 60)
    print("WORKED EXAMPLE: Computing the Covariance Matrix")
    print("=" * 60)
    
    # Data from the book
    data = np.array([
        [2, 3],
        [4, 5],
        [3, 4],
        [5, 7],
        [6, 6]
    ])
    
    n = len(data)
    print(f"\nData ({n} individuals, 2 traits):")
    print("  Individual   Trait 1 (x)   Trait 2 (y)")
    for i, (x, y) in enumerate(data):
        print(f"      {i+1}           {x}            {y}")
    
    # Step 1: Compute means
    mean_x = np.mean(data[:, 0])
    mean_y = np.mean(data[:, 1])
    print(f"\nStep 1: Compute means")
    print(f"  x̄ = {mean_x}")
    print(f"  ȳ = {mean_y}")
    
    # Step 2: Compute deviations
    print(f"\nStep 2: Compute deviations from mean")
    print("  Individual   (xᵢ - x̄)   (yᵢ - ȳ)")
    deviations = data - np.array([mean_x, mean_y])
    for i, (dx, dy) in enumerate(deviations):
        print(f"      {i+1}          {dx:+.0f}         {dy:+.0f}")
    
    # Step 3: Compute sums of squares and cross-products
    print(f"\nStep 3: Compute squared deviations and cross-products")
    print("  Individual   (xᵢ-x̄)²   (yᵢ-ȳ)²   (xᵢ-x̄)(yᵢ-ȳ)")
    
    ss_x = 0
    ss_y = 0
    ss_xy = 0
    
    for i, (dx, dy) in enumerate(deviations):
        dx2 = dx ** 2
        dy2 = dy ** 2
        dxdy = dx * dy
        ss_x += dx2
        ss_y += dy2
        ss_xy += dxdy
        print(f"      {i+1}          {dx2:.0f}         {dy2:.0f}           {dxdy:+.0f}")
    
    print(f"  ─────────────────────────────────────────────")
    print(f"    Sum:       {ss_x:.0f}        {ss_y:.0f}           {ss_xy:+.0f}")
    
    # Step 4: Divide by (n-1)
    print(f"\nStep 4: Divide by (n-1) = {n-1}")
    var_x = ss_x / (n - 1)
    var_y = ss_y / (n - 1)
    cov_xy = ss_xy / (n - 1)
    
    print(f"  Var(X) = {ss_x}/{n-1} = {var_x:.2f}")
    print(f"  Var(Y) = {ss_y}/{n-1} = {var_y:.2f}")
    print(f"  Cov(X,Y) = {ss_xy}/{n-1} = {cov_xy:.2f}")
    
    # Step 5: Assemble covariance matrix
    cov_matrix = np.array([[var_x, cov_xy],
                           [cov_xy, var_y]])
    
    print(f"\nStep 5: Assemble the covariance matrix")
    print(f"  S = [Var(X)    Cov(X,Y)]")
    print(f"      [Cov(X,Y)  Var(Y)  ]")
    print(f"\n  S = [{var_x:.2f}   {cov_xy:.2f}]")
    print(f"      [{cov_xy:.2f}   {var_y:.2f}]")
    
    # Verify with numpy
    cov_numpy = np.cov(data.T)
    print(f"\nVerification with numpy.cov:")
    print(f"  S = [{cov_numpy[0,0]:.2f}   {cov_numpy[0,1]:.2f}]")
    print(f"      [{cov_numpy[1,0]:.2f}   {cov_numpy[1,1]:.2f}]")
    
    # Interpretation
    print(f"\nInterpretation:")
    print(f"  • Var(X) = {var_x:.2f}: Trait 1 has variance {var_x:.2f}")
    print(f"  • Var(Y) = {var_y:.2f}: Trait 2 has variance {var_y:.2f}")
    print(f"  • Cov(X,Y) = {cov_xy:.2f} > 0: Traits INCREASE TOGETHER")
    print(f"  • The cloud is elongated along the diagonal (positive correlation)")
    
    # Correlation
    corr = cov_xy / np.sqrt(var_x * var_y)
    print(f"\n  Correlation r = Cov(X,Y) / √[Var(X)Var(Y)]")
    print(f"              r = {cov_xy:.2f} / √({var_x:.2f} × {var_y:.2f})")
    print(f"              r = {corr:.3f}")
    
    return data, cov_matrix


# -----------------------------------------------------------------------------
# SECTION 4.8: The Covariance Matrix as a Shape
# -----------------------------------------------------------------------------
#
# The covariance matrix is SYMMETRIC: Cov(X,Y) = Cov(Y,X).
# From Chapter 3, we know symmetric matrices describe ELLIPSES.
#
# The eigenvalues of S are the variances along the principal axes.
# The eigenvectors are the directions of those axes.
#
# If eigenvalues are equal → circle (no preferred direction)
# If eigenvalues differ → ellipse (elongated in direction of larger eigenvalue)

def demonstrate_covariance_shapes():
    """
    Show how different covariance matrices produce different shapes.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    np.random.seed(42)
    n = 200
    
    scenarios = [
        ("Uncorrelated\n(r = 0)", np.array([[1.0, 0.0], [0.0, 1.0]])),
        ("Positive Correlation\n(r = 0.8)", np.array([[1.0, 0.8], [0.8, 1.0]])),
        ("Negative Correlation\n(r = -0.8)", np.array([[1.0, -0.8], [-0.8, 1.0]]))
    ]
    
    for ax, (title, cov) in zip(axes, scenarios):
        # Generate data
        data = np.random.multivariate_normal([0, 0], cov, n)
        
        # Plot points
        ax.scatter(data[:, 0], data[:, 1], s=20, alpha=0.5)
        
        # Draw covariance ellipse
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
        
        for k in [1, 2]:  # 1 and 2 SD ellipses
            ellipse = Ellipse(xy=(0, 0), 
                             width=2*k*np.sqrt(eigenvalues[1]), 
                             height=2*k*np.sqrt(eigenvalues[0]),
                             angle=angle, fill=False, 
                             color='red' if k == 1 else 'orange',
                             linewidth=2, linestyle='-' if k == 1 else '--')
            ax.add_patch(ellipse)
        
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.axhline(0, color='gray', lw=0.5)
        ax.axvline(0, color='gray', lw=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Trait 1')
        ax.set_ylabel('Trait 2')
        ax.set_title(title, fontsize=12, fontweight='bold')
    
    plt.suptitle('Covariance Matrices as Shapes: The Ellipse Reveals Correlation', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('ch04_covariance_shapes.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Saved: ch04_covariance_shapes.png")


# -----------------------------------------------------------------------------
# SECTION 4.9: What Squared Distance Assumes
# -----------------------------------------------------------------------------
#
# Euclidean distance (and therefore variance) makes implicit assumptions:
#
# 1. ALL TRAITS ARE ON THE SAME SCALE
#    1 mm of body length = 1 mg of body mass? Usually not!
#
# 2. TRAITS ARE UNCORRELATED
#    A deviation in the "both traits high" direction is treated the same
#    as a deviation in the "one high, one low" direction. But if traits
#    are correlated, these have very different meanings!
#
# These assumptions often FAIL. Chapter 5 will show examples.
# Chapter 6 will introduce Mahalanobis distance as the FIX.

def demonstrate_euclidean_assumptions():
    """
    Show the implicit assumptions of Euclidean distance.
    """
    print("\n" + "=" * 60)
    print("WHAT EUCLIDEAN DISTANCE ASSUMES (and why it often fails)")
    print("=" * 60)
    
    # Example 1: Scale problem
    print("\n--- Assumption 1: All traits on the same scale ---")
    
    beetle_A = np.array([10, 500])  # 10 mm body, 500 mg mass
    beetle_B = np.array([12, 500])  # 12 mm body, 500 mg mass
    beetle_C = np.array([10, 600])  # 10 mm body, 600 mg mass
    
    dist_AB = euclidean_distance(beetle_A, beetle_B)
    dist_AC = euclidean_distance(beetle_A, beetle_C)
    
    print(f"\nBeetle A: 10 mm body, 500 mg mass")
    print(f"Beetle B: 12 mm body, 500 mg mass  (differs by 2 mm)")
    print(f"Beetle C: 10 mm body, 600 mg mass  (differs by 100 mg)")
    print(f"\nEuclidean distances:")
    print(f"  d(A, B) = {dist_AB:.1f}")
    print(f"  d(A, C) = {dist_AC:.1f}")
    print(f"\n→ Beetle C appears 50× more different than B!")
    print(f"  But 2 mm might be 1 SD, while 100 mg might also be 1 SD.")
    print(f"  The SCALE of the trait matters, but Euclidean distance ignores it.")
    
    # Example 2: Correlation problem
    print("\n--- Assumption 2: Traits are uncorrelated ---")
    print("\nImagine body length and wing span are positively correlated (r = 0.9).")
    print("Most individuals are either 'big' (both high) or 'small' (both low).")
    print("\nTwo unusual individuals:")
    print("  D: big body, big wings (along the correlation)")
    print("  E: big body, SMALL wings (against the correlation)")
    print("\nEuclidean distance says D and E are equally 'far' from the mean.")
    print("But E is much more UNUSUAL given the correlation structure!")
    print("\n→ Euclidean distance treats all directions equally.")
    print("  It doesn't know that some directions are rare (perpendicular to correlation)")
    print("  while others are common (along the correlation).")
    
    print("\n" + "=" * 60)
    print("PREVIEW: Mahalanobis distance (Chapter 6) will fix both problems")
    print("by putting the covariance matrix INSIDE the distance formula.")
    print("=" * 60)


# -----------------------------------------------------------------------------
# MAIN DEMONSTRATION
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CHAPTER 4: Distance and Why We Square It")
    print("Seeing the Shape — Code Companion")
    print("=" * 70)
    
    # Section 4.3: Pythagoras
    print("\n" + "-" * 50)
    print("Section 4.3: The Pythagorean formula")
    print("-" * 50)
    demonstrate_pythagoras()
    
    # Section 4.4: Why square?
    print("\n" + "-" * 50)
    print("Section 4.4: Why do we square?")
    print("-" * 50)
    compare_absolute_vs_squared()
    demonstrate_variance_additivity()
    
    # Section 4.5: Variance as distance
    print("\n" + "-" * 50)
    print("Section 4.5: Variance = Mean Squared Distance from Mean")
    print("-" * 50)
    
    data_1d = np.array([2, 4, 4, 4, 5, 5, 7, 9])
    mean, var, sq_dists = compute_variance_as_mean_squared_distance(data_1d)
    
    print(f"\nData: {data_1d}")
    print(f"Mean: {mean}")
    print(f"Squared distances from mean: {sq_dists}")
    print(f"Sum of squared distances: {sq_dists.sum():.2f}")
    print(f"Variance = sum / (n-1) = {sq_dists.sum():.2f} / {len(data_1d)-1} = {var:.2f}")
    
    fig, ax = plt.subplots(figsize=(12, 4))
    visualize_variance_as_distance(data_1d, ax=ax)
    plt.savefig('ch04_variance_as_distance.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Saved: ch04_variance_as_distance.png")
    
    # Section 4.6: Covariance matrix
    print("\n" + "-" * 50)
    print("Section 4.6: The covariance matrix")
    print("-" * 50)
    
    np.random.seed(42)
    n = 100
    cov_true = np.array([[1.0, 0.7], [0.7, 1.2]])
    data_2d = np.random.multivariate_normal([5, 4], cov_true, n)
    
    mean, cov_matrix, devs = compute_covariance_matrix(data_2d)
    print(f"\nGenerated {n} individuals with true covariance:")
    print(f"  [{cov_true[0,0]:.1f}  {cov_true[0,1]:.1f}]")
    print(f"  [{cov_true[1,0]:.1f}  {cov_true[1,1]:.1f}]")
    print(f"\nEstimated covariance matrix:")
    print(f"  [{cov_matrix[0,0]:.3f}  {cov_matrix[0,1]:.3f}]")
    print(f"  [{cov_matrix[1,0]:.3f}  {cov_matrix[1,1]:.3f}]")
    
    fig, ax = plt.subplots(figsize=(10, 10))
    visualize_covariance_geometry(data_2d, ax=ax)
    plt.savefig('ch04_covariance_geometry.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Saved: ch04_covariance_geometry.png")
    
    # Section 4.7: Worked example
    print("\n" + "-" * 50)
    print("Section 4.7: Worked example")
    print("-" * 50)
    data_worked, cov_worked = worked_example()
    
    # Section 4.8: Different shapes
    print("\n" + "-" * 50)
    print("Section 4.8: The covariance matrix as a shape")
    print("-" * 50)
    demonstrate_covariance_shapes()
    
    # Section 4.9: Assumptions
    print("\n" + "-" * 50)
    print("Section 4.9: What Euclidean distance assumes")
    print("-" * 50)
    demonstrate_euclidean_assumptions()
    
    print("\n" + "=" * 70)
    print("END OF CHAPTER 4")
    print("=" * 70)
    print("""
Key Takeaways:
  1. Pythagoras REQUIRES squaring for straight-line distance
  2. Squaring is rewarded: Var(X+Y) = Var(X) + Var(Y) for independent X, Y
  3. Squaring is smooth: the mean minimizes sum of squared deviations
  4. VARIANCE IS MEAN SQUARED DISTANCE FROM THE MEAN
  5. The COVARIANCE MATRIX arises from averaging outer products dᵢ dᵢᵀ
  6. Covariance matrices are SHAPES: ellipses that show spread + correlation
  7. Euclidean distance assumes equal scales and no correlation (often wrong!)
  
The Pattern That Will Keep Returning:
    d² = (z - μ)ᵀ (z - μ)           ← Euclidean (no matrix)
    d² = (z - μ)ᵀ Σ⁻¹ (z - μ)       ← Mahalanobis (Chapter 6)
    
The matrix in the middle changes everything.
  
Next: Chapter 5 shows concrete examples where Euclidean distance fails.
""")


# =============================================================================
# EXERCISES
# =============================================================================
"""
EXERCISE 4.1: Pythagoras by Hand
--------------------------------
Two individuals: z₁ = (2, 3) and z₂ = (5, 7).
(a) Compute Δx and Δy.
(b) Compute the squared distance d² = (Δx)² + (Δy)².
(c) Compute the distance d = √d².
(d) Verify: d² = (z₂ - z₁)ᵀ(z₂ - z₁).

EXERCISE 4.2: Variance as Distance
----------------------------------
Data: X = {1, 2, 3, 4, 5}.
(a) Compute the mean x̄.
(b) Compute each deviation (xᵢ - x̄).
(c) Compute each squared deviation (xᵢ - x̄)².
(d) Compute variance as mean squared distance: s² = Σ(xᵢ - x̄)² / (n-1).
(e) Verify with np.var(X, ddof=1).

EXERCISE 4.3: Why Not Absolute Value?
-------------------------------------
Consider two independent random variables X ~ N(0, 4) and Y ~ N(0, 9).
(a) What is Var(X + Y)? (Hint: variances add for independent variables)
(b) Simulate 10000 samples of X, Y, and Z = X + Y.
(c) Compute sample variances. Does Var(Z) ≈ Var(X) + Var(Y)?
(d) Compute mean absolute deviations. Does MAD(Z) ≈ MAD(X) + MAD(Y)?
(e) Why does additivity fail for MAD?

EXERCISE 4.4: Computing Covariance Matrix
-----------------------------------------
Data (4 individuals, 2 traits):
    [[1, 2], [2, 4], [3, 5], [4, 5]]
(a) Compute the mean of each trait.
(b) Compute the deviation matrix D (subtract mean from each row).
(c) Compute the outer product dᵢ dᵢᵀ for each individual.
(d) Average these to get the covariance matrix S.
(e) Verify with np.cov.

EXERCISE 4.5: Interpreting Covariance
-------------------------------------
Given covariance matrix S = [[4, 3], [3, 9]]:
(a) What is Var(X)? Var(Y)?
(b) What is Cov(X, Y)?
(c) Compute the correlation r = Cov(X,Y) / √[Var(X)Var(Y)].
(d) Is the cloud elongated toward "both traits high" or "one high, one low"?

EXERCISE 4.6: Scale Problems (Preview of Chapter 5)
---------------------------------------------------
Two measurements: body length in mm and body mass in g.
Population means: μ = (15 mm, 8 g), with SDs: σ = (2 mm, 2 g).
(a) Individual A: (17 mm, 10 g). Compute Euclidean distance from mean.
(b) Individual B: (15 mm, 12 g). Compute Euclidean distance from mean.
(c) Both are 2 units from the mean in raw units. Are they equally unusual?
(d) Convert to standard units (z-scores). Who is more unusual?
"""