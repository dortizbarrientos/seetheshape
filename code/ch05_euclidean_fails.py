#!/usr/bin/env python3
"""
================================================================================
CHAPTER 5: When Euclidean Distance Fails
================================================================================
Book: "Seeing the Shape: A Geometric Introduction to Multivariate Quantitative 
       Genetics" by Daniel Ortiz-Barrientos

KEY INSIGHT:
    Euclidean distance makes three implicit assumptions:
    1. All traits are on the same scale (SCALE PROBLEM)
    2. Traits are uncorrelated (CORRELATION PROBLEM)  
    3. Distance should correspond to probability (PROBABILITY PROBLEM)
    
    When these assumptions fail, Euclidean distance gives MISLEADING answers
    about how "different" or "unusual" a phenotype is.

THE THREE FAILURES:
    Problem 1 (Scale): 1 mm of body length ≠ 1 mg of body mass
                       Units contaminate our notion of "distance"
    
    Problem 2 (Correlation): When traits are correlated, some directions are
                             common (along the correlation) while others are
                             rare (perpendicular). Euclidean distance is blind
                             to this structure.
    
    Problem 3 (Probability): Points at the same Euclidean distance can have
                             wildly different probabilities under the data
                             distribution. "Equally far" ≠ "equally unusual"

THE SOLUTION PREVIEW:
    Mahalanobis distance (Chapter 6) fixes ALL THREE problems by putting the
    covariance matrix inside the distance formula:
    
        d²_Mah = (z - μ)ᵀ Σ⁻¹ (z - μ)
    
    The inverse covariance matrix Σ⁻¹ accounts for both scale and correlation.

SECTIONS:
    5.1 Example 1: The Problem of Scale
    5.2 Example 2: The Problem of Correlation
    5.3 Example 3: The Probability Perspective
    5.4 A Biological Interlude: Why This Matters for Selection
    5.5 What We Need from a Better Metric
    5.6 A Geometric Preview
    5.7 A Worked Comparison
    5.8 The Connection to Standardisation

Author: Claude (Anthropic) for Daniel Ortiz-Barrientos
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'mathtext.fontset': 'dejavuserif',
    'figure.figsize': (10, 8),
    'figure.dpi': 150,
})


# =============================================================================
# SECTION 5.1: THE PROBLEM OF SCALE
# =============================================================================
#
# Euclidean distance treats all units equally:
#     d = √[(Δx)² + (Δy)²]
#
# But 1 mm of body length is NOT the same "amount" of difference as 1 mg of
# body mass. If body length has SD = 2 mm and body mass has SD = 80 mg, then:
#     - A difference of 2 mm = 1 SD (moderately unusual)
#     - A difference of 80 mg = 1 SD (moderately unusual)
#     - But Euclidean distance sees these as 2 vs 80!
#
# The result: traits with larger numerical values DOMINATE the distance.

def euclidean_distance(x, y):
    """
    Standard Euclidean distance: d = ||x - y||.
    
    This is the Pythagorean formula:
        d = √[(x₁ - y₁)² + (x₂ - y₂)² + ... + (xₚ - yₚ)²]
    
    Parameters
    ----------
    x, y : array-like
        Two points in p-dimensional space.
    
    Returns
    -------
    float
        The Euclidean distance between x and y.
    """
    x, y = np.asarray(x), np.asarray(y)
    return np.sqrt(np.sum((x - y)**2))


def demonstrate_scale_problem():
    """
    Show how Euclidean distance is contaminated by measurement units.
    
    This is Example 1 from Section 5.1 of the book.
    
    We measure two traits on beetles:
        - Elytra length: mean 12 mm, SD 2 mm
        - Body mass: mean 450 mg, SD 80 mg
    
    Two beetles differ from the mean by 1 SD each, but in different traits.
    Euclidean distance incorrectly says they are vastly different in their
    distance from the mean.
    """
    print("\n" + "=" * 70)
    print("SECTION 5.1: THE PROBLEM OF SCALE")
    print("=" * 70)
    
    # Population parameters
    mean_elytra = 12    # mm
    mean_mass = 450     # mg
    sd_elytra = 2       # mm  
    sd_mass = 80        # mg
    
    reference = np.array([mean_elytra, mean_mass])
    
    # Beetle A: 1 SD above mean in elytra length only
    beetle_A = np.array([mean_elytra + sd_elytra, mean_mass])  # (14 mm, 450 mg)
    
    # Beetle B: 1 SD above mean in body mass only
    beetle_B = np.array([mean_elytra, mean_mass + sd_mass])    # (12 mm, 530 mg)
    
    # Compute Euclidean distances
    dist_A = euclidean_distance(beetle_A, reference)
    dist_B = euclidean_distance(beetle_B, reference)
    
    print(f"\nPopulation reference (mean): ({mean_elytra} mm, {mean_mass} mg)")
    print(f"Standard deviations: elytra = {sd_elytra} mm, mass = {sd_mass} mg")
    
    print(f"\nBeetle A: ({beetle_A[0]} mm, {beetle_A[1]} mg)")
    print(f"         → 1 SD above mean in ELYTRA LENGTH")
    print(f"         → Euclidean distance from reference: {dist_A:.1f}")
    
    print(f"\nBeetle B: ({beetle_B[0]} mm, {beetle_B[1]} mg)")
    print(f"         → 1 SD above mean in BODY MASS")
    print(f"         → Euclidean distance from reference: {dist_B:.1f}")
    
    print(f"\n" + "-" * 50)
    print(f"RATIO: Beetle B appears {dist_B/dist_A:.0f}× more different than A!")
    print("-" * 50)
    
    print("\nBut BIOLOGICALLY both beetles are equally unusual!")
    print("Each is exactly 1 standard deviation from the mean.")
    print("\nThe problem: Euclidean distance treats 1 mm = 1 mg.")
    print("             It doesn't know that these are DIFFERENT scales.")
    
    # Standardised distances (z-scores)
    z_A_elytra = (beetle_A[0] - mean_elytra) / sd_elytra
    z_A_mass = (beetle_A[1] - mean_mass) / sd_mass
    z_B_elytra = (beetle_B[0] - mean_elytra) / sd_elytra
    z_B_mass = (beetle_B[1] - mean_mass) / sd_mass
    
    beetle_A_std = np.array([z_A_elytra, z_A_mass])
    beetle_B_std = np.array([z_B_elytra, z_B_mass])
    ref_std = np.array([0, 0])
    
    dist_A_std = euclidean_distance(beetle_A_std, ref_std)
    dist_B_std = euclidean_distance(beetle_B_std, ref_std)
    
    print(f"\nPARTIAL FIX: Standardise each trait to z-scores first:")
    print(f"  Beetle A (standardised): {beetle_A_std} → distance = {dist_A_std:.2f}")
    print(f"  Beetle B (standardised): {beetle_B_std} → distance = {dist_B_std:.2f}")
    print(f"  Now both have distance 1.0 from the mean (in standard units).")
    
    print("\n⚠️  But standardisation alone doesn't fix the CORRELATION problem!")
    print("   (See Section 5.2)")
    
    return {
        'beetle_A': beetle_A,
        'beetle_B': beetle_B,
        'reference': reference,
        'dist_A': dist_A,
        'dist_B': dist_B,
        'ratio': dist_B / dist_A
    }


# =============================================================================
# SECTION 5.2: THE PROBLEM OF CORRELATION
# =============================================================================
#
# Even after standardising each trait, Euclidean distance ignores the
# CORRELATION structure of the data.
#
# When traits are correlated:
#     - Some trait combinations are COMMON (along the correlation)
#     - Some trait combinations are RARE (perpendicular to the correlation)
#
# Euclidean distance treats these equally. It doesn't know that "big body +
# big wings" is typical while "big body + small wings" is unusual.
#
# The data form an ELLIPSE, not a circle. Euclidean distance uses circles.

def demonstrate_correlation_problem():
    """
    Show how Euclidean distance ignores correlation structure.
    
    This is Example 2 from Section 5.2 of the book.
    
    When traits are positively correlated, the data cloud is elongated.
    Two points at the same Euclidean distance from the mean can have
    very different "unusualness" depending on whether they lie along
    or against the correlation.
    """
    print("\n" + "=" * 70)
    print("SECTION 5.2: THE PROBLEM OF CORRELATION")
    print("=" * 70)
    
    # Define a correlated covariance matrix
    # Both traits have variance 1 (standardised), but correlation = 0.8
    rho = 0.8
    Sigma = np.array([[1.0, rho],
                      [rho, 1.0]])
    
    print(f"\nCovariance matrix (standardised traits, correlation ρ = {rho}):")
    print(f"    Σ = [[1.0, {rho}],")
    print(f"         [{rho}, 1.0]]")
    
    # Generate correlated data
    n = 1000
    mean = np.array([0, 0])
    data = np.random.multivariate_normal(mean, Sigma, n)
    
    # Two points at Euclidean distance 1 from the origin
    # Point C: along the correlation (both traits high)
    point_C = np.array([1/np.sqrt(2), 1/np.sqrt(2)])  # 45° direction
    
    # Point D: against the correlation (one high, one low)
    point_D = np.array([1/np.sqrt(2), -1/np.sqrt(2)])  # -45° direction
    
    dist_C = euclidean_distance(point_C, mean)
    dist_D = euclidean_distance(point_D, mean)
    
    print(f"\nTwo points, BOTH at Euclidean distance 1.0 from the origin:")
    print(f"\n  Point C: ({point_C[0]:.3f}, {point_C[1]:.3f})")
    print(f"           → Along the correlation (both traits elevated)")
    print(f"           → Euclidean distance: {dist_C:.3f}")
    
    print(f"\n  Point D: ({point_D[0]:.3f}, {point_D[1]:.3f})")
    print(f"           → Against the correlation (one high, one low)")
    print(f"           → Euclidean distance: {dist_D:.3f}")
    
    print(f"\n" + "-" * 50)
    print("Euclidean distance says C and D are EQUALLY far from the mean.")
    print("-" * 50)
    
    # Count how many data points are "more extreme" than C and D
    # in terms of their position relative to the correlation structure
    distances_from_mean = np.sqrt(np.sum(data**2, axis=1))
    
    # Project onto the directions of C and D
    direction_C = point_C / np.linalg.norm(point_C)
    direction_D = point_D / np.linalg.norm(point_D)
    
    projections_C = data @ direction_C
    projections_D = data @ direction_D
    
    # How many points are further than |1| in each direction?
    extreme_along = np.sum(np.abs(projections_C) > 1) / n * 100
    extreme_against = np.sum(np.abs(projections_D) > 1) / n * 100
    
    print(f"\nBut look at the DATA:")
    print(f"  Points beyond |1| along correlation (C direction): {extreme_along:.1f}%")
    print(f"  Points beyond |1| against correlation (D direction): {extreme_against:.1f}%")
    
    print(f"\nPoint D is much more UNUSUAL because:")
    print(f"  - It lies in a direction where the data has LESS variance")
    print(f"  - Very few individuals have one trait high and the other low")
    print(f"\nEuclidean distance is BLIND to this correlation structure.")
    
    # Visualise
    fig, ax = plt.subplots(figsize=(9, 8))
    
    # Plot data cloud
    ax.scatter(data[:, 0], data[:, 1], alpha=0.3, s=10, c='steelblue', 
               label='Data (n=1000)')
    
    # Plot the Euclidean circle (radius 1)
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    ax.plot(circle_x, circle_y, 'k--', lw=2, label='Euclidean d = 1 (circle)')
    
    # Plot the Mahalanobis ellipse (preview)
    # For Mahalanobis distance 1, the ellipse is defined by z^T Σ^{-1} z = 1
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
    
    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # The ellipse radii are sqrt(eigenvalues) for Mahalanobis d = 1
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    ellipse_x = np.sqrt(eigenvalues[0]) * np.cos(theta) * np.cos(angle) - \
                np.sqrt(eigenvalues[1]) * np.sin(theta) * np.sin(angle)
    ellipse_y = np.sqrt(eigenvalues[0]) * np.cos(theta) * np.sin(angle) + \
                np.sqrt(eigenvalues[1]) * np.sin(theta) * np.cos(angle)
    ax.plot(ellipse_x, ellipse_y, 'r-', lw=2, 
            label='Mahalanobis d = 1 (ellipse)')
    
    # Mark points C and D
    ax.plot(*point_C, 'go', markersize=15, markeredgecolor='black', 
            markeredgewidth=2, label=f'C: along correlation', zorder=5)
    ax.plot(*point_D, 'r^', markersize=15, markeredgecolor='black',
            markeredgewidth=2, label=f'D: against correlation', zorder=5)
    
    # Mark the origin
    ax.plot(0, 0, 'k+', markersize=15, mew=2, label='Mean')
    
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_xlabel('Trait 1 (standardised)')
    ax.set_ylabel('Trait 2 (standardised)')
    ax.set_title(f'The Correlation Problem: Circles vs Ellipses\n(ρ = {rho})')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add annotations
    ax.annotate('C is COMMON\n(along the cloud)', 
                xy=point_C, xytext=(1.8, 1.8),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='green'))
    ax.annotate('D is RARE\n(against the cloud)', 
                xy=point_D, xytext=(1.8, -1.8),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.tight_layout()
    plt.savefig('fig_05_02_correlation_problem.png', dpi=150, bbox_inches='tight')
    plt.savefig('fig_05_02_correlation_problem.pdf', bbox_inches='tight')
    print(f"\nFigure saved: fig_05_02_correlation_problem.png/pdf")
    plt.show()
    
    return {
        'point_C': point_C,
        'point_D': point_D,
        'data': data,
        'Sigma': Sigma
    }


# =============================================================================
# SECTION 5.3: THE PROBABILITY PERSPECTIVE
# =============================================================================
#
# For multivariate normal data, points at the same PROBABILITY should be
# considered "equally unusual." The probability density depends on Mahalanobis
# distance, not Euclidean distance.
#
# Contours of equal probability are ELLIPSES (matching the covariance structure).
# Euclidean distance uses CIRCLES.
#
# Result: points on the same Euclidean circle can have wildly different
# probabilities.

def demonstrate_probability_problem():
    """
    Show how Euclidean distance disconnects from probability.
    
    This is Example 3 from Section 5.3 of the book.
    
    For a bivariate normal distribution, the probability density at a point
    depends on its Mahalanobis distance from the mean, not its Euclidean
    distance. Points at the same Euclidean distance can have very different
    probabilities.
    """
    print("\n" + "=" * 70)
    print("SECTION 5.3: THE PROBABILITY PERSPECTIVE")
    print("=" * 70)
    
    # Define covariance matrix
    rho = 0.8
    Sigma = np.array([[1.0, rho],
                      [rho, 1.0]])
    mean = np.array([0, 0])
    
    print(f"\nBivariate normal with ρ = {rho}")
    print(f"Contours of equal probability density are ELLIPSES, not circles.")
    
    # Points at Euclidean distance 1
    n_points = 8
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    points = np.array([[np.cos(a), np.sin(a)] for a in angles])
    
    # Compute probability densities
    densities = stats.multivariate_normal.pdf(points, mean=mean, cov=Sigma)
    
    # Compute Mahalanobis distances
    Sigma_inv = np.linalg.inv(Sigma)
    mahal_distances = np.array([np.sqrt(p @ Sigma_inv @ p) for p in points])
    
    print(f"\n8 points on the Euclidean circle (radius 1):")
    print(f"{'Angle':>8} {'Point':>20} {'Density':>12} {'d_Mah':>10}")
    print("-" * 55)
    
    for i, (angle, point, density, d_mah) in enumerate(zip(
            np.degrees(angles), points, densities, mahal_distances)):
        print(f"{angle:>8.0f}° ({point[0]:>6.3f}, {point[1]:>6.3f}) "
              f"{density:>12.4f} {d_mah:>10.3f}")
    
    print(f"\n" + "-" * 55)
    print(f"All points have Euclidean distance 1.0 from the mean.")
    print(f"But densities range from {densities.min():.4f} to {densities.max():.4f}")
    print(f"And Mahalanobis distances range from {mahal_distances.min():.3f} "
          f"to {mahal_distances.max():.3f}")
    print("-" * 55)
    
    # Find the most and least probable points
    most_probable_idx = np.argmax(densities)
    least_probable_idx = np.argmin(densities)
    
    print(f"\nMost probable point: angle = {np.degrees(angles[most_probable_idx]):.0f}°")
    print(f"  → Along the correlation (both traits same sign)")
    print(f"  → Density = {densities[most_probable_idx]:.4f}")
    print(f"  → Mahalanobis distance = {mahal_distances[most_probable_idx]:.3f}")
    
    print(f"\nLeast probable point: angle = {np.degrees(angles[least_probable_idx]):.0f}°")
    print(f"  → Against the correlation (traits opposite signs)")
    print(f"  → Density = {densities[least_probable_idx]:.4f}")
    print(f"  → Mahalanobis distance = {mahal_distances[least_probable_idx]:.3f}")
    
    density_ratio = densities[most_probable_idx] / densities[least_probable_idx]
    print(f"\nThe most probable point is {density_ratio:.1f}× more likely!")
    print(f"Yet Euclidean distance treats them as equally far from the mean.")
    
    # Visualise probability density
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Create a grid for density visualization
    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    Z = stats.multivariate_normal.pdf(pos, mean=mean, cov=Sigma)
    
    # Left panel: probability density with circles and ellipses
    ax = axes[0]
    contour = ax.contourf(X, Y, Z, levels=20, cmap='Blues')
    plt.colorbar(contour, ax=ax, label='Probability density')
    
    # Euclidean circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=2, 
            label='Euclidean d = 1')
    
    # Mark the 8 points
    scatter = ax.scatter(points[:, 0], points[:, 1], c=densities, cmap='RdYlGn',
                        s=100, edgecolor='black', zorder=5)
    
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.set_xlabel('Trait 1')
    ax.set_ylabel('Trait 2')
    ax.set_title('Probability Density with Euclidean Circle')
    ax.legend(loc='upper left')
    
    # Right panel: density vs angle
    ax = axes[1]
    ax.bar(np.degrees(angles), densities, width=40, alpha=0.7, color='steelblue',
           edgecolor='black')
    ax.axhline(np.mean(densities), color='red', linestyle='--', 
               label=f'Mean density = {np.mean(densities):.4f}')
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Probability density')
    ax.set_title('Density Varies by Direction\n(all points at Euclidean d = 1)')
    ax.legend()
    ax.set_xticks(np.degrees(angles))
    
    plt.tight_layout()
    plt.savefig('fig_05_03_probability_problem.png', dpi=150, bbox_inches='tight')
    plt.savefig('fig_05_03_probability_problem.pdf', bbox_inches='tight')
    print(f"\nFigure saved: fig_05_03_probability_problem.png/pdf")
    plt.show()
    
    return {
        'points': points,
        'densities': densities,
        'mahal_distances': mahal_distances
    }


# =============================================================================
# SECTION 5.7: A WORKED COMPARISON
# =============================================================================
#
# We compute both Euclidean and Mahalanobis distances for the same points
# and show how they differ.

def worked_comparison():
    """
    Complete worked example comparing Euclidean and Mahalanobis distances.
    
    This is Section 5.7 of the book.
    """
    print("\n" + "=" * 70)
    print("SECTION 5.7: A WORKED COMPARISON")
    print("=" * 70)
    
    # Covariance matrix
    Sigma = np.array([[1.0, 0.8],
                      [0.8, 1.0]])
    
    # Inverse covariance matrix (for Mahalanobis)
    Sigma_inv = np.linalg.inv(Sigma)
    
    print(f"\nCovariance matrix Σ:")
    print(f"    [[{Sigma[0,0]:.1f}, {Sigma[0,1]:.1f}],")
    print(f"     [{Sigma[1,0]:.1f}, {Sigma[1,1]:.1f}]]")
    
    print(f"\nInverse covariance matrix Σ⁻¹:")
    print(f"    [[{Sigma_inv[0,0]:.3f}, {Sigma_inv[0,1]:.3f}],")
    print(f"     [{Sigma_inv[1,0]:.3f}, {Sigma_inv[1,1]:.3f}]]")
    
    # Three points at Euclidean distance 1
    points = {
        'E': np.array([1/np.sqrt(2), 1/np.sqrt(2)]),    # Along correlation
        'F': np.array([1/np.sqrt(2), -1/np.sqrt(2)]),   # Against correlation
        'G': np.array([1.0, 0.0])                        # Along trait 1
    }
    
    print(f"\nThree points (all at Euclidean distance 1 from origin):")
    print(f"\n{'Point':>6} {'Coordinates':>20} {'Direction':>25} "
          f"{'d_Euc':>8} {'d_Mah':>8}")
    print("-" * 75)
    
    results = {}
    for name, p in points.items():
        d_euc = np.linalg.norm(p)
        d_mah = np.sqrt(p @ Sigma_inv @ p)
        
        if name == 'E':
            direction = 'Along correlation'
        elif name == 'F':
            direction = 'Against correlation'
        else:
            direction = 'Along trait 1 only'
        
        print(f"{name:>6} ({p[0]:>7.3f}, {p[1]:>7.3f}) {direction:>25} "
              f"{d_euc:>8.3f} {d_mah:>8.3f}")
        
        results[name] = {'point': p, 'd_euc': d_euc, 'd_mah': d_mah}
    
    print("-" * 75)
    
    # Detailed calculation for point E
    print(f"\n" + "-" * 40)
    print("Detailed calculation for Point E:")
    print("-" * 40)
    p = points['E']
    print(f"  z = ({p[0]:.3f}, {p[1]:.3f})")
    print(f"\n  Euclidean: d² = {p[0]:.3f}² + {p[1]:.3f}² = {p[0]**2 + p[1]**2:.3f}")
    print(f"             d = √{p[0]**2 + p[1]**2:.3f} = {np.sqrt(p[0]**2 + p[1]**2):.3f}")
    
    Sigma_inv_z = Sigma_inv @ p
    print(f"\n  Mahalanobis: Σ⁻¹ z = [{Sigma_inv_z[0]:.3f}, {Sigma_inv_z[1]:.3f}]")
    print(f"               d² = zᵀ Σ⁻¹ z = {p @ Sigma_inv_z:.3f}")
    print(f"               d = √{p @ Sigma_inv_z:.3f} = {np.sqrt(p @ Sigma_inv_z):.3f}")
    
    # Detailed calculation for point F
    print(f"\n" + "-" * 40)
    print("Detailed calculation for Point F:")
    print("-" * 40)
    p = points['F']
    print(f"  z = ({p[0]:.3f}, {p[1]:.3f})")
    print(f"\n  Euclidean: d² = {p[0]:.3f}² + ({p[1]:.3f})² = {p[0]**2 + p[1]**2:.3f}")
    print(f"             d = √{p[0]**2 + p[1]**2:.3f} = {np.sqrt(p[0]**2 + p[1]**2):.3f}")
    
    Sigma_inv_z = Sigma_inv @ p
    print(f"\n  Mahalanobis: Σ⁻¹ z = [{Sigma_inv_z[0]:.3f}, {Sigma_inv_z[1]:.3f}]")
    print(f"               d² = zᵀ Σ⁻¹ z = {p @ Sigma_inv_z:.3f}")
    print(f"               d = √{p @ Sigma_inv_z:.3f} = {np.sqrt(p @ Sigma_inv_z):.3f}")
    
    print(f"\n" + "=" * 70)
    print("KEY INSIGHT:")
    print("=" * 70)
    print(f"  Point E (along correlation):   d_Mah = {results['E']['d_mah']:.3f}")
    print(f"  Point F (against correlation): d_Mah = {results['F']['d_mah']:.3f}")
    print(f"\n  Same Euclidean distance, but F is {results['F']['d_mah']/results['E']['d_mah']:.1f}× "
          f"more unusual in Mahalanobis terms!")
    print(f"  This is because F lies in a direction of LOW variance.")
    
    return results


# =============================================================================
# SECTION 5.8: THE CONNECTION TO STANDARDISATION
# =============================================================================

def demonstrate_standardisation_limit():
    """
    Show that standardisation alone doesn't fix the correlation problem.
    """
    print("\n" + "=" * 70)
    print("SECTION 5.8: THE CONNECTION TO STANDARDISATION")
    print("=" * 70)
    
    # Generate correlated data (traits already have variance 1)
    rho = 0.8
    Sigma = np.array([[1.0, rho],
                      [rho, 1.0]])
    
    n = 1000
    data = np.random.multivariate_normal([0, 0], Sigma, n)
    
    print(f"\nData: n = {n} points from bivariate normal")
    print(f"      Both traits already standardised (variance = 1)")
    print(f"      Correlation = {rho}")
    
    print(f"\nSample statistics:")
    print(f"  Var(X₁) = {np.var(data[:, 0], ddof=1):.3f} (target: 1.0)")
    print(f"  Var(X₂) = {np.var(data[:, 1], ddof=1):.3f} (target: 1.0)")
    print(f"  Cor(X₁, X₂) = {np.corrcoef(data[:, 0], data[:, 1])[0,1]:.3f} (target: {rho})")
    
    # Points along and against correlation
    along = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    against = np.array([1/np.sqrt(2), -1/np.sqrt(2)])
    
    # Both have the same Euclidean distance
    d_euc_along = np.linalg.norm(along)
    d_euc_against = np.linalg.norm(against)
    
    print(f"\nTwo points at Euclidean distance 1:")
    print(f"  Along correlation:   ({along[0]:.3f}, {along[1]:.3f}), d_Euc = {d_euc_along:.3f}")
    print(f"  Against correlation: ({against[0]:.3f}, {against[1]:.3f}), d_Euc = {d_euc_against:.3f}")
    
    # Mahalanobis distances
    Sigma_inv = np.linalg.inv(Sigma)
    d_mah_along = np.sqrt(along @ Sigma_inv @ along)
    d_mah_against = np.sqrt(against @ Sigma_inv @ against)
    
    print(f"\nMahalanobis distances:")
    print(f"  Along correlation:   d_Mah = {d_mah_along:.3f}")
    print(f"  Against correlation: d_Mah = {d_mah_against:.3f}")
    
    print(f"\n" + "-" * 50)
    print("Standardisation (z-scores) fixed the SCALE problem:")
    print("  Both traits have variance 1, so 1 unit = 1 SD for both.")
    print("\nBut standardisation did NOT fix the CORRELATION problem:")
    print(f"  The point against correlation is still {d_mah_against/d_mah_along:.1f}×")
    print(f"  more unusual in Mahalanobis terms!")
    print("-" * 50)
    
    print(f"\nTo fix BOTH problems, we need WHITENING (Chapter 6):")
    print(f"  w = Σ^{{-1/2}} z")
    print(f"\nWhitening uses the full covariance matrix (including off-diagonals)")
    print(f"to transform the ellipse into a circle.")


# =============================================================================
# SUMMARY
# =============================================================================

def summarise_requirements():
    """
    Summarise what properties a good distance metric should have.
    """
    print("\n" + "=" * 70)
    print("SUMMARY: WHAT WE NEED FROM A BETTER METRIC")
    print("=" * 70)
    
    print("""
Three problems with Euclidean distance:

    1. SCALE DEPENDENCE
       - Euclidean distance changes when we change units
       - Fix: Scale by standard deviations
    
    2. IGNORING CORRELATION
       - All directions are treated equally
       - But some directions have more variance than others
       - Fix: Account for the full covariance structure
    
    3. DISCONNECTION FROM PROBABILITY
       - Points at the same Euclidean distance can have different probabilities
       - Fix: Use a metric where "equally far" = "equally probable"

The solution: MAHALANOBIS DISTANCE (Chapter 6)

    d²_Mah = (z - μ)ᵀ Σ⁻¹ (z - μ)

This formula:
    - Uses the INVERSE covariance matrix
    - DOWNWEIGHTS high-variance directions
    - UPWEIGHTS low-variance directions
    - Makes contours of equal distance = contours of equal probability

Geometrically:
    - Euclidean distance uses CIRCLES
    - Mahalanobis distance uses ELLIPSES that match the data shape

In Chapter 6, we derive this formula and show why the inverse appears.
""")


# =============================================================================
# MAIN DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CHAPTER 5: When Euclidean Distance Fails")
    print("Seeing the Shape — Code Companion")
    print("=" * 70)
    
    # Section 5.1: Scale problem
    print("\n" + "-" * 50)
    scale_results = demonstrate_scale_problem()
    
    # Section 5.2: Correlation problem
    print("\n" + "-" * 50)
    correlation_results = demonstrate_correlation_problem()
    
    # Section 5.3: Probability problem
    print("\n" + "-" * 50)
    probability_results = demonstrate_probability_problem()
    
    # Section 5.7: Worked comparison
    print("\n" + "-" * 50)
    comparison_results = worked_comparison()
    
    # Section 5.8: Standardisation limits
    print("\n" + "-" * 50)
    demonstrate_standardisation_limit()
    
    # Summary
    print("\n" + "-" * 50)
    summarise_requirements()
    
    print("\n" + "=" * 70)
    print("END OF CHAPTER 5")
    print("Next: Chapter 6 — Covariance and Mahalanobis Distance")
    print("=" * 70)


# =============================================================================
# EXERCISES
# =============================================================================
"""
EXERCISE 5.1: Scale Dependence by Hand
--------------------------------------
A population has mean height 170 cm (SD = 10 cm) and mean weight 70 kg (SD = 15 kg).
(a) Individual A: height 180 cm, weight 70 kg
    Individual B: height 170 cm, weight 85 kg
    Compute Euclidean distance from each to the mean.
(b) Convert to z-scores. Recompute distances.
(c) Which individual is more unusual biologically?
(d) Convert height to metres. Recompute Euclidean distances in original units.
    What happens?

EXERCISE 5.2: Correlation Geometry
----------------------------------
Traits X and Y have correlation ρ = 0.9.
(a) Sketch the expected shape of a scatterplot.
(b) Mark two points at Euclidean distance 1 from the mean:
    - One along the major axis of the ellipse
    - One along the minor axis
(c) Which is more unusual? Why?
(d) What would change if ρ = -0.9?

EXERCISE 5.3: Probability Calculation
-------------------------------------
For a bivariate normal with Σ = [[1, 0.6], [0.6, 1]]:
(a) What fraction of the data lies within the Euclidean circle of radius 1?
    (Hint: simulate 10000 points)
(b) What fraction lies within the Mahalanobis ellipse of radius 1?
    (Hint: for 2 variables, the theoretical answer is 1 - exp(-1/2) ≈ 39.3%)
(c) Why do these differ?

EXERCISE 5.4: Biological Application
------------------------------------
In a fish population, body length (L) and body depth (D) have:
    μ_L = 20 cm, σ_L = 3 cm
    μ_D = 8 cm, σ_D = 1 cm
    Correlation = 0.75

Three fish are measured:
    Fish A: L = 26 cm, D = 9 cm (both above mean)
    Fish B: L = 26 cm, D = 7 cm (one above, one below)
    Fish C: L = 20 cm, D = 10 cm (D extreme, L average)

(a) Compute Euclidean distance from mean for each fish (in raw units).
(b) Standardise and recompute.
(c) Without calculating Mahalanobis distance, predict which fish is most unusual
    given the correlation structure. Explain your reasoning.

EXERCISE 5.5: When Euclidean Distance Works
-------------------------------------------
Under what conditions would Euclidean distance be appropriate?
(a) List two conditions on the traits.
(b) Give a biological example where these might hold.
(c) Give an example where they clearly fail.

EXERCISE 5.6: Eigenvalues and Variance by Direction
---------------------------------------------------
A covariance matrix Σ has eigenvalues λ₁ = 4 and λ₂ = 1.
(a) What are the variances along the principal axes?
(b) What are the standard deviations along these axes?
(c) A point z is 1 unit along the first eigenvector. How many SDs from the mean?
(d) A point w is 1 unit along the second eigenvector. How many SDs?
(e) Which point is more unusual?
"""
