#!/usr/bin/env python3
"""
================================================================================
CHAPTER 6: Covariance and Mahalanobis Distance
================================================================================
Book: "Seeing the Shape: A Geometric Introduction to Multivariate Quantitative 
       Genetics" by Daniel Ortiz-Barrientos

KEY INSIGHT:
    Mahalanobis distance fixes ALL THREE problems from Chapter 5 by putting
    the INVERSE covariance matrix between the vectors:
    
        d²_Mah = (z - μ)ᵀ Σ⁻¹ (z - μ)
    
    The inverse covariance matrix Σ⁻¹:
    - DOWNWEIGHTS high-variance directions (they're common)
    - UPWEIGHTS low-variance directions (they're rare)
    - Makes "equally far" = "equally probable"

WHY THE INVERSE?
    Consider what we want:
    - Large variance → direction is common → deviation should count LESS
    - Small variance → direction is rare → deviation should count MORE
    
    The covariance Σ has LARGE eigenvalues in high-variance directions.
    The inverse Σ⁻¹ has SMALL eigenvalues (= 1/λ) in those same directions.
    
    So multiplying by Σ⁻¹ naturally downweights high-variance directions!

THE WHITENING INTERPRETATION:
    Mahalanobis distance can be understood as ordinary Euclidean distance
    AFTER a transformation that makes the covariance matrix = I:
    
        w = Σ^{-1/2} (z - μ)     ← "whitened" coordinates
        d²_Mah = wᵀw = ||w||²    ← just Euclidean distance in whitened space!
    
    In whitened space, the ellipse becomes a circle, and Euclidean works.

SECTIONS:
    6.1 The Key Insight: A Matrix Between the Vectors
    6.2 The Covariance Matrix and Its Inverse
    6.3 A One-Dimensional Sanity Check
    6.4 Geometry: How the Inverse Reshapes Space
    6.5 The Formula in Components
    6.6 Mahalanobis Distance and Probability
    6.7 A Worked Example
    6.8 The Mahalanobis Distance as a Transformation
    6.9 Mahalanobis Distance Between Two Points
    6.10 Connection to Discriminant Analysis
    6.11 Biological Interpretation
    6.12 What Mahalanobis Distance Requires

Author: Claude (Anthropic) for Daniel Ortiz-Barrientos
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import sqrtm

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
# SECTION 6.1: THE KEY INSIGHT — A MATRIX BETWEEN THE VECTORS
# =============================================================================
#
# Recall Euclidean distance:
#     d²_Euc = (z - μ)ᵀ (z - μ) = (z - μ)ᵀ I (z - μ)
#
# The identity matrix I is hiding there! What if we replace it with Σ⁻¹?
#
#     d²_Mah = (z - μ)ᵀ Σ⁻¹ (z - μ)
#
# This is the Mahalanobis distance (squared). The matrix in the middle
# changes the SHAPE of the "unit ball" from a sphere to an ellipsoid.

def euclidean_distance_squared(z, mu):
    """
    Squared Euclidean distance: d² = (z - μ)ᵀ (z - μ).
    
    This is the same as (z - μ)ᵀ I (z - μ) where I is the identity.
    """
    diff = np.asarray(z) - np.asarray(mu)
    return diff @ diff


def mahalanobis_distance_squared(z, mu, Sigma):
    """
    Squared Mahalanobis distance: d² = (z - μ)ᵀ Σ⁻¹ (z - μ).
    
    This puts the INVERSE covariance matrix between the vectors.
    
    Parameters
    ----------
    z : array-like
        The point whose distance we want.
    mu : array-like
        The reference point (typically the mean).
    Sigma : array-like
        The covariance matrix.
    
    Returns
    -------
    float
        The squared Mahalanobis distance.
    
    Notes
    -----
    The inverse Σ⁻¹ appears because we want to:
        - Downweight high-variance directions (common)
        - Upweight low-variance directions (rare)
    
    Eigenvalues of Σ⁻¹ are 1/λᵢ where λᵢ are eigenvalues of Σ.
    So large variances become small weights, and vice versa.
    """
    diff = np.asarray(z) - np.asarray(mu)
    Sigma_inv = np.linalg.inv(Sigma)
    return diff @ Sigma_inv @ diff


def mahalanobis_distance(z, mu, Sigma):
    """
    Mahalanobis distance: d = √[(z - μ)ᵀ Σ⁻¹ (z - μ)].
    """
    return np.sqrt(mahalanobis_distance_squared(z, mu, Sigma))


def demonstrate_matrix_insertion():
    """
    Show the key insight: inserting a matrix changes the distance geometry.
    """
    print("\n" + "=" * 70)
    print("SECTION 6.1: THE KEY INSIGHT — A MATRIX BETWEEN THE VECTORS")
    print("=" * 70)
    
    print("""
Euclidean distance (hidden identity matrix):
    d²_Euc = (z - μ)ᵀ I (z - μ)    ← unit ball is a CIRCLE

Mahalanobis distance (inverse covariance matrix):
    d²_Mah = (z - μ)ᵀ Σ⁻¹ (z - μ)  ← unit ball is an ELLIPSE
    
The matrix in the middle determines the SHAPE of equidistant contours.
""")
    
    # Define covariance matrix
    Sigma = np.array([[1.0, 0.8],
                      [0.8, 1.0]])
    mu = np.array([0, 0])
    
    # A point along the correlation
    z_along = np.array([1, 1]) / np.sqrt(2)  # Euclidean distance 1
    
    # A point against the correlation
    z_against = np.array([1, -1]) / np.sqrt(2)  # Euclidean distance 1
    
    print("Example: Two points, both at Euclidean distance 1 from origin")
    print(f"  Covariance matrix: Σ = [[1.0, 0.8], [0.8, 1.0]]")
    print(f"  Point along correlation:   z = ({z_along[0]:.3f}, {z_along[1]:.3f})")
    print(f"  Point against correlation: z = ({z_against[0]:.3f}, {z_against[1]:.3f})")
    
    d_euc_along = np.sqrt(euclidean_distance_squared(z_along, mu))
    d_euc_against = np.sqrt(euclidean_distance_squared(z_against, mu))
    d_mah_along = mahalanobis_distance(z_along, mu, Sigma)
    d_mah_against = mahalanobis_distance(z_against, mu, Sigma)
    
    print(f"\n{'':>30} {'Euclidean':>12} {'Mahalanobis':>12}")
    print("-" * 55)
    print(f"{'Along correlation':>30} {d_euc_along:>12.3f} {d_mah_along:>12.3f}")
    print(f"{'Against correlation':>30} {d_euc_against:>12.3f} {d_mah_against:>12.3f}")
    print("-" * 55)
    
    print(f"\nThe inverse covariance matrix Σ⁻¹ makes the difference!")
    print(f"It penalises low-variance directions more heavily.")


# =============================================================================
# SECTION 6.2: THE COVARIANCE MATRIX AND ITS INVERSE
# =============================================================================

def demonstrate_inverse_covariance():
    """
    Show the relationship between Σ and Σ⁻¹.
    """
    print("\n" + "=" * 70)
    print("SECTION 6.2: THE COVARIANCE MATRIX AND ITS INVERSE")
    print("=" * 70)
    
    # Define covariance matrix
    sigma1, sigma2 = 2.0, 1.0  # Standard deviations
    rho = 0.8                   # Correlation
    
    Sigma = np.array([[sigma1**2, rho * sigma1 * sigma2],
                      [rho * sigma1 * sigma2, sigma2**2]])
    
    Sigma_inv = np.linalg.inv(Sigma)
    
    print(f"\nCovariance matrix Σ:")
    print(f"  σ₁ = {sigma1}, σ₂ = {sigma2}, ρ = {rho}")
    print(f"  Σ = [[{Sigma[0,0]:.2f}, {Sigma[0,1]:.2f}],")
    print(f"       [{Sigma[1,0]:.2f}, {Sigma[1,1]:.2f}]]")
    
    print(f"\nInverse covariance matrix Σ⁻¹:")
    print(f"  Σ⁻¹ = [[{Sigma_inv[0,0]:.4f}, {Sigma_inv[0,1]:.4f}],")
    print(f"         [{Sigma_inv[1,0]:.4f}, {Sigma_inv[1,1]:.4f}]]")
    
    # Eigenvalues
    eigenvalues_Sigma, eigenvectors_Sigma = np.linalg.eigh(Sigma)
    eigenvalues_Sigma_inv = 1 / eigenvalues_Sigma
    
    print(f"\nEigenvalues of Σ:   λ₁ = {eigenvalues_Sigma[1]:.4f}, λ₂ = {eigenvalues_Sigma[0]:.4f}")
    print(f"Eigenvalues of Σ⁻¹: 1/λ₁ = {eigenvalues_Sigma_inv[1]:.4f}, 1/λ₂ = {eigenvalues_Sigma_inv[0]:.4f}")
    
    print(f"\nKey insight:")
    print(f"  - Σ has large eigenvalue ({eigenvalues_Sigma[1]:.2f}) in high-variance direction")
    print(f"  - Σ⁻¹ has SMALL eigenvalue ({eigenvalues_Sigma_inv[1]:.4f}) in that same direction")
    print(f"  - This DOWNWEIGHTS deviations in the high-variance direction")
    
    # Verify Σ × Σ⁻¹ = I
    product = Sigma @ Sigma_inv
    print(f"\nVerification: Σ × Σ⁻¹ = I")
    print(f"  [[{product[0,0]:.6f}, {product[0,1]:.6f}],")
    print(f"   [{product[1,0]:.6f}, {product[1,1]:.6f}]]")
    
    return Sigma, Sigma_inv


# =============================================================================
# SECTION 6.3: A ONE-DIMENSIONAL SANITY CHECK
# =============================================================================

def demonstrate_1d_case():
    """
    Show that Mahalanobis distance reduces to z-score in 1D.
    """
    print("\n" + "=" * 70)
    print("SECTION 6.3: A ONE-DIMENSIONAL SANITY CHECK")
    print("=" * 70)
    
    # In 1D, covariance matrix is just variance
    mu = 50
    sigma = 10
    variance = sigma**2
    
    # A point
    z = 70
    
    # Z-score
    z_score = (z - mu) / sigma
    
    # Mahalanobis distance (in 1D)
    # d²_Mah = (z - μ)² / σ² = z_score²
    # d_Mah = |z_score|
    Sigma_1d = np.array([[variance]])
    z_array = np.array([z])
    mu_array = np.array([mu])
    d_mah = mahalanobis_distance(z_array, mu_array, Sigma_1d)
    
    print(f"\nIn one dimension:")
    print(f"  Population: μ = {mu}, σ = {sigma}")
    print(f"  Observed value: z = {z}")
    print(f"\n  Z-score: (z - μ) / σ = ({z} - {mu}) / {sigma} = {z_score:.2f}")
    print(f"\n  Mahalanobis distance:")
    print(f"    d²_Mah = (z - μ)ᵀ Σ⁻¹ (z - μ)")
    print(f"           = (z - μ)² / σ²")
    print(f"           = ({z} - {mu})² / {sigma}²")
    print(f"           = {(z - mu)**2} / {sigma**2}")
    print(f"           = {(z - mu)**2 / sigma**2:.2f}")
    print(f"    d_Mah  = √{(z - mu)**2 / sigma**2:.2f} = {d_mah:.2f}")
    
    print(f"\n  ✓ Mahalanobis distance = |z-score| = {abs(z_score):.2f}")
    print(f"\n  In 1D, Mahalanobis distance is just the number of SDs from the mean!")


# =============================================================================
# SECTION 6.4: GEOMETRY — HOW THE INVERSE RESHAPES SPACE
# =============================================================================

def demonstrate_geometry():
    """
    Visualise how Σ⁻¹ reshapes the unit ball from a circle to an ellipse.
    """
    print("\n" + "=" * 70)
    print("SECTION 6.4: GEOMETRY — HOW THE INVERSE RESHAPES SPACE")
    print("=" * 70)
    
    # Define covariance matrix
    Sigma = np.array([[4.0, 2.0],
                      [2.0, 1.5]])
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    print(f"\nCovariance matrix:")
    print(f"  Σ = [[{Sigma[0,0]:.1f}, {Sigma[0,1]:.1f}],")
    print(f"       [{Sigma[1,0]:.1f}, {Sigma[1,1]:.1f}]]")
    print(f"\n  Eigenvalues: λ₁ = {eigenvalues[0]:.3f}, λ₂ = {eigenvalues[1]:.3f}")
    print(f"  Semi-axis lengths: √λ₁ = {np.sqrt(eigenvalues[0]):.3f}, "
          f"√λ₂ = {np.sqrt(eigenvalues[1]):.3f}")
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Generate points on a circle (Euclidean unit ball)
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    
    # Left panel: Euclidean distance = 1 (circle)
    ax = axes[0]
    ax.plot(circle_x, circle_y, 'b-', lw=2, label='d_Euc = 1')
    ax.plot(0, 0, 'k+', markersize=15, mew=2)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_xlabel('Trait 1')
    ax.set_ylabel('Trait 2')
    ax.set_title('Euclidean: d² = zᵀz\n(Circle)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    # Middle panel: Mahalanobis distance = 1 (ellipse)
    # For d²_Mah = zᵀ Σ⁻¹ z = 1, the contour is an ellipse with
    # semi-axes √λᵢ along eigenvector directions
    ax = axes[1]
    
    # Parametric ellipse in eigenvector coordinates, then rotate
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    ellipse_x = np.sqrt(eigenvalues[0]) * np.cos(theta) * np.cos(angle) - \
                np.sqrt(eigenvalues[1]) * np.sin(theta) * np.sin(angle)
    ellipse_y = np.sqrt(eigenvalues[0]) * np.cos(theta) * np.sin(angle) + \
                np.sqrt(eigenvalues[1]) * np.sin(theta) * np.cos(angle)
    
    ax.plot(ellipse_x, ellipse_y, 'r-', lw=2, label='d_Mah = 1')
    ax.plot(0, 0, 'k+', markersize=15, mew=2)
    
    # Draw eigenvectors
    scale = 2.5
    ax.arrow(0, 0, scale*eigenvectors[0, 0], scale*eigenvectors[1, 0], 
             head_width=0.1, head_length=0.1, fc='darkgreen', ec='darkgreen')
    ax.arrow(0, 0, scale*eigenvectors[0, 1], scale*eigenvectors[1, 1],
             head_width=0.1, head_length=0.1, fc='purple', ec='purple')
    
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_xlabel('Trait 1')
    ax.set_ylabel('Trait 2')
    ax.set_title('Mahalanobis: d² = zᵀΣ⁻¹z\n(Ellipse matching Σ)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    # Right panel: Both overlaid
    ax = axes[2]
    ax.plot(circle_x, circle_y, 'b--', lw=2, label='Euclidean d = 1', alpha=0.7)
    ax.plot(ellipse_x, ellipse_y, 'r-', lw=2, label='Mahalanobis d = 1')
    ax.plot(0, 0, 'k+', markersize=15, mew=2)
    
    # Mark key points
    p_along = np.array([1, 1]) / np.sqrt(2)  # Along major axis
    p_against = np.array([1, -1]) / np.sqrt(2)  # Along minor axis
    
    ax.plot(*p_along, 'go', markersize=12, markeredgecolor='black',
            markeredgewidth=2, label='Along (common)')
    ax.plot(*p_against, 'm^', markersize=12, markeredgecolor='black',
            markeredgewidth=2, label='Against (rare)')
    
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_xlabel('Trait 1')
    ax.set_ylabel('Trait 2')
    ax.set_title('Comparison\n(Same Euclidean, different Mahalanobis)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('fig_06_04_geometry.png', dpi=150, bbox_inches='tight')
    plt.savefig('fig_06_04_geometry.pdf', bbox_inches='tight')
    print(f"\nFigure saved: fig_06_04_geometry.png/pdf")
    plt.show()


# =============================================================================
# SECTION 6.5: THE FORMULA IN COMPONENTS
# =============================================================================

def demonstrate_formula_components():
    """
    Expand the Mahalanobis formula in terms of variances and correlations.
    """
    print("\n" + "=" * 70)
    print("SECTION 6.5: THE FORMULA IN COMPONENTS")
    print("=" * 70)
    
    print("""
For two traits with covariance matrix:
    Σ = [[σ₁², ρσ₁σ₂],
         [ρσ₁σ₂, σ₂²]]

The inverse is:
    Σ⁻¹ = (1 / det(Σ)) × [[σ₂², -ρσ₁σ₂],
                          [-ρσ₁σ₂, σ₁²]]
    
    where det(Σ) = σ₁²σ₂²(1 - ρ²)

The squared Mahalanobis distance becomes:
    d²_Mah = (1 / (1 - ρ²)) × [((z₁ - μ₁)/σ₁)² - 2ρ((z₁ - μ₁)/σ₁)((z₂ - μ₂)/σ₂) 
                               + ((z₂ - μ₂)/σ₂)²]

This has three parts:
    1. Squared z-scores for each trait
    2. A cross-term that SUBTRACTS when signs match (along correlation)
       and ADDS when signs oppose (against correlation)
    3. A factor 1/(1 - ρ²) that inflates when correlation is high
""")
    
    # Numerical example
    sigma1, sigma2 = 2.0, 1.5
    rho = 0.6
    mu = np.array([10, 5])
    
    Sigma = np.array([[sigma1**2, rho * sigma1 * sigma2],
                      [rho * sigma1 * sigma2, sigma2**2]])
    
    # Two test points
    z_along = np.array([12, 6])   # Both above mean
    z_against = np.array([12, 4])  # One above, one below
    
    print(f"\nNumerical example:")
    print(f"  σ₁ = {sigma1}, σ₂ = {sigma2}, ρ = {rho}")
    print(f"  Mean μ = ({mu[0]}, {mu[1]})")
    print(f"  1 - ρ² = {1 - rho**2:.3f}")
    
    for name, z in [('Along (both above)', z_along), ('Against (one above)', z_against)]:
        # Z-scores
        z1_std = (z[0] - mu[0]) / sigma1
        z2_std = (z[1] - mu[1]) / sigma2
        
        # Components
        term1 = z1_std**2
        term2 = -2 * rho * z1_std * z2_std
        term3 = z2_std**2
        
        d2_mah = (1 / (1 - rho**2)) * (term1 + term2 + term3)
        d_mah = np.sqrt(d2_mah)
        
        # Verify with matrix formula
        d_mah_check = mahalanobis_distance(z, mu, Sigma)
        
        print(f"\n  Point z = {z} ({name}):")
        print(f"    z₁_std = ({z[0]} - {mu[0]}) / {sigma1} = {z1_std:.3f}")
        print(f"    z₂_std = ({z[1]} - {mu[1]}) / {sigma2} = {z2_std:.3f}")
        print(f"    Term 1 (z₁²):     {term1:.3f}")
        print(f"    Term 2 (cross):   {term2:.3f}")
        print(f"    Term 3 (z₂²):     {term3:.3f}")
        print(f"    Sum:              {term1 + term2 + term3:.3f}")
        print(f"    × 1/(1-ρ²):       {d2_mah:.3f}")
        print(f"    d_Mah = √{d2_mah:.3f} = {d_mah:.3f}")
        print(f"    [Verification: {d_mah_check:.3f}]")
    
    print(f"\nNotice: The cross-term is NEGATIVE when both z-scores have the same sign")
    print(f"        (reducing distance for points along correlation)")
    print(f"        and POSITIVE when they have opposite signs")
    print(f"        (increasing distance for points against correlation).")


# =============================================================================
# SECTION 6.6: MAHALANOBIS DISTANCE AND PROBABILITY
# =============================================================================

def demonstrate_probability_connection():
    """
    Show that Mahalanobis distance connects directly to probability.
    """
    print("\n" + "=" * 70)
    print("SECTION 6.6: MAHALANOBIS DISTANCE AND PROBABILITY")
    print("=" * 70)
    
    print("""
For multivariate normal data, the squared Mahalanobis distance follows
a chi-squared distribution with p degrees of freedom:
    
    d²_Mah ~ χ²_p

This means:
    - P(d²_Mah < χ²_{p, α}) = α
    - Points with d²_Mah = χ²_{p, 0.95} lie on the 95% probability contour
    - Equal Mahalanobis distance = Equal probability density

For p = 2 traits:
    - 50% of data: d²_Mah < χ²_{2, 0.50} = 1.386, so d_Mah < 1.18
    - 95% of data: d²_Mah < χ²_{2, 0.95} = 5.991, so d_Mah < 2.45
    - 99% of data: d²_Mah < χ²_{2, 0.99} = 9.210, so d_Mah < 3.03
""")
    
    # Generate data and verify
    Sigma = np.array([[1.0, 0.7],
                      [0.7, 1.0]])
    mu = np.array([0, 0])
    
    n = 10000
    data = np.random.multivariate_normal(mu, Sigma, n)
    
    # Compute Mahalanobis distances
    Sigma_inv = np.linalg.inv(Sigma)
    d2_mah = np.array([z @ Sigma_inv @ z for z in data])
    
    # Chi-squared quantiles for p = 2
    chi2_50 = stats.chi2.ppf(0.50, df=2)
    chi2_95 = stats.chi2.ppf(0.95, df=2)
    chi2_99 = stats.chi2.ppf(0.99, df=2)
    
    # Empirical fractions
    frac_50 = np.mean(d2_mah < chi2_50) * 100
    frac_95 = np.mean(d2_mah < chi2_95) * 100
    frac_99 = np.mean(d2_mah < chi2_99) * 100
    
    print(f"\nSimulation: n = {n} points from bivariate normal (ρ = 0.7)")
    print(f"\n{'Quantile':>12} {'χ²_{2,α}':>10} {'d_Mah':>10} {'Expected':>10} {'Observed':>10}")
    print("-" * 55)
    print(f"{'50%':>12} {chi2_50:>10.3f} {np.sqrt(chi2_50):>10.3f} {'50.0%':>10} {frac_50:>10.1f}%")
    print(f"{'95%':>12} {chi2_95:>10.3f} {np.sqrt(chi2_95):>10.3f} {'95.0%':>10} {frac_95:>10.1f}%")
    print(f"{'99%':>12} {chi2_99:>10.3f} {np.sqrt(chi2_99):>10.3f} {'99.0%':>10} {frac_99:>10.1f}%")
    print("-" * 55)
    
    print(f"\n✓ Mahalanobis distance correctly calibrates with probability!")
    
    # Compare with Euclidean
    d2_euc = np.sum(data**2, axis=1)
    frac_euc_50 = np.mean(d2_euc < chi2_50) * 100
    frac_euc_95 = np.mean(d2_euc < chi2_95) * 100
    
    print(f"\nComparison with Euclidean (same χ² thresholds):")
    print(f"  50% threshold: Euclidean captures {frac_euc_50:.1f}% (should be 50%)")
    print(f"  95% threshold: Euclidean captures {frac_euc_95:.1f}% (should be 95%)")
    print(f"\n  Euclidean is MISCALIBRATED because it ignores correlation!")
    
    # Visualise
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: histogram of d²_Mah
    ax = axes[0]
    ax.hist(d2_mah, bins=50, density=True, alpha=0.7, color='steelblue',
            edgecolor='black', label='Observed d²_Mah')
    
    # Overlay chi-squared distribution
    x = np.linspace(0, 15, 200)
    ax.plot(x, stats.chi2.pdf(x, df=2), 'r-', lw=2, label='χ²₂ distribution')
    
    ax.axvline(chi2_95, color='orange', linestyle='--', lw=2, 
               label=f'95% quantile = {chi2_95:.2f}')
    ax.set_xlabel('Squared Mahalanobis distance')
    ax.set_ylabel('Density')
    ax.set_title('d²_Mah follows χ² distribution')
    ax.legend()
    ax.set_xlim(0, 15)
    
    # Right: probability contours
    ax = axes[1]
    
    # Plot data
    ax.scatter(data[:, 0], data[:, 1], alpha=0.1, s=5, c='steelblue')
    
    # Draw Mahalanobis contours
    theta = np.linspace(0, 2*np.pi, 100)
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
    angle = np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1])
    
    for level, label, color in [(chi2_50, '50%', 'green'), 
                                 (chi2_95, '95%', 'orange'),
                                 (chi2_99, '99%', 'red')]:
        r = np.sqrt(level)
        ellipse_x = r * np.sqrt(eigenvalues[1]) * np.cos(theta) * np.cos(angle) - \
                    r * np.sqrt(eigenvalues[0]) * np.sin(theta) * np.sin(angle)
        ellipse_y = r * np.sqrt(eigenvalues[1]) * np.cos(theta) * np.sin(angle) + \
                    r * np.sqrt(eigenvalues[0]) * np.sin(theta) * np.cos(angle)
        ax.plot(ellipse_x, ellipse_y, color=color, lw=2, label=f'{label} contour')
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.set_xlabel('Trait 1')
    ax.set_ylabel('Trait 2')
    ax.set_title('Probability Contours (Mahalanobis)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fig_06_06_probability.png', dpi=150, bbox_inches='tight')
    plt.savefig('fig_06_06_probability.pdf', bbox_inches='tight')
    print(f"\nFigure saved: fig_06_06_probability.png/pdf")
    plt.show()


# =============================================================================
# SECTION 6.7: A WORKED EXAMPLE
# =============================================================================

def worked_example():
    """
    Complete worked example from the book.
    """
    print("\n" + "=" * 70)
    print("SECTION 6.7: A WORKED EXAMPLE")
    print("=" * 70)
    
    # From Section 5.7
    Sigma = np.array([[1.0, 0.8],
                      [0.8, 1.0]])
    Sigma_inv = np.linalg.inv(Sigma)
    
    print(f"\nCovariance matrix Σ = [[1.0, 0.8], [0.8, 1.0]]")
    print(f"Inverse Σ⁻¹ = [[{Sigma_inv[0,0]:.3f}, {Sigma_inv[0,1]:.3f}],")
    print(f"               [{Sigma_inv[1,0]:.3f}, {Sigma_inv[1,1]:.3f}]]")
    
    # Three points at Euclidean distance 1
    points = {
        'E': (np.array([1/np.sqrt(2), 1/np.sqrt(2)]), 'Along correlation'),
        'F': (np.array([1/np.sqrt(2), -1/np.sqrt(2)]), 'Against correlation'),
        'G': (np.array([1.0, 0.0]), 'Trait 1 only')
    }
    
    print(f"\n{'Point':>6} {'Coordinates':>20} {'d_Euc':>8} {'d_Mah':>8} {'d²_Mah':>8}")
    print("-" * 60)
    
    for name, (p, desc) in points.items():
        d_euc = np.linalg.norm(p)
        d2_mah = p @ Sigma_inv @ p
        d_mah = np.sqrt(d2_mah)
        print(f"{name:>6} ({p[0]:>6.3f}, {p[1]:>6.3f}) {d_euc:>8.3f} {d_mah:>8.3f} {d2_mah:>8.3f}")
    
    # Detailed calculation for Point E
    p = points['E'][0]
    print(f"\n" + "-" * 50)
    print(f"Detailed calculation for Point E = ({p[0]:.3f}, {p[1]:.3f}):")
    print("-" * 50)
    
    step1 = Sigma_inv @ p
    print(f"\nStep 1: Σ⁻¹ z = [[{Sigma_inv[0,0]:.3f}, {Sigma_inv[0,1]:.3f}],  × [{p[0]:.3f}]")
    print(f"                 [{Sigma_inv[1,0]:.3f}, {Sigma_inv[1,1]:.3f}]]    [{p[1]:.3f}]")
    print(f"              = [{step1[0]:.3f}]")
    print(f"                [{step1[1]:.3f}]")
    
    step2 = p @ step1
    print(f"\nStep 2: zᵀ (Σ⁻¹ z) = [{p[0]:.3f}, {p[1]:.3f}] × [{step1[0]:.3f}]")
    print(f"                                             [{step1[1]:.3f}]")
    print(f"                   = {p[0]:.3f} × {step1[0]:.3f} + {p[1]:.3f} × {step1[1]:.3f}")
    print(f"                   = {p[0] * step1[0]:.3f} + {p[1] * step1[1]:.3f}")
    print(f"                   = {step2:.3f}")
    
    print(f"\nStep 3: d_Mah = √{step2:.3f} = {np.sqrt(step2):.3f}")
    
    # Detailed for Point F
    p = points['F'][0]
    print(f"\n" + "-" * 50)
    print(f"Detailed calculation for Point F = ({p[0]:.3f}, {p[1]:.3f}):")
    print("-" * 50)
    
    step1 = Sigma_inv @ p
    step2 = p @ step1
    
    print(f"\nStep 1: Σ⁻¹ z = [{step1[0]:.3f}, {step1[1]:.3f}]ᵀ")
    print(f"Step 2: zᵀ (Σ⁻¹ z) = {step2:.3f}")
    print(f"Step 3: d_Mah = √{step2:.3f} = {np.sqrt(step2):.3f}")
    
    print(f"\n" + "=" * 50)
    print("KEY RESULT:")
    print("=" * 50)
    d_E = np.sqrt(points['E'][0] @ Sigma_inv @ points['E'][0])
    d_F = np.sqrt(points['F'][0] @ Sigma_inv @ points['F'][0])
    print(f"  Point E (along correlation):   d_Mah = {d_E:.3f}")
    print(f"  Point F (against correlation): d_Mah = {d_F:.3f}")
    print(f"  Ratio: F is {d_F/d_E:.1f}× more unusual than E!")


# =============================================================================
# SECTION 6.8: THE MAHALANOBIS DISTANCE AS A TRANSFORMATION
# =============================================================================

def demonstrate_whitening():
    """
    Show that Mahalanobis distance equals Euclidean distance in whitened space.
    """
    print("\n" + "=" * 70)
    print("SECTION 6.8: THE MAHALANOBIS DISTANCE AS A TRANSFORMATION")
    print("=" * 70)
    
    print("""
Key insight: Mahalanobis distance is Euclidean distance after WHITENING.

The whitening transformation:
    w = Σ^{-1/2} (z - μ)

transforms data so that:
    - Cov(w) = I  (identity matrix)
    - The ellipse becomes a circle
    - Euclidean distance works correctly

Then:
    ||w||² = wᵀw 
           = [Σ^{-1/2}(z-μ)]ᵀ [Σ^{-1/2}(z-μ)]
           = (z-μ)ᵀ Σ^{-1/2} Σ^{-1/2} (z-μ)
           = (z-μ)ᵀ Σ^{-1} (z-μ)
           = d²_Mah

So Mahalanobis distance = Euclidean distance in whitened coordinates!
""")
    
    # Define covariance matrix
    Sigma = np.array([[2.5, 1.5],
                      [1.5, 1.5]])
    mu = np.array([0, 0])
    
    # Compute Σ^{-1/2} using eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
    Lambda_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues))
    Sigma_inv_sqrt = eigenvectors @ Lambda_inv_sqrt @ eigenvectors.T
    
    print(f"Covariance matrix Σ:")
    print(f"  [[{Sigma[0,0]:.2f}, {Sigma[0,1]:.2f}],")
    print(f"   [{Sigma[1,0]:.2f}, {Sigma[1,1]:.2f}]]")
    
    print(f"\nΣ^{{-1/2}}:")
    print(f"  [[{Sigma_inv_sqrt[0,0]:.4f}, {Sigma_inv_sqrt[0,1]:.4f}],")
    print(f"   [{Sigma_inv_sqrt[1,0]:.4f}, {Sigma_inv_sqrt[1,1]:.4f}]]")
    
    # Verify: Σ^{-1/2} Σ Σ^{-1/2} = I
    check = Sigma_inv_sqrt @ Sigma @ Sigma_inv_sqrt
    print(f"\nVerification: Σ^{{-1/2}} Σ Σ^{{-1/2}} =")
    print(f"  [[{check[0,0]:.6f}, {check[0,1]:.6f}],")
    print(f"   [{check[1,0]:.6f}, {check[1,1]:.6f}]]  ✓ (should be identity)")
    
    # Generate data and whiten
    n = 500
    data = np.random.multivariate_normal(mu, Sigma, n)
    data_whitened = (Sigma_inv_sqrt @ data.T).T
    
    # Sample covariance of whitened data
    cov_whitened = np.cov(data_whitened.T)
    print(f"\nSample covariance of whitened data:")
    print(f"  [[{cov_whitened[0,0]:.3f}, {cov_whitened[0,1]:.3f}],")
    print(f"   [{cov_whitened[1,0]:.3f}, {cov_whitened[1,1]:.3f}]]  ✓ (should be ≈ I)")
    
    # Visualise
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Original data
    ax = axes[0]
    ax.scatter(data[:, 0], data[:, 1], alpha=0.5, s=20, c='steelblue')
    
    # Draw 1-SD ellipse
    theta = np.linspace(0, 2*np.pi, 100)
    angle = np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1])
    ellipse_x = np.sqrt(eigenvalues[1]) * np.cos(theta) * np.cos(angle) - \
                np.sqrt(eigenvalues[0]) * np.sin(theta) * np.sin(angle)
    ellipse_y = np.sqrt(eigenvalues[1]) * np.cos(theta) * np.sin(angle) + \
                np.sqrt(eigenvalues[0]) * np.sin(theta) * np.cos(angle)
    ax.plot(ellipse_x, ellipse_y, 'r-', lw=2, label='d_Mah = 1')
    
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.set_xlabel('Trait 1')
    ax.set_ylabel('Trait 2')
    ax.set_title('Original Data\n(Elliptical)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Right: Whitened data
    ax = axes[1]
    ax.scatter(data_whitened[:, 0], data_whitened[:, 1], alpha=0.5, s=20, c='steelblue')
    
    # Draw unit circle
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    ax.plot(circle_x, circle_y, 'r-', lw=2, label='d_Euc = 1')
    
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.set_xlabel('Whitened Trait 1')
    ax.set_ylabel('Whitened Trait 2')
    ax.set_title('Whitened Data\n(Spherical, Cov = I)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fig_06_08_whitening.png', dpi=150, bbox_inches='tight')
    plt.savefig('fig_06_08_whitening.pdf', bbox_inches='tight')
    print(f"\nFigure saved: fig_06_08_whitening.png/pdf")
    plt.show()
    
    return Sigma_inv_sqrt


# =============================================================================
# SECTION 6.11: BIOLOGICAL INTERPRETATION
# =============================================================================

def demonstrate_biological_interpretation():
    """
    Show how Mahalanobis distance applies to G and P matrices.
    """
    print("\n" + "=" * 70)
    print("SECTION 6.11: BIOLOGICAL INTERPRETATION")
    print("=" * 70)
    
    print("""
The choice of covariance matrix changes the QUESTION being asked:

Using PHENOTYPIC covariance P:
    d²_P = (z - μ)ᵀ P⁻¹ (z - μ)
    → "How unusual is this phenotype given the TOTAL variation?"
    → Measures statistical rarity in the population

Using GENETIC covariance G:
    d²_G = (z - μ)ᵀ G⁻¹ (z - μ)
    → "How unusual is this phenotype given the HERITABLE variation?"
    → Measures genetic distance from the mean
    → Relevant for: How hard is it to EVOLVE to this phenotype?

A phenotype can be:
    - Close in P terms (common phenotypically)
    - Far in G terms (requires unusual genetic combinations)
    
This happens when environmental variation is large in certain directions
but genetic variation is small.
""")
    
    # Example
    G = np.array([[0.5, 0.3],
                  [0.3, 0.4]])
    
    P = np.array([[1.0, 0.4],
                  [0.4, 0.8]])
    
    mu = np.array([10, 8])
    z = np.array([12, 7])  # New phenotype
    
    G_inv = np.linalg.inv(G)
    P_inv = np.linalg.inv(P)
    
    d_G = np.sqrt((z - mu) @ G_inv @ (z - mu))
    d_P = np.sqrt((z - mu) @ P_inv @ (z - mu))
    
    print(f"\nExample:")
    print(f"  Genetic covariance G = [[0.5, 0.3], [0.3, 0.4]]")
    print(f"  Phenotypic covariance P = [[1.0, 0.4], [0.4, 0.8]]")
    print(f"  Population mean μ = ({mu[0]}, {mu[1]})")
    print(f"  New phenotype z = ({z[0]}, {z[1]})")
    
    print(f"\n  Phenotypic Mahalanobis distance: d_P = {d_P:.3f}")
    print(f"  Genetic Mahalanobis distance:    d_G = {d_G:.3f}")
    
    print(f"\n  Interpretation:")
    print(f"    - Phenotypically, z is {d_P:.2f} 'units' from the mean")
    print(f"    - Genetically, z is {d_G:.2f} 'units' from the mean")
    print(f"    - The phenotype is {d_G/d_P:.1f}× harder to reach genetically")
    print(f"      than you might expect from phenotypic variation alone.")


# =============================================================================
# MAIN DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CHAPTER 6: Covariance and Mahalanobis Distance")
    print("Seeing the Shape — Code Companion")
    print("=" * 70)
    
    # Section 6.1: Key insight
    demonstrate_matrix_insertion()
    
    # Section 6.2: Covariance and inverse
    demonstrate_inverse_covariance()
    
    # Section 6.3: 1D sanity check
    demonstrate_1d_case()
    
    # Section 6.4: Geometry
    demonstrate_geometry()
    
    # Section 6.5: Formula in components
    demonstrate_formula_components()
    
    # Section 6.6: Probability connection
    demonstrate_probability_connection()
    
    # Section 6.7: Worked example
    worked_example()
    
    # Section 6.8: Whitening
    demonstrate_whitening()
    
    # Section 6.11: Biological interpretation
    demonstrate_biological_interpretation()
    
    print("\n" + "=" * 70)
    print("END OF CHAPTER 6")
    print("Next: Chapter 7 — Diagonalisation and Natural Axes")
    print("=" * 70)


# =============================================================================
# EXERCISES
# =============================================================================
"""
EXERCISE 6.1: Mahalanobis Distance by Hand
------------------------------------------
Given Σ = [[4, 2], [2, 3]] and a point z = (3, 2):
(a) Compute Σ⁻¹.
(b) Compute d²_Mah = zᵀ Σ⁻¹ z.
(c) Compute d_Mah.
(d) Compare to the Euclidean distance ||z||.

EXERCISE 6.2: Eigenvalue Interpretation
---------------------------------------
A covariance matrix has eigenvalues λ₁ = 9 and λ₂ = 1.
(a) What are the eigenvalues of Σ⁻¹?
(b) The eigenvector for λ₁ points along (0.8, 0.6). A point z lies 
    2 units along this direction. What is its Mahalanobis distance?
(c) Another point w lies 2 units along the perpendicular direction.
    What is its Mahalanobis distance?
(d) Which point is more unusual? Why?

EXERCISE 6.3: The Whitening Check
---------------------------------
Generate 1000 samples from a bivariate normal with Σ = [[1, 0.7], [0.7, 1]].
(a) Compute Σ^(-1/2).
(b) Transform the data: w = Σ^(-1/2) z for each point.
(c) Compute the sample covariance of the whitened data.
(d) Verify it's close to the identity matrix.

EXERCISE 6.4: Chi-Squared Verification
--------------------------------------
For the whitened data from Exercise 6.3:
(a) Compute ||w||² for each point (this equals d²_Mah).
(b) What fraction have d²_Mah < χ²_{2, 0.5} = 1.386?
(c) What fraction have d²_Mah < χ²_{2, 0.95} = 5.991?
(d) Compare to the theoretical 50% and 95%.

EXERCISE 6.5: Biological Application
-------------------------------------
A G matrix for two traits is:
    G = [[0.5, 0.3], [0.3, 0.4]]
    
The population mean is (10, 8). A new phenotype z = (12, 7) is observed.
(a) Compute the Mahalanobis distance from z to the mean using G.
(b) Interpret: is this phenotype genetically unusual?
(c) If selection pushes toward z, what does this distance tell us 
    about the expected response?

EXERCISE 6.6: Discriminant Analysis Preview
-------------------------------------------
Two species have means μ₁ = (5, 3) and μ₂ = (8, 6) and share a common
within-group covariance Σ_W = [[2, 1], [1, 2]].
(a) Compute the Mahalanobis distance between species means.
(b) A new individual has phenotype z = (6, 5). Compute d_Mah to each mean.
(c) Which species is the individual closer to (in Mahalanobis terms)?
(d) How does this relate to linear discriminant analysis?
"""
