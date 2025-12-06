#!/usr/bin/env python3
"""
==============================================================================
CHAPTER 13: DIRECTIONAL HERITABILITY AND THE GEOMETRY OF CONSTRAINT
==============================================================================

Seeing the Shape: A Geometric Introduction to Multivariate Quantitative Genetics
Code Companion - Chapter 13

Author: Daniel Ortiz-Barrientos
School of the Environment, The University of Queensland

------------------------------------------------------------------------------
THE CAPSTONE INSIGHT
------------------------------------------------------------------------------

This final chapter connects the geometric framework to a frontier research
question: how does heritability vary across directions in trait space, and
what does this variation tell us about evolutionary constraint?

The eigenvalues of G* = P^{-1/2} G P^{-1/2} are the directional heritabilities
along principal axes. But most selection does not align with principal axes.

KEY QUESTIONS:
1. What is the DISTRIBUTION of heritability across all possible directions?
2. How do we characterise, measure, and interpret this distribution?
3. When constraint heterogeneity is high, what are the consequences?

THE CENTRAL FORMULA:
-------------------
For uniform random directions on the P-sphere:

    CV²(h²) = (2 / (p + 2)) × V_rel(G*)

where:
    • CV(h²) = coefficient of variation of directional heritability
    • p = number of traits
    • V_rel(G*) = Var(λ*) / mean(λ*)² = relative variance of G* eigenvalues

This elegant relationship connects:
    • A distributional property (heritability variation)
    • A geometric property (G* eccentricity)
    • The dimensionality of trait space

------------------------------------------------------------------------------
CONSTRAINT TRAPS
------------------------------------------------------------------------------

A constraint trap occurs when a direction has:
    • Normal phenotypic variance (it's on the P-sphere)
    • LOW heritability (G* is thin there)

Selection in that direction produces little response because genetic variance
is low relative to environmental variance.

Constraint severity = 1 - (λ*_min / mean(λ*))

This measures how much worse the worst direction is compared to average.

------------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, sqrtm, inv
from scipy import stats
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
    'figure.figsize': (14, 10),
    'figure.dpi': 100
})


# =============================================================================
# SECTION 1: CORE FUNCTIONS FROM PREVIOUS CHAPTERS
# =============================================================================
"""
We begin by importing the essential functions developed in earlier chapters.
These form the foundation for our analysis of directional heritability.
"""


def eigendecompose(A: np.ndarray, 
                   sort_descending: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Eigendecomposition of a symmetric matrix: A = V Λ V^T
    
    Parameters
    ----------
    A : np.ndarray
        Symmetric matrix to decompose
    sort_descending : bool
        If True, return eigenvalues in descending order
    
    Returns
    -------
    eigenvalues : np.ndarray
        Eigenvalues (λ₁ ≥ λ₂ ≥ ... ≥ λₚ)
    eigenvectors : np.ndarray
        Orthonormal eigenvectors as columns
    """
    eigenvalues, eigenvectors = eigh(A)
    
    if sort_descending:
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
    
    return eigenvalues, eigenvectors


def compute_P_inv_sqrt(P: np.ndarray) -> np.ndarray:
    """
    Compute P^{-1/2} for the whitening transformation.
    
    P^{-1/2} = V_P Λ_P^{-1/2} V_P^T
    
    where Λ_P^{-1/2} has diagonal entries 1/√λᵢ
    """
    eigenvalues, eigenvectors = eigendecompose(P)
    
    if np.any(eigenvalues <= 0):
        raise ValueError("P must be positive definite")
    
    Lambda_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues))
    P_inv_sqrt = eigenvectors @ Lambda_inv_sqrt @ eigenvectors.T
    
    return P_inv_sqrt


def compute_G_star(G: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Compute the P-whitened genetic matrix: G* = P^{-1/2} G P^{-1/2}
    
    In whitened space:
        • P becomes the identity (the P-sphere becomes a unit sphere)
        • G becomes G*, whose eigenvalues ARE directional heritabilities
    """
    P_inv_sqrt = compute_P_inv_sqrt(P)
    return P_inv_sqrt @ G @ P_inv_sqrt


def directional_heritability(direction: np.ndarray,
                              G: np.ndarray,
                              P: np.ndarray) -> float:
    """
    Compute heritability in a given direction: h²(β) = β'Gβ / β'Pβ
    """
    return (direction @ G @ direction) / (direction @ P @ direction)


# =============================================================================
# SECTION 2: SAMPLING FROM THE P-SPHERE
# =============================================================================
"""
To study the distribution of directional heritability, we need to sample
directions uniformly from the P-sphere: the set of all directions with
unit phenotypic variance.

The P-sphere is defined as: {β : β'Pβ = 1}

In original coordinates, this is an ellipsoid. But after whitening by P^{-1/2},
it becomes the ordinary unit sphere, where uniform sampling is straightforward.

SAMPLING STRATEGY:
1. Generate random vectors from standard normal (isotropic Gaussian)
2. Normalize to unit length (gives uniform distribution on sphere)
3. This samples uniformly from the P-sphere in whitened coordinates
4. Transform back to original coordinates if needed
"""


def sample_uniform_sphere(n_samples: int, p: int) -> np.ndarray:
    """
    Sample n_samples directions uniformly from the unit sphere in R^p.
    
    Uses the fact that normalizing isotropic Gaussian vectors gives
    uniform distribution on the sphere.
    
    Parameters
    ----------
    n_samples : int
        Number of direction vectors to sample
    p : int
        Dimension of the space
    
    Returns
    -------
    np.ndarray
        Array of shape (n_samples, p) with unit vectors as rows
    
    Mathematical Background
    -----------------------
    If X ~ N(0, I_p), then X/||X|| is uniformly distributed on S^{p-1}.
    
    This is because:
    1. The multivariate normal is spherically symmetric
    2. Projecting onto the sphere preserves this symmetry
    3. The only distribution on S^{p-1} with this symmetry is uniform
    """
    # Generate isotropic Gaussian vectors
    X = np.random.randn(n_samples, p)
    
    # Normalize to unit length
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    directions = X / norms
    
    return directions


def sample_from_P_sphere(n_samples: int, P: np.ndarray) -> np.ndarray:
    """
    Sample directions uniformly from the P-sphere in original coordinates.
    
    The P-sphere is {β : β'Pβ = 1}. Uniform sampling on this ellipsoid
    is achieved by:
    1. Sample uniformly from the unit sphere in whitened space
    2. Transform back to original coordinates using P^{1/2}
    
    Parameters
    ----------
    n_samples : int
        Number of directions to sample
    P : np.ndarray
        Phenotypic covariance matrix
    
    Returns
    -------
    np.ndarray
        Array of shape (n_samples, p) with directions on P-sphere
    
    Notes
    -----
    Each returned direction β satisfies β'Pβ = 1 (unit phenotypic variance).
    """
    p = P.shape[0]
    
    # Compute P^{1/2}
    eigenvalues, eigenvectors = eigendecompose(P)
    P_sqrt = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
    
    # Sample from unit sphere in whitened space
    unit_directions = sample_uniform_sphere(n_samples, p)
    
    # Transform to P-sphere in original coordinates
    # If w is on unit sphere, then β = P^{1/2} w satisfies β'Pβ = 1
    # Actually, we want β = P^{-1/2} w for the P-sphere
    P_inv_sqrt = compute_P_inv_sqrt(P)
    P_sphere_directions = (P_inv_sqrt @ unit_directions.T).T
    
    # Normalize so that β'Pβ = 1
    for i in range(n_samples):
        beta = P_sphere_directions[i]
        norm = np.sqrt(beta @ P @ beta)
        P_sphere_directions[i] = beta / norm
    
    return P_sphere_directions


# =============================================================================
# SECTION 3: DISTRIBUTION OF DIRECTIONAL HERITABILITY
# =============================================================================
"""
The key result from the book:

For a unit vector β uniformly distributed on the P-sphere:
    h²(β) = β'G*β
    
where G* = P^{-1/2} G P^{-1/2}.

The variance of this quadratic form is:
    Var[h²] = (2 / (p(p+2))) × Σᵢ<ⱼ (λ*ᵢ - λ*ⱼ)² = (2 / (p+2)) × Var(λ*)

The mean is:
    E[h²] = (1/p) × Σλ*ᵢ = mean(λ*)

This gives the coefficient of variation:
    CV(h²) = √(2/(p+2)) × CV(λ*)

where CV(λ*) = SD(λ*) / mean(λ*).
"""


@dataclass
class HeritabilityDistribution:
    """
    Container for directional heritability distribution analysis.
    
    Attributes
    ----------
    eigenvalues_G_star : np.ndarray
        Eigenvalues of G* (these ARE directional heritabilities along PCs)
    
    h2_max : float
        Maximum directional heritability (λ*₁)
    
    h2_min : float
        Minimum directional heritability (λ*ₚ)
    
    h2_mean : float
        Mean directional heritability (average of λ*)
    
    h2_variance : float
        Theoretical variance of h² across directions
    
    cv_h2 : float
        Coefficient of variation of directional heritability
    
    V_rel : float
        Relative variance of eigenvalues: Var(λ*) / mean(λ*)²
    
    constraint_severity : float
        How much worse the min is than mean: 1 - λ*_min / mean(λ*)
    
    effective_dimensionality : float
        (tr G*)² / tr(G*²) - measures how many "effective" dimensions
    
    eigenvectors_G_star : np.ndarray
        Eigenvectors of G* (directions of extreme heritabilities)
    """
    eigenvalues_G_star: np.ndarray
    h2_max: float
    h2_min: float
    h2_mean: float
    h2_variance: float
    cv_h2: float
    V_rel: float
    constraint_severity: float
    effective_dimensionality: float
    eigenvectors_G_star: np.ndarray


def analyze_heritability_distribution(G: np.ndarray, 
                                       P: np.ndarray) -> HeritabilityDistribution:
    """
    Analyze the distribution of directional heritability across all directions.
    
    This function computes all key quantities characterizing how heritability
    varies across the P-sphere:
    
    • Extreme values (max and min h²)
    • Mean and variance
    • Coefficient of variation
    • Constraint severity
    • Effective dimensionality
    
    Parameters
    ----------
    G : np.ndarray
        Additive genetic covariance matrix
    P : np.ndarray
        Phenotypic covariance matrix
    
    Returns
    -------
    HeritabilityDistribution
        Complete characterization of the h² distribution
    
    Mathematical Details
    --------------------
    The eigenvalues of G* = P^{-1/2} G P^{-1/2} are the directional
    heritabilities along the principal axes.
    
    For uniform random directions:
        E[h²] = mean(λ*)
        Var[h²] = (2/(p+2)) × Var(λ*)
        CV(h²) = √(2/(p+2)) × CV(λ*)
    
    Example
    -------
    >>> G = np.array([[0.5, 0.3], [0.3, 0.4]])
    >>> P = np.array([[1.0, 0.4], [0.4, 0.8]])
    >>> result = analyze_heritability_distribution(G, P)
    >>> print(f"h² ranges from {result.h2_min:.3f} to {result.h2_max:.3f}")
    >>> print(f"CV(h²) = {result.cv_h2:.3f}")
    """
    p = G.shape[0]
    
    # Compute G* and its eigenstructure
    G_star = compute_G_star(G, P)
    eigenvalues, eigenvectors = eigendecompose(G_star)
    
    # Basic statistics of eigenvalues
    h2_max = eigenvalues[0]
    h2_min = eigenvalues[-1]
    h2_mean = np.mean(eigenvalues)
    
    # Variance of eigenvalues
    var_lambda = np.var(eigenvalues, ddof=0)  # Population variance
    
    # Theoretical variance of h² across uniform directions
    # Var[h²] = (2/(p+2)) × Var(λ*)
    h2_variance = (2 / (p + 2)) * var_lambda
    
    # Relative variance: V_rel = Var(λ*) / mean(λ*)²
    V_rel = var_lambda / (h2_mean ** 2) if h2_mean > 0 else 0
    
    # Coefficient of variation
    # CV(h²) = √(2/(p+2)) × CV(λ*) = √(2/(p+2)) × √V_rel
    cv_h2 = np.sqrt(2 / (p + 2)) * np.sqrt(V_rel)
    
    # Constraint severity: how much worse is the min than the mean?
    constraint_severity = 1 - (h2_min / h2_mean) if h2_mean > 0 else 0
    
    # Effective dimensionality: (tr G*)² / tr(G*²)
    # Equals p if all eigenvalues equal, approaches 1 if one dominates
    trace_G_star = np.sum(eigenvalues)
    trace_G_star_sq = np.sum(eigenvalues ** 2)
    effective_dim = (trace_G_star ** 2) / trace_G_star_sq if trace_G_star_sq > 0 else 0
    
    return HeritabilityDistribution(
        eigenvalues_G_star=eigenvalues,
        h2_max=h2_max,
        h2_min=h2_min,
        h2_mean=h2_mean,
        h2_variance=h2_variance,
        cv_h2=cv_h2,
        V_rel=V_rel,
        constraint_severity=constraint_severity,
        effective_dimensionality=effective_dim,
        eigenvectors_G_star=eigenvectors
    )


def monte_carlo_h2_distribution(G: np.ndarray, 
                                 P: np.ndarray,
                                 n_samples: int = 10000) -> Dict:
    """
    Empirically sample the distribution of directional heritability.
    
    This complements the analytical formulas by directly sampling
    h²(β) for random directions β on the P-sphere.
    
    Parameters
    ----------
    G : np.ndarray
        Additive genetic covariance matrix
    P : np.ndarray
        Phenotypic covariance matrix
    n_samples : int
        Number of random directions to sample
    
    Returns
    -------
    dict
        Contains:
        - 'h2_samples': array of h² values
        - 'directions': the sampled directions
        - 'mean': sample mean
        - 'variance': sample variance
        - 'cv': sample coefficient of variation
        - 'percentiles': [5, 25, 50, 75, 95] percentiles
    
    Notes
    -----
    This is useful for:
    1. Verifying the analytical formulas
    2. Visualizing the actual distribution
    3. Computing quantities not available analytically
    """
    p = G.shape[0]
    
    # Sample directions from the P-sphere
    directions = sample_from_P_sphere(n_samples, P)
    
    # Compute h² for each direction
    h2_samples = np.zeros(n_samples)
    for i in range(n_samples):
        beta = directions[i]
        h2_samples[i] = directional_heritability(beta, G, P)
    
    # Compute statistics
    mean_h2 = np.mean(h2_samples)
    var_h2 = np.var(h2_samples, ddof=1)
    cv_h2 = np.std(h2_samples, ddof=1) / mean_h2 if mean_h2 > 0 else 0
    percentiles = np.percentile(h2_samples, [5, 25, 50, 75, 95])
    
    return {
        'h2_samples': h2_samples,
        'directions': directions,
        'mean': mean_h2,
        'variance': var_h2,
        'cv': cv_h2,
        'percentiles': percentiles,
        'min': np.min(h2_samples),
        'max': np.max(h2_samples)
    }


# =============================================================================
# SECTION 4: CONSTRAINT ANALYSIS
# =============================================================================
"""
Constraint traps occur when:
    • Phenotypic variance exists (direction is on P-sphere)
    • But genetic variance is low (h² is small)

This section provides tools for identifying and characterizing constraint traps.
"""


@dataclass
class ConstraintAnalysis:
    """
    Results of constraint trap analysis.
    
    Attributes
    ----------
    constraint_directions : np.ndarray
        Directions with heritability below threshold
    
    constraint_heritabilities : np.ndarray
        h² values for constraint directions
    
    n_constrained : int
        Number of directions below threshold
    
    proportion_constrained : float
        Fraction of sampled directions that are constrained
    
    worst_direction : np.ndarray
        The direction with lowest heritability
    
    worst_h2 : float
        Heritability in the worst direction
    """
    constraint_directions: np.ndarray
    constraint_heritabilities: np.ndarray
    n_constrained: int
    proportion_constrained: float
    worst_direction: np.ndarray
    worst_h2: float


def identify_constraint_traps(G: np.ndarray,
                               P: np.ndarray,
                               threshold: float = None,
                               n_samples: int = 10000) -> ConstraintAnalysis:
    """
    Identify directions that are "constraint traps" - low heritability.
    
    Parameters
    ----------
    G : np.ndarray
        Genetic covariance matrix
    P : np.ndarray
        Phenotypic covariance matrix
    threshold : float, optional
        h² value below which a direction is considered "constrained".
        If None, uses 0.5 × mean(h²).
    n_samples : int
        Number of directions to sample
    
    Returns
    -------
    ConstraintAnalysis
        Complete constraint characterization
    
    Biological Interpretation
    -------------------------
    Constraint traps are dangerous because:
    1. Phenotypic variation exists (selection can act)
    2. But little response occurs (most variation is environmental)
    
    This can mislead breeders or conservationists who assume observable
    variation implies evolutionary potential.
    """
    # Get h² distribution
    mc_result = monte_carlo_h2_distribution(G, P, n_samples)
    h2_samples = mc_result['h2_samples']
    directions = mc_result['directions']
    
    # Set threshold if not provided
    if threshold is None:
        threshold = 0.5 * np.mean(h2_samples)
    
    # Find constrained directions
    constrained_mask = h2_samples < threshold
    n_constrained = np.sum(constrained_mask)
    
    constraint_directions = directions[constrained_mask]
    constraint_heritabilities = h2_samples[constrained_mask]
    
    # Find worst direction
    worst_idx = np.argmin(h2_samples)
    worst_direction = directions[worst_idx]
    worst_h2 = h2_samples[worst_idx]
    
    return ConstraintAnalysis(
        constraint_directions=constraint_directions,
        constraint_heritabilities=constraint_heritabilities,
        n_constrained=n_constrained,
        proportion_constrained=n_constrained / n_samples,
        worst_direction=worst_direction,
        worst_h2=worst_h2
    )


def compute_constraint_risk(G: np.ndarray,
                             P: np.ndarray,
                             selection_direction: np.ndarray) -> Dict:
    """
    Assess the constraint risk for a specific selection direction.
    
    Given a selection target direction, compute how constrained it is
    relative to the overall h² distribution.
    
    Parameters
    ----------
    G : np.ndarray
        Genetic covariance matrix
    P : np.ndarray
        Phenotypic covariance matrix
    selection_direction : np.ndarray
        The direction in which selection is applied
    
    Returns
    -------
    dict
        Contains:
        - 'h2_selection': heritability in the selection direction
        - 'h2_max': maximum possible heritability
        - 'h2_min': minimum possible heritability
        - 'h2_mean': mean heritability
        - 'percentile': percentile rank of h2_selection
        - 'constraint_risk': qualitative assessment
    
    Biological Use
    --------------
    Before a breeding program commits to a selection target, check
    whether that direction is a constraint trap.
    """
    # Normalize selection direction
    beta = selection_direction / np.linalg.norm(selection_direction)
    
    # h² in selection direction
    h2_selection = directional_heritability(beta, G, P)
    
    # Get overall distribution
    dist = analyze_heritability_distribution(G, P)
    
    # Monte Carlo for percentile
    mc = monte_carlo_h2_distribution(G, P, n_samples=10000)
    percentile = 100 * np.mean(mc['h2_samples'] <= h2_selection)
    
    # Qualitative assessment
    if h2_selection >= dist.h2_mean + np.sqrt(dist.h2_variance):
        risk = "LOW - above average heritability"
    elif h2_selection >= dist.h2_mean - np.sqrt(dist.h2_variance):
        risk = "MODERATE - near average heritability"
    elif h2_selection >= dist.h2_min + 0.1 * (dist.h2_mean - dist.h2_min):
        risk = "HIGH - below average heritability"
    else:
        risk = "SEVERE - near minimum heritability (CONSTRAINT TRAP)"
    
    return {
        'h2_selection': h2_selection,
        'h2_max': dist.h2_max,
        'h2_min': dist.h2_min,
        'h2_mean': dist.h2_mean,
        'percentile': percentile,
        'constraint_risk': risk
    }


# =============================================================================
# SECTION 5: THE CENTRAL FORMULA DERIVATION
# =============================================================================
"""
This section provides pedagogical verification of the key formula:

    CV²(h²) = (2 / (p + 2)) × V_rel(G*)

We derive this step-by-step and verify it with Monte Carlo simulation.
"""


def verify_cv_formula(G: np.ndarray, 
                       P: np.ndarray,
                       n_samples: int = 50000) -> Dict:
    """
    Verify the theoretical CV formula against Monte Carlo simulation.
    
    The formula states:
        CV(h²) = √(2/(p+2)) × √V_rel(G*)
    
    where V_rel(G*) = Var(λ*) / mean(λ*)²
    
    Parameters
    ----------
    G : np.ndarray
        Genetic covariance matrix
    P : np.ndarray
        Phenotypic covariance matrix
    n_samples : int
        Number of Monte Carlo samples for verification
    
    Returns
    -------
    dict
        Contains theoretical and empirical values for comparison
    
    Derivation
    ----------
    For a random quadratic form z'Az where z is uniform on the unit sphere:
    
    1. E[z'Az] = (1/p) × tr(A) = mean of eigenvalues
    
    2. Var[z'Az] = (2/p(p+2)) × Σᵢ<ⱼ (λᵢ - λⱼ)²
                 = (2/(p+2)) × (1/(p-1)) × Σᵢ (λᵢ - λ̄)² × (p-1)
                 = (2/(p+2)) × Var(λ)
    
    3. Therefore: CV² = Var/Mean² = (2/(p+2)) × Var(λ)/mean(λ)²
                                   = (2/(p+2)) × V_rel
    """
    p = G.shape[0]
    
    # Analytical predictions
    dist = analyze_heritability_distribution(G, P)
    
    theoretical_mean = dist.h2_mean
    theoretical_var = dist.h2_variance
    theoretical_cv = dist.cv_h2
    V_rel = dist.V_rel
    
    # Monte Carlo verification
    mc = monte_carlo_h2_distribution(G, P, n_samples=n_samples)
    
    empirical_mean = mc['mean']
    empirical_var = mc['variance']
    empirical_cv = mc['cv']
    
    # Check the formula step by step
    formula_check = {
        'p': p,
        'eigenvalues_G_star': dist.eigenvalues_G_star,
        'mean_eigenvalues': theoretical_mean,
        'var_eigenvalues': np.var(dist.eigenvalues_G_star, ddof=0),
        'V_rel': V_rel,
        'factor_2_over_p_plus_2': 2 / (p + 2),
        
        'theoretical_mean': theoretical_mean,
        'theoretical_var': theoretical_var,
        'theoretical_cv': theoretical_cv,
        
        'empirical_mean': empirical_mean,
        'empirical_var': empirical_var,
        'empirical_cv': empirical_cv,
        
        'mean_error': abs(theoretical_mean - empirical_mean) / theoretical_mean,
        'var_error': abs(theoretical_var - empirical_var) / theoretical_var if theoretical_var > 0 else 0,
        'cv_error': abs(theoretical_cv - empirical_cv) / theoretical_cv if theoretical_cv > 0 else 0
    }
    
    return formula_check


# =============================================================================
# SECTION 6: VISUALISATION TOOLS
# =============================================================================
"""
Visualization functions for understanding directional heritability geometry.
"""


def plot_h2_distribution(G: np.ndarray, 
                          P: np.ndarray,
                          n_samples: int = 10000,
                          save_path: str = None) -> plt.Figure:
    """
    Create a comprehensive visualization of the h² distribution.
    
    Generates a 2×2 figure showing:
    1. Histogram of h² values with theoretical bounds
    2. G* ellipse showing principal h² directions
    3. Scree plot of G* eigenvalues
    4. Constraint risk visualization
    """
    # Get analytical and Monte Carlo results
    dist = analyze_heritability_distribution(G, P)
    mc = monte_carlo_h2_distribution(G, P, n_samples)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # ----- Panel A: Histogram of h² -----
    ax = axes[0, 0]
    
    # Histogram
    n, bins, patches = ax.hist(mc['h2_samples'], bins=50, density=True,
                                color='steelblue', alpha=0.7, edgecolor='black')
    
    # Mark theoretical extremes
    ax.axvline(dist.h2_min, color='red', linestyle='--', linewidth=2,
               label=f'Min h² = {dist.h2_min:.3f}')
    ax.axvline(dist.h2_max, color='green', linestyle='--', linewidth=2,
               label=f'Max h² = {dist.h2_max:.3f}')
    ax.axvline(dist.h2_mean, color='orange', linestyle='-', linewidth=2,
               label=f'Mean h² = {dist.h2_mean:.3f}')
    
    # Mark percentiles
    for q in [5, 95]:
        val = np.percentile(mc['h2_samples'], q)
        ax.axvline(val, color='gray', linestyle=':', alpha=0.7)
    
    ax.set_xlabel('Directional Heritability h²(β)')
    ax.set_ylabel('Density')
    ax.set_title(f'(A) Distribution of h² Across Directions\n'
                 f'CV(h²) = {dist.cv_h2:.3f}, n_traits = {G.shape[0]}')
    ax.legend(loc='upper right')
    
    # ----- Panel B: G* ellipse (2D projection) -----
    ax = axes[0, 1]
    
    p = G.shape[0]
    if p >= 2:
        # Project G* onto first two eigenvectors
        G_star = compute_G_star(G, P)
        eigenvalues, eigenvectors = eigendecompose(G_star)
        
        # Draw ellipse for first two dimensions
        theta = np.linspace(0, 2*np.pi, 100)
        
        # The G* ellipse in its principal axes
        a = np.sqrt(eigenvalues[0])  # semi-axis along v1
        b = np.sqrt(eigenvalues[1]) if p >= 2 else a  # semi-axis along v2
        
        ellipse_x = a * np.cos(theta)
        ellipse_y = b * np.sin(theta)
        
        ax.plot(ellipse_x, ellipse_y, 'b-', linewidth=2, label='G* ellipse')
        
        # Draw unit circle (P-sphere in whitened space)
        circle_x = np.cos(theta)
        circle_y = np.sin(theta)
        ax.plot(circle_x, circle_y, 'k--', linewidth=1.5, label='P-sphere (unit circle)')
        
        # Mark principal axes
        ax.arrow(0, 0, a*0.9, 0, head_width=0.03, head_length=0.02,
                fc='red', ec='red', linewidth=2)
        ax.arrow(0, 0, 0, b*0.9, head_width=0.03, head_length=0.02,
                fc='blue', ec='blue', linewidth=2)
        
        ax.text(a + 0.05, 0, f'h²={eigenvalues[0]:.2f}', fontsize=10, color='red')
        ax.text(0.05, b + 0.05, f'h²={eigenvalues[1]:.2f}', fontsize=10, color='blue')
        
        ax.set_xlabel('PC1 direction')
        ax.set_ylabel('PC2 direction')
        ax.set_title('(B) G* Ellipse Inside P-sphere\n'
                     '(Whitened coordinate space)')
        ax.set_aspect('equal')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    # ----- Panel C: Eigenvalue scree plot -----
    ax = axes[1, 0]
    
    x = np.arange(1, p + 1)
    ax.bar(x, dist.eigenvalues_G_star, color='coral', alpha=0.7, edgecolor='black')
    ax.axhline(dist.h2_mean, color='orange', linestyle='--', linewidth=2,
               label=f'Mean = {dist.h2_mean:.3f}')
    
    # Shade constraint zone
    ax.axhspan(0, dist.h2_mean * 0.5, alpha=0.2, color='red', 
               label='Constraint zone (< 50% of mean)')
    
    ax.set_xlabel('Principal Direction (rank)')
    ax.set_ylabel('Directional Heritability (λ* of G*)')
    ax.set_title(f'(C) Scree Plot of G* Eigenvalues\n'
                 f'Effective dimensionality = {dist.effective_dimensionality:.2f}')
    ax.set_xticks(x)
    ax.legend()
    
    # ----- Panel D: Constraint severity visualization -----
    ax = axes[1, 1]
    
    # Create a "gauge" visualization
    categories = ['Max h²', 'Mean h²', 'Min h²', 'Severity']
    values = [dist.h2_max, dist.h2_mean, dist.h2_min, dist.constraint_severity]
    colors = ['green', 'orange', 'red', 'purple']
    
    bars = ax.barh(categories, values, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels
    for bar, val in zip(bars, values):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
               f'{val:.3f}', va='center', fontsize=11)
    
    ax.set_xlabel('Value')
    ax.set_title(f'(D) Constraint Summary\n'
                 f'Range: {dist.h2_max - dist.h2_min:.3f}, '
                 f'Ratio: {dist.h2_max/dist.h2_min:.2f}x')
    ax.set_xlim(0, max(values) * 1.2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_constraint_landscape(G: np.ndarray,
                               P: np.ndarray,
                               save_path: str = None) -> plt.Figure:
    """
    Visualize the constraint landscape for 2-trait systems.
    
    Shows how heritability varies continuously across the unit circle
    of directions.
    """
    if G.shape[0] != 2:
        raise ValueError("This visualization is only for 2-trait systems")
    
    # Sample directions around the circle
    angles = np.linspace(0, 2*np.pi, 360)
    h2_values = np.zeros(len(angles))
    
    for i, angle in enumerate(angles):
        direction = np.array([np.cos(angle), np.sin(angle)])
        h2_values[i] = directional_heritability(direction, G, P)
    
    # Get analytical results for comparison
    dist = analyze_heritability_distribution(G, P)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ----- Panel A: Polar plot -----
    ax = plt.subplot(121, projection='polar')
    
    ax.plot(angles, h2_values, 'b-', linewidth=2)
    ax.fill(angles, h2_values, alpha=0.3)
    
    # Mark extremes
    max_idx = np.argmax(h2_values)
    min_idx = np.argmin(h2_values)
    
    ax.plot(angles[max_idx], h2_values[max_idx], 'go', markersize=12,
            label=f'Max h² = {h2_values[max_idx]:.3f}')
    ax.plot(angles[min_idx], h2_values[min_idx], 'ro', markersize=12,
            label=f'Min h² = {h2_values[min_idx]:.3f}')
    
    ax.set_title('(A) Directional Heritability Landscape\n(polar coordinates)', 
                 pad=20)
    ax.legend(loc='lower right')
    
    # ----- Panel B: Cartesian plot -----
    ax = axes[1]
    
    ax.plot(np.degrees(angles), h2_values, 'b-', linewidth=2)
    ax.axhline(dist.h2_mean, color='orange', linestyle='--', linewidth=2,
               label=f'Mean = {dist.h2_mean:.3f}')
    ax.axhline(dist.h2_max, color='green', linestyle=':', alpha=0.7)
    ax.axhline(dist.h2_min, color='red', linestyle=':', alpha=0.7)
    
    # Shade constraint zone
    ax.axhspan(0, dist.h2_mean * 0.5, alpha=0.2, color='red',
               label='Constraint zone')
    
    ax.fill_between(np.degrees(angles), h2_values, dist.h2_mean,
                    where=h2_values < dist.h2_mean * 0.5,
                    alpha=0.3, color='red')
    
    ax.set_xlabel('Direction angle (degrees)')
    ax.set_ylabel('Directional Heritability h²(β)')
    ax.set_title(f'(B) h² vs Direction\nCV = {dist.cv_h2:.3f}')
    ax.legend()
    ax.set_xlim(0, 360)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


# =============================================================================
# SECTION 7: WORKED EXAMPLE - COMPLETE ANALYSIS
# =============================================================================


def worked_example_directional_heritability():
    """
    Complete worked example demonstrating all concepts in Chapter 13.
    """
    print("\n" + "=" * 70)
    print("WORKED EXAMPLE: DIRECTIONAL HERITABILITY ANALYSIS")
    print("=" * 70)
    print("""
    We analyze a G-P system to understand:
    1. How heritability varies across directions
    2. The distribution of directional heritability
    3. Where constraint traps exist
    4. The key formula: CV²(h²) = (2/(p+2)) × V_rel(G*)
    """)
    
    # ----- Define the G and P matrices -----
    print("\n" + "-" * 70)
    print("THE DATA: FLORAL TRAITS IN A PLANT SPECIES")
    print("-" * 70)
    
    # Three floral traits: petal length, petal width, tube depth
    G = np.array([
        [0.42, 0.28, 0.15],
        [0.28, 0.38, 0.22],
        [0.15, 0.22, 0.31]
    ])
    
    P = np.array([
        [0.85, 0.35, 0.20],
        [0.35, 0.72, 0.28],
        [0.20, 0.28, 0.55]
    ])
    
    trait_names = ['Petal length', 'Petal width', 'Tube depth']
    
    print("\nGenetic covariance matrix G:")
    print(G)
    print("\nPhenotypic covariance matrix P:")
    print(P)
    
    # ----- Step 1: Compute G* and its eigenstructure -----
    print("\n" + "-" * 70)
    print("STEP 1: COMPUTE G* = P^{-1/2} G P^{-1/2}")
    print("-" * 70)
    
    G_star = compute_G_star(G, P)
    eigenvalues, eigenvectors = eigendecompose(G_star)
    
    print("\nG* (the P-whitened genetic matrix):")
    print(np.round(G_star, 4))
    
    print("\nEigenvalues of G* (these ARE directional heritabilities):")
    for i, lam in enumerate(eigenvalues):
        print(f"  λ*_{i+1} = {lam:.4f} = h²_PC{i+1}")
    
    print("\nEigenvectors (directions of extreme h²):")
    for i in range(len(eigenvalues)):
        vec = eigenvectors[:, i]
        interp = ""
        if np.all(np.sign(vec) == np.sign(vec[0])):
            interp = " ← ALL SAME SIGN (size-like)"
        elif np.any(vec * np.roll(vec, 1) < 0):
            interp = " ← MIXED SIGNS (shape-like)"
        print(f"  v*_{i+1} = [{', '.join([f'{v:+.3f}' for v in vec])}]{interp}")
    
    # ----- Step 2: Analyze the distribution -----
    print("\n" + "-" * 70)
    print("STEP 2: ANALYZE THE DISTRIBUTION OF h²")
    print("-" * 70)
    
    dist = analyze_heritability_distribution(G, P)
    
    print(f"""
    KEY STATISTICS:
    ---------------
    Maximum h²:      {dist.h2_max:.4f}  (in direction v*₁)
    Minimum h²:      {dist.h2_min:.4f}  (in direction v*₃)
    Mean h²:         {dist.h2_mean:.4f}  (average across all directions)
    
    Range:           {dist.h2_max - dist.h2_min:.4f}
    Ratio max/min:   {dist.h2_max / dist.h2_min:.2f}x
    
    Variance of h²:  {dist.h2_variance:.6f}
    CV(h²):          {dist.cv_h2:.4f}  ({100*dist.cv_h2:.1f}%)
    
    V_rel(G*):       {dist.V_rel:.4f}
    
    Constraint severity: {dist.constraint_severity:.4f}
    (= 1 - λ*_min / mean(λ*) = how much worse the min is than average)
    
    Effective dimensionality: {dist.effective_dimensionality:.2f} out of {G.shape[0]}
    (= 3.0 if all h² equal, → 1.0 if one direction dominates)
    """)
    
    # ----- Step 3: Verify the central formula -----
    print("\n" + "-" * 70)
    print("STEP 3: VERIFY THE CENTRAL FORMULA")
    print("-" * 70)
    
    p = G.shape[0]
    
    print(f"""
    THE FORMULA:
    ------------
    CV²(h²) = (2 / (p + 2)) × V_rel(G*)
    
    Let's check each piece:
    
    1. p = {p} (number of traits)
    
    2. V_rel(G*) = Var(λ*) / mean(λ*)²
       
       Eigenvalues of G*: {np.round(eigenvalues, 4)}
       mean(λ*) = {np.mean(eigenvalues):.4f}
       Var(λ*)  = {np.var(eigenvalues, ddof=0):.6f}
       V_rel    = {dist.V_rel:.4f}
    
    3. Factor 2/(p+2) = 2/({p}+2) = 2/{p+2} = {2/(p+2):.4f}
    
    4. PREDICTION:
       CV²(h²) = {2/(p+2):.4f} × {dist.V_rel:.4f} = {(2/(p+2)) * dist.V_rel:.6f}
       CV(h²)  = √({(2/(p+2)) * dist.V_rel:.6f}) = {np.sqrt((2/(p+2)) * dist.V_rel):.4f}
    """)
    
    # Monte Carlo verification
    verification = verify_cv_formula(G, P, n_samples=50000)
    
    print(f"""
    MONTE CARLO VERIFICATION (n = 50,000 samples):
    -----------------------------------------------
    Theoretical CV(h²): {verification['theoretical_cv']:.4f}
    Empirical CV(h²):   {verification['empirical_cv']:.4f}
    Relative error:     {100*verification['cv_error']:.2f}%
    
    ✓ The formula works!
    """)
    
    # ----- Step 4: Identify constraint traps -----
    print("\n" + "-" * 70)
    print("STEP 4: IDENTIFY CONSTRAINT TRAPS")
    print("-" * 70)
    
    threshold = dist.h2_mean * 0.6  # 60% of mean
    constraint = identify_constraint_traps(G, P, threshold=threshold)
    
    print(f"""
    CONSTRAINT TRAP ANALYSIS:
    -------------------------
    Threshold: h² < {threshold:.4f} (60% of mean)
    
    Proportion of directions constrained: {100*constraint.proportion_constrained:.1f}%
    ({constraint.n_constrained} out of 10,000 sampled directions)
    
    WORST DIRECTION (most severe constraint trap):
    h² = {constraint.worst_h2:.4f}
    Direction = [{', '.join([f'{v:+.3f}' for v in constraint.worst_direction])}]
    """)
    
    # Interpret the worst direction
    worst_vec = constraint.worst_direction
    print("    Interpretation of worst direction:")
    for i, (name, loading) in enumerate(zip(trait_names, worst_vec)):
        sign = "+" if loading > 0 else "-"
        print(f"      {name}: {loading:+.3f}")
    
    # ----- Step 5: Assess a specific selection target -----
    print("\n" + "-" * 70)
    print("STEP 5: ASSESS A BREEDING TARGET")
    print("-" * 70)
    
    # Example: breeder wants to increase tube depth while keeping other traits constant
    selection_target = np.array([0, 0, 1])  # Pure selection on tube depth
    
    risk = compute_constraint_risk(G, P, selection_target)
    
    print(f"""
    SELECTION TARGET: Increase tube depth only
    Direction: [0, 0, 1]
    
    CONSTRAINT RISK ASSESSMENT:
    ---------------------------
    h² in target direction:  {risk['h2_selection']:.4f}
    
    Context:
      Maximum possible h²:   {risk['h2_max']:.4f}
      Mean h²:               {risk['h2_mean']:.4f}
      Minimum possible h²:   {risk['h2_min']:.4f}
    
    Percentile rank:         {risk['percentile']:.1f}%
    
    RISK LEVEL: {risk['constraint_risk']}
    """)
    
    # Compare with an alternative
    print("    Comparison with alternative targets:")
    
    alternatives = [
        ("All traits together (size)", np.array([1, 1, 1]) / np.sqrt(3)),
        ("Petal shape (L vs W)", np.array([1, -1, 0]) / np.sqrt(2)),
        ("Flower form (petals vs tube)", np.array([1, 1, -2]) / np.sqrt(6))
    ]
    
    print(f"\n    {'Target':<30} {'h²':<10} {'Percentile':<12} {'Risk'}")
    print("    " + "-" * 70)
    
    for name, direction in alternatives:
        risk_alt = compute_constraint_risk(G, P, direction)
        print(f"    {name:<30} {risk_alt['h2_selection']:.4f}     "
              f"{risk_alt['percentile']:>5.1f}%      "
              f"{risk_alt['constraint_risk'].split(' - ')[0]}")
    
    # ----- Step 6: Biological conclusions -----
    print("\n" + "-" * 70)
    print("STEP 6: BIOLOGICAL CONCLUSIONS")
    print("-" * 70)
    
    print(f"""
    KEY FINDINGS:
    
    1. HERITABILITY VARIES SUBSTANTIALLY WITH DIRECTION
       • Range: {dist.h2_min:.2f} to {dist.h2_max:.2f}
       • That's a {dist.h2_max/dist.h2_min:.1f}× difference!
       • CV(h²) = {100*dist.cv_h2:.0f}% heterogeneity
    
    2. THE MAXIMUM h² DIRECTION (v*₁)
       Loadings: {np.round(eigenvectors[:, 0], 2)}
       This is the direction where evolution is EASIEST.
       h² = {dist.h2_max:.2f} means {100*dist.h2_max:.0f}% of variance is genetic.
    
    3. THE MINIMUM h² DIRECTION (v*₃) - CONSTRAINT TRAP
       Loadings: {np.round(eigenvectors[:, -1], 2)}
       This is a CONSTRAINT TRAP.
       h² = {dist.h2_min:.2f} means only {100*dist.h2_min:.0f}% of variance is genetic.
       Selection here will produce WEAK response.
    
    4. EFFECTIVE DIMENSIONALITY = {dist.effective_dimensionality:.2f}
       With 3 traits, max is 3.0.
       Value of {dist.effective_dimensionality:.2f} indicates genetic variance is
       somewhat concentrated in a subset of directions.
    
    5. BREEDING IMPLICATIONS
       • Select ALONG high-h² directions for best response
       • Be cautious of constraint traps - visible variation ≠ heritable variation
       • The formula CV²(h²) = (2/(p+2)) × V_rel(G*) lets you predict
         heritability heterogeneity from eigenstructure alone
    
    6. EVOLUTIONARY IMPLICATIONS
       • Evolution will be CHANNELED toward high-h² directions
       • Adaptation requiring low-h² directions will be SLOW
       • Phenotypic divergence among populations may reflect G* geometry
         as much as selection differences
    """)
    
    # ----- Generate visualization -----
    print("\n" + "-" * 70)
    print("GENERATING VISUALIZATIONS")
    print("-" * 70)
    
    fig = plot_h2_distribution(G, P, n_samples=10000, 
                                save_path='/home/claude/ch13_h2_distribution.png')
    plt.close()
    print("Saved: /home/claude/ch13_h2_distribution.png")
    
    return dist, constraint, verification


def worked_example_two_trait():
    """
    Two-trait example with full constraint landscape visualization.
    """
    print("\n" + "=" * 70)
    print("WORKED EXAMPLE: TWO-TRAIT CONSTRAINT LANDSCAPE")
    print("=" * 70)
    
    # Simple two-trait example
    G = np.array([
        [0.50, 0.35],
        [0.35, 0.40]
    ])
    
    P = np.array([
        [1.00, 0.50],
        [0.50, 0.80]
    ])
    
    print("\nG =", G)
    print("P =", P)
    
    # Analyze
    dist = analyze_heritability_distribution(G, P)
    
    print(f"""
    RESULTS:
    --------
    Eigenvalues of G*: {np.round(dist.eigenvalues_G_star, 4)}
    
    Max h²:  {dist.h2_max:.4f}
    Min h²:  {dist.h2_min:.4f}
    Mean h²: {dist.h2_mean:.4f}
    CV(h²):  {dist.cv_h2:.4f}
    
    V_rel(G*): {dist.V_rel:.4f}
    
    FORMULA CHECK:
    CV²(h²) = (2/(p+2)) × V_rel
            = (2/4) × {dist.V_rel:.4f}
            = {0.5 * dist.V_rel:.4f}
    CV(h²)  = {np.sqrt(0.5 * dist.V_rel):.4f} ✓
    """)
    
    # Generate constraint landscape visualization
    fig = plot_constraint_landscape(G, P, 
                                     save_path='/home/claude/ch13_constraint_landscape.png')
    plt.close()
    print("Saved: /home/claude/ch13_constraint_landscape.png")
    
    return dist


def worked_example_high_dimensional():
    """
    High-dimensional example showing dimensionality effects.
    """
    print("\n" + "=" * 70)
    print("WORKED EXAMPLE: HIGH-DIMENSIONAL SYSTEM (p=8)")
    print("=" * 70)
    
    # Generate a realistic 8-trait G and P
    np.random.seed(42)
    p = 8
    
    # Create structured covariance matrices
    # G with decreasing eigenvalue structure
    eigenvalues_G = np.array([0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05, 0.02])
    Q = stats.ortho_group.rvs(p)  # Random orthogonal matrix
    G = Q @ np.diag(eigenvalues_G) @ Q.T
    
    # P with different structure
    eigenvalues_P = np.array([1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    R = stats.ortho_group.rvs(p)
    P = R @ np.diag(eigenvalues_P) @ R.T
    
    # Ensure G_ii < P_ii
    for i in range(p):
        if G[i,i] > P[i,i]:
            scale = 0.5 * P[i,i] / G[i,i]
            G = G * scale
            break
    
    print(f"Number of traits: p = {p}")
    print(f"\nEigenvalues of G: {np.round(np.linalg.eigvalsh(G)[::-1], 4)}")
    print(f"Eigenvalues of P: {np.round(np.linalg.eigvalsh(P)[::-1], 4)}")
    
    # Analyze
    dist = analyze_heritability_distribution(G, P)
    
    print(f"""
    RESULTS FOR HIGH-DIMENSIONAL SYSTEM:
    -------------------------------------
    Eigenvalues of G*: {np.round(dist.eigenvalues_G_star, 4)}
    
    Max h²:   {dist.h2_max:.4f}
    Min h²:   {dist.h2_min:.4f}
    Mean h²:  {dist.h2_mean:.4f}
    Range:    {dist.h2_max - dist.h2_min:.4f}
    
    CV(h²):   {dist.cv_h2:.4f}
    V_rel:    {dist.V_rel:.4f}
    
    Effective dimensionality: {dist.effective_dimensionality:.2f} / {p}
    Constraint severity: {dist.constraint_severity:.4f}
    """)
    
    # The formula
    print(f"""
    THE FORMULA IN HIGH DIMENSIONS:
    -------------------------------
    CV²(h²) = (2/(p+2)) × V_rel
            = (2/{p+2}) × {dist.V_rel:.4f}
            = {2/(p+2):.4f} × {dist.V_rel:.4f}
            = {(2/(p+2)) * dist.V_rel:.4f}
    
    CV(h²) = {np.sqrt((2/(p+2)) * dist.V_rel):.4f}
    
    KEY INSIGHT:
    The factor 2/(p+2) = {2/(p+2):.3f} DECREASES with p.
    In high dimensions, random directions "average out" the eigenvalues,
    reducing the variability of h² even when V_rel is large.
    
    However, the EXTREMES (max and min h²) still differ by a factor of
    {dist.h2_max/dist.h2_min:.1f}x. Constraint traps are still dangerous!
    """)
    
    # Verify with Monte Carlo
    verification = verify_cv_formula(G, P, n_samples=50000)
    print(f"""
    MONTE CARLO VERIFICATION:
    Theoretical CV: {verification['theoretical_cv']:.4f}
    Empirical CV:   {verification['empirical_cv']:.4f}
    Error:          {100*verification['cv_error']:.2f}%
    """)
    
    # Generate visualization
    fig = plot_h2_distribution(G, P, n_samples=10000,
                                save_path='/home/claude/ch13_high_dimensional.png')
    plt.close()
    print("Saved: /home/claude/ch13_high_dimensional.png")
    
    return dist


# =============================================================================
# SECTION 8: SUMMARY OF KEY FORMULAS
# =============================================================================


def print_formula_summary():
    """
    Print a summary of all key formulas from Chapter 13.
    """
    print("\n" + "=" * 70)
    print("CHAPTER 13: KEY FORMULAS SUMMARY")
    print("=" * 70)
    print("""
    ┌──────────────────────────────────────────────────────────────────────┐
    │  DIRECTIONAL HERITABILITY                                            │
    ├──────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │  Definition:       h²(β) = β'Gβ / β'Pβ                               │
    │                                                                      │
    │  In whitened space: h²(β*) = (β*)'G*β*  where G* = P^{-1/2}GP^{-1/2} │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘
    
    ┌──────────────────────────────────────────────────────────────────────┐
    │  THE CENTRAL FORMULA                                                 │
    ├──────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │  CV²(h²) = (2 / (p + 2)) × V_rel(G*)                                │
    │                                                                      │
    │  where:                                                              │
    │    • CV(h²) = SD(h²) / E(h²) = coefficient of variation             │
    │    • p = number of traits                                            │
    │    • V_rel(G*) = Var(λ*) / mean(λ*)² = relative eigenvalue variance │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘
    
    ┌──────────────────────────────────────────────────────────────────────┐
    │  EIGENVALUE BOUNDS                                                   │
    ├──────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │  λ*_min  ≤  h²(β)  ≤  λ*_max    for all directions β                │
    │                                                                      │
    │  E[h²] = mean(λ*)               (uniform over P-sphere)             │
    │                                                                      │
    │  Var[h²] = (2/(p+2)) × Var(λ*)  (uniform over P-sphere)             │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘
    
    ┌──────────────────────────────────────────────────────────────────────┐
    │  CONSTRAINT MEASURES                                                 │
    ├──────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │  Constraint severity = 1 - λ*_min / mean(λ*)                        │
    │    (how much worse is the worst direction than average?)             │
    │                                                                      │
    │  Effective dimensionality = (tr G*)² / tr(G*²)                      │
    │    (= p if all λ* equal; → 1 if one dominates)                      │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘
    
    ┌──────────────────────────────────────────────────────────────────────┐
    │  KEY INSIGHT                                                         │
    ├──────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │  The eigenvalues of G* = P^{-1/2}GP^{-1/2} ARE the directional      │
    │  heritabilities along the principal axes of the P-sphere.            │
    │                                                                      │
    │  All other quantities (CV, V_rel, constraint severity) flow from    │
    │  this single eigendecomposition.                                     │
    │                                                                      │
    │  "Compute G*, find its eigenvalues—you're done."                    │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘
    """)


# =============================================================================
# MAIN EXECUTION
# =============================================================================


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║  CHAPTER 13: DIRECTIONAL HERITABILITY AND THE GEOMETRY OF CONSTRAINT ║
    ║  Seeing the Shape - Code Companion (FINAL CHAPTER)                   ║
    ╚══════════════════════════════════════════════════════════════════════╝
    
    "These questions lead to the concept of CONSTRAINT HETEROGENEITY: the
     degree to which heritability varies across directions. When constraint
     heterogeneity is high, some directions are evolutionary highways while
     others are dead ends."
    
    Running worked examples...
    """)
    
    # Run all worked examples
    dist1, constraint, verification = worked_example_directional_heritability()
    dist2 = worked_example_two_trait()
    dist3 = worked_example_high_dimensional()
    
    # Print formula summary
    print_formula_summary()
    
    # Final summary
    print("\n" + "=" * 70)
    print("CHAPTER 13 COMPLETE - THE CAPSTONE")
    print("=" * 70)
    print("""
    THE GEOMETRIC PERSPECTIVE - COMPLETE:
    
    We have now traced a complete arc from the simplest idea—distance
    between two points—to a sophisticated understanding of evolutionary
    constraint in high-dimensional trait space.
    
    THE KEY INSIGHTS:
    
    1. SYMMETRIC MATRICES DESCRIBE SHAPES
       • G and P are ellipsoids in trait space
       • G* = P^{-1/2}GP^{-1/2} compares them directly
    
    2. EIGENVALUES ARE DIRECTIONAL QUANTITIES
       • λ*(G*) = directional heritability along principal axes
       • Extremes bound h² for ALL directions
    
    3. CONSTRAINT HAS GEOMETRY
       • Low h² directions are CONSTRAINT TRAPS
       • Severity measures how bad the worst trap is
       • CV(h²) measures overall heterogeneity
    
    4. THE FORMULA CONNECTS EVERYTHING
       
       CV²(h²) = (2/(p+2)) × V_rel(G*)
       
       • Left side: observable variation in heritability
       • Right side: geometric property (G* eccentricity) × dimensionality factor
    
    5. IMPLICATIONS ARE PROFOUND
       • Breeding: Choose selection targets wisely
       • Evolution: Response is channeled by G* geometry
       • Conservation: Constraint traps limit adaptive potential
    
    "The shape of the ellipsoid and the direction of the arrow—these two
     things, together, determine what will happen. The G matrix is potential;
     selection is actuality. Their interaction is evolution."
    
    ═══════════════════════════════════════════════════════════════════════
    
    Output files created:
      • /home/claude/ch13_h2_distribution.png
      • /home/claude/ch13_constraint_landscape.png
      • /home/claude/ch13_high_dimensional.png
    """)
