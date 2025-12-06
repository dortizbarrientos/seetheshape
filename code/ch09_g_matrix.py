#!/usr/bin/env python3
"""
==============================================================================
CHAPTER 9: THE G MATRIX AND THE GENETIC ELLIPSOID
==============================================================================

From "Seeing the Shape: A Geometric Introduction to Multivariate 
Quantitative Genetics" by Daniel Ortiz-Barrientos

This script implements the geometric tools for understanding the additive 
genetic covariance matrix G—the central object of multivariate quantitative 
genetics.

THE CORE INSIGHT:
    The G matrix is not merely a table of numbers. It is a SHAPE—an ellipsoid
    in trait space that determines how populations can and cannot evolve.
    
    "The shape of the ellipsoid and the direction of the arrow—these two 
    things, together, determine what will happen. The G matrix is potential;
    selection is actuality. Their interaction is evolution."

WHAT YOU WILL LEARN:
    1. How to visualize G as an ellipsoid in trait space
    2. How to find g_max—the "line of least evolutionary resistance"
    3. How the breeder's equation Δz̄ = Gβ deflects response toward g_max
    4. How to compute evolvability and respondability
    5. How to quantify constraint via eigenvalue eccentricity
    6. How to compute effective dimensionality

PREREQUISITES:
    - Chapter 7 (Eigendecomposition)
    - Chapter 8 (Whitening and the P-sphere)

Author: Code companion to Ortiz-Barrientos (2025)
License: CC BY-NC-SA 4.0
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
import matplotlib.transforms as transforms
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
import warnings

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
np.set_printoptions(precision=4, suppress=True)


# =============================================================================
# SECTION 1: DATA STRUCTURES FOR G MATRIX ANALYSIS
# =============================================================================

@dataclass
class GMatrixAnalysis:
    """
    Complete analysis of an additive genetic covariance matrix.
    
    This dataclass packages all the key quantities derived from 
    eigendecomposition of G, providing a structured way to access
    the geometric properties of genetic variation.
    
    Attributes
    ----------
    G : np.ndarray
        The original additive genetic covariance matrix
    eigenvalues : np.ndarray
        Genetic variances along principal axes (sorted descending)
    eigenvectors : np.ndarray
        Principal axes of the genetic ellipsoid (columns)
    g_max : np.ndarray
        First eigenvector—direction of maximum genetic variance
    g_min : np.ndarray
        Last eigenvector—direction of minimum genetic variance
    total_variance : float
        Trace of G—sum of genetic variances across all traits
    generalized_variance : float
        Determinant of G—"volume" of the genetic ellipsoid
    effective_dimensionality : float
        How many "independent" directions of genetic variation exist
    eccentricity : float
        Relative variance of eigenvalues—how elongated is the ellipsoid
    condition_number : float
        Ratio of largest to smallest eigenvalue—numerical stability
    """
    G: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    g_max: np.ndarray
    g_min: np.ndarray
    total_variance: float
    generalized_variance: float
    effective_dimensionality: float
    eccentricity: float
    condition_number: float


def analyze_G_matrix(G: np.ndarray, check_validity: bool = True) -> GMatrixAnalysis:
    """
    Perform complete eigenanalysis of a genetic covariance matrix.
    
    This function extracts all the geometric information encoded in G:
    the principal axes, the variance along each axis, and summary 
    statistics that quantify constraint.
    
    Parameters
    ----------
    G : np.ndarray
        Additive genetic covariance matrix (p × p, symmetric)
    check_validity : bool
        If True, verify G is symmetric and positive semi-definite
        
    Returns
    -------
    GMatrixAnalysis
        Dataclass containing all derived quantities
        
    Mathematical Background
    -----------------------
    The eigendecomposition G = VΛV^T reveals:
    
        - V: Matrix whose columns are eigenvectors (principal axes)
        - Λ: Diagonal matrix of eigenvalues (genetic variances along axes)
        
    The eigenvector corresponding to the largest eigenvalue is g_max—
    the direction of maximum genetic variance, also called the 
    "line of least evolutionary resistance" (Schluter 1996).
    
    Example
    -------
    >>> G = np.array([[1.0, 0.8], [0.8, 1.0]])
    >>> analysis = analyze_G_matrix(G)
    >>> print(f"g_max points at {np.degrees(np.arctan2(analysis.g_max[1], analysis.g_max[0])):.1f}°")
    """
    G = np.asarray(G, dtype=float)
    p = G.shape[0]
    
    if check_validity:
        # Check symmetry
        if not np.allclose(G, G.T):
            raise ValueError("G matrix must be symmetric")
        
        # Check positive semi-definiteness (allow small negative eigenvalues from numerical error)
        min_eigenvalue = np.min(np.linalg.eigvalsh(G))
        if min_eigenvalue < -1e-10:
            warnings.warn(f"G has negative eigenvalue ({min_eigenvalue:.6f}). "
                         "This may indicate estimation error.")
    
    # Eigendecomposition using eigh (optimized for symmetric matrices)
    # eigh returns eigenvalues in ASCENDING order
    eigenvalues, eigenvectors = np.linalg.eigh(G)
    
    # Sort in DESCENDING order (largest first—this is the convention)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Extract g_max and g_min
    g_max = eigenvectors[:, 0]      # First column: largest eigenvalue
    g_min = eigenvectors[:, -1]     # Last column: smallest eigenvalue
    
    # Summary statistics
    # ------------------
    
    # Total variance: trace(G) = sum of eigenvalues
    # Measures the overall "size" of the genetic ellipsoid
    total_variance = np.sum(eigenvalues)
    
    # Generalized variance: det(G) = product of eigenvalues
    # Measures the "volume" of the ellipsoid (zero if singular)
    generalized_variance = np.prod(eigenvalues)
    
    # Effective dimensionality: (trace(G))² / trace(G²)
    # Equals p when all eigenvalues equal; approaches 1 when one dominates
    # This tells us how many "independent" directions of variation exist
    if total_variance > 0:
        effective_dimensionality = total_variance**2 / np.sum(eigenvalues**2)
    else:
        effective_dimensionality = 0.0
    
    # Eccentricity: relative variance of eigenvalues
    # V_rel = Var(λ) / mean(λ)²
    # Zero when spherical (no constraint); large when highly elongated
    mean_eigenvalue = np.mean(eigenvalues)
    if mean_eigenvalue > 0:
        var_eigenvalues = np.var(eigenvalues, ddof=0)  # Population variance
        eccentricity = var_eigenvalues / mean_eigenvalue**2
    else:
        eccentricity = 0.0
    
    # Condition number: λ_max / λ_min
    # Measures how "ill-conditioned" the matrix is
    # Large values indicate near-singularity and numerical instability
    if eigenvalues[-1] > 1e-10:
        condition_number = eigenvalues[0] / eigenvalues[-1]
    else:
        condition_number = np.inf
    
    return GMatrixAnalysis(
        G=G,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        g_max=g_max,
        g_min=g_min,
        total_variance=total_variance,
        generalized_variance=generalized_variance,
        effective_dimensionality=effective_dimensionality,
        eccentricity=eccentricity,
        condition_number=condition_number
    )


# =============================================================================
# SECTION 2: THE MULTIVARIATE BREEDER'S EQUATION
# =============================================================================

def selection_response(G: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Compute the response to selection using the multivariate breeder's equation.
    
    THE BREEDER'S EQUATION: Δz̄ = Gβ
    
    This is the fundamental equation of multivariate evolution. It states
    that the change in mean phenotype equals the G matrix acting on the
    selection gradient.
    
    Parameters
    ----------
    G : np.ndarray
        Additive genetic covariance matrix (p × p)
    beta : np.ndarray
        Selection gradient vector (p × 1)—direction of steepest fitness increase
        
    Returns
    -------
    np.ndarray
        Response vector Δz̄—predicted change in mean phenotype
        
    Geometric Interpretation
    ------------------------
    The selection gradient β tells you WHERE SELECTION WANTS TO GO.
    The G matrix tells you WHERE GENETIC VARIATION ALLOWS YOU TO GO.
    The response Δz̄ = Gβ is the COMPROMISE between these.
    
    When G is not proportional to the identity (i.e., genetic correlations
    exist or variances differ), the response is DEFLECTED toward g_max—
    the direction of maximum genetic variance.
    
    Example
    -------
    >>> G = np.array([[1.0, 0.8], [0.8, 1.0]])  # Strong positive correlation
    >>> beta = np.array([0, 1])  # Selection on trait 2 only
    >>> response = selection_response(G, beta)
    >>> print(response)  # Note: trait 1 also responds due to correlation!
    [0.8 1.0]
    """
    G = np.asarray(G, dtype=float)
    beta = np.asarray(beta, dtype=float)
    
    return G @ beta


def response_deflection(G: np.ndarray, beta: np.ndarray) -> Dict:
    """
    Analyze how the G matrix deflects the response away from the selection direction.
    
    This function quantifies the key phenomenon of multivariate evolution:
    genetic correlations cause the evolutionary response to deviate from
    the direction of selection.
    
    Parameters
    ----------
    G : np.ndarray
        Additive genetic covariance matrix
    beta : np.ndarray
        Selection gradient vector
        
    Returns
    -------
    dict containing:
        - 'beta': Original selection gradient
        - 'response': Response vector Δz̄ = Gβ
        - 'beta_direction': Unit vector in direction of selection
        - 'response_direction': Unit vector in direction of response
        - 'deflection_angle': Angle between selection and response (degrees)
        - 'response_magnitude': Length of response vector
        - 'alignment_with_gmax': Cosine of angle between response and g_max
        
    Biological Interpretation
    -------------------------
    When deflection_angle is small, the population evolves roughly where
    selection pushes. When deflection_angle is large, genetic constraints
    are redirecting evolution toward g_max.
    
    The alignment_with_gmax tells you how much the response has been
    "captured" by the line of least resistance.
    """
    beta = np.asarray(beta, dtype=float)
    
    # Compute response
    response = selection_response(G, beta)
    
    # Normalize to get directions
    beta_norm = np.linalg.norm(beta)
    response_norm = np.linalg.norm(response)
    
    if beta_norm < 1e-10 or response_norm < 1e-10:
        raise ValueError("Selection gradient or response is zero")
    
    beta_direction = beta / beta_norm
    response_direction = response / response_norm
    
    # Angle between selection and response
    cos_angle = np.clip(np.dot(beta_direction, response_direction), -1, 1)
    deflection_angle = np.degrees(np.arccos(cos_angle))
    
    # Alignment with g_max
    analysis = analyze_G_matrix(G)
    g_max = analysis.g_max
    alignment_with_gmax = abs(np.dot(response_direction, g_max))
    
    return {
        'beta': beta,
        'response': response,
        'beta_direction': beta_direction,
        'response_direction': response_direction,
        'deflection_angle': deflection_angle,
        'response_magnitude': response_norm,
        'alignment_with_gmax': alignment_with_gmax
    }


# =============================================================================
# SECTION 3: EVOLVABILITY AND RESPONDABILITY
# =============================================================================

def evolvability(beta: np.ndarray, G: np.ndarray) -> float:
    """
    Compute the evolvability in the direction of selection.
    
    EVOLVABILITY: e(β) = β'Gβ (for unit β)
    
    This measures the additive genetic variance in the direction that
    selection is pushing. It answers: "How much genetic raw material
    is available in the direction we want to go?"
    
    Parameters
    ----------
    beta : np.ndarray
        Direction of selection (will be normalized to unit length)
    G : np.ndarray
        Additive genetic covariance matrix
        
    Returns
    -------
    float
        Evolvability—genetic variance in direction β
        
    Bounds
    ------
    From the theory of quadratic forms:
        λ_min(G) ≤ e(β) ≤ λ_max(G)
        
    The evolvability equals λ_max when β aligns with g_max, and
    equals λ_min when β aligns with g_min.
    
    Reference
    ---------
    Hansen, T.F. & Houle, D. (2008). Measuring and comparing evolvability
    and constraint in multivariate characters. J. Evol. Biol. 21: 1201-1219.
    """
    beta = np.asarray(beta, dtype=float)
    G = np.asarray(G, dtype=float)
    
    # Normalize beta to unit length
    beta_norm = np.linalg.norm(beta)
    if beta_norm < 1e-10:
        raise ValueError("Selection direction cannot be zero")
    beta_unit = beta / beta_norm
    
    # Evolvability is the quadratic form β'Gβ
    return float(beta_unit @ G @ beta_unit)


def respondability(beta: np.ndarray, G: np.ndarray, P: np.ndarray) -> float:
    """
    Compute the respondability (directional heritability) in direction β.
    
    RESPONDABILITY: r(β) = β'Gβ / β'Pβ = h²(β)
    
    This is the fraction of phenotypic variance that is genetic in
    direction β. It generalizes univariate heritability to arbitrary
    directions in trait space.
    
    Parameters
    ----------
    beta : np.ndarray
        Direction in trait space (will be normalized)
    G : np.ndarray
        Additive genetic covariance matrix
    P : np.ndarray
        Phenotypic covariance matrix
        
    Returns
    -------
    float
        Directional heritability h²(β) ∈ [0, 1]
        
    Relationship to Evolvability
    ----------------------------
    Evolvability asks: "How much genetic variance is there?"
    Respondability asks: "What fraction of variance is genetic?"
    
    A direction can have high evolvability but low respondability
    (lots of genetic variance, but even more environmental variance),
    or vice versa.
    """
    beta = np.asarray(beta, dtype=float)
    G = np.asarray(G, dtype=float)
    P = np.asarray(P, dtype=float)
    
    # Normalize beta
    beta_norm = np.linalg.norm(beta)
    if beta_norm < 1e-10:
        raise ValueError("Direction cannot be zero")
    beta_unit = beta / beta_norm
    
    # Compute both quadratic forms
    genetic_variance = float(beta_unit @ G @ beta_unit)
    phenotypic_variance = float(beta_unit @ P @ beta_unit)
    
    if phenotypic_variance < 1e-10:
        raise ValueError("Phenotypic variance in direction β is zero")
    
    return genetic_variance / phenotypic_variance


def evolvability_surface(G: np.ndarray, n_directions: int = 100) -> Dict:
    """
    Compute evolvability across all directions in 2D trait space.
    
    This function samples directions uniformly around the unit circle
    and computes the evolvability in each direction, creating a "polar
    plot" of genetic potential.
    
    Parameters
    ----------
    G : np.ndarray
        2×2 genetic covariance matrix
    n_directions : int
        Number of directions to sample
        
    Returns
    -------
    dict containing:
        - 'angles': Array of angles in radians
        - 'evolvabilities': Array of e(β) values
        - 'max_evolvability': Maximum (equals λ_max)
        - 'min_evolvability': Minimum (equals λ_min)
        - 'angle_at_max': Angle (radians) where maximum occurs
        - 'angle_at_min': Angle (radians) where minimum occurs
        
    Note
    ----
    For 2D, this fully characterizes the evolvability landscape.
    In higher dimensions, one would sample from the unit sphere.
    """
    if G.shape != (2, 2):
        raise ValueError("This function is designed for 2D G matrices")
    
    angles = np.linspace(0, 2*np.pi, n_directions, endpoint=False)
    evolvabilities = np.zeros(n_directions)
    
    for i, theta in enumerate(angles):
        beta = np.array([np.cos(theta), np.sin(theta)])
        evolvabilities[i] = evolvability(beta, G)
    
    # Find extremes
    max_idx = np.argmax(evolvabilities)
    min_idx = np.argmin(evolvabilities)
    
    return {
        'angles': angles,
        'evolvabilities': evolvabilities,
        'max_evolvability': evolvabilities[max_idx],
        'min_evolvability': evolvabilities[min_idx],
        'angle_at_max': angles[max_idx],
        'angle_at_min': angles[min_idx]
    }


# =============================================================================
# SECTION 4: CONSTRAINT QUANTIFICATION
# =============================================================================

def constraint_metrics(G: np.ndarray) -> Dict:
    """
    Compute comprehensive metrics of evolutionary constraint from G.
    
    These metrics quantify different aspects of how G limits evolution:
    
    1. ECCENTRICITY: How elongated is the genetic ellipsoid?
    2. EFFECTIVE DIMENSIONALITY: How many independent directions exist?
    3. EIGENVALUE RATIOS: How much does the largest direction dominate?
    4. CONSTRAINT SEVERITY: How bad is the worst-constrained direction?
    
    Parameters
    ----------
    G : np.ndarray
        Additive genetic covariance matrix
        
    Returns
    -------
    dict with comprehensive constraint metrics
    
    Interpretation Guide
    --------------------
    - High eccentricity → Strong channeling toward g_max
    - Low effective dimensionality → Variation concentrated in few directions
    - High first eigenvalue proportion → g_max dominates
    - High constraint severity → Some directions have very low variance
    """
    analysis = analyze_G_matrix(G)
    eigenvalues = analysis.eigenvalues
    p = len(eigenvalues)
    
    # Proportion of variance explained by each PC
    total = np.sum(eigenvalues)
    proportions = eigenvalues / total if total > 0 else np.zeros(p)
    
    # Cumulative proportion
    cumulative = np.cumsum(proportions)
    
    # Number of dimensions to explain 95% of variance
    if total > 0:
        n_for_95 = int(np.searchsorted(cumulative, 0.95) + 1)
    else:
        n_for_95 = 0
    
    # Constraint severity: 1 - (λ_min / λ_mean)
    mean_eigenvalue = np.mean(eigenvalues)
    if mean_eigenvalue > 0:
        constraint_severity = 1 - eigenvalues[-1] / mean_eigenvalue
    else:
        constraint_severity = 0.0
    
    # Dimensionality reduction ratio
    dimensionality_ratio = analysis.effective_dimensionality / p
    
    return {
        'eigenvalues': eigenvalues,
        'proportions': proportions,
        'cumulative_proportions': cumulative,
        'effective_dimensionality': analysis.effective_dimensionality,
        'max_possible_dimensionality': p,
        'dimensionality_ratio': dimensionality_ratio,
        'eccentricity': analysis.eccentricity,
        'condition_number': analysis.condition_number,
        'constraint_severity': constraint_severity,
        'n_dims_for_95pct': n_for_95,
        'first_pc_proportion': proportions[0] if len(proportions) > 0 else 0,
        'g_max': analysis.g_max,
        'g_min': analysis.g_min
    }


def compare_G_matrices(G1: np.ndarray, G2: np.ndarray, 
                       n_random_vectors: int = 1000) -> Dict:
    """
    Compare two G matrices using multiple approaches.
    
    This function implements several methods for comparing genetic
    covariance structures:
    
    1. RANDOM SKEWERS: Correlation of responses to random selection vectors
    2. EIGENVALUE COMPARISON: Do they have similar variance structure?
    3. EIGENVECTOR ALIGNMENT: Do principal axes point in similar directions?
    
    Parameters
    ----------
    G1, G2 : np.ndarray
        Two genetic covariance matrices to compare
    n_random_vectors : int
        Number of random selection gradients for random skewers test
        
    Returns
    -------
    dict with comparison metrics
    
    Reference
    ---------
    Cheverud, J.M. (1988). A comparison of genetic and phenotypic correlations.
    Evolution 42: 958-968.
    
    Krzanowski, W.J. (1979). Between-groups comparison of principal components.
    JASA 74: 703-707.
    """
    p = G1.shape[0]
    if G2.shape[0] != p:
        raise ValueError("G matrices must have the same dimensions")
    
    # Analyze both matrices
    analysis1 = analyze_G_matrix(G1)
    analysis2 = analyze_G_matrix(G2)
    
    # -------------------------------------------------------------------------
    # Random Skewers Test
    # -------------------------------------------------------------------------
    # Generate random unit vectors and compute responses
    responses_corr = []
    
    for _ in range(n_random_vectors):
        # Random unit vector
        beta = np.random.randn(p)
        beta = beta / np.linalg.norm(beta)
        
        # Responses
        r1 = G1 @ beta
        r2 = G2 @ beta
        
        # Normalize responses
        norm1 = np.linalg.norm(r1)
        norm2 = np.linalg.norm(r2)
        
        if norm1 > 1e-10 and norm2 > 1e-10:
            # Correlation between response directions
            responses_corr.append(np.dot(r1/norm1, r2/norm2))
    
    random_skewers_corr = np.mean(responses_corr)
    
    # -------------------------------------------------------------------------
    # Eigenvalue Comparison
    # -------------------------------------------------------------------------
    # Correlation between eigenvalue spectra
    eigenvalue_corr = np.corrcoef(analysis1.eigenvalues, 
                                   analysis2.eigenvalues)[0, 1]
    
    # -------------------------------------------------------------------------
    # Eigenvector Alignment (Krzanowski's subspace comparison)
    # -------------------------------------------------------------------------
    # Compare first k eigenvectors (we'll use k = min(p, 3))
    k = min(p, 3)
    
    # Subspace spanned by first k eigenvectors
    V1 = analysis1.eigenvectors[:, :k]
    V2 = analysis2.eigenvectors[:, :k]
    
    # Krzanowski's statistic: sum of squared projections
    # Ranges from 0 (orthogonal subspaces) to k (identical subspaces)
    S = V1.T @ V2 @ V2.T @ V1
    krzanowski_stat = np.trace(S)
    
    # Normalize to [0, 1]
    krzanowski_normalized = krzanowski_stat / k
    
    # -------------------------------------------------------------------------
    # g_max alignment
    # -------------------------------------------------------------------------
    gmax_alignment = abs(np.dot(analysis1.g_max, analysis2.g_max))
    
    return {
        'random_skewers_correlation': random_skewers_corr,
        'eigenvalue_correlation': eigenvalue_corr,
        'krzanowski_statistic': krzanowski_stat,
        'krzanowski_normalized': krzanowski_normalized,
        'gmax_alignment': gmax_alignment,
        'G1_effective_dim': analysis1.effective_dimensionality,
        'G2_effective_dim': analysis2.effective_dimensionality
    }


# =============================================================================
# SECTION 5: VISUALIZATION
# =============================================================================

def plot_genetic_ellipse(G: np.ndarray, ax: Optional[plt.Axes] = None,
                         center: Tuple[float, float] = (0, 0),
                         n_std: float = 2.0,
                         color: str = 'steelblue',
                         alpha: float = 0.3,
                         show_axes: bool = True,
                         label: str = 'G ellipse') -> plt.Axes:
    """
    Plot the genetic covariance ellipse for a 2D G matrix.
    
    The ellipse shows the shape of genetic variation: its principal axes
    are the eigenvectors of G, and its semi-axis lengths are proportional
    to the square roots of the eigenvalues.
    
    Parameters
    ----------
    G : np.ndarray
        2×2 genetic covariance matrix
    ax : plt.Axes, optional
        Axes to plot on (creates new figure if None)
    center : tuple
        Center of the ellipse (typically the mean phenotype)
    n_std : float
        Number of standard deviations for ellipse radius
    color : str
        Color for the ellipse
    alpha : float
        Transparency
    show_axes : bool
        If True, draw lines along the principal axes
    label : str
        Label for the legend
        
    Returns
    -------
    plt.Axes
        The axes with the ellipse drawn
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    analysis = analyze_G_matrix(G)
    eigenvalues = analysis.eigenvalues
    eigenvectors = analysis.eigenvectors
    
    # Ellipse parameters
    # Width and height are 2 * n_std * sqrt(eigenvalue)
    width = 2 * n_std * np.sqrt(eigenvalues[0])
    height = 2 * n_std * np.sqrt(eigenvalues[1])
    
    # Rotation angle (from first eigenvector)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    
    # Create and add ellipse
    ellipse = Ellipse(center, width, height, angle=angle,
                      facecolor=color, alpha=alpha,
                      edgecolor=color, linewidth=2, label=label)
    ax.add_patch(ellipse)
    
    # Draw principal axes
    if show_axes:
        for i in range(2):
            vec = eigenvectors[:, i]
            length = n_std * np.sqrt(eigenvalues[i])
            ax.arrow(center[0], center[1],
                    length * vec[0], length * vec[1],
                    head_width=0.1, head_length=0.05,
                    fc='black' if i == 0 else 'gray',
                    ec='black' if i == 0 else 'gray',
                    linewidth=2 if i == 0 else 1)
            
            # Label
            label_pos = (center[0] + length * vec[0] * 1.15,
                        center[1] + length * vec[1] * 1.15)
            ax.annotate(f'g{"_max" if i == 0 else "_min"}\n(λ={eigenvalues[i]:.2f})',
                       label_pos, fontsize=10, ha='center')
    
    ax.set_aspect('equal')
    ax.axhline(y=center[1], color='lightgray', linestyle='--', alpha=0.5)
    ax.axvline(x=center[0], color='lightgray', linestyle='--', alpha=0.5)
    
    return ax


def plot_selection_response(G: np.ndarray, beta: np.ndarray,
                           ax: Optional[plt.Axes] = None,
                           scale: float = 1.0) -> plt.Axes:
    """
    Visualize how G transforms selection into response.
    
    This plot shows:
    - The G ellipse (genetic variation)
    - The selection gradient β (where selection pushes)
    - The response Δz̄ = Gβ (where evolution goes)
    - The deflection angle (how much constraint redirects evolution)
    
    Parameters
    ----------
    G : np.ndarray
        2×2 genetic covariance matrix
    beta : np.ndarray
        Selection gradient
    ax : plt.Axes, optional
        Axes to plot on
    scale : float
        Scale factor for arrows
        
    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot G ellipse
    plot_genetic_ellipse(G, ax=ax, show_axes=True, alpha=0.2)
    
    # Compute response and deflection
    deflection_info = response_deflection(G, beta)
    response = deflection_info['response']
    deflection_angle = deflection_info['deflection_angle']
    
    # Scale arrows for visualization
    beta_scaled = scale * beta
    response_scaled = scale * response
    
    # Plot selection gradient
    ax.arrow(0, 0, beta_scaled[0], beta_scaled[1],
            head_width=0.15, head_length=0.08,
            fc='red', ec='red', linewidth=2,
            label=f'Selection β')
    
    # Plot response
    ax.arrow(0, 0, response_scaled[0], response_scaled[1],
            head_width=0.15, head_length=0.08,
            fc='blue', ec='blue', linewidth=2,
            label=f'Response Δz̄ = Gβ')
    
    # Add deflection angle arc
    angle_beta = np.arctan2(beta[1], beta[0])
    angle_response = np.arctan2(response[1], response[0])
    
    # Annotation
    ax.annotate(f'Deflection: {deflection_angle:.1f}°',
               xy=(0.05, 0.95), xycoords='axes fraction',
               fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Legend
    ax.legend(loc='lower right', fontsize=10)
    
    ax.set_xlabel('Trait 1', fontsize=12)
    ax.set_ylabel('Trait 2', fontsize=12)
    ax.set_title('G Matrix: Transforming Selection into Response', fontsize=14)
    
    # Set axis limits to show all elements
    all_x = [0, beta_scaled[0], response_scaled[0]]
    all_y = [0, beta_scaled[1], response_scaled[1]]
    margin = 0.5
    ax.set_xlim(min(all_x) - margin - 2, max(all_x) + margin + 2)
    ax.set_ylim(min(all_y) - margin - 2, max(all_y) + margin + 2)
    
    return ax


def plot_evolvability_polar(G: np.ndarray, ax: Optional[plt.Axes] = None,
                            n_directions: int = 100) -> plt.Axes:
    """
    Create a polar plot of evolvability across all directions.
    
    This visualization shows how genetic variance changes as you rotate
    around the trait space. The distance from the origin in each direction
    equals the evolvability (genetic variance) in that direction.
    
    Parameters
    ----------
    G : np.ndarray
        2×2 genetic covariance matrix
    ax : plt.Axes, optional
        Polar axes to plot on
    n_directions : int
        Number of directions to sample
        
    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # Compute evolvability surface
    surface = evolvability_surface(G, n_directions)
    angles = surface['angles']
    evolvabilities = surface['evolvabilities']
    
    # Close the loop by appending first point
    angles_plot = np.append(angles, angles[0])
    evolv_plot = np.append(evolvabilities, evolvabilities[0])
    
    # Plot
    ax.plot(angles_plot, evolv_plot, 'b-', linewidth=2)
    ax.fill(angles_plot, evolv_plot, alpha=0.3)
    
    # Mark maximum and minimum
    ax.plot(surface['angle_at_max'], surface['max_evolvability'], 
           'go', markersize=12, label=f"Max: {surface['max_evolvability']:.3f}")
    ax.plot(surface['angle_at_min'], surface['min_evolvability'],
           'ro', markersize=12, label=f"Min: {surface['min_evolvability']:.3f}")
    
    ax.set_title('Evolvability by Direction\n(distance = genetic variance)', fontsize=12)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    return ax


def plot_G_vs_P(G: np.ndarray, P: np.ndarray, 
                ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot G and P ellipses together to visualize the heritability structure.
    
    This comparison reveals:
    - Where G nearly fills P (high heritability)
    - Where G is thin relative to P (low heritability, constraint traps)
    
    Parameters
    ----------
    G : np.ndarray
        Genetic covariance matrix
    P : np.ndarray
        Phenotypic covariance matrix
    ax : plt.Axes, optional
        Axes to plot on
        
    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot P ellipse
    plot_genetic_ellipse(P, ax=ax, color='gray', alpha=0.15,
                        show_axes=False, label='P (phenotypic)', n_std=2)
    
    # Plot G ellipse
    plot_genetic_ellipse(G, ax=ax, color='steelblue', alpha=0.3,
                        show_axes=True, label='G (genetic)', n_std=2)
    
    # Compute and display heritabilities along axes
    P_analysis = analyze_G_matrix(P)
    
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xlabel('Trait 1', fontsize=12)
    ax.set_ylabel('Trait 2', fontsize=12)
    ax.set_title('G Matrix Nested Within P Matrix', fontsize=14)
    
    # Add heritability annotations
    for i in range(2):
        h2 = G[i, i] / P[i, i]
        ax.annotate(f'h²(trait {i+1}) = {h2:.2f}',
                   xy=(0.02, 0.98 - i*0.05), xycoords='axes fraction',
                   fontsize=10)
    
    return ax


# =============================================================================
# SECTION 6: WORKED EXAMPLES FROM THE BOOK
# =============================================================================

def example_two_trait_G_analysis():
    """
    Reproduce Example 1 from Chapter 12: Two-trait G matrix analysis.
    
    This example analyzes a G matrix for flowering time and plant height
    from a plant breeding program.
    """
    print("=" * 70)
    print("EXAMPLE: Two-Trait G Matrix Analysis")
    print("=" * 70)
    
    # Define matrices from the book
    G = np.array([
        [25, 15],
        [15, 36]
    ])
    
    P = np.array([
        [50, 20],
        [20, 60]
    ])
    
    trait_names = ['Flowering time (days)', 'Plant height (cm)']
    
    print("\nAdditive Genetic Covariance Matrix G:")
    print(G)
    print("\nPhenotypic Covariance Matrix P:")
    print(P)
    
    # Step 1: Eigendecomposition of G
    print("\n" + "-" * 50)
    print("STEP 1: Eigendecomposition of G")
    print("-" * 50)
    
    analysis = analyze_G_matrix(G)
    
    print(f"\nEigenvalues: λ₁ = {analysis.eigenvalues[0]:.2f}, λ₂ = {analysis.eigenvalues[1]:.2f}")
    print(f"\ng_max = ({analysis.g_max[0]:.3f}, {analysis.g_max[1]:.3f})")
    print(f"  → Points toward: {trait_names[1]} with positive {trait_names[0]}")
    print(f"\ng_min = ({analysis.g_min[0]:.3f}, {analysis.g_min[1]:.3f})")
    print(f"  → Contrasts the traits: tall-but-early vs short-but-late")
    
    # Step 2: Univariate heritabilities
    print("\n" + "-" * 50)
    print("STEP 2: Univariate Heritabilities")
    print("-" * 50)
    
    for i in range(2):
        h2 = G[i, i] / P[i, i]
        print(f"h²({trait_names[i]}) = {G[i,i]}/{P[i,i]} = {h2:.2f}")
    
    # Step 3: Directional heritabilities
    print("\n" + "-" * 50)
    print("STEP 3: Directional Heritabilities")
    print("-" * 50)
    
    directions = {
        'g_max': analysis.g_max,
        'g_min': analysis.g_min,
        'Flowering only': np.array([1, 0]),
        'Height only': np.array([0, 1])
    }
    
    for name, direction in directions.items():
        e = evolvability(direction, G)
        r = respondability(direction, G, P)
        print(f"\nDirection: {name}")
        print(f"  Direction vector: ({direction[0]:.3f}, {direction[1]:.3f})")
        print(f"  Evolvability e(β) = {e:.2f}")
        print(f"  Respondability h²(β) = {r:.3f}")
    
    # Step 4: Response to selection
    print("\n" + "-" * 50)
    print("STEP 4: Response to Selection on Height Only")
    print("-" * 50)
    
    beta = np.array([0, 1])  # Select on height only
    deflection = response_deflection(G, beta)
    
    print(f"\nSelection gradient β = {beta}")
    print(f"Response Δz̄ = Gβ = {deflection['response']}")
    print(f"\nDeflection angle: {deflection['deflection_angle']:.1f}°")
    print(f"  → Selection targeted only height, but flowering time also responds")
    print(f"  → The genetic correlation 'drags' flowering time along")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: G and P ellipses
    plot_G_vs_P(G, P, ax=axes[0])
    axes[0].set_title('Genetic (G) and Phenotypic (P) Covariance', fontsize=12)
    
    # Right: Selection and response
    plot_selection_response(G, beta, ax=axes[1], scale=3)
    axes[1].set_title('Selection on Height Only → Correlated Response', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('ch09_two_trait_example.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Figure saved: ch09_two_trait_example.png")
    
    return analysis


def example_constraint_analysis():
    """
    Demonstrate constraint quantification using multiple G matrices.
    
    This example compares three different G matrices to illustrate
    how eigenvalue structure relates to evolutionary constraint.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE: Constraint Analysis")
    print("=" * 70)
    
    # Three G matrices with different constraint structures
    matrices = {
        'Spherical (no constraint)': np.array([[1.0, 0.0], [0.0, 1.0]]),
        'Moderate constraint': np.array([[2.0, 1.2], [1.2, 1.0]]),
        'Strong constraint': np.array([[5.0, 4.5], [4.5, 5.0]])
    }
    
    print("\nConstraint Metrics Comparison:")
    print("-" * 70)
    print(f"{'Matrix':<25} {'EffDim':>8} {'Eccentr':>10} {'λ₁/λ₂':>10} {'Severity':>10}")
    print("-" * 70)
    
    for name, G in matrices.items():
        metrics = constraint_metrics(G)
        print(f"{name:<25} {metrics['effective_dimensionality']:>8.2f} "
              f"{metrics['eccentricity']:>10.3f} "
              f"{metrics['condition_number']:>10.1f} "
              f"{metrics['constraint_severity']:>10.3f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, (name, G) in zip(axes, matrices.items()):
        plot_genetic_ellipse(G, ax=ax, n_std=1.5)
        metrics = constraint_metrics(G)
        ax.set_title(f'{name}\nEff. Dim = {metrics["effective_dimensionality"]:.2f}', 
                    fontsize=11)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_xlabel('Trait 1')
        ax.set_ylabel('Trait 2')
    
    plt.tight_layout()
    plt.savefig('ch09_constraint_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Figure saved: ch09_constraint_comparison.png")


def example_g_matrix_comparison():
    """
    Compare two G matrices using multiple methods.
    
    This simulates comparing G matrices from two populations to ask:
    "Do these populations have similar genetic constraint structures?"
    """
    print("\n" + "=" * 70)
    print("EXAMPLE: Comparing G Matrices")
    print("=" * 70)
    
    # Two G matrices: similar structure but different scale
    G1 = np.array([
        [1.0, 0.7],
        [0.7, 1.0]
    ])
    
    G2 = np.array([
        [2.0, 1.4],  # Same correlations, doubled variances
        [1.4, 2.0]
    ])
    
    # A third matrix with different structure
    G3 = np.array([
        [1.0, -0.5],  # Negative correlation
        [-0.5, 1.0]
    ])
    
    print("\nG1 (Population A):")
    print(G1)
    print("\nG2 (Population B - same structure, different scale):")
    print(G2)
    print("\nG3 (Population C - different structure):")
    print(G3)
    
    # Compare G1 vs G2 (should be similar)
    print("\n" + "-" * 50)
    print("Comparison: G1 vs G2 (expected: SIMILAR)")
    print("-" * 50)
    comparison_12 = compare_G_matrices(G1, G2)
    for key, value in comparison_12.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
    
    # Compare G1 vs G3 (should be different)
    print("\n" + "-" * 50)
    print("Comparison: G1 vs G3 (expected: DIFFERENT)")
    print("-" * 50)
    comparison_13 = compare_G_matrices(G1, G3)
    for key, value in comparison_13.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, (G, name) in zip(axes, [(G1, 'G1: Pop A'), (G2, 'G2: Pop B'), (G3, 'G3: Pop C')]):
        plot_genetic_ellipse(G, ax=ax, n_std=1.5)
        ax.set_title(name, fontsize=12)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_xlabel('Trait 1')
        ax.set_ylabel('Trait 2')
    
    plt.tight_layout()
    plt.savefig('ch09_g_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Figure saved: ch09_g_comparison.png")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   CHAPTER 9: THE G MATRIX AND THE GENETIC ELLIPSOID                          ║
║                                                                              ║
║   "The G matrix is not merely a table of numbers. It is a SHAPE—an          ║
║    ellipsoid in trait space that determines how populations can and          ║
║    cannot evolve."                                                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Run all examples
    example_two_trait_G_analysis()
    example_constraint_analysis()
    example_g_matrix_comparison()
    
    # Final demonstration: Evolvability polar plot
    print("\n" + "=" * 70)
    print("BONUS: Evolvability Polar Plot")
    print("=" * 70)
    
    G = np.array([[2.0, 1.5], [1.5, 2.0]])
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    plot_evolvability_polar(G, ax=ax)
    plt.savefig('ch09_evolvability_polar.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Figure saved: ch09_evolvability_polar.png")
    
    print("\n" + "=" * 70)
    print("Chapter 9 code execution complete!")
    print("=" * 70)
