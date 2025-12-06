#!/usr/bin/env python3
"""
==============================================================================
CHAPTER 10: THE FITNESS SURFACE AND γ (GAMMA)
==============================================================================

From "Seeing the Shape: A Geometric Introduction to Multivariate 
Quantitative Genetics" by Daniel Ortiz-Barrientos

This script implements tools for understanding fitness surfaces and the 
quadratic selection gradient γ (gamma)—the matrix that captures the 
curvature of selection.

THE CORE INSIGHT:
    The fitness surface is a landscape over trait space. Its SLOPE (β) 
    determines directional selection. Its CURVATURE (γ) determines whether
    selection is stabilizing, disruptive, or correlational.
    
    The interaction between γ (the geometry of selection) and G (the geometry
    of genetic variation) determines whether evolution is fast or slow,
    aligned or deflected.

WHAT YOU WILL LEARN:
    1. How to interpret fitness as a surface over trait space
    2. How to compute and interpret the selection gradient β
    3. How to compute and interpret the curvature matrix γ
    4. The meaning of stabilizing vs. disruptive selection geometrically
    5. How correlational selection shapes trait combinations
    6. How to estimate γ from fitness data via quadratic regression
    7. Canonical analysis of the fitness surface
    8. How to analyze G-γ alignment

PREREQUISITES:
    - Chapter 7 (Eigendecomposition)
    - Chapter 9 (The G Matrix)

Author: Code companion to Ortiz-Barrientos (2025)
License: CC BY-NC-SA 4.0
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from scipy.optimize import minimize
from typing import Tuple, Dict, Optional, Callable, List
from dataclasses import dataclass
import warnings

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
np.set_printoptions(precision=4, suppress=True)


# =============================================================================
# SECTION 1: DATA STRUCTURES
# =============================================================================

@dataclass
class FitnessSurfaceAnalysis:
    """
    Complete analysis of a fitness surface.
    
    Packages the selection gradient β, curvature matrix γ, and their
    eigenstructure for biological interpretation.
    
    Attributes
    ----------
    beta : np.ndarray
        Selection gradient—direction of steepest fitness increase
    gamma : np.ndarray
        Quadratic selection gradient—curvature matrix
    gamma_eigenvalues : np.ndarray
        Eigenvalues of γ (curvatures along principal axes)
    gamma_eigenvectors : np.ndarray
        Eigenvectors of γ (principal axes of the fitness surface)
    selection_type : List[str]
        Classification of selection along each principal axis
    overall_selection : str
        Summary of selection regime
    """
    beta: np.ndarray
    gamma: np.ndarray
    gamma_eigenvalues: np.ndarray
    gamma_eigenvectors: np.ndarray
    selection_type: List[str]
    overall_selection: str


@dataclass
class GammaGAnalysis:
    """
    Analysis of alignment between fitness surface (γ) and genetic variation (G).
    
    This captures the key question: does the geometry of selection align with
    the geometry of genetic constraint?
    
    Attributes
    ----------
    gamma : np.ndarray
        Curvature matrix
    G : np.ndarray
        Genetic covariance matrix
    alignment_angle : float
        Angle between leading axes of γ and G (degrees)
    evolvability_at_peak_curvature : float
        Genetic variance in the direction of strongest curvature
    heritability_at_peak_curvature : float
        Directional heritability where selection is strongest
    evolutionary_prediction : str
        Qualitative prediction about evolutionary dynamics
    """
    gamma: np.ndarray
    G: np.ndarray
    alignment_angle: float
    evolvability_at_peak_curvature: float
    heritability_at_peak_curvature: Optional[float]
    evolutionary_prediction: str


# =============================================================================
# SECTION 2: SELECTION GRADIENTS (β)
# =============================================================================

def compute_selection_gradient(z: np.ndarray, w: np.ndarray,
                                standardize: bool = True) -> np.ndarray:
    """
    Compute the directional selection gradient β from phenotype and fitness data.
    
    THE SELECTION GRADIENT: β = P⁻¹s = P⁻¹Cov(z, w̃)
    
    where w̃ = w/w̄ is relative fitness.
    
    The selection gradient is the vector of partial regression coefficients
    of relative fitness on standardized traits. It points in the direction
    of steepest fitness ascent.
    
    Parameters
    ----------
    z : np.ndarray
        Phenotype matrix (n × p) where each row is an individual
    w : np.ndarray
        Absolute fitness values (n,)
    standardize : bool
        If True, standardize traits to mean 0, variance 1 before analysis
        
    Returns
    -------
    np.ndarray
        Selection gradient vector β (p,)
        
    Mathematical Background
    -----------------------
    The selection gradient is defined as:
    
        β = ∂(ln w̄) / ∂z̄
        
    For small selection, this equals P⁻¹s where s is the selection differential
    (covariance between traits and relative fitness).
    
    Reference
    ---------
    Lande, R. & Arnold, S.J. (1983). The measurement of selection on 
    correlated characters. Evolution 37: 1210-1226.
    
    Example
    -------
    >>> z = np.random.randn(100, 2)  # 100 individuals, 2 traits
    >>> w = np.exp(0.5 * z[:, 0])    # Fitness depends on trait 1
    >>> beta = compute_selection_gradient(z, w)
    >>> print(f"Selection on trait 1: {beta[0]:.3f}")
    """
    z = np.asarray(z, dtype=float)
    w = np.asarray(w, dtype=float)
    
    n, p = z.shape
    
    # Standardize traits if requested
    if standardize:
        z_mean = np.mean(z, axis=0)
        z_std = np.std(z, axis=0, ddof=1)
        z_std[z_std == 0] = 1  # Avoid division by zero
        z = (z - z_mean) / z_std
    
    # Compute relative fitness
    w_mean = np.mean(w)
    if w_mean <= 0:
        raise ValueError("Mean fitness must be positive")
    w_rel = w / w_mean
    
    # Method: Multiple regression of relative fitness on traits
    # This gives the selection gradient directly
    # We add a column of ones for the intercept
    X = np.column_stack([np.ones(n), z])
    
    # Solve normal equations: (X'X)⁻¹X'w
    beta_with_intercept = np.linalg.lstsq(X, w_rel, rcond=None)[0]
    
    # Extract β (excluding intercept)
    beta = beta_with_intercept[1:]
    
    return beta


def interpret_selection_gradient(beta: np.ndarray, 
                                  trait_names: Optional[List[str]] = None) -> Dict:
    """
    Provide biological interpretation of a selection gradient.
    
    Parameters
    ----------
    beta : np.ndarray
        Selection gradient vector
    trait_names : list of str, optional
        Names of traits
        
    Returns
    -------
    dict with interpretation including:
        - magnitude: Overall strength of directional selection
        - direction: Unit vector in direction of selection
        - strongest_trait: Which trait is under strongest selection
        - interpretation: Text description
    """
    p = len(beta)
    if trait_names is None:
        trait_names = [f"Trait {i+1}" for i in range(p)]
    
    # Magnitude of selection
    magnitude = np.linalg.norm(beta)
    
    # Direction (normalized)
    if magnitude > 1e-10:
        direction = beta / magnitude
    else:
        direction = np.zeros(p)
    
    # Which trait is under strongest selection?
    abs_beta = np.abs(beta)
    strongest_idx = np.argmax(abs_beta)
    strongest_trait = trait_names[strongest_idx]
    strongest_direction = "+" if beta[strongest_idx] > 0 else "-"
    
    # Build interpretation
    interpretation = []
    for i in range(p):
        if abs(beta[i]) < 0.05:
            strength = "weak"
        elif abs(beta[i]) < 0.15:
            strength = "moderate"
        else:
            strength = "strong"
        
        direction_word = "positive" if beta[i] > 0 else "negative"
        interpretation.append(f"{trait_names[i]}: {strength} {direction_word} selection (β = {beta[i]:.3f})")
    
    return {
        'beta': beta,
        'magnitude': magnitude,
        'direction': direction,
        'strongest_trait': strongest_trait,
        'strongest_direction': strongest_direction,
        'trait_interpretations': interpretation,
        'summary': f"Overall directional selection strength: {magnitude:.3f}"
    }


# =============================================================================
# SECTION 3: THE CURVATURE MATRIX γ (GAMMA)
# =============================================================================

def compute_gamma(z: np.ndarray, w: np.ndarray,
                   standardize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the quadratic selection gradient γ (curvature matrix) from data.
    
    THE CURVATURE MATRIX: γᵢⱼ = ∂²(ln w̄) / ∂zᵢ∂zⱼ
    
    γ describes the curvature of the fitness surface:
    - Negative eigenvalues → stabilizing selection (fitness peak)
    - Positive eigenvalues → disruptive selection (fitness valley)
    - Off-diagonal elements → correlational selection
    
    Parameters
    ----------
    z : np.ndarray
        Phenotype matrix (n × p)
    w : np.ndarray
        Absolute fitness values (n,)
    standardize : bool
        If True, standardize traits before analysis
        
    Returns
    -------
    beta : np.ndarray
        Selection gradient (p,)
    gamma : np.ndarray
        Quadratic selection gradient matrix (p × p)
        
    Mathematical Background
    -----------------------
    We fit the quadratic model:
    
        w̃ = α + Σᵢ βᵢzᵢ + ½ΣᵢΣⱼ γᵢⱼzᵢzⱼ + ε
        
    The factor of ½ ensures γᵢᵢ equals the second derivative ∂²w̃/∂zᵢ².
    
    Note: We report γ directly, not the doubled version sometimes seen.
    
    Reference
    ---------
    Lande, R. & Arnold, S.J. (1983). Evolution 37: 1210-1226.
    Phillips, P.C. & Arnold, S.J. (1989). Evolution 43: 1209-1222.
    
    Example
    -------
    >>> # Simulate stabilizing selection on 2 traits
    >>> z = np.random.randn(200, 2)
    >>> w = np.exp(-0.5 * (z[:, 0]**2 + z[:, 1]**2))  # Gaussian peak at origin
    >>> beta, gamma = compute_gamma(z, w)
    >>> print(f"γ₁₁ = {gamma[0,0]:.3f}, γ₂₂ = {gamma[1,1]:.3f}")  # Should be ~ -1
    """
    z = np.asarray(z, dtype=float)
    w = np.asarray(w, dtype=float)
    
    n, p = z.shape
    
    # Standardize traits
    if standardize:
        z_mean = np.mean(z, axis=0)
        z_std = np.std(z, axis=0, ddof=1)
        z_std[z_std == 0] = 1
        z = (z - z_mean) / z_std
    
    # Compute relative fitness
    w_mean = np.mean(w)
    if w_mean <= 0:
        raise ValueError("Mean fitness must be positive")
    w_rel = w / w_mean
    
    # Build design matrix for quadratic regression
    # Columns: 1, z₁, z₂, ..., z₁², z₂², ..., z₁z₂, z₁z₃, ...
    
    # Linear terms
    X_linear = z  # (n, p)
    
    # Quadratic terms (diagonal of γ)
    X_quad_diag = z ** 2  # (n, p)
    
    # Cross-product terms (off-diagonal of γ)
    X_cross = []
    cross_indices = []
    for i in range(p):
        for j in range(i + 1, p):
            X_cross.append(z[:, i] * z[:, j])
            cross_indices.append((i, j))
    
    if X_cross:
        X_cross = np.column_stack(X_cross)
    else:
        X_cross = np.zeros((n, 0))
    
    # Full design matrix
    X = np.column_stack([np.ones(n), X_linear, X_quad_diag, X_cross])
    
    # Solve via least squares
    coeffs = np.linalg.lstsq(X, w_rel, rcond=None)[0]
    
    # Extract coefficients
    idx = 1  # Start after intercept
    
    # β (linear terms)
    beta = coeffs[idx:idx + p]
    idx += p
    
    # γ diagonal (need to multiply by 2 because model has ½γᵢᵢzᵢ²)
    gamma_diag = 2 * coeffs[idx:idx + p]
    idx += p
    
    # γ off-diagonal
    gamma_offdiag = coeffs[idx:]
    
    # Assemble γ matrix
    gamma = np.zeros((p, p))
    np.fill_diagonal(gamma, gamma_diag)
    
    for k, (i, j) in enumerate(cross_indices):
        gamma[i, j] = gamma_offdiag[k]
        gamma[j, i] = gamma_offdiag[k]  # Symmetric
    
    return beta, gamma


def analyze_gamma(gamma: np.ndarray, 
                   trait_names: Optional[List[str]] = None) -> FitnessSurfaceAnalysis:
    """
    Perform eigenanalysis of γ to understand the geometry of selection.
    
    The eigenvalues of γ reveal:
    - Negative eigenvalue → stabilizing selection (fitness curves down)
    - Positive eigenvalue → disruptive selection (fitness curves up)
    - Zero eigenvalue → no curvature (flat in that direction)
    
    Parameters
    ----------
    gamma : np.ndarray
        Quadratic selection gradient matrix
    trait_names : list of str, optional
        Names of traits
        
    Returns
    -------
    FitnessSurfaceAnalysis
        Complete analysis including eigenstructure and interpretation
    """
    p = gamma.shape[0]
    if trait_names is None:
        trait_names = [f"Trait {i+1}" for i in range(p)]
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(gamma)
    
    # Sort by absolute value (strongest curvature first)
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Classify selection type for each axis
    selection_type = []
    for lam in eigenvalues:
        if lam < -0.05:
            selection_type.append("stabilizing")
        elif lam > 0.05:
            selection_type.append("disruptive")
        else:
            selection_type.append("neutral/weak")
    
    # Overall classification
    n_stab = sum(1 for s in selection_type if s == "stabilizing")
    n_disr = sum(1 for s in selection_type if s == "disruptive")
    
    if n_stab == p:
        overall = "pure stabilizing (fitness peak)"
    elif n_disr == p:
        overall = "pure disruptive (fitness valley)"
    elif n_stab > 0 and n_disr > 0:
        overall = "saddle point (stabilizing + disruptive)"
    else:
        overall = "weak/neutral curvature"
    
    # Note: beta is not computed here—would need to be passed in
    # For now, we leave it as zeros
    beta = np.zeros(p)
    
    return FitnessSurfaceAnalysis(
        beta=beta,
        gamma=gamma,
        gamma_eigenvalues=eigenvalues,
        gamma_eigenvectors=eigenvectors,
        selection_type=selection_type,
        overall_selection=overall
    )


def interpret_gamma(gamma: np.ndarray, 
                     trait_names: Optional[List[str]] = None) -> Dict:
    """
    Provide detailed interpretation of the γ matrix.
    
    Parameters
    ----------
    gamma : np.ndarray
        Quadratic selection gradient matrix
    trait_names : list of str, optional
        Names of traits
        
    Returns
    -------
    dict with detailed interpretation
    """
    p = gamma.shape[0]
    if trait_names is None:
        trait_names = [f"Trait {i+1}" for i in range(p)]
    
    analysis = analyze_gamma(gamma, trait_names)
    
    interpretation = {
        'gamma': gamma,
        'eigenvalues': analysis.gamma_eigenvalues,
        'eigenvectors': analysis.gamma_eigenvectors,
        'overall_selection': analysis.overall_selection,
        'axis_interpretations': [],
        'correlational_selection': []
    }
    
    # Interpret each principal axis
    for i, (lam, sel_type) in enumerate(zip(analysis.gamma_eigenvalues, 
                                             analysis.selection_type)):
        vec = analysis.gamma_eigenvectors[:, i]
        
        # Describe the direction
        dominant_traits = []
        for j, v in enumerate(vec):
            if abs(v) > 0.3:
                sign = "+" if v > 0 else "-"
                dominant_traits.append(f"{sign}{trait_names[j]}")
        
        direction_desc = " & ".join(dominant_traits) if dominant_traits else "mixed"
        
        interpretation['axis_interpretations'].append({
            'axis': i + 1,
            'eigenvalue': lam,
            'selection_type': sel_type,
            'direction': direction_desc,
            'eigenvector': vec
        })
    
    # Interpret correlational selection (off-diagonals)
    for i in range(p):
        for j in range(i + 1, p):
            gamma_ij = gamma[i, j]
            if abs(gamma_ij) > 0.02:
                if gamma_ij > 0:
                    interp = f"Positive correlation favored between {trait_names[i]} and {trait_names[j]}"
                else:
                    interp = f"Negative correlation favored between {trait_names[i]} and {trait_names[j]}"
                
                interpretation['correlational_selection'].append({
                    'traits': (trait_names[i], trait_names[j]),
                    'gamma_ij': gamma_ij,
                    'interpretation': interp
                })
    
    return interpretation


# =============================================================================
# SECTION 4: GAUSSIAN FITNESS SURFACES
# =============================================================================

def gaussian_fitness_surface(z: np.ndarray, theta: np.ndarray, 
                              omega: np.ndarray, w_max: float = 1.0) -> np.ndarray:
    """
    Compute fitness for a Gaussian (quadratic) fitness surface.
    
    THE GAUSSIAN FITNESS FUNCTION:
        w(z) = w_max × exp(-½(z - θ)'ω⁻¹(z - θ))
    
    This models stabilizing selection with optimum at θ and width ω.
    
    Parameters
    ----------
    z : np.ndarray
        Phenotype(s)—either (p,) for one individual or (n, p) for many
    theta : np.ndarray
        Optimum phenotype (p,)
    omega : np.ndarray
        Width matrix (p × p)—larger ω means weaker selection
    w_max : float
        Maximum fitness (at the optimum)
        
    Returns
    -------
    np.ndarray
        Fitness value(s)
        
    Mathematical Background
    -----------------------
    For this fitness function:
    - Selection gradient: β = ω⁻¹(θ - z̄)
    - Curvature matrix: γ = -ω⁻¹
    
    The curvature is constant everywhere and equals -ω⁻¹.
    Narrow peaks (small ω) have strong curvature (large |γ|).
    
    Example
    -------
    >>> theta = np.array([0, 0])  # Optimum at origin
    >>> omega = np.array([[1, 0], [0, 1]])  # Width = 1 in both directions
    >>> z = np.array([0.5, 0.5])  # Suboptimal phenotype
    >>> w = gaussian_fitness_surface(z, theta, omega)
    >>> print(f"Fitness: {w:.4f}")  # Less than 1
    """
    z = np.asarray(z, dtype=float)
    theta = np.asarray(theta, dtype=float)
    omega = np.asarray(omega, dtype=float)
    
    # Handle single individual or multiple
    single = (z.ndim == 1)
    if single:
        z = z.reshape(1, -1)
    
    # Deviation from optimum
    delta = z - theta
    
    # Mahalanobis distance squared using ω as metric
    omega_inv = np.linalg.inv(omega)
    
    # For each individual: δ'ω⁻¹δ
    mahal_sq = np.sum((delta @ omega_inv) * delta, axis=1)
    
    # Gaussian fitness
    fitness = w_max * np.exp(-0.5 * mahal_sq)
    
    if single:
        return fitness[0]
    return fitness


def derive_selection_from_gaussian(theta: np.ndarray, omega: np.ndarray,
                                    z_bar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analytically derive β and γ for a Gaussian fitness surface.
    
    For a Gaussian surface centered at θ with width ω:
    - β = ω⁻¹(θ - z̄)  [points toward the optimum]
    - γ = -ω⁻¹         [constant negative curvature]
    
    Parameters
    ----------
    theta : np.ndarray
        Optimum phenotype
    omega : np.ndarray
        Width matrix
    z_bar : np.ndarray
        Current mean phenotype
        
    Returns
    -------
    beta : np.ndarray
        Selection gradient
    gamma : np.ndarray
        Curvature matrix
    """
    omega_inv = np.linalg.inv(omega)
    
    beta = omega_inv @ (theta - z_bar)
    gamma = -omega_inv
    
    return beta, gamma


# =============================================================================
# SECTION 5: G-γ ALIGNMENT ANALYSIS
# =============================================================================

def analyze_G_gamma_alignment(G: np.ndarray, gamma: np.ndarray,
                               P: Optional[np.ndarray] = None) -> GammaGAnalysis:
    """
    Analyze how the geometry of selection (γ) aligns with genetic variation (G).
    
    This is a central question in evolutionary biology: does selection push
    in a direction where genetic variation is abundant or scarce?
    
    Parameters
    ----------
    G : np.ndarray
        Genetic covariance matrix
    gamma : np.ndarray
        Curvature matrix (quadratic selection gradient)
    P : np.ndarray, optional
        Phenotypic covariance matrix (for heritability calculations)
        
    Returns
    -------
    GammaGAnalysis
        Complete analysis of alignment
        
    Biological Interpretation
    -------------------------
    ALIGNED (low angle): The direction of strongest selection curvature
    coincides with a direction of high/low genetic variance. This can mean:
    - If g_max aligns with stabilizing: variation maintained along ridge
    - If g_max aligns with disruptive: rapid divergence possible
    
    MISALIGNED (high angle): Selection and constraint operate in different
    planes of trait space. Evolutionary dynamics are complex.
    """
    # Eigenanalyze both matrices
    G_eigenvalues, G_eigenvectors = np.linalg.eigh(G)
    gamma_eigenvalues, gamma_eigenvectors = np.linalg.eigh(gamma)
    
    # Sort G by largest eigenvalue first
    G_idx = np.argsort(G_eigenvalues)[::-1]
    G_eigenvalues = G_eigenvalues[G_idx]
    G_eigenvectors = G_eigenvectors[:, G_idx]
    
    # Sort gamma by largest ABSOLUTE eigenvalue first (strongest curvature)
    gamma_idx = np.argsort(np.abs(gamma_eigenvalues))[::-1]
    gamma_eigenvalues = gamma_eigenvalues[gamma_idx]
    gamma_eigenvectors = gamma_eigenvectors[:, gamma_idx]
    
    # g_max and γ_strongest (direction of strongest curvature)
    g_max = G_eigenvectors[:, 0]
    gamma_strongest = gamma_eigenvectors[:, 0]
    
    # Alignment angle between g_max and strongest curvature direction
    cos_angle = abs(np.dot(g_max, gamma_strongest))
    cos_angle = np.clip(cos_angle, 0, 1)
    alignment_angle = np.degrees(np.arccos(cos_angle))
    
    # Evolvability in the direction of strongest curvature
    evolvability = float(gamma_strongest @ G @ gamma_strongest)
    
    # Heritability in that direction (if P provided)
    if P is not None:
        pheno_var = float(gamma_strongest @ P @ gamma_strongest)
        heritability = evolvability / pheno_var if pheno_var > 0 else None
    else:
        heritability = None
    
    # Generate evolutionary prediction
    strongest_selection_type = "stabilizing" if gamma_eigenvalues[0] < 0 else "disruptive"
    
    if alignment_angle < 30:
        if strongest_selection_type == "stabilizing":
            prediction = ("ALIGNED STABILIZING: Strong stabilizing selection aligns with "
                         "direction of high genetic variance. Expect rapid erosion of "
                         "variance along g_max.")
        else:
            prediction = ("ALIGNED DISRUPTIVE: Disruptive selection aligns with g_max. "
                         "Expect rapid divergence if population can split.")
    elif alignment_angle > 60:
        if strongest_selection_type == "stabilizing":
            prediction = ("ORTHOGONAL STABILIZING: Stabilizing selection acts perpendicular "
                         "to g_max. Variance maintained along g_max; constrained traits eroded.")
        else:
            prediction = ("ORTHOGONAL DISRUPTIVE: Disruptive selection acts perpendicular "
                         "to g_max. Complex dynamics; divergence limited by low variance.")
    else:
        prediction = ("INTERMEDIATE ALIGNMENT: Selection and genetic variation partially "
                     "aligned. Expect mixed dynamics with both constraint and response.")
    
    return GammaGAnalysis(
        gamma=gamma,
        G=G,
        alignment_angle=alignment_angle,
        evolvability_at_peak_curvature=evolvability,
        heritability_at_peak_curvature=heritability,
        evolutionary_prediction=prediction
    )


# =============================================================================
# SECTION 6: CANONICAL ANALYSIS
# =============================================================================

def canonical_analysis(beta: np.ndarray, gamma: np.ndarray) -> Dict:
    """
    Perform canonical analysis of the fitness surface.
    
    Following Phillips & Arnold (1989), we transform to the coordinate
    system defined by the eigenvectors of γ. In this system, the fitness
    surface has the form:
    
        w̃ = w̄ + Σᵢ θᵢmᵢ + ½Σᵢ λᵢmᵢ²
        
    where mᵢ is the projection onto the ith eigenvector, θᵢ is the
    directional selection along that axis, and λᵢ is the curvature.
    
    Parameters
    ----------
    beta : np.ndarray
        Selection gradient
    gamma : np.ndarray
        Curvature matrix
        
    Returns
    -------
    dict containing:
        - eigenvalues: Curvatures along canonical axes
        - eigenvectors: Canonical axes (columns)
        - theta: Directional selection along each canonical axis
        - canonical_interpretation: Text description
        
    Reference
    ---------
    Phillips, P.C. & Arnold, S.J. (1989). Visualizing multivariate selection.
    Evolution 43: 1209-1222.
    """
    # Eigendecomposition of gamma
    eigenvalues, eigenvectors = np.linalg.eigh(gamma)
    
    # Sort by eigenvalue (most negative first = strongest stabilizing)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Project beta onto canonical axes
    # θᵢ = vᵢ'β (directional selection in canonical direction i)
    theta = eigenvectors.T @ beta
    
    # Build interpretation
    interpretations = []
    for i in range(len(eigenvalues)):
        lam = eigenvalues[i]
        th = theta[i]
        
        # Curvature type
        if lam < -0.05:
            curv_type = "stabilizing"
        elif lam > 0.05:
            curv_type = "disruptive"
        else:
            curv_type = "flat"
        
        # Directional component
        if abs(th) < 0.05:
            dir_type = "no directional"
        elif th > 0:
            dir_type = "positive directional"
        else:
            dir_type = "negative directional"
        
        interpretations.append(
            f"Canonical axis {i+1}: λ={lam:.3f} ({curv_type}), θ={th:.3f} ({dir_type})"
        )
    
    return {
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'theta': theta,
        'canonical_interpretations': interpretations
    }


# =============================================================================
# SECTION 7: VISUALIZATION
# =============================================================================

def plot_fitness_surface_2d(fitness_func: Callable, 
                            xlim: Tuple[float, float] = (-3, 3),
                            ylim: Tuple[float, float] = (-3, 3),
                            n_points: int = 100,
                            ax: Optional[plt.Axes] = None,
                            cmap: str = 'RdYlGn') -> plt.Axes:
    """
    Plot a 2D fitness surface as a contour plot.
    
    Parameters
    ----------
    fitness_func : callable
        Function that takes (x, y) and returns fitness
    xlim, ylim : tuples
        Axis limits
    n_points : int
        Grid resolution
    ax : plt.Axes, optional
        Axes to plot on
    cmap : str
        Colormap name
        
    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    x = np.linspace(xlim[0], xlim[1], n_points)
    y = np.linspace(ylim[0], ylim[1], n_points)
    X, Y = np.meshgrid(x, y)
    
    # Compute fitness at each point
    Z = np.zeros_like(X)
    for i in range(n_points):
        for j in range(n_points):
            Z[i, j] = fitness_func(X[i, j], Y[i, j])
    
    # Contour plot
    levels = np.linspace(Z.min(), Z.max(), 20)
    cs = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, alpha=0.8)
    ax.contour(X, Y, Z, levels=levels, colors='k', alpha=0.3, linewidths=0.5)
    
    plt.colorbar(cs, ax=ax, label='Fitness')
    
    ax.set_xlabel('Trait 1', fontsize=12)
    ax.set_ylabel('Trait 2', fontsize=12)
    ax.set_aspect('equal')
    
    return ax


def plot_fitness_surface_3d(fitness_func: Callable,
                            xlim: Tuple[float, float] = (-3, 3),
                            ylim: Tuple[float, float] = (-3, 3),
                            n_points: int = 50,
                            ax: Optional[Axes3D] = None) -> Axes3D:
    """
    Plot a 2D fitness surface as a 3D surface plot.
    
    Parameters
    ----------
    fitness_func : callable
        Function that takes (x, y) and returns fitness
    xlim, ylim : tuples
        Axis limits
    n_points : int
        Grid resolution
    ax : Axes3D, optional
        3D axes to plot on
        
    Returns
    -------
    Axes3D
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    x = np.linspace(xlim[0], xlim[1], n_points)
    y = np.linspace(ylim[0], ylim[1], n_points)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X)
    for i in range(n_points):
        for j in range(n_points):
            Z[i, j] = fitness_func(X[i, j], Y[i, j])
    
    surf = ax.plot_surface(X, Y, Z, cmap='RdYlGn', alpha=0.8,
                           linewidth=0, antialiased=True)
    
    ax.set_xlabel('Trait 1')
    ax.set_ylabel('Trait 2')
    ax.set_zlabel('Fitness')
    
    return ax


def plot_gamma_ellipse(gamma: np.ndarray, ax: Optional[plt.Axes] = None,
                       center: Tuple[float, float] = (0, 0),
                       color: str = 'red', alpha: float = 0.3,
                       n_std: float = 1.0) -> plt.Axes:
    """
    Visualize the γ matrix as an ellipse showing curvature structure.
    
    Unlike covariance ellipses (which show extent of variation), the γ ellipse
    shows the shape of the fitness surface curvature. The axes point along
    the principal curvature directions.
    
    Parameters
    ----------
    gamma : np.ndarray
        2×2 curvature matrix
    ax : plt.Axes, optional
        Axes to plot on
    center : tuple
        Center of ellipse
    color : str
        Color
    alpha : float
        Transparency
    n_std : float
        Scale factor
        
    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Use absolute values of eigenvalues for visualization
    eigenvalues, eigenvectors = np.linalg.eigh(gamma)
    
    # Scale by absolute eigenvalues (curvature strength)
    abs_eigenvalues = np.abs(eigenvalues)
    
    # Create ellipse
    from matplotlib.patches import Ellipse
    
    width = 2 * n_std * np.sqrt(abs_eigenvalues[1]) if abs_eigenvalues[1] > 0 else 0.1
    height = 2 * n_std * np.sqrt(abs_eigenvalues[0]) if abs_eigenvalues[0] > 0 else 0.1
    
    angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
    
    ellipse = Ellipse(center, width, height, angle=angle,
                      facecolor=color, alpha=alpha,
                      edgecolor=color, linewidth=2,
                      linestyle='--' if any(eigenvalues > 0) else '-')
    ax.add_patch(ellipse)
    
    # Draw principal axes with curvature labels
    for i in range(2):
        vec = eigenvectors[:, i]
        length = n_std * np.sqrt(abs_eigenvalues[i])
        
        # Color by sign: red for negative (stabilizing), blue for positive (disruptive)
        arrow_color = 'darkred' if eigenvalues[i] < 0 else 'darkblue'
        
        ax.arrow(center[0], center[1],
                length * vec[0], length * vec[1],
                head_width=0.1, head_length=0.05,
                fc=arrow_color, ec=arrow_color, linewidth=2)
        
        sel_type = "stab" if eigenvalues[i] < 0 else "disr"
        ax.annotate(f'γ={eigenvalues[i]:.2f}\n({sel_type})',
                   (center[0] + length * vec[0] * 1.2,
                    center[1] + length * vec[1] * 1.2),
                   fontsize=9, ha='center')
    
    ax.set_aspect('equal')
    ax.axhline(y=center[1], color='lightgray', linestyle='--', alpha=0.5)
    ax.axvline(x=center[0], color='lightgray', linestyle='--', alpha=0.5)
    
    return ax


def plot_G_and_gamma(G: np.ndarray, gamma: np.ndarray,
                      ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot G and γ ellipses together to visualize alignment.
    
    This shows how the geometry of selection (γ) relates to the
    geometry of genetic variation (G).
    
    Parameters
    ----------
    G : np.ndarray
        Genetic covariance matrix
    gamma : np.ndarray
        Curvature matrix
    ax : plt.Axes, optional
        Axes to plot on
        
    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    # Import here to avoid circular dependency with ch09
    from matplotlib.patches import Ellipse
    
    # G ellipse
    G_eigenvalues, G_eigenvectors = np.linalg.eigh(G)
    G_idx = np.argsort(G_eigenvalues)[::-1]
    G_eigenvalues = G_eigenvalues[G_idx]
    G_eigenvectors = G_eigenvectors[:, G_idx]
    
    G_width = 2 * 2 * np.sqrt(G_eigenvalues[0])
    G_height = 2 * 2 * np.sqrt(G_eigenvalues[1])
    G_angle = np.degrees(np.arctan2(G_eigenvectors[1, 0], G_eigenvectors[0, 0]))
    
    G_ellipse = Ellipse((0, 0), G_width, G_height, angle=G_angle,
                        facecolor='steelblue', alpha=0.3,
                        edgecolor='steelblue', linewidth=2,
                        label='G (genetic variance)')
    ax.add_patch(G_ellipse)
    
    # g_max arrow
    g_max = G_eigenvectors[:, 0]
    g_max_len = 2 * np.sqrt(G_eigenvalues[0])
    ax.arrow(0, 0, g_max_len * g_max[0], g_max_len * g_max[1],
            head_width=0.15, head_length=0.08,
            fc='steelblue', ec='steelblue', linewidth=2)
    
    # γ representation: arrows showing curvature directions
    gamma_eigenvalues, gamma_eigenvectors = np.linalg.eigh(gamma)
    
    for i in range(2):
        vec = gamma_eigenvectors[:, i]
        lam = gamma_eigenvalues[i]
        length = 1.5 * np.sqrt(np.abs(lam))
        
        color = 'darkred' if lam < 0 else 'darkgreen'
        style = '-' if lam < 0 else '--'
        
        # Draw bidirectional
        ax.annotate('', xy=(length * vec[0], length * vec[1]),
                   xytext=(-length * vec[0], -length * vec[1]),
                   arrowprops=dict(arrowstyle='<->', color=color, lw=2, ls=style))
    
    # Compute alignment
    alignment = analyze_G_gamma_alignment(G, gamma)
    
    ax.set_aspect('equal')
    ax.axhline(y=0, color='lightgray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='lightgray', linestyle='--', alpha=0.5)
    
    # Adjust limits
    max_extent = max(G_width, G_height) / 2 + 0.5
    ax.set_xlim(-max_extent, max_extent)
    ax.set_ylim(-max_extent, max_extent)
    
    ax.set_xlabel('Trait 1', fontsize=12)
    ax.set_ylabel('Trait 2', fontsize=12)
    ax.set_title(f'G-γ Alignment\n(angle = {alignment.alignment_angle:.1f}°)', 
                fontsize=14)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Ellipse((0, 0), 0.1, 0.1, facecolor='steelblue', alpha=0.3, 
                edgecolor='steelblue', label='G ellipse'),
        Line2D([0], [0], color='darkred', linewidth=2, label='γ: stabilizing'),
        Line2D([0], [0], color='darkgreen', linewidth=2, linestyle='--', 
               label='γ: disruptive')
    ]
    ax.legend(handles=[Line2D([0], [0], color='steelblue', linewidth=2, label='g_max'),
                       Line2D([0], [0], color='darkred', linewidth=2, label='γ (stabilizing)'),
                       Line2D([0], [0], color='darkgreen', linewidth=2, ls='--', label='γ (disruptive)')],
             loc='upper right')
    
    return ax


# =============================================================================
# SECTION 8: WORKED EXAMPLES
# =============================================================================

def example_gaussian_fitness():
    """
    Demonstrate a Gaussian fitness surface and its selection properties.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE: Gaussian Fitness Surface")
    print("=" * 70)
    
    # Define the fitness surface
    theta = np.array([0.5, 0.3])  # Optimum slightly off-center
    omega = np.array([
        [1.0, 0.3],
        [0.3, 0.5]
    ])  # Elliptical surface—wider along trait 1
    
    print("\nOptimum phenotype θ:", theta)
    print("\nWidth matrix ω:")
    print(omega)
    
    # Current population mean
    z_bar = np.array([0, 0])
    
    # Derive selection analytically
    beta, gamma = derive_selection_from_gaussian(theta, omega, z_bar)
    
    print("\n" + "-" * 50)
    print("Selection gradients (analytical)")
    print("-" * 50)
    print(f"β = {beta}")
    print("  → Selection pushes toward the optimum")
    
    print(f"\nγ = \n{gamma}")
    print("  → Curvature matrix (= -ω⁻¹)")
    
    # Analyze gamma
    analysis = analyze_gamma(gamma)
    print(f"\nγ eigenvalues: {analysis.gamma_eigenvalues}")
    print(f"Selection types: {analysis.selection_type}")
    print(f"Overall: {analysis.overall_selection}")
    
    # Visualize
    fig = plt.figure(figsize=(16, 6))
    
    # 2D contour
    ax1 = fig.add_subplot(131)
    
    def fitness_2d(x, y):
        z = np.array([x, y])
        return gaussian_fitness_surface(z, theta, omega)
    
    plot_fitness_surface_2d(fitness_2d, ax=ax1, xlim=(-2, 3), ylim=(-2, 3))
    ax1.plot(theta[0], theta[1], 'k*', markersize=15, label='Optimum θ')
    ax1.plot(z_bar[0], z_bar[1], 'ro', markersize=10, label='Pop. mean')
    ax1.arrow(z_bar[0], z_bar[1], beta[0]*0.5, beta[1]*0.5,
             head_width=0.1, head_length=0.05, fc='red', ec='red')
    ax1.legend()
    ax1.set_title('Fitness Surface (contours)', fontsize=12)
    
    # 3D surface
    ax2 = fig.add_subplot(132, projection='3d')
    plot_fitness_surface_3d(fitness_2d, ax=ax2, xlim=(-2, 3), ylim=(-2, 3))
    ax2.set_title('Fitness Surface (3D)', fontsize=12)
    
    # γ ellipse
    ax3 = fig.add_subplot(133)
    plot_gamma_ellipse(gamma, ax=ax3, n_std=1.0)
    ax3.set_title('Curvature Structure (γ)', fontsize=12)
    ax3.set_xlim(-2, 2)
    ax3.set_ylim(-2, 2)
    
    plt.tight_layout()
    plt.savefig('ch10_gaussian_fitness.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Figure saved: ch10_gaussian_fitness.png")


def example_selection_analysis():
    """
    Reproduce Example 3 from Chapter 12: Selection analysis with β and γ.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE: Selection Analysis with G and γ")
    print("=" * 70)
    
    # From the book
    beta = np.array([0.18, 0.12])
    gamma = np.array([
        [-0.15, 0.08],
        [0.08, -0.10]
    ])
    G = np.array([
        [0.45, 0.30],
        [0.30, 0.35]
    ])
    
    print("\nSelection gradient β:", beta)
    print("\nCurvature matrix γ:")
    print(gamma)
    print("\nGenetic covariance G:")
    print(G)
    
    # Step 1: Interpret β
    print("\n" + "-" * 50)
    print("STEP 1: Interpret Selection Gradient β")
    print("-" * 50)
    
    beta_interp = interpret_selection_gradient(beta, ['Trait 1', 'Trait 2'])
    print(f"Magnitude: {beta_interp['magnitude']:.3f}")
    print(f"Strongest selection on: {beta_interp['strongest_trait']}")
    for line in beta_interp['trait_interpretations']:
        print(f"  {line}")
    
    # Step 2: Interpret γ
    print("\n" + "-" * 50)
    print("STEP 2: Interpret Curvature Matrix γ")
    print("-" * 50)
    
    gamma_interp = interpret_gamma(gamma, ['Trait 1', 'Trait 2'])
    print(f"Overall: {gamma_interp['overall_selection']}")
    print(f"Eigenvalues: {gamma_interp['eigenvalues']}")
    
    for axis_info in gamma_interp['axis_interpretations']:
        print(f"\n  Axis {axis_info['axis']}: λ = {axis_info['eigenvalue']:.3f}")
        print(f"    Type: {axis_info['selection_type']}")
        print(f"    Direction: {axis_info['direction']}")
    
    if gamma_interp['correlational_selection']:
        print("\nCorrelational selection:")
        for cs in gamma_interp['correlational_selection']:
            print(f"  {cs['interpretation']}")
    
    # Step 3: Predict response
    print("\n" + "-" * 50)
    print("STEP 3: Predict Response to Selection")
    print("-" * 50)
    
    response = G @ beta
    print(f"Δz̄ = Gβ = {response}")
    
    # Step 4: G-γ alignment
    print("\n" + "-" * 50)
    print("STEP 4: G-γ Alignment Analysis")
    print("-" * 50)
    
    alignment = analyze_G_gamma_alignment(G, gamma)
    print(f"Alignment angle: {alignment.alignment_angle:.1f}°")
    print(f"Evolvability at peak curvature: {alignment.evolvability_at_peak_curvature:.3f}")
    print(f"\nPrediction: {alignment.evolutionary_prediction}")
    
    # Step 5: Canonical analysis
    print("\n" + "-" * 50)
    print("STEP 5: Canonical Analysis")
    print("-" * 50)
    
    canonical = canonical_analysis(beta, gamma)
    print("\nCanonical axes:")
    for interp in canonical['canonical_interpretations']:
        print(f"  {interp}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: G and γ together
    plot_G_and_gamma(G, gamma, ax=axes[0])
    
    # Right: Selection and response with breeder's equation
    from matplotlib.patches import Ellipse
    
    # Plot G ellipse
    G_eigenvalues, G_eigenvectors = np.linalg.eigh(G)
    G_idx = np.argsort(G_eigenvalues)[::-1]
    G_eigenvalues = G_eigenvalues[G_idx]
    G_eigenvectors = G_eigenvectors[:, G_idx]
    
    G_width = 2 * 2 * np.sqrt(G_eigenvalues[0])
    G_height = 2 * 2 * np.sqrt(G_eigenvalues[1])
    G_angle = np.degrees(np.arctan2(G_eigenvectors[1, 0], G_eigenvectors[0, 0]))
    
    ellipse = Ellipse((0, 0), G_width, G_height, angle=G_angle,
                      facecolor='steelblue', alpha=0.2,
                      edgecolor='steelblue', linewidth=2)
    axes[1].add_patch(ellipse)
    
    # Selection and response arrows
    scale = 3
    axes[1].arrow(0, 0, scale*beta[0], scale*beta[1],
                 head_width=0.05, head_length=0.03,
                 fc='red', ec='red', linewidth=2, label='β (selection)')
    axes[1].arrow(0, 0, scale*response[0], scale*response[1],
                 head_width=0.05, head_length=0.03,
                 fc='blue', ec='blue', linewidth=2, label='Δz̄ = Gβ (response)')
    
    axes[1].set_aspect('equal')
    axes[1].set_xlim(-1.5, 1.5)
    axes[1].set_ylim(-1.5, 1.5)
    axes[1].axhline(y=0, color='lightgray', linestyle='--')
    axes[1].axvline(x=0, color='lightgray', linestyle='--')
    axes[1].legend(loc='lower right')
    axes[1].set_xlabel('Trait 1')
    axes[1].set_ylabel('Trait 2')
    axes[1].set_title("Breeder's Equation: Δz̄ = Gβ", fontsize=12)
    
    plt.tight_layout()
    plt.savefig('ch10_selection_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Figure saved: ch10_selection_analysis.png")


def example_estimate_gamma_from_data():
    """
    Demonstrate estimation of β and γ from simulated fitness data.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE: Estimating γ from Data")
    print("=" * 70)
    
    np.random.seed(42)
    
    # True parameters
    theta_true = np.array([0, 0])
    omega_true = np.array([
        [1.0, 0.5],
        [0.5, 2.0]
    ])
    
    # Generate population
    n = 300
    z = np.random.multivariate_normal([0, 0], [[1, 0.3], [0.3, 1]], size=n)
    
    # Compute fitness with noise
    w_true = gaussian_fitness_surface(z, theta_true, omega_true)
    w_noisy = w_true + 0.1 * np.random.randn(n)
    w_noisy = np.maximum(w_noisy, 0.01)  # Ensure positive fitness
    
    print(f"\nSimulated {n} individuals")
    print(f"True optimum: {theta_true}")
    print(f"True ω:\n{omega_true}")
    
    # True gamma = -ω⁻¹
    gamma_true = -np.linalg.inv(omega_true)
    print(f"\nTrue γ = -ω⁻¹:\n{gamma_true}")
    
    # Estimate from data
    print("\n" + "-" * 50)
    print("Estimation via quadratic regression")
    print("-" * 50)
    
    beta_est, gamma_est = compute_gamma(z, w_noisy, standardize=True)
    
    print(f"\nEstimated β: {beta_est}")
    print(f"  (should be near zero since pop. mean ≈ optimum)")
    
    print(f"\nEstimated γ:\n{gamma_est}")
    print(f"\nTrue γ:\n{gamma_true}")
    
    # Compare eigenvalues
    true_eigenvalues = np.linalg.eigvalsh(gamma_true)
    est_eigenvalues = np.linalg.eigvalsh(gamma_est)
    
    print(f"\nEigenvalue comparison:")
    print(f"  True: {np.sort(true_eigenvalues)}")
    print(f"  Est:  {np.sort(est_eigenvalues)}")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Left: Data with fitness coloring
    scatter = axes[0].scatter(z[:, 0], z[:, 1], c=w_noisy, cmap='RdYlGn',
                              alpha=0.7, s=30)
    plt.colorbar(scatter, ax=axes[0], label='Fitness')
    axes[0].set_xlabel('Trait 1')
    axes[0].set_ylabel('Trait 2')
    axes[0].set_title('Simulated Data', fontsize=12)
    
    # Middle: True fitness surface
    def true_fitness(x, y):
        return gaussian_fitness_surface(np.array([x, y]), theta_true, omega_true)
    
    plot_fitness_surface_2d(true_fitness, ax=axes[1], xlim=(-3, 3), ylim=(-3, 3))
    axes[1].set_title('True Fitness Surface', fontsize=12)
    
    # Right: Estimated γ vs true γ
    plot_gamma_ellipse(gamma_true, ax=axes[2], color='blue', alpha=0.2, n_std=1.0)
    plot_gamma_ellipse(gamma_est, ax=axes[2], color='red', alpha=0.2, n_std=1.0)
    axes[2].set_xlim(-2, 2)
    axes[2].set_ylim(-2, 2)
    axes[2].set_title('True γ (blue) vs Estimated γ (red)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('ch10_gamma_estimation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Figure saved: ch10_gamma_estimation.png")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   CHAPTER 10: THE FITNESS SURFACE AND γ (GAMMA)                              ║
║                                                                              ║
║   "The fitness surface is a landscape over trait space. Its SLOPE (β)       ║
║    determines directional selection. Its CURVATURE (γ) determines whether   ║
║    selection is stabilizing, disruptive, or correlational."                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Run all examples
    example_gaussian_fitness()
    example_selection_analysis()
    example_estimate_gamma_from_data()
    
    print("\n" + "=" * 70)
    print("Chapter 10 code execution complete!")
    print("=" * 70)
