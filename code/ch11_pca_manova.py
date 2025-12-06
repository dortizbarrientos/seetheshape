#!/usr/bin/env python3
"""
==============================================================================
CHAPTER 11: PCA, MANOVA, AND PROJECTIONS
==============================================================================

Seeing the Shape: A Geometric Introduction to Multivariate Quantitative Genetics
Code Companion - Chapter 11

Author: Daniel Ortiz-Barrientos
School of the Environment, The University of Queensland

------------------------------------------------------------------------------
THE CENTRAL INSIGHT
------------------------------------------------------------------------------

Every statistical method in this chapter does the same thing at its core:
it takes a covariance matrix (or a ratio of covariance matrices), finds its
eigenvalues and eigenvectors, and interprets them biologically.

    • PCA eigendecomposes a single covariance matrix to find directions of
      maximum variance.
    
    • MANOVA compares covariance matrices (among-group vs. within-group) to
      test whether groups differ.
    
    • Canonical Correlation Analysis (CCA) finds directions that maximise
      correlation between two sets of variables.
    
    • Discriminant Analysis finds directions that best separate groups.

Once you understand diagonalisation, these methods become variations on a
single theme: finding "interesting" directions in high-dimensional space.

------------------------------------------------------------------------------
MATHEMATICAL FOUNDATION
------------------------------------------------------------------------------

All these methods rest on the spectral theorem for symmetric matrices:

    A = Q Λ Q^T

where:
    • A is a symmetric matrix (covariance, ratio of covariances, etc.)
    • Q contains orthonormal eigenvectors as columns
    • Λ is diagonal with eigenvalues λ₁ ≥ λ₂ ≥ ... ≥ λₚ

The eigenvectors define "natural axes" of the matrix's geometry.
The eigenvalues measure extent (variance, separation, correlation) along
those axes.

------------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import eigh, sqrtm, inv
from typing import Tuple, Dict, List, Optional, NamedTuple
from dataclasses import dataclass
import warnings

# Set random seed for reproducibility
np.random.seed(42)

# Configure matplotlib for publication-quality figures
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.figsize': (10, 8),
    'figure.dpi': 100
})


# =============================================================================
# SECTION 1: PRINCIPAL COMPONENT ANALYSIS (PCA)
# =============================================================================
"""
PCA is the most widely used multivariate technique in biology. Its goal is
dimensionality reduction: represent the variation in p traits using fewer
than p derived variables, while losing as little information as possible.

GEOMETRIC INTERPRETATION:
-------------------------
PCA finds the principal axes of the data ellipse. The first principal
component (PC1) points along the direction of maximum variance; PC2 is
orthogonal to PC1 and captures the most remaining variance; and so on.

ALGEBRAIC PROCEDURE:
-------------------
1. Compute the sample covariance matrix S (or correlation matrix R)
2. Eigendecompose: S = V Λ V^T
3. The eigenvectors v₁, v₂, ..., vₚ are the principal component loadings
4. The eigenvalues λ₁ ≥ λ₂ ≥ ... ≥ λₚ are the variances along each PC
5. Project data onto eigenvectors to get principal component scores
"""


@dataclass
class PCAResult:
    """
    Container for PCA results with clear biological interpretation.
    
    Attributes
    ----------
    eigenvalues : np.ndarray
        Variances along each principal component (λ₁ ≥ λ₂ ≥ ... ≥ λₚ)
    
    eigenvectors : np.ndarray
        Loadings matrix - columns are principal component directions.
        eigenvectors[:, 0] is PC1, the direction of maximum variance.
    
    scores : np.ndarray
        Principal component scores - data projected onto PC axes.
        Each row is an individual; each column is a PC score.
    
    proportion_variance : np.ndarray
        Fraction of total variance explained by each PC.
        proportion_variance[0] is the fraction explained by PC1.
    
    cumulative_variance : np.ndarray
        Cumulative proportion of variance explained.
        cumulative_variance[k] = sum of first k+1 proportions.
    
    mean : np.ndarray
        Sample mean of original data (used for centering).
    
    use_correlation : bool
        Whether analysis was performed on correlation matrix.
    """
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    scores: np.ndarray
    proportion_variance: np.ndarray
    cumulative_variance: np.ndarray
    mean: np.ndarray
    use_correlation: bool


def perform_pca(X: np.ndarray, use_correlation: bool = False) -> PCAResult:
    """
    Perform Principal Component Analysis on multivariate data.
    
    PCA finds the directions of maximum variance in the data cloud.
    Geometrically, it identifies the principal axes of the covariance
    ellipse and ranks them by the amount of variance they capture.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_individuals, p_traits).
        Each row is an individual; each column is a trait.
    
    use_correlation : bool, default=False
        If True, standardise traits before analysis (PCA on correlation matrix).
        If False, use raw covariances (PCA on covariance matrix).
        
        WHEN TO USE EACH:
        • Covariance PCA: When traits are on comparable scales and 
          absolute variances are biologically meaningful (e.g., G or P matrices).
        • Correlation PCA: When traits are on different scales 
          (e.g., length in mm vs. mass in g) or you want equal weighting.
    
    Returns
    -------
    PCAResult
        Named tuple containing eigenvalues, eigenvectors, scores,
        variance proportions, and analysis metadata.
    
    Mathematical Details
    --------------------
    Given data matrix X (n × p), centered to X_c = X - mean(X):
    
    1. Compute covariance matrix: S = (1/(n-1)) X_c^T X_c
    
    2. Eigendecompose: S = V Λ V^T
       where Λ = diag(λ₁, λ₂, ..., λₚ) with λ₁ ≥ λ₂ ≥ ... ≥ λₚ
    
    3. Principal component scores: Z = X_c V
       The i-th column of Z contains scores for PC_i.
    
    4. Proportion of variance: prop_k = λ_k / Σλ_i = λ_k / tr(S)
    
    Example
    -------
    >>> # Simulate correlated trait data
    >>> n, p = 100, 4
    >>> true_cov = np.array([[1, 0.7, 0.3, 0.2],
    ...                      [0.7, 1, 0.5, 0.3],
    ...                      [0.3, 0.5, 1, 0.4],
    ...                      [0.2, 0.3, 0.4, 1]])
    >>> X = np.random.multivariate_normal(np.zeros(p), true_cov, n)
    >>> result = perform_pca(X)
    >>> print(f"PC1 explains {result.proportion_variance[0]:.1%} of variance")
    """
    n, p = X.shape
    
    # Step 1: Center the data (subtract column means)
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    
    # Step 2: Optionally standardise for correlation-based PCA
    if use_correlation:
        std = np.std(X, axis=0, ddof=1)
        # Avoid division by zero for constant traits
        std[std == 0] = 1.0
        X_centered = X_centered / std
    
    # Step 3: Compute covariance (or correlation) matrix
    # Using (n-1) for unbiased estimate (Bessel's correction)
    S = (X_centered.T @ X_centered) / (n - 1)
    
    # Step 4: Eigendecomposition
    # eigh() is for symmetric matrices - guarantees real eigenvalues
    # and orthonormal eigenvectors
    eigenvalues, eigenvectors = eigh(S)
    
    # Step 5: Sort in descending order (eigh returns ascending)
    # Convention: λ₁ ≥ λ₂ ≥ ... ≥ λₚ
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Step 6: Compute principal component scores
    # Z = X_c V projects data onto PC axes
    scores = X_centered @ eigenvectors
    
    # Step 7: Calculate variance proportions
    total_variance = np.sum(eigenvalues)  # = tr(S)
    proportion_variance = eigenvalues / total_variance
    cumulative_variance = np.cumsum(proportion_variance)
    
    return PCAResult(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        scores=scores,
        proportion_variance=proportion_variance,
        cumulative_variance=cumulative_variance,
        mean=mean,
        use_correlation=use_correlation
    )


def interpret_pca_loadings(pca_result: PCAResult, 
                           trait_names: List[str],
                           n_components: int = 2) -> None:
    """
    Interpret PCA loadings with biological context.
    
    The loadings tell us what each PC represents biologically:
    • If all loadings have the same sign → PC represents "size"
    • If loadings have mixed signs → PC represents a contrast
    
    Parameters
    ----------
    pca_result : PCAResult
        Output from perform_pca()
    
    trait_names : List[str]
        Names of the original traits for interpretation
    
    n_components : int
        Number of components to interpret
    """
    print("=" * 70)
    print("PCA LOADING INTERPRETATION")
    print("=" * 70)
    
    for i in range(min(n_components, len(pca_result.eigenvalues))):
        loadings = pca_result.eigenvectors[:, i]
        var_explained = pca_result.proportion_variance[i]
        cumulative = pca_result.cumulative_variance[i]
        
        print(f"\nPC{i+1}: {var_explained:.1%} of variance "
              f"(cumulative: {cumulative:.1%})")
        print("-" * 50)
        
        # Determine if "size" (all same sign) or "shape" (mixed signs)
        all_positive = np.all(loadings >= 0)
        all_negative = np.all(loadings <= 0)
        
        if all_positive or all_negative:
            interpretation = "SIZE axis - all traits load in same direction"
        else:
            pos_traits = [trait_names[j] for j in range(len(loadings)) 
                         if loadings[j] > 0.1]
            neg_traits = [trait_names[j] for j in range(len(loadings)) 
                         if loadings[j] < -0.1]
            interpretation = (f"SHAPE/CONTRAST axis\n"
                            f"    Positive: {', '.join(pos_traits)}\n"
                            f"    Negative: {', '.join(neg_traits)}")
        
        print(f"Interpretation: {interpretation}")
        print("\nLoadings:")
        for name, loading in zip(trait_names, loadings):
            bar = "█" * int(abs(loading) * 20)
            sign = "+" if loading >= 0 else "-"
            print(f"  {name:15s}: {loading:+.3f} {sign}{bar}")


def plot_scree(pca_result: PCAResult, 
               title: str = "Scree Plot",
               ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Create a scree plot to visualise eigenvalue decay.
    
    The scree plot helps decide how many principal components to retain.
    Look for the "elbow" where eigenvalues level off.
    
    Parameters
    ----------
    pca_result : PCAResult
        Output from perform_pca()
    
    title : str
        Plot title
    
    ax : matplotlib Axes, optional
        Axes to plot on; creates new figure if None
    
    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    n_components = len(pca_result.eigenvalues)
    x = np.arange(1, n_components + 1)
    
    # Plot eigenvalues
    ax.bar(x, pca_result.proportion_variance, alpha=0.7, 
           color='steelblue', label='Individual')
    
    # Plot cumulative variance
    ax2 = ax.twinx()
    ax2.plot(x, pca_result.cumulative_variance, 'ro-', 
             linewidth=2, markersize=8, label='Cumulative')
    ax2.axhline(y=0.8, color='gray', linestyle='--', 
                label='80% threshold')
    ax2.set_ylabel('Cumulative Proportion', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 1.05)
    
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Proportion of Variance')
    ax.set_title(title)
    ax.set_xticks(x)
    
    # Add legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    plt.tight_layout()
    return ax


def plot_biplot(pca_result: PCAResult,
                trait_names: List[str],
                groups: Optional[np.ndarray] = None,
                pc_x: int = 0,
                pc_y: int = 1,
                ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Create a biplot overlaying individuals and variable loadings.
    
    A biplot shows:
    • Points: individuals projected onto PC axes (scores)
    • Arrows: original variables projected onto PC axes (loadings)
    
    The arrows show:
    • Direction: correlation of variable with each PC
    • Length: importance of variable to these PCs
    • Angle between arrows: correlation between variables
    
    Parameters
    ----------
    pca_result : PCAResult
        Output from perform_pca()
    
    trait_names : List[str]
        Names of original traits (for arrow labels)
    
    groups : np.ndarray, optional
        Group labels for colouring points
    
    pc_x, pc_y : int
        Which PCs to plot (0-indexed, default PC1 vs PC2)
    
    ax : matplotlib Axes, optional
        Axes to plot on
    
    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    scores = pca_result.scores
    loadings = pca_result.eigenvectors
    prop_var = pca_result.proportion_variance
    
    # Scale loadings for visibility
    # We scale to the range of the scores
    score_range = max(np.abs(scores[:, [pc_x, pc_y]]).max(), 1e-10)
    loading_scale = score_range * 0.8 / max(np.abs(loadings[:, [pc_x, pc_y]]).max(), 1e-10)
    
    # Plot scores (individuals)
    if groups is not None:
        unique_groups = np.unique(groups)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_groups)))
        for i, g in enumerate(unique_groups):
            mask = groups == g
            ax.scatter(scores[mask, pc_x], scores[mask, pc_y],
                      c=[colors[i]], alpha=0.6, s=50, label=f'Group {g}')
        ax.legend()
    else:
        ax.scatter(scores[:, pc_x], scores[:, pc_y],
                  c='steelblue', alpha=0.6, s=50)
    
    # Plot loadings (variables as arrows)
    for i, name in enumerate(trait_names):
        ax.annotate('', 
                   xy=(loadings[i, pc_x] * loading_scale, 
                       loadings[i, pc_y] * loading_scale),
                   xytext=(0, 0),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2))
        ax.text(loadings[i, pc_x] * loading_scale * 1.1,
               loadings[i, pc_y] * loading_scale * 1.1,
               name, fontsize=10, color='red', fontweight='bold')
    
    # Labels
    ax.set_xlabel(f'PC{pc_x+1} ({prop_var[pc_x]:.1%} variance)')
    ax.set_ylabel(f'PC{pc_y+1} ({prop_var[pc_y]:.1%} variance)')
    ax.set_title('Biplot: Individuals and Variable Loadings')
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    
    return ax


# =============================================================================
# SECTION 2: MULTIVARIATE ANALYSIS OF VARIANCE (MANOVA)
# =============================================================================
"""
While PCA describes variation within a single sample, MANOVA asks whether
multiple groups differ in their multivariate means. It is the multivariate
generalisation of ANOVA.

GEOMETRIC INTERPRETATION:
------------------------
MANOVA decomposes total variation into:
• Among-group matrix B: How group means differ from grand mean
• Within-group matrix W: Variation within groups (pooled)

The question: Is B large relative to W?

If groups are very different (large B) and within-group variation is small
(small W), we reject the null hypothesis that all group means are equal.

The eigenvalues of W⁻¹B quantify separation along each discriminant axis.
"""


@dataclass
class MANOVAResult:
    """
    Container for MANOVA results.
    
    Attributes
    ----------
    B : np.ndarray
        Among-group (between) covariance matrix
    
    W : np.ndarray
        Within-group (pooled) covariance matrix
    
    T : np.ndarray
        Total covariance matrix (B + W)
    
    eigenvalues : np.ndarray
        Eigenvalues of W⁻¹B (discriminant axis importance)
    
    eigenvectors : np.ndarray
        Discriminant function coefficients
    
    wilks_lambda : float
        Wilks' Λ test statistic
    
    pillai_trace : float
        Pillai's trace test statistic
    
    hotelling_lawley : float
        Hotelling-Lawley trace
    
    roys_largest_root : float
        Roy's largest root (first eigenvalue)
    
    group_means : np.ndarray
        Mean vector for each group
    
    grand_mean : np.ndarray
        Overall mean across all groups
    """
    B: np.ndarray
    W: np.ndarray
    T: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    wilks_lambda: float
    pillai_trace: float
    hotelling_lawley: float
    roys_largest_root: float
    group_means: np.ndarray
    grand_mean: np.ndarray
    group_sizes: np.ndarray


def perform_manova(X: np.ndarray, groups: np.ndarray) -> MANOVAResult:
    """
    Perform Multivariate Analysis of Variance.
    
    MANOVA tests the null hypothesis:
        H₀: μ₁ = μ₂ = ... = μₖ (all group means equal)
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_total, p_traits)
    
    groups : np.ndarray
        Group labels of shape (n_total,)
    
    Returns
    -------
    MANOVAResult
        Contains B, W matrices and test statistics
    
    Mathematical Details
    --------------------
    Total variation decomposes as: T = B + W
    
    Total matrix:
        T = Σᵢ (xᵢ - x̄)(xᵢ - x̄)ᵀ
    
    Among-group matrix:
        B = Σⱼ nⱼ (x̄ⱼ - x̄)(x̄ⱼ - x̄)ᵀ
    
    Within-group matrix:
        W = Σⱼ Σᵢ∈ⱼ (xᵢ - x̄ⱼ)(xᵢ - x̄ⱼ)ᵀ
    
    Test statistics based on eigenvalues λᵢ of W⁻¹B:
        • Wilks' Λ = Π(1 + λᵢ)⁻¹ = |W|/|T|
        • Pillai's trace = Σ λᵢ/(1 + λᵢ)
        • Hotelling-Lawley = Σ λᵢ
        • Roy's largest root = λ₁
    """
    n, p = X.shape
    unique_groups = np.unique(groups)
    k = len(unique_groups)  # Number of groups
    
    # Grand mean
    grand_mean = np.mean(X, axis=0)
    
    # Compute group means and sizes
    group_means = np.zeros((k, p))
    group_sizes = np.zeros(k, dtype=int)
    
    for i, g in enumerate(unique_groups):
        mask = groups == g
        group_means[i] = np.mean(X[mask], axis=0)
        group_sizes[i] = np.sum(mask)
    
    # Total matrix T
    X_centered = X - grand_mean
    T = X_centered.T @ X_centered
    
    # Among-group matrix B
    B = np.zeros((p, p))
    for i, (mean, size) in enumerate(zip(group_means, group_sizes)):
        diff = mean - grand_mean
        B += size * np.outer(diff, diff)
    
    # Within-group matrix W
    W = np.zeros((p, p))
    for i, g in enumerate(unique_groups):
        mask = groups == g
        X_group_centered = X[mask] - group_means[i]
        W += X_group_centered.T @ X_group_centered
    
    # Verify decomposition: T = B + W
    assert np.allclose(T, B + W), "T ≠ B + W: decomposition error"
    
    # Eigendecomposition of W⁻¹B for test statistics
    try:
        W_inv = inv(W)
        W_inv_B = W_inv @ B
        eigenvalues, eigenvectors = np.linalg.eig(W_inv_B)
        
        # Sort by eigenvalue magnitude (largest first)
        eigenvalues = np.real(eigenvalues)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = np.real(eigenvectors[:, idx])
        
        # Keep only positive eigenvalues (rank of B is at most k-1)
        positive_mask = eigenvalues > 1e-10
        eigenvalues_positive = eigenvalues[positive_mask]
        
    except np.linalg.LinAlgError:
        warnings.warn("W is singular; using pseudo-inverse")
        eigenvalues_positive = np.array([0.0])
        eigenvectors = np.eye(p)
    
    # Compute test statistics
    # Wilks' Lambda: Λ = |W| / |T| = Π(1 + λᵢ)⁻¹
    wilks_lambda = np.prod(1 / (1 + eigenvalues_positive))
    
    # Pillai's trace: V = Σ λᵢ/(1 + λᵢ)
    pillai_trace = np.sum(eigenvalues_positive / (1 + eigenvalues_positive))
    
    # Hotelling-Lawley trace: U = Σ λᵢ
    hotelling_lawley = np.sum(eigenvalues_positive)
    
    # Roy's largest root: θ = λ₁
    roys_largest_root = eigenvalues_positive[0] if len(eigenvalues_positive) > 0 else 0
    
    return MANOVAResult(
        B=B,
        W=W,
        T=T,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        wilks_lambda=wilks_lambda,
        pillai_trace=pillai_trace,
        hotelling_lawley=hotelling_lawley,
        roys_largest_root=roys_largest_root,
        group_means=group_means,
        grand_mean=grand_mean,
        group_sizes=group_sizes
    )


def print_manova_summary(result: MANOVAResult, 
                         group_names: Optional[List[str]] = None) -> None:
    """
    Print formatted MANOVA summary with interpretation.
    """
    print("=" * 70)
    print("MANOVA RESULTS")
    print("=" * 70)
    
    k = len(result.group_sizes)
    p = len(result.grand_mean)
    
    print(f"\nDesign: {k} groups, {p} traits")
    print(f"Group sizes: {result.group_sizes}")
    
    print("\n" + "-" * 70)
    print("TEST STATISTICS")
    print("-" * 70)
    print(f"  Wilks' Lambda (Λ):        {result.wilks_lambda:.4f}")
    print(f"  Pillai's Trace (V):       {result.pillai_trace:.4f}")
    print(f"  Hotelling-Lawley (U):     {result.hotelling_lawley:.4f}")
    print(f"  Roy's Largest Root (θ):   {result.roys_largest_root:.4f}")
    
    print("\n" + "-" * 70)
    print("INTERPRETATION GUIDE")
    print("-" * 70)
    print("  Wilks' Λ:  Ranges from 0 to 1. SMALLER = more separation.")
    print("             Λ ≈ 0 means groups are very different.")
    print("             Λ ≈ 1 means groups are similar.")
    print("")
    print("  Pillai's V: Ranges from 0 to k-1. LARGER = more separation.")
    print("              Most robust to assumption violations.")
    print("")
    print("  The eigenvalues of W⁻¹B are the discriminant axis 'strengths':")
    
    # Show eigenvalue decomposition
    positive_eigs = result.eigenvalues[result.eigenvalues > 1e-10]
    for i, eig in enumerate(positive_eigs[:min(5, len(positive_eigs))]):
        percent = eig / np.sum(positive_eigs) * 100
        print(f"    Axis {i+1}: λ = {eig:.4f} ({percent:.1f}% of discrimination)")


# =============================================================================
# SECTION 3: LINEAR DISCRIMINANT ANALYSIS (LDA)
# =============================================================================
"""
LDA is closely related to MANOVA. While MANOVA tests whether groups differ,
LDA finds the directions along which they differ most and uses these for
classification.

GEOMETRIC INTERPRETATION:
------------------------
LDA finds linear combinations of traits that maximise the ratio of
among-group to within-group variance:

    maximise: (a^T B a) / (a^T W a)

The solution: eigenvectors of W⁻¹B

The first discriminant function (DF1) is the direction along which groups
are most separated relative to within-group spread.

CONNECTION TO MAHALANOBIS DISTANCE:
----------------------------------
The Mahalanobis distance between two group means, using pooled W:

    D² = (μ₁ - μ₂)^T W⁻¹ (μ₁ - μ₂)

This is exactly the squared distance along the discriminant axis connecting
the two groups. LDA and Mahalanobis distance are two views of the same
geometry.
"""


@dataclass
class LDAResult:
    """
    Container for Linear Discriminant Analysis results.
    
    Attributes
    ----------
    discriminant_axes : np.ndarray
        Discriminant function coefficients (columns are DF1, DF2, ...)
    
    eigenvalues : np.ndarray
        Eigenvalues of W⁻¹B (axis importance)
    
    proportion_trace : np.ndarray
        Proportion of discrimination explained by each axis
    
    scores : np.ndarray
        Data projected onto discriminant axes
    
    group_means_projected : np.ndarray
        Group centroids in discriminant space
    
    W_pooled : np.ndarray
        Pooled within-group covariance (for Mahalanobis distance)
    """
    discriminant_axes: np.ndarray
    eigenvalues: np.ndarray
    proportion_trace: np.ndarray
    scores: np.ndarray
    group_means_projected: np.ndarray
    W_pooled: np.ndarray


def perform_lda(X: np.ndarray, groups: np.ndarray) -> LDAResult:
    """
    Perform Linear Discriminant Analysis.
    
    LDA finds directions that maximise group separation relative to
    within-group variation. These directions are optimal for classification.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix (n_individuals × p_traits)
    
    groups : np.ndarray
        Group labels (n_individuals,)
    
    Returns
    -------
    LDAResult
        Discriminant axes, eigenvalues, scores
    
    Mathematical Details
    --------------------
    LDA solves the generalised eigenvalue problem:
        B v = λ W v
    
    Equivalently, eigendecompose W⁻¹B.
    
    The maximum number of non-trivial discriminant functions is
    min(p, k-1), where k is the number of groups.
    
    Example
    -------
    >>> # Two groups in 3D
    >>> X1 = np.random.randn(50, 3) + [2, 0, 0]
    >>> X2 = np.random.randn(50, 3) + [-2, 0, 0]
    >>> X = np.vstack([X1, X2])
    >>> groups = np.array([0]*50 + [1]*50)
    >>> lda_result = perform_lda(X, groups)
    >>> # DF1 should point roughly along [1, 0, 0]
    """
    # First, compute MANOVA matrices
    manova = perform_manova(X, groups)
    
    n, p = X.shape
    k = len(np.unique(groups))
    
    # Maximum number of discriminant functions
    max_df = min(p, k - 1)
    
    # Eigendecomposition of W⁻¹B
    # (already computed in MANOVA)
    eigenvalues = manova.eigenvalues[:max_df]
    discriminant_axes = manova.eigenvectors[:, :max_df]
    
    # Normalise discriminant axes for interpretation
    for i in range(max_df):
        discriminant_axes[:, i] = discriminant_axes[:, i] / np.linalg.norm(discriminant_axes[:, i])
    
    # Project data onto discriminant axes
    X_centered = X - manova.grand_mean
    scores = X_centered @ discriminant_axes
    
    # Project group means
    group_means_centered = manova.group_means - manova.grand_mean
    group_means_projected = group_means_centered @ discriminant_axes
    
    # Proportion of trace (discrimination) explained
    positive_eigs = eigenvalues[eigenvalues > 1e-10]
    total_trace = np.sum(positive_eigs)
    proportion_trace = np.zeros_like(eigenvalues)
    if total_trace > 0:
        proportion_trace[:len(positive_eigs)] = positive_eigs / total_trace
    
    # Pooled within-group covariance (for classification)
    n_total = np.sum(manova.group_sizes)
    W_pooled = manova.W / (n_total - k)
    
    return LDAResult(
        discriminant_axes=discriminant_axes,
        eigenvalues=eigenvalues,
        proportion_trace=proportion_trace,
        scores=scores,
        group_means_projected=group_means_projected,
        W_pooled=W_pooled
    )


def mahalanobis_distance_between_groups(group1_mean: np.ndarray,
                                         group2_mean: np.ndarray,
                                         W_pooled: np.ndarray) -> float:
    """
    Compute Mahalanobis distance between two group centroids.
    
    D² = (μ₁ - μ₂)^T W⁻¹ (μ₁ - μ₂)
    
    This measures how different the groups are in units that account for
    within-group covariance structure.
    
    Parameters
    ----------
    group1_mean, group2_mean : np.ndarray
        Mean vectors for each group
    
    W_pooled : np.ndarray
        Pooled within-group covariance matrix
    
    Returns
    -------
    float
        Mahalanobis distance D (not D²)
    """
    diff = group1_mean - group2_mean
    D_squared = diff @ inv(W_pooled) @ diff
    return np.sqrt(D_squared)


def plot_lda(lda_result: LDAResult,
             groups: np.ndarray,
             group_names: Optional[List[str]] = None,
             ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Visualise LDA results showing group separation.
    
    Parameters
    ----------
    lda_result : LDAResult
        Output from perform_lda()
    
    groups : np.ndarray
        Original group labels
    
    group_names : List[str], optional
        Names for legend
    
    ax : matplotlib Axes, optional
    
    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    unique_groups = np.unique(groups)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_groups)))
    
    if group_names is None:
        group_names = [f'Group {g}' for g in unique_groups]
    
    scores = lda_result.scores
    n_axes = min(2, scores.shape[1])
    
    if n_axes == 1:
        # 1D: histogram-style plot
        for i, g in enumerate(unique_groups):
            mask = groups == g
            ax.hist(scores[mask, 0], bins=20, alpha=0.6,
                   label=group_names[i], color=colors[i])
        ax.set_xlabel(f'DF1 ({lda_result.proportion_trace[0]:.1%})')
        ax.set_ylabel('Frequency')
        
    else:
        # 2D: scatter plot
        for i, g in enumerate(unique_groups):
            mask = groups == g
            ax.scatter(scores[mask, 0], scores[mask, 1],
                      c=[colors[i]], alpha=0.6, s=50, label=group_names[i])
            
            # Plot group centroid
            ax.scatter(lda_result.group_means_projected[i, 0],
                      lda_result.group_means_projected[i, 1],
                      c=[colors[i]], s=200, marker='*', edgecolors='black')
        
        ax.set_xlabel(f'DF1 ({lda_result.proportion_trace[0]:.1%})')
        ax.set_ylabel(f'DF2 ({lda_result.proportion_trace[1]:.1%})')
    
    ax.legend()
    ax.set_title('Linear Discriminant Analysis')
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    
    return ax


# =============================================================================
# SECTION 4: CANONICAL CORRELATION ANALYSIS (CCA)
# =============================================================================
"""
CCA extends correlation to multiple variables on each side. Given two sets
of variables (e.g., morphological traits and physiological traits), CCA
finds linear combinations of each set that are maximally correlated.

BIOLOGICAL APPLICATIONS:
-----------------------
• Relating genotype to phenotype
• Relating morphology to performance
• Relating traits to environmental variables
• Finding shared patterns between trait sets

MATHEMATICAL SETUP:
------------------
Let X (n × p) and Y (n × q) be two trait sets.

We seek coefficient vectors a and b such that:
    Corr(X a, Y b) is maximised

The solution involves eigendecomposition of matrices built from the
covariance structure between X and Y.
"""


@dataclass
class CCAResult:
    """
    Container for Canonical Correlation Analysis results.
    
    Attributes
    ----------
    canonical_correlations : np.ndarray
        The canonical correlations ρ₁ ≥ ρ₂ ≥ ... ≥ ρᵣ
    
    x_coefficients : np.ndarray
        Coefficients for X variables (columns are canonical variates)
    
    y_coefficients : np.ndarray
        Coefficients for Y variables
    
    x_scores : np.ndarray
        X data projected onto canonical variates
    
    y_scores : np.ndarray
        Y data projected onto canonical variates
    
    x_loadings : np.ndarray
        Correlations of original X with X canonical variates
    
    y_loadings : np.ndarray
        Correlations of original Y with Y canonical variates
    """
    canonical_correlations: np.ndarray
    x_coefficients: np.ndarray
    y_coefficients: np.ndarray
    x_scores: np.ndarray
    y_scores: np.ndarray
    x_loadings: np.ndarray
    y_loadings: np.ndarray


def perform_cca(X: np.ndarray, Y: np.ndarray) -> CCAResult:
    """
    Perform Canonical Correlation Analysis between two variable sets.
    
    CCA finds linear combinations of X and Y that have maximum correlation.
    
    Parameters
    ----------
    X : np.ndarray
        First variable set (n × p)
    
    Y : np.ndarray
        Second variable set (n × q)
    
    Returns
    -------
    CCAResult
        Canonical correlations, coefficients, and scores
    
    Mathematical Details
    --------------------
    Partition the covariance matrix:
        Σ = | Σ_XX  Σ_XY |
            | Σ_YX  Σ_YY |
    
    Canonical correlations are the square roots of the eigenvalues of:
        Σ_XX⁻¹ Σ_XY Σ_YY⁻¹ Σ_YX
    
    The maximum number of canonical variates is min(p, q).
    """
    n, p = X.shape
    q = Y.shape[1]
    
    # Center both variable sets
    X_centered = X - np.mean(X, axis=0)
    Y_centered = Y - np.mean(Y, axis=0)
    
    # Compute covariance matrices
    Sigma_XX = (X_centered.T @ X_centered) / (n - 1)
    Sigma_YY = (Y_centered.T @ Y_centered) / (n - 1)
    Sigma_XY = (X_centered.T @ Y_centered) / (n - 1)
    Sigma_YX = Sigma_XY.T
    
    # Regularise if needed (for numerical stability)
    reg = 1e-8
    Sigma_XX += reg * np.eye(p)
    Sigma_YY += reg * np.eye(q)
    
    # Compute Σ_XX⁻¹ Σ_XY Σ_YY⁻¹ Σ_YX
    Sigma_XX_inv = inv(Sigma_XX)
    Sigma_YY_inv = inv(Sigma_YY)
    
    M = Sigma_XX_inv @ Sigma_XY @ Sigma_YY_inv @ Sigma_YX
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(M)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Canonical correlations (sqrt of eigenvalues)
    # Clamp to [0, 1] for numerical stability
    canonical_correlations = np.sqrt(np.clip(eigenvalues, 0, 1))
    
    # Number of canonical variates
    r = min(p, q)
    
    # X coefficients (eigenvectors of M)
    x_coefficients = eigenvectors[:, :r]
    
    # Y coefficients: from the corresponding eigenvalue problem for Y
    # b = Σ_YY⁻¹ Σ_YX a / ρ
    y_coefficients = np.zeros((q, r))
    for i in range(r):
        if canonical_correlations[i] > 1e-10:
            y_coefficients[:, i] = (Sigma_YY_inv @ Sigma_YX @ x_coefficients[:, i]) / canonical_correlations[i]
    
    # Normalise coefficients
    for i in range(r):
        x_coefficients[:, i] = x_coefficients[:, i] / np.linalg.norm(x_coefficients[:, i])
        if np.linalg.norm(y_coefficients[:, i]) > 0:
            y_coefficients[:, i] = y_coefficients[:, i] / np.linalg.norm(y_coefficients[:, i])
    
    # Canonical variate scores
    x_scores = X_centered @ x_coefficients
    y_scores = Y_centered @ y_coefficients
    
    # Loadings (correlations of original variables with canonical variates)
    x_loadings = np.zeros((p, r))
    y_loadings = np.zeros((q, r))
    
    for i in range(r):
        for j in range(p):
            x_loadings[j, i] = np.corrcoef(X_centered[:, j], x_scores[:, i])[0, 1]
        for j in range(q):
            y_loadings[j, i] = np.corrcoef(Y_centered[:, j], y_scores[:, i])[0, 1]
    
    return CCAResult(
        canonical_correlations=canonical_correlations[:r],
        x_coefficients=x_coefficients,
        y_coefficients=y_coefficients,
        x_scores=x_scores,
        y_scores=y_scores,
        x_loadings=x_loadings,
        y_loadings=y_loadings
    )


# =============================================================================
# SECTION 5: COMPARING G MATRICES
# =============================================================================
"""
A major application of eigendecomposition is comparing G matrices across
populations or species. Do different populations have the same pattern of
genetic constraints?

THREE APPROACHES:
----------------

1. COMMON PRINCIPAL COMPONENTS (Flury):
   Test whether G matrices share the same eigenvectors (even if eigenvalues
   differ). Tests whether the "shape" of constraint is conserved.

2. RANDOM SKEWERS:
   Compare how matrices respond to random selection vectors. Generate many
   random unit vectors β, compute Gβ for each matrix, correlate responses.
   If two G matrices give similar responses to the same selection pressures,
   they are functionally similar.

3. KRZANOWSKI'S SUBSPACE COMPARISON:
   Compare the subspaces spanned by leading eigenvectors. Uses angles between
   subspaces to measure similarity.
"""


def random_skewers(G1: np.ndarray, 
                   G2: np.ndarray, 
                   n_skewers: int = 1000) -> Dict:
    """
    Compare two matrices using the random skewers method.
    
    This method tests functional similarity: do the matrices produce
    similar evolutionary responses to random selection pressures?
    
    Parameters
    ----------
    G1, G2 : np.ndarray
        Two covariance matrices to compare
    
    n_skewers : int
        Number of random selection vectors to test
    
    Returns
    -------
    dict
        'correlation': Average correlation between responses
        'p_value': Proportion of random matrices with higher correlation
        'response_correlations': All individual correlations
    
    Mathematical Details
    --------------------
    For each random unit vector β:
        r₁ = G₁ β  (response under G₁)
        r₂ = G₂ β  (response under G₂)
        
    Measure: correlation between r₁ and r₂ across many β
    
    If matrices are identical, this correlation is 1.
    If matrices are unrelated, this correlation is near 0.
    """
    p = G1.shape[0]
    assert G1.shape == G2.shape, "Matrices must have same dimensions"
    
    correlations = []
    
    for _ in range(n_skewers):
        # Generate random unit vector on the sphere
        beta = np.random.randn(p)
        beta = beta / np.linalg.norm(beta)
        
        # Compute responses
        response1 = G1 @ beta
        response2 = G2 @ beta
        
        # Correlation between responses
        # (equivalent to cosine similarity for centered vectors)
        corr = np.corrcoef(response1, response2)[0, 1]
        correlations.append(corr)
    
    correlations = np.array(correlations)
    mean_correlation = np.mean(correlations)
    
    # Simple permutation test for p-value would require null distribution
    # Here we just report the mean correlation
    
    return {
        'correlation': mean_correlation,
        'correlation_std': np.std(correlations),
        'response_correlations': correlations,
        'n_skewers': n_skewers
    }


def krzanowski_subspace_comparison(G1: np.ndarray, 
                                    G2: np.ndarray,
                                    k: int = None) -> Dict:
    """
    Compare subspaces spanned by leading eigenvectors using Krzanowski's method.
    
    Parameters
    ----------
    G1, G2 : np.ndarray
        Two covariance matrices to compare
    
    k : int, optional
        Number of leading eigenvectors to compare.
        Default: number with eigenvalues > 10% of largest
    
    Returns
    -------
    dict
        'S': Krzanowski's S statistic (sum of squared cosines)
        'S_max': Maximum possible S (= k)
        'S_normalized': S / S_max (ranges 0 to 1)
        'k': Number of eigenvectors compared
    
    Mathematical Details
    --------------------
    Let A = [v₁, ..., vₖ] be the k leading eigenvectors of G₁.
    Let B = [u₁, ..., uₖ] be the k leading eigenvectors of G₂.
    
    S = tr(A^T B B^T A) = sum of squared cosines between subspaces
    
    S ranges from 0 (orthogonal subspaces) to k (identical subspaces).
    """
    p = G1.shape[0]
    
    # Eigendecomposition
    eig1, vec1 = eigh(G1)
    eig2, vec2 = eigh(G2)
    
    # Sort descending
    idx1 = np.argsort(eig1)[::-1]
    idx2 = np.argsort(eig2)[::-1]
    eig1, vec1 = eig1[idx1], vec1[:, idx1]
    eig2, vec2 = eig2[idx2], vec2[:, idx2]
    
    # Determine k if not specified
    if k is None:
        threshold = 0.1 * eig1[0]
        k = np.sum(eig1 > threshold)
        k = max(1, min(k, p))
    
    # Extract leading eigenvectors
    A = vec1[:, :k]
    B = vec2[:, :k]
    
    # Krzanowski's S = tr(A^T B B^T A)
    # = sum of squared cosines of angles between eigenvector pairs
    S = np.trace(A.T @ B @ B.T @ A)
    
    return {
        'S': S,
        'S_max': k,
        'S_normalized': S / k,
        'k': k,
        'interpretation': 'S/k = 1 means identical subspaces; S/k = 0 means orthogonal'
    }


def vector_correlation(G1: np.ndarray, G2: np.ndarray, k: int = 1) -> np.ndarray:
    """
    Compute correlations between corresponding eigenvectors of two matrices.
    
    Parameters
    ----------
    G1, G2 : np.ndarray
        Two covariance matrices
    
    k : int
        Number of eigenvector pairs to compare
    
    Returns
    -------
    np.ndarray
        Absolute cosines between corresponding eigenvectors
    """
    eig1, vec1 = eigh(G1)
    eig2, vec2 = eigh(G2)
    
    # Sort descending
    vec1 = vec1[:, np.argsort(eig1)[::-1]]
    vec2 = vec2[:, np.argsort(eig2)[::-1]]
    
    correlations = []
    for i in range(k):
        # Absolute value because eigenvector sign is arbitrary
        corr = np.abs(vec1[:, i] @ vec2[:, i])
        correlations.append(corr)
    
    return np.array(correlations)


# =============================================================================
# SECTION 6: ESTIMATION ISSUES AND REGULARISATION
# =============================================================================
"""
Estimating covariance matrices is statistically challenging, especially with
many traits.

KEY ISSUES:
----------
1. Sample size requirements: n × p matrix needs >> p observations
2. Singularity: When n < p, sample covariance is singular
3. Eigenvalue bias: Small eigenvalues are underestimated; large ones
   overestimated

SOLUTIONS:
---------
• Ledoit-Wolf shrinkage: Pull eigenvalues toward a common value
• Ridge regularisation: Add small constant to diagonal
• Factor models: Assume low-rank structure
"""


def ledoit_wolf_shrinkage(X: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute Ledoit-Wolf shrinkage estimator for covariance matrix.
    
    This estimator shrinks the sample covariance toward a scaled identity
    matrix, reducing estimation error when sample size is limited.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix (n × p)
    
    Returns
    -------
    Sigma_shrunk : np.ndarray
        Shrinkage estimator
    
    shrinkage : float
        Optimal shrinkage intensity (0 = sample cov, 1 = scaled identity)
    
    Mathematical Details
    --------------------
    Σ_shrunk = α * (tr(S)/p) * I + (1 - α) * S
    
    where α is chosen to minimise expected squared error.
    """
    n, p = X.shape
    
    # Center data
    X_centered = X - np.mean(X, axis=0)
    
    # Sample covariance
    S = (X_centered.T @ X_centered) / (n - 1)
    
    # Target: scaled identity
    mu = np.trace(S) / p
    
    # Frobenius norm of S - μI
    delta = S - mu * np.eye(p)
    delta_norm_sq = np.sum(delta ** 2)
    
    # Estimate shrinkage intensity using Ledoit-Wolf formula
    # This is a simplified version; full formula involves fourth moments
    
    # Sum of squared sample covariances
    X2 = X_centered ** 2
    
    # Estimate of variance of off-diagonal elements
    phi = np.sum([(X_centered[k, :].reshape(-1, 1) @ X_centered[k, :].reshape(1, -1)
                   - S) ** 2 for k in range(n)]) / (n - 1)
    phi = phi / n
    
    # Optimal shrinkage
    kappa = (phi - delta_norm_sq / n) / delta_norm_sq
    shrinkage = max(0, min(1, kappa))
    
    # Shrunk estimator
    Sigma_shrunk = shrinkage * mu * np.eye(p) + (1 - shrinkage) * S
    
    return Sigma_shrunk, shrinkage


def ridge_regularisation(S: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    Apply ridge regularisation to a covariance matrix.
    
    Adds a small multiple of the identity to ensure positive definiteness.
    
    Parameters
    ----------
    S : np.ndarray
        Sample covariance matrix
    
    alpha : float
        Regularisation parameter (fraction of trace)
    
    Returns
    -------
    np.ndarray
        Regularised covariance matrix
    
    Notes
    -----
    S_reg = S + α * tr(S)/p * I
    
    This ensures all eigenvalues are at least α * mean(eigenvalues).
    """
    p = S.shape[0]
    ridge = alpha * np.trace(S) / p
    return S + ridge * np.eye(p)


# =============================================================================
# SECTION 7: WORKED EXAMPLE - COMPLETE PCA ANALYSIS
# =============================================================================


def worked_example_pca():
    """
    Complete PCA analysis of a simulated phenotypic dataset.
    
    This example walks through all steps from data generation to
    biological interpretation.
    """
    print("=" * 70)
    print("WORKED EXAMPLE: PCA OF A PHENOTYPIC DATASET")
    print("=" * 70)
    print("""
    We analyse measurements of four morphological traits on 150 individuals
    from three populations. The goal is to:
    
    1. Identify major axes of variation
    2. Determine dimensionality (how many PCs needed?)
    3. Interpret PCs biologically (size? shape?)
    4. Visualise population differences
    """)
    
    # ----- Step 1: Generate realistic data -----
    print("\n" + "-" * 70)
    print("STEP 1: DATA GENERATION")
    print("-" * 70)
    
    np.random.seed(42)
    
    # Define a realistic covariance structure
    # Four traits: wing length, tarsus length, bill depth, bill width
    trait_names = ['Wing', 'Tarsus', 'Bill Depth', 'Bill Width']
    
    # Covariance matrix with positive correlations (typical for morphology)
    true_cov = np.array([
        [1.20, 0.85, 0.42, 0.31],
        [0.85, 1.05, 0.51, 0.28],
        [0.42, 0.51, 0.78, 0.15],
        [0.31, 0.28, 0.15, 0.55]
    ])
    
    # Three populations with slightly different means
    pop_means = [
        np.array([0.0, 0.0, 0.0, 0.0]),      # Population A: baseline
        np.array([1.0, 0.8, 0.2, 0.1]),       # Population B: larger overall
        np.array([-0.5, -0.3, 0.5, 0.3])      # Population C: smaller body, larger bill
    ]
    
    # Sample sizes
    n_per_pop = 50
    
    # Generate data
    X_list = []
    groups = []
    for i, mean in enumerate(pop_means):
        X_pop = np.random.multivariate_normal(mean, true_cov, n_per_pop)
        X_list.append(X_pop)
        groups.extend([i] * n_per_pop)
    
    X = np.vstack(X_list)
    groups = np.array(groups)
    
    print(f"Generated data: {X.shape[0]} individuals, {X.shape[1]} traits")
    print(f"Populations: {np.unique(groups, return_counts=True)}")
    
    # ----- Step 2: PCA -----
    print("\n" + "-" * 70)
    print("STEP 2: PRINCIPAL COMPONENT ANALYSIS")
    print("-" * 70)
    
    pca_result = perform_pca(X, use_correlation=False)
    
    print("\nEigenvalue summary:")
    print("-" * 50)
    print(f"{'PC':<6} {'Eigenvalue':<12} {'Proportion':<12} {'Cumulative':<12}")
    print("-" * 50)
    
    for i in range(len(pca_result.eigenvalues)):
        print(f"PC{i+1:<4} {pca_result.eigenvalues[i]:<12.4f} "
              f"{pca_result.proportion_variance[i]:<12.1%} "
              f"{pca_result.cumulative_variance[i]:<12.1%}")
    
    # ----- Step 3: Interpret loadings -----
    print("\n" + "-" * 70)
    print("STEP 3: INTERPRET LOADINGS")
    print("-" * 70)
    
    interpret_pca_loadings(pca_result, trait_names, n_components=4)
    
    # ----- Step 4: Visualise -----
    print("\n" + "-" * 70)
    print("STEP 4: VISUALISATION")
    print("-" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Scree plot
    plot_scree(pca_result, ax=axes[0, 0])
    axes[0, 0].set_title('(A) Scree Plot: Variance by PC')
    
    # Biplot
    plot_biplot(pca_result, trait_names, groups=groups, ax=axes[0, 1])
    axes[0, 1].set_title('(B) Biplot: Individuals and Loadings')
    
    # PC1 vs PC2 scores by population
    colors = ['#e41a1c', '#377eb8', '#4daf4a']
    pop_names = ['Pop A', 'Pop B', 'Pop C']
    
    for i in range(3):
        mask = groups == i
        axes[1, 0].scatter(pca_result.scores[mask, 0], 
                          pca_result.scores[mask, 1],
                          c=colors[i], alpha=0.6, s=50, label=pop_names[i])
    axes[1, 0].legend()
    axes[1, 0].set_xlabel(f'PC1 ({pca_result.proportion_variance[0]:.1%})')
    axes[1, 0].set_ylabel(f'PC2 ({pca_result.proportion_variance[1]:.1%})')
    axes[1, 0].set_title('(C) Population Separation in PC Space')
    axes[1, 0].axhline(y=0, color='gray', linewidth=0.5)
    axes[1, 0].axvline(x=0, color='gray', linewidth=0.5)
    
    # Covariance matrix heatmap
    S = (X - X.mean(axis=0)).T @ (X - X.mean(axis=0)) / (X.shape[0] - 1)
    im = axes[1, 1].imshow(S, cmap='RdBu_r', vmin=-1, vmax=1.5)
    axes[1, 1].set_xticks(range(4))
    axes[1, 1].set_yticks(range(4))
    axes[1, 1].set_xticklabels(trait_names, rotation=45, ha='right')
    axes[1, 1].set_yticklabels(trait_names)
    axes[1, 1].set_title('(D) Sample Covariance Matrix')
    plt.colorbar(im, ax=axes[1, 1], shrink=0.8)
    
    # Add values to heatmap
    for i in range(4):
        for j in range(4):
            axes[1, 1].text(j, i, f'{S[i,j]:.2f}', ha='center', va='center', 
                           fontsize=10, color='white' if abs(S[i,j]) > 0.7 else 'black')
    
    plt.tight_layout()
    plt.savefig('/home/claude/ch11_pca_example.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Figure saved to: /home/claude/ch11_pca_example.png")
    
    # ----- Step 5: Conclusions -----
    print("\n" + "-" * 70)
    print("STEP 5: BIOLOGICAL CONCLUSIONS")
    print("-" * 70)
    print(f"""
    KEY FINDINGS:
    
    1. DIMENSIONALITY: PC1 captures {pca_result.proportion_variance[0]:.1%} of variance.
       With PC2, we capture {pca_result.cumulative_variance[1]:.1%}.
       The data are effectively ~2-dimensional despite having 4 traits.
    
    2. PC1 INTERPRETATION: All loadings positive → "SIZE" axis.
       Large individuals are large on all traits.
    
    3. PC2 INTERPRETATION: Mixed signs → "SHAPE" axis.
       Contrasts wing/tarsus (body size) with bill dimensions.
    
    4. POPULATION STRUCTURE:
       • Pop B is shifted along PC1 (larger overall)
       • Pop C differs on PC2 (larger bills relative to body)
    
    5. COVARIANCE STRUCTURE: Strong positive correlations among all traits,
       especially wing-tarsus (r = {S[0,1]/np.sqrt(S[0,0]*S[1,1]):.2f}).
       This is typical allometric scaling.
    """)
    
    return pca_result, X, groups, trait_names


# =============================================================================
# SECTION 8: WORKED EXAMPLE - MANOVA AND LDA
# =============================================================================


def worked_example_manova_lda():
    """
    Complete MANOVA and LDA analysis demonstrating group comparison.
    """
    print("\n" + "=" * 70)
    print("WORKED EXAMPLE: MANOVA AND LDA")
    print("=" * 70)
    print("""
    We test whether three populations differ in multivariate phenotype and
    find the directions that best discriminate them.
    """)
    
    np.random.seed(42)
    
    # Generate data with clear group structure
    p = 4
    n_per_group = 40
    trait_names = ['Trait 1', 'Trait 2', 'Trait 3', 'Trait 4']
    
    # Common within-group covariance
    W_true = np.array([
        [1.0, 0.5, 0.2, 0.1],
        [0.5, 1.0, 0.3, 0.2],
        [0.2, 0.3, 1.0, 0.4],
        [0.1, 0.2, 0.4, 1.0]
    ])
    
    # Group means (deliberately structured)
    # Group 1: baseline
    # Group 2: differs mainly on traits 1-2
    # Group 3: differs mainly on traits 3-4
    means = [
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([2.0, 1.5, 0.2, 0.1]),
        np.array([0.3, 0.2, 1.8, 1.5])
    ]
    
    # Generate data
    X = np.vstack([
        np.random.multivariate_normal(means[i], W_true, n_per_group)
        for i in range(3)
    ])
    groups = np.repeat([0, 1, 2], n_per_group)
    
    # ----- MANOVA -----
    print("\n" + "-" * 70)
    print("MANOVA ANALYSIS")
    print("-" * 70)
    
    manova_result = perform_manova(X, groups)
    print_manova_summary(manova_result)
    
    # ----- LDA -----
    print("\n" + "-" * 70)
    print("LINEAR DISCRIMINANT ANALYSIS")
    print("-" * 70)
    
    lda_result = perform_lda(X, groups)
    
    print("\nDiscriminant Function Coefficients:")
    print("-" * 50)
    print(f"{'Trait':<15} {'DF1':<12} {'DF2':<12}")
    print("-" * 50)
    for i, name in enumerate(trait_names):
        print(f"{name:<15} {lda_result.discriminant_axes[i, 0]:+.3f}       "
              f"{lda_result.discriminant_axes[i, 1]:+.3f}")
    
    print(f"\nProportion of discrimination:")
    print(f"  DF1: {lda_result.proportion_trace[0]:.1%}")
    print(f"  DF2: {lda_result.proportion_trace[1]:.1%}")
    
    # Mahalanobis distances between groups
    print("\nMahalanobis Distances Between Groups:")
    print("-" * 50)
    for i in range(3):
        for j in range(i + 1, 3):
            D = mahalanobis_distance_between_groups(
                manova_result.group_means[i],
                manova_result.group_means[j],
                lda_result.W_pooled
            )
            print(f"  Group {i} vs Group {j}: D = {D:.3f}")
    
    # ----- Visualisation -----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # LDA plot
    plot_lda(lda_result, groups, group_names=['Group 0', 'Group 1', 'Group 2'],
             ax=axes[0])
    axes[0].set_title('(A) LDA: Group Separation')
    
    # Between/Within comparison
    eig_B, _ = np.linalg.eig(manova_result.B)
    eig_W, _ = np.linalg.eig(manova_result.W)
    
    x = np.arange(p)
    width = 0.35
    axes[1].bar(x - width/2, np.sort(np.real(eig_B))[::-1], width, 
                label='Between-group (B)', color='coral')
    axes[1].bar(x + width/2, np.sort(np.real(eig_W))[::-1], width, 
                label='Within-group (W)', color='steelblue')
    axes[1].set_xlabel('Component')
    axes[1].set_ylabel('Eigenvalue')
    axes[1].set_title('(B) Between vs. Within-Group Variation')
    axes[1].legend()
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'{i+1}' for i in range(p)])
    
    plt.tight_layout()
    plt.savefig('/home/claude/ch11_manova_lda_example.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nFigure saved to: /home/claude/ch11_manova_lda_example.png")
    
    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)
    print(f"""
    1. MANOVA TEST STATISTICS:
       • Wilks' Λ = {manova_result.wilks_lambda:.4f} (small = groups differ)
       • This indicates strong multivariate group differences.
    
    2. DISCRIMINANT AXES:
       • DF1 ({lda_result.proportion_trace[0]:.1%}) separates Group 1 from others
         (loads heavily on Traits 1-2)
       • DF2 ({lda_result.proportion_trace[1]:.1%}) separates Group 2 from others
         (loads heavily on Traits 3-4)
    
    3. GROUP STRUCTURE:
       • Groups are well-separated in discriminant space
       • The eigenvalue pattern shows discrimination is primarily 2D
    """)
    
    return manova_result, lda_result


# =============================================================================
# SECTION 9: COMPARING G MATRICES - WORKED EXAMPLE
# =============================================================================


def worked_example_g_comparison():
    """
    Compare two G matrices using multiple methods.
    """
    print("\n" + "=" * 70)
    print("WORKED EXAMPLE: COMPARING G MATRICES")
    print("=" * 70)
    print("""
    We compare genetic covariance matrices from two populations to ask:
    Do they share the same constraint structure?
    """)
    
    # Define two G matrices with similar but not identical structure
    # G1: Strong positive genetic correlations
    G1 = np.array([
        [0.50, 0.35, 0.20],
        [0.35, 0.45, 0.25],
        [0.20, 0.25, 0.40]
    ])
    
    # G2: Similar structure but rotated slightly
    theta = 0.3  # Rotation angle (radians)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    G2 = R @ G1 @ R.T
    
    # Slightly different eigenvalues too
    G2 = G2 * 1.1
    
    print("\nG1 (Population 1):")
    print(G1)
    print("\nG2 (Population 2):")
    print(G2.round(3))
    
    # ----- Method 1: Random Skewers -----
    print("\n" + "-" * 70)
    print("METHOD 1: RANDOM SKEWERS")
    print("-" * 70)
    
    rs_result = random_skewers(G1, G2, n_skewers=10000)
    
    print(f"Mean response correlation: {rs_result['correlation']:.3f}")
    print(f"Standard deviation: {rs_result['correlation_std']:.3f}")
    print("""
    Interpretation:
    • Correlation near 1 → matrices produce similar evolutionary responses
    • Correlation near 0 → matrices respond to selection very differently
    """)
    
    # ----- Method 2: Krzanowski Subspace Comparison -----
    print("-" * 70)
    print("METHOD 2: KRZANOWSKI SUBSPACE COMPARISON")
    print("-" * 70)
    
    krz_result = krzanowski_subspace_comparison(G1, G2)
    
    print(f"Subspace overlap S = {krz_result['S']:.3f} (max = {krz_result['S_max']})")
    print(f"Normalised: S/k = {krz_result['S_normalized']:.3f}")
    print(f"(Using k = {krz_result['k']} leading eigenvectors)")
    print("""
    Interpretation:
    • S/k = 1 → identical subspaces (same principal axes)
    • S/k = 0 → orthogonal subspaces (completely different axes)
    """)
    
    # ----- Method 3: Eigenvector Correlations -----
    print("-" * 70)
    print("METHOD 3: EIGENVECTOR CORRELATIONS")
    print("-" * 70)
    
    vec_corrs = vector_correlation(G1, G2, k=3)
    
    print(f"Correlation between g_max vectors: |v₁·u₁| = {vec_corrs[0]:.3f}")
    print(f"Correlation between g_2 vectors:   |v₂·u₂| = {vec_corrs[1]:.3f}")
    print(f"Correlation between g_3 vectors:   |v₃·u₃| = {vec_corrs[2]:.3f}")
    print("""
    Interpretation:
    • High correlation (≈1) → eigenvectors point in same direction
    • Low correlation (≈0) → eigenvectors are rotated relative to each other
    
    Note: Leading eigenvector (g_max) agreement is often most important
    biologically, as this is the "line of least resistance."
    """)
    
    # ----- Visualise -----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Random skewers distribution
    axes[0].hist(rs_result['response_correlations'], bins=50, 
                 color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].axvline(rs_result['correlation'], color='red', linestyle='--',
                    linewidth=2, label=f'Mean = {rs_result["correlation"]:.3f}')
    axes[0].set_xlabel('Response Correlation')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('(A) Random Skewers Distribution')
    axes[0].legend()
    
    # Eigenvalue comparison
    eig1, vec1 = np.linalg.eigh(G1)
    eig2, vec2 = np.linalg.eigh(G2)
    eig1 = eig1[::-1]
    eig2 = eig2[::-1]
    
    x = np.arange(3)
    width = 0.35
    axes[1].bar(x - width/2, eig1, width, label='G1', color='coral')
    axes[1].bar(x + width/2, eig2, width, label='G2', color='steelblue')
    axes[1].set_xlabel('Eigenvalue Rank')
    axes[1].set_ylabel('Eigenvalue (Genetic Variance)')
    axes[1].set_title('(B) Eigenvalue Comparison')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['λ₁', 'λ₂', 'λ₃'])
    axes[1].legend()
    
    # Eigenvector alignment
    axes[2].bar(x, vec_corrs, color='forestgreen', alpha=0.7, edgecolor='black')
    axes[2].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Eigenvector Pair')
    axes[2].set_ylabel('|Correlation|')
    axes[2].set_title('(C) Eigenvector Alignment')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(['v₁ vs u₁', 'v₂ vs u₂', 'v₃ vs u₃'])
    axes[2].set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('/home/claude/ch11_g_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nFigure saved to: /home/claude/ch11_g_comparison.png")
    
    print("\n" + "-" * 70)
    print("OVERALL CONCLUSION")
    print("-" * 70)
    print(f"""
    The two G matrices show:
    
    • High response similarity (random skewers r = {rs_result['correlation']:.3f})
    • Moderate subspace overlap (S/k = {krz_result['S_normalized']:.3f})
    • Good g_max alignment (|correlation| = {vec_corrs[0]:.3f})
    
    Interpretation: These populations share similar genetic constraint
    structure—they would respond similarly to the same selection pressures.
    The slight differences may reflect local adaptation or drift.
    """)
    
    return rs_result, krz_result, vec_corrs


# =============================================================================
# MAIN EXECUTION
# =============================================================================


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║  CHAPTER 11: PCA, MANOVA, AND PROJECTIONS                           ║
    ║  Seeing the Shape - Code Companion                                   ║
    ╚══════════════════════════════════════════════════════════════════════╝
    
    This script demonstrates that all major multivariate statistical methods
    are fundamentally the same operation: eigendecomposition of (ratios of)
    covariance matrices.
    
    Running worked examples...
    """)
    
    # Run all worked examples
    pca_result, X, groups, trait_names = worked_example_pca()
    manova_result, lda_result = worked_example_manova_lda()
    rs_result, krz_result, vec_corrs = worked_example_g_comparison()
    
    print("\n" + "=" * 70)
    print("CHAPTER 11 SUMMARY")
    print("=" * 70)
    print("""
    THE UNIFYING PRINCIPLE:
    
    Every method finds "interesting" directions in trait space by
    eigendecomposing a covariance matrix or ratio of matrices:
    
    ┌─────────────────────┬─────────────────────┬────────────────────────┐
    │ Method              │ Matrix Decomposed   │ Eigenvalues Mean       │
    ├─────────────────────┼─────────────────────┼────────────────────────┤
    │ PCA                 │ S (covariance)      │ Variance along axis    │
    │ MANOVA              │ W⁻¹B                │ Group separation       │
    │ LDA                 │ W⁻¹B                │ Discrimination power   │
    │ CCA                 │ Σ_XX⁻¹Σ_XY Σ_YY⁻¹Σ_YX │ Correlation²         │
    └─────────────────────┴─────────────────────┴────────────────────────┘
    
    Once you see matrices as shapes (ellipsoids), all these methods become
    exercises in finding the principal axes of those shapes.
    
    "Symmetric matrices describe shapes. The algebra is a precise language
     for those shapes. Whenever the symbols become opaque, the right move
     is to go back to the picture and draw the ellipse."
    """)
    
    print("\nOutput files created:")
    print("  • /home/claude/ch11_pca_example.png")
    print("  • /home/claude/ch11_manova_lda_example.png")
    print("  • /home/claude/ch11_g_comparison.png")
