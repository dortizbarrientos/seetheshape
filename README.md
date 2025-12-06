[README.md](https://github.com/user-attachments/files/23980986/README.md)
# Seeing the Shape: Code Companion

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![R 4.0+](https://img.shields.io/badge/R-4.0+-blue.svg)](https://www.r-project.org/)

**Computational companion to:**  
*Seeing the Shape: A Geometric Introduction to Multivariate Quantitative Genetics*  
by Daniel Ortiz-Barrientos  
School of the Environment, The University of Queensland

---

## 📖 About This Repository

This repository contains annotated code in **Python** and **R** that accompanies each chapter of *Seeing the Shape*. The code is designed to be pedagogical: each script explains not just *how* to perform calculations, but *why* they work geometrically and biologically.

The guiding philosophy mirrors that of the book:

> **Symmetric matrices describe shapes. The algebra is a precise language for those shapes. Whenever the symbols become opaque, the right move is to go back to the picture and draw the ellipse.**

## 🗂️ Repository Structure

```
seeing-the-shape-code/
├── README.md                 # This file
├── python/                   # Python implementations
│   ├── requirements.txt      # Dependencies
│   ├── ch01_points_trait_space.py
│   ├── ch02_vectors_coordinates.py
│   ├── ch03_matrices_transformations.py
│   ├── ch04_distance_variance.py
│   ├── ch05_euclidean_fails.py
│   ├── ch06_mahalanobis.py
│   ├── ch07_eigendecomposition.py
│   ├── ch08_whitening_psphere.py
│   ├── ch09_g_matrix.py
│   ├── ch10_fitness_gamma.py
│   ├── ch11_pca_manova.py
│   ├── ch12_worked_examples.py
│   └── ch13_directional_heritability.py
├── R/                        # R implementations
│   ├── ch01_points_trait_space.R
│   ├── ...
│   └── ch13_directional_heritability.R
└── figures/                  # Figure generation scripts
    ├── README.md
    └── generate_all_figures.py
```

## 🚀 Quick Start

### Python

```bash
# Clone the repository
git clone https://github.com/ortizbarrientoslab/seeing-the-shape-code.git
cd seeing-the-shape-code/python

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run a chapter script
python ch01_points_trait_space.py
```

### R

```r
# Install required packages
install.packages(c("MASS", "Matrix", "ggplot2", "ellipse", "mvtnorm"))

# Source a chapter script
source("R/ch01_points_trait_space.R")
```

## 📚 Chapter Guide

| Part | Chapter | Topic | Key Concepts |
|------|---------|-------|--------------|
| **I** | 1 | Points and Trait Space | Phenotypes as points, differences as arrows |
| | 2 | Vectors and Coordinates | Dot product, lengths, angles, projections |
| | 3 | Matrices as Machines | Linear transformations, stretch/rotate/shear |
| **II** | 4 | Distance and Variance | Pythagoras, squaring, covariance matrix |
| | 5 | When Euclidean Fails | Scale, correlation, probability problems |
| | 6 | Mahalanobis Distance | Inverse covariance, whitening preview |
| **III** | 7 | Diagonalisation | Eigenvalues, eigenvectors, spectral theorem |
| | 8 | Whitening and P-sphere | P⁻¹/², directional heritability, G* |
| **IV** | 9 | The G Matrix | Genetic ellipsoid, g_max, evolvability |
| | 10 | Fitness Surface and γ | Selection gradients, curvature, γ matrix |
| | 11 | PCA, MANOVA, Projections | Statistical applications of eigendecomposition |
| **V** | 12 | Worked Examples | Complete analyses from data to interpretation |
| | 13 | Directional Heritability | CV(h²), constraint geometry, dimensionality |

## 🔑 Key Functions by Chapter

### Core Geometric Operations

```python
# Chapter 6-7: The essential operations
def mahalanobis_distance(x, mu, Sigma):
    """Mahalanobis distance from x to mu given covariance Sigma."""
    diff = x - mu
    return np.sqrt(diff @ np.linalg.inv(Sigma) @ diff)

def eigendecompose(A):
    """Eigendecomposition of symmetric matrix A = V Λ V^T."""
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvalues[idx], eigenvectors[:, idx]
```

### G-P Analysis (Chapter 8, 12, 13)

```python
# The whitening transformation
def compute_G_star(G, P):
    """Compute P-whitened genetic matrix G* = P^{-1/2} G P^{-1/2}."""
    eigenvalues_P, V_P = np.linalg.eigh(P)
    P_inv_sqrt = V_P @ np.diag(1/np.sqrt(eigenvalues_P)) @ V_P.T
    return P_inv_sqrt @ G @ P_inv_sqrt

# Directional heritability
def directional_heritability(beta, G, P):
    """Heritability in direction beta: h²(β) = β'Gβ / β'Pβ."""
    return (beta @ G @ beta) / (beta @ P @ beta)
```

### Evolutionary Response (Chapter 9-10)

```python
# The multivariate breeder's equation
def selection_response(G, beta):
    """Δz̄ = Gβ: Response to selection given G and selection gradient β."""
    return G @ beta

def evolvability(beta, G):
    """Evolvability e(β) = β'Gβ: genetic variance in direction of selection."""
    beta_unit = beta / np.linalg.norm(beta)
    return beta_unit @ G @ beta_unit
```

## 📊 Notation Conventions

The code follows the book's notation:

| Symbol | Code (Python) | Code (R) | Meaning |
|--------|---------------|----------|---------|
| **x**, **z** | `x`, `z` | `x`, `z` | Phenotype vectors |
| **P** | `P` | `P` | Phenotypic covariance matrix |
| **G** | `G` | `G` | Additive genetic covariance matrix |
| **G*** | `G_star` | `G_star` | P-whitened G: P⁻¹/²GP⁻¹/² |
| **β** | `beta` | `beta` | Selection gradient |
| **γ** | `gamma` | `gamma` | Quadratic selection gradient (curvature) |
| λᵢ | `eigenvalues[i]` | `eigenvalues[i]` | Eigenvalue i |
| **vᵢ** | `eigenvectors[:, i]` | `eigenvectors[, i]` | Eigenvector i |
| h²(β) | `h2_beta` | `h2_beta` | Directional heritability |
| g_max | `g_max` | `g_max` | First eigenvector of G |

## 🎯 Learning Objectives

After working through this code, you should be able to:

1. **Visualize** covariance matrices as ellipses/ellipsoids
2. **Compute** eigendecompositions and interpret them biologically
3. **Apply** Mahalanobis distance and understand why it works
4. **Perform** P-whitening to analyze G in the context of P
5. **Calculate** directional heritability for any direction
6. **Predict** selection response using the breeder's equation
7. **Interpret** fitness surfaces via the γ matrix
8. **Run** complete G-P analyses from raw data

## 📝 Exercises

Each chapter file includes exercises at the end. Solutions are provided in a separate `solutions/` folder (available to instructors upon request).

## 🐛 Issues and Contributions

Found a bug? Have a suggestion? Please open an issue or submit a pull request.

## 📜 License

This code is released under the **CC BY-NC-SA 4.0** license, matching the book. You are free to share and adapt for non-commercial purposes with attribution.

## 📧 Contact

- **Author:** Daniel Ortiz-Barrientos
- **Lab website:** https://www.ortizbarrientoslab.org
- **Institution:** School of the Environment, The University of Queensland

---

*"The shape of the ellipsoid and the direction of the arrow—these two things, together, determine what will happen. The G matrix is potential; selection is actuality. Their interaction is evolution."*
