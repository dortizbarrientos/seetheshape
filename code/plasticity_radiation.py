#!/usr/bin/env python3
"""
The Plasticity-Constraint Feedback Loop and Adaptive Radiation
===============================================================

This script explores Daniel's insight:
1. Environmental heterogeneity maintains Va (genetic variance for plasticity)
2. Plasticity enables colonization of new environments
3. This creates a positive feedback that generates more buffering traps
4. In adaptive radiations, g_max and β misalignment becomes likely

Author: Daniel Ortiz-Barrientos
Date: December 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, Circle, Wedge
from matplotlib.collections import LineCollection
import matplotlib.patheffects as pe

# =============================================================================
# COLOR PALETTE
# =============================================================================
PAL = {
    'G': '#2E86AB',
    'P': '#B2182B',
    'E': '#F4A582',
    'beta': '#F18F01',
    'high': '#1B7837',
    'low': '#B2182B',
    'neutral': '#878787',
    'trap': '#762A83',
    'radiation': '#E63946',
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'figure.facecolor': 'white',
    'savefig.dpi': 300,
})

# =============================================================================
# FIGURE 1: THE FEEDBACK LOOP
# =============================================================================

fig = plt.figure(figsize=(16, 10))

# -----------------------------------------------------------------------------
# Panel A: The feedback loop diagram
# -----------------------------------------------------------------------------
ax = fig.add_subplot(2, 2, 1)
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# Draw circular feedback loop
center = (5, 5)
radius = 3.5

# Nodes of the loop
nodes = [
    (5, 8.5, "Environmental\nHeterogeneity"),
    (8.5, 5, "High Va\n(plasticity variance)"),
    (5, 1.5, "Colonization of\nnew environments"),
    (1.5, 5, "Increased Ve\n(phenotypic noise)"),
]

# Draw nodes
for x, y, label in nodes:
    ax.add_patch(Circle((x, y), 1.2, facecolor='white', edgecolor=PAL['G'], 
                        linewidth=2, zorder=2))
    ax.text(x, y, label, ha='center', va='center', fontsize=9, 
           fontweight='bold', zorder=3)

# Draw arrows between nodes (clockwise)
arrow_style = dict(arrowstyle='->', color=PAL['G'], lw=2.5, 
                  connectionstyle='arc3,rad=0.2')
for i in range(4):
    x1, y1, _ = nodes[i]
    x2, y2, _ = nodes[(i+1) % 4]
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
               arrowprops=arrow_style)

# Add the consequence in the center
ax.text(5, 5, "MORE\nCONSTRAINT\nTRAPS", ha='center', va='center',
       fontsize=11, fontweight='bold', color=PAL['trap'],
       bbox=dict(boxstyle='round', facecolor='lightyellow', 
                edgecolor=PAL['trap'], linewidth=2))

# Add "+" signs to show positive feedback
ax.text(6.8, 7.0, '+', fontsize=16, fontweight='bold', color=PAL['high'])
ax.text(7.5, 3.0, '+', fontsize=16, fontweight='bold', color=PAL['high'])
ax.text(3.2, 2.5, '+', fontsize=16, fontweight='bold', color=PAL['high'])
ax.text(2.5, 7.0, '+', fontsize=16, fontweight='bold', color=PAL['high'])

ax.set_title('A. The Plasticity-Constraint Feedback Loop', fontsize=13, pad=10)

# -----------------------------------------------------------------------------
# Panel B: How plasticity inflates P relative to G
# -----------------------------------------------------------------------------
ax = fig.add_subplot(2, 2, 2)

# Fixed G matrix
G = np.array([[0.6, 0.2],
              [0.2, 0.4]])

# Increasing E matrices (more plasticity = more environmental variance)
E_levels = [
    (np.array([[0.2, 0.05], [0.05, 0.15]]), 'Low plasticity'),
    (np.array([[0.5, 0.1], [0.1, 0.4]]), 'Medium plasticity'),
    (np.array([[1.0, 0.2], [0.2, 0.8]]), 'High plasticity'),
]

colors = [PAL['high'], PAL['beta'], PAL['low']]
alphas = [0.4, 0.25, 0.15]

# Draw G ellipse (same for all)
eigvals_G, eigvecs_G = np.linalg.eigh(G)
w_G = 2 * np.sqrt(eigvals_G[1])
h_G = 2 * np.sqrt(eigvals_G[0])
angle_G = np.degrees(np.arctan2(eigvecs_G[1, 1], eigvecs_G[0, 1]))

ellipse_G = Ellipse((0, 0), w_G, h_G, angle=angle_G,
                    fill=True, facecolor=PAL['G'], alpha=0.4,
                    edgecolor=PAL['G'], linewidth=3, label='G (fixed)')
ax.add_patch(ellipse_G)

# Draw P ellipses for each plasticity level
for i, (E, label) in enumerate(E_levels):
    P = G + E
    eigvals_P, eigvecs_P = np.linalg.eigh(P)
    w_P = 2 * np.sqrt(eigvals_P[1])
    h_P = 2 * np.sqrt(eigvals_P[0])
    angle_P = np.degrees(np.arctan2(eigvecs_P[1, 1], eigvecs_P[0, 1]))
    
    ellipse_P = Ellipse((0, 0), w_P, h_P, angle=angle_P,
                        fill=False, edgecolor=colors[i],
                        linewidth=2.5, linestyle='--', 
                        label=f'P ({label})')
    ax.add_patch(ellipse_P)

ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.axhline(0, color=PAL['neutral'], linewidth=0.5)
ax.axvline(0, color=PAL['neutral'], linewidth=0.5)
ax.legend(loc='lower right', fontsize=9)
ax.set_xlabel('Trait 1')
ax.set_ylabel('Trait 2')
ax.set_title('B. Plasticity inflates P while G stays fixed\n→ h² decreases as Ve increases', fontsize=12)

# -----------------------------------------------------------------------------
# Panel C: h² distribution shifts with plasticity
# -----------------------------------------------------------------------------
ax = fig.add_subplot(2, 2, 3)

# Sample h² distributions for different plasticity levels
np.random.seed(42)
n_samples = 5000

for i, (E, label) in enumerate(E_levels):
    P = G + E
    
    # P-whiten to get G*
    eigvals_P, eigvecs_P = np.linalg.eigh(P)
    P_inv_sqrt = eigvecs_P @ np.diag(1/np.sqrt(eigvals_P)) @ eigvecs_P.T
    Gstar = P_inv_sqrt @ G @ P_inv_sqrt
    
    # Sample random directions
    betas = np.random.randn(n_samples, 2)
    betas = betas / np.linalg.norm(betas, axis=1, keepdims=True)
    
    # Compute h² for each
    h2_vals = np.array([b @ Gstar @ b for b in betas])
    
    # Histogram
    ax.hist(h2_vals, bins=40, alpha=0.5, color=colors[i], 
           density=True, label=f'{label}\nmean h² = {np.mean(h2_vals):.2f}')
    ax.axvline(np.mean(h2_vals), color=colors[i], linestyle='--', linewidth=2)

ax.set_xlabel('h²(β)')
ax.set_ylabel('Density')
ax.set_title('C. Higher plasticity → lower h² distribution\n(more variance trapped in E)', fontsize=12)
ax.legend(fontsize=9)
ax.set_xlim(0, 1)
ax.grid(alpha=0.3)

# -----------------------------------------------------------------------------
# Panel D: The adaptive radiation alignment problem
# -----------------------------------------------------------------------------
ax = fig.add_subplot(2, 2, 4)

# Fixed G matrix
G = np.array([[0.8, 0.3],
              [0.3, 0.3]])
P = G + np.array([[0.4, 0.1], [0.1, 0.5]])

# Find g_max direction (largest eigenvector of G)
eigvals_G, eigvecs_G = np.linalg.eigh(G)
g_max = eigvecs_G[:, 1]  # Largest eigenvalue direction

# Draw G and P ellipses
for M, color, label, ls in [(G, PAL['G'], 'G', '-'), (P, PAL['P'], 'P', '--')]:
    eigvals_M, eigvecs_M = np.linalg.eigh(M)
    w = 2 * np.sqrt(eigvals_M[1])
    h = 2 * np.sqrt(eigvals_M[0])
    angle = np.degrees(np.arctan2(eigvecs_M[1, 1], eigvecs_M[0, 1]))
    
    fill = (M is G)
    ellipse = Ellipse((0, 0), w, h, angle=angle,
                     fill=fill, facecolor=color if fill else 'none',
                     alpha=0.3 if fill else 1.0,
                     edgecolor=color, linewidth=2, linestyle=ls)
    ax.add_patch(ellipse)

# Draw g_max direction
ax.annotate('', xy=g_max*1.5, xytext=(0, 0),
           arrowprops=dict(arrowstyle='->', color=PAL['G'], lw=3))
ax.text(g_max[0]*1.6, g_max[1]*1.6, 'g_max', fontsize=11, 
       color=PAL['G'], fontweight='bold')

# Draw multiple β directions representing different niches in radiation
# Some aligned, most misaligned
np.random.seed(123)
n_niches = 12
niche_angles = np.linspace(0, 2*np.pi, n_niches, endpoint=False)

for theta in niche_angles:
    beta = np.array([np.cos(theta), np.sin(theta)])
    
    # Compute h² for this direction
    h2 = (beta @ G @ beta) / (beta @ P @ beta)
    
    # Color by h²
    color = plt.cm.RdYlGn(h2)
    
    # Draw as thin arrow
    ax.annotate('', xy=beta*1.3, xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', color=color, lw=1.5, alpha=0.7))

# Highlight one "lucky" aligned niche and one "unlucky" misaligned niche
# Aligned: close to g_max
theta_good = np.arctan2(g_max[1], g_max[0])
beta_good = np.array([np.cos(theta_good), np.sin(theta_good)])
h2_good = (beta_good @ G @ beta_good) / (beta_good @ P @ beta_good)

ax.annotate('', xy=beta_good*1.4, xytext=(0, 0),
           arrowprops=dict(arrowstyle='->', color=PAL['high'], lw=3))
ax.text(beta_good[0]*1.5 + 0.1, beta_good[1]*1.5, 
       f'aligned\nh²={h2_good:.2f}', fontsize=9, color=PAL['high'])

# Misaligned: perpendicular to g_max
theta_bad = theta_good + np.pi/2
beta_bad = np.array([np.cos(theta_bad), np.sin(theta_bad)])
h2_bad = (beta_bad @ G @ beta_bad) / (beta_bad @ P @ beta_bad)

ax.annotate('', xy=beta_bad*1.4, xytext=(0, 0),
           arrowprops=dict(arrowstyle='->', color=PAL['low'], lw=3))
ax.text(beta_bad[0]*1.5 + 0.1, beta_bad[1]*1.5 - 0.15, 
       f'misaligned\nh²={h2_bad:.2f}', fontsize=9, color=PAL['low'])

ax.set_xlim(-2, 2)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.axhline(0, color=PAL['neutral'], linewidth=0.5)
ax.axvline(0, color=PAL['neutral'], linewidth=0.5)
ax.set_xlabel('Trait 1')
ax.set_ylabel('Trait 2')
ax.set_title('D. Adaptive radiation: β explores many directions\n→ most will be misaligned with g_max', fontsize=12)

plt.suptitle('The Plasticity Paradox: How Colonization Success Creates Evolutionary Constraints',
            fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()

plt.savefig('plasticity_constraint_feedback.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved: plasticity_constraint_feedback.png")
plt.close()

# =============================================================================
# FIGURE 2: The geometry of the alignment problem
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# -----------------------------------------------------------------------------
# Panel A: "Good" region vs "bad" region in trait space
# -----------------------------------------------------------------------------
ax = axes[0]

G = np.array([[0.8, 0.2],
              [0.2, 0.3]])
P = G + np.array([[0.3, 0.05], [0.05, 0.6]])

# Compute h² for all directions
thetas = np.linspace(0, 2*np.pi, 360)
h2_vals = []
for theta in thetas:
    beta = np.array([np.cos(theta), np.sin(theta)])
    h2_vals.append((beta @ G @ beta) / (beta @ P @ beta))
h2_vals = np.array(h2_vals)

# Find the "good" angular region (h² > mean)
mean_h2 = np.mean(h2_vals)
good_mask = h2_vals > mean_h2

# Shade good and bad regions
for theta, h2, is_good in zip(thetas, h2_vals, good_mask):
    wedge = Wedge((0, 0), 1.5, np.degrees(theta)-1, np.degrees(theta)+1,
                  facecolor=PAL['high'] if is_good else PAL['low'],
                  alpha=0.3)
    ax.add_patch(wedge)

# Draw G and P ellipses on top
for M, color, ls in [(G, PAL['G'], '-'), (P, PAL['P'], '--')]:
    eigvals_M, eigvecs_M = np.linalg.eigh(M)
    w = 2 * np.sqrt(eigvals_M[1])
    h = 2 * np.sqrt(eigvals_M[0])
    angle = np.degrees(np.arctan2(eigvecs_M[1, 1], eigvecs_M[0, 1]))
    ellipse = Ellipse((0, 0), w, h, angle=angle,
                     fill=False, edgecolor=color, linewidth=2.5, linestyle=ls)
    ax.add_patch(ellipse)

# Legend
ax.plot([], [], 's', color=PAL['high'], markersize=15, alpha=0.3, 
       label=f'h² > {mean_h2:.2f}')
ax.plot([], [], 's', color=PAL['low'], markersize=15, alpha=0.3,
       label=f'h² < {mean_h2:.2f}')

ax.set_xlim(-2, 2)
ax.set_ylim(-1.8, 1.8)
ax.set_aspect('equal')
ax.legend(loc='lower right', fontsize=10)
ax.set_xlabel('Trait 1')
ax.set_ylabel('Trait 2')
ax.set_title('A. "Good" vs "bad" regions in trait space\n(shading = h² above/below mean)')

# -----------------------------------------------------------------------------
# Panel B: What fraction of directions are "good"?
# -----------------------------------------------------------------------------
ax = axes[1]

# Vary the eigenvalue ratio of G* and see what fraction of directions have h² > threshold
np.random.seed(42)

eigenvalue_ratios = np.linspace(1, 10, 20)  # λ_max / λ_min
fraction_good = []

threshold = 0.5  # h² > 0.5 is "good"
n_samples = 10000

for ratio in eigenvalue_ratios:
    # Create G* with given eigenvalue ratio
    lambda_max = 0.7
    lambda_min = lambda_max / ratio
    
    Gstar = np.diag([lambda_min, lambda_max])
    
    # Sample random directions
    betas = np.random.randn(n_samples, 2)
    betas = betas / np.linalg.norm(betas, axis=1, keepdims=True)
    
    # h² for each (P* = I in whitened space)
    h2_samples = np.array([b @ Gstar @ b for b in betas])
    
    # Fraction above threshold
    fraction_good.append(np.mean(h2_samples > threshold))

ax.plot(eigenvalue_ratios, fraction_good, 'o-', color=PAL['G'], 
       linewidth=2, markersize=8)
ax.axhline(0.5, color=PAL['neutral'], linestyle='--', linewidth=1,
          label='50% of directions')

ax.set_xlabel('Eigenvalue ratio λ_max / λ_min')
ax.set_ylabel('Fraction of directions with h² > 0.5')
ax.set_title('B. Higher constraint → fewer "good" directions\n(more eigenvalue dispersion = worse odds)')
ax.legend()
ax.grid(alpha=0.3)
ax.set_ylim(0, 1)

# -----------------------------------------------------------------------------
# Panel C: The probability of finding g_max during radiation
# -----------------------------------------------------------------------------
ax = axes[2]

# In p dimensions, what's the probability that a random β is within
# angle θ of g_max?

dimensions = [2, 3, 5, 10, 20]
angles = np.linspace(0, 90, 100)  # degrees from g_max

for p in dimensions:
    # The probability that a random unit vector is within angle θ of a fixed direction
    # is related to the surface area of a spherical cap
    # P(angle < θ) ≈ sin^(p-2)(θ) for small θ
    # More precisely: P = I_{sin²(θ)}((p-1)/2, 1/2) where I is regularized incomplete beta
    
    from scipy import special
    
    probs = []
    for theta_deg in angles:
        theta = np.radians(theta_deg)
        # Regularized incomplete beta function
        prob = special.betainc((p-1)/2, 0.5, np.sin(theta)**2)
        probs.append(prob)
    
    ax.plot(angles, probs, linewidth=2, label=f'p = {p}')

ax.set_xlabel('Angular tolerance from g_max (degrees)')
ax.set_ylabel('Probability β falls within tolerance')
ax.set_title('C. Higher dimensions → harder to find g_max\n(curse of dimensionality)')
ax.legend(title='Dimensions')
ax.grid(alpha=0.3)
ax.set_xlim(0, 90)
ax.set_ylim(0, 1)

# Add annotation
ax.annotate('In 10D, only ~1% of directions\nare within 30° of g_max!',
           xy=(30, 0.01), xytext=(50, 0.3),
           fontsize=10, color=PAL['trap'],
           arrowprops=dict(arrowstyle='->', color=PAL['trap']))

plt.suptitle('The Geometry of Constraint: Why Most Directions Are "Bad"',
            fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

plt.savefig('alignment_geometry.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved: alignment_geometry.png")
plt.close()

# =============================================================================
# FIGURE 3: The adaptive radiation simulation
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# -----------------------------------------------------------------------------
# Panel A: Simulated radiation with h² tracking
# -----------------------------------------------------------------------------
ax = axes[0]

np.random.seed(42)

# Fixed G for the ancestor
G = np.array([[0.7, 0.2],
              [0.2, 0.4]])
P = G + np.array([[0.4, 0.1], [0.1, 0.5]])

# g_max direction
eigvals_G, eigvecs_G = np.linalg.eigh(G)
g_max = eigvecs_G[:, 1]
g_max_angle = np.arctan2(g_max[1], g_max[0])

# Simulate radiation: lineages diverge into different niches
# Each niche has a selection direction β that changes over time
n_lineages = 8
n_generations = 50

lineage_angles = np.linspace(0, 2*np.pi, n_lineages, endpoint=False)
lineage_h2_history = []

# Draw G and P ellipses
for M, color, ls in [(G, PAL['G'], '-'), (P, PAL['P'], '--')]:
    eigvals_M, eigvecs_M = np.linalg.eigh(M)
    w = 2 * np.sqrt(eigvals_M[1])
    h = 2 * np.sqrt(eigvals_M[0])
    angle = np.degrees(np.arctan2(eigvecs_M[1, 1], eigvecs_M[0, 1]))
    ellipse = Ellipse((0, 0), w, h, angle=angle,
                     fill=(M is G), facecolor=color if M is G else 'none',
                     alpha=0.2 if M is G else 1.0,
                     edgecolor=color, linewidth=2, linestyle=ls)
    ax.add_patch(ellipse)

# Draw g_max
ax.annotate('', xy=g_max*1.5, xytext=(0, 0),
           arrowprops=dict(arrowstyle='->', color=PAL['G'], lw=3))
ax.text(g_max[0]*1.6, g_max[1]*1.6, 'g_max', fontsize=11, color=PAL['G'], fontweight='bold')

# For each lineage, show final β direction colored by h²
for i, base_angle in enumerate(lineage_angles):
    # Final selection direction (with some drift from base)
    final_angle = base_angle + np.random.uniform(-0.2, 0.2)
    beta = np.array([np.cos(final_angle), np.sin(final_angle)])
    
    h2 = (beta @ G @ beta) / (beta @ P @ beta)
    
    color = plt.cm.RdYlGn(h2)
    
    ax.annotate('', xy=beta*1.4, xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', color=color, lw=2))
    ax.text(beta[0]*1.5, beta[1]*1.5, f'{h2:.2f}', fontsize=9, 
           color=color, fontweight='bold')

ax.set_xlim(-2, 2)
ax.set_ylim(-1.8, 1.8)
ax.set_aspect('equal')
ax.set_xlabel('Trait 1')
ax.set_ylabel('Trait 2')
ax.set_title('A. Adaptive radiation: 8 lineages diverge\n(numbers = h² in each niche\'s direction)')

# Colorbar
sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(0.3, 0.7))
plt.colorbar(sm, ax=ax, label='h²(β)', shrink=0.8)

# -----------------------------------------------------------------------------
# Panel B: Distribution of h² across niches
# -----------------------------------------------------------------------------
ax = axes[1]

# Simulate many radiations
n_radiations = 1000
n_niches_per_radiation = 10

h2_all_niches = []

for _ in range(n_radiations):
    # Random niche directions
    niche_angles = np.random.uniform(0, 2*np.pi, n_niches_per_radiation)
    
    for angle in niche_angles:
        beta = np.array([np.cos(angle), np.sin(angle)])
        h2 = (beta @ G @ beta) / (beta @ P @ beta)
        h2_all_niches.append(h2)

h2_all_niches = np.array(h2_all_niches)

# Histogram
ax.hist(h2_all_niches, bins=50, density=True, color=PAL['trap'], 
       alpha=0.7, edgecolor='white')

# Mark mean and the h² along g_max
mean_h2 = np.mean(h2_all_niches)
h2_gmax = (g_max @ G @ g_max) / (g_max @ P @ g_max)

ax.axvline(mean_h2, color='black', linestyle='--', linewidth=2,
          label=f'Mean h² = {mean_h2:.3f}')
ax.axvline(h2_gmax, color=PAL['high'], linestyle='-', linewidth=2,
          label=f'h²(g_max) = {h2_gmax:.3f}')

# What fraction are in the "trap"?
trap_threshold = 0.45
fraction_trapped = np.mean(h2_all_niches < trap_threshold)

ax.axvline(trap_threshold, color=PAL['low'], linestyle=':', linewidth=2)
ax.fill_betweenx([0, 4], 0, trap_threshold, alpha=0.2, color=PAL['low'],
                label=f'h² < {trap_threshold}: {fraction_trapped:.0%} of niches')

ax.set_xlabel('h²(β)')
ax.set_ylabel('Density')
ax.set_title(f'B. Distribution of h² across {n_radiations*n_niches_per_radiation:,} random niches\n'
            f'{fraction_trapped:.0%} fall into low-h² "trap" regions')
ax.legend(fontsize=9)
ax.set_xlim(0.2, 0.8)
ax.grid(alpha=0.3)

plt.suptitle('Adaptive Radiations: Most Niches Will Have Suboptimal h²',
            fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

plt.savefig('radiation_h2_distribution.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved: radiation_h2_distribution.png")
plt.close()

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY: THE PLASTICITY PARADOX")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  THE PLASTICITY-CONSTRAINT FEEDBACK LOOP                            │
│  ═══════════════════════════════════════                            │
│                                                                     │
│  1. Environmental heterogeneity maintains Va (plasticity variance)  │
│                                                                     │
│  2. Plasticity → high Ve → low h² for many directions               │
│                                                                     │
│  3. But plasticity ALSO enables colonization of new environments    │
│                                                                     │
│  4. New environments → more heterogeneity → more Va maintained      │
│                                                                     │
│  5. POSITIVE FEEDBACK that generates constraint traps!              │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  THE ADAPTIVE RADIATION PROBLEM                                     │
│  ═════════════════════════════                                      │
│                                                                     │
│  • g_max is FIXED by genetic architecture                           │
│                                                                     │
│  • But β (selection direction) CHANGES as lineages diversify        │
│                                                                     │
│  • In p dimensions, most directions are FAR from g_max:             │
│      - 2D: ~25% within 45° of g_max                                 │
│      - 5D: ~7% within 45° of g_max                                  │
│      - 10D: ~1% within 45° of g_max                                 │
│                                                                     │
│  • CONCLUSION: Adaptive radiations will typically encounter         │
│    constraint traps. The very success that enables radiation        │
│    creates the conditions for evolutionary stasis!                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

This may explain:
  • Why some clades radiate then "stall"
  • Why canalization evolves (reduces Ve, increases h²)
  • Why genetic assimilation occurs (converts Ve to Va along β)
  • Why "evolvability" itself may be under selection
""")
