#!/usr/bin/env python3
"""
Figures for Chapter 41: Directional Heritability and the Geometry of Constraint
================================================================================

Generates:
- fig41_cv_formula.png: The CV formula components
- fig41_constraint_severity.png: Constraint severity visualization
- fig41_h2_distribution.png: Distribution of directional heritability

Author: Daniel Ortiz-Barrientos
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from scipy import stats

# Style settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Color palette
BLUE = '#2E86AB'
GREEN = '#06D6A0'
ORANGE = '#F18F01'
RED = '#E63946'
PURPLE = '#762A83'
GRAY = '#666666'
LIGHT_BLUE = '#a8d4e6'
LIGHT_GREEN = '#a8f0d4'
TEAL = '#20B2AA'

# =============================================================================
# Figure 1: CV Formula Components
# =============================================================================

print("Creating fig41_cv_formula.png...")

fig = plt.figure(figsize=(14, 5))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.2])

# Left panel: Effect of eigenvalue dispersion
ax1 = fig.add_subplot(gs[0])

# Two scenarios: low vs high dispersion
eigenvalues_low = [0.6, 0.55, 0.5, 0.45]
eigenvalues_high = [0.9, 0.6, 0.4, 0.2]

x = np.arange(1, 5)
width = 0.35

bars1 = ax1.bar(x - width/2, eigenvalues_low, width, label='Low dispersion', 
                color=BLUE, alpha=0.7, edgecolor='black')
bars2 = ax1.bar(x + width/2, eigenvalues_high, width, label='High dispersion', 
                color=ORANGE, alpha=0.7, edgecolor='black')

ax1.axhline(np.mean(eigenvalues_low), color=BLUE, linestyle='--', linewidth=2, alpha=0.7)
ax1.axhline(np.mean(eigenvalues_high), color=ORANGE, linestyle='--', linewidth=2, alpha=0.7)

# CV values
cv_low = np.std(eigenvalues_low) / np.mean(eigenvalues_low)
cv_high = np.std(eigenvalues_high) / np.mean(eigenvalues_high)

ax1.text(0.95, 0.95, f'CV(λ*) = {cv_low:.2f}', transform=ax1.transAxes, 
         fontsize=10, color=BLUE, ha='right', va='top', fontweight='bold')
ax1.text(0.95, 0.85, f'CV(λ*) = {cv_high:.2f}', transform=ax1.transAxes, 
         fontsize=10, color=ORANGE, ha='right', va='top', fontweight='bold')

ax1.set_xlabel('Principal Axis')
ax1.set_ylabel('Directional Heritability (λ*)')
ax1.set_xticks(x)
ax1.set_xticklabels(['1', '2', '3', '4'])
ax1.set_ylim(0, 1)
ax1.legend(loc='lower left', fontsize=9)
ax1.set_title('Factor 1: Eigenvalue Dispersion\nCV(λ*) measures G* eccentricity', 
              fontsize=11, fontweight='bold')

# Middle panel: Effect of dimensionality
ax2 = fig.add_subplot(gs[1])

p_values = np.arange(2, 21)
scaling_factor = np.sqrt(2 / (p_values + 2))

ax2.plot(p_values, scaling_factor, 'o-', color=PURPLE, linewidth=2, markersize=6)
ax2.fill_between(p_values, 0, scaling_factor, color=PURPLE, alpha=0.2)

# Mark specific values
for p in [2, 5, 10, 20]:
    sf = np.sqrt(2 / (p + 2))
    ax2.scatter([p], [sf], s=100, c=PURPLE, zorder=5, edgecolor='white', linewidth=2)
    ax2.annotate(f'p={p}\n{sf:.2f}', xy=(p, sf), xytext=(p+1, sf+0.05),
                fontsize=9, ha='left')

ax2.set_xlabel('Number of Traits (p)')
ax2.set_ylabel('Scaling Factor √(2/(p+2))')
ax2.set_ylim(0, 0.85)
ax2.set_title('Factor 2: Dimensionality\nMore traits → more averaging', 
              fontsize=11, fontweight='bold')

# Right panel: The formula
ax3 = fig.add_subplot(gs[2])
ax3.axis('off')

formula_text = """THE CV FORMULA

CV(h²) = √(2/(p+2)) × CV(λ*)

Or equivalently:

CV²(h²) = [2/(p+2)] × V_rel(G*)

where V_rel = Var(λ*) / mean(λ*)²


INTERPRETATION:

• High CV(λ*) → G* is eccentric
  → heritability varies greatly by direction

• Low CV(λ*) → G* is spherical
  → heritability similar in all directions

• Large p → averaging effect
  → random directions sample many axes
  → CV(h²) reduced


EXAMPLE:

If CV(λ*) = 0.5 and p = 4:
  CV(h²) = √(2/6) × 0.5 = 0.41 × 0.5 = 0.20

Heritability varies by ~20% (CV) across directions
"""

ax3.text(0.05, 0.95, formula_text, transform=ax3.transAxes, fontsize=10,
         fontfamily='serif', va='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95))

ax3.set_title('The Master Formula', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('fig41_cv_formula.png', dpi=300, facecolor='white')
print("  Saved: fig41_cv_formula.png")
plt.close()

# =============================================================================
# Figure 2: Constraint Severity
# =============================================================================

print("Creating fig41_constraint_severity.png...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

# Left panel: G* ellipse showing severity
ax = axes[0]

# G* in whitened space (P = unit circle)
h2_max = 0.75
h2_min = 0.30
h2_mean = (h2_max + h2_min) / 2  # Simplified for 2D

# P-sphere (unit circle)
theta = np.linspace(0, 2*np.pi, 100)
ax.plot(np.cos(theta), np.sin(theta), '-', color=BLUE, linewidth=2.5, label='P-sphere')
ax.fill(np.cos(theta), np.sin(theta), color=LIGHT_BLUE, alpha=0.15)

# G* ellipse
rotation = 25
theta_rad = np.radians(rotation)
ellipse_Gstar = Ellipse((0, 0), 2*np.sqrt(h2_max), 2*np.sqrt(h2_min),
                        angle=rotation, fill=True, facecolor=LIGHT_GREEN, alpha=0.4,
                        edgecolor=GREEN, linewidth=2.5, label='G* ellipse')
ax.add_patch(ellipse_Gstar)

# Mark max and min heritability directions
R = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
              [np.sin(theta_rad), np.cos(theta_rad)]])

h2_max_dir = R @ np.array([1, 0])
h2_min_dir = R @ np.array([0, 1])

# Max h² direction
ax.annotate('', xy=h2_max_dir*1.2, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color=ORANGE, lw=2.5))
ax.text(h2_max_dir[0]*1.3 + 0.1, h2_max_dir[1]*1.3, 
        f'$h^2_{{max}} = {h2_max:.2f}$', fontsize=10, color=ORANGE, fontweight='bold')

# Min h² direction (constraint trap)
ax.annotate('', xy=h2_min_dir*1.2, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color=RED, lw=2.5))
ax.text(h2_min_dir[0]*1.3 - 0.35, h2_min_dir[1]*1.3, 
        f'$h^2_{{min}} = {h2_min:.2f}$\nCONSTRAINT', fontsize=10, color=RED, 
        fontweight='bold', ha='center')

# Show the gap = severity
gap_point_on_circle = h2_min_dir * 1.0
gap_point_on_ellipse = h2_min_dir * np.sqrt(h2_min)
ax.annotate('', xy=gap_point_on_circle, xytext=gap_point_on_ellipse,
            arrowprops=dict(arrowstyle='<->', color=PURPLE, lw=2))

severity = 1 - h2_min / h2_mean
ax.text(-0.5, 0.75, f'Severity = {severity:.2f}', fontsize=10, color=PURPLE,
        fontweight='bold')

ax.axhline(0, color='lightgray', linewidth=0.5, zorder=0)
ax.axvline(0, color='lightgray', linewidth=0.5, zorder=0)
ax.scatter([0], [0], s=60, c='black', marker='+', zorder=4, linewidths=2)

ax.set_xlim(-1.8, 1.8)
ax.set_ylim(-1.5, 1.8)
ax.set_aspect('equal')
ax.set_xlabel('Whitened trait 1')
ax.set_ylabel('Whitened trait 2')
ax.set_title('Constraint Severity in G* Geometry', fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)

# Right panel: Severity across scenarios
ax = axes[1]

# Different scenarios
scenarios = [
    ('Low constraint', [0.60, 0.55, 0.50, 0.45]),
    ('Moderate', [0.70, 0.55, 0.45, 0.30]),
    ('High constraint', [0.85, 0.60, 0.35, 0.15]),
    ('Severe trap', [0.90, 0.70, 0.30, 0.05]),
]

x_pos = np.arange(len(scenarios))
severities = []
ranges = []

for name, eigenvalues in scenarios:
    mean_h2 = np.mean(eigenvalues)
    min_h2 = min(eigenvalues)
    sev = 1 - min_h2 / mean_h2
    severities.append(sev)
    ranges.append(max(eigenvalues) - min(eigenvalues))

colors_sev = [GREEN, TEAL, ORANGE, RED]
bars = ax.bar(x_pos, severities, color=colors_sev, alpha=0.7, edgecolor='black')

for i, (s, name_tuple) in enumerate(zip(severities, scenarios)):
    ax.text(i, s + 0.02, f'{s:.2f}', ha='center', fontsize=10, fontweight='bold')

ax.set_xticks(x_pos)
ax.set_xticklabels([s[0] for s in scenarios], fontsize=9)
ax.set_ylabel('Constraint Severity')
ax.set_ylim(0, 1)
ax.set_title('Severity Across Different G* Shapes', fontsize=12, fontweight='bold')

# Annotation
ax.text(0.98, 0.98, 
        'Severity = 1 - (min h²) / (mean h²)\n\n'
        '0 = all directions equal\n'
        '1 = worst direction has h² ≈ 0',
        transform=ax.transAxes, fontsize=9, ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()
plt.savefig('fig41_constraint_severity.png', dpi=300, facecolor='white')
print("  Saved: fig41_constraint_severity.png")
plt.close()

# =============================================================================
# Figure 3: Distribution of Directional Heritability
# =============================================================================

print("Creating fig41_h2_distribution.png...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

# Left panel: Simulated h² distribution
ax = axes[0]

# Simulate sampling from P-sphere and computing h²
np.random.seed(42)

# G* eigenvalues
lambda_star = np.array([0.78, 0.62, 0.52, 0.41])
p = len(lambda_star)

# Sample many random directions
n_samples = 10000
h2_samples = []

for _ in range(n_samples):
    # Random unit vector (uniform on sphere)
    direction = np.random.randn(p)
    direction = direction / np.linalg.norm(direction)
    
    # h² = sum of lambda_i * (direction_i)^2
    # In the eigenbasis of G*, this is simple
    h2 = np.sum(lambda_star * direction**2)
    h2_samples.append(h2)

h2_samples = np.array(h2_samples)

# Histogram
ax.hist(h2_samples, bins=50, density=True, color=LIGHT_BLUE, edgecolor=BLUE,
        alpha=0.7, label='Simulated')

# Mark eigenvalues
for i, (lam, color) in enumerate(zip(lambda_star, [GREEN, TEAL, BLUE, PURPLE])):
    ax.axvline(lam, color=color, linewidth=2, linestyle='--', alpha=0.8)
    ax.text(lam, ax.get_ylim()[1]*0.9 - i*0.8, f'$\\lambda_{i+1}^* = {lam:.2f}$',
            fontsize=9, color=color, ha='center')

# Mark mean
ax.axvline(np.mean(h2_samples), color=ORANGE, linewidth=3, label=f'Mean = {np.mean(h2_samples):.2f}')

# Theoretical stats
cv_observed = np.std(h2_samples) / np.mean(h2_samples)
cv_lambda = np.std(lambda_star) / np.mean(lambda_star)
cv_theoretical = np.sqrt(2/(p+2)) * cv_lambda

ax.set_xlabel('Directional Heritability h²(β)')
ax.set_ylabel('Density')
ax.set_title('Distribution of h² Across Random Directions\n(Uniform sampling from P-sphere)',
             fontsize=11, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)

# Stats box
stats_text = (f'Observed CV(h²) = {cv_observed:.3f}\n'
              f'Predicted CV(h²) = {cv_theoretical:.3f}\n'
              f'CV(λ*) = {cv_lambda:.3f}')
ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
        ha='right', va='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

# Right panel: Effect of trait number
ax = axes[1]

# Different numbers of traits
p_values = [2, 4, 8, 16]
colors_p = [ORANGE, GREEN, BLUE, PURPLE]

for p_val, color in zip(p_values, colors_p):
    # Create eigenvalues that span from 0.3 to 0.8
    lambda_star_p = np.linspace(0.8, 0.3, p_val)
    
    # Simulate
    h2_samples_p = []
    for _ in range(5000):
        direction = np.random.randn(p_val)
        direction = direction / np.linalg.norm(direction)
        h2 = np.sum(lambda_star_p * direction**2)
        h2_samples_p.append(h2)
    
    h2_samples_p = np.array(h2_samples_p)
    
    # KDE for smooth curve
    kde = stats.gaussian_kde(h2_samples_p)
    x_range = np.linspace(0.3, 0.8, 100)
    ax.plot(x_range, kde(x_range), '-', color=color, linewidth=2.5,
            label=f'p = {p_val}, CV = {np.std(h2_samples_p)/np.mean(h2_samples_p):.2f}')
    ax.fill_between(x_range, 0, kde(x_range), color=color, alpha=0.1)

ax.set_xlabel('Directional Heritability h²(β)')
ax.set_ylabel('Density')
ax.set_title('Effect of Dimensionality\nMore traits → narrower distribution',
             fontsize=11, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)

# Annotation
ax.text(0.98, 0.98, 
        'Same eigenvalue range (0.3 to 0.8)\n'
        'but more traits → more averaging\n'
        '→ h² concentrates near mean',
        transform=ax.transAxes, fontsize=9, ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()
plt.savefig('fig41_h2_distribution.png', dpi=300, facecolor='white')
print("  Saved: fig41_h2_distribution.png")
plt.close()

# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*60)
print("Chapter 41 Figures Complete")
print("="*60)
print("""
Generated figures:
  1. fig41_cv_formula.png
     - Left: Effect of eigenvalue dispersion on CV
     - Middle: Effect of dimensionality (scaling factor)
     - Right: The master formula explained
     
  2. fig41_constraint_severity.png
     - Left: G* geometry showing severity measure
     - Right: Severity across different scenarios
     
  3. fig41_h2_distribution.png
     - Left: Simulated h² distribution with eigenvalues marked
     - Right: Effect of dimensionality on distribution width
""")
