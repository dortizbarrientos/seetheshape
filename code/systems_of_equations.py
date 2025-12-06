#!/usr/bin/env python3
"""
Systems of Linear Equations: The Classic View Connected to Geometry
====================================================================

A system of linear equations can be viewed THREE ways:
1. ROW PICTURE: Each equation is a line/plane; solution is intersection
2. COLUMN PICTURE: Solution gives coefficients for linear combination
3. TRANSFORMATION PICTURE: Solution is pre-image under matrix transformation

All three give the same answer, but reveal different geometric insights.

Author: Daniel Ortiz-Barrientos
Date: December 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, FancyArrowPatch, Polygon
from matplotlib.collections import LineCollection
import matplotlib.patheffects as pe

# =============================================================================
# COLOR PALETTE
# =============================================================================
PAL = {
    'eq1': '#2E86AB',      # First equation
    'eq2': '#E63946',      # Second equation
    'eq3': '#1B7837',      # Third equation
    'solution': '#F18F01', # Solution point
    'col1': '#2E86AB',     # Column 1
    'col2': '#E63946',     # Column 2
    'target': '#F18F01',   # Target vector b
    'transform': '#762A83',
    'neutral': '#878787',
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
# FIGURE 1: THE ROW PICTURE
# =============================================================================

print("Creating Figure 1: The Row Picture...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Define a simple 2x2 system:
# 2x + y = 5
# x - y = 1
# Solution: x = 2, y = 1

A = np.array([[2, 1],
              [1, -1]])
b = np.array([5, 1])
x_sol = np.linalg.solve(A, b)

# -----------------------------------------------------------------------------
# Panel A: The row picture in 2D
# -----------------------------------------------------------------------------
ax = axes[0, 0]

x_range = np.linspace(-1, 4, 100)

# Equation 1: 2x + y = 5  =>  y = 5 - 2x
y1 = 5 - 2*x_range
ax.plot(x_range, y1, color=PAL['eq1'], linewidth=2.5, label='2x + y = 5')

# Equation 2: x - y = 1  =>  y = x - 1
y2 = x_range - 1
ax.plot(x_range, y2, color=PAL['eq2'], linewidth=2.5, label='x - y = 1')

# Solution point
ax.plot(x_sol[0], x_sol[1], 'o', color=PAL['solution'], markersize=15,
       markeredgecolor='white', markeredgewidth=2, zorder=5)
ax.text(x_sol[0]+0.2, x_sol[1]+0.3, f'({x_sol[0]:.0f}, {x_sol[1]:.0f})', 
       fontsize=12, fontweight='bold', color=PAL['solution'])

ax.set_xlim(-1, 4)
ax.set_ylim(-2, 4)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(loc='upper right', fontsize=10)
ax.grid(alpha=0.3)
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_title('A. Row Picture (2D)\nEach equation = a LINE', fontsize=11)

# -----------------------------------------------------------------------------
# Panel B: The row picture explained
# -----------------------------------------------------------------------------
ax = axes[0, 1]
ax.axis('off')

text = """
THE ROW PICTURE

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

System of equations:
    2x + y = 5
    x - y = 1

Each equation defines a LINE:
  • All (x, y) satisfying that equation
  • The "row" of coefficients [2, 1] is 
    the normal vector to the line

THE SOLUTION:
  • The point where ALL lines intersect
  • Satisfies ALL equations simultaneously

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

In 3D: each equation = a PLANE
       solution = where planes intersect

In nD: each equation = a HYPERPLANE
       solution = intersection of all

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=10,
       ha='center', va='center', fontfamily='monospace',
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
ax.set_title('B. Row picture interpretation', fontsize=11)

# -----------------------------------------------------------------------------
# Panel C: The row picture in 3D
# -----------------------------------------------------------------------------
ax = axes[0, 2]

# 3D system:
# x + y + z = 6
# x - y + 2z = 5  
# 2x + y - z = 1
# Solution: x = 1, y = 2, z = 3

# Show the planes conceptually (simplified 2D sketch)
ax.fill([0, 3, 3, 0], [0, 0, 3, 3], color=PAL['eq1'], alpha=0.3, label='Plane 1')
ax.fill([0.5, 3.5, 2.5, -0.5], [0, 0.5, 3.5, 3], color=PAL['eq2'], alpha=0.3, label='Plane 2')
ax.fill([0, 2, 3, 1], [1, 0, 2, 3], color=PAL['eq3'], alpha=0.3, label='Plane 3')

# Intersection line (conceptual)
ax.plot([1, 2], [1.5, 2.5], 'k--', linewidth=2, label='Intersection')

# Solution point
ax.plot(1.5, 2, 'o', color=PAL['solution'], markersize=12,
       markeredgecolor='white', markeredgewidth=2, zorder=5)
ax.text(1.7, 2.2, 'Solution', fontsize=10, color=PAL['solution'], fontweight='bold')

ax.set_xlim(-0.5, 4)
ax.set_ylim(-0.5, 4)
ax.legend(loc='upper right', fontsize=9)
ax.set_title('C. Row Picture (3D sketch)\nThree planes intersect at a point', fontsize=11)
ax.axis('off')

# -----------------------------------------------------------------------------
# Panel D: When there's no solution
# -----------------------------------------------------------------------------
ax = axes[1, 0]

x_range = np.linspace(-1, 4, 100)

# Parallel lines (no solution):
# 2x + y = 5
# 2x + y = 3  (same slope, different intercept)

y1 = 5 - 2*x_range
y2 = 3 - 2*x_range

ax.plot(x_range, y1, color=PAL['eq1'], linewidth=2.5, label='2x + y = 5')
ax.plot(x_range, y2, color=PAL['eq2'], linewidth=2.5, label='2x + y = 3')

ax.text(2, 2.5, 'PARALLEL!\nNo intersection\nNo solution', fontsize=11, 
       color=PAL['neutral'], ha='center', fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_xlim(-1, 4)
ax.set_ylim(-2, 4)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(loc='upper right', fontsize=10)
ax.grid(alpha=0.3)
ax.set_title('D. No solution\n(parallel lines never meet)', fontsize=11)

# -----------------------------------------------------------------------------
# Panel E: Infinitely many solutions
# -----------------------------------------------------------------------------
ax = axes[1, 1]

x_range = np.linspace(-1, 4, 100)

# Same line (infinitely many solutions):
# 2x + y = 5
# 4x + 2y = 10  (same line!)

y1 = 5 - 2*x_range

ax.plot(x_range, y1, color=PAL['eq1'], linewidth=4, label='2x + y = 5')
ax.plot(x_range, y1, color=PAL['eq2'], linewidth=2, linestyle='--', 
       label='4x + 2y = 10 (same!)')

# Show multiple solutions
for x_val in [0.5, 1.5, 2.5]:
    y_val = 5 - 2*x_val
    ax.plot(x_val, y_val, 'o', color=PAL['solution'], markersize=10,
           markeredgecolor='white', markeredgewidth=2)

ax.text(2.5, 3, 'SAME LINE!\nInfinitely many\nsolutions', fontsize=11, 
       color=PAL['neutral'], ha='center', fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_xlim(-1, 4)
ax.set_ylim(-2, 5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(loc='upper right', fontsize=10)
ax.grid(alpha=0.3)
ax.set_title('E. Infinitely many solutions\n(lines are the same)', fontsize=11)

# -----------------------------------------------------------------------------
# Panel F: The determinant
# -----------------------------------------------------------------------------
ax = axes[1, 2]
ax.axis('off')

text = """
WHEN DOES A UNIQUE SOLUTION EXIST?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

det(A) ≠ 0  →  UNIQUE SOLUTION
  • Lines/planes intersect at ONE point
  • Matrix is "invertible"
  • x = A⁻¹b

det(A) = 0  →  NO UNIQUE SOLUTION
  • Lines/planes are parallel (no solution)
  • OR lines/planes are same (∞ solutions)
  • Matrix is "singular"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE GEOMETRIC MEANING:

det(A) = signed area/volume of the
parallelogram/parallelepiped formed
by the column vectors

det(A) = 0 means columns are PARALLEL
(the transformation "squashes" space)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=10,
       ha='center', va='center', fontfamily='monospace',
       bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))
ax.set_title('F. The determinant test', fontsize=11)

plt.suptitle('VIEW 1: THE ROW PICTURE — Each Equation is a Hyperplane',
            fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('row_picture.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved: row_picture.png")
plt.close()

# =============================================================================
# FIGURE 2: THE COLUMN PICTURE
# =============================================================================

print("Creating Figure 2: The Column Picture...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Same system: Ax = b
# [2  1] [x]   [5]
# [1 -1] [y] = [1]
#
# Column picture: x*[2,1] + y*[1,-1] = [5,1]

col1 = A[:, 0]  # [2, 1]
col2 = A[:, 1]  # [1, -1]

# -----------------------------------------------------------------------------
# Panel A: The column picture
# -----------------------------------------------------------------------------
ax = axes[0, 0]

# Draw column vectors
ax.annotate('', xy=col1, xytext=(0, 0),
           arrowprops=dict(arrowstyle='->', color=PAL['col1'], lw=3))
ax.text(col1[0]+0.1, col1[1]+0.2, 'a₁ = [2, 1]', fontsize=11, 
       color=PAL['col1'], fontweight='bold')

ax.annotate('', xy=col2, xytext=(0, 0),
           arrowprops=dict(arrowstyle='->', color=PAL['col2'], lw=3))
ax.text(col2[0]+0.1, col2[1]-0.3, 'a₂ = [1, -1]', fontsize=11, 
       color=PAL['col2'], fontweight='bold')

# Target vector b
ax.annotate('', xy=b, xytext=(0, 0),
           arrowprops=dict(arrowstyle='->', color=PAL['target'], lw=3))
ax.text(b[0]+0.1, b[1]+0.2, 'b = [5, 1]', fontsize=11, 
       color=PAL['target'], fontweight='bold')

ax.set_xlim(-1, 6)
ax.set_ylim(-2, 3)
ax.set_aspect('equal')
ax.grid(alpha=0.3)
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_title('A. The column vectors and target\na₁, a₂, and b', fontsize=11)

# -----------------------------------------------------------------------------
# Panel B: Linear combination that reaches b
# -----------------------------------------------------------------------------
ax = axes[0, 1]

# Solution: x = 2, y = 1
# So: 2*a₁ + 1*a₂ = b

# Draw scaled vectors
scaled_col1 = x_sol[0] * col1  # 2 * [2, 1] = [4, 2]
scaled_col2 = x_sol[1] * col2  # 1 * [1, -1] = [1, -1]

# First vector (2*a₁)
ax.annotate('', xy=scaled_col1, xytext=(0, 0),
           arrowprops=dict(arrowstyle='->', color=PAL['col1'], lw=3))
ax.text(scaled_col1[0]/2, scaled_col1[1]/2+0.3, '2·a₁', fontsize=11, 
       color=PAL['col1'], fontweight='bold')

# Second vector (1*a₂) starting from end of first
ax.annotate('', xy=scaled_col1 + scaled_col2, xytext=scaled_col1,
           arrowprops=dict(arrowstyle='->', color=PAL['col2'], lw=3))
ax.text(scaled_col1[0]+0.5, scaled_col1[1]-0.5, '1·a₂', fontsize=11, 
       color=PAL['col2'], fontweight='bold')

# Result = b
ax.annotate('', xy=b, xytext=(0, 0),
           arrowprops=dict(arrowstyle='->', color=PAL['target'], lw=2, 
                          linestyle='--'))
ax.plot(b[0], b[1], 'o', color=PAL['target'], markersize=12,
       markeredgecolor='white', markeredgewidth=2)
ax.text(b[0]+0.2, b[1]+0.2, 'b = 2·a₁ + 1·a₂', fontsize=10, 
       color=PAL['target'], fontweight='bold')

ax.set_xlim(-1, 6)
ax.set_ylim(-2, 3)
ax.set_aspect('equal')
ax.grid(alpha=0.3)
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_title('B. The solution as linear combination\n2·a₁ + 1·a₂ = b', fontsize=11)

# -----------------------------------------------------------------------------
# Panel C: The column picture explained
# -----------------------------------------------------------------------------
ax = axes[0, 2]
ax.axis('off')

text = """
THE COLUMN PICTURE

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Ax = b becomes:

x₁·(column 1) + x₂·(column 2) = b

"What LINEAR COMBINATION of columns
gives the target vector b?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For our system:

x·[2]  +  y·[1]  =  [5]
  [1]      [-1]     [1]

The solution x=2, y=1 means:

2·[2] + 1·[1]  = [4+1]  = [5]  ✓
  [1]    [-1]    [2-1]    [1]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The solution coefficients ARE the 
coordinates of b in the basis 
formed by the columns of A!
"""

ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=10,
       ha='center', va='center', fontfamily='monospace',
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
ax.set_title('C. Column picture interpretation', fontsize=11)

# -----------------------------------------------------------------------------
# Panel D: The column space
# -----------------------------------------------------------------------------
ax = axes[1, 0]

# Show all possible linear combinations (the column space)
# This is the entire 2D plane if columns are independent

# Draw a grid of linear combinations
alphas = np.linspace(-1, 3, 20)
betas = np.linspace(-2, 2, 20)

for alpha in alphas:
    for beta in betas:
        point = alpha * col1 + beta * col2
        ax.plot(point[0], point[1], '.', color=PAL['neutral'], 
               markersize=2, alpha=0.3)

# Highlight the columns
ax.annotate('', xy=col1*1.5, xytext=(0, 0),
           arrowprops=dict(arrowstyle='->', color=PAL['col1'], lw=2))
ax.annotate('', xy=col2*1.5, xytext=(0, 0),
           arrowprops=dict(arrowstyle='->', color=PAL['col2'], lw=2))

# Target
ax.plot(b[0], b[1], 'o', color=PAL['target'], markersize=12,
       markeredgecolor='white', markeredgewidth=2)
ax.text(b[0]+0.2, b[1]+0.2, 'b', fontsize=12, color=PAL['target'], fontweight='bold')

ax.text(0, -3, 'The COLUMN SPACE is all\npoints reachable by linear combinations', 
       ha='center', fontsize=10)

ax.set_xlim(-3, 7)
ax.set_ylim(-3.5, 4)
ax.set_aspect('equal')
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_title('D. Column space\n(all reachable points)', fontsize=11)

# -----------------------------------------------------------------------------
# Panel E: When columns are parallel (no solution for most b)
# -----------------------------------------------------------------------------
ax = axes[1, 1]

# Parallel columns
col1_parallel = np.array([2, 1])
col2_parallel = np.array([4, 2])  # = 2 * col1

# Show the line they span
t = np.linspace(-2, 3, 100)
line = np.outer(t, col1_parallel)
ax.plot(line[:, 0], line[:, 1], color=PAL['col1'], linewidth=3,
       label='Column space (just a line!)')

# Draw columns
ax.annotate('', xy=col1_parallel, xytext=(0, 0),
           arrowprops=dict(arrowstyle='->', color=PAL['col1'], lw=2))
ax.text(col1_parallel[0]+0.1, col1_parallel[1]+0.2, 'a₁', fontsize=11, color=PAL['col1'])

ax.annotate('', xy=col2_parallel, xytext=(0, 0),
           arrowprops=dict(arrowstyle='->', color=PAL['col2'], lw=2))
ax.text(col2_parallel[0]+0.1, col2_parallel[1]+0.2, 'a₂ = 2·a₁', fontsize=11, color=PAL['col2'])

# A target NOT on the line
b_unreachable = np.array([3, 0])
ax.plot(b_unreachable[0], b_unreachable[1], 'x', color=PAL['target'], 
       markersize=15, markeredgewidth=3)
ax.text(b_unreachable[0]+0.2, b_unreachable[1]+0.3, 'b (unreachable!)', 
       fontsize=10, color=PAL['target'], fontweight='bold')

ax.set_xlim(-3, 6)
ax.set_ylim(-2, 4)
ax.set_aspect('equal')
ax.legend(loc='lower right', fontsize=9)
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_title('E. Parallel columns (singular)\nColumn space is just a line!', fontsize=11)

# -----------------------------------------------------------------------------
# Panel F: Connection to the row picture
# -----------------------------------------------------------------------------
ax = axes[1, 2]
ax.axis('off')

text = """
ROW vs COLUMN: SAME ANSWER!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ROW PICTURE:
  "Where do the lines/planes meet?"
  Each equation constrains the solution

COLUMN PICTURE:
  "What combination of columns gives b?"
  Solution = coordinates in column basis

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THEY GIVE THE SAME SOLUTION!

The duality:
  • Rows define hyperplanes to intersect
  • Columns define vectors to combine

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SINGULAR MATRIX (det = 0):
  Row view:  parallel hyperplanes
  Column view: parallel columns
             → column space is "thin"
             → most b are unreachable
"""

ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=10,
       ha='center', va='center', fontfamily='monospace',
       bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))
ax.set_title('F. Two views, same answer', fontsize=11)

plt.suptitle('VIEW 2: THE COLUMN PICTURE — Find the Linear Combination',
            fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('column_picture.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved: column_picture.png")
plt.close()

# =============================================================================
# FIGURE 3: THE TRANSFORMATION PICTURE
# =============================================================================

print("Creating Figure 3: The Transformation Picture...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# -----------------------------------------------------------------------------
# Panel A: A transforms the whole space
# -----------------------------------------------------------------------------
ax = axes[0, 0]

# Show how A deforms the unit square
unit_square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]).T
transformed_square = A @ unit_square

# Original
ax.plot(unit_square[0], unit_square[1], '--', color=PAL['neutral'], 
       linewidth=2, label='Original unit square')
ax.fill(unit_square[0], unit_square[1], color=PAL['neutral'], alpha=0.1)

# Transformed
ax.plot(transformed_square[0], transformed_square[1], '-', color=PAL['transform'], 
       linewidth=2, label='Transformed')
ax.fill(transformed_square[0], transformed_square[1], color=PAL['transform'], alpha=0.2)

# Show where standard basis goes
ax.annotate('', xy=A[:, 0], xytext=(0, 0),
           arrowprops=dict(arrowstyle='->', color=PAL['col1'], lw=2))
ax.annotate('', xy=A[:, 1], xytext=(0, 0),
           arrowprops=dict(arrowstyle='->', color=PAL['col2'], lw=2))

ax.set_xlim(-1, 4)
ax.set_ylim(-2, 3)
ax.set_aspect('equal')
ax.legend(loc='upper left', fontsize=9)
ax.grid(alpha=0.3)
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_title('A. Matrix A transforms space\nUnit square → parallelogram', fontsize=11)

# -----------------------------------------------------------------------------
# Panel B: Finding the pre-image of b
# -----------------------------------------------------------------------------
ax = axes[0, 1]

# The question: what x maps to b?

# Show target b
ax.annotate('', xy=b, xytext=(0, 0),
           arrowprops=dict(arrowstyle='->', color=PAL['target'], lw=3))
ax.text(b[0]+0.1, b[1]+0.2, 'b = [5, 1]', fontsize=11, 
       color=PAL['target'], fontweight='bold')

# Show solution x
ax.annotate('', xy=x_sol, xytext=(0, 0),
           arrowprops=dict(arrowstyle='->', color=PAL['solution'], lw=3))
ax.text(x_sol[0]+0.1, x_sol[1]+0.2, 'x = [2, 1]', fontsize=11, 
       color=PAL['solution'], fontweight='bold')

# Show the transformation arrow
mid_point = (x_sol + b) / 2
ax.annotate('A', xy=b*0.8, xytext=x_sol*1.2,
           fontsize=14, fontweight='bold', color=PAL['transform'],
           arrowprops=dict(arrowstyle='->', color=PAL['transform'], lw=2,
                          connectionstyle='arc3,rad=0.3'))

ax.set_xlim(-1, 6)
ax.set_ylim(-1, 3)
ax.set_aspect('equal')
ax.grid(alpha=0.3)
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_title('B. Ax = b means\n"Find pre-image of b under A"', fontsize=11)

# -----------------------------------------------------------------------------
# Panel C: The inverse transformation
# -----------------------------------------------------------------------------
ax = axes[0, 2]

# A⁻¹ maps b back to x

A_inv = np.linalg.inv(A)

# Show b
ax.annotate('', xy=b, xytext=(0, 0),
           arrowprops=dict(arrowstyle='->', color=PAL['target'], lw=3))
ax.text(b[0]+0.1, b[1]+0.2, 'b', fontsize=12, color=PAL['target'], fontweight='bold')

# Show x = A⁻¹b
ax.annotate('', xy=x_sol, xytext=(0, 0),
           arrowprops=dict(arrowstyle='->', color=PAL['solution'], lw=3))
ax.text(x_sol[0]-0.5, x_sol[1]+0.2, 'x = A⁻¹b', fontsize=11, 
       color=PAL['solution'], fontweight='bold')

# Show the inverse transformation
ax.annotate('A⁻¹', xy=x_sol*1.1, xytext=b*0.9,
           fontsize=14, fontweight='bold', color=PAL['transform'],
           arrowprops=dict(arrowstyle='->', color=PAL['transform'], lw=2,
                          connectionstyle='arc3,rad=-0.3'))

ax.set_xlim(-1, 6)
ax.set_ylim(-1, 3)
ax.set_aspect('equal')
ax.grid(alpha=0.3)
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_title('C. x = A⁻¹b\n"Apply inverse transformation"', fontsize=11)

# -----------------------------------------------------------------------------
# Panel D: Connection to column picture
# -----------------------------------------------------------------------------
ax = axes[1, 0]
ax.axis('off')

text = """
TRANSFORMATION = COLUMN PICTURE!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

What does Ax actually compute?

A·x = [a₁ a₂]·[x₁] = x₁·a₁ + x₂·a₂
              [x₂]

The transformation Ax IS the linear
combination of columns weighted by x!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

So:
  • Columns of A = where basis vectors go
  • Ax = linear combination of columns
  • x = A⁻¹b finds weights to reach b

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This is why:
  COLUMN PICTURE = TRANSFORMATION PICTURE
  (just different emphasis)
"""

ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=10,
       ha='center', va='center', fontfamily='monospace',
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
ax.set_title('D. Columns = transformation', fontsize=11)

# -----------------------------------------------------------------------------
# Panel E: Connection to the metric
# -----------------------------------------------------------------------------
ax = axes[1, 1]

# For symmetric A, the transformation picture connects to the metric
A_sym = np.array([[1.5, 0.5],
                  [0.5, 1.2]])

# Unit circle
theta = np.linspace(0, 2*np.pi, 100)
circle = np.vstack([np.cos(theta), np.sin(theta)])

# Transformed circle
ellipse = A_sym @ circle

ax.plot(circle[0], circle[1], '--', color=PAL['neutral'], linewidth=2, 
       alpha=0.5, label='Unit circle')
ax.plot(ellipse[0], ellipse[1], '-', color=PAL['transform'], linewidth=2,
       label='A × circle = ellipse')

# The ellipse x'A⁻¹x = 1 (the metric unit ball)
eigvals, eigvecs = np.linalg.eigh(A_sym)
w = 2 * np.sqrt(eigvals[1])
h = 2 * np.sqrt(eigvals[0])
ang = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))

ellipse_patch = Ellipse((0, 0), w, h, angle=ang,
                        fill=False, edgecolor=PAL['target'], 
                        linewidth=2, linestyle=':', label="x'Ax = 1 (metric)")
ax.add_patch(ellipse_patch)

ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.legend(loc='upper right', fontsize=9)
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_title("E. Transformation connects to metric\nA × circle and x'Ax = 1", fontsize=11)

# -----------------------------------------------------------------------------
# Panel F: The complete unification
# -----------------------------------------------------------------------------
ax = axes[1, 2]
ax.axis('off')

text = """
ALL THREE VIEWS UNIFIED

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ROW PICTURE:
  Each row = a hyperplane
  Solution = intersection point

COLUMN PICTURE:
  Columns = basis vectors
  Solution = coordinates in that basis

TRANSFORMATION PICTURE:
  Matrix = space deformation
  Solution = pre-image of b

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

+ FOR SYMMETRIC MATRICES:

METRIC PICTURE:
  Matrix = way to measure length
  Ellipse = unit ball
  Eigenvalues = directional lengths

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

All roads lead to THE SAME SOLUTION!
Different views reveal different insights.
"""

ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=10,
       ha='center', va='center', fontfamily='monospace',
       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3,
                edgecolor=PAL['solution'], linewidth=2))
ax.set_title('F. Four views, one solution', fontsize=11)

plt.suptitle('VIEW 3: THE TRANSFORMATION PICTURE — Find the Pre-Image',
            fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('transformation_picture.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved: transformation_picture.png")
plt.close()

# =============================================================================
# FIGURE 4: THE GRAND UNIFICATION
# =============================================================================

print("Creating Figure 4: The Grand Unification...")

fig = plt.figure(figsize=(16, 14))

# Create a layout
ax = fig.add_subplot(111)
ax.axis('off')

grand_text = """
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                                                                        ┃
┃                    T H E   F O U R   V I E W S   O F   L I N E A R   A L G E B R A                      ┃
┃                                                                                                        ┃
┃                                    Ax = b   and   x'Σx = 1                                             ┃
┃                                                                                                        ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃                                                                                                        ┃
┃  VIEW 1: ROWS (Equations)                      VIEW 2: COLUMNS (Basis)                                 ┃
┃  ─────────────────────────                     ───────────────────────                                  ┃
┃                                                                                                        ┃
┃  Each row of A defines a hyperplane:           Each column of A is a basis vector:                     ┃
┃    a₁₁x₁ + a₁₂x₂ + ... = b₁                     [a₁ | a₂ | ... | aₙ]                                   ┃
┃                                                                                                        ┃
┃  The solution is where ALL planes meet.        The solution x gives coefficients:                      ┃
┃                                                  x₁·a₁ + x₂·a₂ + ... = b                               ┃
┃  ┌─────────────────┐                                                                                   ┃
┃  │     ╲   ╱       │                           "What linear combination of columns                     ┃
┃  │      ╲ ╱        │  ← intersection            gives the target vector b?"                            ┃
┃  │       ●         │                                                                                   ┃
┃  │      ╱ ╲        │                           ┌─────────────────┐                                     ┃
┃  │     ╱   ╲       │                           │   →a₁           │                                     ┃
┃  └─────────────────┘                           │     ↘  →b       │                                     ┃
┃                                                │       ●         │ ← b as combination                  ┃
┃  Classic "solving simultaneous equations"      │     ↗           │                                     ┃
┃                                                │   →a₂           │                                     ┃
┃                                                └─────────────────┘                                     ┃
┃                                                                                                        ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃                                                                                                        ┃
┃  VIEW 3: TRANSFORMATION (Mapping)              VIEW 4: METRIC (Covariance)                             ┃
┃  ────────────────────────────────              ───────────────────────────                              ┃
┃                                                                                                        ┃
┃  A transforms every vector x to Ax:            For symmetric Σ, the quadratic form                     ┃
┃    y = Ax  (matrix multiplication)              x'Σx defines a "squared length"                        ┃
┃                                                                                                        ┃
┃  The solution is the PRE-IMAGE of b:           The set x'Σ⁻¹x = 1 is an ellipse                        ┃
┃    "What x transforms into b?"                  (the "unit ball" of the Σ-metric)                      ┃
┃                                                                                                        ┃
┃  ┌─────────────────┐                           ┌─────────────────┐                                     ┃
┃  │  □ ──A──→ ◇     │                           │      ⬭          │                                     ┃
┃  │  ↑        ↑     │                           │   (ellipse)     │                                     ┃
┃  │  x        b     │                           │                 │                                     ┃
┃  │  ↓              │                           │  eigenvalues =  │                                     ┃
┃  │  x = A⁻¹b       │                           │  axis lengths²  │                                     ┃
┃  └─────────────────┘                           └─────────────────┘                                     ┃
┃                                                                                                        ┃
┃  A deforms space: circles → ellipses           This IS the covariance matrix view!                     ┃
┃  A⁻¹ reverses the deformation                  G, P, E are all metrics (rulers)                        ┃
┃                                                                                                        ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃                                                                                                        ┃
┃  EIGENVALUES IN EACH VIEW                                                                              ┃
┃  ────────────────────────                                                                              ┃
┃                                                                                                        ┃
┃    VIEW 1 (Rows):        Eigenvalues don't appear directly                                             ┃
┃                          (but condition number = λ_max/λ_min affects stability)                        ┃
┃                                                                                                        ┃
┃    VIEW 2 (Columns):     Eigenvalues measure "how independent" columns are                             ┃
┃                          (zero eigenvalue = linearly dependent columns)                                ┃
┃                                                                                                        ┃
┃    VIEW 3 (Transform):   Eigenvalues = stretch factors along principal axes                            ┃
┃                          (how much A expands/contracts each direction)                                 ┃
┃                                                                                                        ┃
┃    VIEW 4 (Metric):      Eigenvalues = squared lengths along principal axes                            ┃
┃                          = directional variances!                                                      ┃
┃                                                                                                        ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃                                                                                                        ┃
┃  FOR QUANTITATIVE GENETICS                                                                             ┃
┃  ════════════════════════                                                                              ┃
┃                                                                                                        ┃
┃  The Breeder's Equation:  Δz̄ = G P⁻¹ S = G β                                                           ┃
┃                                                                                                        ┃
┃    • P⁻¹S is VIEW 3: solving Pβ = S (find pre-image of S under P)                                      ┃
┃    • Gβ is VIEW 3: transforming β by G                                                                 ┃
┃    • G, P as metrics is VIEW 4: measuring length with different rulers                                 ┃
┃    • h²(β) = β'Gβ / β'Pβ compares two metrics                                                          ┃
┃                                                                                                        ┃
┃  Directional heritability unifies VIEWS 3 and 4:                                                       ┃
┃    • The ratio of two metric measurements                                                              ┃
┃    • Determines how effectively selection transforms into response                                     ┃
┃                                                                                                        ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
"""

ax.text(0.5, 0.5, grand_text, transform=ax.transAxes, fontsize=10.5,
       ha='center', va='center', fontfamily='monospace')

plt.savefig('grand_unification.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved: grand_unification.png")
plt.close()

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("THE FOUR VIEWS OF LINEAR ALGEBRA")
print("=" * 70)

print("""
┌───────────────────────────────────────────────────────────────────────┐
│                                                                       │
│  SOLVING Ax = b: FOUR EQUIVALENT PERSPECTIVES                        │
│  ════════════════════════════════════════════                         │
│                                                                       │
│  1. ROW PICTURE                                                       │
│     Each row = a hyperplane (line in 2D, plane in 3D)                 │
│     Solution = where all hyperplanes intersect                        │
│     This is the "classic" view taught first                           │
│                                                                       │
│  2. COLUMN PICTURE                                                    │
│     Columns = basis vectors                                           │
│     Solution = coefficients for linear combination                    │
│     "What weights on columns give b?"                                 │
│                                                                       │
│  3. TRANSFORMATION PICTURE                                            │
│     A = a function that maps vectors                                  │
│     Solution = pre-image of b under A                                 │
│     A⁻¹ reverses the transformation                                   │
│                                                                       │
│  4. METRIC PICTURE (for symmetric matrices)                           │
│     A = defines "length" in different directions                      │
│     The ellipse x'A⁻¹x = 1 is the "unit ball"                         │
│     Eigenvalues = squared lengths along principal axes                │
│                                                                       │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ALL FOUR GIVE THE SAME ANSWER!                                       │
│                                                                       │
│  The eigenvalues appear in each view:                                 │
│    • Rows: affect numerical stability (condition number)              │
│    • Columns: measure independence (zero = dependent)                 │
│    • Transformation: stretch factors along principal axes             │
│    • Metric: squared lengths = directional variances                  │
│                                                                       │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  FOR QUANTITATIVE GENETICS:                                           │
│                                                                       │
│  Δz̄ = GP⁻¹S combines TRANSFORMATION + METRIC views                    │
│                                                                       │
│  • P⁻¹S: solve the system Pβ = S                                      │
│  • Gβ: transform β by the G matrix                                    │
│  • h²(β): compare G-metric to P-metric in direction β                 │
│                                                                       │
│  The geometry of constraint IS the geometry of linear algebra!        │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
""")
