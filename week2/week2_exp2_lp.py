# ============================================================
# Week 2 — Part B: Linear Programming with PuLP
# ============================================================
# Goal: Use LP to find the optimal product mix that maximises
# profit subject to limited resources.
#
# Problem: A factory makes two products P1 and P2.
#   Profit per unit: P1 → ₹25, P2 → ₹30
#
#   Resource constraints (per unit):
#                   P1    P2    Capacity
#   Machine A (hr):  3     2      ≤ 120
#   Machine B (hr):  1     2      ≤  80
#   Labour  (hr):    2     1      ≤  60
#
#   Non-negativity: x1 ≥ 0, x2 ≥ 0
#
# LP formulation:
#   Maximise    25 x1 + 30 x2
#   subject to  3x1 + 2x2 ≤ 120   (Machine A)
#                x1 + 2x2 ≤  80   (Machine B)
#               2x1 +  x2 ≤  60   (Labour)
#                x1 ≥ 0,  x2 ≥ 0
# ============================================================

# ---- install PuLP if needed ----
try:
    import pulp
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pulp', '--quiet'])
    import pulp

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ------------------------------------------------------------------
# Utility: solve and print an LP problem
# ------------------------------------------------------------------
def solve_lp(name, c, A_ub, b_ub, var_names=None, sense='max'):
    """
    Solve an LP with PuLP and return the solution dict.

    Parameters
    ----------
    name      : Problem name (string)
    c         : Objective coefficients  [list]
    A_ub      : Constraint LHS matrix   [list of lists]
    b_ub      : Constraint RHS          [list]
    var_names : Variable labels         [list of strings]
    sense     : 'max' or 'min'
    """
    n = len(c)
    if var_names is None:
        var_names = [f'x{i+1}' for i in range(n)]

    prob = pulp.LpProblem(name, pulp.LpMaximize if sense == 'max' else pulp.LpMinimize)

    # Decision variables (non-negative)
    x = [pulp.LpVariable(var_names[i], lowBound=0) for i in range(n)]

    # Objective
    prob += pulp.lpDot(c, x), "Objective"

    # Constraints
    for j, (row, rhs) in enumerate(zip(A_ub, b_ub)):
        prob += (pulp.lpDot(row, x) <= rhs), f"C{j+1}"

    # Solve silently
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    solution = {
        'status'   : pulp.LpStatus[prob.status],
        'objective': pulp.value(prob.objective),
        'variables': {var_names[i]: pulp.value(x[i]) for i in range(n)},
    }
    return solution, prob


# ==================================================================
# BASE LP — Product Mix Problem
# ==================================================================
c      = [25, 30]
A_ub   = [[3, 2], [1, 2], [2, 1]]
b_ub   = [120, 80, 60]
names  = ['P1', 'P2']

sol, prob = solve_lp("Product_Mix", c, A_ub, b_ub, var_names=names)

print("=" * 50)
print("   LP — Product Mix (Base Case)")
print("=" * 50)
print(f"  Status: {sol['status']}")
print(f"  Optimal Profit: ₹{sol['objective']:.2f}")
for var, val in sol['variables'].items():
    print(f"    {var} = {val:.2f} units")

# Verify binding constraints
x1_opt = sol['variables']['P1']
x2_opt = sol['variables']['P2']
print("\n  Constraint utilisation at optimum:")
constraint_labels = ['Machine A', 'Machine B', 'Labour']
for label, row, rhs in zip(constraint_labels, A_ub, b_ub):
    used = row[0]*x1_opt + row[1]*x2_opt
    slack = rhs - used
    status = "BINDING" if slack < 1e-6 else f"slack={slack:.1f}"
    print(f"    {label:12s}: used={used:.1f}/{rhs}  [{status}]")


# ------------------------------------------------------------------
# Feasible region plot (2-variable LP only)
# ------------------------------------------------------------------
def plot_feasible_region(A_ub, b_ub, c, opt_point, opt_val,
                         var_names, constraint_names, title,
                         filename):
    """Plot the feasible region and optimal point for a 2-var LP."""
    fig, ax = plt.subplots(figsize=(8, 6))

    x_max = max(b_ub[i] / A_ub[i][0] for i in range(len(b_ub)) if A_ub[i][0] > 0) * 1.1
    y_max = max(b_ub[i] / A_ub[i][1] for i in range(len(b_ub)) if A_ub[i][1] > 0) * 1.1

    x_vals = np.linspace(0, x_max, 400)

    colors = ['tomato', 'steelblue', 'orange', 'purple', 'green']
    for j, (row, rhs, name) in enumerate(zip(A_ub, b_ub, constraint_names)):
        a1, a2 = row
        if a2 != 0:
            y_line = (rhs - a1 * x_vals) / a2
            ax.plot(x_vals, y_line, color=colors[j % len(colors)],
                    linewidth=2, label=f'{name}: {a1}{var_names[0]}+{a2}{var_names[1]}≤{rhs}')
        else:
            x_val = rhs / a1
            ax.axvline(x_val, color=colors[j % len(colors)], linewidth=2,
                       label=f'{name}: {a1}{var_names[0]}≤{rhs}')

    # Shade feasible region
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    # Enumerate corner vertices of feasible region
    from scipy.spatial import ConvexHull

    # Sample grid to find feasible points
    xs = np.linspace(0, x_max, 200)
    ys = np.linspace(0, y_max, 200)
    XX, YY = np.meshgrid(xs, ys)
    feasible = np.ones(XX.shape, dtype=bool)
    for row, rhs in zip(A_ub, b_ub):
        feasible &= (row[0]*XX + row[1]*YY <= rhs + 1e-9)
    ax.contourf(XX, YY, feasible.astype(float), levels=[0.5, 1.5],
                colors=['lightgreen'], alpha=0.3)

    # Plot optimal point
    ax.scatter(*opt_point, color='darkgreen', s=150, zorder=5,
               label=f'Optimal: ({opt_point[0]:.1f}, {opt_point[1]:.1f})  Z=₹{opt_val:.0f}')
    ax.annotate(f'  Opt={opt_point}', xy=opt_point, fontsize=10, color='darkgreen')

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel(var_names[0] + ' (units)')
    ax.set_ylabel(var_names[1] + ' (units)')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.show(block=False)
    print(f"\n  [Chart saved: {filename}]")

try:
    from scipy.spatial import ConvexHull
    plot_feasible_region(
        A_ub, b_ub, c,
        opt_point=(x1_opt, x2_opt),
        opt_val=sol['objective'],
        var_names=names,
        constraint_names=constraint_labels,
        title='LP — Feasible Region & Optimal Solution',
        filename='week2_lp_feasible_region.png'
    )
except ImportError:
    print("  (scipy not available – skipping feasible region plot)")


# ==================================================================
# EXPERIMENT 1 — What happens if Machine A capacity increases?
# ==================================================================
print("\n" + "=" * 50)
print("   EXPERIMENT 1 — Machine A Capacity Sensitivity")
print("=" * 50)

capacities_A = [60, 90, 120, 150, 180]
profits_A    = []

print(f"  {'Cap A':>8} | {'P1*':>8} | {'P2*':>8} | {'Profit*':>10}")
print("  " + "-" * 42)

for cap in capacities_A:
    A_ub_mod = [[3, 2], [1, 2], [2, 1]]
    b_mod    = [cap, 80, 60]
    s, _     = solve_lp("CapA", c, A_ub_mod, b_mod, var_names=names)
    p1 = s['variables']['P1']
    p2 = s['variables']['P2']
    pr = s['objective']
    profits_A.append(pr)
    print(f"  {cap:>8} | {p1:>8.1f} | {p2:>8.1f} | {pr:>10.2f}")

print("""
  Observation:
    Increasing Machine A capacity allows a better product mix
    and grows profit — but gains diminish as other constraints
    become binding.
""")

plt.figure(figsize=(7, 4))
plt.plot(capacities_A, profits_A, marker='o', color='steelblue', linewidth=2)
plt.xlabel('Machine A Capacity (hours)')
plt.ylabel('Optimal Profit (₹)')
plt.title('Exp 1: Profit vs Machine A Capacity')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('week2_exp3_capA_sensitivity.png', dpi=120)
plt.show(block=False)
print("  [Chart saved: week2_exp3_capA_sensitivity.png]")


# ==================================================================
# EXPERIMENT 2 — Shadow Prices / Dual Values
#   Which constraint is worth relaxing the most?
# ==================================================================
print("\n" + "=" * 50)
print("   EXPERIMENT 2 — Shadow Prices (marginal value of each resource)")
print("=" * 50)

delta    = 1          # increase RHS by 1 unit
base_obj = sol['objective']

print(f"  {'Constraint':15s} | {'Base RHS':>10} | {'Shadow Price':>14}")
print("  " + "-" * 45)

for j, (label, row, rhs) in enumerate(zip(constraint_labels, A_ub, b_ub)):
    b_mod = list(b_ub)
    b_mod[j] += delta
    s2, _  = solve_lp("Shadow", c, A_ub, b_mod, var_names=names)
    shadow = (s2['objective'] - base_obj) / delta
    print(f"  {label:15s} | {rhs:>10} | {shadow:>14.2f}")

print("""
  Interpretation:
    The shadow price (₹ / extra hour) tells you how much
    additional profit you gain from one more unit of that resource.
    A higher shadow price → that resource is the most valuable
    bottleneck to relax (invest here first).
""")


# ==================================================================
# EXPERIMENT 3 — New product P3 with reduced P2 profit
# ==================================================================
print("\n" + "=" * 50)
print("   EXPERIMENT 3 — Adding a Third Product P3")
print("=" * 50)
print("  P3 uses: Machine A=1hr, Machine B=3hr, Labour=2hr, Profit=₹20")

c3     = [25, 30, 20]
A_ub3  = [[3, 2, 1], [1, 2, 3], [2, 1, 2]]
b_ub3  = [120, 80, 60]
names3 = ['P1', 'P2', 'P3']

sol3, _ = solve_lp("ThreeProducts", c3, A_ub3, b_ub3, var_names=names3)

print(f"  Status: {sol3['status']}")
print(f"  Optimal Profit: ₹{sol3['objective']:.2f}")
for var, val in sol3['variables'].items():
    print(f"    {var} = {val:.2f} units")
print("""
  Note: Adding P3 allows the solver to use leftover Machine A
  capacity — the new mix may shift profit upward if P3 absorbs
  slack resources profitably.
""")


# ==================================================================
# EXPERIMENT 4 — Minimisation LP (Cost Minimisation)
#   Minimum cost diet-style problem
#   Minimise:  4 F1 + 6 F2
#   s.t.       3 F1 + 2 F2 ≥ 90  (protein min)
#              1 F1 + 4 F2 ≥ 80  (carbs   min)
#              F1, F2 ≥ 0
# ==================================================================
print("\n" + "=" * 50)
print("   EXPERIMENT 4 — Minimisation LP (Diet / Cost Problem)")
print("=" * 50)

prob_min = pulp.LpProblem("Diet_Min", pulp.LpMinimize)
f1 = pulp.LpVariable('F1', lowBound=0)
f2 = pulp.LpVariable('F2', lowBound=0)

prob_min += 4*f1 + 6*f2, "Cost"
prob_min += 3*f1 + 2*f2 >= 90, "Protein"
prob_min += 1*f1 + 4*f2 >= 80, "Carbs"

prob_min.solve(pulp.PULP_CBC_CMD(msg=False))

print(f"  Status         : {pulp.LpStatus[prob_min.status]}")
print(f"  Min Cost       : ₹{pulp.value(prob_min.objective):.2f}")
print(f"  F1 (units)     : {pulp.value(f1):.2f}")
print(f"  F2 (units)     : {pulp.value(f2):.2f}")
print("""
  This shows LP can handle ≥ constraints (≥ requirements)
  for minimisation problems — same solver, different sense.
""")


# ------------------------------------------------------------------
# Summary comparison bar chart
# ------------------------------------------------------------------
experiments = ['Base\n(2 products)', '3 Products\n(+P3)', 'Cap A=180\n(relaxed)']
profits_cmp = [sol['objective'], sol3['objective'], profits_A[-1]]

plt.figure(figsize=(7, 4))
bars = plt.bar(experiments, profits_cmp, color=['steelblue', 'green', 'orange'], width=0.5)
for bar, val in zip(bars, profits_cmp):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f'₹{val:.0f}', ha='center', fontsize=11, fontweight='bold')
plt.ylabel('Optimal Profit (₹)')
plt.title('LP Experiments — Profit Comparison')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('week2_lp_summary.png', dpi=120)
plt.show(block=False)
print("  [Chart saved: week2_lp_summary.png]")

plt.show()
print("\n[All LP experiments complete.]\n")
