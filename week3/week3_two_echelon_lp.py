# ============================================================
# Week 3 — Two-Echelon Systems and Inventory Balance
# ============================================================
# Goal: Model a single-period, two-echelon supply chain with
#       one warehouse feeding one retailer using PuLP LP.
#
# Structure:
#   Supplier (∞) → Warehouse → Retailer → Customer
#
# Decision Variables:
#   o_w  : units ordered by warehouse from supplier
#   o_r  : units shipped from warehouse to retailer
#   I_w  : warehouse ending inventory
#   I_r  : retailer ending inventory
#
# Objective (Minimise total cost):
#   c_order × o_w  +  h_w × I_w  +  h_r × I_r
#
# Constraints:
#   Warehouse balance : I_w = I_w0 + o_w - o_r
#   Retailer  balance : I_r = I_r0 + o_r - demand
# ============================================================

# ---- install PuLP if needed --------------------------------
try:
    import pulp
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pulp', '--quiet'])
    import pulp

import matplotlib
matplotlib.use('Agg')           # non-interactive backend — saves PNGs without a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# All output files go into the week3 directory alongside this script
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# Helper — build and solve the two-echelon LP
# ============================================================
def solve_two_echelon(demand, h_w, h_r, c_order, I_w0, I_r0,
                      name='TwoEchelon', extra_constraints=None,
                      verbose=True):
    """
    Solve the single-period two-echelon LP.

    Parameters
    ----------
    demand            : int/float  — retailer demand this period
    h_w               : float      — warehouse holding cost per unit
    h_r               : float      — retailer  holding cost per unit
    c_order           : float      — variable ordering cost per unit (warehouse←supplier)
    I_w0              : float      — warehouse starting inventory
    I_r0              : float      — retailer  starting inventory
    name              : str        — LP problem label
    extra_constraints : list of (expr, label) — additional PuLP constraints
    verbose           : bool       — print a summary of the solution

    Returns
    -------
    dict with keys: status, total_cost, o_w, o_r, I_w, I_r
    """
    m = pulp.LpProblem(name, pulp.LpMinimize)

    # --- Decision variables (all ≥ 0) ---
    o_w = pulp.LpVariable('o_w', lowBound=0)   # ordered from supplier
    o_r = pulp.LpVariable('o_r', lowBound=0)   # shipped warehouse→retailer
    I_w = pulp.LpVariable('I_w', lowBound=0)   # warehouse ending inventory
    I_r = pulp.LpVariable('I_r', lowBound=0)   # retailer  ending inventory

    # --- Objective: minimise total cost ---
    m += c_order * o_w + h_w * I_w + h_r * I_r, 'TotalCost'

    # --- Constraint 1: warehouse inventory balance ---
    m += I_w == I_w0 + o_w - o_r, 'warehouse_balance'

    # --- Constraint 2: retailer inventory balance ---
    m += I_r == I_r0 + o_r - demand, 'retailer_balance'

    # --- Any additional experiment-specific constraints ---
    if extra_constraints:
        for expr, label in extra_constraints:
            m += expr, label

    # --- Solve ---
    m.solve(pulp.PULP_CBC_CMD(msg=0))

    result = {
        'status'    : pulp.LpStatus[m.status],
        'total_cost': pulp.value(m.objective) if m.status == 1 else None,
        'o_w'       : pulp.value(o_w),
        'o_r'       : pulp.value(o_r),
        'I_w'       : pulp.value(I_w),
        'I_r'       : pulp.value(I_r),
    }

    if verbose:
        _print_solution(result, demand, I_w0, I_r0)

    return result


def _print_solution(res, demand, I_w0, I_r0):
    """Pretty-print a solution dictionary."""
    print(f"  Status         : {res['status']}")
    if res['status'] != 'Optimal':
        print("  *** Problem is not optimal — check constraints. ***")
        return
    print(f"  Total Cost     : ${res['total_cost']:.2f}")
    print()
    print(f"  o_w (ordered supplier→warehouse) : {res['o_w']:.1f} units")
    print(f"  o_r (shipped  warehouse→retailer) : {res['o_r']:.1f} units")
    print(f"  I_w (warehouse ending inventory)  : {res['I_w']:.1f} units")
    print(f"  I_r (retailer  ending inventory)  : {res['I_r']:.1f} units")
    print()
    # Manual verification
    wh_check  = I_w0  + res['o_w'] - res['o_r']
    ret_check = I_r0  + res['o_r'] - demand
    print(f"  Warehouse check : {I_w0} + {res['o_w']:.0f} - {res['o_r']:.0f}"
          f" = {wh_check:.0f}   (I_w = {res['I_w']:.0f}) "
          f"{'✓' if abs(wh_check - res['I_w']) < 1e-4 else '✗'}")
    print(f"  Retailer  check : {I_r0} + {res['o_r']:.0f} - {demand}"
          f" = {ret_check:.0f}   (I_r = {res['I_r']:.0f}) "
          f"{'✓' if abs(ret_check - res['I_r']) < 1e-4 else '✗'}")


# ============================================================
# BASE MODEL  (as given in the PDF)
# ============================================================
print("=" * 60)
print("   WEEK 3 — Base Model: Single-Period Two-Echelon LP")
print("=" * 60)

# Parameters
demand  = 30     # retailer must sell exactly 30 units
h_w     = 0.5   # warehouse holding cost  ($0.50 / unit)
h_r     = 1.0   # retailer  holding cost  ($1.00 / unit)
c_order = 2.0   # ordering cost from supplier ($ / unit)
I_w0    = 50    # warehouse starting inventory
I_r0    = 20    # retailer  starting inventory

base_result = solve_two_echelon(
    demand, h_w, h_r, c_order, I_w0, I_r0,
    name='Base_TwoEchelon'
)

print("""
  Insight:
    The warehouse starts with 50 units — already more than enough to
    supply the retailer's demand of 30. The solver orders nothing from
    the supplier (o_w = 0) because ordering costs money and existing
    stock is sufficient. The retailer receives exactly enough to cover
    demand, and the solver minimises leftover (expensive) retailer stock.
""")


# ============================================================
# EXPERIMENT 1 — What if the retailer starts with zero stock?
# ============================================================
print("=" * 60)
print("   EXPERIMENT 1 — Retailer starts with zero stock  (I_r0 = 0)")
print("=" * 60)
print("  Parameters: demand=30, h_w=0.5, h_r=1.0, c_order=2.0, I_w0=50, I_r0=0")

exp1 = solve_two_echelon(
    demand=30, h_w=0.5, h_r=1.0, c_order=2.0,
    I_w0=50, I_r0=0,
    name='Exp1_ZeroRetailerStock'
)

print("""
  Insight (Exp 1):
    With I_r0 = 0 the retailer has no opening stock so the warehouse
    must ship exactly 30 units to satisfy demand. The warehouse itself
    still has 50 units on hand, so it does NOT need to order from the
    supplier. Total cost rises because the retailer's holding cost now
    covers a larger shipment received this period.
""")


# ============================================================
# EXPERIMENT 2 — Demand higher than starting inventory
# ============================================================
print("=" * 60)
print("   EXPERIMENT 2 — High demand: I_r0=0, demand=80")
print("=" * 60)
print("  Parameters: demand=80, h_w=0.5, h_r=1.0, c_order=2.0, I_w0=50, I_r0=0")

exp2 = solve_two_echelon(
    demand=80, h_w=0.5, h_r=1.0, c_order=2.0,
    I_w0=50, I_r0=0,
    name='Exp2_HighDemand'
)

print("""
  Insight (Exp 2):
    The retailer needs 80 units but starts with 0. The warehouse has
    only 50. The gap (80 - 50 = 30) must come from the supplier, so the
    solver orders exactly 30 units (o_w = 30). The warehouse then ships
    all 80 units to the retailer, ending with zero warehouse inventory.
    Ordering cost appears in the objective for the first time.
""")


# ============================================================
# EXPERIMENT 3 — Expensive retailer holding cost  (h_r = 5.0)
# ============================================================
print("=" * 60)
print("   EXPERIMENT 3 — Expensive retailer holding cost  (h_r = 5.0)")
print("=" * 60)
print("  Parameters: demand=30, h_w=0.5, h_r=5.0, c_order=2.0, I_w0=50, I_r0=20")

exp3 = solve_two_echelon(
    demand=30, h_w=0.5, h_r=5.0, c_order=2.0,
    I_w0=50, I_r0=20,
    name='Exp3_ExpensiveRetailerHolding'
)

print("""
  Insight (Exp 3):
    With h_r = 5.0 the solver aggressively minimises retailer ending
    inventory (I_r). It ships the warehouse only exactly what the
    retailer needs to cover demand (opting for I_r = 0). Raising h_r
    makes it extremely costly to leave stock at the retailer, so the
    solver avoids it — confirming that the cost structure drives the
    supply decision.
""")


# ============================================================
# EXPERIMENT 4 — Warehouse starts with zero stock
#   Part A: c_order = 2.0
#   Part B: c_order = 0.0  (free ordering)
# ============================================================
print("=" * 60)
print("   EXPERIMENT 4 — Warehouse starts with zero stock  (I_w0=0, I_r0=0)")
print("=" * 60)

print("  --- Part A: c_order = 2.0 ---")
exp4a = solve_two_echelon(
    demand=30, h_w=0.5, h_r=1.0, c_order=2.0,
    I_w0=0, I_r0=0,
    name='Exp4A_NoStock_PaidOrdering'
)

print("  --- Part B: c_order = 0.0 (free ordering) ---")
exp4b = solve_two_echelon(
    demand=30, h_w=0.5, h_r=1.0, c_order=0.0,
    I_w0=0, I_r0=0,
    name='Exp4B_NoStock_FreeOrdering'
)

print("""
  Insight (Exp 4):
    Part A: Both nodes start empty. The solver must order exactly 30
    units from the supplier and route them directly to the retailer.
    Total cost = ordering cost + holding costs.

    Part B: Setting c_order = 0 makes ordering free. The solver still
    orders the same 30 units (just enough to meet demand) — there is
    no incentive to over-order because holding costs (h_w, h_r) remain
    positive. The total cost drops by the ordering portion only.
""")


# ============================================================
# EXPERIMENT 5 — Break it: Infeasible problem
#   I_w0=0, I_r0=0, demand=30, but o_w ≤ 10  → infeasible
# ============================================================
print("=" * 60)
print("   EXPERIMENT 5 — Try to break it: o_w ≤ 10 with demand=30")
print("=" * 60)
print("  Parameters: I_w0=0, I_r0=0, demand=30, o_w ≤ 10")

m_inf = pulp.LpProblem('Exp5_Infeasible', pulp.LpMinimize)
o_w_i  = pulp.LpVariable('o_w', lowBound=0)
o_r_i  = pulp.LpVariable('o_r', lowBound=0)
I_w_i  = pulp.LpVariable('I_w', lowBound=0)
I_r_i  = pulp.LpVariable('I_r', lowBound=0)

m_inf += 2.0 * o_w_i + 0.5 * I_w_i + 1.0 * I_r_i
m_inf += I_w_i == 0 + o_w_i - o_r_i,   'warehouse_balance'
m_inf += I_r_i == 0 + o_r_i - 30,      'retailer_balance'
m_inf += o_w_i <= 10,                   'order_cap'         # <- artificial cap

m_inf.solve(pulp.PULP_CBC_CMD(msg=0))

print(f"  Status : {pulp.LpStatus[m_inf.status]}")
print("""
  Insight (Exp 5):
    The retailer needs 30 units. Both nodes start at zero. The only
    source is the supplier (via o_w), but we cap o_w ≤ 10. The system
    can source at most 10 units but must deliver 30 → infeasible.

    To make it feasible again you could:
      • Remove or raise the order cap (o_w ≤ 30+).
      • Give the warehouse or retailer a positive opening stock ≥ 30.
      • Reduce demand so that demand ≤ I_w0 + I_r0 + o_w_max.
""")


# ============================================================
# DELIVERABLE EXTENSION — Two Retailers
# ============================================================
print("=" * 60)
print("   DELIVERABLE EXTENSION — Two Retailers")
print("   Retailer 1 demand=20, Retailer 2 demand=15")
print("=" * 60)
print("  Parameters: I_w0=50, I_r1_0=5, I_r2_0=5")
print("              h_w=0.5, h_r1=1.0, h_r2=1.2, c_order=2.0")

# Parameters
demand_r1 = 20
demand_r2 = 15
h_r1      = 1.0
h_r2      = 1.2      # slightly more expensive shelf space at retailer 2
I_r1_0    = 5
I_r2_0    = 5
I_w0_2r   = 50
h_w_2r    = 0.5
c_order_2r= 2.0

m2r = pulp.LpProblem('TwoRetailers', pulp.LpMinimize)

o_w2  = pulp.LpVariable('o_w',  lowBound=0)   # warehouse ← supplier
o_r1  = pulp.LpVariable('o_r1', lowBound=0)   # warehouse → retailer 1
o_r2  = pulp.LpVariable('o_r2', lowBound=0)   # warehouse → retailer 2
I_w2  = pulp.LpVariable('I_w',  lowBound=0)   # warehouse ending inventory
I_r1  = pulp.LpVariable('I_r1', lowBound=0)   # retailer 1 ending inventory
I_r2  = pulp.LpVariable('I_r2', lowBound=0)   # retailer 2 ending inventory

# Objective
m2r += c_order_2r * o_w2 + h_w_2r * I_w2 + h_r1 * I_r1 + h_r2 * I_r2, 'TotalCost'

# Warehouse balance: receives from supplier, ships to BOTH retailers
m2r += I_w2 == I_w0_2r + o_w2 - o_r1 - o_r2, 'warehouse_balance'

# Retailer 1 balance
m2r += I_r1 == I_r1_0 + o_r1 - demand_r1, 'retailer1_balance'

# Retailer 2 balance
m2r += I_r2 == I_r2_0 + o_r2 - demand_r2, 'retailer2_balance'

m2r.solve(pulp.PULP_CBC_CMD(msg=0))

print(f"\n  Status         : {pulp.LpStatus[m2r.status]}")
if m2r.status == 1:
    print(f"  Total Cost     : ${pulp.value(m2r.objective):.2f}")
    print()
    print(f"  o_w  (supplier → warehouse)  : {o_w2.varValue:.1f} units")
    print(f"  o_r1 (warehouse → retailer1) : {o_r1.varValue:.1f} units")
    print(f"  o_r2 (warehouse → retailer2) : {o_r2.varValue:.1f} units")
    print(f"  I_w  (warehouse ending inv.) : {I_w2.varValue:.1f} units")
    print(f"  I_r1 (retailer 1 ending inv.): {I_r1.varValue:.1f} units")
    print(f"  I_r2 (retailer 2 ending inv.): {I_r2.varValue:.1f} units")
    print()

    # Manual verification
    wh2_check   = I_w0_2r  + o_w2.varValue  - o_r1.varValue - o_r2.varValue
    ret1_check  = I_r1_0   + o_r1.varValue  - demand_r1
    ret2_check  = I_r2_0   + o_r2.varValue  - demand_r2
    print(f"  Warehouse check  : {I_w0_2r} + {o_w2.varValue:.0f} "
          f"- {o_r1.varValue:.0f} - {o_r2.varValue:.0f} "
          f"= {wh2_check:.0f}   (I_w = {I_w2.varValue:.0f}) "
          f"{'✓' if abs(wh2_check - I_w2.varValue) < 1e-4 else '✗'}")
    print(f"  Retailer 1 check : {I_r1_0} + {o_r1.varValue:.0f} "
          f"- {demand_r1} = {ret1_check:.0f}   "
          f"(I_r1 = {I_r1.varValue:.0f}) "
          f"{'✓' if abs(ret1_check - I_r1.varValue) < 1e-4 else '✗'}")
    print(f"  Retailer 2 check : {I_r2_0} + {o_r2.varValue:.0f} "
          f"- {demand_r2} = {ret2_check:.0f}   "
          f"(I_r2 = {I_r2.varValue:.0f}) "
          f"{'✓' if abs(ret2_check - I_r2.varValue) < 1e-4 else '✗'}")

print("""
  Insight (Two Retailers):
    The warehouse balance equation now subtracts shipments to BOTH
    retailers. The solver independently decides o_r1 and o_r2 to
    minimise cost. Because h_r2 > h_r1, the solver is slightly more
    careful about leaving stock at retailer 2. The warehouse still has
    enough opening stock (50) to cover total demand (20+15=35) without
    ordering from the supplier.
""")


# ============================================================
# VISUALISATION  — Cost comparison across all experiments
# ============================================================

# Collect total costs (None if infeasible)
exp_labels = [
    'Base\n(I_r0=20)',
    'Exp1\n(I_r0=0)',
    'Exp2\nDmd=80',
    'Exp3\nh_r=5',
    'Exp4A\nAll-zero',
    'Exp4B\nFree ord.',
    '2-Retailer\nExtension',
]
exp_costs = [
    base_result['total_cost'],
    exp1['total_cost'],
    exp2['total_cost'],
    exp3['total_cost'],
    exp4a['total_cost'],
    exp4b['total_cost'],
    pulp.value(m2r.objective) if m2r.status == 1 else None,
]

colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2',
          '#CCB974', '#64B5CD', '#DD8452']

fig, ax = plt.subplots(figsize=(12, 5))
for i, (label, cost, color) in enumerate(zip(exp_labels, exp_costs, colors)):
    if cost is not None:
        bar = ax.bar(i, cost, color=color, width=0.6, edgecolor='white', linewidth=1.2)
        ax.text(i, cost + 0.3, f'${cost:.1f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='#222')
    else:
        ax.bar(i, 0, color='#ccc', width=0.6, edgecolor='white', linewidth=1.2)
        ax.text(i, 1, 'Infeasible', ha='center', va='bottom',
                fontsize=8, color='#c00', fontweight='bold')

ax.set_xticks(range(len(exp_labels)))
ax.set_xticklabels(exp_labels, fontsize=9)
ax.set_ylabel('Total Cost ($)', fontsize=11)
ax.set_title('Week 3 — Two-Echelon LP: Cost Comparison Across Experiments',
             fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
chart_path = os.path.join(OUT_DIR, 'week3_cost_comparison.png')
plt.savefig(chart_path, dpi=130)
print(f"  [Chart saved: {chart_path}]")


# ============================================================
# VISUALISATION  — Sensitivity: demand vs total cost
#   (varying demand from 10 to 100, base parameters)
# ============================================================
demands     = list(range(10, 101, 5))
costs_dem   = []
o_w_vals    = []
o_r_vals    = []

for d in demands:
    r = solve_two_echelon(d, h_w=0.5, h_r=1.0, c_order=2.0,
                          I_w0=50, I_r0=20, verbose=False)
    costs_dem.append(r['total_cost'] if r['status'] == 'Optimal' else None)
    o_w_vals.append(r['o_w'] if r['status'] == 'Optimal' else None)
    o_r_vals.append(r['o_r'] if r['status'] == 'Optimal' else None)

fig2, axes = plt.subplots(1, 2, figsize=(13, 5))
fig2.suptitle('Week 3 — Sensitivity Analysis: Demand vs Cost & Shipment Decisions',
              fontsize=12, fontweight='bold')

# Left: total cost vs demand
axes[0].plot(demands, costs_dem, marker='o', color='#4C72B0', linewidth=2, markersize=5)
axes[0].axvline(x=70, color='#C44E52', linestyle='--', linewidth=1.2,
                label='Warehouse stock exhausted (I_w0=50, I_r0=20 → 70 total)')
axes[0].set_xlabel('Demand (units)', fontsize=11)
axes[0].set_ylabel('Total Cost ($)', fontsize=11)
axes[0].set_title('Total Cost vs Demand', fontsize=11)
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Right: o_w and o_r vs demand
axes[1].plot(demands, o_w_vals, marker='s', color='#C44E52', linewidth=2,
             markersize=5, label='o_w (ordered from supplier)')
axes[1].plot(demands, o_r_vals, marker='^', color='#55A868', linewidth=2,
             markersize=5, label='o_r (shipped to retailer)')
axes[1].set_xlabel('Demand (units)', fontsize=11)
axes[1].set_ylabel('Units', fontsize=11)
axes[1].set_title('Ordering & Shipment Quantities vs Demand', fontsize=11)
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.tight_layout()
sens_path = os.path.join(OUT_DIR, 'week3_demand_sensitivity.png')
plt.savefig(sens_path, dpi=130)
print(f"  [Chart saved: {sens_path}]")


# ============================================================
# VISUALISATION  — Two-retailer inventory bar chart
# ============================================================
if m2r.status == 1:
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    nodes      = ['Warehouse', 'Retailer 1', 'Retailer 2']
    start_inv  = [I_w0_2r,  I_r1_0,         I_r2_0]
    end_inv    = [I_w2.varValue, I_r1.varValue, I_r2.varValue]

    x     = np.arange(len(nodes))
    width = 0.35

    b1 = ax3.bar(x - width/2, start_inv, width, label='Opening Inventory',
                 color='#64B5CD', edgecolor='white')
    b2 = ax3.bar(x + width/2, end_inv,   width, label='Closing Inventory',
                 color='#4C72B0', edgecolor='white')

    for bar, val in zip(list(b1) + list(b2),
                        start_inv + end_inv):
        ax3.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.4,
                 f'{val:.0f}', ha='center', va='bottom', fontsize=10)

    ax3.set_xticks(x)
    ax3.set_xticklabels(nodes, fontsize=11)
    ax3.set_ylabel('Inventory (units)', fontsize=11)
    ax3.set_title('Two-Retailer Extension — Opening vs Closing Inventory',
                  fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    plt.tight_layout()
    two_ret_path = os.path.join(OUT_DIR, 'week3_two_retailer_inventory.png')
    plt.savefig(two_ret_path, dpi=130)
    print(f"  [Chart saved: {two_ret_path}]")

print("\n" + "=" * 60)
print("   All Week 3 experiments complete.")
print("=" * 60)
