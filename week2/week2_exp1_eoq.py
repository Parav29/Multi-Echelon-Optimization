# ============================================================
# Week 2 — Part A: Economic Order Quantity (EOQ)
# ============================================================
# Goal: Find the optimal order quantity Q* that minimises
# total annual cost = annual ordering cost + annual holding cost.
#
# Key assumptions:
#   • Constant and known demand rate D
#   • Zero lead time (orders arrive instantly)
#   • No shortages allowed
#
# Parameters:
#   D  – Annual demand (units / year)
#   K  – Fixed cost per order placed   (₹ / order)
#   h  – Holding cost per unit per year (₹ / unit / year)
#
# Formulas:
#   Q* = sqrt(2KD / h)          <- optimal order quantity
#   TC* = sqrt(2KDh)            <- minimum total annual cost
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Helper: compute EOQ metrics given parameters
# ------------------------------------------------------------------
def eoq(D, K, h):
    """Return (Q*, TC*, orders_per_year, days_between_orders)."""
    Q_star  = (2 * K * D / h) ** 0.5
    TC_star = (2 * K * D * h) ** 0.5
    orders_per_year      = D / Q_star
    days_between_orders  = 365 / orders_per_year
    return Q_star, TC_star, orders_per_year, days_between_orders


# ------------------------------------------------------------------
# Base-case parameters (from the lecture)
# ------------------------------------------------------------------
D_base = 1300     # annual demand (units / year)
K_base = 8        # fixed ordering cost (₹ / order)
h_base = 0.225    # holding cost (₹ / unit / year)

Q_star, TC_star, n_orders, cycle_days = eoq(D_base, K_base, h_base)

print("=" * 50)
print("   EOQ — Base Case")
print("=" * 50)
print(f"  Annual demand (D)              = {D_base} units/year")
print(f"  Ordering cost (K)              = ₹{K_base}/order")
print(f"  Holding cost (h)               = ₹{h_base}/unit/year")
print(f"  Optimal order quantity  Q*     = {Q_star:.1f} units")
print(f"  Minimum total annual cost TC*  = ₹{TC_star:.2f}")
print(f"  Orders per year                = {n_orders:.1f}")
print(f"  Days between orders            = {cycle_days:.0f} days")

# Verify: at Q* the two cost components should be equal
annual_ordering = K_base * (D_base / Q_star)
annual_holding  = h_base * (Q_star / 2)
print(f"\n  [Verification at Q*]")
print(f"    Annual ordering cost = ₹{annual_ordering:.2f}")
print(f"    Annual holding cost  = ₹{annual_holding:.2f}  (should match)")


# ------------------------------------------------------------------
# Total-cost curve: how TC changes with order quantity Q
# ------------------------------------------------------------------
Q_range = np.linspace(10, 500, 500)
ordering_cost = K_base * (D_base / Q_range)
holding_cost  = h_base * (Q_range / 2)
total_cost    = ordering_cost + holding_cost

plt.figure(figsize=(9, 5))
plt.plot(Q_range, ordering_cost, label='Annual Ordering Cost',  color='tomato',    linewidth=2)
plt.plot(Q_range, holding_cost,  label='Annual Holding Cost',   color='steelblue', linewidth=2)
plt.plot(Q_range, total_cost,    label='Total Annual Cost',     color='purple',    linewidth=2.5)
plt.axvline(Q_star, color='green', linestyle='--', linewidth=1.5, label=f'Q* = {Q_star:.1f}')
plt.axhline(TC_star, color='orange', linestyle=':', linewidth=1.5, label=f'TC* = ₹{TC_star:.2f}')
plt.xlabel('Order Quantity Q (units)')
plt.ylabel('Annual Cost (₹)')
plt.title('EOQ — Cost Curves')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('week2_eoq_cost_curve.png', dpi=120)
plt.show(block=False)
print("\n  [Chart saved: week2_eoq_cost_curve.png]")


# ==================================================================
# EXPERIMENT 1 — Sensitivity to Ordering Cost K
# ==================================================================
print("\n" + "=" * 50)
print("   EXPERIMENT 1 — Sensitivity to Ordering Cost K")
print("=" * 50)

K_values = [1, 8, 80]          # low / base / high ordering costs
print(f"  {'K':>6} | {'Q*':>10} | {'TC*':>12} | {'Orders/yr':>10} | {'Cycle days':>12}")
print("  " + "-" * 58)

Q_list = []
for K in K_values:
    Qs, TCs, n, d = eoq(D_base, K, h_base)
    Q_list.append(Qs)
    print(f"  {K:>6} | {Qs:>10.1f} | {TCs:>12.2f} | {n:>10.1f} | {d:>12.0f}")

print("""
  Observation:
    When K is very small (K=1), ordering is cheap so we order
    tiny batches very frequently.
    When K is large (K=80), each order is expensive so we bulk-
    buy infrequently with a large Q*.
""")

# Bar chart for experiment 1
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar([str(k) for k in K_values], Q_list, color=['steelblue', 'green', 'tomato'])
axes[0].set_xlabel('Ordering Cost K (₹)')
axes[0].set_ylabel('Optimal Q* (units)')
axes[0].set_title('Exp 1: Q* vs Ordering Cost K')
axes[0].grid(axis='y', alpha=0.3)

tc_list = [eoq(D_base, K, h_base)[1] for K in K_values]
axes[1].bar([str(k) for k in K_values], tc_list, color=['steelblue', 'green', 'tomato'])
axes[1].set_xlabel('Ordering Cost K (₹)')
axes[1].set_ylabel('Min Total Cost TC* (₹)')
axes[1].set_title('Exp 1: TC* vs Ordering Cost K')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('week2_exp1_sensitivity_K.png', dpi=120)
plt.show(block=False)
print("  [Chart saved: week2_exp1_sensitivity_K.png]")


# ==================================================================
# EXPERIMENT 2 — Sensitivity to Holding Cost h
# ==================================================================
print("\n" + "=" * 50)
print("   EXPERIMENT 2 — Sensitivity to Holding Cost h")
print("=" * 50)

# h=3.0 models refrigerated / perishable goods (expensive to store)
h_values = [0.05, 0.225, 1.0, 3.0]
labels    = ['0.05 (dry goods)', '0.225 (base)', '1.0 (moderate cold)', '3.0 (refrigerated)']

print(f"  {'h':>6} | {'Q*':>10} | {'TC*':>12} | {'Orders/yr':>10} | {'Cycle days':>12}")
print("  " + "-" * 58)

Q_h_list = []
for h_val, lbl in zip(h_values, labels):
    Qs, TCs, n, d = eoq(D_base, K_base, h_val)
    Q_h_list.append(Qs)
    print(f"  {h_val:>6} | {Qs:>10.1f} | {TCs:>12.2f} | {n:>10.1f} | {d:>12.0f}  ({lbl})")

print("""
  Observation:
    Higher holding costs (e.g. refrigeration at h=3.0) shrink Q*
    dramatically — it's cheaper to order more often in small batches
    than to hold large inventories in cold storage.
""")

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(labels, Q_h_list, color=['steelblue', 'green', 'orange', 'tomato'])
ax.set_xlabel('Holding Cost h (₹/unit/year)')
ax.set_ylabel('Optimal Q* (units)')
ax.set_title('Exp 2: Q* vs Holding Cost h')
ax.tick_params(axis='x', rotation=15)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('week2_exp2_sensitivity_h.png', dpi=120)
plt.show(block=False)
print("  [Chart saved: week2_exp2_sensitivity_h.png]")


# ==================================================================
# ROBUSTNESS DEMO — Square-root buffers estimation errors
# ==================================================================
print("\n" + "=" * 50)
print("   ROBUSTNESS — Square-root dampens demand estimation error")
print("=" * 50)

D_true  = D_base
D_over  = D_base * 1.20   # 20 % over-estimate
D_under = D_base * 0.80   # 20 % under-estimate

for label, D_val in [("True D", D_true), ("+20% D", D_over), ("-20% D", D_under)]:
    Qs, TCs, _, _ = eoq(D_val, K_base, h_base)
    pct_change = (Qs / Q_star - 1) * 100
    print(f"  {label}: D={D_val:.0f}  →  Q*={Qs:.1f}  ({pct_change:+.1f}% change)")

print("""
  A ±20 % error in demand only changes Q* by ~±10 %.
  The square-root formula makes EOQ robust to estimation noise.
""")

plt.show()   # keep all figures open until the user closes them
print("\n[All EOQ experiments complete.]\n")
