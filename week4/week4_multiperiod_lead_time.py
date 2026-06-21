"""
week4_multiperiod_lead_time.py
==============================
Week 4 — Lead Times and Multi-Period Inventory
Two-echelon supply chain (Supplier → Warehouse → Retailer) extended to:
  • T periods  (multi-period planning horizon)
  • Lead time L (order placed in period t arrives in period t + L)

Decision variables per period (dictionary variables via LpVariable.dicts):
  o_w[t]  — units ordered by warehouse from supplier in period t
  o_r[t]  — units shipped from warehouse to retailer in period t
  I_w[t]  — warehouse ending inventory in period t
  I_r[t]  — retailer  ending inventory in period t

Objective: minimise total cost over T periods
  Σ_t  c_order * o_w[t]  +  h_w * I_w[t]  +  h_r * I_r[t]

Inventory balance (warehouse):
  I_w[t] = I_w[t-1] + received[t] - o_r[t]
  where received[t] = o_w[t - L]  if t - L >= 0  else  0

Inventory balance (retailer):
  I_r[t] = I_r[t-1] + o_r[t] - demand[t]
"""

import os
import pulp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─── Output directory ────────────────────────────────────────────────────────
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Colour palette ──────────────────────────────────────────────────────────
PALETTE = {
    'warehouse': '#4C72B0',
    'retailer':  '#55A868',
    'order_w':   '#C44E52',
    'order_r':   '#DD8452',
    'demand':    '#8172B2',
    'cost':      '#CCB974',
    'grid':      '#e8e8e8',
    'bg':        '#F9F9F9',
}


# ═══════════════════════════════════════════════════════════════════════════════
# CORE SOLVER
# ═══════════════════════════════════════════════════════════════════════════════

def solve_multiperiod(
    T=5,
    LT=1,
    demand=None,
    h_w=0.5,
    h_r=1.0,
    c_order=2.0,
    I_w0=50,
    I_r0=10,
    verbose=True,
    label='Base',
):
    """
    Solve the multi-period two-echelon inventory LP.

    Parameters
    ----------
    T       : int   — number of planning periods
    LT      : int   — lead time (periods)
    demand  : list  — retailer demand per period (length T); uniform 20 if None
    h_w     : float — warehouse holding cost per unit per period
    h_r     : float — retailer  holding cost per unit per period
    c_order : float — ordering cost per unit (warehouse ← supplier)
    I_w0    : float — warehouse opening inventory
    I_r0    : float — retailer  opening inventory
    verbose : bool  — whether to print results
    label   : str   — experiment label for printing

    Returns
    -------
    dict with keys: status, total_cost, o_w, o_r, I_w, I_r  (lists length T)
    """
    if demand is None:
        demand = [20] * T

    m = pulp.LpProblem(f'MultiPeriod_LT{LT}_T{T}', pulp.LpMinimize)

    # ── Decision variables (dictionary variables) ────────────────────────────
    o_w = pulp.LpVariable.dicts('o_w', range(T), lowBound=0)   # warehouse orders
    o_r = pulp.LpVariable.dicts('o_r', range(T), lowBound=0)   # retailer shipments
    I_w = pulp.LpVariable.dicts('I_w', range(T), lowBound=0)   # warehouse inventory
    I_r = pulp.LpVariable.dicts('I_r', range(T), lowBound=0)   # retailer  inventory

    # ── Objective: minimise total cost across all T periods ──────────────────
    m += pulp.lpSum(
        c_order * o_w[t] + h_w * I_w[t] + h_r * I_r[t]
        for t in range(T)
    ), 'TotalCost'

    # ── Constraints loop ─────────────────────────────────────────────────────
    for t in range(T):
        # Lead-time: what arrives in period t was ordered LT periods ago
        if t - LT >= 0:
            received = o_w[t - LT]   # LP variable from LT periods back
        else:
            received = 0             # nothing was ordered before period 0

        prev_Iw = I_w[t - 1] if t > 0 else I_w0
        prev_Ir = I_r[t - 1] if t > 0 else I_r0

        # Warehouse balance
        m += I_w[t] == prev_Iw + received - o_r[t], f'wh_balance_{t}'

        # Retailer balance
        m += I_r[t] == prev_Ir + o_r[t] - demand[t], f'ret_balance_{t}'

    # ── Solve ────────────────────────────────────────────────────────────────
    m.solve(pulp.PULP_CBC_CMD(msg=0))

    status = pulp.LpStatus[m.status]
    if m.status != 1:
        if verbose:
            print(f'\n  [{label}] Status: {status} — no feasible solution found.')
        return {'status': status, 'total_cost': None,
                'o_w': None, 'o_r': None, 'I_w': None, 'I_r': None}

    ow_vals = [o_w[t].varValue for t in range(T)]
    or_vals = [o_r[t].varValue for t in range(T)]
    Iw_vals = [I_w[t].varValue for t in range(T)]
    Ir_vals = [I_r[t].varValue for t in range(T)]
    total   = pulp.value(m.objective)

    if verbose:
        print(f'\n{"─"*62}')
        print(f'  Experiment : {label}')
        print(f'  T={T}  Lead Time={LT}  c_order={c_order}  h_w={h_w}  h_r={h_r}')
        print(f'  Opening: I_w0={I_w0}  I_r0={I_r0}')
        print(f'  Demand : {demand}')
        print(f'  Status : {status}')
        print(f'  Total Cost: ${total:.2f}')
        print(f'  {"Period":<8} {"o_w":>7} {"o_r":>7} {"I_w":>7} {"I_r":>7}  {"demand":>7}')
        print(f'  {"------":<8} {"---":>7} {"---":>7} {"---":>7} {"---":>7}  {"------":>7}')
        for t in range(T):
            print(f'  t={t:<6} {ow_vals[t]:>7.1f} {or_vals[t]:>7.1f} '
                  f'{Iw_vals[t]:>7.1f} {Ir_vals[t]:>7.1f}  {demand[t]:>7}')
        print(f'{"─"*62}')

    return {
        'status': status,
        'total_cost': total,
        'o_w': ow_vals,
        'o_r': or_vals,
        'I_w': Iw_vals,
        'I_r': Ir_vals,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════════════════

print('\n' + '=' * 62)
print('   WEEK 4 — Multi-Period Inventory LP with Lead Times')
print('=' * 62)

# ── BASE CASE ────────────────────────────────────────────────────────────────
print('\n[BASE CASE] T=5, Lead Time=1, uniform demand=20')
base = solve_multiperiod(
    T=5, LT=1,
    demand=[20, 20, 20, 20, 20],
    h_w=0.5, h_r=1.0, c_order=2.0,
    I_w0=50, I_r0=10,
    verbose=True, label='Base (LT=1, T=5)',
)

# ── EXPERIMENT 1: Lead Time = 0 ───────────────────────────────────────────────
print('\n[EXP 1] Lead Time = 0 (instant delivery)')
exp1 = solve_multiperiod(
    T=5, LT=0,
    demand=[20, 20, 20, 20, 20],
    h_w=0.5, h_r=1.0, c_order=2.0,
    I_w0=50, I_r0=10,
    verbose=True, label='Exp1: LT=0',
)

# ── EXPERIMENT 2: Lead Time = 2 ───────────────────────────────────────────────
print('\n[EXP 2] Lead Time = 2 (delayed delivery)')
exp2 = solve_multiperiod(
    T=5, LT=2,
    demand=[20, 20, 20, 20, 20],
    h_w=0.5, h_r=1.0, c_order=2.0,
    I_w0=50, I_r0=10,
    verbose=True, label='Exp2: LT=2',
)

# ── EXPERIMENT 3: Longer horizon T=10 ─────────────────────────────────────────
print('\n[EXP 3] T=10, Lead Time=1, uniform demand=15')
exp3 = solve_multiperiod(
    T=10, LT=1,
    demand=[15] * 10,
    h_w=0.5, h_r=1.0, c_order=2.0,
    I_w0=50, I_r0=10,
    verbose=True, label='Exp3: T=10, LT=1',
)

# ── EXPERIMENT 4: Seasonal demand pattern ─────────────────────────────────────
print('\n[EXP 4] Seasonal demand: [10, 15, 30, 40, 25], LT=1')
exp4 = solve_multiperiod(
    T=5, LT=1,
    demand=[10, 15, 30, 40, 25],
    h_w=0.5, h_r=1.0, c_order=2.0,
    I_w0=50, I_r0=10,
    verbose=True, label='Exp4: Seasonal demand',
)

# ── EXPERIMENT 5: High ordering cost ──────────────────────────────────────────
print('\n[EXP 5] High ordering cost c_order=5, LT=1')
exp5 = solve_multiperiod(
    T=5, LT=1,
    demand=[20, 20, 20, 20, 20],
    h_w=0.5, h_r=1.0, c_order=5.0,
    I_w0=20, I_r0=5,
    verbose=True, label='Exp5: High c_order=5',
)

# ── EXPERIMENT 6: Low initial inventory — stresses lead time ──────────────────
print('\n[EXP 6] Low opening stock, LT=2, demand=20/period')
exp6 = solve_multiperiod(
    T=5, LT=2,
    demand=[20, 20, 20, 20, 20],
    h_w=0.5, h_r=1.0, c_order=2.0,
    I_w0=10, I_r0=5,
    verbose=True, label='Exp6: LT=2, low stock',
)


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════════════

T_base  = 5
periods = list(range(T_base))

# ── FIGURE 1: Base-case timeline ──────────────────────────────────────────────
fig1, axes1 = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
fig1.patch.set_facecolor(PALETTE['bg'])
fig1.suptitle('Week 4 — Base Case: Multi-Period Inventory with Lead Time = 1',
              fontsize=14, fontweight='bold', y=0.98)

# Panel 1: Orders
ax = axes1[0]
ax.set_facecolor(PALETTE['bg'])
w1 = ax.bar([t - 0.18 for t in periods], base['o_w'], width=0.35,
            label='o_w (warehouse order)', color=PALETTE['order_w'], alpha=0.85)
w2 = ax.bar([t + 0.18 for t in periods], base['o_r'], width=0.35,
            label='o_r (retailer shipment)', color=PALETTE['order_r'], alpha=0.85)
for bar in list(w1) + list(w2):
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                f'{h:.0f}', ha='center', va='bottom', fontsize=8)
ax.set_ylabel('Units', fontsize=10)
ax.set_title('Orders & Shipments per Period', fontsize=10, pad=4)
ax.legend(fontsize=9, loc='upper right')
ax.grid(axis='y', color=PALETTE['grid'], linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel 2: Inventory levels
ax = axes1[1]
ax.set_facecolor(PALETTE['bg'])
ax.plot(periods, base['I_w'], marker='o', color=PALETTE['warehouse'],
        linewidth=2.2, label='I_w (warehouse)', markersize=7)
ax.plot(periods, base['I_r'], marker='s', color=PALETTE['retailer'],
        linewidth=2.2, label='I_r (retailer)', markersize=7)
for t in periods:
    ax.annotate(f"{base['I_w'][t]:.0f}", (t, base['I_w'][t]),
                textcoords='offset points', xytext=(0, 7),
                ha='center', fontsize=8, color=PALETTE['warehouse'])
    ax.annotate(f"{base['I_r'][t]:.0f}", (t, base['I_r'][t]),
                textcoords='offset points', xytext=(0, -14),
                ha='center', fontsize=8, color=PALETTE['retailer'])
ax.set_ylabel('Inventory (units)', fontsize=10)
ax.set_title('Inventory Levels per Period', fontsize=10, pad=4)
ax.legend(fontsize=9)
ax.grid(color=PALETTE['grid'], linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel 3: Demand vs shipment to retailer
ax = axes1[2]
ax.set_facecolor(PALETTE['bg'])
ax.bar(periods, [20] * T_base, width=0.4, label='Demand', color=PALETTE['demand'],
       alpha=0.6, align='center')
ax.plot(periods, base['o_r'], marker='^', color=PALETTE['order_r'],
        linewidth=2, label='Shipment to retailer', markersize=8)
ax.set_xlabel('Period', fontsize=10)
ax.set_ylabel('Units', fontsize=10)
ax.set_title('Demand vs. Shipment to Retailer', fontsize=10, pad=4)
ax.set_xticks(periods)
ax.legend(fontsize=9)
ax.grid(color=PALETTE['grid'], linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig1_path = os.path.join(OUT_DIR, 'week4_base_timeline.png')
plt.savefig(fig1_path, dpi=130, bbox_inches='tight')
print(f'\n  [Chart saved: {fig1_path}]')
plt.close(fig1)


# ── FIGURE 2: Lead Time Comparison (LT=0, LT=1, LT=2) ────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
fig2.patch.set_facecolor(PALETTE['bg'])
fig2.suptitle('Week 4 — Inventory Levels: Comparing Lead Times (LT = 0, 1, 2)',
              fontsize=13, fontweight='bold', y=1.01)

for ax, result, lt, title in zip(
    axes2,
    [exp1, base, exp2],
    [0, 1, 2],
    ['Lead Time = 0\n(instant delivery)',
     'Lead Time = 1\n(base case)',
     'Lead Time = 2\n(2-period delay)'],
):
    ax.set_facecolor(PALETTE['bg'])
    if result['status'] == 'Optimal':
        ax.plot(periods, result['I_w'], marker='o', color=PALETTE['warehouse'],
                linewidth=2.2, label='I_w (warehouse)', markersize=7)
        ax.plot(periods, result['I_r'], marker='s', color=PALETTE['retailer'],
                linewidth=2.2, label='I_r (retailer)', markersize=7)
        cost_txt = f"Total Cost: ${result['total_cost']:.2f}"
    else:
        cost_txt = 'Infeasible'
    ax.set_title(f'{title}\n{cost_txt}', fontsize=10)
    ax.set_xlabel('Period', fontsize=9)
    ax.set_ylabel('Inventory (units)', fontsize=9)
    ax.set_xticks(periods)
    ax.legend(fontsize=8)
    ax.grid(color=PALETTE['grid'], linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
fig2_path = os.path.join(OUT_DIR, 'week4_lead_time_comparison.png')
plt.savefig(fig2_path, dpi=130, bbox_inches='tight')
print(f'  [Chart saved: {fig2_path}]')
plt.close(fig2)


# ── FIGURE 3: Seasonal demand — inventory & ordering ─────────────────────────
demand_seasonal = [10, 15, 30, 40, 25]
fig3, axes3 = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
fig3.patch.set_facecolor(PALETTE['bg'])
fig3.suptitle('Week 4 — Seasonal Demand: Inventory & Ordering Decisions',
              fontsize=13, fontweight='bold')

ax = axes3[0]
ax.set_facecolor(PALETTE['bg'])
ax.bar([t - 0.18 for t in periods], exp4['o_w'], width=0.35,
       label='o_w (warehouse order)', color=PALETTE['order_w'], alpha=0.85)
ax.bar([t + 0.18 for t in periods], exp4['o_r'], width=0.35,
       label='o_r (retailer shipment)', color=PALETTE['order_r'], alpha=0.85)
ax.plot(periods, demand_seasonal, marker='D', color=PALETTE['demand'],
        linewidth=2, label='Demand', markersize=7, linestyle='--')
ax.set_ylabel('Units', fontsize=10)
ax.set_title('Orders vs Demand (Seasonal Pattern)', fontsize=10)
ax.legend(fontsize=9)
ax.grid(axis='y', color=PALETTE['grid'], linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax = axes3[1]
ax.set_facecolor(PALETTE['bg'])
ax.plot(periods, exp4['I_w'], marker='o', color=PALETTE['warehouse'],
        linewidth=2.2, label='I_w (warehouse)', markersize=7)
ax.plot(periods, exp4['I_r'], marker='s', color=PALETTE['retailer'],
        linewidth=2.2, label='I_r (retailer)', markersize=7)
ax.fill_between(periods, exp4['I_w'], alpha=0.12, color=PALETTE['warehouse'])
ax.fill_between(periods, exp4['I_r'], alpha=0.12, color=PALETTE['retailer'])
ax.set_xlabel('Period', fontsize=10)
ax.set_ylabel('Inventory (units)', fontsize=10)
ax.set_title('Inventory Levels (Seasonal Pattern)', fontsize=10)
ax.set_xticks(periods)
ax.legend(fontsize=9)
ax.grid(color=PALETTE['grid'], linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
fig3_path = os.path.join(OUT_DIR, 'week4_seasonal_demand.png')
plt.savefig(fig3_path, dpi=130, bbox_inches='tight')
print(f'  [Chart saved: {fig3_path}]')
plt.close(fig3)


# ── FIGURE 4: Sensitivity — Lead Time vs Total Cost ──────────────────────────
lead_times  = list(range(0, 6))
lt_costs    = []
for lt in lead_times:
    r = solve_multiperiod(
        T=5, LT=lt,
        demand=[20] * 5,
        h_w=0.5, h_r=1.0, c_order=2.0,
        I_w0=50, I_r0=10,
        verbose=False,
    )
    lt_costs.append(r['total_cost'])

fig4, ax4 = plt.subplots(figsize=(9, 5))
fig4.patch.set_facecolor(PALETTE['bg'])
ax4.set_facecolor(PALETTE['bg'])
colors_lt = [PALETTE['warehouse'] if c is not None else '#cccccc' for c in lt_costs]
bars = ax4.bar(lead_times, [c if c is not None else 0 for c in lt_costs],
               color=colors_lt, edgecolor='white', linewidth=1.2, width=0.55)
for bar, cost in zip(bars, lt_costs):
    label = f'${cost:.1f}' if cost is not None else 'Infeasible'
    ax4.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.3,
             label, ha='center', va='bottom', fontsize=9, fontweight='bold')
ax4.set_xlabel('Lead Time (periods)', fontsize=11)
ax4.set_ylabel('Total Cost ($)', fontsize=11)
ax4.set_title('Week 4 — Sensitivity: Lead Time vs Total Cost\n'
              '(T=5, demand=20/period, I_w0=50, I_r0=10)',
              fontsize=12, fontweight='bold')
ax4.set_xticks(lead_times)
ax4.grid(axis='y', color=PALETTE['grid'], linewidth=0.8)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
plt.tight_layout()
fig4_path = os.path.join(OUT_DIR, 'week4_leadtime_sensitivity.png')
plt.savefig(fig4_path, dpi=130, bbox_inches='tight')
print(f'  [Chart saved: {fig4_path}]')
plt.close(fig4)


# ── FIGURE 5: Sensitivity — Horizon T vs Total Cost ──────────────────────────
horizons  = list(range(2, 13))
T_costs   = []
for T_val in horizons:
    r = solve_multiperiod(
        T=T_val, LT=1,
        demand=[20] * T_val,
        h_w=0.5, h_r=1.0, c_order=2.0,
        I_w0=50, I_r0=10,
        verbose=False,
    )
    T_costs.append(r['total_cost'])

fig5, ax5 = plt.subplots(figsize=(10, 5))
fig5.patch.set_facecolor(PALETTE['bg'])
ax5.set_facecolor(PALETTE['bg'])
ax5.plot(horizons, T_costs, marker='o', color=PALETTE['cost'],
         linewidth=2.5, markersize=8, label='Total Cost')
ax5.fill_between(horizons, T_costs, alpha=0.18, color=PALETTE['cost'])
for i, (T_val, cost) in enumerate(zip(horizons, T_costs)):
    if cost is not None:
        ax5.annotate(f'${cost:.0f}', (T_val, cost),
                     textcoords='offset points', xytext=(0, 8),
                     ha='center', fontsize=8, color='#555')
ax5.set_xlabel('Planning Horizon T (periods)', fontsize=11)
ax5.set_ylabel('Total Cost ($)', fontsize=11)
ax5.set_title('Week 4 — Sensitivity: Planning Horizon T vs Total Cost\n'
              '(LT=1, demand=20/period, I_w0=50, I_r0=10)',
              fontsize=12, fontweight='bold')
ax5.set_xticks(horizons)
ax5.legend(fontsize=10)
ax5.grid(color=PALETTE['grid'], linewidth=0.8)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
plt.tight_layout()
fig5_path = os.path.join(OUT_DIR, 'week4_horizon_sensitivity.png')
plt.savefig(fig5_path, dpi=130, bbox_inches='tight')
print(f'  [Chart saved: {fig5_path}]')
plt.close(fig5)


# ── FIGURE 6: Cost breakdown — T=10 base case ────────────────────────────────
T10 = 10
exp3_r = solve_multiperiod(
    T=T10, LT=1,
    demand=[15] * T10,
    h_w=0.5, h_r=1.0, c_order=2.0,
    I_w0=50, I_r0=10,
    verbose=False,
)
if exp3_r['status'] == 'Optimal':
    per_period_cost = [
        2.0 * exp3_r['o_w'][t] + 0.5 * exp3_r['I_w'][t] + 1.0 * exp3_r['I_r'][t]
        for t in range(T10)
    ]
    order_cost  = [2.0 * exp3_r['o_w'][t] for t in range(T10)]
    hold_w_cost = [0.5 * exp3_r['I_w'][t] for t in range(T10)]
    hold_r_cost = [1.0 * exp3_r['I_r'][t] for t in range(T10)]

    fig6, ax6 = plt.subplots(figsize=(12, 5))
    fig6.patch.set_facecolor(PALETTE['bg'])
    ax6.set_facecolor(PALETTE['bg'])
    T10_periods = list(range(T10))

    ax6.bar(T10_periods, order_cost,  label='Ordering cost',     color=PALETTE['order_w'],  alpha=0.85)
    ax6.bar(T10_periods, hold_w_cost, bottom=order_cost,         label='Holding cost (WH)',  color=PALETTE['warehouse'], alpha=0.85)
    ax6.bar(T10_periods, hold_r_cost,
            bottom=[a + b for a, b in zip(order_cost, hold_w_cost)],
            label='Holding cost (Retailer)', color=PALETTE['retailer'], alpha=0.85)

    ax6.set_xlabel('Period', fontsize=11)
    ax6.set_ylabel('Cost ($)', fontsize=11)
    ax6.set_title('Week 4 — Cost Breakdown per Period (T=10, LT=1, demand=15/period)',
                  fontsize=12, fontweight='bold')
    ax6.set_xticks(T10_periods)
    ax6.legend(fontsize=10)
    ax6.grid(axis='y', color=PALETTE['grid'], linewidth=0.8)
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    plt.tight_layout()
    fig6_path = os.path.join(OUT_DIR, 'week4_cost_breakdown.png')
    plt.savefig(fig6_path, dpi=130, bbox_inches='tight')
    print(f'  [Chart saved: {fig6_path}]')
    plt.close(fig6)


# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 62)
print('   All Week 4 experiments complete.')
print('   Key Concepts Demonstrated:')
print('   1. LpVariable.dicts() for T-indexed decision variables')
print('   2. lpSum() to build objective over all T periods')
print('   3. Lead-time constraint: received[t] = o_w[t - LT]')
print('   4. Constraint loop for all T periods (balance equations)')
print('   5. Experiments: vary LT, T, demand pattern, costs')
print('=' * 62)
