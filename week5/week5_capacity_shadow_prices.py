"""
week5_capacity_shadow_prices.py
===============================
Week 5 — Two Retailers, Capacity, and Shadow Prices
Two-echelon supply chain (Supplier → Warehouse → {Retailer 1, Retailer 2})
extended from Week 4 to add:
  • A SECOND retailer  (the warehouse now supplies two independent retailers)
  • Warehouse STORAGE CAPACITY:  I_w[t] <= CAP_STORE   for all t
  • SHADOW PRICE analysis on the capacity constraint (dual variables)

Decision variables per period (dictionary variables via LpVariable.dicts):
  o_w[t]   — units ordered by warehouse from supplier in period t
  o_r1[t]  — units shipped from warehouse to retailer 1 in period t
  o_r2[t]  — units shipped from warehouse to retailer 2 in period t
  I_w[t]   — warehouse   ending inventory in period t
  I_r1[t]  — retailer 1  ending inventory in period t
  I_r2[t]  — retailer 2  ending inventory in period t

Objective: minimise total cost over T periods
  Σ_t  c_order*o_w[t] + h_w*I_w[t] + h_r1*I_r1[t] + h_r2*I_r2[t]

Warehouse balance (one warehouse, TWO downstream shipments):
  I_w[t] = I_w[t-1] + received[t] - o_r1[t] - o_r2[t]
  where received[t] = o_w[t - LT]  if t - LT >= 0  else  0

Retailer balances (each retailer is independent):
  I_r1[t] = I_r1[t-1] + o_r1[t] - demand1[t]
  I_r2[t] = I_r2[t-1] + o_r2[t] - demand2[t]

Warehouse storage capacity (NAMED constraint so we can read its shadow price):
  I_w[t] <= CAP_STORE                       for all t   ('storage_cap_t{t}')

──────────────────────────────────────────────────────────────────────────────
SHADOW PRICE — how to read it
  A shadow price answers: "if I relax this constraint by one unit, how much does
  total cost change?"
    • shadow price == 0  → constraint is NOT binding. The warehouse never reaches
                           CAP_STORE, so extra capacity would not help.
    • shadow price != 0  → constraint IS binding (a genuine bottleneck). Adding one
                           more unit of capacity changes total cost by that amount.
  In PuLP the shadow price of a named constraint is  m.constraints['name'].pi
──────────────────────────────────────────────────────────────────────────────
"""

import os
import pulp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── Output directory ────────────────────────────────────────────────────────
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Colour palette (shared with earlier weeks) ──────────────────────────────
PALETTE = {
    'warehouse': '#4C72B0',
    'retailer1': '#55A868',
    'retailer2': '#8172B2',
    'order_w':   '#C44E52',
    'order_r':   '#DD8452',
    'cap':       '#C44E52',
    'shadow':    '#CCB974',
    'grid':      '#e8e8e8',
    'bg':        '#F9F9F9',
}


# ═══════════════════════════════════════════════════════════════════════════════
# CORE SOLVER
# ═══════════════════════════════════════════════════════════════════════════════

def solve_two_retailer(
    T=6,
    LT=1,
    demand1=None,
    demand2=None,
    CAP_STORE=60,
    h_w=0.5,
    h_r1=1.0,
    h_r2=1.0,
    c_order=2.0,
    I_w0=50,
    I_r1_0=10,
    I_r2_0=10,
    verbose=True,
    label='Base',
):
    """
    Solve the multi-period, two-retailer inventory LP with a warehouse
    storage-capacity constraint, and return shadow prices of that constraint.

    Parameters
    ----------
    T         : int   — number of planning periods
    LT        : int   — lead time (periods)
    demand1   : list  — retailer 1 demand per period (length T); uniform 20 if None
    demand2   : list  — retailer 2 demand per period (length T); uniform 15 if None
    CAP_STORE : float — warehouse maximum storage capacity (units, per period)
    h_w       : float — warehouse holding cost per unit per period
    h_r1      : float — retailer 1 holding cost per unit per period
    h_r2      : float — retailer 2 holding cost per unit per period
    c_order   : float — ordering cost per unit (warehouse ← supplier)
    I_w0      : float — warehouse opening inventory
    I_r1_0    : float — retailer 1 opening inventory
    I_r2_0    : float — retailer 2 opening inventory
    verbose   : bool  — whether to print results
    label     : str   — experiment label for printing

    Returns
    -------
    dict with keys:
        status, total_cost,
        o_w, o_r1, o_r2, I_w, I_r1, I_r2  (lists length T),
        shadow   (list length T — shadow price of storage_cap_t{t}),
        binding  (list length T — True where the capacity is binding)
    """
    if demand1 is None:
        demand1 = [20] * T
    if demand2 is None:
        demand2 = [15] * T

    m = pulp.LpProblem(f'TwoRetailer_LT{LT}_T{T}', pulp.LpMinimize)

    # ── Decision variables (dictionary variables) ────────────────────────────
    o_w  = pulp.LpVariable.dicts('o_w',  range(T), lowBound=0)   # warehouse orders
    o_r1 = pulp.LpVariable.dicts('o_r1', range(T), lowBound=0)   # ship → retailer 1
    o_r2 = pulp.LpVariable.dicts('o_r2', range(T), lowBound=0)   # ship → retailer 2
    I_w  = pulp.LpVariable.dicts('I_w',  range(T), lowBound=0)   # warehouse inventory
    I_r1 = pulp.LpVariable.dicts('I_r1', range(T), lowBound=0)   # retailer 1 inventory
    I_r2 = pulp.LpVariable.dicts('I_r2', range(T), lowBound=0)   # retailer 2 inventory

    # ── Objective: minimise total cost across all T periods ──────────────────
    m += pulp.lpSum(
        c_order * o_w[t] + h_w * I_w[t] + h_r1 * I_r1[t] + h_r2 * I_r2[t]
        for t in range(T)
    ), 'TotalCost'

    # ── Constraints loop ─────────────────────────────────────────────────────
    for t in range(T):
        # Lead-time: what arrives in period t was ordered LT periods ago
        received = o_w[t - LT] if t - LT >= 0 else 0

        prev_Iw  = I_w[t - 1]  if t > 0 else I_w0
        prev_Ir1 = I_r1[t - 1] if t > 0 else I_r1_0
        prev_Ir2 = I_r2[t - 1] if t > 0 else I_r2_0

        # Warehouse balance — one source feeding TWO retailers
        m += I_w[t] == prev_Iw + received - o_r1[t] - o_r2[t], f'wh_balance_{t}'

        # Retailer balances — completely independent of each other
        m += I_r1[t] == prev_Ir1 + o_r1[t] - demand1[t], f'r1_balance_{t}'
        m += I_r2[t] == prev_Ir2 + o_r2[t] - demand2[t], f'r2_balance_{t}'

        # Warehouse storage capacity — NAMED so its shadow price is readable
        m += I_w[t] <= CAP_STORE, f'storage_cap_t{t}'

    # ── Solve ────────────────────────────────────────────────────────────────
    m.solve(pulp.PULP_CBC_CMD(msg=0))

    status = pulp.LpStatus[m.status]
    if m.status != 1:
        if verbose:
            print(f'\n  [{label}] Status: {status} — no feasible solution found.')
        return {'status': status, 'total_cost': None, 'o_w': None,
                'o_r1': None, 'o_r2': None, 'I_w': None, 'I_r1': None,
                'I_r2': None, 'shadow': None, 'binding': None}

    ow_vals  = [o_w[t].varValue  for t in range(T)]
    or1_vals = [o_r1[t].varValue for t in range(T)]
    or2_vals = [o_r2[t].varValue for t in range(T)]
    Iw_vals  = [I_w[t].varValue  for t in range(T)]
    Ir1_vals = [I_r1[t].varValue for t in range(T)]
    Ir2_vals = [I_r2[t].varValue for t in range(T)]
    total    = pulp.value(m.objective)

    # ── Read shadow prices of the capacity constraints ───────────────────────
    # m.constraints['storage_cap_t{t}'].pi  is the dual value.
    shadow  = [m.constraints[f'storage_cap_t{t}'].pi for t in range(T)]
    # A period is "binding" when the warehouse actually sits at CAP_STORE.
    binding = [abs(Iw_vals[t] - CAP_STORE) < 1e-6 for t in range(T)]

    if verbose:
        print(f'\n{"─"*78}')
        print(f'  Experiment : {label}')
        print(f'  T={T}  Lead Time={LT}  CAP_STORE={CAP_STORE}  c_order={c_order}')
        print(f'  h_w={h_w}  h_r1={h_r1}  h_r2={h_r2}')
        print(f'  Opening: I_w0={I_w0}  I_r1_0={I_r1_0}  I_r2_0={I_r2_0}')
        print(f'  Demand r1: {demand1}')
        print(f'  Demand r2: {demand2}')
        print(f'  Status : {status}')
        print(f'  Total Cost: ${total:.2f}')
        print(f'  {"Period":<7} {"o_w":>6} {"o_r1":>6} {"o_r2":>6} '
              f'{"I_w":>6} {"I_r1":>6} {"I_r2":>6} {"shadow":>8} {"binding":>8}')
        print(f'  {"-"*7} {"-"*6} {"-"*6} {"-"*6} {"-"*6} {"-"*6} {"-"*6} '
              f'{"-"*8} {"-"*8}')
        for t in range(T):
            print(f'  t={t:<5} {ow_vals[t]:>6.1f} {or1_vals[t]:>6.1f} '
                  f'{or2_vals[t]:>6.1f} {Iw_vals[t]:>6.1f} {Ir1_vals[t]:>6.1f} '
                  f'{Ir2_vals[t]:>6.1f} {shadow[t]:>8.3f} '
                  f'{"YES" if binding[t] else "no":>8}')
        n_bind = sum(binding)
        print(f'  Capacity binding in {n_bind} of {T} periods.')
        if n_bind == 0:
            print('  → Shadow prices are 0 everywhere: capacity is NOT a bottleneck.')
        else:
            print('  → Non-zero shadow prices: capacity IS a bottleneck in those periods.')
        print(f'{"─"*78}')

    return {
        'status': status,
        'total_cost': total,
        'o_w': ow_vals,
        'o_r1': or1_vals,
        'o_r2': or2_vals,
        'I_w': Iw_vals,
        'I_r1': Ir1_vals,
        'I_r2': Ir2_vals,
        'shadow': shadow,
        'binding': binding,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════════════════

print('\n' + '=' * 78)
print('   WEEK 5 — Two Retailers, Warehouse Capacity, and Shadow Prices')
print('=' * 78)

# Demand patterns used throughout
D1 = [20, 22, 25, 20, 18, 24]   # retailer 1
D2 = [15, 18, 20, 16, 14, 19]   # retailer 2

# ── BASE CASE: generous capacity — capacity should NOT bind ───────────────────
print('\n[BASE CASE] Generous capacity CAP_STORE=60 — expect NO binding')
base = solve_two_retailer(
    T=6, LT=1, demand1=D1, demand2=D2,
    CAP_STORE=60, h_w=0.5, h_r1=1.0, h_r2=1.0, c_order=2.0,
    I_w0=50, I_r1_0=10, I_r2_0=10,
    verbose=True, label='Base (CAP_STORE=60)',
)

# ── EXPERIMENT 1: tighten the capacity — force it to bind ─────────────────────
print('\n[EXP 1] Tighten capacity to CAP_STORE=30 — expect binding & shadow prices')
exp1 = solve_two_retailer(
    T=6, LT=1, demand1=D1, demand2=D2,
    CAP_STORE=30, h_w=0.5, h_r1=1.0, h_r2=1.0, c_order=2.0,
    I_w0=50, I_r1_0=10, I_r2_0=10,
    verbose=True, label='Exp1: CAP_STORE=30 (tight)',
)

# ── EXPERIMENT 2: even tighter capacity ───────────────────────────────────────
print('\n[EXP 2] Very tight capacity CAP_STORE=20')
exp2 = solve_two_retailer(
    T=6, LT=1, demand1=D1, demand2=D2,
    CAP_STORE=20, h_w=0.5, h_r1=1.0, h_r2=1.0, c_order=2.0,
    I_w0=50, I_r1_0=10, I_r2_0=10,
    verbose=True, label='Exp2: CAP_STORE=20 (very tight)',
)

# ── EXPERIMENT 3: change h_r2 — make retailer 2 expensive to hold ────────────
print('\n[EXP 3] Change h_r2: retailer 2 holding cost 1.0 → 3.0')
exp3 = solve_two_retailer(
    T=6, LT=1, demand1=D1, demand2=D2,
    CAP_STORE=30, h_w=0.5, h_r1=1.0, h_r2=3.0, c_order=2.0,
    I_w0=50, I_r1_0=10, I_r2_0=10,
    verbose=True, label='Exp3: h_r2=3.0 (tight cap)',
)

# ── EXPERIMENT 4: change lead time — LT=2 with tight capacity ────────────────
print('\n[EXP 4] Change Lead Time: LT=1 → LT=2 (tight capacity)')
exp4 = solve_two_retailer(
    T=6, LT=2, demand1=D1, demand2=D2,
    CAP_STORE=30, h_w=0.5, h_r1=1.0, h_r2=1.0, c_order=2.0,
    I_w0=50, I_r1_0=10, I_r2_0=10,
    verbose=True, label='Exp4: LT=2 (tight cap)',
)


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════════════

T_base  = 6
periods = list(range(T_base))

# ── FIGURE 1: Two-retailer inventory timeline (base case) ────────────────────
fig1, axes1 = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
fig1.patch.set_facecolor(PALETTE['bg'])
fig1.suptitle('Week 5 — Two Retailers Served by One Warehouse (Base Case)',
              fontsize=14, fontweight='bold', y=0.98)

# Panel 1: shipments to each retailer
ax = axes1[0]
ax.set_facecolor(PALETTE['bg'])
ax.bar([t - 0.2 for t in periods], base['o_r1'], width=0.38,
       label='o_r1 (→ retailer 1)', color=PALETTE['retailer1'], alpha=0.85)
ax.bar([t + 0.2 for t in periods], base['o_r2'], width=0.38,
       label='o_r2 (→ retailer 2)', color=PALETTE['retailer2'], alpha=0.85)
ax.plot(periods, base['o_w'], marker='D', color=PALETTE['order_w'],
        linewidth=2, label='o_w (warehouse order)', markersize=7, linestyle='--')
ax.set_ylabel('Units', fontsize=10)
ax.set_title('Warehouse Order & Shipments to Each Retailer', fontsize=10, pad=4)
ax.legend(fontsize=9)
ax.grid(axis='y', color=PALETTE['grid'], linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel 2: inventory levels
ax = axes1[1]
ax.set_facecolor(PALETTE['bg'])
ax.plot(periods, base['I_w'],  marker='o', color=PALETTE['warehouse'],
        linewidth=2.2, label='I_w (warehouse)', markersize=7)
ax.plot(periods, base['I_r1'], marker='s', color=PALETTE['retailer1'],
        linewidth=2.2, label='I_r1 (retailer 1)', markersize=7)
ax.plot(periods, base['I_r2'], marker='^', color=PALETTE['retailer2'],
        linewidth=2.2, label='I_r2 (retailer 2)', markersize=7)
ax.axhline(60, color=PALETTE['cap'], linestyle=':', linewidth=1.8,
           label='CAP_STORE = 60')
ax.set_xlabel('Period', fontsize=10)
ax.set_ylabel('Inventory (units)', fontsize=10)
ax.set_title('Inventory Levels per Period', fontsize=10, pad=4)
ax.set_xticks(periods)
ax.legend(fontsize=9)
ax.grid(color=PALETTE['grid'], linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig1_path = os.path.join(OUT_DIR, 'week5_two_retailer_timeline.png')
plt.savefig(fig1_path, dpi=130, bbox_inches='tight')
print(f'\n  [Chart saved: {fig1_path}]')
plt.close(fig1)


# ── FIGURE 2: Capacity binding & shadow prices (tight case, Exp1) ────────────
fig2, axes2 = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
fig2.patch.set_facecolor(PALETTE['bg'])
fig2.suptitle('Week 5 — Capacity Constraint & Shadow Prices (CAP_STORE = 30)',
              fontsize=14, fontweight='bold', y=0.98)

# Panel 1: warehouse inventory vs the capacity line
ax = axes2[0]
ax.set_facecolor(PALETTE['bg'])
bar_colors = [PALETTE['cap'] if b else PALETTE['warehouse'] for b in exp1['binding']]
ax.bar(periods, exp1['I_w'], width=0.55, color=bar_colors, alpha=0.85,
       label='I_w (warehouse)')
ax.axhline(30, color=PALETTE['cap'], linestyle='--', linewidth=2,
           label='CAP_STORE = 30')
for t in periods:
    if exp1['binding'][t]:
        ax.text(t, exp1['I_w'][t] + 0.5, 'binding', ha='center', va='bottom',
                fontsize=8, color=PALETTE['cap'], fontweight='bold')
ax.set_ylabel('Warehouse inventory', fontsize=10)
ax.set_title('Warehouse Inventory vs Capacity  (red = capacity binding)',
             fontsize=10, pad=4)
ax.legend(fontsize=9)
ax.grid(axis='y', color=PALETTE['grid'], linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel 2: shadow prices per period
ax = axes2[1]
ax.set_facecolor(PALETTE['bg'])
shadow_abs = [abs(s) for s in exp1['shadow']]
bars = ax.bar(periods, shadow_abs, width=0.55, color=PALETTE['shadow'],
              edgecolor='white', linewidth=1.2)
for bar, s in zip(bars, exp1['shadow']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{s:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_xlabel('Period', fontsize=10)
ax.set_ylabel('|Shadow price|', fontsize=10)
ax.set_title('Shadow Price of the Storage-Capacity Constraint per Period\n'
             '(0 → not binding;  non-zero → one more unit of capacity saves this much)',
             fontsize=10, pad=4)
ax.set_xticks(periods)
ax.grid(axis='y', color=PALETTE['grid'], linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig2_path = os.path.join(OUT_DIR, 'week5_shadow_prices.png')
plt.savefig(fig2_path, dpi=130, bbox_inches='tight')
print(f'  [Chart saved: {fig2_path}]')
plt.close(fig2)


# ── FIGURE 3: Capacity sweep — total cost & total shadow price vs CAP_STORE ───
caps        = list(range(15, 61, 5))
sweep_cost  = []
sweep_shadow = []   # sum of |shadow| across periods
for cap in caps:
    r = solve_two_retailer(
        T=6, LT=1, demand1=D1, demand2=D2,
        CAP_STORE=cap, h_w=0.5, h_r1=1.0, h_r2=1.0, c_order=2.0,
        I_w0=50, I_r1_0=10, I_r2_0=10,
        verbose=False,
    )
    sweep_cost.append(r['total_cost'])
    sweep_shadow.append(sum(abs(s) for s in r['shadow']) if r['shadow'] else None)

fig3, ax3 = plt.subplots(figsize=(10, 5.5))
fig3.patch.set_facecolor(PALETTE['bg'])
ax3.set_facecolor(PALETTE['bg'])
ax3.plot(caps, sweep_cost, marker='o', color=PALETTE['warehouse'],
         linewidth=2.5, markersize=7, label='Total cost ($)')
for cap, cost in zip(caps, sweep_cost):
    if cost is not None:
        ax3.annotate(f'${cost:.0f}', (cap, cost), textcoords='offset points',
                     xytext=(0, 8), ha='center', fontsize=8, color='#555')
ax3.set_xlabel('Warehouse storage capacity CAP_STORE (units)', fontsize=11)
ax3.set_ylabel('Total cost ($)', fontsize=11, color=PALETTE['warehouse'])
ax3.tick_params(axis='y', labelcolor=PALETTE['warehouse'])

ax3b = ax3.twinx()
ax3b.bar(caps, sweep_shadow, width=2.4, color=PALETTE['shadow'], alpha=0.45,
         label='Σ |shadow price|')
ax3b.set_ylabel('Σ |shadow price| across periods', fontsize=11,
                color='#9c8a3d')
ax3b.tick_params(axis='y', labelcolor='#9c8a3d')

ax3.set_title('Week 5 — Sensitivity: Tightening Capacity Raises Cost & Shadow Prices\n'
              '(as CAP_STORE shrinks, the constraint binds and dual values grow)',
              fontsize=12, fontweight='bold')
ax3.set_xticks(caps)
ax3.grid(color=PALETTE['grid'], linewidth=0.8)
ax3.spines['top'].set_visible(False)
ax3b.spines['top'].set_visible(False)
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3b.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper right')
plt.tight_layout()
fig3_path = os.path.join(OUT_DIR, 'week5_capacity_sweep.png')
plt.savefig(fig3_path, dpi=130, bbox_inches='tight')
print(f'  [Chart saved: {fig3_path}]')
plt.close(fig3)


# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 78)
print('   All Week 5 experiments complete.')
print('   Key Concepts Demonstrated:')
print('   1. Second retailer: independent balance equation per retailer')
print('   2. Warehouse feeds two retailers: I_w -= o_r1 + o_r2')
print('   3. Named capacity constraint  I_w[t] <= CAP_STORE  ("storage_cap_t{t}")')
print('   4. Shadow prices via m.constraints[name].pi')
print('   5. Experiments: tighten capacity, change h_r2, change lead time')
print('   6. Reading duals: 0 → slack, non-zero → binding bottleneck')
print('=' * 78)
