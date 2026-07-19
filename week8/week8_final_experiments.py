"""
week8_final_experiments.py
==========================
Week 8 — Final Experiments & Report

This is the capstone script. It brings together everything built across the
project into ONE resilience model and then stress-tests it against the
disruption cases named in the Plan of Action:
    • reduced supply
    • delayed replenishment
    • limited warehouse capacity

Model (a consolidation of Weeks 4–6)
  Two-echelon, multi-period, TWO retailers, with:
    • lead time                         (Week 4)
    • a warehouse storage-capacity limit + shadow prices   (Week 5)
    • a shortage penalty with recourse  (Week 6)  ← so a disruption shows up as
      measurable shortage & cost instead of plain infeasibility.

Decision variables per period (LpVariable.dicts):
  o_w[t]            — warehouse order from supplier
  o_r1[t], o_r2[t]  — shipments to retailer 1 / retailer 2
  I_w[t]            — warehouse ending inventory
  I_r1[t], I_r2[t]  — retailer ending inventories
  sh1[t], sh2[t]    — unmet demand (shortage) at each retailer

Objective: minimise
  Σ_t c_order*o_w[t] + h_w*I_w[t] + h_r1*I_r1[t] + h_r2*I_r2[t]
        + p_short*(sh1[t] + sh2[t])

Disruption levers (all optional — the base case turns them all off):
  supply_cap : list[T] or None — max units the supplier can deliver in period t
  delay_map  : dict {t: extra} — orders placed in period t arrive `extra` periods
               later than usual (temporary lead-time spike / delayed replenishment)
  CAP_STORE  : float           — warehouse storage capacity (tighten to disrupt)
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
    'short':     '#C44E52',
    'base':      '#4C72B0',
    'supply':    '#DD8452',
    'delay':     '#8172B2',
    'capacity':  '#C44E52',
    'combined':  '#937860',
    'ok':        '#55A868',
    'grid':      '#e8e8e8',
    'bg':        '#F9F9F9',
}


# ═══════════════════════════════════════════════════════════════════════════════
# CORE RESILIENCE SOLVER
# ═══════════════════════════════════════════════════════════════════════════════

def solve_resilience(
    T=8,
    LT=1,
    demand1=None,
    demand2=None,
    CAP_STORE=40,
    CAP_R=22,
    supply_cap=None,
    delay_map=None,
    h_w=0.5,
    h_r1=1.0,
    h_r2=1.0,
    c_order=2.0,
    p_short=15.0,
    I_w0=40,
    I_r1_0=12,
    I_r2_0=12,
    verbose=True,
    label='Base',
):
    """
    Solve the two-retailer, multi-period resilience LP, optionally under a
    disruption (reduced supply / delayed replenishment / tight capacity).

    Parameters
    ----------
    T          : int          — planning horizon
    LT         : int          — base lead time
    demand1/2  : list[T]       — retailer demands (defaults filled in below)
    CAP_STORE  : float         — warehouse storage capacity
    CAP_R      : float         — per-retailer storage capacity (limits pre-build,
                                 so a foreseen shock cannot be fully absorbed early)
    supply_cap : list[T]|None  — per-period cap on o_w[t] (None = unlimited)
    delay_map  : dict|None     — {t: extra}: orders placed at t arrive LT+extra
    h_w,h_r1,h_r2,c_order,p_short : costs
    I_w0,I_r1_0,I_r2_0 : opening inventories

    Returns
    -------
    dict with status, total_cost, cost breakdown, per-period arrays, shortages,
    service_level (% of demand met), and capacity shadow prices.
    """
    if demand1 is None:
        demand1 = [22, 24, 26, 23, 25, 21, 24, 22]
    if demand2 is None:
        demand2 = [16, 18, 17, 19, 15, 18, 16, 17]
    if delay_map is None:
        delay_map = {}

    m = pulp.LpProblem('Resilience', pulp.LpMinimize)

    o_w  = pulp.LpVariable.dicts('o_w',  range(T), lowBound=0)
    o_r1 = pulp.LpVariable.dicts('o_r1', range(T), lowBound=0)
    o_r2 = pulp.LpVariable.dicts('o_r2', range(T), lowBound=0)
    I_w  = pulp.LpVariable.dicts('I_w',  range(T), lowBound=0)
    I_r1 = pulp.LpVariable.dicts('I_r1', range(T), lowBound=0)
    I_r2 = pulp.LpVariable.dicts('I_r2', range(T), lowBound=0)
    sh1  = pulp.LpVariable.dicts('sh1',  range(T), lowBound=0)
    sh2  = pulp.LpVariable.dicts('sh2',  range(T), lowBound=0)

    # ── Objective ─────────────────────────────────────────────────────────────
    m += pulp.lpSum(
        c_order * o_w[t] + h_w * I_w[t] + h_r1 * I_r1[t] + h_r2 * I_r2[t]
        + p_short * (sh1[t] + sh2[t])
        for t in range(T)
    ), 'TotalCost'

    # ── Arrival schedule: which order lands in each period? ──────────────────
    # Order placed at period τ normally arrives at τ + LT. Delayed replenishment
    # pushes orders placed in disrupted periods further out by delay_map[τ].
    def received_in(t):
        terms = []
        for tau in range(T):
            arrival = tau + LT + delay_map.get(tau, 0)
            if arrival == t:
                terms.append(o_w[tau])
        return pulp.lpSum(terms) if terms else 0

    for t in range(T):
        prev_Iw  = I_w[t - 1]  if t > 0 else I_w0
        prev_Ir1 = I_r1[t - 1] if t > 0 else I_r1_0
        prev_Ir2 = I_r2[t - 1] if t > 0 else I_r2_0

        # Warehouse balance
        m += I_w[t] == prev_Iw + received_in(t) - o_r1[t] - o_r2[t], f'wh_bal_{t}'

        # Retailer balances with shortage recourse (unmet demand -> sh)
        m += I_r1[t] == prev_Ir1 + o_r1[t] - demand1[t] + sh1[t], f'r1_bal_{t}'
        m += I_r2[t] == prev_Ir2 + o_r2[t] - demand2[t] + sh2[t], f'r2_bal_{t}'

        # Warehouse storage capacity (named → shadow price available)
        m += I_w[t] <= CAP_STORE, f'storage_cap_t{t}'

        # Retailer storage capacity — small shops cannot hoard without limit,
        # so they cannot fully pre-stock ahead of a foreseen disruption.
        m += I_r1[t] <= CAP_R, f'r1_cap_t{t}'
        m += I_r2[t] <= CAP_R, f'r2_cap_t{t}'

        # Reduced-supply disruption: cap the order the supplier can fulfil
        if supply_cap is not None:
            m += o_w[t] <= supply_cap[t], f'supply_cap_t{t}'

    m.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[m.status]
    if m.status != 1:
        if verbose:
            print(f'\n  [{label}] Status: {status}')
        return {'status': status, 'total_cost': None}

    ow  = [o_w[t].varValue  for t in range(T)]
    or1 = [o_r1[t].varValue for t in range(T)]
    or2 = [o_r2[t].varValue for t in range(T)]
    Iw  = [I_w[t].varValue  for t in range(T)]
    Ir1 = [I_r1[t].varValue for t in range(T)]
    Ir2 = [I_r2[t].varValue for t in range(T)]
    s1  = [sh1[t].varValue  for t in range(T)]
    s2  = [sh2[t].varValue  for t in range(T)]

    order_cost = sum(c_order * ow[t] for t in range(T))
    hold_cost  = sum(h_w * Iw[t] + h_r1 * Ir1[t] + h_r2 * Ir2[t] for t in range(T))
    short_cost = sum(p_short * (s1[t] + s2[t]) for t in range(T))
    total      = pulp.value(m.objective)

    total_demand   = sum(demand1) + sum(demand2)
    total_short    = sum(s1) + sum(s2)
    service_level  = 100.0 * (total_demand - total_short) / total_demand

    shadow = [m.constraints[f'storage_cap_t{t}'].pi for t in range(T)]

    if verbose:
        print(f'\n{"─"*74}')
        print(f'  {label}')
        print(f'  T={T}  LT={LT}  CAP_STORE={CAP_STORE}  p_short={p_short}')
        if supply_cap is not None:
            print(f'  supply_cap : {supply_cap}')
        if delay_map:
            print(f'  delay_map  : {delay_map}  (order at t arrives LT+extra later)')
        print(f'  Status: {status}   Total Cost: ${total:.2f}')
        print(f'    ordering ${order_cost:.1f}  |  holding ${hold_cost:.1f}  '
              f'|  shortage ${short_cost:.1f}')
        print(f'  Total demand={total_demand}  total shortage={total_short:.1f}  '
              f'→ service level {service_level:.1f}%')
        print(f'  {"t":>3} {"o_w":>6} {"o_r1":>6} {"o_r2":>6} {"I_w":>6} '
              f'{"I_r1":>6} {"I_r2":>6} {"sh1":>5} {"sh2":>5} {"shadow":>7}')
        for t in range(T):
            print(f'  {t:>3} {ow[t]:>6.1f} {or1[t]:>6.1f} {or2[t]:>6.1f} '
                  f'{Iw[t]:>6.1f} {Ir1[t]:>6.1f} {Ir2[t]:>6.1f} '
                  f'{s1[t]:>5.1f} {s2[t]:>5.1f} {shadow[t]:>7.2f}')
        print(f'{"─"*74}')

    return {
        'status': status,
        'total_cost': total,
        'order_cost': order_cost,
        'hold_cost': hold_cost,
        'short_cost': short_cost,
        'o_w': ow, 'o_r1': or1, 'o_r2': or2,
        'I_w': Iw, 'I_r1': Ir1, 'I_r2': Ir2,
        'sh1': s1, 'sh2': s2,
        'total_short': total_short,
        'service_level': service_level,
        'shadow': shadow,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL EXPERIMENTS — DISRUPTION CASES
# ═══════════════════════════════════════════════════════════════════════════════

print('\n' + '=' * 74)
print('   WEEK 8 — Final Experiments: Supply-Chain Disruption Stress Tests')
print('=' * 74)

T = 8
D1 = [22, 24, 26, 23, 25, 21, 24, 22]
D2 = [16, 18, 17, 19, 15, 18, 16, 17]
COMMON = dict(T=T, LT=1, demand1=D1, demand2=D2, CAP_R=22,
              h_w=0.5, h_r1=1.0, h_r2=1.0, c_order=2.0, p_short=15.0,
              I_w0=40, I_r1_0=12, I_r2_0=12)

# Shared shock definitions (reused so the combined case is the union of shocks)
SUPPLY_SHOCK = [999, 999, 10, 10, 10, 999, 999, 999]   # supply cut in t=2..4
DELAY_SHOCK  = {1: 2, 2: 2}                             # orders at t=1,2 arrive +2 late
TIGHT_CAP    = 12                                       # warehouse squeezed

# ── BASE CASE — no disruption, healthy supply chain ──────────────────────────
print('\n[BASE] No disruption — the healthy supply chain')
base = solve_resilience(CAP_STORE=40, supply_cap=None, delay_map=None,
                        label='BASE — no disruption', **COMMON)

# ── DISRUPTION 1 — reduced supply in periods 2..4 ────────────────────────────
print('\n[D1] Reduced supply — supplier capped at 10 units/period during t=2..4')
d1 = solve_resilience(CAP_STORE=40, supply_cap=SUPPLY_SHOCK, delay_map=None,
                      label='D1 — reduced supply (t=2..4)', **COMMON)

# ── DISRUPTION 2 — delayed replenishment: orders at t=1,2 arrive 2 periods late ─
print('\n[D2] Delayed replenishment — orders placed at t=1,2 arrive 2 periods late')
d2 = solve_resilience(CAP_STORE=40, supply_cap=None,
                      delay_map=DELAY_SHOCK,
                      label='D2 — delayed replenishment', **COMMON)

# ── DISRUPTION 3 — limited warehouse capacity (tight) ────────────────────────
print('\n[D3] Limited warehouse capacity — CAP_STORE tightened 40 → 12')
d3 = solve_resilience(CAP_STORE=TIGHT_CAP, supply_cap=None, delay_map=None,
                      label='D3 — limited capacity (CAP_STORE=12)', **COMMON)

# ── DISRUPTION 4 — combined shock (all three at once) ────────────────────────
print('\n[D4] Combined shock — reduced supply + delayed replenishment + tight capacity')
d4 = solve_resilience(CAP_STORE=TIGHT_CAP, supply_cap=SUPPLY_SHOCK,
                      delay_map=DELAY_SHOCK,
                      label='D4 — combined shock', **COMMON)

cases = {
    'Base':                  base,
    'D1: reduced\nsupply':   d1,
    'D2: delayed\nreplen.':  d2,
    'D3: tight\ncapacity':   d3,
    'D4: combined\nshock':   d4,
}
case_colors = [PALETTE['base'], PALETTE['supply'], PALETTE['delay'],
               PALETTE['capacity'], PALETTE['combined']]


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════════════

periods = list(range(T))

# ── FIGURE 1: Cost breakdown & shortage across all disruption cases ──────────
fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5.5))
fig1.patch.set_facecolor(PALETTE['bg'])
fig1.suptitle('Week 8 — Cost & Shortage Across Disruption Scenarios',
              fontsize=14, fontweight='bold', y=1.01)

# Left: stacked cost breakdown
ax = axes1[0]
ax.set_facecolor(PALETTE['bg'])
names = list(cases.keys())
x = np.arange(len(names))
order_c = [c['order_cost'] for c in cases.values()]
hold_c  = [c['hold_cost']  for c in cases.values()]
short_c = [c['short_cost'] for c in cases.values()]
ax.bar(x, order_c, width=0.6, label='Ordering', color=PALETTE['warehouse'], alpha=0.88)
ax.bar(x, hold_c, bottom=order_c, width=0.6, label='Holding',
       color=PALETTE['retailer1'], alpha=0.88)
ax.bar(x, short_c, bottom=[a + b for a, b in zip(order_c, hold_c)], width=0.6,
       label='Shortage penalty', color=PALETTE['short'], alpha=0.88)
for xi, c in zip(x, cases.values()):
    ax.text(xi, c['total_cost'] + 3, f"${c['total_cost']:.0f}", ha='center',
            va='bottom', fontsize=9, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel('Total cost ($)', fontsize=11)
ax.set_title('Cost Breakdown (ordering / holding / shortage)', fontsize=11)
ax.legend(fontsize=9)
ax.grid(axis='y', color=PALETTE['grid'], linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Right: service level per case
ax = axes1[1]
ax.set_facecolor(PALETTE['bg'])
svc = [c['service_level'] for c in cases.values()]
bars = ax.bar(x, svc, width=0.6, color=case_colors, alpha=0.88)
for bar, s in zip(bars, svc):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f'{s:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.axhline(100, color=PALETTE['ok'], linestyle=':', linewidth=1.6,
           label='100% demand met')
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel('Service level (% demand met)', fontsize=11)
ax.set_ylim(min(svc) - 3, 102)
ax.set_title('Resilience: Service Level Under Each Disruption', fontsize=11)
ax.legend(fontsize=9, loc='lower left')
ax.grid(axis='y', color=PALETTE['grid'], linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
fig1_path = os.path.join(OUT_DIR, 'week8_disruption_costs.png')
plt.savefig(fig1_path, dpi=130, bbox_inches='tight')
print(f'\n  [Chart saved: {fig1_path}]')
plt.close(fig1)


# ── FIGURE 2: Timeline under the combined shock (D4) ─────────────────────────
fig2, axes2 = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
fig2.patch.set_facecolor(PALETTE['bg'])
fig2.suptitle('Week 8 — Inventory, Orders & Shortage Under the Combined Shock (D4)',
              fontsize=14, fontweight='bold', y=0.99)

# Panel 1: orders & warehouse receipts
ax = axes2[0]
ax.set_facecolor(PALETTE['bg'])
ax.bar([t - 0.2 for t in periods], d4['o_r1'], width=0.38,
       label='o_r1 (→ retailer 1)', color=PALETTE['retailer1'], alpha=0.85)
ax.bar([t + 0.2 for t in periods], d4['o_r2'], width=0.38,
       label='o_r2 (→ retailer 2)', color=PALETTE['retailer2'], alpha=0.85)
ax.plot(periods, d4['o_w'], marker='D', color=PALETTE['order_w'], linewidth=2,
        linestyle='--', label='o_w (warehouse order)', markersize=7)
ax.set_ylabel('Units', fontsize=10)
ax.set_title('Orders & Shipments (supply capped at t=2..4, deliveries delayed)',
             fontsize=10)
ax.legend(fontsize=8)
ax.grid(axis='y', color=PALETTE['grid'], linewidth=0.8)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Panel 2: inventory levels vs capacity
ax = axes2[1]
ax.set_facecolor(PALETTE['bg'])
ax.plot(periods, d4['I_w'],  marker='o', color=PALETTE['warehouse'], linewidth=2.2,
        label='I_w (warehouse)', markersize=7)
ax.plot(periods, d4['I_r1'], marker='s', color=PALETTE['retailer1'], linewidth=2.2,
        label='I_r1 (retailer 1)', markersize=7)
ax.plot(periods, d4['I_r2'], marker='^', color=PALETTE['retailer2'], linewidth=2.2,
        label='I_r2 (retailer 2)', markersize=7)
ax.axhline(TIGHT_CAP, color=PALETTE['capacity'], linestyle=':', linewidth=1.8,
           label=f'CAP_STORE = {TIGHT_CAP}')
ax.set_ylabel('Inventory (units)', fontsize=10)
ax.set_title('Inventory Levels (warehouse pinned against tight capacity)', fontsize=10)
ax.legend(fontsize=8)
ax.grid(color=PALETTE['grid'], linewidth=0.8)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Panel 3: shortage per period
ax = axes2[2]
ax.set_facecolor(PALETTE['bg'])
ax.bar([t - 0.2 for t in periods], d4['sh1'], width=0.38,
       label='Shortage retailer 1', color=PALETTE['retailer1'], alpha=0.9)
ax.bar([t + 0.2 for t in periods], d4['sh2'], width=0.38,
       label='Shortage retailer 2', color=PALETTE['retailer2'], alpha=0.9)
ax.set_xlabel('Period', fontsize=10)
ax.set_ylabel('Unmet demand', fontsize=10)
ax.set_title('Where the Chain Breaks: Shortage per Period', fontsize=10)
ax.set_xticks(periods)
ax.legend(fontsize=8)
ax.grid(axis='y', color=PALETTE['grid'], linewidth=0.8)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.97])
fig2_path = os.path.join(OUT_DIR, 'week8_combined_shock_timeline.png')
plt.savefig(fig2_path, dpi=130, bbox_inches='tight')
print(f'  [Chart saved: {fig2_path}]')
plt.close(fig2)


# ── FIGURE 3: Resilience frontier — how deep a supply cut can the chain absorb? ─
# Sweep the supply cap during the shock window (t=2..4) from generous down to a
# total cut, and watch service level fall & cost rise. The "knee" is the point
# beyond which the chain can no longer cover demand — its resilience threshold.
cut_levels = [40, 30, 25, 20, 15, 12, 10, 8, 5, 2, 0]
svc_curve, cost_curve = [], []
for cut in cut_levels:
    ss = [999, 999, cut, cut, cut, 999, 999, 999]
    r = solve_resilience(CAP_STORE=40, supply_cap=ss, delay_map=None,
                         **COMMON, verbose=False, label=f'cut={cut}')
    svc_curve.append(r['service_level'])
    cost_curve.append(r['total_cost'])

# Resilience threshold = the largest cut level that still keeps 100% service.
threshold = next((cut_levels[i] for i in range(len(cut_levels))
                  if svc_curve[i] < 99.99), cut_levels[-1])

fig3, ax3 = plt.subplots(figsize=(10, 5.5))
fig3.patch.set_facecolor(PALETTE['bg'])
ax3.set_facecolor(PALETTE['bg'])
ax3.plot(cut_levels, svc_curve, marker='o', color=PALETTE['ok'], linewidth=2.6,
         markersize=8, label='Service level (%)')
for c, s in zip(cut_levels, svc_curve):
    ax3.annotate(f'{s:.0f}%', (c, s), textcoords='offset points', xytext=(0, 9),
                 ha='center', fontsize=8, color='#3d7a4d')
ax3.axvline(threshold, color=PALETTE['capacity'], linestyle='--', linewidth=1.8,
            label=f'Resilience threshold (~{threshold} units/period)')
ax3.set_xlabel('Supplier capacity during shock window t=2..4  (units/period)',
               fontsize=11)
ax3.set_ylabel('Service level (% demand met)', fontsize=11, color=PALETTE['ok'])
ax3.tick_params(axis='y', labelcolor=PALETTE['ok'])
ax3.invert_xaxis()   # read left→right as the disruption WORSENS
ax3b = ax3.twinx()
ax3b.plot(cut_levels, cost_curve, marker='s', color=PALETTE['short'], linewidth=2.2,
          linestyle='--', markersize=7, label='Total cost ($)')
ax3b.set_ylabel('Total cost ($)', fontsize=11, color=PALETTE['short'])
ax3b.tick_params(axis='y', labelcolor=PALETTE['short'])
ax3.set_title('Week 8 — The Resilience Frontier: How Deep a Supply Cut Can the '
              'Chain Absorb?\n(reading left→right the supply cut deepens; service '
              'holds at 100% until the threshold, then falls)',
              fontsize=12, fontweight='bold')
ax3.set_xticks(cut_levels)
ax3.grid(color=PALETTE['grid'], linewidth=0.8)
ax3.spines['top'].set_visible(False); ax3b.spines['top'].set_visible(False)
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3b.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='center left')
plt.tight_layout()
fig3_path = os.path.join(OUT_DIR, 'week8_resilience_frontier.png')
plt.savefig(fig3_path, dpi=130, bbox_inches='tight')
print(f'  [Chart saved: {fig3_path}]')
plt.close(fig3)


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '─' * 74)
print('   FINAL DISRUPTION SUMMARY')
print('─' * 74)
print(f'  {"Scenario":<26} {"Cost ($)":>10} {"Shortage":>10} {"Service %":>10}')
print(f'  {"-"*26} {"-"*10} {"-"*10} {"-"*10}')
labels_clean = ['Base (no disruption)', 'D1 reduced supply',
                'D2 delayed replenishment', 'D3 limited capacity',
                'D4 combined shock']
for name, c in zip(labels_clean, cases.values()):
    print(f'  {name:<26} {c["total_cost"]:>10.1f} {c["total_short"]:>10.1f} '
          f'{c["service_level"]:>9.1f}%')
print('─' * 74)

print('\n' + '=' * 74)
print('   All Week 8 experiments complete.')
print('   Key Concepts Demonstrated:')
print('   1. One consolidated model = Weeks 4–6 combined (multi-period, 2')
print('      retailers, capacity + shadow prices, shortage recourse)')
print('   2. Disruption levers: reduced supply, delayed replenishment, tight cap')
print('   3. Metrics: cost breakdown, total shortage, service level')
print('   4. Combined shocks compound — service falls fastest when all hit at once')
print('   5. Resilience frontier: the chain absorbs shocks up to a threshold,')
print('      then some shortage becomes structural (unavoidable at any price)')
print('   See REPORT.md for the consolidated final report (Weeks 1–8).')
print('=' * 74)
