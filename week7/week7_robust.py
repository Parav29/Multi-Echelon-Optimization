"""
week7_robust.py
===============
Week 7 — Robust Optimization

Stochastic programming (Week 6) needed probabilities for each scenario. But where
do those probabilities come from? For a new business, or shifting demand, reliable
probabilities may not exist. Robust optimization needs only the RANGE demand can
take — no probabilities — and it protects against the WORST case within that range.

The uncertainty set (box)
  For each period t, demand can be any value in
      d_t ∈ [ d_bar_t − d_hat_t ,  d_bar_t + d_hat_t ]
  d_bar_t is the nominal demand;  d_hat_t is the maximum deviation either side.

The Gamma parameter (budget of uncertainty, Bertsimas–Sim)
  Protecting against EVERY period hitting its worst case at once is far too
  expensive and very unlikely. Gamma ∈ [0, T] caps how many periods may be at
  their worst case simultaneously:
      Gamma = 0  → all periods at nominal   (same as deterministic / optimistic)
      Gamma = k  → at most k periods at worst case            (balanced)
      Gamma = T  → every period at worst case  (fully conservative, most costly)

Robust "no-stockout" (safety) constraint
  The retailer must never stock out under any demand allowed by the budget. Up to
  period t the cumulative shipment (plus opening stock) must cover the worst-case
  cumulative demand:
      I_r0 + Σ_{τ≤t} o_r[τ]  ≥  Σ_{τ≤t} d_bar[τ]  +  protection_t(Gamma)
  where protection_t(Gamma) is the largest possible extra demand the budget allows
  over periods 0..t = the sum of the  min(Gamma, t+1)  largest deviations d_hat.
  Because d_hat and Gamma are parameters, protection_t is a constant, so the model
  stays a clean linear program.

The price of robustness curve
  Solve the robust LP for every Gamma from 0 to T and plot optimal cost. The curve
  is typically flat at low Gamma (cheap protection) and steep at high Gamma
  (expensive protection). The ELBOW is the natural Gamma: most of the protection
  for a fraction of the maximum cost.

Deliverable:
  • Run the Gamma sweep and print cost at each Gamma from 0 to T
  • Find the elbow, recommend a Gamma, and report the extra cost vs Gamma = 0
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
    'retailer':  '#55A868',
    'nominal':   '#4C72B0',
    'worst':     '#C44E52',
    'cost':      '#CCB974',
    'elbow':     '#C44E52',
    'det':       '#55A868',
    'robust':    '#C44E52',
    'grid':      '#e8e8e8',
    'bg':        '#F9F9F9',
}


# ═══════════════════════════════════════════════════════════════════════════════
# PROTECTION FUNCTION (Bertsimas–Sim budget over a box uncertainty set)
# ═══════════════════════════════════════════════════════════════════════════════

def cumulative_protection(d_hat, Gamma):
    """
    protection_t(Gamma) for every t = the worst-case EXTRA cumulative demand that
    a budget of Gamma allows over periods 0..t.

    With at most Gamma periods at worst case, the worst extra demand up to period t
    is the sum of the  min(Gamma, t+1)  largest deviations among d_hat[0..t].
    (Gamma may be fractional: the last, partial period contributes a fraction.)

    Returns a list prot of length T where prot[t] is the protection for the
    cumulative constraint ending at period t.
    """
    T = len(d_hat)
    prot = []
    for t in range(T):
        devs = sorted(d_hat[:t + 1], reverse=True)   # largest deviations first
        g = min(Gamma, t + 1)
        full = int(np.floor(g))
        frac = g - full
        total = sum(devs[:full])
        if frac > 1e-9 and full < len(devs):
            total += frac * devs[full]               # partial period contribution
        prot.append(total)
    return prot


# ═══════════════════════════════════════════════════════════════════════════════
# ROBUST SOLVER
# ═══════════════════════════════════════════════════════════════════════════════

def solve_robust(
    d_bar,
    d_hat,
    Gamma,
    T=None,
    LT=1,
    h_w=0.5,
    h_r=1.0,
    c_order=2.0,
    I_w0=80,
    I_r0=20,
    verbose=False,
    label='Robust',
):
    """
    Robust two-echelon inventory LP with a box uncertainty set and a Gamma budget.

    Decision variables (nominal plan, feasible for the whole uncertainty set):
      o_w[t]  — warehouse order            I_w[t] — warehouse inventory
      o_r[t]  — shipment to retailer       I_r[t] — retailer inventory (nominal)

    Robust safety constraint (no stockout under the budget):
      I_r0 + Σ_{τ≤t} o_r[τ]  ≥  Σ_{τ≤t} d_bar[τ] + protection_t(Gamma)

    Cost is evaluated on the nominal demand path (holding + ordering); the robust
    constraints force enough early shipment to survive the worst case, which is
    what raises cost as Gamma grows.

    Returns dict: status, total_cost, o_w, o_r, I_w, I_r, prot
    """
    if T is None:
        T = len(d_bar)
    prot = cumulative_protection(d_hat, Gamma)

    m = pulp.LpProblem(f'Robust_G{Gamma}', pulp.LpMinimize)

    o_w = pulp.LpVariable.dicts('o_w', range(T), lowBound=0)
    o_r = pulp.LpVariable.dicts('o_r', range(T), lowBound=0)
    I_w = pulp.LpVariable.dicts('I_w', range(T), lowBound=0)
    I_r = pulp.LpVariable.dicts('I_r', range(T), lowBound=0)

    # Objective: ordering + holding cost on the nominal path
    m += pulp.lpSum(c_order * o_w[t] + h_w * I_w[t] + h_r * I_r[t]
                    for t in range(T)), 'TotalCost'

    for t in range(T):
        received = o_w[t - LT] if t - LT >= 0 else 0
        prev_Iw  = I_w[t - 1] if t > 0 else I_w0
        prev_Ir  = I_r[t - 1] if t > 0 else I_r0

        # Warehouse balance
        m += I_w[t] == prev_Iw + received - o_r[t], f'wh_balance_{t}'
        # Retailer nominal balance (for cost accounting)
        m += I_r[t] == prev_Ir + o_r[t] - d_bar[t], f'ret_balance_{t}'

        # Robust no-stockout: cumulative shipment covers worst-case cumulative demand
        cum_ship   = I_r0 + pulp.lpSum(o_r[tau] for tau in range(t + 1))
        cum_demand = sum(d_bar[tau] for tau in range(t + 1)) + prot[t]
        m += cum_ship >= cum_demand, f'robust_safety_{t}'

    m.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[m.status]
    if m.status != 1:
        if verbose:
            print(f'  [{label}] Gamma={Gamma}: {status}')
        return {'status': status, 'total_cost': None, 'o_w': None, 'o_r': None,
                'I_w': None, 'I_r': None, 'prot': prot}

    result = {
        'status': status,
        'total_cost': pulp.value(m.objective),
        'o_w': [o_w[t].varValue for t in range(T)],
        'o_r': [o_r[t].varValue for t in range(T)],
        'I_w': [I_w[t].varValue for t in range(T)],
        'I_r': [I_r[t].varValue for t in range(T)],
        'prot': prot,
    }
    if verbose:
        print(f'  [{label}] Gamma={Gamma:>4}: cost=${result["total_cost"]:.2f}  '
              f'protection={[round(p, 1) for p in prot]}')
    return result


def find_elbow(gammas, costs):
    """
    Find the elbow of the price-of-robustness curve using the maximum-distance
    (Kneedle-style) heuristic: the point on the curve farthest from the straight
    line joining the first and last points.

    Returns (elbow_index, elbow_gamma).
    """
    x = np.array(gammas, dtype=float)
    y = np.array(costs, dtype=float)
    x0, y0, x1, y1 = x[0], y[0], x[-1], y[-1]
    # Perpendicular distance of each point from the line (x0,y0)-(x1,y1)
    denom = np.hypot(x1 - x0, y1 - y0)
    if denom < 1e-12:
        return 0, gammas[0]
    dist = np.abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0) / denom
    idx = int(np.argmax(dist))
    return idx, gammas[idx]


# ═══════════════════════════════════════════════════════════════════════════════
# RUN — GAMMA SWEEP
# ═══════════════════════════════════════════════════════════════════════════════

print('\n' + '=' * 72)
print('   WEEK 7 — Robust Optimization (Box Uncertainty + Gamma Budget)')
print('=' * 72)

# Problem data
T      = 8
d_bar  = [30, 32, 28, 35, 31, 29, 33, 30]   # nominal demand per period
d_hat  = [ 8,  6,  7,  9,  8,  5,  7,  6]    # max deviation per period
params = dict(LT=1, h_w=0.5, h_r=1.0, c_order=2.0, I_w0=80, I_r0=20)

print(f'\n  Horizon T = {T}')
print(f'  Nominal demand d_bar : {d_bar}')
print(f'  Max deviation d_hat  : {d_hat}')
print(f'  Demand ranges [d_bar - d_hat, d_bar + d_hat]:')
for t in range(T):
    print(f'     t={t}: [{d_bar[t] - d_hat[t]:>3}, {d_bar[t] + d_hat[t]:>3}]')

print('\n  --- Gamma sweep: cost at each Gamma from 0 to T ---')
gammas = list(range(0, T + 1))
costs  = []
results = {}
for G in gammas:
    r = solve_robust(d_bar, d_hat, G, T=T, verbose=True, label='sweep', **params)
    results[G] = r
    costs.append(r['total_cost'])

# ── Elbow analysis ────────────────────────────────────────────────────────────
elbow_idx, elbow_gamma = find_elbow(gammas, costs)
cost_g0    = costs[0]
cost_gT    = costs[-1]
cost_elbow = costs[elbow_idx]
extra_elbow = cost_elbow - cost_g0
extra_full  = cost_gT - cost_g0

print('\n' + '─' * 72)
print('   PRICE OF ROBUSTNESS — SUMMARY')
print('─' * 72)
print(f'  {"Gamma":>6} {"Cost ($)":>12} {"Extra vs G=0":>14} {"% of full range":>16}')
for G, c in zip(gammas, costs):
    extra = c - cost_g0
    pct = (extra / extra_full * 100) if extra_full > 1e-9 else 0.0
    tag = '  <-- ELBOW' if G == elbow_gamma else ''
    print(f'  {G:>6} {c:>12.2f} {extra:>14.2f} {pct:>15.1f}%{tag}')
print('─' * 72)
print(f'  Gamma = 0 (deterministic)  : ${cost_g0:.2f}')
print(f'  Gamma = T (fully robust)   : ${cost_gT:.2f}   '
      f'(+${extra_full:.2f} over deterministic)')
print(f'  RECOMMENDED Gamma (elbow)  : {elbow_gamma}   cost ${cost_elbow:.2f}')
print(f'    → adds only ${extra_elbow:.2f} over Gamma=0 '
      f'({(extra_elbow / extra_full * 100) if extra_full > 1e-9 else 0:.1f}% of the '
      f'full price of robustness)')
print(f'    → yet protects against up to {elbow_gamma} of {T} periods hitting '
      f'their worst case simultaneously.')
print('─' * 72)


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════════════

periods = list(range(T))

# ── FIGURE 1: The price of robustness curve with the elbow marked ────────────
fig1, ax1 = plt.subplots(figsize=(10, 6))
fig1.patch.set_facecolor(PALETTE['bg'])
ax1.set_facecolor(PALETTE['bg'])
ax1.plot(gammas, costs, marker='o', color=PALETTE['cost'], linewidth=2.6,
         markersize=8, label='Optimal robust cost')
ax1.fill_between(gammas, costs, min(costs) - 2, alpha=0.15, color=PALETTE['cost'])
# straight reference line from first to last point
ax1.plot([gammas[0], gammas[-1]], [costs[0], costs[-1]], linestyle='--',
         color='#999', linewidth=1.4, label='Chord (endpoints)')
# elbow marker
ax1.scatter([elbow_gamma], [cost_elbow], s=190, color=PALETTE['elbow'],
            zorder=5, edgecolor='white', linewidth=1.6,
            label=f'Elbow  (Gamma = {elbow_gamma})')
ax1.annotate(f'Elbow: Gamma={elbow_gamma}\n${cost_elbow:.0f}  (+${extra_elbow:.0f})',
             (elbow_gamma, cost_elbow), textcoords='offset points',
             xytext=(15, -35), fontsize=10, fontweight='bold',
             color=PALETTE['elbow'],
             arrowprops=dict(arrowstyle='->', color=PALETTE['elbow']))
for G, c in zip(gammas, costs):
    ax1.annotate(f'${c:.0f}', (G, c), textcoords='offset points',
                 xytext=(0, 10), ha='center', fontsize=8, color='#555')
ax1.set_xlabel('Gamma  (budget of uncertainty — periods allowed at worst case)',
               fontsize=11)
ax1.set_ylabel('Optimal total cost ($)', fontsize=11)
ax1.set_title('Week 7 — The Price of Robustness Curve\n'
              'flat & cheap at low Gamma, steep & expensive near Gamma = T',
              fontsize=13, fontweight='bold')
ax1.set_xticks(gammas)
ax1.legend(fontsize=9, loc='upper left')
ax1.grid(color=PALETTE['grid'], linewidth=0.8)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
plt.tight_layout()
fig1_path = os.path.join(OUT_DIR, 'week7_price_of_robustness.png')
plt.savefig(fig1_path, dpi=130, bbox_inches='tight')
print(f'\n  [Chart saved: {fig1_path}]')
plt.close(fig1)


# ── FIGURE 2: Deterministic vs Robust — demand band & inventory plans ────────
fig2, axes2 = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
fig2.patch.set_facecolor(PALETTE['bg'])
fig2.suptitle('Week 7 — Deterministic (Gamma=0) vs Robust (elbow Gamma) Plans',
              fontsize=14, fontweight='bold', y=0.98)

# Panel 1: demand band (nominal ± deviation)
ax = axes2[0]
ax.set_facecolor(PALETTE['bg'])
upper = [d_bar[t] + d_hat[t] for t in periods]
lower = [d_bar[t] - d_hat[t] for t in periods]
ax.fill_between(periods, lower, upper, alpha=0.20, color=PALETTE['worst'],
                label='Uncertainty band [d_bar ± d_hat]')
ax.plot(periods, d_bar, marker='o', color=PALETTE['nominal'], linewidth=2.4,
        label='Nominal demand d_bar', markersize=7)
ax.plot(periods, upper, linestyle='--', color=PALETTE['worst'], linewidth=1.6,
        label='Worst-case (upper)')
ax.set_ylabel('Demand (units)', fontsize=10)
ax.set_title('Box Uncertainty Set: demand can land anywhere in the band',
             fontsize=10, pad=4)
ax.legend(fontsize=9)
ax.grid(color=PALETTE['grid'], linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel 2: retailer inventory under det (G=0) vs robust (elbow)
ax = axes2[1]
ax.set_facecolor(PALETTE['bg'])
det_r   = results[0]
rob_r   = results[elbow_gamma]
ax.plot(periods, det_r['I_r'], marker='s', color=PALETTE['det'], linewidth=2.4,
        label=f'Retailer inventory — Gamma=0 (cost ${det_r["total_cost"]:.0f})',
        markersize=7)
ax.plot(periods, rob_r['I_r'], marker='^', color=PALETTE['robust'], linewidth=2.4,
        label=f'Retailer inventory — Gamma={elbow_gamma} '
              f'(cost ${rob_r["total_cost"]:.0f})', markersize=8)
ax.fill_between(periods, det_r['I_r'], alpha=0.10, color=PALETTE['det'])
ax.fill_between(periods, rob_r['I_r'], alpha=0.10, color=PALETTE['robust'])
ax.set_xlabel('Period', fontsize=10)
ax.set_ylabel('Retailer inventory (units)', fontsize=10)
ax.set_title('Robust plan carries extra safety stock to survive the worst case',
             fontsize=10, pad=4)
ax.set_xticks(periods)
ax.legend(fontsize=9)
ax.grid(color=PALETTE['grid'], linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig2_path = os.path.join(OUT_DIR, 'week7_det_vs_robust.png')
plt.savefig(fig2_path, dpi=130, bbox_inches='tight')
print(f'  [Chart saved: {fig2_path}]')
plt.close(fig2)


# ── FIGURE 3: Three-way comparison — Deterministic / Stochastic / Robust ─────
# A simple worst-case stress test: evaluate each plan's stockout risk if demand
# jumps to its upper bound in every period.
def worst_case_shortfall(o_r, o_r_offset_I_r0, d_upper):
    """Total shortfall if demand equals d_upper every period, given a shipment
    plan o_r and opening retailer stock I_r0. Positive = would stock out."""
    inv = o_r_offset_I_r0
    short = 0.0
    cum_ship = 0.0
    cum_dem = 0.0
    for t in range(len(o_r)):
        cum_ship += o_r[t]
        cum_dem += d_upper[t]
        avail = o_r_offset_I_r0 + cum_ship
        if avail < cum_dem:
            short = max(short, cum_dem - avail)
    return short


d_upper = [d_bar[t] + d_hat[t] for t in periods]
plans = {
    'Deterministic\n(Gamma=0)': results[0],
    f'Balanced robust\n(Gamma={elbow_gamma})': results[elbow_gamma],
    'Fully robust\n(Gamma=T)': results[T],
}
plan_costs  = [p['total_cost'] for p in plans.values()]
plan_shorts = [worst_case_shortfall(p['o_r'], 20, d_upper)
               for p in plans.values()]

fig3, ax3 = plt.subplots(figsize=(10, 5.5))
fig3.patch.set_facecolor(PALETTE['bg'])
ax3.set_facecolor(PALETTE['bg'])
x = np.arange(len(plans))
bars = ax3.bar(x, plan_costs, width=0.5, color=PALETTE['warehouse'], alpha=0.88,
               label='Total cost ($)')
for bar, c in zip(bars, plan_costs):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
             f'${c:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(list(plans.keys()), fontsize=10)
ax3.set_ylabel('Total cost ($)', fontsize=11, color=PALETTE['warehouse'])
ax3.tick_params(axis='y', labelcolor=PALETTE['warehouse'])

ax3b = ax3.twinx()
ax3b.plot(x, plan_shorts, marker='D', color=PALETTE['worst'], linewidth=2.5,
          markersize=11, label='Worst-case shortfall')
for xi, sh in zip(x, plan_shorts):
    ax3b.annotate(f'{sh:.0f}', (xi, sh), textcoords='offset points',
                  xytext=(8, 6), fontsize=10, color=PALETTE['worst'],
                  fontweight='bold')
ax3b.set_ylabel('Worst-case shortfall (units)', fontsize=11, color=PALETTE['worst'])
ax3b.tick_params(axis='y', labelcolor=PALETTE['worst'])
ax3b.set_ylim(bottom=-1)

ax3.set_title('Week 7 — The Trade-off: Cost vs Protection\n'
              'more robustness costs more but removes worst-case stockouts',
              fontsize=12, fontweight='bold')
ax3.grid(axis='y', color=PALETTE['grid'], linewidth=0.8)
ax3.spines['top'].set_visible(False)
ax3b.spines['top'].set_visible(False)
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3b.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left')
plt.tight_layout()
fig3_path = os.path.join(OUT_DIR, 'week7_tradeoff.png')
plt.savefig(fig3_path, dpi=130, bbox_inches='tight')
print(f'  [Chart saved: {fig3_path}]')
plt.close(fig3)


# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 72)
print('   All Week 7 experiments complete.')
print('   Key Concepts Demonstrated:')
print('   1. Box uncertainty set: d_t in [d_bar - d_hat, d_bar + d_hat]')
print('   2. Gamma budget: how many periods may hit worst case at once')
print('   3. Protection term = sum of the Gamma largest deviations (LP-safe)')
print('   4. Robust no-stockout constraint on cumulative shipments')
print('   5. Gamma sweep 0..T and the price-of-robustness curve')
print('   6. Elbow detection → recommended Gamma at a fraction of full cost')
print('   7. Trade-off: deterministic vs balanced-robust vs fully-robust')
print('=' * 72)
