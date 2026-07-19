"""
week6_stochastic.py
===================
Week 6 — Stochastic Programming: Demand is Uncertain

The deterministic model of Weeks 4–5 assumed we knew demand exactly. Real demand
is uncertain. Stochastic programming captures that uncertainty with SCENARIOS:
each scenario is one complete possible demand path over all periods, and each has
a probability. We minimise the EXPECTED cost across all scenarios.

Two-stage structure
  • First-stage decisions  : o_w[t]  (warehouse orders). Chosen HERE-AND-NOW,
    before demand is revealed, so they are the SAME across every scenario.
  • Second-stage decisions : o_r[t][s], I_w[t][s], I_r[t][s], short[t][s].
    These are RECOURSE — they adapt to the demand actually realised in scenario s.

New objective (expected cost)
  Minimise  Σ_t c_order*o_w[t]
          + Σ_s p[s] · Σ_t ( h_w*I_w[t][s] + h_r*I_r[t][s] + p_short*short[t][s] )
  Ordering cost carries no scenario index (first stage). Holding and shortage
  costs are averaged across scenarios, each weighted by its probability p[s].

Part A — three MANUAL scenarios, one retailer, T = 5, S = 3   (structure clearly visible)
Part B — ten GENERATED scenarios (numpy), TWO retailers, T = 5, S = 10

Experiments (deliverable):
  • Compare the stochastic plan with the deterministic plan
  • Increase the shortage penalty p_short
  • Increase demand variability
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
    'retailer2': '#8172B2',
    'order_w':   '#C44E52',
    'short':     '#C44E52',
    'low':       '#55A868',
    'normal':    '#4C72B0',
    'high':      '#C44E52',
    'det':       '#DD8452',
    'stoch':     '#4C72B0',
    'grid':      '#e8e8e8',
    'bg':        '#F9F9F9',
}


# ═══════════════════════════════════════════════════════════════════════════════
# PART A — STOCHASTIC LP, ONE RETAILER, MANUAL SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════

def solve_stochastic_1r(
    scenarios,
    prob,
    T=5,
    LT=1,
    h_w=0.5,
    h_r=1.0,
    c_order=2.0,
    p_short=8.0,
    I_w0=80,
    I_r0=20,
    verbose=True,
    label='Stochastic (1 retailer)',
):
    """
    Two-stage stochastic LP: one warehouse, one retailer, S scenarios.

    First stage : o_w[t]                      (same across all scenarios)
    Second stage: o_r[t][s], I_w[t][s],
                  I_r[t][s], short[t][s]       (recourse, per scenario)

    Parameters
    ----------
    scenarios : list[list]  — S demand paths, each of length T
    prob      : list        — scenario probabilities (must sum to 1)
    T, LT     : int         — horizon and lead time
    h_w, h_r  : float       — holding costs (warehouse / retailer)
    c_order   : float       — ordering cost per unit
    p_short   : float       — shortage penalty per unit of unmet demand
    I_w0, I_r0: float       — opening inventories

    Returns
    -------
    dict: status, exp_cost, o_w (list length T, first stage),
          I_w, I_r, o_r, short  (each list[s][t]),
          scen_cost (list length S — realised cost per scenario)
    """
    S = len(scenarios)
    m = pulp.LpProblem('Stochastic_1R', pulp.LpMinimize)

    # ── First-stage variables: warehouse orders (no scenario index) ──────────
    o_w = pulp.LpVariable.dicts('o_w', range(T), lowBound=0)

    # ── Second-stage variables: indexed by (period, scenario) ────────────────
    o_r   = pulp.LpVariable.dicts('o_r',   (range(T), range(S)), lowBound=0)
    I_w   = pulp.LpVariable.dicts('I_w',   (range(T), range(S)), lowBound=0)
    I_r   = pulp.LpVariable.dicts('I_r',   (range(T), range(S)), lowBound=0)
    short = pulp.LpVariable.dicts('short', (range(T), range(S)), lowBound=0)

    # ── Objective: expected cost ─────────────────────────────────────────────
    order_cost = pulp.lpSum(c_order * o_w[t] for t in range(T))
    recourse_cost = pulp.lpSum(
        prob[s] * (h_w * I_w[t][s] + h_r * I_r[t][s] + p_short * short[t][s])
        for s in range(S) for t in range(T)
    )
    m += order_cost + recourse_cost, 'ExpectedCost'

    # ── Constraints: balance equations for EACH period AND EACH scenario ─────
    for s in range(S):
        for t in range(T):
            received = o_w[t - LT] if t - LT >= 0 else 0   # first-stage arrival

            prev_Iw = I_w[t - 1][s] if t > 0 else I_w0
            prev_Ir = I_r[t - 1][s] if t > 0 else I_r0

            # Warehouse balance (per scenario)
            m += I_w[t][s] == prev_Iw + received - o_r[t][s], f'wh_s{s}_t{t}'

            # Retailer balance with shortage recourse: unmet demand -> short
            #   prev + shipment - demand + short = ending inventory
            m += I_r[t][s] == prev_Ir + o_r[t][s] - scenarios[s][t] + short[t][s], \
                f'ret_s{s}_t{t}'

    # ── Solve ────────────────────────────────────────────────────────────────
    m.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[m.status]
    if m.status != 1:
        if verbose:
            print(f'\n  [{label}] Status: {status}')
        return {'status': status, 'exp_cost': None}

    ow_vals    = [o_w[t].varValue for t in range(T)]
    Iw_vals    = [[I_w[t][s].varValue   for t in range(T)] for s in range(S)]
    Ir_vals    = [[I_r[t][s].varValue   for t in range(T)] for s in range(S)]
    or_vals    = [[o_r[t][s].varValue   for t in range(T)] for s in range(S)]
    short_vals = [[short[t][s].varValue for t in range(T)] for s in range(S)]
    exp_cost   = pulp.value(m.objective)

    # Realised cost in each scenario (ordering cost + that scenario's recourse)
    order_only = sum(c_order * ow_vals[t] for t in range(T))
    scen_cost = []
    for s in range(S):
        c = order_only + sum(h_w * Iw_vals[s][t] + h_r * Ir_vals[s][t]
                             + p_short * short_vals[s][t] for t in range(T))
        scen_cost.append(c)

    if verbose:
        print(f'\n{"─"*72}')
        print(f'  {label}')
        print(f'  T={T}  LT={LT}  S={S}  c_order={c_order}  h_w={h_w}  h_r={h_r}  '
              f'p_short={p_short}')
        print(f'  Opening: I_w0={I_w0}  I_r0={I_r0}')
        print(f'  Expected Cost: ${exp_cost:.2f}')
        print(f'  First-stage warehouse orders o_w (same for every scenario):')
        print('     ' + '  '.join(f't{t}={ow_vals[t]:.1f}' for t in range(T)))
        print(f'  Per-scenario realised cost & total shortage:')
        for s in range(S):
            tot_short = sum(short_vals[s][t] for t in range(T))
            print(f'     Scenario {s} (p={prob[s]:.2f}): cost=${scen_cost[s]:.2f}  '
                  f'demand={scenarios[s]}  total_short={tot_short:.1f}')
        print(f'{"─"*72}')

    return {
        'status': status,
        'exp_cost': exp_cost,
        'o_w': ow_vals,
        'I_w': Iw_vals,
        'I_r': Ir_vals,
        'o_r': or_vals,
        'short': short_vals,
        'scen_cost': scen_cost,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PART B — STOCHASTIC LP, TWO RETAILERS, GENERATED SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════

def solve_stochastic_2r(
    scen1,
    scen2,
    prob,
    T=5,
    LT=1,
    h_w=0.5,
    h_r1=1.0,
    h_r2=1.0,
    c_order=2.0,
    p_short=8.0,
    I_w0=120,
    I_r1_0=20,
    I_r2_0=20,
    verbose=True,
    label='Stochastic (2 retailers)',
):
    """
    Two-stage stochastic LP with TWO retailers and S generated scenarios.
    Same structure as Part A; the warehouse now feeds two retailers, each with
    its own scenario demand array (scen1[s][t], scen2[s][t]).
    """
    S = len(scen1)
    m = pulp.LpProblem('Stochastic_2R', pulp.LpMinimize)

    # First stage
    o_w = pulp.LpVariable.dicts('o_w', range(T), lowBound=0)

    # Second stage (per scenario)
    o_r1 = pulp.LpVariable.dicts('o_r1', (range(T), range(S)), lowBound=0)
    o_r2 = pulp.LpVariable.dicts('o_r2', (range(T), range(S)), lowBound=0)
    I_w  = pulp.LpVariable.dicts('I_w',  (range(T), range(S)), lowBound=0)
    I_r1 = pulp.LpVariable.dicts('I_r1', (range(T), range(S)), lowBound=0)
    I_r2 = pulp.LpVariable.dicts('I_r2', (range(T), range(S)), lowBound=0)
    sh1  = pulp.LpVariable.dicts('sh1',  (range(T), range(S)), lowBound=0)
    sh2  = pulp.LpVariable.dicts('sh2',  (range(T), range(S)), lowBound=0)

    m += (
        pulp.lpSum(c_order * o_w[t] for t in range(T))
        + pulp.lpSum(
            prob[s] * (h_w * I_w[t][s] + h_r1 * I_r1[t][s] + h_r2 * I_r2[t][s]
                       + p_short * sh1[t][s] + p_short * sh2[t][s])
            for s in range(S) for t in range(T)
        )
    ), 'ExpectedCost'

    for s in range(S):
        for t in range(T):
            received = o_w[t - LT] if t - LT >= 0 else 0
            prev_Iw  = I_w[t - 1][s]  if t > 0 else I_w0
            prev_Ir1 = I_r1[t - 1][s] if t > 0 else I_r1_0
            prev_Ir2 = I_r2[t - 1][s] if t > 0 else I_r2_0

            m += I_w[t][s] == prev_Iw + received - o_r1[t][s] - o_r2[t][s], \
                f'wh_s{s}_t{t}'
            m += I_r1[t][s] == prev_Ir1 + o_r1[t][s] - scen1[s][t] + sh1[t][s], \
                f'r1_s{s}_t{t}'
            m += I_r2[t][s] == prev_Ir2 + o_r2[t][s] - scen2[s][t] + sh2[t][s], \
                f'r2_s{s}_t{t}'

    m.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[m.status]
    if m.status != 1:
        if verbose:
            print(f'\n  [{label}] Status: {status}')
        return {'status': status, 'exp_cost': None}

    ow_vals  = [o_w[t].varValue for t in range(T)]
    exp_cost = pulp.value(m.objective)
    tot_short = sum(prob[s] * sum(sh1[t][s].varValue + sh2[t][s].varValue
                                  for t in range(T)) for s in range(S))

    if verbose:
        print(f'\n{"─"*72}')
        print(f'  {label}')
        print(f'  T={T}  LT={LT}  S={S}  p_short={p_short}  '
              f'h_r1={h_r1}  h_r2={h_r2}')
        print(f'  Expected Cost: ${exp_cost:.2f}')
        print(f'  Expected total shortage (both retailers): {tot_short:.2f} units')
        print(f'  First-stage warehouse orders o_w:')
        print('     ' + '  '.join(f't{t}={ow_vals[t]:.1f}' for t in range(T)))
        print(f'{"─"*72}')

    return {'status': status, 'exp_cost': exp_cost, 'o_w': ow_vals,
            'exp_short': tot_short}


def deterministic_1r_expected(scenarios, prob, **kwargs):
    """
    Solve the DETERMINISTIC model using the expected (probability-weighted mean)
    demand path, then evaluate that single plan's EXPECTED cost across the real
    scenarios. This is what "ignoring uncertainty" costs — the benchmark the
    stochastic model is compared against.
    """
    T = kwargs.get('T', 5)
    S = len(scenarios)
    mean_demand = [sum(prob[s] * scenarios[s][t] for s in range(S))
                   for t in range(T)]
    # Solve deterministic with the mean path as a single "scenario".
    det = solve_stochastic_1r([mean_demand], [1.0], verbose=False, **kwargs)
    # Fix its first-stage o_w and evaluate the expected cost over real scenarios.
    return det, mean_demand


# ═══════════════════════════════════════════════════════════════════════════════
# RUN PART A — THREE MANUAL SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════

print('\n' + '=' * 72)
print('   WEEK 6 — Stochastic Programming (Demand is Uncertain)')
print('=' * 72)

print('\n########## PART A — Three Manual Scenarios (1 retailer, T=5, S=3) ##########')

scenariosA = [
    [18, 22, 20, 25, 19],   # Scenario 0: low demand
    [28, 32, 30, 35, 29],   # Scenario 1: normal demand
    [40, 45, 42, 48, 38],   # Scenario 2: high demand
]
probA = [1/3, 1/3, 1/3]

params_A = dict(T=5, LT=1, h_w=0.5, h_r=1.0, c_order=2.0,
                p_short=8.0, I_w0=80, I_r0=20)

resA = solve_stochastic_1r(scenariosA, probA, label='PART A — Stochastic',
                           **params_A)

# ── Compare against the deterministic (mean-demand) plan ──────────────────────
detA, mean_demandA = deterministic_1r_expected(scenariosA, probA, **params_A)
print('\n  --- Deterministic benchmark (plans for MEAN demand only) ---')
print(f'  Mean demand path: {[round(d, 1) for d in mean_demandA]}')
# Evaluate the deterministic first-stage orders across the real scenarios by
# re-solving the recourse with o_w fixed to the deterministic plan.
det_ow = detA['o_w']


def evaluate_fixed_ow(ow_fixed, scenarios, prob, **kwargs):
    """Evaluate the expected cost of a FIXED first-stage order plan ow_fixed
    when the true demand follows `scenarios`. Only recourse re-optimises."""
    T = kwargs.get('T', 5); LT = kwargs.get('LT', 1)
    h_w = kwargs.get('h_w', 0.5); h_r = kwargs.get('h_r', 1.0)
    c_order = kwargs.get('c_order', 2.0); p_short = kwargs.get('p_short', 8.0)
    I_w0 = kwargs.get('I_w0', 80); I_r0 = kwargs.get('I_r0', 20)
    S = len(scenarios)
    m = pulp.LpProblem('EvalFixed', pulp.LpMinimize)
    o_r   = pulp.LpVariable.dicts('o_r',   (range(T), range(S)), lowBound=0)
    I_w   = pulp.LpVariable.dicts('I_w',   (range(T), range(S)), lowBound=0)
    I_r   = pulp.LpVariable.dicts('I_r',   (range(T), range(S)), lowBound=0)
    short = pulp.LpVariable.dicts('short', (range(T), range(S)), lowBound=0)
    m += (sum(c_order * ow_fixed[t] for t in range(T))
          + pulp.lpSum(prob[s] * (h_w * I_w[t][s] + h_r * I_r[t][s]
                                  + p_short * short[t][s])
                       for s in range(S) for t in range(T))), 'ExpCost'
    for s in range(S):
        for t in range(T):
            received = ow_fixed[t - LT] if t - LT >= 0 else 0
            prev_Iw = I_w[t - 1][s] if t > 0 else I_w0
            prev_Ir = I_r[t - 1][s] if t > 0 else I_r0
            m += I_w[t][s] == prev_Iw + received - o_r[t][s]
            m += I_r[t][s] == prev_Ir + o_r[t][s] - scenarios[s][t] + short[t][s]
    m.solve(pulp.PULP_CBC_CMD(msg=0))
    return pulp.value(m.objective)


det_expected_cost = evaluate_fixed_ow(det_ow, scenariosA, probA, **params_A)
print(f'  Deterministic plan first-stage o_w: {[round(x, 1) for x in det_ow]}')
print(f'  Deterministic plan expected cost (evaluated on real scenarios): '
      f'${det_expected_cost:.2f}')
print(f'  Stochastic  plan expected cost                               : '
      f'${resA["exp_cost"]:.2f}')
vss = det_expected_cost - resA['exp_cost']
print(f'  Value of the Stochastic Solution (VSS) = ${vss:.2f}  '
      f'(cost avoided by modelling uncertainty)')

# ── Experiment: increase the shortage penalty ────────────────────────────────
print('\n  --- Experiment A1: increase shortage penalty p_short 8 -> 20 ---')
resA_highpen = solve_stochastic_1r(scenariosA, probA,
                                   label='PART A — high p_short=20',
                                   **{**params_A, 'p_short': 20.0})


# ═══════════════════════════════════════════════════════════════════════════════
# RUN PART B — TEN GENERATED SCENARIOS, TWO RETAILERS
# ═══════════════════════════════════════════════════════════════════════════════

print('\n########## PART B — Ten Generated Scenarios (2 retailers, T=5, S=10) ##########')

rng = np.random.default_rng(42)
S_B = 10
T_B = 5


def generate_scenarios(mean, std, S, T, rng):
    """Generate S demand paths of length T ~ Normal(mean, std), clipped at 0."""
    return [[max(0, int(round(rng.normal(mean[t], std)))) for t in range(T)]
            for _ in range(S)]


mean_r1 = [30, 32, 28, 35, 31]
mean_r2 = [22, 24, 20, 26, 23]
std_B   = 6.0

scen1_B = generate_scenarios(mean_r1, std_B, S_B, T_B, rng)
scen2_B = generate_scenarios(mean_r2, std_B, S_B, T_B, rng)
probB   = [1.0 / S_B] * S_B   # equally likely

print(f'  Generated {S_B} scenarios for retailer 1 (mean {mean_r1}, std {std_B}):')
for s in range(S_B):
    print(f'     s{s}: r1={scen1_B[s]}  r2={scen2_B[s]}')

resB = solve_stochastic_2r(scen1_B, scen2_B, probB,
                           label='PART B — Stochastic (2 retailers, S=10)',
                           T=T_B, LT=1, h_w=0.5, h_r1=1.0, h_r2=1.0,
                           c_order=2.0, p_short=8.0,
                           I_w0=120, I_r1_0=20, I_r2_0=20)

# ── Experiment: increase demand variability (std 6 -> 14) ────────────────────
print('\n  --- Experiment B1: increase demand variability std 6 -> 14 ---')
scen1_hv = generate_scenarios(mean_r1, 14.0, S_B, T_B, rng)
scen2_hv = generate_scenarios(mean_r2, 14.0, S_B, T_B, rng)
resB_hv = solve_stochastic_2r(scen1_hv, scen2_hv, probB,
                              label='PART B — high variability (std=14)',
                              T=T_B, LT=1, h_w=0.5, h_r1=1.0, h_r2=1.0,
                              c_order=2.0, p_short=8.0,
                              I_w0=120, I_r1_0=20, I_r2_0=20)


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════════════

periodsA = list(range(5))

# ── FIGURE 1: The three manual scenarios & the single first-stage plan ───────
fig1, axes1 = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
fig1.patch.set_facecolor(PALETTE['bg'])
fig1.suptitle('Week 6 (Part A) — Three Demand Scenarios & One Robust Order Plan',
              fontsize=14, fontweight='bold', y=0.98)

ax = axes1[0]
ax.set_facecolor(PALETTE['bg'])
scen_colors = [PALETTE['low'], PALETTE['normal'], PALETTE['high']]
scen_names  = ['Scenario 0 (low)', 'Scenario 1 (normal)', 'Scenario 2 (high)']
for s in range(3):
    ax.plot(periodsA, scenariosA[s], marker='o', color=scen_colors[s],
            linewidth=2.2, label=scen_names[s], markersize=7)
ax.plot(periodsA, mean_demandA, marker='D', color='#444', linewidth=2,
        linestyle='--', label='Expected (mean) demand', markersize=6)
ax.set_ylabel('Demand (units)', fontsize=10)
ax.set_title('Three Manual Demand Scenarios (each equally likely, p = 1/3)',
             fontsize=10, pad=4)
ax.legend(fontsize=9)
ax.grid(color=PALETTE['grid'], linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax = axes1[1]
ax.set_facecolor(PALETTE['bg'])
ax.bar(periodsA, resA['o_w'], width=0.5, color=PALETTE['order_w'], alpha=0.85,
       label='o_w — first-stage order (same for all scenarios)')
for t in periodsA:
    ax.text(t, resA['o_w'][t] + 0.5, f"{resA['o_w'][t]:.0f}", ha='center',
            va='bottom', fontsize=9)
ax.set_xlabel('Period', fontsize=10)
ax.set_ylabel('Units ordered', fontsize=10)
ax.set_title('First-Stage Warehouse Orders — chosen BEFORE demand is known',
             fontsize=10, pad=4)
ax.set_xticks(periodsA)
ax.legend(fontsize=9)
ax.grid(axis='y', color=PALETTE['grid'], linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig1_path = os.path.join(OUT_DIR, 'week6_partA_scenarios.png')
plt.savefig(fig1_path, dpi=130, bbox_inches='tight')
print(f'\n  [Chart saved: {fig1_path}]')
plt.close(fig1)


# ── FIGURE 2: Deterministic vs Stochastic expected cost + per-scenario cost ──
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
fig2.patch.set_facecolor(PALETTE['bg'])
fig2.suptitle('Week 6 (Part A) — Stochastic vs Deterministic Planning',
              fontsize=14, fontweight='bold', y=1.01)

ax = axes2[0]
ax.set_facecolor(PALETTE['bg'])
bars = ax.bar(['Deterministic\n(plans for mean)', 'Stochastic\n(plans for all)'],
              [det_expected_cost, resA['exp_cost']],
              color=[PALETTE['det'], PALETTE['stoch']], alpha=0.88, width=0.55)
for bar, val in zip(bars, [det_expected_cost, resA['exp_cost']]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f'${val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('Expected cost ($)', fontsize=11)
ax.set_title(f'Expected Cost  (VSS = ${vss:.1f} saved)', fontsize=11)
ax.grid(axis='y', color=PALETTE['grid'], linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax = axes2[1]
ax.set_facecolor(PALETTE['bg'])
x = np.arange(3)
ax.bar(x - 0.2, resA['scen_cost'], width=0.38, color=PALETTE['stoch'],
       alpha=0.85, label='Stochastic plan')
ax.bar(x + 0.2, resA_highpen['scen_cost'], width=0.38, color=PALETTE['short'],
       alpha=0.85, label='Stochastic, high p_short=20')
ax.set_xticks(x)
ax.set_xticklabels(['Low', 'Normal', 'High'])
ax.set_xlabel('Realised scenario', fontsize=11)
ax.set_ylabel('Realised cost ($)', fontsize=11)
ax.set_title('Per-Scenario Cost & Effect of Shortage Penalty', fontsize=11)
ax.legend(fontsize=9)
ax.grid(axis='y', color=PALETTE['grid'], linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
fig2_path = os.path.join(OUT_DIR, 'week6_partA_comparison.png')
plt.savefig(fig2_path, dpi=130, bbox_inches='tight')
print(f'  [Chart saved: {fig2_path}]')
plt.close(fig2)


# ── FIGURE 3: Part B — generated scenarios & variability effect ──────────────
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
fig3.patch.set_facecolor(PALETTE['bg'])
fig3.suptitle('Week 6 (Part B) — Ten Generated Scenarios & Effect of Variability',
              fontsize=14, fontweight='bold', y=1.01)

ax = axes3[0]
ax.set_facecolor(PALETTE['bg'])
periodsB = list(range(T_B))
for s in range(S_B):
    ax.plot(periodsB, scen1_B[s], color=PALETTE['retailer'], alpha=0.35,
            linewidth=1.2)
    ax.plot(periodsB, scen2_B[s], color=PALETTE['retailer2'], alpha=0.35,
            linewidth=1.2)
ax.plot(periodsB, mean_r1, color=PALETTE['retailer'], linewidth=3,
        marker='o', label='Retailer 1 mean')
ax.plot(periodsB, mean_r2, color=PALETTE['retailer2'], linewidth=3,
        marker='s', label='Retailer 2 mean')
ax.set_xlabel('Period', fontsize=11)
ax.set_ylabel('Demand (units)', fontsize=11)
ax.set_title(f'{S_B} Generated Scenarios per Retailer (std={std_B})', fontsize=11)
ax.set_xticks(periodsB)
ax.legend(fontsize=9)
ax.grid(color=PALETTE['grid'], linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax = axes3[1]
ax.set_facecolor(PALETTE['bg'])
labels = ['std = 6\n(base)', 'std = 14\n(high variability)']
costs  = [resB['exp_cost'], resB_hv['exp_cost']]
shorts = [resB['exp_short'], resB_hv['exp_short']]
xB = np.arange(2)
bars = ax.bar(xB, costs, width=0.5, color=PALETTE['stoch'], alpha=0.88,
              label='Expected cost')
for bar, c in zip(bars, costs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            f'${c:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_xticks(xB)
ax.set_xticklabels(labels)
ax.set_ylabel('Expected cost ($)', fontsize=11, color=PALETTE['stoch'])
ax.tick_params(axis='y', labelcolor=PALETTE['stoch'])
axb = ax.twinx()
axb.plot(xB, shorts, marker='D', color=PALETTE['short'], linewidth=2.5,
         markersize=9, label='Expected shortage')
axb.set_ylabel('Expected shortage (units)', fontsize=11, color=PALETTE['short'])
axb.tick_params(axis='y', labelcolor=PALETTE['short'])
ax.set_title('Higher Variability → Higher Cost & Shortage Risk', fontsize=11)
ax.grid(axis='y', color=PALETTE['grid'], linewidth=0.8)
ax.spines['top'].set_visible(False)
axb.spines['top'].set_visible(False)
plt.tight_layout()
fig3_path = os.path.join(OUT_DIR, 'week6_partB_variability.png')
plt.savefig(fig3_path, dpi=130, bbox_inches='tight')
print(f'  [Chart saved: {fig3_path}]')
plt.close(fig3)


# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 72)
print('   All Week 6 experiments complete.')
print('   Key Concepts Demonstrated:')
print('   1. Scenarios: each is one full demand path with a probability')
print('   2. First-stage o_w[t] (here-and-now) vs second-stage recourse[t][s]')
print('   3. Objective = expected cost  Σ_s p[s] · Cost[s]')
print('   4. Balance equations for each period AND each scenario')
print('   5. Shortage penalty p_short prices unmet demand')
print('   6. VSS: stochastic plan beats deterministic mean-demand plan')
print('   7. Higher shortage penalty / variability raises cost & safety stock')
print('=' * 72)
