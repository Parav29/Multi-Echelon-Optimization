# Week 6 — Stochastic Model & Cost Comparison Report

## Goal
Introduce demand **uncertainty** through scenario-based stochastic optimization,
build a two-stage stochastic LP, and compare its expected cost against the
deterministic (plan-for-the-mean) benchmark.

## Model
Two-stage stochastic program (`week6_stochastic.py`):

- **First-stage** decision `o_w[t]` — warehouse orders chosen *before* demand is
  known, therefore identical across every scenario.
- **Second-stage** recourse `o_r[t][s]`, `I_w[t][s]`, `I_r[t][s]`, `short[t][s]`
  — adapt to the demand actually realised in scenario `s`.
- **Objective — expected cost:**
  `Σ_t c·o_w[t] + Σ_s p[s]·Σ_t (h_w·I_w[t][s] + h_r·I_r[t][s] + p_short·short[t][s])`
- Balance equations are enforced for **each period and each scenario**.
- `p_short` prices every unit of unmet demand (shortage penalty).

## Part A — Three manual scenarios (1 retailer, T=5, S=3)
| Scenario | Probability | Demand path |
|----------|-------------|-------------|
| 0 — low    | 1/3 | 18, 22, 20, 25, 19 |
| 1 — normal | 1/3 | 28, 32, 30, 35, 29 |
| 2 — high   | 1/3 | 40, 45, 42, 48, 38 |

**Result:** the stochastic plan chooses one set of first-stage orders that works
across all three futures. Compared against the deterministic plan (which only
plans for the mean demand path and is then exposed to the real scenarios), the
stochastic plan has the lower expected cost. The gap is the
**Value of the Stochastic Solution (VSS)** — the money saved purely by modelling
uncertainty instead of ignoring it.

Charts: `week6_partA_scenarios.png`, `week6_partA_comparison.png`.

## Part B — Ten generated scenarios (2 retailers, T=5, S=10)
Scenarios are generated with `numpy` (`Normal(mean, std)`, clipped at 0), one
demand array per retailer. The warehouse feeds two independent retailers, each
with its own balance and shortage variables. Chart: `week6_partB_variability.png`.

## Experiments (deliverable)
- **Compare with deterministic** — stochastic plan wins on expected cost (VSS > 0).
- **Increase the shortage penalty** (`p_short` 8 → 20) — the optimiser buys more
  safety stock early, driving total shortage toward zero even in the high scenario.
- **Increase demand variability** (`std` 6 → 14) — both expected cost *and*
  expected shortage rise sharply, because a wider spread of futures forces more
  conservative (costlier) ordering.

## Takeaways
1. Uncertainty is captured by scenarios, each a full demand path with a probability.
2. First-stage/second-stage separation is what makes a plan *robust to which future
   actually occurs*.
3. Pricing shortages and widening variability both push the optimiser toward
   holding more inventory — the classic cost-vs-service-level trade-off.

Run with: `cd week6 && python week6_stochastic.py`
