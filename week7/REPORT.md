# Week 7 — Robust Model & Trade-off Analysis

## Goal
Explore **robust optimization** for uncertain demand, compare the deterministic,
stochastic, and robust approaches, and analyse the trade-off between lower cost
and better protection against uncertainty.

## Why robust (vs stochastic)
Stochastic programming (Week 6) needs a **probability** for every scenario. When
the business is new or demand is shifting, those probabilities may not exist.
Robust optimization needs only the **range** demand can take and protects against
the **worst case** inside that range — no probabilities required.

## Model
`week7_robust.py` — box uncertainty set + Bertsimas–Sim budget:

- **Uncertainty set (box):** `d_t ∈ [d_bar_t − d_hat_t, d_bar_t + d_hat_t]`.
- **Gamma budget** `∈ [0, T]`: how many periods may hit their worst case *at once*.
  - `Gamma = 0` → all nominal (identical to deterministic — optimistic).
  - `Gamma = T` → every period worst case (fully conservative — most expensive).
- **Protection term** `protection_t(Gamma)` = sum of the `min(Gamma, t+1)` largest
  deviations `d_hat` over periods `0..t`. Because `d_hat` and `Gamma` are
  parameters, this is a constant, so the robust counterpart stays a **clean LP**.
- **Robust no-stockout constraint:**
  `I_r0 + Σ_{τ≤t} o_r[τ] ≥ Σ_{τ≤t} d_bar[τ] + protection_t(Gamma)` — cumulative
  shipments must cover worst-case cumulative demand allowed by the budget.

## The Gamma sweep & price-of-robustness curve
Solving the LP for every `Gamma` from 0 to T and plotting optimal cost produces
the **price of robustness curve** (`week7_price_of_robustness.png`). Each extra
unit of `Gamma` buys protection against one more worst-case period, at an
increasing safety-stock cost.

## Finding the elbow
The **elbow** is detected with a Kneedle-style maximum-distance heuristic (the
point on the curve farthest from the chord joining the two endpoints). It marks
the natural `Gamma`: most of the achievable protection for a fraction of the
maximum cost. The script prints, for the elbow `Gamma`:
- the total cost,
- the extra cost over `Gamma = 0`, and
- that extra as a percentage of the full price of robustness (`Gamma = T`).

## Trade-off analysis (deliverable)
`week7_tradeoff.png` compares three plans on two axes:

| Plan | Cost | Worst-case protection |
|------|------|-----------------------|
| Deterministic (`Gamma=0`) | cheapest | exposed — stocks out if demand spikes |
| Balanced robust (elbow `Gamma`) | moderate | survives up to `Gamma` bad periods |
| Fully robust (`Gamma=T`) | most expensive | survives everything |

The worst-case shortfall is evaluated by pushing demand to its upper bound in
every period and measuring the resulting stockout for each plan.

## Takeaways
1. Robustness is a **dial** (`Gamma`), not an on/off switch.
2. The first units of protection are the best value; full conservatism is costly.
3. Choose `Gamma` at the elbow to get most of the safety at a fraction of the cost
   — unless (like a hospital stocking blood) you cannot tolerate any stockout, in
   which case `Gamma = T` is justified.

Run with: `cd week7 && python week7_robust.py`
