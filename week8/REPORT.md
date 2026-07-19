# Week 8 — Final Report: Multi-Echelon Supply Chain Optimization

**Author:** Parav Solanki (24b4515)
**Project:** Multi-Echelon Optimization — Plan of Action, Weeks 1–8

This is the capstone report. It (1) summarises every model built across the
project, (2) presents the Week 8 disruption stress-tests, and (3) collects the
observations and learnings.

---

## 1. Project Journey (Weeks 1–8)

| Week | Model / Topic | Core idea added | Deliverable |
|------|---------------|-----------------|-------------|
| 1 | Supply-chain & demand basics | Echelons, costs, simulated demand | Demand notebook |
| 2 | EOQ & Linear Programming | EOQ trade-off; first LPs with PuLP | EOQ solver + LP set |
| 3 | Two-echelon LP | Warehouse → retailer, balance & cost min | Two-echelon LP |
| 4 | Multi-period + lead times | Time-indexed variables; `received[t]=o_w[t−L]` | Multi-period model |
| 5 | Multiple retailers + capacity | 2nd retailer; `I_w≤CAP_STORE`; **shadow prices** | Multi-retailer LP |
| 6 | Stochastic optimization | Scenarios; first/second-stage; expected cost | Stochastic model + report |
| 7 | Robust optimization | Box uncertainty + Gamma budget; price of robustness | Robust model + trade-off |
| 8 | Final experiments & report | Disruption stress-tests; consolidation | **This report** |

The through-line: each week made the model **one step more realistic** — from a
single deterministic period, to many periods with delays, to many retailers with
physical limits, to demand that is *uncertain* (stochastic), to demand whose
*worst case* must be survived (robust), to a chain that must keep running through
**disruptions** (Week 8).

---

## 2. The Consolidated Model (Week 8)

`week8_final_experiments.py` fuses the earlier building blocks into one model:

- **Two-echelon, multi-period, two retailers** (Weeks 3–5)
- **Lead time** on supplier → warehouse replenishment (Week 4)
- **Warehouse *and* retailer storage capacity** (Week 5, extended) — retailer caps
  matter because without them a retailer could pre-hoard unlimited stock ahead of
  a *foreseen* disruption, hiding its impact
- **Shortage recourse with a penalty** `p_short` (Week 6) — so a disruption shows
  up as measurable, priced unmet demand rather than plain infeasibility
- **Warehouse-capacity shadow prices** reported per period (Week 5)

**Objective:** minimise
`Σ_t c·o_w[t] + h_w·I_w[t] + h_r1·I_r1[t] + h_r2·I_r2[t] + p_short·(sh1[t]+sh2[t])`

---

## 3. Disruption Stress-Tests

Three disruption levers from the Plan of Action, applied to the same base chain
(T=8, LT=1, two retailers), plus a combined shock:

| Scenario | What changes | Total cost | Shortage | Service level |
|----------|--------------|-----------:|---------:|--------------:|
| **Base** | none (healthy chain) | $531 | 0 | **100.0%** |
| **D1 — reduced supply** | supplier capped at 10 u/period during t=2–4 | $729 | 7 | 97.8% |
| **D2 — delayed replenishment** | orders placed at t=1,2 arrive 2 periods late | $631 | 1 | 99.7% |
| **D3 — limited warehouse capacity** | `CAP_STORE` 40 → 12 | $538 | 0 | 100.0% |
| **D4 — combined shock** | all three at once | $987 | 29 | **91.0%** |

*(exact numbers reproduced by running the script)*

**Charts**
- `week8_disruption_costs.png` — cost breakdown (ordering/holding/shortage) and
  service level for every scenario.
- `week8_combined_shock_timeline.png` — orders, inventory and where the shortage
  lands, period by period, under the combined shock D4.
- `week8_resilience_frontier.png` — service level & cost as the supply cut is
  deepened, exposing the chain's **resilience threshold**.

---

## 4. Observations & Learnings

1. **Each single disruption is survivable, but they compound.** Reduced supply
   (97.8%), delay (99.7%) and tight capacity (100%) each dent the chain only
   slightly. Hit together (**D4**), service collapses to **91%** and cost nearly
   doubles — risks are not additive, they interact.

2. **Limited warehouse capacity alone rarely causes stockouts — it raises cost.**
   Because the warehouse can ship to retailers in the same period, a tight
   `CAP_STORE` mostly forces more frequent ordering (D3: cost +$7, service still
   100%). Its real danger is *combinatorial*: with a supply cut it removes the
   buffer that would have absorbed the shock.

3. **Some shortage is structural, not a pricing choice.** Under D4, raising the
   shortage penalty from $5 to $60 per unit does **not** improve the 91% service
   level — the goods physically cannot reach the retailers in time. Beyond the
   resilience threshold, you cannot *buy* your way out; you must change the
   physical system (more capacity, shorter lead time, safety stock, dual sourcing).

4. **The resilience frontier has a knee.** Service holds at 100% while the supply
   cut is mild, then falls once supplier capacity drops below ~10–15 units/period
   (`week8_resilience_frontier.png`). Knowing where that knee sits tells a planner
   exactly how much disruption the current design can absorb.

5. **Shadow prices flag the true bottleneck.** The warehouse-capacity dual is 0
   when capacity is slack and non-zero (binding) exactly in the periods where more
   capacity would reduce cost — pointing capital investment at the periods that
   matter (Week 5 concept, reused here).

6. **Deterministic → Stochastic → Robust is a spectrum of caution.** Deterministic
   plans for the average, stochastic hedges across weighted futures (Week 6, VSS
   quantifies the benefit), robust guarantees the worst case within a budget
   (Week 7, the Gamma dial). Week 8 shows why that caution matters: real chains
   face shocks the average never predicts.

---

## 5. How to Reproduce

```bash
pip install pulp numpy matplotlib

cd week5 && python week5_capacity_shadow_prices.py   # capacity & shadow prices
cd week6 && python week6_stochastic.py               # stochastic (Parts A & B)
cd week7 && python week7_robust.py                   # robust + Gamma sweep
cd week8 && python week8_final_experiments.py        # disruption stress-tests
```

Each script prints its results to the console and writes its charts (`.png`) into
the same folder. Per-week deliverable notes live in each week's `REPORT.md`.

---

*End of final report.*
