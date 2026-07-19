# Multi-Echelon Supply Chain Optimization

This repository contains Python-based linear programming (LP) models using the `PuLP` library to optimize inventory and ordering decisions in a multi-echelon supply chain. 

The project progresses week-by-week, starting from a basic single-period model and evolving into a complex multi-period model with lead times and multiple retailers.

---

## 📅 Project Progression

### 🔹 Week 1: Basic Two-Echelon LP
**Objective:** Model a basic supply chain consisting of a Supplier, a Warehouse, and a Retailer for a single period.
- Set up decision variables for ordering ($o_w$, $o_r$) and inventory ($I_w$, $I_r$).
- Defined objective function to minimize total costs (ordering cost + holding costs).
- Implemented inventory balance equations.
- **Tools used:** `pulp.LpVariable`, `pulp.LpProblem`.

### 🔹 Week 2: Sensitivity Analysis & Experiments
**Objective:** Test the robustness of the basic model by running multiple scenarios.
- Handled different edge cases such as zero initial inventory, extreme demand spikes, and varying holding/ordering costs.
- Analyzed how the solver prioritizes holding inventory at the warehouse vs. the retailer based on cost parameters.
- Demonstrated the mathematical trade-offs in supply chain economics.

### 🔹 Week 3: Multi-Retailer Extension
**Objective:** Expand the two-echelon model to support one warehouse serving **two** independent retailers.
- Added decision variables for Retailer 2.
- Updated warehouse inventory balance constraints to account for shipments to multiple downstream nodes.
- Visualized cost comparisons and inventory levels across different demand sensitivity scenarios.

### 🔹 Week 4: Multi-Period Inventory with Lead Times
**Objective:** Transition from a single-period snapshot to a dynamic, multi-period planning horizon ($T$) with delayed deliveries (Lead Time $L$).
- Implemented dictionary variables using `pulp.LpVariable.dicts` to handle time-indexed variables ($o_w[t], I_w[t]$, etc.).
- Used `pulp.lpSum` to efficiently aggregate costs over all $T$ periods.
- Added complex lead-time constraints: Orders placed at time $t$ arrive at time $t + L$.
- Ran stress tests to observe how lead times impact feasibility when initial stock is low.

### 🔹 Week 5: Multiple Retailers, Capacity & Shadow Prices
**Objective:** Serve **two** independent retailers from one warehouse, add a warehouse **storage-capacity** limit, and read the **dual (shadow) prices** of that limit.
- Added a second retailer with its own demand and balance equation; the warehouse now ships to both ($I_w[t] \mathrel{-}= o_{r1}[t] + o_{r2}[t]$).
- Imposed the capacity constraint as a **named** constraint `storage_cap_t{t}` (`I_w[t] <= CAP_STORE`) so its shadow price is retrievable.
- Read shadow prices via `m.constraints['storage_cap_t{t}'].pi` — `0` means the constraint is slack, non-zero means it is a **binding bottleneck**.
- Experiments: tighten `CAP_STORE`, change $h_{r2}$, change lead time, and observe how the shadow prices and total cost respond.
- **Deliverable:** multi-retailer LP model with shadow-price analysis.

### 🔹 Week 6: Stochastic Optimization (Demand is Uncertain)
**Objective:** Capture demand uncertainty with **scenarios** and minimise **expected cost** in a two-stage stochastic LP.
- **First-stage** orders `o_w[t]` are chosen before demand is known (same across all scenarios); **second-stage** recourse (`o_r`, `I_w`, `I_r`, `short`) adapts per scenario.
- Objective: $\sum_t c\,o_w[t] + \sum_s p[s]\sum_t (h_w I_w[t][s] + h_r I_r[t][s] + p_{short}\,\text{short}[t][s])$, with balance equations per period **and** per scenario.
- **Part A:** three manual scenarios (1 retailer, T=5). **Part B:** ten `numpy`-generated scenarios (2 retailers).
- Compared against the deterministic mean-demand plan (**Value of the Stochastic Solution**), then increased the shortage penalty and demand variability.
- **Deliverable:** stochastic model + cost-comparison report (`week6/REPORT.md`).

### 🔹 Week 7: Robust Optimization
**Objective:** Protect against the **worst case** within a demand range — no probabilities needed — and analyse the cost-vs-protection trade-off.
- **Box uncertainty set:** $d_t \in [\bar d_t - \hat d_t,\ \bar d_t + \hat d_t]$.
- **Gamma budget** (Bertsimas–Sim): how many periods may hit their worst case simultaneously ($\Gamma=0$ = deterministic, $\Gamma=T$ = fully conservative).
- Robust no-stockout constraint on cumulative shipments; the protection term is the sum of the $\Gamma$ largest deviations, keeping the model a clean LP.
- Ran the **Gamma sweep** ($0 \to T$), plotted the **price-of-robustness curve**, and auto-detected the **elbow** to recommend a $\Gamma$.
- **Deliverable:** robust model + trade-off analysis (`week7/REPORT.md`).

### 🔹 Week 8: Final Experiments & Report
**Objective:** Consolidate Weeks 4–6 into one **resilience model** and stress-test it against real-world **disruptions**, then compile the final report.
- One model = multi-period + two retailers + warehouse **and** retailer capacity + shortage recourse (so shocks show up as priced unmet demand, not infeasibility).
- **Disruption cases:** reduced supply (`supply_cap`), delayed replenishment (`delay_map`), and limited warehouse capacity (tight `CAP_STORE`), plus a combined shock.
- **Metrics:** cost breakdown (ordering/holding/shortage), total shortage, and **service level** (% demand met); plus a **resilience-frontier** sweep exposing the chain's disruption threshold.
- **Finding:** single shocks are survivable but **compound** when combined; tight capacity alone raises cost without stockouts; beyond a threshold some shortage is **structural** (unavoidable at any price).
- **Deliverable:** final experiments + consolidated report covering Weeks 1–8 (`week8/REPORT.md`).

---

## 🚀 How to Run

Each week is contained within its own directory and can be executed independently. The scripts automatically run all experiments and generate corresponding visualizations (charts) in the same directory.

```bash
# Example: Running the Week 4 Multi-Period LP model
cd week4
python week4_multiperiod_lead_time.py
```

Each later week follows the same pattern:

```bash
cd week5 && python week5_capacity_shadow_prices.py   # two retailers, capacity, shadow prices
cd week6 && python week6_stochastic.py               # stochastic optimization (Parts A & B)
cd week7 && python week7_robust.py                   # robust optimization + Gamma sweep
cd week8 && python week8_final_experiments.py        # disruption stress-tests + final report
```

## 📦 Dependencies

Ensure you have the required libraries installed:
```bash
pip install pulp numpy matplotlib
```
