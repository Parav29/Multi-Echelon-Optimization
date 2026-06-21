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

---

## 🚀 How to Run

Each week is contained within its own directory and can be executed independently. The scripts automatically run all experiments and generate corresponding visualizations (charts) in the same directory.

```bash
# Example: Running the Week 4 Multi-Period LP model
cd week4
python week4_multiperiod_lead_time.py
```

## 📦 Dependencies

Ensure you have the required libraries installed:
```bash
pip install pulp numpy matplotlib
```
