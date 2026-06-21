# Week 1 — Inventory Simulation
import numpy as np
import matplotlib.pyplot as plt
# 30 days of customer demand (units per day)
demand = [45, 52, 38, 61, 49, 55, 42, 58, 47, 50,
            63, 41, 56, 48, 53, 39, 62, 44, 57, 51,
            46, 60, 43, 54, 49, 58, 40, 55, 48, 52]
# We decide to stock this many units every day

stocks = [50, 55, 60, 65, 70]
results = {}

for stock in stocks:
    inventory = []
    current = stock
    stockout_days = 0
    for d in demand:
        current = current - d # customers buy d units
        if current < 0:
            current = 0 # can't have negative inventory
        inventory.append(current) # record end-of-day inventory
        if current == 0:
            stockout_days += 1 # Count days where we hit 0 stock
        current = stock # restock at start of next day
    
    total_excess = sum(inventory)
    results[stock] = {'stockouts': stockout_days, 'excess': total_excess}
    
    # Plot demand vs inventory
    plt.figure(figsize=(10, 4))
    plt.plot(demand, label='Daily Demand', color='tomato', marker='o',
        markersize=4)
    plt.plot(inventory, label='End-of-day Stock', color='steelblue', marker='s',
        markersize=4)
    plt.xlabel('Day')
    plt.ylabel('Units')
    plt.title(f'Inventory Simulation — Stock Level = {stock}\nStockouts: {stockout_days}, Excess Inventory: {total_excess}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show(block=False) # Don't block execution so we can print results

print("\n--- Simulation Results ---")
for stock, data in results.items():
    print(f"Stock: {stock} | Stockout Days: {data['stockouts']} | Total Excess Inventory: {data['excess']}")

# Find best: prioritize lowest stockouts, then lowest excess inventory
best_stock = sorted(results.keys(), key=lambda s: (results[s]['stockouts'], results[s]['excess']))[0]
print(f"\n=> The 'best' stock level found is {best_stock} (Zero or lowest stockouts with minimal excess inventory)")

plt.show() # Keep windows open until user closes them