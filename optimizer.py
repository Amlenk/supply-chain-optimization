import os
import pandas as pd
import numpy as np

# File Paths
DATA_DIR = r"C:\Users\amanl\.gemini\antigravity\scratch\supply-chain-optimization"
PRODUCT_METADATA_PATH = os.path.join(DATA_DIR, "product_metadata.csv")
FORECAST_METRICS_PATH = os.path.join(DATA_DIR, "forecast_metrics.csv")
PREDICTIONS_PATH = os.path.join(DATA_DIR, "predictions.csv")
SIMULATION_RESULTS_PATH = os.path.join(DATA_DIR, "simulation_results.csv")

def get_z_score(service_level):
    """Returns the z-score multiplier for a desired service level (percentage)."""
    # Simple lookup for common service levels
    levels = {
        80.0: 0.842,
        85.0: 1.036,
        90.0: 1.282,
        95.0: 1.645,
        98.0: 2.054,
        99.0: 2.326,
        99.9: 3.090
    }
    # Closest match helper
    closest = min(levels.keys(), key=lambda x: abs(x - service_level))
    return levels[closest]

def simulate_inventory_policy(sku, actual_demand, predicted_demand, lead_time, holding_cost_daily, stockout_penalty, forecast_uncertainty, service_level=95.0, initial_stock=100):
    """Simulates daily inventory levels and costs for both Baseline and ML-Optimized replenishment policies."""
    days = len(actual_demand)
    
    # 1. BASELINE POLICY SETUP (Standard Heuristic)
    # Baseline uses historical average demand and baseline standard deviation of demand for safety stock
    avg_demand_hist = actual_demand.mean()
    std_demand_hist = actual_demand.std()
    
    baseline_ss = 1.5 * std_demand_hist * np.sqrt(lead_time)
    baseline_rop = (avg_demand_hist * lead_time) + baseline_ss
    baseline_order_qty = max(50, avg_demand_hist * 14) # order 14 days worth of demand
    
    # 2. ML-OPTIMIZED POLICY SETUP (Dynamic Forecast-driven)
    # ML uses active forecast demand and residual uncertainty for safety stock
    z_score = get_z_score(service_level)
    ml_ss = z_score * forecast_uncertainty * np.sqrt(lead_time)
    
    # ML ROP integrates predicted demand over lead time + ML safety stock
    # ROP = (predicted daily demand * lead time) + SS
    
    # --- SIMULATION LOOPS ---
    def run_sim(policy_type):
        inventory = initial_stock
        holding_cost = 0.0
        stockout_cost = 0.0
        stockouts_count = 0
        stockout_units = 0
        
        # Pending orders list of dicts: {"eta": day_to_arrive, "qty": order_quantity}
        pending_orders = []
        
        inventory_log = []
        order_placed_log = []
        
        for day in range(days):
            # Check for order arrivals at start of day
            arrived_qty = sum(ord["qty"] for ord in pending_orders if ord["eta"] == day)
            inventory += arrived_qty
            # Remove arrived orders
            pending_orders = [ord for ord in pending_orders if ord["eta"] != day]
            
            # Record starting inventory
            day_start_inv = inventory
            
            # Customer demand occurs
            demand = actual_demand[day]
            
            # Satisfy demand
            if inventory >= demand:
                inventory -= demand
                sold = demand
                unfulfilled = 0
            else:
                sold = inventory
                unfulfilled = demand - inventory
                inventory = 0
                stockouts_count += 1
                stockout_units += unfulfilled
                stockout_cost += unfulfilled * stockout_penalty
                
            # End of day holding cost calculation
            holding_cost += inventory * holding_cost_daily
            
            inventory_log.append(inventory)
            
            # Order decision
            # We calculate current effective inventory (On-Hand + On-Order)
            on_hand = inventory
            on_order = sum(ord["qty"] for ord in pending_orders)
            effective_inv = on_hand + on_order
            
            # Policy rules
            order_placed = False
            if policy_type == "baseline":
                if effective_inv <= baseline_rop:
                    eta = day + lead_time
                    pending_orders.append({"eta": eta, "qty": baseline_order_qty})
                    order_placed = True
            elif policy_type == "ml":
                # ML dynamically updates prediction demand for the upcoming lead time
                lead_time_forecast = predicted_demand[day] * lead_time
                ml_rop = lead_time_forecast + ml_ss
                
                # Order quantity adjusts to match demand target
                ml_order_qty = max(50, predicted_demand[day] * 14)
                
                if effective_inv <= ml_rop:
                    eta = day + lead_time
                    pending_orders.append({"eta": eta, "qty": ml_order_qty})
                    order_placed = True
                    
            order_placed_log.append(order_placed)
            
        return {
            "inventory": inventory_log,
            "orders": order_placed_log,
            "holding_cost": round(holding_cost, 2),
            "stockout_cost": round(stockout_cost, 2),
            "total_cost": round(holding_cost + stockout_cost, 2),
            "stockouts_count": stockouts_count,
            "stockout_units": stockout_units
        }

    baseline_res = run_sim("baseline")
    ml_res = run_sim("ml")
    
    return baseline_res, ml_res, baseline_ss, ml_ss

def run_optimization(service_level=95.0):
    """Runs inventory simulations for all top SKUs and compiles optimization statistics."""
    # Load required data
    meta_df = pd.read_csv(PRODUCT_METADATA_PATH)
    metrics_df = pd.read_csv(FORECAST_METRICS_PATH)
    preds_df = pd.read_csv(PREDICTIONS_PATH)
    
    # Merge metadata and ML forecast metrics
    skus_data = pd.merge(meta_df, metrics_df, on="StockCode")
    
    simulation_records = []
    
    print(f"[*] Starting Inventory Policy Simulation (Target Service Level: {service_level}%)...")
    
    for _, row in skus_data.iterrows():
        sku = row["StockCode"]
        desc = row["Description"]
        lead_time = int(row["LeadTime"])
        holding_cost = float(row["HoldingCostDaily"])
        penalty = float(row["StockoutPenalty"])
        uncertainty = float(row["Uncertainty"])
        
        # Get predictions and actual demand for SKU
        sku_preds = preds_df[preds_df["StockCode"] == sku].copy().sort_values("Date")
        
        actuals = sku_preds["Demand"].values
        predictions = sku_preds["Predicted"].values
        
        # Guard in case test array is empty
        if len(actuals) == 0:
            continue
            
        baseline, ml, b_ss, m_ss = simulate_inventory_policy(
            sku=sku,
            actual_demand=actuals,
            predicted_demand=predictions,
            lead_time=lead_time,
            holding_cost_daily=holding_cost,
            stockout_penalty=penalty,
            forecast_uncertainty=uncertainty,
            service_level=service_level
        )
        
        cost_saved = baseline["total_cost"] - ml["total_cost"]
        pct_savings = (cost_saved / max(1.0, baseline["total_cost"])) * 100
        
        print(f"    SKU [{sku}]: Baseline Cost: ${baseline['total_cost']:,} | ML Cost: ${ml['total_cost']:,} | Savings: ${cost_saved:,.2f} ({pct_savings:.1f}%)")
        
        simulation_records.append({
            "StockCode": sku,
            "Description": desc,
            "LeadTime": lead_time,
            "HoldingCostDaily": holding_cost,
            "StockoutPenalty": penalty,
            "Uncertainty": uncertainty,
            "Baseline_SS": round(b_ss, 1),
            "ML_SS": round(m_ss, 1),
            "Baseline_HoldingCost": baseline["holding_cost"],
            "Baseline_StockoutCost": baseline["stockout_cost"],
            "Baseline_TotalCost": baseline["total_cost"],
            "Baseline_Stockouts": baseline["stockouts_count"],
            "ML_HoldingCost": ml["holding_cost"],
            "ML_StockoutCost": ml["stockout_cost"],
            "ML_TotalCost": ml["total_cost"],
            "ML_Stockouts": ml["stockouts_count"],
            "CostSavings": round(cost_saved, 2),
            "SavingsPercentage": round(pct_savings, 1)
        })
        
    results_df = pd.DataFrame(simulation_records)
    results_df.to_csv(SIMULATION_RESULTS_PATH, index=False)
    print(f"\n[+] Inventory optimization simulation results saved to: {SIMULATION_RESULTS_PATH}")
    print("[+] INVENTORY OPTIMIZATION SIMULATOR PROCESS COMPLETED!")

if __name__ == "__main__":
    run_optimization()
