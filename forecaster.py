import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Set seed for reproducibility
np.random.seed(42)

# File Paths
DATA_DIR = r"C:\Users\amanl\.gemini\antigravity\scratch\supply-chain-optimization"
PROCESSED_DEMAND_PATH = os.path.join(DATA_DIR, "processed_demand.csv")
MODELS_DIR = os.path.join(DATA_DIR, "models")
PREDICTIONS_PATH = os.path.join(DATA_DIR, "predictions.csv")
FORECAST_METRICS_PATH = os.path.join(DATA_DIR, "forecast_metrics.csv")

# Create models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)

# Select Model Type with automated fallback
try:
    from xgboost import XGBRegressor
    USE_XGB = True
    print("[*] Successfully imported XGBoost Regressor as primary modeling engine.")
except ImportError:
    USE_XGB = False
    print("[!] XGBoost not detected. Falling back to RandomForestRegressor as robust ML engine.")

def train_and_evaluate():
    """Trains individual forecasting ML models for each of the top SKUs and saves predictions."""
    if not os.path.exists(PROCESSED_DEMAND_PATH):
        raise FileNotFoundError(f"Processed demand file not found at: {PROCESSED_DEMAND_PATH}. Run data_pipeline.py first!")

    df = pd.read_csv(PROCESSED_DEMAND_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    
    top_skus = df["StockCode"].unique()
    
    predictions_list = []
    metrics_records = []
    
    # Feature columns
    feature_cols = ["DayOfWeek", "Month", "IsWeekend", "Lag_7", "Lag_14", "Rolling_Mean_7", "Rolling_Std_7"]
    target_col = "Demand"
    
    print(f"[*] Beginning model training for {len(top_skus)} products...")
    
    for sku in top_skus:
        sku_df = df[df["StockCode"] == sku].copy().sort_values("Date")
        
        # 1. TIME-SERIES TRAIN/TEST SPLIT
        # We hold out the last 30 days as the validation set
        split_date = sku_df["Date"].max() - pd.Timedelta(days=30)
        train_df = sku_df[sku_df["Date"] <= split_date]
        test_df = sku_df[sku_df["Date"] > split_date]
        
        X_train, y_train = train_df[feature_cols], train_df[target_col]
        X_test, y_test = test_df[feature_cols], test_df[target_col]
        
        # In case test data is empty or too small, fall back to simple row division
        if len(test_df) < 5:
            split_idx = int(len(sku_df) * 0.9)
            train_df = sku_df.iloc[:split_idx]
            test_df = sku_df.iloc[split_idx:]
            X_train, y_train = train_df[feature_cols], train_df[target_col]
            X_test, y_test = test_df[feature_cols], test_df[target_col]

        # 2. MODEL SELECTION & TRAINING
        if USE_XGB:
            model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
            
        model.fit(X_train, y_train)
        
        # 3. SAVE TRAINED MODEL
        model_path = os.path.join(MODELS_DIR, f"model_{sku}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
            
        # 4. EVALUATION
        # Predictions
        train_pred = np.clip(model.predict(X_train), 0, None)
        test_pred = np.clip(model.predict(X_test), 0, None)
        
        # Calculate uncertainty (residuals standard deviation of training set)
        residuals = y_train - train_pred
        forecast_uncertainty = np.std(residuals)
        if forecast_uncertainty <= 0:
            forecast_uncertainty = 1.0 # Guard against perfectly zero variance
            
        # Metrics on Test Set
        mae = mean_absolute_error(y_test, test_pred)
        r2 = r2_score(y_test, test_pred)
        mean_actual = y_test.mean()
        # Mean Absolute Percentage Error (MAPE) with small denominator guard
        mape = np.mean(np.abs(y_test - test_pred) / np.clip(y_test, 1.0, None)) * 100
        
        print(f"    SKU [{sku}]: Test MAPE = {mape:.2f}% | R2 = {r2:.2f} | Uncertainty (StdDev) = {forecast_uncertainty:.2f}")
        
        metrics_records.append({
            "StockCode": sku,
            "MAE": round(mae, 2),
            "MAPE": round(mape, 2),
            "R2": round(r2, 4),
            "Uncertainty": round(forecast_uncertainty, 4)
        })
        
        # Store predictions for the test interval
        sku_preds = test_df[["Date", "StockCode", "Demand"]].copy()
        sku_preds["Predicted"] = np.round(test_pred, 1)
        # Dynamic prediction intervals: Forecast +/- 1.96 * Uncertainty (95% confidence bounds)
        sku_preds["Lower_Bound"] = np.clip(np.round(test_pred - 1.96 * forecast_uncertainty, 1), 0, None)
        sku_preds["Upper_Bound"] = np.round(test_pred + 1.96 * forecast_uncertainty, 1)
        predictions_list.append(sku_preds)
        
    # Save predictions and metrics
    predictions_df = pd.concat(predictions_list, ignore_index=True)
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)
    
    metrics_df = pd.DataFrame(metrics_records)
    metrics_df.to_csv(FORECAST_METRICS_PATH, index=False)
    
    print(f"\n[+] Validation predictions saved to: {PREDICTIONS_PATH}")
    print(f"[+] Model performance metrics saved to: {FORECAST_METRICS_PATH}")
    print("[+] MACHINE LEARNING FORECASTING PIPELINE COMPLETE!")

if __name__ == "__main__":
    train_and_evaluate()
