import os
import pandas as pd
import numpy as np
import requests

# Set random seed for reproducibility
np.random.seed(42)

# File Paths
DATA_DIR = r"C:\Users\amanl\.gemini\antigravity\scratch\supply-chain-optimization"
RAW_DATA_PATH = os.path.join(DATA_DIR, "online_retail.csv")
PROCESSED_DEMAND_PATH = os.path.join(DATA_DIR, "processed_demand.csv")
PRODUCT_METADATA_PATH = os.path.join(DATA_DIR, "product_metadata.csv")

# UCI Online Retail Dataset URL (Clean Databricks Hosted Raw CSV)
DATASET_URL = "https://raw.githubusercontent.com/databricks/Spark-The-Definitive-Guide/master/data/retail-data/all/online-retail-dataset.csv"

def download_data():
    """Downloads the raw UCI Online Retail dataset if not already present."""
    if os.path.exists(RAW_DATA_PATH):
        print(f"[*] Raw dataset already exists at: {RAW_DATA_PATH}")
        return

    print(f"[*] Downloading raw UCI Online Retail dataset from:\n    {DATASET_URL}")
    print("[*] Please wait, downloading ~22MB file...")
    
    try:
        response = requests.get(DATASET_URL, stream=True)
        response.raise_for_status()
        
        with open(RAW_DATA_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("[+] Download complete!")
    except Exception as e:
        print(f"[-] Failed to download dataset: {e}")
        raise

def process_pipeline():
    """Ingests raw transactions, cleans, aggregates, and builds lag features."""
    download_data()

    print("[*] Loading raw transactions into memory...")
    # Ingest CSV
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"[+] Loaded {len(df):,} transactions.")
    
    # 1. CLEANING & FILTERING
    print("[*] Performing data cleaning...")
    
    # Drop rows without InvoiceNo or StockCode
    df = df.dropna(subset=["InvoiceNo", "StockCode"])
    
    # Convert InvoiceDate to datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors='coerce')
    df = df.dropna(subset=["InvoiceDate"])
    
    # Convert InvoiceNo to string to handle checking
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    
    # Filter out transaction cancellations (InvoiceNo starts with 'C')
    df = df[~df["InvoiceNo"].str.startswith("C")]
    
    # Filter out non-positive quantities and prices
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    
    # Strip whitespace from Description and StockCode
    df["Description"] = df["Description"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    
    # Formulate Daily Date
    df["Date"] = df["InvoiceDate"].dt.normalize()
    
    print(f"[+] Cleaned transactions remaining: {len(df):,}")

    # 2. SELECT TOP SKUs (ABC Analysis - High Velocity Products)
    # To build high-precision time series, we select the top 15 most frequent SKUs
    print("[*] Performing ABC Product Velocity Analysis...")
    product_frequency = df.groupby("StockCode")["Quantity"].sum().sort_values(ascending=False)
    top_skus = product_frequency.head(15).index.tolist()
    
    print(f"[+] Selected top 15 high-velocity SKUs for predictive modeling:")
    for i, sku in enumerate(top_skus, 1):
        desc = df[df["StockCode"] == sku]["Description"].iloc[0]
        qty = product_frequency[sku]
        print(f"    {i}. SKU [{sku}]: {desc} (Total Sold: {qty:,})")

    # Create Product Metadata (mapping StockCode to Description and Avg UnitPrice)
    print("[*] Creating product metadata dictionary...")
    metadata_df = df[df["StockCode"].isin(top_skus)].groupby("StockCode").agg({
        "Description": "first",
        "UnitPrice": "mean"
    }).reset_index()
    
    # Let's add simulated Lead Time (days), holding cost, and stockout penalty for the optimizer
    # This simulates a real supply chain database
    metadata_df["LeadTime"] = np.random.randint(3, 8, size=len(metadata_df)) # 3 to 7 days lead time
    metadata_df["HoldingCostDaily"] = np.round(metadata_df["UnitPrice"] * 0.005, 3) # 0.5% of price daily
    metadata_df["StockoutPenalty"] = np.round(metadata_df["UnitPrice"] * 2.5, 2)    # 2.5x price penalty
    
    metadata_df.to_csv(PRODUCT_METADATA_PATH, index=False)
    print(f"[+] Product metadata saved to: {PRODUCT_METADATA_PATH}")

    # 3. DAILY DEMAND AGGREGATION
    print("[*] Aggregating transaction logs to daily SKU demand time-series...")
    
    # Filter transactions for top SKUs
    df_top = df[df["StockCode"].isin(top_skus)]
    
    # Group by Date and StockCode to get total daily Quantity sold
    daily_demand = df_top.groupby(["Date", "StockCode"])["Quantity"].sum().reset_index()
    
    # We must ensure continuous daily time series for each SKU (filling missing days with 0 demand)
    all_dates = pd.date_range(start=daily_demand["Date"].min(), end=daily_demand["Date"].max(), freq="D")
    
    reindexed_records = []
    for sku in top_skus:
        sku_df = daily_demand[daily_demand["StockCode"] == sku].set_index("Date")
        # Reindex to full date range
        sku_df = sku_df.reindex(all_dates)
        # Fill missing values in 'Quantity' with 0, and safely assign StockCode afterwards
        sku_df["Quantity"] = sku_df["Quantity"].fillna(0)
        sku_df["StockCode"] = sku
        sku_df = sku_df.reset_index().rename(columns={"index": "Date"})
        reindexed_records.append(sku_df)
        
    full_ts_df = pd.concat(reindexed_records, ignore_index=True)
    print(f"[+] Continuous time-series generated. Total rows: {len(full_ts_df):,}")

    # 4. TIME-SERIES FEATURE ENGINEERING (Lags & Rolling Metrics)
    print("[*] Executing time-series feature engineering...")
    
    processed_records = []
    for sku in top_skus:
        sku_df = full_ts_df[full_ts_df["StockCode"] == sku].copy().sort_values("Date")
        
        # Calendar Features
        sku_df["DayOfWeek"] = sku_df["Date"].dt.dayofweek
        sku_df["Month"] = sku_df["Date"].dt.month
        sku_df["IsWeekend"] = sku_df["DayOfWeek"].isin([5, 6]).astype(int)
        
        # Target Variable
        sku_df["Demand"] = sku_df["Quantity"]
        
        # Lag Features (Past Sales)
        sku_df["Lag_7"] = sku_df["Demand"].shift(7)
        sku_df["Lag_14"] = sku_df["Demand"].shift(14)
        
        # Rolling Statistics (7-day window)
        sku_df["Rolling_Mean_7"] = sku_df["Demand"].shift(1).rolling(window=7).mean()
        sku_df["Rolling_Std_7"] = sku_df["Demand"].shift(1).rolling(window=7).std()
        
        # Drop rows with NaN (due to shifting/rolling)
        sku_df = sku_df.dropna()
        
        processed_records.append(sku_df)
        
    final_df = pd.concat(processed_records, ignore_index=True)
    
    # Save the fully processed time-series feature dataset
    final_df.to_csv(PROCESSED_DEMAND_PATH, index=False)
    print(f"[+] Processed daily aggregated time-series saved to: {PROCESSED_DEMAND_PATH}")
    print(f"[+] Feature matrix columns: {list(final_df.columns)}")
    print("[+] DATA PIPELINE PROCESS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    process_pipeline()
