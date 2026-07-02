# Supply Chain Optimization

End-to-end demand forecasting and inventory optimization pipeline built on the UCI Online Retail dataset.

## What it does

**Data pipeline** (`data_pipeline.py`): cleans raw transaction data and aggregates it into per-product demand series (`processed_demand.csv`).

**Forecasting** (`forecaster.py`): trains demand forecasting models per product and writes predictions and accuracy metrics (`predictions.csv`, `forecast_metrics.csv`).

**Optimization** (`optimizer.py`): converts forecasts into inventory decisions, including safety stock and reorder points, and simulates outcomes (`simulation_results.csv`).

**App** (`app.py`): ties the pipeline together for interactive exploration.

## Run

Install dependencies with `pip install -r requirements.txt`, then run `data_pipeline.py`, `forecaster.py`, and `optimizer.py` in that order, or use `run_setup.ps1` on Windows.

## Related work

See my `sap-s4hana-procurement-orchestrator` repo for a deeper treatment of the same problem with attention-based LSTM quantile forecasting and RAG contract parsing.
