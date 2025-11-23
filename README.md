# OPSD PowerDesk: Day-Ahead Load Forecasting & Anomaly Detection

## Overview
This project implements a production-grade pipeline for day-ahead electric load forecasting across three European countries: **Germany (DE), France (FR), and Spain (ES)**. 

The system includes:
1.  **Forecasting:** * **Core:** SARIMA with expanding window backtesting.
    * **Bonus:** GRU/LSTM Neural Network (Input 168h -> Output 24h).
2.  **Anomaly Detection:** * Statistical (Rolling Z-Score).
    * Supervised ML Classifier (LightGBM) using "Silver Labels".
3.  **Live Simulation:** * Simulates a streaming environment with drift detection.
    * **Adaptation Strategies:** Rolling SARIMA Refit vs. Tiny Neural Fine-Tuning.
4.  **Dashboard:** Interactive Streamlit app for monitoring.

## 🛠️ Setup & Installation

1.  **Environment:**
    Ensure you have Python 3.8+ installed. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  2.  **Data:**
    * Download the OPSD (Open Power Systems Data) Time Series CSV from the [OPSD repository](https://data.open-power-system-data.org/time_series/2020-10-06).
    * The required file is `time_series_60min_singleindex.csv` (hourly resolution).
    * Place the file in the `data/` directory.
    * Ensure the filename matches the `data_path` setting in `config.yaml` (default: `data/time_series_60min_singleindex.csv`).
    * The data should contain columns: `utc_timestamp`, `DE_load_actual_entsoe_transparency`, `FR_load_actual_entsoe_transparency`, and `ES_load_actual_entsoe_transparency`.


Run the scripts in the following order to generate all artifacts:

### Phase 1: Ingestion & Analysis
1.  `python src/load_opsd.py` - Cleans raw data and saves per-country CSVs.
2.  `python src/decompose_acf_pcf.py` - Generates STL plots and performs AIC/BIC grid search.

### Phase 2: Core Forecasting (SARIMA)
3.  `python src/forecast.py` - Runs expanding window backtest (Dev/Test splits).

### Phase 3: Anomaly Detection
4.  `python src/anomaly.py` - Detects statistical anomalies (Z-score > 3.0).
5.  `python src/anomaly_ml.py` - Trains LightGBM classifier and verifies labels.

### Phase 4: Bonus Forecasting (LSTM)
6.  `python src/forecast_lstm.py` - Trains GRU models and generates forecasts.
7.  `python src/lstm_metrics.py` - Calculates comparable metrics for the LSTM models.

### Phase 5: Live Simulation
8.  `python src/live_loop.py` - Simulates "Rolling SARIMA Refit" (DE).
9.  `python src/live_loop_lstm.py` - Simulates "Tiny Neural Fine-tune" (DE).

### Phase 6: Visualization
10. `streamlit run src/dashboard_app.py` - Launches the interactive dashboard.

## Outputs
All artifacts are saved in the `outputs/` directory:
* `*_forecasts_test.csv`: Model predictions.
* `*_anomalies.csv`: Detected anomalies.
* `*_online_updates.csv`: Logs of model adaptation events.
* `test_metrics_comparison.csv`: Final performance table.

## Expected Performance Metrics

The following table shows typical test set performance metrics for SARIMA models across all three countries:

| Country | Test MASE | Test SMAPE (%) | Test RMSE | Test Coverage 80% |
|---------|-----------|----------------|-----------|-------------------|
| DE      | 0.68      | 6.07           | 4270.67   | 85.08%            |
| FR      | 0.72      | 5.03           | 3549.62   | 88.02%            |
| ES      | 0.73      | 5.01           | 1846.04   | 84.13%            |

### LSTM Performance Metrics (Bonus)

| Country | Test MASE | Test SMAPE (%) | Test RMSE | Test Coverage 80% |
|---------|-----------|----------------|-----------|-------------------|
| DE      | 0.459     | 4.16           | 2911.10   | 81.01%            |
| FR      | 0.8548    | 5.86           | 4225.92   | 79.67%            |
| ES      | 0.6308    | 4.50           | 1625.57   | 76.89%            |

**Metric Definitions:**
- **MASE (Mean Absolute Scaled Error)**: Values < 1.0 indicate the model outperforms a naive seasonal forecast. Lower is better.
- **SMAPE (Symmetric Mean Absolute Percentage Error)**: Percentage error metric, typically 5-7% for well-performing load forecasts.
- **RMSE (Root Mean Squared Error)**: Absolute error in load units (MW). Varies by country size.
- **Coverage 80%**: Percentage of actual values falling within the 80% prediction interval. Target is ~80%.

These metrics are saved in `outputs/test_metrics_comparison.csv`and `outputs/lstm_metrics.csv` after running the forecasting pipeline.

## Model Architecture Details

### SARIMA (Seasonal ARIMA)
- **Model Selection**: Grid search over SARIMA orders using AIC/BIC criteria
- **Grid Search Space**: 
  - AR order (p): [0, 1, 2]
  - Differencing (d): [0, 1]
  - MA order (q): [0, 1, 2]
  - Seasonal AR (P): [0, 1]
  - Seasonal differencing (D): [0, 1]
  - Seasonal MA (Q): [0, 1]
  - Seasonality (s): 24 hours
- **Backtesting**: Expanding window approach with 24-hour stride and 24-hour forecast horizon
- **Adaptation**: Rolling refit strategy in live simulation (refits model when drift detected)

### LSTM/GRU Neural Network
- **Architecture**: 
  - Input: 168 hours (7 days) of historical load data
  - Layer 1: GRU with 32 units (return_sequences=False)
  - Layer 2: Dense layer with 32 units (ReLU activation)
  - Output: Dense layer with 24 units (24-hour forecast)
- **Training**: 
  - Optimizer: Adam (learning_rate=0.001)
  - Loss: Mean Squared Error (MSE)
  - Epochs: 10
  - Batch size: 32
  - Validation split: 10%
- **Data Preprocessing**: MinMaxScaler (fitted on training data only)
- **Uncertainty Estimation**: 80% prediction intervals based on residual standard deviation from training set
- **Adaptation**: Tiny fine-tuning approach in live simulation (updates model weights incrementally)

### Anomaly Detection
- **Statistical Method**: Rolling Z-score with window of 336 hours (14 days), threshold of |z| ≥ 3.0
- **ML Method**: LightGBM classifier trained on "silver labels" (z-score thresholds: high=3.5, low=2.5)
- **Features**: Temporal features (hour, day of week, month) + rolling statistics

## Configuration
Project settings (SARIMA grids, thresholds, paths) are managed in `config.yaml`.
