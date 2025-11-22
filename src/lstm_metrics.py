import pandas as pd
import numpy as np
import os
import yaml
from metrics import calculate_metrics

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    countries = config['countries']
    output_dir = config['outputs_dir']
    
    metrics_summary = []

    print("--- Calculating Full LSTM Metrics ---")

    for country in countries:
        # 1. Load the Saved LSTM Forecasts
        fc_path = os.path.join(output_dir, f"{country}_forecasts_lstm_test.csv")
        
        if not os.path.exists(fc_path):
            print(f"Warning: No forecast file found for {country}. Skipping.")
            continue
            
        df_test = pd.read_csv(fc_path)
        
        # 2. Load Training Data (Required for MASE scaling)
        data_path = os.path.join(config['processed_data_dir'], f"cleaned_{country}.csv")
        df_data = pd.read_csv(data_path)
        
        # Identify Training Split (Same logic as before)
        n = len(df_data)
        train_end = int(n * config['split']['train_ratio'])
        train_series = df_data['load'].iloc[:train_end].values
        
        # 3. Calculate ALL Metrics
        met = calculate_metrics(df_test, train_series)
        
        print(f"[{country}] MASE: {met['MASE']} | SMAPE: {met['SMAPE']}% | RMSE: {met['RMSE']}")
        
        metrics_summary.append({
            "Country": country,
            "Model": "LSTM",
            "Test_MASE": met['MASE'],
            "Test_SMAPE": met['SMAPE'],
            "Test_MSE": met['MSE'],
            "Test_RMSE": met['RMSE'],
            "Test_MAPE": met['MAPE'],
            "Test_Cov80": met['Coverage_80']
        })

    # 4. Overwrite the incomplete CSV
    out_path = os.path.join(output_dir, "lstm_metrics.csv")
    pd.DataFrame(metrics_summary).to_csv(out_path, index=False)
    print(f"\nSuccess! Full metrics saved to {out_path}")

if __name__ == "__main__":
    main()