import pandas as pd
import numpy as np
import yaml
import json
import os
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
from metrics import calculate_metrics

warnings.filterwarnings("ignore")

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def run_backtest(series, split_idx, order, seasonal_order, steps_ahead=24):
    """
    Efficient expanding window backtest.
    """
    # 1. Initial Fit on Training Data
    train_data = series.iloc[:split_idx]
    
    print(f"  Fitting initial model on {len(train_data)} records...")
    try:
        model = SARIMAX(train_data, 
                        order=order, 
                        seasonal_order=seasonal_order, 
                        trend='c',
                        enforce_stationarity=False, 
                        enforce_invertibility=False)
        model_res = model.fit(disp=False)
    except Exception as e:
        print(f"  Error fitting model: {e}")
        return pd.DataFrame()
    
    history_res = model_res
    predictions = []
    total_len = len(series)
    current_idx = split_idx
    
    print("  Starting expanding window forecast loop...")
    
    while current_idx < total_len:
        steps = min(steps_ahead, total_len - current_idx)
        
        try:
            fc_obj = history_res.get_forecast(steps=steps)
            fc_mean = fc_obj.predicted_mean
            fc_conf = fc_obj.conf_int(alpha=0.2) 
        except Exception as e:
            print(f"  Error forecasting at idx {current_idx}: {e}")
            break
        
        y_true_chunk = series.iloc[current_idx : current_idx + steps]
        
        for i in range(steps):
            ts = y_true_chunk.index[i]
            
            predictions.append({
                'timestamp': ts,
                'y_true': y_true_chunk.iloc[i],
                'yhat': fc_mean.iloc[i],
                'lo': fc_conf.iloc[i, 0],
                'hi': fc_conf.iloc[i, 1],
                'horizon': i + 1
            })
            
        if current_idx + steps < total_len:
            new_obs = series.iloc[current_idx : current_idx + steps]
            try:
                history_res = history_res.extend(new_obs)
            except Exception as e:
                print(f"  Error extending model at idx {current_idx}: {e}")
                
                break
            
        current_idx += steps
        
        if (current_idx - split_idx) % (24 * 100) == 0:
            print(f"    Processed {current_idx} / {total_len} hours...")

    return pd.DataFrame(predictions)

def main():
    config = load_config()
    countries = config['countries']
    output_dir = config['outputs_dir']
    
    with open(os.path.join(output_dir, "model_orders.json"), "r") as f:
        all_orders = json.load(f)
    
    metrics_summary = []

    for country in countries:
        print(f"\n--- Forecasting {country} ---")
        
        df_path = os.path.join(config['processed_data_dir'], f"cleaned_{country}.csv")
        df = pd.read_csv(df_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        
        full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')
        df = df.reindex(full_idx)
        
        df['load'] = df['load'].interpolate(method='linear', limit_direction='both')
        
        df = df.asfreq('h')
        series = df['load']
        
        n = len(df)
        train_end = int(n * config['split']['train_ratio'])
        dev_end = int(n * (config['split']['train_ratio'] + config['split']['dev_ratio']))
        
        order = tuple(all_orders[country]['order'])
        seasonal_order = tuple(all_orders[country]['seasonal_order'])
        
        print(f"  Order: {order} x {seasonal_order}")
        
        full_forecast_df = run_backtest(series, train_end, order, seasonal_order)
        
        if full_forecast_df.empty:
            print(f"Skipping results for {country} due to errors.")
            continue

        dev_mask = (full_forecast_df['timestamp'] >= df.index[train_end]) & \
                   (full_forecast_df['timestamp'] < df.index[dev_end])
        test_mask = (full_forecast_df['timestamp'] >= df.index[dev_end])
        
        df_dev = full_forecast_df[dev_mask].copy()
        df_test = full_forecast_df[test_mask].copy()
        
        train_series = series.iloc[:train_end].values
        
        met_dev = calculate_metrics(df_dev, train_series)
        met_test = calculate_metrics(df_test, train_series)
        
        print(f"  [Dev]  MASE: {met_dev['MASE']} | MAPE: {met_dev['MAPE']}% | Cov: {met_dev['Coverage_80']}%")
        print(f"  [Test] MASE: {met_test['MASE']} | MAPE: {met_test['MAPE']}% | Cov: {met_test['Coverage_80']}%")
        
        df_dev['train_end'] = str(df.index[train_end])
        df_test['train_end'] = str(df.index[train_end])
        
        df_dev.to_csv(os.path.join(output_dir, f"{country}_forecasts_dev.csv"), index=False)
        df_test.to_csv(os.path.join(output_dir, f"{country}_forecasts_test.csv"), index=False)
        
        metrics_summary.append({
            "Country": country,
            "Test_MASE": met_test['MASE'],
            "Test_SMAPE": met_test['SMAPE'],
            "Test_RMSE": met_test['RMSE'],
            "Test_Cov80": met_test['Coverage_80']
        })

    summary_df = pd.DataFrame(metrics_summary)
    summary_df.to_csv(os.path.join(output_dir, "test_metrics_comparison.csv"), index=False)
    print("\nAll forecasts complete. Summary saved.")

if __name__ == "__main__":
    main()