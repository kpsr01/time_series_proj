import pandas as pd
import numpy as np
import yaml
import os
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
import time

warnings.filterwarnings("ignore")

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def fit_sarima(series, order, seasonal_order):
    """Helper to fit SARIMA on a specific series."""
    try:
        model = SARIMAX(series, 
                        order=order, 
                        seasonal_order=seasonal_order, 
                        trend='c',
                        enforce_stationarity=False, 
                        enforce_invertibility=False)
        res = model.fit(disp=False, maxiter=50)
        return res
    except Exception as e:
        print(f"    [Error] Fit failed: {e}")
        return None

def main():
    config = load_config()
    country = config['countries'][0]
    print(f"\n--- Starting Live Simulation for {country} ---")
    
    df_path = os.path.join(config['processed_data_dir'], f"cleaned_{country}.csv")
    df = pd.read_csv(df_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')
    df = df.reindex(full_idx)
    df['load'] = df['load'].interpolate(method='linear', limit_direction='both')
    series = df['load']
    
    sim_hours = config['live']['sim_hours'] # 2000
    history_start_days = config['live']['history_start_days'] # 120
    
   
    sim_start_idx = len(series) - sim_hours
    if sim_start_idx < 24 * history_start_days:
        raise ValueError("Not enough history for the requested simulation configuration.")
        
    print(f"  Simulation Period: Last {sim_hours} hours of data.")
    print("  Rolling SARIMA Refit")
    
    with open(os.path.join(config['outputs_dir'], "model_orders.json"), "r") as f:
        import json
        all_orders = json.load(f)
        order = tuple(all_orders[country]['order'])
        seasonal_order = tuple(all_orders[country]['seasonal_order'])

    current_idx = sim_start_idx
    
    # Drift Monitors
    z_history = [] 
    ewma_z = 0.0
    alpha = config['live']['drift_alpha']
    drift_window = config['live']['drift_window'] # 720h (30 days)
    
    updates_log = []
    
   
    train_window = 90 * 24
    
    # Initial fit
    history_series = series.iloc[current_idx - train_window : current_idx]
    print("  Initializing model...")
    model_res = fit_sarima(history_series, order, seasonal_order)
    
    
    next_forecast_val = model_res.forecast(steps=1).iloc[0]
    
    print("  Entering Live Loop")
    
    for i in range(sim_hours):
        current_ts = series.index[current_idx]
        actual_val = series.iloc[current_idx]
        
        resid = actual_val - next_forecast_val
       
        
        sigma = np.std(model_res.resid[-drift_window:]) if len(model_res.resid) > 0 else 1.0
        if sigma == 0: sigma = 1.0
        
        z_score = resid / sigma
        abs_z = abs(z_score)
        
        ewma_z = alpha * abs_z + (1 - alpha) * ewma_z
        z_history.append(abs_z)
        if len(z_history) > drift_window:
            z_history.pop(0)
            
        if len(z_history) >= 24:
            drift_threshold = np.percentile(z_history, config['live']['drift_percentile'] * 100)
        else:
            drift_threshold = 5.0 
            
        triggered = False
        reason = ""
        
        if ewma_z > drift_threshold and len(z_history) > 168:
            triggered = True
            reason = "Drift"
            
        is_midnight = (current_ts.hour == 0)
        if is_midnight:
            triggered = True
            reason = "Scheduled" if not reason else "Scheduled+Drift"
            
        if triggered:
            t0 = time.time()
            train_start = current_idx - train_window
            train_end = current_idx + 1 
            
            new_history = series.iloc[train_start : train_end]
            
           
            model_res = fit_sarima(new_history, order, seasonal_order)
            
            duration = round(time.time() - t0, 2)
            
            updates_log.append({
                'timestamp': current_ts,
                'strategy': 'Rolling Refit',
                'reason': reason,
                'duration_s': duration,
                'ewma_z': round(ewma_z, 3),
                'threshold': round(drift_threshold, 3)
            })
            
            
        else:
            
            new_obs = series.iloc[current_idx : current_idx+1]
            model_res = model_res.extend(new_obs)
        

        next_forecast_val = model_res.forecast(steps=1).iloc[0]
        
        current_idx += 1
        
        if i % 168 == 0: # Weekly print
            print(f"    [Hour {i}/{sim_hours}] {current_ts} | EWMA: {ewma_z:.2f} | Thresh: {drift_threshold:.2f} | Updates: {len(updates_log)}")

    log_df = pd.DataFrame(updates_log)
    out_path = os.path.join(config['outputs_dir'], f"{country}_online_updates.csv")
    log_df.to_csv(out_path, index=False)
    
    print(f"\nSimulation Complete. Total Updates: {len(log_df)}")
    print(f"Updates saved to {out_path}")

if __name__ == "__main__":
    main()