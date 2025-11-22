import pandas as pd
import numpy as np
import os
import yaml

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def detect_anomalies(df, config):
    """
    Computes residuals, rolling Z-score, and flags anomalies.
    """
    # 1. Compute Residuals
    df['residual'] = df['y_true'] - df['yhat']
    
    # 2. Rolling Z-score
    window = config['anomaly']['z_window']
    min_periods = config['anomaly']['min_periods']
    
    # Compute rolling statistics on residuals
    roll_mean = df['residual'].rolling(window=window, min_periods=min_periods).mean()
    roll_std = df['residual'].rolling(window=window, min_periods=min_periods).std()
    
    # Calculate Z-score
    df['z_resid'] = (df['residual'] - roll_mean) / roll_std
    
    df['z_resid'] = df['z_resid'].fillna(0.0)
    
    # 3. Flag Anomalies
    threshold = config['anomaly']['z_threshold']
    df['flag_z'] = (df['z_resid'].abs() >= threshold).astype(int)
    
    # 4. CUSUM Method
    k = 0.5
    h = 5.0
    
    
    s_pos = np.zeros(len(df))
    s_neg = np.zeros(len(df))
    cusum_flags = np.zeros(len(df))
    
    z_vals = df['z_resid'].values
    
    for i in range(1, len(z_vals)):
        s_pos[i] = max(0, s_pos[i-1] + z_vals[i] - k)
        s_neg[i] = max(0, s_neg[i-1] - z_vals[i] - k)
        
        if s_pos[i] > h or s_neg[i] > h:
            cusum_flags[i] = 1
            
            
    df['flag_cusum'] = cusum_flags.astype(int)
    
    return df

def main():
    config = load_config()
    output_dir = config['outputs_dir']
    countries = config['countries']
    
    for country in countries:
        print(f"Processing anomalies for {country}...")
        
        dev_path = os.path.join(output_dir, f"{country}_forecasts_dev.csv")
        test_path = os.path.join(output_dir, f"{country}_forecasts_test.csv")
        
        if not os.path.exists(dev_path) or not os.path.exists(test_path):
            print(f"  Missing forecast files for {country}. Skipping.")
            continue
            
        df_dev = pd.read_csv(dev_path)
        df_test = pd.read_csv(test_path)
        
        # Mark source to split later
        df_dev['split'] = 'dev'
        df_test['split'] = 'test'
        
        # Concatenate
        df_full = pd.concat([df_dev, df_test], axis=0).reset_index(drop=True)
        df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])
        
        # Detect
        df_scored = detect_anomalies(df_full, config)
        
        df_out = df_scored[df_scored['split'] == 'test'].copy()
        
        cols = ['timestamp', 'y_true', 'yhat', 'z_resid', 'flag_z', 'flag_cusum']
        out_path = os.path.join(output_dir, f"{country}_anomalies.csv")
        df_out[cols].to_csv(out_path, index=False)
        
        n_anom = df_out['flag_z'].sum()
        print(f"  [{country}] Saved. Found {n_anom} statistical anomalies")

if __name__ == "__main__":
    main()