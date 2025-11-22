import pandas as pd
import numpy as np
import yaml
import os
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import joblib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def create_sequences(data, input_width=168, label_width=24):
   
    X, y = [], []
    if len(data) < input_width + label_width:
        return np.array([]), np.array([])
        
    for i in range(len(data) - input_width - label_width + 1):
        X.append(data[i : i + input_width])
        y.append(data[i + input_width : i + input_width + label_width].flatten())
        
    return np.array(X), np.array(y)

def main():
    config = load_config()
    country = config['countries'][0] # DE
    print(f"\n--- Starting Live Simulation (LSTM Fine-Tuning) for {country} ---")
    
    df_path = os.path.join(config['processed_data_dir'], f"cleaned_{country}.csv")
    df = pd.read_csv(df_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')
    df = df.reindex(full_idx)
    df['load'] = df['load'].interpolate(method='linear', limit_direction='both')
    
    model_path = os.path.join(config['outputs_dir'], f"{country}_lstm_model.keras")
    scaler_path = os.path.join(config['outputs_dir'], f"{country}_scaler.pkl")
    
    if not os.path.exists(model_path):
        print("Model not found. Run forecast_lstm.py first.")
        return
        
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    sim_hours = config['live']['sim_hours'] # 2000
    input_width = 168
    label_width = 24
    
    data_scaled = scaler.transform(df['load'].values.reshape(-1, 1))
    
    sim_start_idx = len(df) - sim_hours
    current_idx = sim_start_idx
    

    for layer in model.layers[:-1]:
        layer.trainable = False
    
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse')
    
    fine_tune_window = 14 * 24 # 14 days
    updates_log = []
    
    z_history = []
    ewma_z = 0.0
    alpha = config['live']['drift_alpha']
    drift_window = config['live']['drift_window']
    

    hist_slice = data_scaled[current_idx - input_width : current_idx]
    curr_input = hist_slice.reshape(1, input_width, 1)
    next_forecast_scaled = model.predict(curr_input, verbose=0)[0] # (24,)
    
    print("  Entering Live Loop...")
    
    for i in range(sim_hours):
        current_ts = df.index[current_idx]
        
        actual_scaled = data_scaled[current_idx][0]
       
        pred_val_scaled = next_forecast_scaled[0] 
        resid = actual_scaled - pred_val_scaled
        
        ewma_z = alpha * abs(resid) + (1 - alpha) * ewma_z 
        
        triggered = False
        reason = ""
        
        if i % 6 == 0:
            triggered = True
            reason = "Scheduled (6h)"
            
        if ewma_z > 0.15: # Empirical threshold for scaled data
            if not triggered:
                triggered = True
                reason = "Drift"

        if triggered:
            t0 = time.time()
            
            train_start = current_idx - fine_tune_window
            train_end = current_idx
            
            subset_data = data_scaled[train_start : train_end]
            X_ft, y_ft = create_sequences(subset_data, input_width, label_width)
            
            if len(X_ft) > 0:
                model.fit(X_ft, y_ft, epochs=1, batch_size=32, verbose=0)
            
            duration = round(time.time() - t0, 2)
            
            updates_log.append({
                'timestamp': current_ts,
                'strategy': 'Tiny Neural Fine-tune',
                'reason': reason,
                'duration_s': duration,
                'ewma_z': round(ewma_z, 4)
            })
            
        hist_slice = data_scaled[current_idx - input_width + 1 : current_idx + 1]
        curr_input = hist_slice.reshape(1, input_width, 1)
        next_forecast_scaled = model.predict(curr_input, verbose=0)[0]
        
        current_idx += 1
        
        if i % 168 == 0:
            print(f"    [Hour {i}/{sim_hours}] {current_ts} | Updates: {len(updates_log)}")

    # Save Logs
    log_df = pd.DataFrame(updates_log)
    out_path = os.path.join(config['outputs_dir'], f"{country}_online_updates_lstm.csv")
    log_df.to_csv(out_path, index=False)
    print(f"\nSimulation Complete. Log saved to {out_path}")

if __name__ == "__main__":
    main()