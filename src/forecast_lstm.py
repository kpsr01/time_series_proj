import pandas as pd
import numpy as np
import yaml
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import joblib
from metrics import calculate_metrics

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def create_sequences(data, input_width=168, label_width=24):
    """
    Converts a time series array into X (history) and y (future) samples.
    X: (num_samples, 168, 1)
    y: (num_samples, 24) -> Flattened to match model output
    """
    X, y = [], []
    for i in range(len(data) - input_width - label_width + 1):
        X.append(data[i : i + input_width]) 
        y.append(data[i + input_width : i + input_width + label_width].flatten())
    return np.array(X), np.array(y)

def build_model(input_width, label_width):
   
    model = Sequential([
        Input(shape=(input_width, 1)),
        GRU(32, return_sequences=False), 
        Dense(32, activation='relu'),
        Dense(label_width)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def main():
    config = load_config()
    countries = config['countries']
    output_dir = config['outputs_dir']
    
    input_width = 168 # 7 days history
    label_width = 24  # 24h forecast
    
    metrics_summary = []

    for country in countries:
        print(f"\n--- Training LSTM/GRU for {country} ---")
        
        # 1. Load Data
        df_path = os.path.join(config['processed_data_dir'], f"cleaned_{country}.csv")
        df = pd.read_csv(df_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Ensure continuous
        full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')
        df = df.reindex(full_idx)
        df['load'] = df['load'].interpolate(method='linear', limit_direction='both')
        
        data = df['load'].values.reshape(-1, 1)
        
        # 2. Scale Data
        scaler = MinMaxScaler()
        scaler.fit(data[:int(len(data)*config['split']['train_ratio'])]) # Fit only on train
        data_scaled = scaler.transform(data)
        
        # 3. Create Splits
        n = len(data)
        train_end = int(n * config['split']['train_ratio'])
        dev_end = int(n * (config['split']['train_ratio'] + config['split']['dev_ratio']))
        
        # X_train uses data up to train_end
        train_data = data_scaled[:train_end]
        X_train, y_train = create_sequences(train_data, input_width, label_width)
        
        # 4. Train Model
        print(f"  Training on {len(X_train)} samples...")
        model = build_model(input_width, label_width)
        
        history = model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, validation_split=0.1)
        
        # Save Model & Scaler
        model.save(os.path.join(output_dir, f"{country}_lstm_model.keras"))
        joblib.dump(scaler, os.path.join(output_dir, f"{country}_scaler.pkl"))
        
        # 5. Estimate Uncertainty (Residuals on Train)
        train_preds = model.predict(X_train, verbose=0)
        train_errs = y_train - train_preds # Shapes now match: (N, 24) - (N, 24)
        
        # Std dev of error (in scaled space)
        sigma_scaled = np.std(train_errs)
        
        # 6. Forecast Loop (Dev + Test)
        eval_start_idx = train_end - input_width
        eval_data = data_scaled[eval_start_idx:]
        X_eval, y_eval_true = create_sequences(eval_data, input_width, label_width)
        
        print(f"  Generating forecasts for {len(X_eval)} steps (Dev+Test)...")
        y_eval_pred_scaled = model.predict(X_eval, verbose=0)
        
        predictions_list = []
        start_ts_idx = train_end
        
        scale_factor = scaler.data_max_[0] - scaler.data_min_[0]
        sigma_raw = sigma_scaled * scale_factor
        
        for i in range(len(X_eval)):
            curr_idx = start_ts_idx + i
            
            if curr_idx + label_width > len(df):
                break
                
            # Inverse scale
            pred_raw = scaler.inverse_transform(y_eval_pred_scaled[i].reshape(-1, 1)).flatten()
            true_raw = scaler.inverse_transform(y_eval_true[i].reshape(-1, 1)).flatten()
            
            # Intervals
            lo = pred_raw - 1.28 * sigma_raw 
            hi = pred_raw + 1.28 * sigma_raw
            
            ts_range = df.index[curr_idx : curr_idx + label_width]
            
            for h in range(label_width):
                predictions_list.append({
                    'timestamp': ts_range[h],
                    'y_true': true_raw[h],
                    'yhat': pred_raw[h],
                    'lo': lo[h],
                    'hi': hi[h],
                    'horizon': h + 1,
                    'split': 'dev' if curr_idx < dev_end else 'test'
                })
                
        res_df = pd.DataFrame(predictions_list)
        
        # Save Test Results
        test_res = res_df[res_df['split'] == 'test'].copy()
        test_res.to_csv(os.path.join(output_dir, f"{country}_forecasts_lstm_test.csv"), index=False)
        
        # Metrics
        train_series = df['load'].iloc[:train_end].values
        met = calculate_metrics(test_res, train_series)
        
        print(f"  [LSTM Test] MASE: {met['MASE']} | MAPE: {met['MAPE']}% | Cov: {met['Coverage_80']}%")
        
        metrics_summary.append({
            "Country": country,
            "Model": "LSTM",
            "Test_MASE": met['MASE'],
            "Test_Cov80": met['Coverage_80']
        })

    pd.DataFrame(metrics_summary).to_csv(os.path.join(output_dir, "lstm_metrics.csv"), index=False)
    print("\nLSTM Forecasting Complete.")

if __name__ == "__main__":
    main()