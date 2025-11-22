import pandas as pd
import numpy as np
import os
import yaml
import json
import lightgbm as lgb
from sklearn.metrics import precision_recall_curve, auc, f1_score

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def create_silver_labels(df, config):
    """
    Creates silver labels
    """
    z = df['z_resid'].abs()
    out_of_bounds = (df['y_true'] < df['lo']) | (df['y_true'] > df['hi'])
    
    high = config['anomaly']['silver_label_thresh_high'] # 3.5
    low = config['anomaly']['silver_label_thresh_low']   # 2.5
    
    conditions = [
        (z >= high) | (out_of_bounds & (z >= low)), # Positive
        (z < 1.0) & (~out_of_bounds)                # Negative
    ]
    choices = [1, 0]
    
    df['silver_label'] = np.select(conditions, choices, default=-1)
    return df

def extract_features(df):
    """
    Extract features for ML classifier.
    """
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    
    df['yhat_err'] = df['y_true'] - df['yhat']
   
    for lag in [1, 2, 3, 24]:
        df[f'resid_lag_{lag}'] = df['yhat_err'].shift(lag)
        
    df['resid_roll_mean_24'] = df['yhat_err'].rolling(24).mean()
    df['resid_roll_std_24'] = df['yhat_err'].rolling(24).std()
    
    df = df.dropna().copy()
    return df

def train_and_eval(country, config):
    output_dir = config['outputs_dir']
    
    
    dev_path = os.path.join(output_dir, f"{country}_forecasts_dev.csv")
    test_path = os.path.join(output_dir, f"{country}_forecasts_test.csv")
    
    df_dev = pd.read_csv(dev_path)
    df_test = pd.read_csv(test_path)
    df_dev['timestamp'] = pd.to_datetime(df_dev['timestamp'])
    df_test['timestamp'] = pd.to_datetime(df_test['timestamp'])
   
    df_full = pd.concat([df_dev, df_test], axis=0).reset_index(drop=True)
    
    df_full['residual'] = df_full['y_true'] - df_full['yhat']
    roll_mean = df_full['residual'].rolling(window=336, min_periods=168).mean()
    roll_std = df_full['residual'].rolling(window=336, min_periods=168).std()
    df_full['z_resid'] = (df_full['residual'] - roll_mean) / roll_std
    df_full['z_resid'] = df_full['z_resid'].fillna(0)
    
    df_full = create_silver_labels(df_full, config)
    
    df_feats = extract_features(df_full)
    
    
    train_mask = (df_feats['timestamp'] < df_test['timestamp'].min())
    test_mask = (df_feats['timestamp'] >= df_test['timestamp'].min())
    
    train_data = df_feats[train_mask & (df_feats['silver_label'] != -1)]
    test_data = df_feats[test_mask & (df_feats['silver_label'] != -1)] # Evaluation only on clearly labeled data
    
    # Features to use
    features = [c for c in df_feats.columns if 'resid_' in c or c in ['hour', 'dayofweek']]
    X_train = train_data[features]
    y_train = train_data['silver_label']
    
    X_test = test_data[features]
    y_test = test_data['silver_label']
    
    #Train LightGBM
    clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
    clf.fit(X_train, y_train)
    
    #Predict
    y_probs = clf.predict_proba(X_test)[:, 1]
    
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    pr_auc = auc(recall, precision)
    
    
    valid_idxs = np.where(precision >= 0.8)[0]
    if len(valid_idxs) > 0:
        target_idx = valid_idxs[0]
        f1_at_p80 = 2 * (precision[target_idx] * recall[target_idx]) / (precision[target_idx] + recall[target_idx])
    else:
        f1_at_p80 = 0.0
        
    print(f"  [{country}] ML Result: PR-AUC={pr_auc:.4f}, F1@P0.8={f1_at_p80:.4f}")
    
    
    sample_pos = df_feats[df_feats['silver_label'] == 1].sample(n=min(50, len(df_feats[df_feats['silver_label']==1])), random_state=42)
    sample_neg = df_feats[df_feats['silver_label'] == 0].sample(n=min(50, len(df_feats[df_feats['silver_label']==0])), random_state=42)
    
    verification_df = pd.concat([sample_pos, sample_neg]).sort_values('timestamp')
    verification_path = os.path.join(output_dir, f"{country}_anomaly_labels_verified.csv")
    verification_df[['timestamp', 'y_true', 'yhat', 'z_resid', 'silver_label']].to_csv(verification_path, index=False)
    
    return {
        "country": country,
        "pr_auc": round(pr_auc, 4),
        "f1_p80": round(f1_at_p80, 4)
    }

def main():
    config = load_config()
    countries = config['countries']
    results = []
    
    for country in countries:
        res = train_and_eval(country, config)
        results.append(res)
        
    # Save ML Eval Summary
    with open(os.path.join(config['outputs_dir'], "anomaly_ml_eval.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nML Anomaly classification complete.")

if __name__ == "__main__":
    main()