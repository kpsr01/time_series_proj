import numpy as np
import pandas as pd

def mase(y_true, y_pred, y_train, seasonality=24):
    """
    Mean Absolute Scaled Error (MASE).
    """
    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scale = np.mean(naive_errors)
    
    if scale == 0:
        return np.nan
        
    mae = np.mean(np.abs(y_true - y_pred))
    return mae / scale

def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 100 * np.mean(diff)

def calculate_metrics(df_res, train_series, seasonality=24):
    """
    Computes all required metrics for the dataframe.
    """
    y_true = df_res['y_true'].values
    y_pred = df_res['yhat'].values
    
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    mape = 100 * np.mean(np.abs((y_true - y_pred) / y_true))
    
    _smape = smape(y_true, y_pred)
    _mase = mase(y_true, y_pred, train_series, seasonality)
    
    
    if 'lo' in df_res.columns and 'hi' in df_res.columns:
        covered = ((y_true >= df_res['lo']) & (y_true <= df_res['hi']))
        coverage = 100 * np.mean(covered)
    else:
        coverage = np.nan
        
    return {
        "MASE": round(_mase, 4),
        "SMAPE": round(_smape, 2),
        "MSE": round(mse, 2),
        "RMSE": round(rmse, 2),
        "MAPE": round(mape, 2),
        "Coverage_80": round(coverage, 2)
    }