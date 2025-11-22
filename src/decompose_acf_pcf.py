import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import yaml
import json
import os
import itertools
import warnings

warnings.filterwarnings("ignore")

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def plot_stl(df, country, output_dir):
    """
    Generates STL decomposition plot (Seasonal-Trend).
    Period = 24 (Daily seasonality).
    """
    # Use last 30 days for clear visualization
    subset = df.iloc[-24*30:].copy()
    subset.set_index('timestamp', inplace=True)
    
    res = seasonal_decompose(subset['load'], model='additive', period=24)
    
    fig = res.plot()
    fig.set_size_inches(12, 10)
    plt.suptitle(f'STL Decomposition: {country} (Last 30 Days)', y=1.02)
    plt.savefig(os.path.join(output_dir, f'{country}_stl.png'), bbox_inches='tight')
    plt.close()
    print(f"[{country}] STL plot saved.")

def plot_acf_pacf_diagnostics(df, country, output_dir):
    """
    Plots ACF and PACF for raw, d=1, and D=1 (seasonal) differencing.
    """
    y = df['load'].iloc[-24*14:]
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    
    plot_acf(y, lags=48, ax=axes[0, 0], title=f'{country} Raw ACF')
    plot_pacf(y, lags=48, ax=axes[0, 1], title=f'{country} Raw PACF')
    
    y_d1 = y.diff().dropna()
    plot_acf(y_d1, lags=48, ax=axes[1, 0], title=f'{country} d=1 ACF')
    plot_pacf(y_d1, lags=48, ax=axes[1, 1], title=f'{country} d=1 PACF')
    
    y_sd1 = y.diff(24).dropna()
    plot_acf(y_sd1, lags=48, ax=axes[2, 0], title=f'{country} D=1 (s=24) ACF')
    plot_pacf(y_sd1, lags=48, ax=axes[2, 1], title=f'{country} D=1 (s=24) PACF')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{country}_acf_pacf.png'))
    plt.close()
    print(f"[{country}] ACF/PACF plots saved.")

def optimize_sarima_grid(df, country, config):
    """
    Grid search for SARIMA order based on BIC.
    Uses a subset of data for speed.
    """
    print(f"[{country}] Starting Grid Search")
    
    train_subset = df['load'].iloc[-2000:].values
    
    
    ps = [0, 1, 2]
    ds = [0, 1]
    qs = [0, 1, 2]
    Ps = [0, 1]
    Ds = [1] 
    Qs = [0, 1]
    s = 24
    
    best_bic = float("inf")
    best_order = None
    best_seasonal_order = None
    
    param_combinations = list(itertools.product(ps, ds, qs, Ps, Ds, Qs))
    
    results = []

    for params in param_combinations:
        p, d, q, P, D, Q = params
        
        if p == 0 and q == 0 and P == 0 and Q == 0:
            continue
            
        try:
            model = sm.tsa.statespace.SARIMAX(train_subset,
                                            order=(p, d, q),
                                            seasonal_order=(P, D, Q, s),
                                            enforce_stationarity=False,
                                            enforce_invertibility=False,
                                            trend='c') 
            
            results_opt = model.fit(disp=False, maxiter=50)
            
            results.append({
                'order': (p, d, q),
                'seasonal_order': (P, D, Q, s),
                'aic': results_opt.aic,
                'bic': results_opt.bic
            })
            
            if results_opt.bic < best_bic:
                best_bic = results_opt.bic
                best_order = (p, d, q)
                best_seasonal_order = (P, D, Q, s)
                
        except Exception as e:
            continue

    results_df = pd.DataFrame(results)
    results_df.sort_values('bic', inplace=True)
    
    top5_path = f"outputs/{country}_top5_grid.csv"
    results_df.head(5).to_csv(top5_path, index=False)
    print(f"[{country}] Top 5 models saved to {top5_path}")
    print(f"[{country}] Best Model: Order={best_order}, Seasonal={best_seasonal_order} (BIC={best_bic:.2f})")
    
    return best_order, best_seasonal_order

def main():
    config = load_config()
    
    processed_dir = config['processed_data_dir']
    output_dir = config['outputs_dir']
    countries = config['countries']
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    final_orders = {}
    
    for country in countries:
        print(f"\n--- Processing {country} ---")
        file_path = os.path.join(processed_dir, f"cleaned_{country}.csv")
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        plot_stl(df, country, output_dir)
        
        plot_acf_pacf_diagnostics(df, country, output_dir)
        
        
        train_size = int(len(df) * config['split']['train_ratio'])
        train_df = df.iloc[:train_size]
        
        order, seasonal_order = optimize_sarima_grid(train_df, country, config)
        
        final_orders[country] = {
            'order': order,
            'seasonal_order': seasonal_order
        }
    
    orders_path = os.path.join(output_dir, "model_orders.json")
    with open(orders_path, "w") as f:
        json.dump(final_orders, f, indent=4)
    
    print(f"\nSuccess! Model orders saved to {orders_path}")

if __name__ == "__main__":
    main()