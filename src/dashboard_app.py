import streamlit as st
import pandas as pd
import numpy as np
import yaml
import os
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(page_title="OPSD PowerDesk", layout="wide")

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def load_data(country, model_type, config):
    """Load forecasts based on Country and Model Type."""
    output_dir = config['outputs_dir']
    
    # Select File based on Model
    if model_type == "SARIMA":
        fc_path = os.path.join(output_dir, f"{country}_forecasts_test.csv")
        anom_path = os.path.join(output_dir, f"{country}_anomalies.csv")
    else: # LSTM
        fc_path = os.path.join(output_dir, f"{country}_forecasts_lstm_test.csv")
        # We didn't run explicit anomaly detection on LSTM, so no separate anomaly file
        anom_path = None 
        
    if not os.path.exists(fc_path):
        return None, None
        
    df = pd.read_csv(fc_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Load Anomalies (Only available for SARIMA usually, unless you ran anomaly.py on LSTM)
    if anom_path and os.path.exists(anom_path):
        df_anom = pd.read_csv(anom_path)
        df_anom['timestamp'] = pd.to_datetime(df_anom['timestamp'])
        df = pd.merge(df, df_anom[['timestamp', 'flag_z', 'z_resid']], on='timestamp', how='left')
        df['flag_z'] = df['flag_z'].fillna(0)
    else:
        df['flag_z'] = 0 # No anomaly flags for LSTM view
        
    return df

def load_update_log(country, model_type, config):
    """Load the online adaptation log."""
    if model_type == "SARIMA":
        log_path = os.path.join(config['outputs_dir'], f"{country}_online_updates.csv")
    else:
        log_path = os.path.join(config['outputs_dir'], f"{country}_online_updates_lstm.csv")
        
    if os.path.exists(log_path):
        return pd.read_csv(log_path)
    return pd.DataFrame()

def main():
    config = load_config()
    st.title("⚡ OPSD PowerDesk: Forecasting Dashboard")
    
    # --- Sidebar ---
    st.sidebar.header("Configuration")
    selected_country = st.sidebar.selectbox("Select Country", config['countries'])
    
    # NEW: Model Selector
    model_type = st.sidebar.radio("Select Model", ["SARIMA", "LSTM"])
    
    # Load Data
    df = load_data(selected_country, model_type, config)
    
    if df is None:
        st.error(f"No data found for {selected_country} ({model_type}). Run forecasts first.")
        return

    # Filter to Last 14 Days for "Live" view
    max_date = df['timestamp'].max()
    min_date = max_date - pd.Timedelta(days=14)
    df_view = df.loc[df['timestamp'] >= min_date].copy()

    # --- KPI Tiles ---
    st.subheader(f"Status: {selected_country} Grid ({model_type})")
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    # Metrics Calculation
    mape = 100 * np.mean(np.abs((df_view['y_true'] - df_view['yhat']) / df_view['y_true']))
    covered = ((df_view['y_true'] >= df_view['lo']) & (df_view['y_true'] <= df_view['hi']))
    coverage = 100 * np.mean(covered)
    anom_count = df_view['flag_z'].sum()
    
    kpi1.metric("MAPE (Last 14d)", f"{mape:.2f}%")
    kpi2.metric("80% PI Coverage", f"{coverage:.1f}%")
    kpi3.metric("Anomalies (Visible)", int(anom_count))
    kpi4.metric("Last Forecast", str(max_date.date()))

    # --- Main Plot ---
    st.markdown("### Live Load Monitor")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df_view['timestamp'], df_view['y_true'], label='Actual Load', color='black', alpha=0.7)
    ax.plot(df_view['timestamp'], df_view['yhat'], label=f'{model_type} Forecast', color='blue', linestyle='--')
    ax.fill_between(df_view['timestamp'], df_view['lo'], df_view['hi'], color='blue', alpha=0.1, label='80% PI')
    
    if anom_count > 0:
        anomalies = df_view[df_view['flag_z'] == 1]
        ax.scatter(anomalies['timestamp'], anomalies['y_true'], color='red', s=50, label='Anomaly', zorder=5)

    ax.set_title(f"Forecast vs Actuals ({model_type})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # --- Online Adaptation Log ---
    df_updates = load_update_log(selected_country, model_type, config)
    
    st.markdown(f"### Online Adaptation Log ({model_type})")
    if not df_updates.empty:
        st.dataframe(df_updates)
        st.caption(f"Strategy: {'Rolling Refit' if model_type=='SARIMA' else 'Tiny Neural Fine-tune'}")
    else:
        st.info(f"No adaptation log found for {model_type} (Offline or not Live country).")

    # --- Global Metrics ---
    st.markdown("---")
    st.markdown("### Global Test Metrics")
    
    # Show different table based on model
    if model_type == "SARIMA":
        comp_path = os.path.join(config['outputs_dir'], "test_metrics_comparison.csv")
    else:
        comp_path = os.path.join(config['outputs_dir'], "lstm_metrics.csv")
        
    if os.path.exists(comp_path):
        st.table(pd.read_csv(comp_path))

if __name__ == "__main__":
    main()