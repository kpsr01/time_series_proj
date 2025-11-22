import streamlit as st
import pandas as pd
import numpy as np
import yaml
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="OPSD PowerDesk", layout="wide")

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def load_data(country, config):
    """Load Test forecasts and Anomaly data for the selected country."""
    output_dir = config['outputs_dir']
    
    
    fc_path = os.path.join(output_dir, f"{country}_forecasts_test.csv")
    if not os.path.exists(fc_path):
        return None, None
        
    df = pd.read_csv(fc_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    anom_path = os.path.join(output_dir, f"{country}_anomalies.csv")
    if os.path.exists(anom_path):
        df_anom = pd.read_csv(anom_path)
        df_anom['timestamp'] = pd.to_datetime(df_anom['timestamp'])
       
        df = pd.merge(df, df_anom[['timestamp', 'flag_z', 'z_resid']], on='timestamp', how='left')
        df['flag_z'] = df['flag_z'].fillna(0)
    else:
        df['flag_z'] = 0
        
    return df

def load_update_log(country, config):
    """Load the online adaptation log (Only exists for the Live country)."""
    log_path = os.path.join(config['outputs_dir'], f"{country}_online_updates.csv")
    if os.path.exists(log_path):
        return pd.read_csv(log_path)
    return pd.DataFrame()

def main():
    config = load_config()
    st.title("⚡ OPSD PowerDesk: Day-Ahead Forecasting")
    
    st.sidebar.header("Configuration")
    
    selected_country = st.sidebar.selectbox("Select Country", config['countries'])
    
    df = load_data(selected_country, config)
    
    if df is None:
        st.error(f"No data found for {selected_country}. Run forecasts first.")
        return

    max_date = df['timestamp'].max()
    min_date = max_date - pd.Timedelta(days=14)
    mask = df['timestamp'] >= min_date
    df_view = df.loc[mask].copy()

    st.subheader(f"Status: {selected_country} Grid")
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    mae = np.mean(np.abs(df_view['y_true'] - df_view['yhat']))
   
    
    mape = 100 * np.mean(np.abs((df_view['y_true'] - df_view['yhat']) / df_view['y_true']))
    
    # Coverage
    covered = ((df_view['y_true'] >= df_view['lo']) & (df_view['y_true'] <= df_view['hi']))
    coverage = 100 * np.mean(covered)
    
    # Anomalies
    anom_count = df_view['flag_z'].sum()
    
    kpi1.metric("MAPE (Last 14d)", f"{mape:.2f}%")
    kpi2.metric("80% PI Coverage", f"{coverage:.1f}%")
    kpi3.metric("Anomalies Detected", int(anom_count), delta_color="inverse")
    kpi4.metric("Last Update", str(max_date.date()))

    st.markdown("### Live Load Monitor")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot Actuals
    ax.plot(df_view['timestamp'], df_view['y_true'], label='Actual Load', color='black', alpha=0.7, linewidth=1.5)
    
    # Plot Forecast
    ax.plot(df_view['timestamp'], df_view['yhat'], label='Forecast (Day-Ahead)', color='blue', linestyle='--', alpha=0.8)
    
    # Plot Cone (Confidence Interval)
    ax.fill_between(df_view['timestamp'], df_view['lo'], df_view['hi'], color='blue', alpha=0.1, label='80% Conf. Interval')
    
    # Plot Anomalies 
    anomalies = df_view[df_view['flag_z'] == 1]
    if not anomalies.empty:
        ax.scatter(anomalies['timestamp'], anomalies['y_true'], color='red', s=50, label='Anomaly (|Z| >= 3.0)', zorder=5)

    ax.set_ylabel("Load (MW)")
    ax.set_title(f"Load Forecast vs Actuals (Last 14 Days) - {selected_country}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)

    
    df_updates = load_update_log(selected_country, config)
    
    st.markdown("### Online Model Adaptation Log")
    if not df_updates.empty:
        st.dataframe(
            df_updates.style.format({
                "duration_s": "{:.2f}",
                "ewma_z": "{:.2f}",
                "threshold": "{:.2f}"
            })
        )
        st.caption(f"Showing {len(df_updates)} adaptation events triggered by Drift or Schedule.")
    else:
        st.info("No online adaptation log found for this country (Offline Mode).")

    st.markdown("---")
    st.markdown("### Global Performance Overview (Test Set)")
    
    comp_path = os.path.join(config['outputs_dir'], "test_metrics_comparison.csv")
    if os.path.exists(comp_path):
        df_comp = pd.read_csv(comp_path)
        st.table(df_comp)

if __name__ == "__main__":
    main()