import pandas as pd
import yaml
import os

def load_config():
    """Load project configuration from yaml."""
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def clean_and_save_country(raw_df, country_code, config):
    """Extracts, cleans, and saves data for a single country."""
    
    
    target_load_col = None
    
    
    
    load_col_candidate = f"{country_code}_load_actual_entsoe_transparency"
    
    if load_col_candidate not in raw_df.columns:
        print(f"Warning: Column {load_col_candidate} not found. Checking alternatives...")
        candidates = [c for c in raw_df.columns if c.startswith(country_code) and "load_actual" in c]
        if candidates:
            load_col_candidate = candidates[0]
            print(f"Found alternative: {load_col_candidate}")
        else:
            print(f"Skipping {country_code}: No load column found.")
            return

    
    df = raw_df[['utc_timestamp', load_col_candidate]].copy()
    
    df.rename(columns={
        'utc_timestamp': config['column_mapping']['utc_timestamp'],
        load_col_candidate: config['column_mapping']['DE_load_actual_entsoe_transparency'] 
    }, inplace=True)

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    original_len = len(df)
    df.dropna(subset=['load'], inplace=True)
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    print(f"[{country_code}] Dropped {original_len - len(df)} rows with missing values.")

    out_dir = config['processed_data_dir']
    out_path = os.path.join(out_dir, f"cleaned_{country_code}.csv")
    df.to_csv(out_path, index=False)
    print(f"[{country_code}] Saved {len(df)} rows to {out_path}")
    
    print(f"[{country_code}] Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")

def main():
    config = load_config()
    raw_path = config['data_path']
    
    print(f"Loading raw data from {raw_path}...")
    try:
        raw_df = pd.read_csv(raw_path, low_memory=False)
    except FileNotFoundError:
        print(f"Error: File not found at {raw_path}. Please check config.yaml or file placement.")
        return

    countries = config['countries']
    
    for country in countries:
        clean_and_save_country(raw_df, country, config)

if __name__ == "__main__":
    main()