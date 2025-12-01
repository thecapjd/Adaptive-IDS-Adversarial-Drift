# 00_data_harmonization.py
# Description: ETL Pipeline to merge Specific UNSW-NB15 & CIC-IDS-2017 files.
# Output: 'master_dataset.csv' with Standardized English Feature Names.

def main():
    print("--- [Phase 0] Data Harmonization (Target: English Schema) ---")
    
    # CORRECCIÓN PARA COLAB/JUPYTER:
    # En lugar de usar __file__, usamos os.getcwd() que funciona en notebooks
    current_dir = os.getcwd()
    print(f"Working Directory: {current_dir}")
    
    # 1. Detect Files
    # A. Check Specific CIC Files
    found_cic = []
    for fname in REQUIRED_CIC_FILES:
        fpath = os.path.join(current_dir, fname)
        if os.path.exists(fpath): found_cic.append(fpath)
        else: print(f"Warning: Missing CIC file -> {fname}")

    # B. Check Specific UNSW Files
    found_unsw = []
    for fname in REQUIRED_UNSW_FILES:
        fpath = os.path.join(current_dir, fname)
        if os.path.exists(fpath): found_unsw.append(fpath)
        else: print(f"Warning: Missing UNSW file -> {fname}")

    if not found_cic and not found_unsw:
        print("Error: No valid source datasets found from the required list.")
        return

    processed_dfs = []

    # 2. Process Sources
    if found_cic:
        print(f"\nStatus: Processing {len(found_cic)} CIC-IDS-2017 files.")
        for f in found_cic:
            df = load_and_standardize(f, 'CIC')
            if df is not None and not df.empty: processed_dfs.append(df)

    if found_unsw:
        print(f"\nStatus: Processing {len(found_unsw)} UNSW-NB15 files.")
        for f in found_unsw:
            df = load_and_standardize(f, 'UNSW')
            if df is not None and not df.empty: processed_dfs.append(df)

    if not processed_dfs:
        print("Error: No data remaining after filtering.")
        return # Cambiado sys.exit() por return para no matar el kernel de Colab

    # 3. Merge
    print("\nStatus: Merging datasets...")
    df_master = pd.concat(processed_dfs, ignore_index=True)
    
    # 4. Global Sanitation
    print(f"Total Rows (Raw): {len(df_master)}")
    df_master.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_master.dropna(inplace=True)
    
    # 5. Type Enforcement
    print("Status: Enforcing Data Types...")
    
    int_cols = ['protocol', 'port_src', 'port_dst', 
                'bytes_fwd', 'bytes_bwd', 
                'packets_fwd', 'packets_bwd', 
                'window_size_fwd', 'window_size_bwd', 
                'label', 'timestamp']
    
    # Protocol Mapping (String -> IANA Number)
    if df_master['protocol'].dtype == 'O':
        df_master['protocol'] = df_master['protocol'].astype(str).str.lower().map({'tcp': 6, 'udp': 17})
        df_master['protocol'] = df_master['protocol'].fillna(0)

    for col in int_cols:
        if col in df_master.columns:
            df_master[col] = pd.to_numeric(df_master[col], errors='coerce').fillna(0).astype(np.int64)

    float_cols = ['flow_duration', 'iat_mean', 'payload_mean_fwd', 'payload_mean_bwd']
    for col in float_cols:
        if col in df_master.columns:
            df_master[col] = pd.to_numeric(df_master[col], errors='coerce').fillna(0.0).astype(float)

    # 6. Final Select
    final_cols_present = [c for c in FINAL_SCHEMA if c in df_master.columns]
    df_master = df_master[final_cols_present]

    # 7. Save
    print(f"\nFinal Dataset Shape: {df_master.shape}")
    df_master.to_csv(OUTPUT_FILENAME, index=False)
    print(f"SUCCESS: Unified dataset saved to '{OUTPUT_FILENAME}'")
    print("Features:", list(df_master.columns))

if __name__ == "__main__":
    main()