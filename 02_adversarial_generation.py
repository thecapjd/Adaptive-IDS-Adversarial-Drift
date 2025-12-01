# 02_adversarial_generation.py
# Description: Generates adversarial traffic using Statistical Mimicry and Low-Rate modification.
# Input: 'master_dataset.csv' (English Schema)
# Output: 'adversarial_dataset.csv'

import pandas as pd
import numpy as np
import os

# Configuration
INPUT_FILE = 'master_dataset.csv'
OUTPUT_FILE = 'adversarial_dataset.csv'
SEED = 42

# Reproducibility
np.random.seed(SEED)

def generate_attack():
    print("--- [Phase 2] Adversarial Attack Generation ---")
    
    # 1. Load Data
    if not os.path.exists(INPUT_FILE):
        print(f"Error: '{INPUT_FILE}' not found. Please run '00_data_harmonization.py' first.")
        return

    print("Status: Loading Master Dataset...")
    df = pd.read_csv(INPUT_FILE)
    
    # 2. Split Traffic
    # Label 0 = Benign, 1 = PortScan/Malicious
    df_benign = df[df['label'] == 0]
    df_malicious = df[df['label'] == 1].copy()
    
    print(f"  > Benign Samples: {len(df_benign)}")
    print(f"  > Malicious Samples to Mutate: {len(df_malicious)}")

    if df_benign.empty or df_malicious.empty:
        print("Error: Dataset must contain both Benign (0) and Malicious (1) samples.")
        return

    # 3. ATTACK STRATEGY 1: STATISTICAL MIMICRY
    # We replace the attacker's statistical footprint with samples from benign traffic.
    # CRITICAL: Using ENGLISH Column Names
    mimicry_features = [
        'bytes_fwd', 'bytes_bwd', 
        'payload_mean_fwd', 'payload_mean_bwd', 
        'window_size_fwd', 'window_size_bwd',
        'packets_fwd', 'packets_bwd'
    ]

    print("\nStatus: Applying Statistical Mimicry (Morphing)...")
    for col in mimicry_features:
        if col in df.columns:
            # Randomly sample from benign distribution to overwrite malicious values
            mimic_values = np.random.choice(df_benign[col].values, size=len(df_malicious))
            df_malicious[col] = mimic_values
        else:
            print(f"  [Warning] Feature '{col}' not found in dataset. Skipping.")

    # 4. ATTACK STRATEGY 2: LOW-RATE EVASION (Temporal Modification)
    # We increase the time between packets to evade rate-limiting.
    print("\nStatus: Applying Low-Rate Temporal Evasion...")
    SLOW_FACTOR = 10  # 10x slower
    
    # Using ENGLISH Column Names
    if 'iat_mean' in df_malicious.columns:
        df_malicious['iat_mean'] = df_malicious['iat_mean'] * SLOW_FACTOR
        print("  -> Modified 'iat_mean' (Increased delay)")
        
    if 'flow_duration' in df_malicious.columns:
        df_malicious['flow_duration'] = df_malicious['flow_duration'] * SLOW_FACTOR
        print("  -> Modified 'flow_duration'")

    # 5. Save Adversarial Dataset
    df_malicious.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSUCCESS: Adversarial dataset generated at '{OUTPUT_FILE}'")
    print("Features:", list(df_malicious.columns))

if __name__ == "__main__":
    generate_attack()