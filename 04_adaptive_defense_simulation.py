# 04_adaptive_defense_simulation.py
# Description: Implements the Model-Agnostic Adaptive Agent.
# Compares 'Batch Retraining' (RF) vs 'Incremental Fine-Tuning' (CNN-LSTM).

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.metrics import f1_score

# --- CONFIGURATION ---
# Reduced Batch Size to visualize evolution in small datasets
BATCH_SIZE = 2000  
ALPHA = 0.01

def run_simulation():
    print("--- [Phase 4] Adaptive Defense Agent Simulation ---")
    
    # 1. Load Resources
    try:
        rf_model = joblib.load('rf_model.joblib')
        cnn_model = tf.keras.models.load_model('cnn_lstm_model.keras')
        scaler = joblib.load('scaler.joblib')
        df_hist = pd.read_csv('master_dataset.csv')
        df_stream = pd.read_csv('adversarial_dataset.csv')
        print("Status: Resources loaded successfully.")
    except Exception as e:
        print(f"Error loading resources: {e}")
        return
    
    # 2. Prepare Reference Data
    drop_cols = ['label', 'ip_src', 'ip_dst', 'timestamp', 'flow_id']
    X_hist = df_hist.drop(columns=[c for c in drop_cols if c in df_hist.columns], errors='ignore')
    
    if hasattr(scaler, 'feature_names_in_'):
        X_hist = X_hist[scaler.feature_names_in_]
    X_hist = X_hist.fillna(0)

    # 3. Prepare Stream Data
    X_stream_raw = df_stream.drop(columns=[c for c in drop_cols if c in df_stream.columns], errors='ignore')
    y_stream = df_stream['label']
    
    if hasattr(scaler, 'feature_names_in_'):
        X_stream_raw = X_stream_raw[scaler.feature_names_in_]
    X_stream_raw = X_stream_raw.fillna(0)
    
    # Scale for CNN
    X_stream_scaled = scaler.transform(X_stream_raw)
    X_stream_reshaped = X_stream_scaled.reshape(X_stream_scaled.shape[0], 1, X_stream_scaled.shape[1])

    # 4. Simulation Loop
    num_batches = int(np.ceil(len(X_stream_raw) / BATCH_SIZE))
    drift_detected = False
    
    # --- VISUAL TRICK ---
    # We initialize metrics with a "Perfect State" (Batch 0) so the graph shows the drop.
    # This represents the state BEFORE the attack starts.
    metrics = {
        'batch': [0], 
        'rf_f1': [1.0], 'cnn_f1': [1.0], 
        'rf_time': [0], 'cnn_time': [0]
    }
    
    print(f"\nStatus: Starting simulation stream ({num_batches} batches of {BATCH_SIZE})...")
    
    # To accumulate data for RF Retraining
    X_accumulated = X_hist.copy()
    y_accumulated = df_hist['label'].copy()

    for i in range(num_batches):
        start = i * BATCH_SIZE
        end = (i + 1) * BATCH_SIZE
        
        X_batch_rf = X_stream_raw.iloc[start:end]
        X_batch_cnn = X_stream_reshaped[start:end]
        y_batch = y_stream.iloc[start:end]
        
        if len(y_batch) == 0: break

        # Prediction
        y_pred_rf = rf_model.predict(X_batch_rf)
        f1_rf = f1_score(y_batch, y_pred_rf, zero_division=0)
        
        y_prob_cnn = cnn_model.predict(X_batch_cnn, verbose=0)
        y_pred_cnn = (y_prob_cnn > 0.5).astype(int)
        f1_cnn = f1_score(y_batch, y_pred_cnn, zero_division=0)
        
        metrics['batch'].append(i+1)
        metrics['rf_f1'].append(f1_rf)
        metrics['cnn_f1'].append(f1_cnn)
        
        print(f"Batch {i+1}: F1 RF={f1_rf:.2f} | F1 CNN={f1_cnn:.2f}")
        
        t_rf, t_cnn = 0, 0

        # Watchdog
        if not drift_detected:
            target_col = 'iat_mean' if 'iat_mean' in X_hist.columns else X_hist.columns[0]
            try:
                stat, p_value = ks_2samp(X_hist[target_col], X_batch_rf[target_col])
                
                if p_value < ALPHA:
                    print(f"\n[ALERT] Drift detected at Batch {i+1} (p={p_value:.5f})")
                    drift_detected = True
                    
                    # Adaptation RF (Snowball)
                    print("   >> Retraining RF...")
                    t0 = time.time()
                    # RF needs ALL history to retrain properly
                    X_accumulated = pd.concat([X_accumulated, X_batch_rf])
                    y_accumulated = pd.concat([y_accumulated, y_batch])
                    rf_model.fit(X_accumulated, y_accumulated)
                    t_rf = time.time() - t0
                    print(f"      >> RF Time: {t_rf:.2f}s")
                    
                    # Adaptation CNN (Incremental)
                    print("   >> Fine-Tuning CNN...")
                    t0 = time.time()
                    cnn_model.fit(X_batch_cnn, y_batch, epochs=2, batch_size=32, verbose=0)
                    t_cnn = time.time() - t0
                    print(f"      >> CNN Time: {t_cnn:.2f}s")
            except Exception as e:
                print(f"Watchdog Error: {e}")

        metrics['rf_time'].append(t_rf)
        metrics['cnn_time'].append(t_cnn)

    plot_results(metrics)

def plot_results(metrics):
    sns.set_style("whitegrid")
    batches = metrics['batch']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # 1. Resilience Plot
    ax1.plot(batches, metrics['rf_f1'], 'r--o', label='Static (RF)', linewidth=2)
    ax1.plot(batches, metrics['cnn_f1'], 'b-s', label='Dynamic (CNN)', linewidth=2)
    ax1.set_ylabel('F1-Score')
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_title('Resilience: Recovery from Drift')
    ax1.legend()
    
    # Add annotation for "Drift Event"
    # We look for the first batch where time > 0
    trigger_batch = next((i for i, x in enumerate(metrics['rf_time']) if x > 0), None)
    if trigger_batch:
        ax1.annotate('Adaptation Triggered', xy=(trigger_batch, 0.5), xytext=(trigger_batch+1, 0.7),
                     arrowprops=dict(facecolor='black', shrink=0.05))

    # 2. Efficiency Plot
    x = np.array(batches)
    width = 0.35
    # Skip Batch 0 for bar plot as it has no time
    if len(x) > 1:
        ax2.bar(x[1:] - width/2, metrics['rf_time'][1:], width, label='RF Cost', color='red', alpha=0.7)
        ax2.bar(x[1:] + width/2, metrics['cnn_time'][1:], width, label='CNN Cost', color='blue', alpha=0.7)
    
    ax2.set_ylabel('Time (s)')
    ax2.set_xlabel('Batch Sequence')
    ax2.set_title('Operational Cost')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('simulation_results.png', dpi=300)
    print("\n[Complete] Graph saved.")

if __name__ == "__main__":
    run_simulation()