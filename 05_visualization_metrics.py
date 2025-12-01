# 05_visualization_metrics.py
# Description: Generates Confusion Matrices for the Vulnerability Assessment (Phase 3).
# Input: 'adversarial_dataset.csv'
# Output: 'confusion_matrix_comparison.png'

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Configuration
INPUT_FILE = 'adversarial_dataset.csv'
MODEL_RF = 'rf_model.joblib'
MODEL_CNN = 'cnn_lstm_model.keras'
SCALER = 'scaler.joblib'

def plot_confusion_matrices():
    print("--- [Phase 5] Generating Confusion Matrices ---")

    # 1. Load Resources
    try:
        rf_model = joblib.load(MODEL_RF)
        cnn_model = tf.keras.models.load_model(MODEL_CNN)
        scaler = joblib.load(SCALER)
        df = pd.read_csv(INPUT_FILE)
        print("Status: Models and Data loaded.")
    except Exception as e:
        print(f"Error: {e}. Make sure previous phases are completed.")
        return

    # 2. Prepare Data (English Schema)
    drop_cols = ['label', 'ip_src', 'ip_dst', 'timestamp', 'flow_id']
    X_raw = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    y_true = df['label']

    # Align columns
    if hasattr(scaler, 'feature_names_in_'):
        X_raw = X_raw[scaler.feature_names_in_]
    X_raw = X_raw.fillna(0)

    # 3. Predict - Random Forest
    print("Generating RF Predictions...")
    y_pred_rf = rf_model.predict(X_raw)
    cm_rf = confusion_matrix(y_true, y_pred_rf)

    # 4. Predict - CNN-LSTM
    print("Generating CNN Predictions...")
    X_scaled = scaler.transform(X_raw)
    X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
    y_prob_cnn = cnn_model.predict(X_reshaped, verbose=0)
    y_pred_cnn = (y_prob_cnn > 0.5).astype(int)
    cm_cnn = confusion_matrix(y_true, y_pred_cnn)

    # 5. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Define labels
    labels = ['Benign (0)', 'Adversarial PortScan (1)']

    # Plot RF
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Reds', ax=axes[0], 
                xticklabels=labels, yticklabels=labels, cbar=False)
    axes[0].set_title('Random Forest (Static) - Vulnerability', fontsize=14)
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')

    # Plot CNN
    sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues', ax=axes[1], 
                xticklabels=labels, yticklabels=labels, cbar=False)
    axes[1].set_title('CNN-LSTM (Frozen) - Vulnerability', fontsize=14)
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')

    plt.tight_layout()
    plt.savefig('confusion_matrix_comparison.png', dpi=300)
    print("\nSUCCESS: Confusion Matrix saved as 'confusion_matrix_comparison.png'")
    
    # Interpretation Helper
    print("\n--- INTERPRETATION FOR YOUR PAPER ---")
    tn_rf, fp_rf, fn_rf, tp_rf = cm_rf.ravel()
    print(f"Random Forest Missed Attacks (False Negatives): {fn_rf}")
    print(f"Random Forest Detection Rate (Recall): {tp_rf / (tp_rf + fn_rf):.4f}")

if __name__ == "__main__":
    plot_confusion_matrices()