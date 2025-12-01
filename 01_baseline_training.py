# 01_baseline_training.py
# Description: Trains baseline Static (Random Forest) and Dynamic (CNN-LSTM) models.
# Output: Saves trained models (.joblib, .keras) and the scaler object.

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Input, Dropout

# Configuration
DATASET_FILE = 'master_dataset.csv'
SEED = 42

# Reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)

def train_models():
    print("--- [Phase 1] Baseline Model Training ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv(DATASET_FILE)
        print(f"Data Loaded: {df.shape}")
    except FileNotFoundError:
        print(f"Error: '{DATASET_FILE}' not found. Please upload the dataset.")
        return

    # 2. Preprocessing
    # UPDATED DROP LIST to match '00_data_harmonization.py' English Schema
    drop_cols = ['label', 'ip_src', 'ip_dst', 'timestamp', 'flow_id'] 
    
    # Remove columns that exist
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    y = df['label']
    
    print(f"Features selected for training: {list(X.columns)}")
    
    # Scaling
    print("Status: Scaling features...")
    scaler = MinMaxScaler()
    try:
        X_scaled = scaler.fit_transform(X)
    except ValueError as e:
        print(f"\nCRITICAL ERROR during scaling: {e}")
        print("Check if there are still non-numeric columns in X:")
        print(X.dtypes)
        return

    # Save feature names for inference validity
    scaler.feature_names_in_ = list(X.columns) 
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=SEED)

    # 3. Train Static Model (Random Forest)
    print("Status: Training Random Forest (Static Paradigm)...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    # 4. Train Dynamic Model (CNN-LSTM)
    print("Status: Training CNN-LSTM (Dynamic Paradigm)...")
    # Reshape for 3D input: [samples, time_steps, features]
    X_train_cnn = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    
    cnn_model = Sequential([
        Input(shape=(1, X_train.shape[1])),
        Conv1D(filters=64, kernel_size=1, activation='relu'),
        LSTM(64, return_sequences=False),
        Dropout(0.2), # Added Dropout for regularization
        Dense(1, activation='sigmoid')
    ])
    
    cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    cnn_model.fit(X_train_cnn, y_train, epochs=5, batch_size=64, verbose=1)

    # 5. Save Artifacts
    print("Status: Saving models and artifacts...")
    joblib.dump(rf_model, 'rf_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    cnn_model.save('cnn_lstm_model.keras')
    print("Success: Phase 1 complete.")

if __name__ == "__main__":
    train_models()