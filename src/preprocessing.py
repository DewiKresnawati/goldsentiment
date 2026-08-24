import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SEQUENCE_LENGTH, TEST_SIZE, FEATURES
from src.feature_eng import calculate_technical_indicators, create_target_variable, clean_feature_data
from src.data_loader import load_gold_data, load_sentiment_data, merge_data

def prepare_data():
    """Memuat data dan melakukan preprocessing lengkap"""
    print("="*60)
    print("STARTING PREPROCESSING PIPELINE")
    print("="*60)
    
    # 1. Load & Feature Engineering
    df_gold = load_gold_data()
    df_sentiment = load_sentiment_data()
    df_merged = merge_data(df_gold, df_sentiment)
    df_features = calculate_technical_indicators(df_merged)
    df_target = create_target_variable(df_features)
    df_final = clean_feature_data(df_target)
    
    # 2. Pastikan HANYA 7 fitur yang diambil
    available_cols = [col for col in FEATURES if col in df_final.columns]
    print(f"Fitur yang digunakan ({len(available_cols)}): {available_cols}")
    
    X = df_final[available_cols].values # Rumus Early Fusion (Konkatenasi / Penggabungan)
    y = df_final['target_price'].values
    
    print(f"\nTotal data valid: {len(X)} rows")
    
    # 3. Train/Test Split (Chronological)
    split_idx = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Train set: {len(X_train)} rows")
    print(f"Test set:  {len(X_test)} rows")
    
        # 4. Normalisasi (Scaling)
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    scaler_X.fit(X) # <-- Mencari X_min dan X_max PER KOLOM
    scaler_y.fit(y.reshape(-1, 1))
    
    X_train_scaled = scaler_X.transform(X_train) # <-- Menerapkan rumus
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.transform(y_train.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))
    
    # 5. Buat Sequence
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, SEQUENCE_LENGTH)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, SEQUENCE_LENGTH)
    
    print(f"\n✓ Sequences created with window size: {SEQUENCE_LENGTH} days")
    print(f"  X_train shape: {X_train_seq.shape}")
    print(f"  X_test shape:  {X_test_seq.shape}")
    
    return X_train_seq, y_train_seq, X_test_seq, y_test_seq, scaler_X, scaler_y, df_final

def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)]) # Mengambil 60 baris berurutan
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)