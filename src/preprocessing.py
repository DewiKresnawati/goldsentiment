import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SEQUENCE_LENGTH, TEST_SIZE
from src.feature_eng import calculate_technical_indicators, create_target_variable, clean_feature_data
from src.data_loader import load_gold_data, load_sentiment_data, merge_data

# Kolom yang akan digunakan sebagai fitur (input)
# Kita buang 'date' dan 'target_price' dari fitur input
FEATURES = ['price', 'open', 'high', 'low', 'change_%', 'vol', 
            'sentiment_score', 'avg_confidence', 'RSI', 'MACD', 
            'MACD_Signal', 'MACD_Diff', 'SMA_20']

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
    
    # 2. Pisahkan Fitur (X) dan Target (y)
    X = df_final[FEATURES].values
    y = df_final['target_price'].values
    
    print(f"\nTotal data: {len(X)} rows")
    print(f"Features shape: {X.shape}")
    
    # 3. Train/Test Split (Time Series Split - TIDAK BOLEH ACAK!)
    # Kita ambil 80% data pertama untuk training, 20% terakhir untuk testing
    split_idx = int(len(X) * (1 - TEST_SIZE))
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Train set: {len(X_train)} rows")
    print(f"Test set:  {len(X_test)} rows")
    
    # 4. Normalisasi (Scaling)
    # PENTING: Fit scaler HANYA pada data train untuk mencegah Data Leakage!
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test) # Hanya transform, jangan fit!
    
    # Target (y) juga perlu di-scale karena LSTM memprediksi nilai 0-1
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))
    
    print("\n✓ Data scaled successfully (0 to 1)")
    
    # 5. Buat Sequence (Sliding Window)
    # Mengubah data 2D menjadi 3D: [Samples, Time_Steps, Features]
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, SEQUENCE_LENGTH)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, SEQUENCE_LENGTH)
    
    print(f"\n✓ Sequences created with window size: {SEQUENCE_LENGTH} days")
    print(f"  X_train shape: {X_train_seq.shape} -> (Samples, Time_Steps, Features)")
    print(f"  X_test shape:  {X_test_seq.shape}")
    
    return X_train_seq, y_train_seq, X_test_seq, y_test_seq, scaler_X, scaler_y

def create_sequences(X, y, time_steps):
    """Membuat sliding window untuk LSTM"""
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps]) # Target adalah hari ke-(time_steps)
    return np.array(Xs), np.array(ys)

if __name__ == "__main__":
    # Jalankan preprocessing
    X_train, y_train, X_test, y_test, scaler_X, scaler_y = prepare_data()
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)
    print("Data siap untuk dimasukkan ke model LSTM.")