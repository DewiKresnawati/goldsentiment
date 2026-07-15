import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import sys
import os

# Setup path untuk import config dan preprocessing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EPOCHS, BATCH_SIZE
from src.preprocessing import prepare_data

def build_lstm_model(input_shape):
    """Membangun arsitektur model LSTM"""
    print("\nBuilding LSTM Architecture...")
    model = Sequential()
    
    # Layer LSTM 1 (Menangkap pola kompleks)
    model.add(LSTM(units=64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2)) # Mencegah overfitting (menghafal data)
    
    # Layer LSTM 2 (Menyempurnakan pola)
    model.add(LSTM(units=32))
    model.add(Dropout(0.2))
    
    # Layer Output (Memprediksi 1 nilai: harga emas)
    model.add(Dense(units=1))
    
    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print("✓ LSTM Model built successfully")
    return model

def train_and_save_model():
    print("="*60)
    print("STARTING LSTM TRAINING PROCESS")
    print("="*60)
    
    # 1. Siapkan data (Load, Feature Eng, Preprocessing)
    X_train, y_train, X_test, y_test, scaler_X, scaler_y = prepare_data()
    
    # 2. Bangun model
    input_shape = (X_train.shape[1], X_train.shape[2]) # (30, 13)
    model = build_lstm_model(input_shape)
    
    # Tampilkan ringkasan model
    model.summary()
    
    # 3. Setup Early Stopping 
    # (Otomatis berhenti jika error tidak turun selama 10 epoch, menghemat waktu)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # 4. Training Model
    print(f"\nStarting training for {EPOCHS} epochs...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1, # 10% dari data train untuk validasi
        callbacks=[early_stop],
        verbose=1
    )
    
    # 5. Simpan Model dan Scaler
    print("\nSaving model and scalers...")
    os.makedirs('models', exist_ok=True)
    
    # Simpan model LSTM
    model.save('models/lstm_model.h5')
    
    # PENTING: Simpan Scaler! 
    # Dashboard butuh ini untuk mengubah prediksi 0-1 kembali ke harga Dollar asli.
    joblib.dump(scaler_X, 'models/scaler_X.pkl')
    joblib.dump(scaler_y, 'models/scaler_y.pkl')
    
    print("✓ Model saved to models/lstm_model.h5")
    print("✓ Scalers saved to models/scaler_X.pkl and scaler_y.pkl")
    print("="*60)
    print("TRAINING COMPLETE!")

if __name__ == "__main__":
    train_and_save_model()