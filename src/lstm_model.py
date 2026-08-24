import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import random
import joblib
import sys
import os

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EPOCHS, BATCH_SIZE, SEQUENCE_LENGTH
from src.preprocessing import prepare_data

def build_lstm_model(input_shape):
    print("\nBuilding LSTM Architecture (7 Features)...")
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=input_shape, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(LSTM(units=64, return_sequences=False, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=1, activation='linear'))
    
    # Kita import Adam secara eksplisit dan turunkan learning rate jadi 0.0005
    from tensorflow.keras.optimizers import Adam
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error', metrics=['mae', 'mape'])
    print("✓ LSTM Model built successfully")
    return model

def train_and_save_model():
    print("="*60)
    print("STARTING LSTM TRAINING PROCESS")
    print("="*60)
    
    X_train, y_train, X_test, y_test, scaler_X, scaler_y, df_final = prepare_data()
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"Input shape: {input_shape}")
    model = build_lstm_model(input_shape)
    model.summary()
    
    # Naikkan patience dari 10 jadi 20, biar model punya waktu lebih buat cari loss terendah
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
    checkpoint = ModelCheckpoint('models/best_model_early_fusion.h5', monitor='val_loss', save_best_only=True, verbose=1)
    
    print(f"\nStarting training for {EPOCHS} epochs...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )
    
    print("\nSaving model and scalers...")
    os.makedirs('models', exist_ok=True)
    
    model.save('models/lstm_model.h5')
    joblib.dump(scaler_X, 'models/scaler_X.pkl')
    joblib.dump(scaler_y, 'models/scaler_y.pkl')
    
    print("✓ Model saved to models/lstm_model.h5")
    print("✓ Scalers saved")
    print("="*60)

if __name__ == "__main__":
    train_and_save_model()
    
    # Plot learning curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss (MSE) per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('MAE Value')
    plt.legend()

    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=300)
    plt.show()