import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import sys
import os

# Setup path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import prepare_data

def evaluate_model():
    print("="*60)
    print("STARTING MODEL EVALUATION")
    print("="*60)

    # 1. Load Model dan Scaler yang sudah disimpan
    print("\nLoading trained model and scalers...")
    model = load_model('models/lstm_model.h5')
    scaler_X = joblib.load('models/scaler_X.pkl')
    scaler_y = joblib.load('models/scaler_y.pkl')
    print("✓ Loaded successfully")

    # 2. Siapkan Data Test (Menggunakan pipeline yang sama)
    print("\nPreparing test data...")
    # Kita hanya butuh X_test dan y_test, serta scaler_y untuk inverse transform
    X_train, y_train, X_test, y_test, _, _ = prepare_data()
    print(f"✓ Test set shape: {X_test.shape}")

    # 3. Lakukan Prediksi
    print("\nMaking predictions on unseen test data...")
    y_pred_scaled = model.predict(X_test)
    
    # 4. Kembalikan ke Nilai Asli (Inverse Transform)
    # Saat ini prediksi masih dalam bentuk 0-1. Kita ubah jadi harga Dollar asli.
    y_test_real = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    y_pred_real = scaler_y.inverse_transform(y_pred_scaled)

    # 5. Hitung Metrik Evaluasi
    mae = mean_absolute_error(y_test_real, y_pred_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    
    # MAPE (Mean Absolute Percentage Error)
    # Tambahkan epsilon kecil untuk menghindari pembagian dengan nol
    mape = np.mean(np.abs((y_test_real - y_pred_real) / (y_test_real + 1e-8))) * 100

    print("\n" + "="*60)
    print("EVALUATION RESULTS (ON TEST DATA)")
    print("="*60)
    print(f"Mean Absolute Error (MAE)      : ${mae:,.2f}")
    print(f"Root Mean Square Error (RMSE)  : ${rmse:,.2f}")
    print(f"Mean Absolute Percentage Error : {mape:,.2f}%")
    print("="*60)

    # 6. Simpan Hasil Prediksi untuk Dashboard
    # Ini akan sangat berguna untuk Tab 1 di Streamlit nanti!
    results = pd.DataFrame({
        'Actual_Price': y_test_real.flatten(),
        'Predicted_Price': y_pred_real.flatten()
    })
    
    # Kita juga perlu tanggal untuk data test
    # (Ambil dari data asli, skip 30 hari pertama karena sequence, lalu ambil bagian test)
    from src.data_loader import load_gold_data, load_sentiment_data, merge_data
    from src.feature_eng import calculate_technical_indicators, create_target_variable, clean_feature_data
    
    df_gold = load_gold_data()
    df_sentiment = load_sentiment_data()
    df_merged = merge_data(df_gold, df_sentiment)
    df_features = calculate_technical_indicators(df_merged)
    df_target = create_target_variable(df_features)
    df_final = clean_feature_data(df_target)
    
    # Ambil tanggal sesuai dengan panjang data test
    test_dates = df_final['date'].iloc[-len(y_test_real):].values
    results['Date'] = test_dates
    
    results.to_csv('models/test_predictions.csv', index=False)
    print("✓ Predictions saved to models/test_predictions.csv for Dashboard")

if __name__ == "__main__":
    evaluate_model()