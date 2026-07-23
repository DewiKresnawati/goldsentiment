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
    print("="*70)
    print("STARTING MODEL EVALUATION (Matching Report & Journal Metrics)")
    print("="*70)

    # 1. Load Model dan Scaler yang sudah disimpan
    print("\n[1/6] Loading trained model and scalers...")
    model = load_model('models/lstm_model.h5')
    scaler_X = joblib.load('models/scaler_X.pkl')
    scaler_y = joblib.load('models/scaler_y.pkl')
    print("✓ Loaded successfully")

    # 2. Siapkan Data Test (Menggunakan pipeline yang sama)
    print("\n[2/6] Preparing test data...")
    X_train, y_train, X_test, y_test, _, _ = prepare_data()
    print(f"✓ Test set shape: {X_test.shape}")

    # 3. Lakukan Prediksi
    print("\n[3/6] Making predictions on unseen test data...")
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    # 4. Kembalikan ke Nilai Asli (Inverse Transform)
    print("\n[4/6] Inverse transforming predictions to USD scale...")
    y_test_real = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_real = scaler_y.inverse_transform(y_pred_scaled).flatten()

    # 5. Hitung Metrik Evaluasi (Sesuai Laporan & Jurnal)
    print("\n[5/6] Calculating evaluation metrics...")
    
    # MAE & RMSE
    mae = mean_absolute_error(y_test_real, y_pred_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_test_real - y_pred_real) / (y_test_real + 1e-8))) * 100
    
    # Directional Accuracy (DA) - BARU DITAMBAHKAN
    # Menghitung seberapa sering model benar menebak arah harga (Naik/Turun)
    actual_direction = np.sign(np.diff(y_test_real))
    pred_direction = np.sign(np.diff(y_pred_real))
    da = np.mean(actual_direction == pred_direction) * 100

    # 6. Tampilkan Hasil dalam Format Tabel yang Rapi
    print("\n" + "="*70)
    print("EVALUATION RESULTS (ON TEST DATA)")
    print("="*70)
    print(f"{'Metric':<35} | {'Value':<10} | {'Unit':<5} | {'Interpretation'}")
    print("-" * 70)
    print(f"{'Mean Absolute Error (MAE)':<35} | {mae:<10.2f} | {'USD':<5} | Avg absolute deviation")
    print(f"{'Root Mean Square Error (RMSE)':<35} | {rmse:<10.2f} | {'USD':<5} | Penalizes larger errors")
    print(f"{'Mean Absolute Percentage Error':<35} | {mape:<10.2f} | {'%':<5} | Relative error percentage")
    print(f"{'Directional Accuracy (DA)':<35} | {da:<10.2f} | {'%':<5} | Correctly predicted direction")
    print("="*70)

    # 7. Simpan Hasil Prediksi untuk Dashboard & Laporan
    print("\n[6/6] Saving predictions to CSV for Dashboard and Report...")
    results = pd.DataFrame({
        'Actual_Price': y_test_real,
        'Predicted_Price': y_pred_real
    })
    
    # Ambil tanggal untuk data test agar sesuai dengan laporan
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
    
    # Simpan ke CSV
    results.to_csv('models/test_predictions.csv', index=False)
    print("✓ Predictions saved to models/test_predictions.csv")
    print("="*70)
    print("EVALUATION COMPLETE! Metrics match the final report/journal.")
    print("="*70)

if __name__ == "__main__":
    evaluate_model()