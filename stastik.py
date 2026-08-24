import sys
import os
import pandas as pd

# Tambahkan root folder ke path agar bisa import modul dari folder 'src'
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from src.data_loader import load_gold_data, load_sentiment_data, merge_data
from src.feature_eng import calculate_technical_indicators, create_target_variable, clean_feature_data

print("="*60)
print("MEMULAI PROSES PEMUATAN DATA & FEATURE ENGINEERING")
print("="*60)

# 1. Load dan Merge Data
df_gold = load_gold_data()
df_sentiment = load_sentiment_data()
df_merged = merge_data(df_gold, df_sentiment)

# 2. Feature Engineering (Menambahkan RSI, MACD, SMA, dll)
df_features = calculate_technical_indicators(df_merged)
df_final = clean_feature_data(df_features) # Ini adalah dataframe final yang sudah bersih

print("\n" + "="*60)
print("MENAMPILKAN STATISTIK DESKRIPTIF FITUR")
print("="*60)

# 3. Kode yang ingin kamu jalankan
features_to_check = ['price', 'RSI', 'MACD', 'SMA_20', 'SMA_50', 'sentiment_score']

# Pastikan semua kolom ada sebelum di-describe
available_features = [col for col in features_to_check if col in df_final.columns]

if len(available_features) == len(features_to_check):
    # Tampilkan statistik deskriptif
    stats = df_final[available_features].describe()
    print(stats)
    
    # Opsional: Simpan ke CSV agar mudah kamu copy ke Word/Laporan
    stats.to_csv('statistik_fitur_laporan.csv')
    print("\n✓ Statistik juga telah disimpan ke file 'statistik_fitur_laporan.csv'")
else:
    print(f"\n⚠️ Peringatan: Beberapa fitur tidak ditemukan di dataframe.")
    print(f"Kolom yang tersedia: {df_final.columns.tolist()}")