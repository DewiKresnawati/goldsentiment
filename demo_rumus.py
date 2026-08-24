import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Setup path agar bisa import modul src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.data_loader import load_gold_data, load_sentiment_data, merge_data
from src.feature_eng import calculate_technical_indicators, clean_feature_data
from config import FEATURES, SEQUENCE_LENGTH

print("="*70)
print("DEMO VISUALISASI: RUMUS 1 (SCALER) & RUMUS 2 (EARLY FUSION)")
print("="*70)

# 1. Load & Prepare Data (Sama seperti di preprocessing.py)
df_gold = load_gold_data()
df_sentiment = load_sentiment_data()
df_merged = merge_data(df_gold, df_sentiment)
df_features = calculate_technical_indicators(df_merged)
df_final = clean_feature_data(df_features)

# ==========================================
# 🔵 DEMO RUMUS 2: EARLY FUSION (Penggabungan Fitur)
# ==========================================
print("\n🔵 [RUMUS 2: EARLY FUSION] Penggabungan 7 Fitur per Hari")
print("Di kode, ini terjadi saat kita memanggil df_final[FEATURES].")
print("5 Fitur Numerik + 2 Sentimen disatukan (dikonkatenasi) menjadi 1 matriks.")

df_fused = df_final[FEATURES]
print("\nContoh 3 Hari Pertama (Vektor Gabungan / Early Fusion):")
print(df_fused.head(3).round(4))
print(f"\nBentuk Data Early Fusion 2D: {df_fused.shape} -> (Jumlah Hari, 7 Fitur Gabungan)")

# ==========================================
# 🟢 DEMO RUMUS 1: MINMAX SCALER (Normalisasi)
# ==========================================
print("\n\n🟢 [RUMUS 1: MINMAX SCALER] Normalisasi Per Kolom")
print("Rumus: (X - X_min) / (X_max - X_min)")

# Ambil 3 hari pertama untuk contoh
sample_data = df_fused.head(3).values

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(df_fused.values) # Scaler menghafal min & max TIAP KOLOM
sample_scaled = scaler.transform(sample_data)

print("\n[SEBELUM SCALING] (Skala Asli / Mentah):")
print(pd.DataFrame(sample_data, columns=FEATURES).round(2))

print("\n[SESUDAH SCALING] (Skala 0 sampai 1):")
print(pd.DataFrame(sample_scaled, columns=FEATURES).round(4))

print("\n💡 Contoh Pembuktian Manual untuk kolom 'price' di hari pertama:")
print(f"   Harga Asli : {sample_data[0][0]:.2f}")
print(f"   Min Price  : {scaler.data_min_[0]:.2f}")
print(f"   Max Price  : {scaler.data_max_[0]:.2f}")
hitung_manual = (sample_data[0][0] - scaler.data_min_[0]) / (scaler.data_max_[0] - scaler.data_min_[0])
print(f"   Perhitungan: ({sample_data[0][0]:.2f} - {scaler.data_min_[0]:.2f}) / ({scaler.data_max_[0]:.2f} - {scaler.data_min_[0]:.2f}) = {hitung_manual:.4f}")
print(f"   Hasil Kode : {sample_scaled[0][0]:.4f} (Cocok!)")

# ==========================================
# 🟣 DEMO SLIDING WINDOW (Input ke LSTM)
# ==========================================
print("\n\n🟣 [INPUT LSTM] Sliding Window (Look-back 60 Hari)")
X_scaled = scaler.transform(df_fused.values)

Xs = []
for i in range(len(X_scaled) - SEQUENCE_LENGTH):
    Xs.append(X_scaled[i:(i + SEQUENCE_LENGTH)])
X_seq = np.array(Xs)

print(f"\nBentuk akhir input LSTM (X): {X_seq.shape}")
print("Artinya: (Jumlah Sampel, 60 Hari Historis, 7 Fitur Gabungan)")
print("Inilah bentuk 3D Array yang diminta oleh lapisan LSTM di Keras!")
print("="*70)