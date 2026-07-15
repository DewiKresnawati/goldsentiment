import pandas as pd
import numpy as np
import ta # Library Technical Analysis
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_gold_data, load_sentiment_data, merge_data

def calculate_technical_indicators(df):
    """Menghitung indikator teknikal: RSI, MACD, SMA"""
    print("\nCalculating technical indicators...")
    
    # 1. RSI (Relative Strength Index) - window 14 hari
    df['RSI'] = ta.momentum.RSIIndicator(close=df['price'], window=14).rsi()
    
    # 2. MACD (Moving Average Convergence Divergence)
    macd = ta.trend.MACD(close=df['price'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()
    
    # 3. SMA (Simple Moving Average) - window 20 hari
    df['SMA_20'] = ta.trend.SMAIndicator(close=df['price'], window=20).sma_indicator()
    
    print("✓ Technical indicators calculated (RSI, MACD, SMA_20)")
    return df

def create_target_variable(df):
    """Membuat target prediksi (Harga Emas Hari Besok)"""
    print("\nCreating target variable for prediction...")
    
    # Target kita adalah harga 'price' di hari berikutnya (shift -1)
    df['target_price'] = df['price'].shift(-1)
    
    # Hapus baris terakhir karena tidak ada data hari esok untuk diprediksi
    df = df.iloc[:-1]
    
    print("✓ Target variable created")
    return df

def clean_feature_data(df):
    """Membersihkan NaN yang dihasilkan dari perhitungan indikator"""
    print("\nCleaning data...")
    
    # Indikator seperti RSI (butuh 14 hari) dan MACD (butuh 26 hari) 
    # akan menghasilkan NaN di baris-baris awal.
    initial_rows = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped_rows = initial_rows - len(df)
    
    print(f"✓ Dropped {dropped_rows} rows with NaN values.")
    print(f"  Final dataset size: {len(df)} rows")
    
    return df

if __name__ == "__main__":
    # 1. Load & Merge Data
    print("Loading and merging data...")
    df_gold = load_gold_data()
    df_sentiment = load_sentiment_data()
    df_merged = merge_data(df_gold, df_sentiment)
    
    # 2. Feature Engineering
    df_features = calculate_technical_indicators(df_merged)
    df_target = create_target_variable(df_features)
    df_final = clean_feature_data(df_target)
    
    # 3. Tampilkan Hasil
    print("\n" + "="*60)
    print("FINAL DATASET READY FOR LSTM")
    print("="*60)
    print(df_final.head(10))
    
    print("\nColumns available:")
    print(list(df_final.columns))