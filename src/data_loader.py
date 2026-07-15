import pandas as pd
import numpy as np
import sys
import os

# Tambahkan parent directory ke path agar bisa import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GOLD_DATA_FILE, SENTIMENT_DATA_FILE

def convert_volume_to_numeric(vol_str):
    """Mengubah teks volume (misal: 273.37K) menjadi angka (273370)"""
    if pd.isna(vol_str):
        return np.nan
    
    vol_str = str(vol_str).replace(',', '').strip()
    
    if vol_str.endswith('K'):
        return float(vol_str[:-1]) * 1000
    elif vol_str.endswith('M'):
        return float(vol_str[:-1]) * 1000000
    elif vol_str.endswith('B'):
        return float(vol_str[:-1]) * 1000000000
    else:
        try:
            return float(vol_str)
        except ValueError:
            return np.nan

def load_gold_data():
    """Load data harga emas dari CSV"""
    print(f"Loading gold data from {GOLD_DATA_FILE}...")
    
    df_gold = pd.read_csv(GOLD_DATA_FILE)
    
    # Normalisasi nama kolom
    df_gold.columns = [col.strip().lower().replace(' ', '_') for col in df_gold.columns]
    
    # Konversi Date
    df_gold['date'] = pd.to_datetime(df_gold['date'])
    
    # Bersihkan data numerik (hapus tanda kutip, koma, dan persen)
    numeric_cols = ['price', 'open', 'high', 'low']
    for col in numeric_cols:
        if col in df_gold.columns:
            df_gold[col] = df_gold[col].astype(str).str.replace('"', '').str.replace(',', '')
            df_gold[col] = pd.to_numeric(df_gold[col], errors='coerce')
            
    # Khusus Volume (karena ada huruf K, M)
    if 'vol.' in df_gold.columns:
        df_gold['vol'] = df_gold['vol.'].apply(convert_volume_to_numeric)
        df_gold = df_gold.drop(columns=['vol.'])
        
    # Khusus Change % (hapus tanda %)
    if 'change_%' in df_gold.columns:
        df_gold['change_%'] = df_gold['change_%'].astype(str).str.replace('"', '').str.replace('%', '').str.replace(',', '')
        df_gold['change_%'] = pd.to_numeric(df_gold['change_%'], errors='coerce')
    
    df_gold = df_gold.sort_values('date').reset_index(drop=True)
    
    print(f"✓ Loaded {len(df_gold)} rows of gold data")
    print(f"  Date range: {df_gold['date'].min()} to {df_gold['date'].max()}")
    
    return df_gold

def load_sentiment_data():
    """Load data sentimen dari Excel"""
    print(f"Loading sentiment data from {SENTIMENT_DATA_FILE}...")
    
    df_sentiment = pd.read_excel(SENTIMENT_DATA_FILE)
    df_sentiment.columns = [col.strip().lower().replace(' ', '_') for col in df_sentiment.columns]
    
    df_sentiment['date'] = pd.to_datetime(df_sentiment['date'])
    
    # PENTING: Hilangkan jam/menik/detik agar bisa di-merge dengan data emas
    df_sentiment['date'] = df_sentiment['date'].dt.normalize()
    
    df_sentiment = df_sentiment.sort_values('date').reset_index(drop=True)
    
    print(f"✓ Loaded {len(df_sentiment)} rows of sentiment data")
    print(f"  Date range: {df_sentiment['date'].min()} to {df_sentiment['date'].max()}")
    
    return df_sentiment

def merge_data(df_gold, df_sentiment):
    """Gabungkan data emas dan sentimen berdasarkan tanggal"""
    print("\nMerging gold and sentiment data...")
    
    # Konversi sentimen text ke numeric score
    sentiment_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
    df_sentiment['sentiment_numeric'] = df_sentiment['sentiment'].map(sentiment_mapping)
    
    # Agregasi sentimen per hari (Rata-rata)
    daily_sentiment = df_sentiment.groupby('date').agg({
        'sentiment_numeric': 'mean',
        'confidence': 'mean'
    }).reset_index()
    
    daily_sentiment.columns = ['date', 'sentiment_score', 'avg_confidence']
    
    # Merge data
    df_merged = df_gold.merge(daily_sentiment, on='date', how='left')
    
    # Fill NaN (Hari tanpa berita dianggap netral 0)
    df_merged['sentiment_score'] = df_merged['sentiment_score'].fillna(0)
    df_merged['avg_confidence'] = df_merged['avg_confidence'].fillna(0.5)
    
    print(f"✓ Merged data: {len(df_merged)} rows")
    
    return df_merged

if __name__ == "__main__":
    df_gold = load_gold_data()
    df_sentiment = load_sentiment_data()
    df_merged = merge_data(df_gold, df_sentiment)
    
    print("\n" + "="*50)
    print("Sample of merged data:")
    print(df_merged.head(10))
    print("\nData info:")
    print(df_merged.info())