import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import random
import joblib
import os
import sys
import ta

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()

# Setup path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.preprocessing import prepare_data

print("="*70)
print("ABLATION STUDY: MULTIMODAL VS UNIMODAL MODELS")
print("="*70)

# ============================================
# MODEL 1: MULTIMODAL LSTM (LOAD MODEL TERLATIH - SAMA PERSIS DENGAN EVALUATION)
# ============================================
print("\n" + "="*70)
print("MODEL 1: MULTIMODAL LSTM (Proposed - loaded from models/lstm_model.h5)")
print("="*70)

model1 = load_model('models/lstm_model.h5')
scaler_y1 = joblib.load('models/scaler_y.pkl')

# Pakai pipeline yang SAMA PERSIS dengan evaluation.py
X_train_seq_1, y_train_seq_1, X_test_seq_1, y_test_seq_1, _, _, df_final = prepare_data()

pred1_scaled = model1.predict(X_test_seq_1, verbose=0)
pred1 = scaler_y1.inverse_transform(pred1_scaled).flatten()
y_test_1_true = scaler_y1.inverse_transform(y_test_seq_1.reshape(-1, 1)).flatten()

mae1 = mean_absolute_error(y_test_1_true, pred1)
rmse1 = np.sqrt(mean_squared_error(y_test_1_true, pred1))
mape1 = np.mean(np.abs((y_test_1_true - pred1) / (y_test_1_true + 1e-8))) * 100

print(f"\n✓ MODEL 1 RESULTS (IDENTIK DENGAN evaluation.py & Tabel 4.8):")
print(f"  MAE:  ${mae1:.2f}")
print(f"  RMSE: ${rmse1:.2f}")
print(f"  MAPE: {mape1:.2f}%")

# ============================================
# LOAD DATA UNTUK MODEL BASELINE (ABLATION)
# ============================================
print("\n[2/6] Loading data for baseline models...")

df_gold = pd.read_csv('data/Gold Futures Historical Data.csv')
df_gold.columns = [col.strip().lower().replace(' ', '_') for col in df_gold.columns]
df_gold['date'] = pd.to_datetime(df_gold['date'])
df_gold = df_gold.sort_values('date').reset_index(drop=True)

numeric_cols = ['price', 'open', 'high', 'low']
for col in numeric_cols:
    if col in df_gold.columns:
        df_gold[col] = df_gold[col].astype(str).str.replace('"', '').str.replace(',', '')
        df_gold[col] = pd.to_numeric(df_gold[col], errors='coerce')

df_sentiment = pd.read_excel('data/dataset_indonesia_sentimen.xlsx')
df_sentiment.columns = [col.strip().lower().replace(' ', '_') for col in df_sentiment.columns]
df_sentiment['date_only'] = pd.to_datetime(df_sentiment['date']).dt.date
df_gold['date_only'] = df_gold['date'].dt.date

sentiment_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
if 'sentiment' in df_sentiment.columns:
    df_sentiment['sentiment_numeric'] = df_sentiment['sentiment'].astype(str).str.lower().map(sentiment_mapping).fillna(0)
else:
    df_sentiment['sentiment_numeric'] = 0

daily_sentiment = df_sentiment.groupby('date_only').agg({
    'sentiment_numeric': 'mean',
    'confidence': 'mean'
}).reset_index()
daily_sentiment.columns = ['date_only', 'sentiment_score', 'avg_confidence']

df_merged = df_gold.merge(daily_sentiment, on='date_only', how='left')
df_merged['sentiment_score'] = df_merged['sentiment_score'].fillna(0)
df_merged['avg_confidence'] = df_merged['avg_confidence'].fillna(0.5)

df_merged['price'] = pd.to_numeric(df_merged['price'], errors='coerce')
df_merged['RSI'] = ta.momentum.RSIIndicator(close=df_merged['price'], window=14).rsi()
df_merged['MACD'] = ta.trend.MACD(close=df_merged['price']).macd()
df_merged['SMA_20'] = ta.trend.SMAIndicator(close=df_merged['price'], window=20).sma_indicator()
df_merged['SMA_50'] = ta.trend.SMAIndicator(close=df_merged['price'], window=50).sma_indicator()

df_merged['target'] = df_merged['price'].shift(-1)
df_merged = df_merged.dropna().reset_index(drop=True)

SEQUENCE_LENGTH = 60
TEST_SIZE = 0.2

def create_sequences(data, target, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(target[i + seq_length])
    return np.array(X), np.array(y)

split_idx = int(len(df_merged) * (1 - TEST_SIZE))
train_df = df_merged.iloc[:split_idx].copy()
test_df = df_merged.iloc[split_idx:].copy()

print(f"  Training samples: {len(train_df)}")
print(f"  Testing samples: {len(test_df)}")

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)

# ============================================
# MODEL 2: NO-SENTIMENT (Price + Technical)
# ============================================
print("\n" + "="*70)
print("MODEL 2: NO-SENTIMENT (Price + Technical Indicators)")
print("="*70)

feature_cols_2 = ['price', 'RSI', 'MACD', 'SMA_20', 'SMA_50']
X_2 = train_df[feature_cols_2].values
y_2 = train_df['target'].values.reshape(-1, 1)
X_test_2 = test_df[feature_cols_2].values
y_test_2 = test_df['target'].values.reshape(-1, 1)

scaler_X2 = MinMaxScaler(feature_range=(0, 1))
scaler_y2 = MinMaxScaler(feature_range=(0, 1))

X_2_scaled = scaler_X2.fit_transform(X_2)
y_2_scaled = scaler_y2.fit_transform(y_2)
X_test_2_scaled = scaler_X2.transform(X_test_2)
y_test_2_scaled = scaler_y2.transform(y_test_2)

X_train_seq_2, y_train_seq_2 = create_sequences(X_2_scaled, y_2_scaled, SEQUENCE_LENGTH)
X_test_seq_2, y_test_seq_2 = create_sequences(X_test_2_scaled, y_test_2_scaled, SEQUENCE_LENGTH)

model2 = Sequential([
    Input(shape=(SEQUENCE_LENGTH, len(feature_cols_2))),
    LSTM(128, return_sequences=True, activation='tanh'),
    Dropout(0.2),
    LSTM(64, activation='tanh'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])
model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
model2.fit(X_train_seq_2, y_train_seq_2, epochs=100, batch_size=32, validation_split=0.1, callbacks=[early_stop], verbose=0)

pred2_scaled = model2.predict(X_test_seq_2, verbose=0)
pred2 = scaler_y2.inverse_transform(pred2_scaled)
y_test_2_true = scaler_y2.inverse_transform(y_test_seq_2)

mae2 = mean_absolute_error(y_test_2_true, pred2)
rmse2 = np.sqrt(mean_squared_error(y_test_2_true, pred2))
mape2 = np.mean(np.abs((y_test_2_true - pred2) / (y_test_2_true + 1e-8))) * 100

print(f"\n✓ MODEL 2 RESULTS:")
print(f"  MAE:  ${mae2:.2f}")
print(f"  RMSE: ${rmse2:.2f}")
print(f"  MAPE: {mape2:.2f}%")

# ============================================
# MODEL 3: NO-TECHNICAL (Price + Sentiment)
# ============================================
print("\n" + "="*70)
print("MODEL 3: NO-TECHNICAL (Price + Sentiment)")
print("="*70)

feature_cols_3 = ['price', 'sentiment_score', 'avg_confidence']
X_3 = train_df[feature_cols_3].values
y_3 = train_df['target'].values.reshape(-1, 1)
X_test_3 = test_df[feature_cols_3].values
y_test_3 = test_df['target'].values.reshape(-1, 1)

scaler_X3 = MinMaxScaler(feature_range=(0, 1))
scaler_y3 = MinMaxScaler(feature_range=(0, 1))

X_3_scaled = scaler_X3.fit_transform(X_3)
y_3_scaled = scaler_y3.fit_transform(y_3)
X_test_3_scaled = scaler_X3.transform(X_test_3)
y_test_3_scaled = scaler_y3.transform(y_test_3)

X_train_seq_3, y_train_seq_3 = create_sequences(X_3_scaled, y_3_scaled, SEQUENCE_LENGTH)
X_test_seq_3, y_test_seq_3 = create_sequences(X_test_3_scaled, y_test_3_scaled, SEQUENCE_LENGTH)

model3 = Sequential([
    Input(shape=(SEQUENCE_LENGTH, len(feature_cols_3))),
    LSTM(128, return_sequences=True, activation='tanh'),
    Dropout(0.2),
    LSTM(64, activation='tanh'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])
model3.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
model3.fit(X_train_seq_3, y_train_seq_3, epochs=100, batch_size=32, validation_split=0.1, callbacks=[early_stop], verbose=0)

pred3_scaled = model3.predict(X_test_seq_3, verbose=0)
pred3 = scaler_y3.inverse_transform(pred3_scaled)
y_test_3_true = scaler_y3.inverse_transform(y_test_seq_3)

mae3 = mean_absolute_error(y_test_3_true, pred3)
rmse3 = np.sqrt(mean_squared_error(y_test_3_true, pred3))
mape3 = np.mean(np.abs((y_test_3_true - pred3) / (y_test_3_true + 1e-8))) * 100

print(f"\n✓ MODEL 3 RESULTS:")
print(f"  MAE:  ${mae3:.2f}")
print(f"  RMSE: ${rmse3:.2f}")
print(f"  MAPE: {mape3:.2f}%")

# ============================================
# SUMMARY TABLE
# ============================================
print("\n" + "="*70)
print("ABLATION STUDY SUMMARY")
print("="*70)
print(f"{'Model':<45} {'MAE (USD)':<15} {'RMSE (USD)':<15} {'MAPE (%)':<10}")
print("-"*85)
print(f"{'1. Multimodal LSTM (Proposed)':<45} {mae1:<15.2f} {rmse1:<15.2f} {mape1:<10.2f}")
print(f"{'2. No-Sentiment (Technical Only)':<45} {mae2:<15.2f} {rmse2:<15.2f} {mape2:<10.2f}")
print(f"{'3. No-Technical (Sentiment Only)':<45} {mae3:<15.2f} {rmse3:<15.2f} {mape3:<10.2f}")
print("="*85)

print("\n✓ Ablation study complete!")
print("  Model 1 results are IDENTICAL to evaluation.py (Table 4.8)!")