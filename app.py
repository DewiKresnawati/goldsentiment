import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import os
import sys

# Setup path agar bisa import modul src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import SEQUENCE_LENGTH, FEATURES
from src.data_loader import load_gold_data, load_sentiment_data, merge_data
from src.feature_eng import calculate_technical_indicators, create_target_variable, clean_feature_data

# ==========================================
# 1. CACHING FUNCTIONS (Agar Dashboard Cepat)
# ==========================================
@st.cache_resource
def load_trained_model():
    """Memuat model LSTM hanya sekali saat aplikasi dimulai"""
    return load_model('models/lstm_model.h5', compile=False)

@st.cache_data
def load_and_process_data():
    """Memuat dan memproses data hanya sekali, kecuali data berubah"""
    df_gold = load_gold_data()
    df_sentiment = load_sentiment_data()
    df_merged = merge_data(df_gold, df_sentiment)
    df_features = calculate_technical_indicators(df_merged)
    df_final = create_target_variable(df_features)
    df_final = clean_feature_data(df_final)
    return df_final

@st.cache_data
def load_predictions():
    """Memuat hasil prediksi dari file CSV"""
    if os.path.exists('models/test_predictions.csv'):
        return pd.read_csv('models/test_predictions.csv', parse_dates=['Date'])
    return pd.DataFrame()

# ==========================================
# 2. KONFIGURASI HALAMAN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Gold Price Prediction Dashboard", 
    page_icon="📈", 
    layout="wide" # Menggunakan layout lebar agar grafik lebih lega
)

# ==========================================
# 3. SIDEBAR & HEADER
# ==========================================
st.sidebar.title("⚙️ Pengaturan Dashboard")
st.sidebar.markdown("---")
st.sidebar.info("""
**Tentang Proyek Ini:**
Dashboard ini memprediksi harga emas (Gold Futures) menggunakan model Deep Learning LSTM Multimodal yang menggabungkan:
1. Data Historis Harga
2. Indikator Teknikal (RSI, MACD, SMA)
3. Analisis Sentimen Berita Keuangan Indonesia
""")

st.title("📈 Dashboard Prediksi Harga Emas (Gold Futures)")
st.markdown("Analisis historis, sentimen pasar, dan prediksi harga berbasis *Deep Learning* (LSTM).")

# Muat semua data dan model di awal
with st.spinner("Memuat data dan model AI... (Ini mungkin memakan waktu beberapa detik)"):
    df_full = load_and_process_data()
    model = load_trained_model()
    df_pred = load_predictions()

st.markdown("---")

# ==========================================
# 4. TAB NAVIGASI
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Prediksi vs Aktual", 
    "📊 Analisis Historis Harga", 
    "📰 Analisis Sentimen Berita", 
    "📋 Detail Data"
])

# ==========================================
# TAB 1: Prediksi vs Aktual
# ==========================================
with tab1:
    st.header("Prediksi vs Harga Aktual (Data Testing)")
    st.markdown("Perbandingan antara harga emas sebenarnya dengan prediksi model LSTM pada data *testing* (20% data terakhir).")
    
    if not df_pred.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_pred['Date'], y=df_pred['Actual_Price'], 
            mode='lines', name='Harga Aktual', 
            line=dict(color='royalblue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=df_pred['Date'], y=df_pred['Predicted_Price'], 
            mode='lines', name='Prediksi LSTM', 
            line=dict(color='crimson', width=2, dash='dot')
        ))
        
        fig.update_layout(
            title="Perbandingan Harga Emas Aktual vs Prediksi",
            xaxis_title="Tanggal",
            yaxis_title="Harga Emas (USD)",
            hovermode="x unified",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrik Evaluasi
        st.markdown("### 📏 Metrik Akurasi Model")
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Absolute Error (MAE)", "$612.49", delta="-", delta_color="inverse")
        col2.metric("Root Mean Square Error (RMSE)", "$747.34", delta="-", delta_color="inverse")
        col3.metric("Mean Absolute Percentage Error (MAPE)", "13.33%", delta="-", delta_color="inverse")
    else:
        st.warning("⚠️ Data prediksi belum tersedia. Silakan jalankan `python src/evaluation.py` terlebih dahulu.")

# ==========================================
# TAB 2: Analisis Historis Harga
# ==========================================
with tab2:
    st.header("Analisis Historis Harga & Indikator Teknikal")
    
    # Filter Tanggal
    min_date = df_full['date'].min().date()
    max_date = df_full['date'].max().date()
    start_date, end_date = st.date_input(
        "Pilih Rentang Tanggal", 
        [min_date, max_date], 
        min_value=min_date, 
        max_value=max_date
    )
    
    df_filtered = df_full[(df_full['date'].dt.date >= start_date) & (df_full['date'].dt.date <= end_date)]
    
    # Grafik Harga + SMA
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['price'], mode='lines', name='Harga Close', line=dict(color='gold', width=2)))
    fig_hist.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['SMA_20'], mode='lines', name='SMA 20', line=dict(color='blue', width=1, dash='dash')))
    
    fig_hist.update_layout(
        title="Tren Harga Emas dan Simple Moving Average (SMA 20)",
        xaxis_title="Tanggal", yaxis_title="Harga (USD)",
        hovermode="x unified", template="plotly_white"
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Grafik RSI
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['RSI'], mode='lines', name='RSI (14)', line=dict(color='purple', width=1)))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (Jenuh Beli)")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (Jenuh Jual)")
    
    fig_rsi.update_layout(
        title="Relative Strength Index (RSI)",
        xaxis_title="Tanggal", yaxis_title="RSI", yaxis_range=[0, 100],
        hovermode="x unified", template="plotly_white"
    )
    st.plotly_chart(fig_rsi, use_container_width=True)

# ==========================================
# TAB 3: Analisis Sentimen Berita
# ==========================================
with tab3:
    st.header("Analisis Sentimen Berita")
    st.markdown("Skor sentimen berkisar antara **-1 (Sangat Negatif)** hingga **1 (Sangat Positif)**. Skor 0 berarti Netral.")
    
    # Agregasi sentimen per bulan agar grafik lebih rapi
    df_sentiment_trend = df_full.copy()
    df_sentiment_trend['Month'] = df_sentiment_trend['date'].dt.to_period('M').astype(str)
    monthly_sentiment = df_sentiment_trend.groupby('Month')['sentiment_score'].mean().reset_index()
    monthly_sentiment['Month'] = pd.to_datetime(monthly_sentiment['Month'])
    
    fig_sent = px.bar(
        monthly_sentiment, x='Month', y='sentiment_score',
        title="Rata-rata Skor Sentimen Berita per Bulan",
        color='sentiment_score', color_continuous_scale='RdYlGn',
        labels={'sentiment_score': 'Skor Sentimen', 'Month': 'Bulan'}
    )
    fig_sent.update_layout(template="plotly_white")
    st.plotly_chart(fig_sent, use_container_width=True)
    
    # Scatter Plot Korelasi
    st.subheader("Korelasi Sentimen vs Perubahan Harga Harian")
    fig_corr = px.scatter(
        df_full, x='sentiment_score', y='change_%',
        title="Scatter Plot: Skor Sentimen vs Perubahan Harga (%)",
        opacity=0.6, trendline="ols",
        labels={'sentiment_score': 'Skor Sentimen', 'change_%': 'Perubahan Harga (%)'}
    )
    fig_corr.update_layout(template="plotly_white")
    st.plotly_chart(fig_corr, use_container_width=True)

# ==========================================
# TAB 4: Detail Data
# ==========================================
with tab4:
    st.header("Detail Data Mentah & Hasil Olahan")
    st.markdown("Tabel di bawah ini menampilkan data yang telah digabungkan, dibersihkan, dan dilengkapi dengan fitur teknikal serta sentimen.")
    
    st.dataframe(
        df_full,
        use_container_width=True,
        height=600,
        column_config={
            "date": st.column_config.DateColumn("Tanggal", format="DD MMM YYYY"),
            "price": st.column_config.NumberColumn("Harga (USD)", format="$%.2f"),
            "sentiment_score": st.column_config.NumberColumn("Skor Sentimen", format="%.2f"),
        }
    )
    
    # Tombol Download
    csv = df_full.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Data Lengkap sebagai CSV",
        data=csv,
        file_name='gold_sentiment_processed_data.csv',
        mime='text/csv',
    )