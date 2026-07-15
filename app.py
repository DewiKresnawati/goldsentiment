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
from datetime import datetime, timedelta

# ==========================================
# SETUP PATH & IMPORT MODUL
# ==========================================
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
    """Memuat hasil prediksi dari file CSV (data testing)"""
    if os.path.exists('models/test_predictions.csv'):
        return pd.read_csv('models/test_predictions.csv', parse_dates=['Date'])
    return pd.DataFrame()


@st.cache_resource
def load_scalers():
    """Memuat scaler yang sudah disimpan"""
    scaler_X = joblib.load('models/scaler_X.pkl')
    scaler_y = joblib.load('models/scaler_y.pkl')
    return scaler_X, scaler_y


# ==========================================
# 2. FUNGSI PREDIKSI MASA DEPAN (RECURSIVE)
# ==========================================
def predict_future_days(model, last_sequence_scaled, scaler_y, days_ahead):
    """
    Melakukan prediksi rekursif untuk N hari ke depan.
    Model memprediksi 1 hari, lalu menggunakan hasilnya sebagai input untuk hari berikutnya.
    """
    predictions_scaled = []
    current_seq = last_sequence_scaled.copy()  # Shape: (1, 30, 13)

    for _ in range(days_ahead):
        # 1. Prediksi 1 langkah ke depan
        pred_scaled = model.predict(current_seq, verbose=0)
        predictions_scaled.append(pred_scaled[0][0])

        # 2. Siapkan input untuk langkah berikutnya
        # Asumsi: Fitur lain (Open, High, RSI, dll) dianggap konstan sama dengan hari terakhir
        last_row = current_seq[0, -1, :]
        new_row = last_row.copy()
        new_row[0] = pred_scaled[0][0]  # Update hanya kolom 'price' (index 0)

        # Geser array (buang hari tertua, tambah hari prediksi)
        new_seq = np.append(current_seq[0, 1:, :], new_row.reshape(1, 13), axis=0)
        current_seq = new_seq.reshape(1, SEQUENCE_LENGTH, len(FEATURES))

    # Kembalikan ke nilai Dollar asli
    predictions_real = scaler_y.inverse_transform(
        np.array(predictions_scaled).reshape(-1, 1)
    ).flatten()
    return predictions_real


# ==========================================
# 3. KONFIGURASI HALAMAN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Gold Price Prediction Dashboard",
    page_icon="📈",
    layout="wide"  # Layout lebar agar grafik lebih lega
)

# ==========================================
# 4. SIDEBAR & HEADER
# ==========================================
st.sidebar.title("️ Pengaturan Dashboard")
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
    scaler_X, scaler_y = load_scalers()

st.markdown("---")

# ==========================================
# 5. TAB NAVIGASI
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Integrasi Data & Sentimen",
    "📊 Analisis Historis Harga",
    " Analisis Sentimen Berita",
    " Detail Data"
])


# ==========================================
# TAB 1: Prediksi vs Aktual + Future Forecasting
# ==========================================
with tab1:
    st.header("Integrasi Data & Sentimen")

    # --- BAGIAN A: Filter Tanggal untuk SELURUH Data Historis ---
    min_date_full = df_full['date'].min().date()
    max_date_full = df_full['date'].max().date()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Tanggal Mulai",
            min_date_full,
            min_value=min_date_full,
            max_value=max_date_full
        )
    with col2:
        end_date = st.date_input(
            "Tanggal Akhir",
            max_date_full,
            min_value=min_date_full,
            max_value=max_date_full
        )

    # Filter data historis lengkap
    mask_full = (df_full['date'].dt.date >= start_date) & (df_full['date'].dt.date <= end_date)
    df_filtered_full = df_full[mask_full]

    # Plot Grafik Harga Aktual (Seluruh Data)
    fig_actual = go.Figure()
    fig_actual.add_trace(go.Scatter(
        x=df_filtered_full['date'],
        y=df_filtered_full['price'],
        mode='lines',
        name='Harga Aktual',
        line=dict(color='gold', width=2)
    ))

    # Tambahkan prediksi jika ada di rentang tanggal yang dipilih
    if not df_pred.empty:
        mask_pred = (df_pred['Date'].dt.date >= start_date) & (df_pred['Date'].dt.date <= end_date)
        df_filtered_pred = df_pred[mask_pred]

        if not df_filtered_pred.empty:
            fig_actual.add_trace(go.Scatter(
                x=df_filtered_pred['Date'],
                y=df_filtered_pred['Predicted_Price'],
                mode='lines',
                name='Prediksi LSTM',
                line=dict(color='crimson', width=2, dash='dot')
            ))

    fig_actual.update_layout(
        title=f"Perbandingan Harga Emas Aktual vs Prediksi ({start_date} s/d {end_date})",
        xaxis_title="Tanggal",
        yaxis_title="Harga Emas (USD)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_actual, width="stretch")

    # Info Box Penjelasan
    st.info(f"""
    **📊 Informasi Data:**
    - Rentang data lengkap: **{min_date_full}** sampai **{max_date_full}**
    - Data Training (80%): Januari 2021 - ~Juni 2025 (model belajar dari data ini)
    - Data Testing (20%): ~Juni 2025 - April 2026 (model diuji pada data ini)
    - **Garis prediksi (merah putus-putus) hanya muncul di periode Data Testing** karena model hanya diprediksi pada data yang belum pernah dilihat.
    """)

        # Metrik Akurasi (Dihitung Dinamis dalam Persentase)
    st.markdown("### 📏 Metrik Akurasi Model (Pada Data Testing 20%)")
    
    if not df_pred.empty:
        actuals = df_pred['Actual_Price'].values
        preds = df_pred['Predicted_Price'].values
        
        # 1. MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((actuals - preds) / actuals)) * 100
        
        # 2. NRMSE (Normalized Root Mean Square Error)
        rmse = np.sqrt(np.mean((actuals - preds)**2))
        nrmse = (rmse / np.mean(actuals)) * 100
        
        # 3. Directional Accuracy (DA) - Seberapa sering tebakan arah (Naik/Turun) benar
        actual_direction = np.sign(np.diff(actuals))
        pred_direction = np.sign(np.diff(preds))
        da = np.mean(actual_direction == pred_direction) * 100
        
        col1, col2, col3 = st.columns(3)
        
        col1.metric(
            label="MAPE (Mean Absolute % Error)",
            value=f"{mape:.2f}%",
            help="Rata-rata persentase penyimpangan prediksi dari harga asli."
        )
        
        col2.metric(
            label="NRMSE (Normalized RMSE)",
            value=f"{nrmse:.2f}%",
            help="Root Mean Square Error yang dinormalisasi terhadap rata-rata harga, untuk perbandingan yang adil."
        )
        
        col3.metric(
            label="DA (Directional Accuracy)",
            value=f"{da:.2f}%",
            delta="Tebakan Arah",
            delta_color="normal" if da > 50 else "inverse",
            help="Persentase keberhasilan model dalam menebak arah harga (Naik/Turun) dengan benar. >50% berarti lebih baik dari tebakan acak."
        )
    else:
        st.warning("Data testing tidak tersedia untuk menghitung metrik.")

    st.markdown("---")

    # --- BAGIAN B: Prediksi Masa Depan (Future Forecasting) ---
    st.header("🔮 Prediksi Masa Depan (Future Forecasting)")
    st.markdown("Gunakan model yang sudah dilatih untuk meramal harga emas di masa depan.")

    last_date = df_full['date'].max()
    last_date_str = last_date.strftime("%d %B %Y")
    last_price = df_full['price'].iloc[-1]

    st.info(f"📅 **Titik Mulai Prediksi:** Hari terakhir data tersedia adalah **{last_date_str}** (Harga: ${last_price:,.2f}).")
    st.warning("⚠️ **Catatan Ilmiah:** Model memprediksi hari demi hari secara berantai (rekursif). Memprediksi lebih dari 30 hari ke depan tidak disarankan karena akumulasi error.")

    duration_options = {
        "1 Minggu (7 Hari)": 7,
        "2 Minggu (14 Hari)": 14,
        "1 Bulan (30 Hari)": 30
    }

    selected_duration = st.selectbox(
        "Pilih durasi prediksi ke depan:",
        options=list(duration_options.keys()),
        index=0,
        key="duration_selector_tab1"
    )

    days_ahead = duration_options[selected_duration]

    if st.button("🚀 Jalankan Prediksi Masa Depan", type="primary"):
        with st.spinner(f"Sedang meramal harga untuk {days_ahead} hari ke depan dari {last_date_str}..."):

            # 1. Ambil 30 hari terakhir dari data asli yang sudah di-scale
            last_30_days_raw = df_full[FEATURES].iloc[-SEQUENCE_LENGTH:].values
            last_30_days_scaled = scaler_X.transform(last_30_days_raw).reshape(1, SEQUENCE_LENGTH, len(FEATURES))

            # 2. Jalankan fungsi prediksi rekursif
            future_predictions = predict_future_days(model, last_30_days_scaled, scaler_y, days_ahead)

            # 3. Buat tanggal untuk masa depan (hanya hari kerja)
            future_dates = []
            current_date = last_date
            days_generated = 0

            while days_generated < days_ahead:
                current_date += timedelta(days=1)
                # Skip Sabtu (5) dan Minggu (6)
                if current_date.weekday() < 5:
                    future_dates.append(current_date)
                    days_generated += 1

            # SIMPAN HASIL PREDIKSI KE SESSION STATE agar bisa diakses Tab 4
            st.session_state['future_predictions'] = pd.DataFrame({
                'Tanggal_Prediksi': future_dates,
                'Prediksi_Harga_Emas_USD': future_predictions,
                'Sumber': 'Prediksi LSTM (Masa Depan)'
            })

            st.success(f"✅ Prediksi {days_ahead} hari berhasil dibuat! Lihat hasilnya di grafik ini dan Tab 4 (tabel detail).")

            # 4. Plot Future Prediction
            fig_future = go.Figure()

            # Tambahkan 5 titik terakhir data aktual sebagai konteks (anchor)
            last_5_dates = df_full['date'].iloc[-5:].tolist()
            last_5_prices = df_full['price'].iloc[-5:].tolist()

            fig_future.add_trace(go.Scatter(
                x=last_5_dates,
                y=last_5_prices,
                mode='lines+markers',
                name='5 Hari Terakhir (Aktual)',
                line=dict(color='royalblue', width=2),
                marker=dict(size=6)
            ))

            fig_future.add_trace(go.Scatter(
                x=future_dates,
                y=future_predictions,
                mode='lines+markers',
                name=f'Prediksi ({days_ahead} Hari)',
                line=dict(color='crimson', width=3, dash='dot'),
                marker=dict(size=8, symbol='diamond')
            ))

            # Tambahkan garis vertikal pemisah
            fig_future.add_vline(x=last_date, line_dash="dash", line_color="gray", annotation_text="Batas Data Historis")

            fig_future.update_layout(
                title=f"Proyeksi Harga Emas: {days_ahead} Hari Kerja Ke Depan",
                xaxis_title="Tanggal",
                yaxis_title="Harga Emas (USD)",
                hovermode="x unified",
                template="plotly_white",
                xaxis=dict(tickformat="%d %b %Y")
            )
            st.plotly_chart(fig_future, width="stretch")

            # Tampilkan tabel prediksi
            st.subheader("Detail Angka Prediksi")
            st.dataframe(
                st.session_state['future_predictions'].style.format({'Prediksi_Harga_Emas_USD': '${:,.2f}'}),
                width="stretch",
                hide_index=True
            )


# ==========================================
# TAB 2: Analisis Historis Harga & Indikator Teknikal
# ==========================================
with tab2:
    st.header("Analisis Historis Harga & Indikator Teknikal")

    # Filter Tanggal
    min_date_hist = df_full['date'].min().date()
    max_date_hist = df_full['date'].max().date()
    start_date_hist, end_date_hist = st.date_input(
        "Pilih Rentang Tanggal",
        [min_date_hist, max_date_hist],
        min_value=min_date_hist,
        max_value=max_date_hist
    )

    df_filtered_hist = df_full[
        (df_full['date'].dt.date >= start_date_hist) &
        (df_full['date'].dt.date <= end_date_hist)
    ]

    # Grafik Harga + SMA
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=df_filtered_hist['date'],
        y=df_filtered_hist['price'],
        mode='lines',
        name='Harga Close',
        line=dict(color='gold', width=2)
    ))
    fig_hist.add_trace(go.Scatter(
        x=df_filtered_hist['date'],
        y=df_filtered_hist['SMA_20'],
        mode='lines',
        name='SMA 20',
        line=dict(color='blue', width=1, dash='dash')
    ))

    fig_hist.update_layout(
        title="Tren Harga Emas dan Simple Moving Average (SMA 20)",
        xaxis_title="Tanggal",
        yaxis_title="Harga (USD)",
        hovermode="x unified",
        template="plotly_white"
    )
    st.plotly_chart(fig_hist, width="stretch")

    # Grafik RSI
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(
        x=df_filtered_hist['date'],
        y=df_filtered_hist['RSI'],
        mode='lines',
        name='RSI (14)',
        line=dict(color='purple', width=1)
    ))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (Jenuh Beli)")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (Jenuh Jual)")

    fig_rsi.update_layout(
        title="Relative Strength Index (RSI)",
        xaxis_title="Tanggal",
        yaxis_title="RSI",
        yaxis_range=[0, 100],
        hovermode="x unified",
        template="plotly_white"
    )
    st.plotly_chart(fig_rsi, width="stretch")


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
        monthly_sentiment,
        x='Month',
        y='sentiment_score',
        title="Rata-rata Skor Sentimen Berita per Bulan",
        color='sentiment_score',
        color_continuous_scale='RdYlGn',
        labels={'sentiment_score': 'Skor Sentimen', 'Month': 'Bulan'}
    )
    fig_sent.update_layout(template="plotly_white")
    st.plotly_chart(fig_sent, width="stretch")

    # Scatter Plot Korelasi
    st.subheader("Korelasi Sentimen vs Perubahan Harga Harian")
    fig_corr = px.scatter(
        df_full,
        x='sentiment_score',
        y='change_%',
        title="Scatter Plot: Skor Sentimen vs Perubahan Harga (%)",
        opacity=0.6,
        trendline="ols",
        labels={'sentiment_score': 'Skor Sentimen', 'change_%': 'Perubahan Harga (%)'}
    )
    fig_corr.update_layout(template="plotly_white")
    st.plotly_chart(fig_corr, width="stretch")


# ==========================================
# TAB 4: Detail Data
# ==========================================
with tab4:
    st.header("Detail Data Mentah & Hasil Olahan")
    st.markdown("Tabel di bawah ini menampilkan data yang telah digabungkan, dibersihkan, dan dilengkapi dengan fitur teknikal serta sentimen.")

    # --- TABEL DATA HISTORIS ---
    st.subheader("📊 Data Historis (2021 - April 2026)")
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

    # Tombol Download Data Historis
    csv_historis = df_full.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=" Download Data Historis sebagai CSV",
        data=csv_historis,
        file_name='gold_sentiment_historis.csv',
        mime='text/csv',
    )

    st.markdown("---")

    # --- TABEL PREDIKSI MASA DEPAN (dari Session State) ---
    st.subheader("🔮 Data Prediksi Masa Depan (Future Forecasting)")

    if 'future_predictions' in st.session_state:
        df_future = st.session_state['future_predictions']

        st.info(f"✨ Menampilkan **{len(df_future)} hari** prediksi ke depan, dimulai dari {df_future['Tanggal_Prediksi'].min().strftime('%d %B %Y')}.")

        # Tampilkan tabel prediksi
        st.dataframe(
            df_future.style.format({'Prediksi_Harga_Emas_USD': '${:,.2f}'}),
            use_container_width=True,
            height=400,
            hide_index=True,
            column_config={
                "Tanggal_Prediksi": st.column_config.DateColumn("Tanggal Prediksi", format="DD MMM YYYY"),
                "Prediksi_Harga_Emas_USD": st.column_config.NumberColumn("Prediksi Harga (USD)", format="$%.2f"),
            }
        )

        # Tombol Download Data Prediksi
        csv_prediksi = df_future.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Data Prediksi sebagai CSV",
            data=csv_prediksi,
            file_name='gold_sentiment_prediksi_masa_depan.csv',
            mime='text/csv',
        )

        st.success("✅ Data prediksi ini juga bisa digabungkan dengan data historis untuk analisis lebih lanjut.")

    else:
        st.warning("⚠️ **Belum ada data prediksi.** Silakan buka **Tab 1 (Prediksi vs Aktual)**, pilih durasi prediksi, dan klik tombol **'🚀 Jalankan Prediksi Masa Depan'** terlebih dahulu. Setelah itu, kembali ke tab ini untuk melihat hasilnya.")