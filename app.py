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
    current_seq = last_sequence_scaled.copy()  # Shape: (1, 60, 7)

    for _ in range(days_ahead):
        # 1. Prediksi 1 langkah ke depan
        pred_scaled = model.predict(current_seq, verbose=0)
        predictions_scaled.append(pred_scaled[0][0])

        # 2. Siapkan input untuk langkah berikutnya
        # Asumsi: Fitur lain (selain price) dianggap konstan sama dengan hari terakhir
        last_row = current_seq[0, -1, :].copy()
        last_row[0] = pred_scaled[0][0]  # Update hanya kolom 'price' (index 0)

        # Geser array (buang hari tertua, tambah hari prediksi)
        new_seq = np.append(current_seq[0, 1:, :], last_row.reshape(1, len(FEATURES)), axis=0)
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
st.sidebar.title("⚙️ Pengaturan Dashboard")
st.sidebar.markdown("---")
st.sidebar.info("""
**Tentang Proyek Ini:**

Dashboard ini memprediksi harga emas (Gold Futures) menggunakan model Deep Learning LSTM Multimodal yang menggabungkan:

1. Data Historis Harga
2. Indikator Teknikal (RSI, MACD, SMA)
3. Analisis Sentimen Berita Keuangan Indonesia
""")

st.title(" Dashboard Prediksi Harga Emas (Gold Futures)")
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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Integrasi Data & Sentimen",
    "📊 Analisis Historis Harga",
    "📰 Analisis Sentimen Berita",
    "📁 Detail Data",
    "🚀 Prediksi Masa Depan"  # TAB BARU
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
    st.plotly_chart(fig_actual, use_container_width=True)

    # Info Box Penjelasan
    st.info(f"""
    **📊 Informasi Data:**
    - Rentang data lengkap: **{min_date_full}** sampai **{max_date_full}**
    - Data Training (80%): Januari 2021 - Agustus 2025 (model belajar dari data ini)
    - Data Testing (20%): Agustus 2025 - April 2026 (model diuji pada data ini)
    - **Garis prediksi (merah putus-putus) hanya muncul di periode Data Testing** karena model hanya diprediksi pada data yang belum pernah dilihat.
    """)

    # Metrik Akurasi (Dihitung Dinamis dalam Persentase)
    st.markdown("### 📏 Metrik Akurasi Model (Pada Data Testing 20%)")

    if not df_pred.empty:
        actuals = df_pred['Actual_Price'].values
        preds = df_pred['Predicted_Price'].values

        # 1. MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((actuals - preds) / (actuals + 1e-8))) * 100

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
        st.warning("⚠️ Data testing tidak tersedia. Jalankan `python -m src.evaluation` terlebih dahulu di terminal!")

    st.markdown("---")


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
    if 'SMA_20' in df_filtered_hist.columns:
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
    st.plotly_chart(fig_hist, use_container_width=True)

    # Grafik RSI
    if 'RSI' in df_filtered_hist.columns:
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
        monthly_sentiment,
        x='Month',
        y='sentiment_score',
        title="Rata-rata Skor Sentimen Berita per Bulan",
        color='sentiment_score',
        color_continuous_scale='RdYlGn',
        labels={'sentiment_score': 'Skor Sentimen', 'Month': 'Bulan'}
    )
    fig_sent.update_layout(template="plotly_white")
    st.plotly_chart(fig_sent, use_container_width=True)

    # Scatter Plot Korelasi
    if 'change_%' in df_full.columns:
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
        st.plotly_chart(fig_corr, use_container_width=True)


# ==========================================
# TAB 4: Detail Data
# ==========================================
with tab4:
    st.header("Detail Data Mentah & Hasil Olahan")
    st.markdown("Tabel di bawah ini menampilkan data yang telah digabungkan, dibersihkan, dan dilengkapi dengan fitur teknikal serta sentimen.")

    # --- TABEL DATA HISTORIS ---
    st.subheader(" Data Historis (2021 - April 2026)")
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
        label="📥 Download Data Historis sebagai CSV",
        data=csv_historis,
        file_name='gold_sentiment_historis.csv',
        mime='text/csv',
    )

    st.markdown("---")

    # --- TABEL PREDIKSI TESTING ---
    if not df_pred.empty:
        st.subheader("📋 Data Prediksi Testing (Data Testing 20%)")
        st.dataframe(
            df_pred,
            use_container_width=True,
            height=400,
            column_config={
                "Date": st.column_config.DateColumn("Tanggal", format="DD MMM YYYY"),
                "Actual_Price": st.column_config.NumberColumn("Harga Aktual (USD)", format="$%.2f"),
                "Predicted_Price": st.column_config.NumberColumn("Harga Prediksi (USD)", format="$%.2f"),
            }
        )

        # Tombol Download Data Prediksi Testing
        csv_prediksi_testing = df_pred.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Data Prediksi Testing sebagai CSV",
            data=csv_prediksi_testing,
            file_name='gold_sentiment_prediksi_testing.csv',
            mime='text/csv',
        )


# ==========================================
# TAB 5: PREDIKSI MASA DEPAN (FUTURE FORECASTING)
# ==========================================
with tab5:
    st.header("🔮 Prediksi Harga Emas Masa Depan")
    st.markdown("""
    Gunakan model yang sudah dilatih untuk memprediksi harga emas di masa depan. 
    Prediksi dilakukan secara **rekursif** - model memprediksi 1 hari, lalu menggunakan 
    hasil tersebut sebagai input untuk hari berikutnya.
    
    **⚠️ Catatan Penting:**
    - Prediksi lebih dari 30 hari tidak disarankan karena akumulasi error
    - Prediksi ini mengasumsikan kondisi pasar relatif stabil
    - Gunakan sebagai referensi, bukan satu-satunya dasar keputusan investasi
    """)

    # Informasi data terakhir
    last_date = df_full['date'].max()
    last_date_str = last_date.strftime("%d %B %Y")
    last_price = df_full['price'].iloc[-1]

    st.info(f"📅 **Titik Mulai Prediksi:** {last_date_str} (Harga Terakhir: ${last_price:,.2f})")

    # Pilihan durasi prediksi
    duration_options = {
        "7 Hari (1 Minggu)": 7,
        "14 Hari (2 Minggu)": 14,
        "30 Hari (1 Bulan)": 30
    }

    selected_duration = st.selectbox(
        "Pilih durasi prediksi:",
        options=list(duration_options.keys()),
        index=0
    )

    days_ahead = duration_options[selected_duration]

    # Tombol jalankan prediksi
    if st.button("🚀 Jalankan Prediksi", type="primary"):
        with st.spinner(f"Sedang memprediksi {days_ahead} hari ke depan..."):

            # 1. Ambil 60 hari terakhir untuk input awal
            last_60_days_raw = df_full[FEATURES].iloc[-SEQUENCE_LENGTH:].values
            last_60_days_scaled = scaler_X.transform(last_60_days_raw).reshape(1, SEQUENCE_LENGTH, len(FEATURES))

            # 2. Jalankan prediksi rekursif
            future_predictions = predict_future_days(
                model,
                last_60_days_scaled,
                scaler_y,
                days_ahead
            )

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

            # 4. Simpan hasil ke session state
            st.session_state['future_predictions'] = pd.DataFrame({
                'Tanggal': future_dates,
                'Prediksi_Harga_USD': future_predictions,
                'Durasi': f"{days_ahead} Hari"
            })

            st.success(f"✅ Prediksi {days_ahead} hari berhasil!")

            # 5. Visualisasi Grafik
            fig_future = go.Figure()

            # Tambahkan 10 hari terakhir sebagai konteks
            last_10_dates = df_full['date'].iloc[-10:].tolist()
            last_10_prices = df_full['price'].iloc[-10:].tolist()

            fig_future.add_trace(go.Scatter(
                x=last_10_dates,
                y=last_10_prices,
                mode='lines+markers',
                name='10 Hari Terakhir (Aktual)',
                line=dict(color='royalblue', width=2),
                marker=dict(size=6)
            ))

            # Tambahkan prediksi
            fig_future.add_trace(go.Scatter(
                x=future_dates,
                y=future_predictions,
                mode='lines+markers',
                name=f'Prediksi ({days_ahead} Hari)',
                line=dict(color='crimson', width=3, dash='dot'),
                marker=dict(size=8, symbol='diamond', color='crimson')
            ))

            # Garis vertikal pemisah
            fig_future.add_vline(
                x=last_date,
                line_dash="dash",
                line_color="gray",
                annotation_text="Batas Data Historis"
            )

            fig_future.update_layout(
                title=f"Proyeksi Harga Emas: {days_ahead} Hari Kerja Ke Depan",
                xaxis_title="Tanggal",
                yaxis_title="Harga Emas (USD)",
                hovermode="x unified",
                template="plotly_white",
                xaxis=dict(tickformat="%d %b %Y"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            st.plotly_chart(fig_future, use_container_width=True)

            # 6. Statistik Prediksi
            st.subheader("📊 Statistik Prediksi")

            col1, col2, col3, col4 = st.columns(4)

            pred_min = future_predictions.min()
            pred_max = future_predictions.max()
            pred_mean = future_predictions.mean()
            pred_change = ((future_predictions[-1] - last_price) / last_price) * 100

            col1.metric(
                label="Harga Prediksi Terendah",
                value=f"${pred_min:,.2f}"
            )

            col2.metric(
                label="Harga Prediksi Tertinggi",
                value=f"${pred_max:,.2f}"
            )

            col3.metric(
                label="Rata-rata Prediksi",
                value=f"${pred_mean:,.2f}"
            )

            col4.metric(
                label="Perubahan dari Harga Terakhir",
                value=f"{pred_change:+.2f}%",
                delta=f"${future_predictions[-1] - last_price:+,.2f}",
                delta_color="normal" if pred_change > 0 else "inverse"
            )

            # 7. Tabel Detail Prediksi
            st.subheader("📋 Detail Prediksi per Hari")

            df_pred_detail = st.session_state['future_predictions'].copy()
            df_pred_detail['Perubahan_Harian'] = df_pred_detail['Prediksi_Harga_USD'].pct_change() * 100

            # Format tampilan
            st.dataframe(
                df_pred_detail.style.format({
                    'Prediksi_Harga_USD': '${:,.2f}',
                    'Perubahan_Harian': '{:+.2f}%'
                }),
                use_container_width=True,
                hide_index=True
            )

            # 8. Tombol Download
            csv_prediksi = df_pred_detail.to_csv(index=False, encoding='utf-8')
            st.download_button(
                label="📥 Download Hasil Prediksi (CSV)",
                data=csv_prediksi,
                file_name=f'prediksi_emas_{days_ahead}_hari_{last_date.strftime("%Y%m%d")}.csv',
                mime='text/csv'
            )

            # 9. Analisis Tren
            st.subheader("📈 Analisis Tren")

            if pred_change > 0:
                st.success(f"📈 **Tren Naik:** Model memprediksi kenaikan harga sebesar {pred_change:.2f}% dalam {days_ahead} hari ke depan.")
            else:
                st.warning(f"📉 **Tren Turun:** Model memprediksi penurunan harga sebesar {abs(pred_change):.2f}% dalam {days_ahead} hari ke depan.")

            # Volatilitas
            volatility = future_predictions.std() / future_predictions.mean() * 100
            if volatility < 2:
                st.info(f" **Volatilitas Rendah:** {volatility:.2f}% - Pasar diprediksi stabil")
            elif volatility < 5:
                st.warning(f"🟡 **Volatilitas Sedang:** {volatility:.2f}% - Waspada fluktuasi")
            else:
                st.error(f"🔴 **Volatilitas Tinggi:** {volatility:.2f}% - Siapkan strategi hedging")