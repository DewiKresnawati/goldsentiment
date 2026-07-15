# Konfigurasi Proyek

# Path Data
DATA_PATH = "data"
GOLD_DATA_FILE = f"{DATA_PATH}/Gold Futures Historical Data.csv"  # CSV, bukan XLSX
SENTIMENT_DATA_FILE = f"{DATA_PATH}/dataset_indonesia_sentimen.xlsx"

# Hyperparameters LSTM
SEQUENCE_LENGTH = 30  # Jumlah hari yang digunakan untuk prediksi
TEST_SIZE = 0.2       # 20% data untuk testing
EPOCHS = 50           # Jumlah iterasi training
BATCH_SIZE = 32       # Ukuran batch

# Fitur yang akan digunakan
FEATURES = ['Close', 'Open', 'High', 'Low', 'Vol', 
            'RSI', 'MACD', 'SMA_20', 'Sentiment_Score']