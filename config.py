# Konfigurasi Proyek

# Path Data
DATA_PATH = "data"
GOLD_DATA_FILE = f"{DATA_PATH}/Gold Futures Historical Data.csv"
SENTIMENT_DATA_FILE = f"{DATA_PATH}/dataset_indonesia_sentimen.xlsx"

# Hyperparameters LSTM
SEQUENCE_LENGTH = 30  # Jumlah hari yang digunakan untuk prediksi
TEST_SIZE = 0.2       # 20% data untuk testing
EPOCHS = 50           # Jumlah iterasi training
BATCH_SIZE = 32       # Ukuran batch

# Fitur yang akan digunakan (HARUS SAMA PERSIS dengan nama kolom hasil olahan)
FEATURES = [
    'price', 'open', 'high', 'low', 'change_%', 'vol', 
    'sentiment_score', 'avg_confidence', 'RSI', 'MACD', 
    'MACD_Signal', 'MACD_Diff', 'SMA_20'
]