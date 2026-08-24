# Konfigurasi Proyek
DATA_PATH = "data"
GOLD_DATA_FILE = f"{DATA_PATH}/Gold Futures Historical Data.csv"
SENTIMENT_DATA_FILE = f"{DATA_PATH}/dataset_indonesia_sentimen.xlsx"

# Hyperparameters LSTM
SEQUENCE_LENGTH = 60
TEST_SIZE = 0.2
EPOCHS = 100
BATCH_SIZE = 32

# Fitur yang digunakan (13 fitur sesuai jurnal)
FEATURES = [
    'price', 'RSI', 'MACD', 'SMA_20', 'SMA_50', 
    'sentiment_score', 'avg_confidence'
]