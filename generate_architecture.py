import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import plot_model

# 1. Dapatkan path absolut dari folder tempat script ini berada
current_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_dir, 'model_architecture_early_fusion.png')

print("="*60)
print("MEMBANGUN ARSITEKTUR MODEL UNTUK VISUALISASI...")
print("="*60)
print(f"📍 File akan disimpan persis di:")
print(f"   {output_path}")
print("="*60)

SEQUENCE_LENGTH = 60
NUM_FEATURES = 7

model = Sequential([
    Input(shape=(SEQUENCE_LENGTH, NUM_FEATURES)), # Input Shape (60, 7)
    LSTM(128, return_sequences=True, activation='tanh'), # LSTM 1
    Dropout(0.2),
    LSTM(64, return_sequences=False, activation='tanh'), # LSTM 2
    Dropout(0.2),
    Dense(32, activation='relu'), # Dense
    Dense(1, activation='linear') # Output
])

# 2. Generate Gambar Arsitektur (Flowchart)
plot_model(
    model,
    to_file=output_path, # Gunakan path absolut yang sudah dipastikan
    show_shapes=True,          # Menampilkan dimensi shape
    show_layer_names=True,     # Menampilkan nama layer
    rankdir='TB',              # TB = Top to Bottom (atas ke bawah)
    dpi=300                    # Resolusi tinggi agar tidak pecah di PPT
)

print("\n✅ BERHASIL 100%!")
print(f"Silakan buka folder ini di Windows Explorer:")
print(f"📂 {current_dir}")
print("="*60)