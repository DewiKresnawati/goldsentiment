import re
import pandas as pd
from transformers import pipeline
from deep_translator import GoogleTranslator

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)   # hapus karakter khusus
    return text.lower().strip()               # case folding

def translate_text(text):                     # translasi ID->EN (in-memory, tidak disimpan)
    return GoogleTranslator(source='id', target='en').translate(text)

sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")

df_news = pd.read_excel('data/berita_mentah.xlsx')
df_news['cleaned']   = df_news['headline'].apply(clean_text)
df_news['translated'] = df_news['cleaned'].apply(translate_text)

results = df_news['translated'].apply(lambda x: sentiment_pipeline(x)[0])
df_news['sentiment']  = results.apply(lambda x: x['label'])      # positive/neutral/negative
df_news['confidence'] = results.apply(lambda x: x['score'])      # confidence score

# Yang disimpan hanya headline asli + hasil akhir (teks translasi = artefak sementara)
df_news[['date', 'headline', 'source', 'category', 'sentiment', 'confidence']].to_excel(
    'data/dataset_indonesia_sentimen.xlsx', index=False)