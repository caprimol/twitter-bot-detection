import pandas as pd
from sentence_transformers import SentenceTransformer

class BertEncoder:
    def __init__(self):
        print("Ładowanie modelu BERT do pamięci...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def encode_tweets(self, tweets_dataframe: pd.DataFrame) -> pd.DataFrame:
        print("Kodowanie tweetów za pomocą modelu BERT (to potrwa dłużej niż TF-IDF)...")
        tweets_dataframe['text'] = tweets_dataframe['text'].fillna('')
        
        vectors = self.model.encode(tweets_dataframe['text'].tolist(), show_progress_bar=True)
        
        print("Dodawanie wektorów BERT do ramki danych...")
        tweets_dataframe['text_vector'] = vectors.tolist()
        
        return tweets_dataframe