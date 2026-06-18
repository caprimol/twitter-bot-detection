# legacy code
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class TextEncoder:
    def __init__(self, max_features: int = 50):
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')

    def encode_tweets(self, tweets_dataframe: pd.DataFrame) -> pd.DataFrame:
        print("Fitting TF-IDF on tweets and transforming text to vectors...")
        
        tweets_dataframe['text'] = tweets_dataframe['text'].fillna('')
        
        vectors = self.vectorizer.fit_transform(tweets_dataframe['text']).toarray()
        
        print("Adding vectors to the dataframe...")
        tweets_dataframe['text_vector'] = vectors.tolist()
        
        return tweets_dataframe