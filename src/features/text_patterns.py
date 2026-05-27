import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class TFIDFTextExtractor:
    def __init__(self, max_features: int = 500):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )

    def extract_features(self, df_aggregated: pd.DataFrame, is_train: bool = True) -> np.ndarray:
        texts = df_aggregated['aggregated_text'].tolist()

        if is_train:
            features = self.vectorizer.fit_transform(texts)
        else:
            features = self.vectorizer.transform(texts)

        return features.toarray()