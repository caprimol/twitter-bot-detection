import numpy as np
import pandas as pd
from tqdm import tqdm

class SequenceBuilder:
    def __init__(self, sequence_length: int = 100, feature_mode: str = 'both'):
        self.sequence_length = sequence_length
        self.feature_mode = feature_mode # Dodany przełącznik trybu

    def build_sequences(self, tweets_dataframe: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        print(f"Budowanie sekwencji (Tryb: {self.feature_mode})...")
        
        # Logika wyboru cech (Ablation)
        def combine_features(row):
            if self.feature_mode == 'text_only':
                return row['text_vector']
            elif self.feature_mode == 'time_only':
                return [row['timestamp_normalized']]
            else: # 'both'
                return [row['timestamp_normalized']] + row['text_vector']
            
        tweets_dataframe['combined_vector'] = tweets_dataframe.apply(combine_features, axis=1)
        
        grouped = tweets_dataframe.groupby('user_id')
        sequences = []
        labels = []
        
        feature_dim = len(tweets_dataframe.iloc[0]['combined_vector'])
        
        for user_id, group in tqdm(grouped, desc=f"Przetwarzanie uzytkownikow ({self.feature_mode})"):
            user_sequence = group['combined_vector'].tolist()
            user_label = group['label'].iloc[0]
            
            if len(user_sequence) < self.sequence_length:
                padding_length = self.sequence_length - len(user_sequence)
                padding = [[0.0] * feature_dim for _ in range(padding_length)]
                user_sequence = padding + user_sequence
                
            sequences.append(user_sequence)
            labels.append(user_label)
            
        X = np.array(sequences, dtype=np.float32) # Utrzymujemy oszczędzanie RAM!
        y = np.array(labels)
        
        return X, y