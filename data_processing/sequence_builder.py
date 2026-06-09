import numpy as np
import pandas as pd
from tqdm import tqdm

class SequenceBuilder:
    def __init__(self, sequence_length: int = 100):
        self.sequence_length = sequence_length

    def build_sequences(self, tweets_dataframe: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        print("Combining timestamp and text vectors...")
        
        def combine_features(row):
            return [row['timestamp_normalized']] + row['text_vector']
            
        tweets_dataframe['combined_vector'] = tweets_dataframe.apply(combine_features, axis=1)
        
        print("Grouping tweets by user to form sequences...")
        grouped = tweets_dataframe.groupby('user_id')
        
        sequences = []
        labels = []
        
        feature_dim = len(tweets_dataframe.iloc[0]['combined_vector'])
        
        for user_id, group in tqdm(grouped, desc="Budowanie sekwencji użytkowników"):
            user_sequence = group['combined_vector'].tolist()
            user_label = group['label'].iloc[0]
            
            if len(user_sequence) < self.sequence_length:
                padding_length = self.sequence_length - len(user_sequence)
                padding = [[0.0] * feature_dim for _ in range(padding_length)]
                user_sequence = padding + user_sequence
                
            sequences.append(user_sequence)
            labels.append(user_label)
            
        print("Converting sequences to numpy arrays...")
        X = np.array(sequences)
        y = np.array(labels)
        
        return X, y