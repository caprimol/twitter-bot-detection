import numpy as np
import pandas as pd
from tqdm import tqdm

class SequenceBuilder:
    def __init__(self, sequence_length: int = 100, feature_mode: str = 'both'):
        self.sequence_length = sequence_length
        self.feature_mode = feature_mode

    def build_sequences(self, tweets_dataframe: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        print(f"Budowanie sekwencji (Tryb: {self.feature_mode})...")
        
        def get_dna_features(row):
            text = str(row.get('text', ''))
            reply_id = str(row.get('in_reply_to_status_id', '0')).strip().upper()
            retweet_id = str(row.get('retweeted_status_id', '0')).strip().upper()
            
            is_retweet = text.startswith('RT @') or (retweet_id not in ['0', 'NULL', 'NAN', 'NONE', ''])
            
            is_reply = reply_id not in ['0', 'NULL', 'NAN', 'NONE', '']
            
            if is_retweet:
                return [0.0, 0.0, 1.0]
            elif is_reply:
                return [0.0, 1.0, 0.0]
            else:
                return [1.0, 0.0, 0.0]

        def combine_features(row):
            if self.feature_mode == 'text_only':
                return row['text_vector']
            elif self.feature_mode == 'time_only':
                return [row['timestamp_normalized']]
            elif self.feature_mode == 'dna_only':
                return get_dna_features(row)
            elif self.feature_mode == 'both':
                return [row['timestamp_normalized']] + row['text_vector']
            elif self.feature_mode == 'both_with_dna':
                dna_vector = get_dna_features(row)
                return [row['timestamp_normalized']] + dna_vector + row['text_vector']
            
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
            
        X = np.array(sequences, dtype=np.float32)
        y = np.array(labels)
        
        return X, y