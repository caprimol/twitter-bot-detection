import numpy as np
import pandas as pd
from tqdm import tqdm

class SequenceBuilder:
    def __init__(self, sequence_length: int = 100, feature_mode: str = 'both'):
        self.sequence_length = sequence_length
        self.feature_mode = feature_mode

    def build_sequences(self, tweets_dataframe: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        print(f"Budowanie sekwencji (Tryb: {self.feature_mode})...")
        
        # --- NOWA FUNKCJA: Ekstrakcja Digital DNA ---
        def get_dna_features(row):
            # Zabezpieczenie przed brakującymi danymi (NaN -> string)
            text = str(row.get('text', ''))
            reply_id = str(row.get('in_reply_to_status_id', '0')).strip().upper()
            retweet_id = str(row.get('retweeted_status_id', '0')).strip().upper()
            
            # Logika T (Retweet): Zaczyna się od "RT @" lub ma przypisane ID retweeta
            is_retweet = text.startswith('RT @') or (retweet_id not in ['0', 'NULL', 'NAN', 'NONE', ''])
            
            # Logika C (Conversation/Reply): Ma przypisane ID odpowiedzi
            is_reply = reply_id not in ['0', 'NULL', 'NAN', 'NONE', '']
            
            # Kodowanie One-Hot [A, C, T]
            if is_retweet:
                return [0.0, 0.0, 1.0] # T (Retweet)
            elif is_reply:
                return [0.0, 1.0, 0.0] # C (Reply)
            else:
                return [1.0, 0.0, 0.0] # A (Action / Autorski wpis)

        def combine_features(row):
            if self.feature_mode == 'text_only':
                return row['text_vector']
            elif self.feature_mode == 'time_only':
                return [row['timestamp_normalized']]
            elif self.feature_mode == 'both':
                return [row['timestamp_normalized']] + row['text_vector']
            elif self.feature_mode == 'both_with_dna':
                # NOWY TRYB: Czas + 3 bity Digital DNA + BERT
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