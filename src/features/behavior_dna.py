import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class DigitalDNAExtractor:
    def __init__(self, max_sequence_length: int = 100):
        self.max_len = max_sequence_length
        self.tokenizer = Tokenizer(char_level=True, lower=False, oov_token='U')

    def _map_to_action(self, row: pd.Series) -> str:
        if pd.notna(row.get('retweeted_status_id')):
            return 'C'
        elif pd.notna(row.get('in_reply_to_status_id')):
            return 'T'
        else:
            return 'A'

    def extract_dna_sequences(self, data: pd.DataFrame) -> pd.DataFrame:
        data['action_char'] = data.apply(self._map_to_action, axis=1)
        dna_series = data.groupby('user_id')['action_char'].apply(lambda chars: ''.join(chars))
        df_dna = dna_series.reset_index()
        df_dna.rename(columns={'action_char': 'dna_string'}, inplace=True)
        return df_dna

    def tokenize_and_pad(self, df_dna: pd.DataFrame, is_train: bool = True):
        dna_strings = df_dna['dna_string'].values

        if is_train:
            self.tokenizer.fit_on_texts(dna_strings)

        sequences = self.tokenizer.texts_to_sequences(dna_strings)

        padded_seqs = pad_sequences(
            sequences,
            maxlen=self.max_len,
            padding='post',
            truncating='post'
        )

        return padded_seqs, self.tokenizer.word_index