import pandas as pd
import re

class NLPPreprocessor:
    def clean_text(self, data: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        df = data.copy()

        if text_column not in df.columns:
            if 'body' in df.columns: text_column = 'body'
            elif 'content' in df.columns: text_column = 'content'
            else: return df

        df[text_column] = df[text_column].fillna("")
        df[text_column] = df[text_column].astype(str).str.lower()
        df[text_column] = df[text_column].apply(lambda x: re.sub(r'http\S+|www\S+|https\S+', '', str(x), flags=re.MULTILINE))
        df[text_column] = df[text_column].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)))

        self.active_text_column = text_column
        return df

    def aggregate_texts(self, data: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, 'active_text_column'):
            return data

        grouped = data.groupby('user_id')[self.active_text_column].apply(lambda x: ' '.join(x.astype(str))).reset_index()
        grouped.rename(columns={self.active_text_column: 'aggregated_text'}, inplace=True)
        return grouped