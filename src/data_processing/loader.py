import pandas as pd

class CSVDataLoader:
    def load_data(self, filepath: str, limit_rows: int = None) -> pd.DataFrame:
        return pd.read_csv(filepath, low_memory=False, nrows=limit_rows)

    def preprocess_and_filter(self, data: pd.DataFrame, min_quantile: float = 0.25) -> pd.DataFrame:
        user_col = 'user_id' if 'user_id' in data.columns else 'author_id'

        counts = data[user_col].value_counts()
        threshold = counts.quantile(min_quantile)
        valid_users = counts[counts >= threshold].index
        filtered_data = data[data[user_col].isin(valid_users)].copy()

        time_col = 'created_at' if 'created_at' in filtered_data.columns else 'timestamp'

        if time_col in filtered_data.columns:
            filtered_data[time_col] = pd.to_datetime(filtered_data[time_col], errors='coerce', format='mixed')
            filtered_data = filtered_data.sort_values(by=[user_col, time_col])

        filtered_data.rename(columns={user_col: 'user_id'}, inplace=True)
        return filtered_data