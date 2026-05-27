import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    def __init__(self, sequence_length: int = 100):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()

    def preprocess_tweets(self, tweets_dataframe: pd.DataFrame) -> pd.DataFrame:
        print("Converting timestamps to datetime format...")
        tweets_dataframe['timestamp'] = pd.to_datetime(tweets_dataframe['timestamp'], errors='coerce')
        
        print("Dropping rows with invalid timestamps...")
        tweets_dataframe = tweets_dataframe.dropna(subset=['timestamp'])
        
        print("Sorting tweets by user and time...")
        tweets_dataframe = tweets_dataframe.sort_values(by=['user_id', 'timestamp'])
        
        print("Extracting latest tweets sequence per user...")
        tweets_dataframe = tweets_dataframe.groupby('user_id').tail(self.sequence_length)
        
        print("Normalizing timestamps...")
        timestamps_numeric = tweets_dataframe['timestamp'].astype('int64').values.reshape(-1, 1)
        tweets_dataframe['timestamp_normalized'] = self.scaler.fit_transform(timestamps_numeric)
        
        return tweets_dataframe