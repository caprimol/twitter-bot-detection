import os
import pandas as pd
from abc import ABC, abstractmethod

class DataLoader(ABC):
    @abstractmethod
    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        pass

class TwitterDatasetLoader(DataLoader):
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.bot_folders = [
            "fake_followers.csv",
            "social_spambots_1.csv",
            "social_spambots_2.csv",
            "social_spambots_3.csv",
            "traditional_spambots_1.csv"
        ]
        self.real_folders = ["genuine_accounts.csv"]
        self.users_cols = ['id', 'followers_count', 'friends_count', 'statuses_count', 'favourites_count', 'listed_count']
        self.tweets_cols = ['user_id', 'text', 'timestamp', 'retweet_count', 'favorite_count', 'num_hashtags', 'num_urls']

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        all_users = []
        all_tweets = []
        
        for folder in self.bot_folders + self.real_folders:
            folder_path = os.path.join(self.base_path, folder)
            users_file = os.path.join(folder_path, "users.csv")
            tweets_file = os.path.join(folder_path, "tweets.csv")
            
            is_bot = 1 if folder in self.bot_folders else 0
            
            try:
                df_users = pd.read_csv(
                    users_file, 
                    usecols=lambda c: c in self.users_cols, 
                    low_memory=False,
                    encoding="utf-8",
                    encoding_errors="replace"
                )
                df_users['label'] = is_bot
                all_users.append(df_users)
                
                df_tweets = pd.read_csv(
                    tweets_file, 
                    usecols=lambda c: c in self.tweets_cols, 
                    low_memory=False,
                    encoding="utf-8",
                    encoding_errors="replace"
                )
                df_tweets['label'] = is_bot
                all_tweets.append(df_tweets)
                
            except Exception as e:
                print(f"Error while loading data from {folder}: {e}")
                
        final_users = pd.concat(all_users, ignore_index=True) if all_users else pd.DataFrame()
        final_tweets = pd.concat(all_tweets, ignore_index=True) if all_tweets else pd.DataFrame()
        
        return final_users, final_tweets