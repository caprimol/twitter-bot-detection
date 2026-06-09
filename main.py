from data_processing.data_loader import TwitterDatasetLoader
from data_processing.preprocessing import DataPreprocessor
from content_analysis.text_encoder import TextEncoder
from data_processing.sequence_builder import SequenceBuilder
from sequence_analysis.lstm_classifier import LSTMModel
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np

def main():
    base_directory = "datasets_full.csv"
    
    print("Initializing data loader...")
    loader = TwitterDatasetLoader(base_path=base_directory)
    users_dataframe, tweets_dataframe = loader.load_data()
    
    if not tweets_dataframe.empty:
        print("\nInitializing preprocessor...")
        preprocessor = DataPreprocessor(sequence_length=100)
        processed_tweets = preprocessor.preprocess_tweets(tweets_dataframe)
        
        print("\nInitializing text encoder...")
        encoder = TextEncoder(max_features=50)
        encoded_tweets = encoder.encode_tweets(processed_tweets)
        
        print("\nInitializing sequence builder...")
        sequence_builder = SequenceBuilder(sequence_length=100)
        X, y = sequence_builder.build_sequences(encoded_tweets)
        
        print("\nSplitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print("\nInitializing LSTM model...")
        lstm = LSTMModel(sequence_length=100, num_features=51)
        
        print("\nTraining model...")
        lstm.train(X_train, y_train, epochs=10, batch_size=64)
        
        print("\nFinal evaluation...")
        lstm.evaluate(X_test, y_test)
        plt.plot(np.array(lstm.training_history.history['accuracy'], label='training_acuracy'))
        plt.plot(np.array(lstm.training_history.history['val_accuracy']), label='validation accuracy')
        plt.legend()
        plt.savefig("fil.png")
if __name__ == "__main__":
    main()