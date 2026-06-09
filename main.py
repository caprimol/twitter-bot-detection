from data_processing.data_loader import TwitterDatasetLoader
from data_processing.preprocessing import DataPreprocessor
from content_analysis.text_encoder import TextEncoder
from data_processing.sequence_builder import SequenceBuilder
from sequence_analysis.lstm_classifier import LSTMModel
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
import datetime
import os

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
        # Zmieniamy liczbę epok na 50 (Early Stopping przerwie szybciej jeśli trzeba)
        lstm.train(X_train, y_train, epochs=50, batch_size=64) 
        
        print("\nFinal evaluation...")
        lstm.evaluate(X_test, y_test)
        
        # --- NOWY KOD GENERUJĄCY WYKRESY I TIMESTAMP ---
        print("\nGenerowanie wykresów...")
        plt.figure(figsize=(12, 5)) # Większe okno na 2 wykresy

        # Wykres 1: Skuteczność (Accuracy)
        plt.subplot(1, 2, 1)
        plt.plot(lstm.training_history.history['accuracy'], label='Trening')
        plt.plot(lstm.training_history.history['val_accuracy'], label='Walidacja')
        plt.title('Skuteczność modelu (Accuracy)')
        plt.xlabel('Epoka')
        plt.ylabel('Skuteczność')
        plt.legend()

        # Wykres 2: Funkcja straty (Loss)
        plt.subplot(1, 2, 2)
        plt.plot(lstm.training_history.history['loss'], label='Trening')
        plt.plot(lstm.training_history.history['val_loss'], label='Walidacja')
        plt.title('Funkcja straty (Loss)')
        plt.xlabel('Epoka')
        plt.ylabel('Strata')
        plt.legend()

        plt.tight_layout()
        
        # --- ZMIANY DLA FOLDERU CHARTS ---
        # Upewniamy się, że folder 'charts' istnieje (jeśli nie, Python go stworzy)
        os.makedirs('charts', exist_ok=True)
        
        # Generujemy nazwę pliku z datą i godziną
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"wykres_treningu_{timestamp}.png"
        
        # Łączymy folder 'charts' z nazwą pliku
        filepath = os.path.join('charts', filename)
        
        # Zapisujemy pod nową ścieżką
        plt.savefig(filepath)
        print(f"Wykresy zapisane pomyślnie jako: {filepath}\n")

if __name__ == "__main__":
    main()