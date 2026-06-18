import os
import gc
import datetime
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from data_processing.data_loader import TwitterDatasetLoader
from data_processing.preprocessing import DataPreprocessor
from content_analysis.bert_encoder import BertEncoder
from data_processing.sequence_builder import SequenceBuilder
from sequence_analysis.lstm_classifier import LSTMFunctionalWithEmbeddingModel, LSTMDoubleFunctionalWithEmbeddingModel

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def run_baseline(encoded_tweets):
    print("\n" + "="*50)
    print("EKSPERYMENT: Rozwiązanie Bazowe (Naiwny Bayes)")
    print("="*50)
    sb = SequenceBuilder(sequence_length=100, feature_mode='both')
    X, y = sb.build_sequences(encoded_tweets)
    
    liczba_probek, czas, cechy = X.shape
    X_flat = X.reshape((liczba_probek, czas * cechy))
    X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)
    
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Skuteczność (Baseline Bayes): {acc:.4f}")
    
    del sb, X, y, X_flat, X_train, X_test, y_train, y_test, model
    gc.collect()
    return acc

def run_dl_experiment(encoded_tweets, exp_name, feature_mode, model_class):
    print("\n" + "="*50)
    print(f"EKSPERYMENT: {exp_name}")
    print("="*50)
    
    sb = SequenceBuilder(sequence_length=100, feature_mode=feature_mode)
    X, y = sb.build_sequences(encoded_tweets)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    feature_dim = X_train.shape[2]
    
    model = model_class(sequence_length=100, num_features=feature_dim, embedding_dim=40)
    
    model.train(X_train, y_train, epochs=25, batch_size=64)
    acc = model.evaluate(X_test, y_test)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(model.training_history.history['accuracy'], label='Trening')
    plt.plot(model.training_history.history['val_accuracy'], label='Walidacja')
    plt.title(f'Skuteczność - {exp_name}')
    plt.xlabel('Epoka')
    plt.ylabel('Skuteczność')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(model.training_history.history['loss'], label='Trening')
    plt.plot(model.training_history.history['val_loss'], label='Walidacja')
    plt.title(f'Strata - {exp_name}')
    plt.xlabel('Epoka')
    plt.ylabel('Strata')
    plt.legend()
    plt.tight_layout()
    
    os.makedirs('charts', exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_name = exp_name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
    filepath = os.path.join('charts', f"wykres_{safe_name}_{timestamp}.png")
    plt.savefig(filepath)
    plt.close()
    
    print(f"Wykres zapisany w: {filepath}")
    
    del sb, X, y, X_train, X_test, y_train, y_test, model
    gc.collect()
    return acc


def main():
    base_directory = "datasets_full.csv"
    print("Wczytywanie i preprocessowanie danych...")
    loader = TwitterDatasetLoader(base_path=base_directory)
    users_dataframe, tweets_dataframe = loader.load_data()
    
    if tweets_dataframe.empty:
        print("Brak danych!")
        return
        
    preprocessor = DataPreprocessor(sequence_length=100)
    processed_tweets = preprocessor.preprocess_tweets(tweets_dataframe)
    
    checkpoint_file = "bert_encoded_tweets.pkl"
    if os.path.exists(checkpoint_file):
        print(f"Wczytuję gotowy BERT z pliku {checkpoint_file}...")
        encoded_tweets = pd.read_pickle(checkpoint_file)
    else:
        print("Brak pliku checkpoint. Generuję wektory BERT...")
        encoder = BertEncoder()
        encoded_tweets = encoder.encode_tweets(processed_tweets)
        encoded_tweets.to_pickle(checkpoint_file)
        
    wyniki = {}
    
    wyniki['Naiwny Bayes (Baseline)'] = run_baseline(encoded_tweets)
    
    wyniki['Tylko Tekst (1xLSTM)'] = run_dl_experiment(
        encoded_tweets, "Tylko Tekst", 'text_only', LSTMFunctionalWithEmbeddingModel)
        
    wyniki['Tylko Czas (1xLSTM)'] = run_dl_experiment(
        encoded_tweets, "Tylko Czas", 'time_only', LSTMFunctionalWithEmbeddingModel)

    wyniki['Tylko DNA [ACT] (1xLSTM)'] = run_dl_experiment(
        encoded_tweets, "Tylko DNA", 'dna_only', LSTMFunctionalWithEmbeddingModel)
        
    wyniki['Czas + Tekst (1xLSTM)'] = run_dl_experiment(
        encoded_tweets, "Czas i Tekst 1xLSTM", 'both', LSTMFunctionalWithEmbeddingModel)

    wyniki['Czas + Tekst + DNA (1xLSTM)'] = run_dl_experiment(
        encoded_tweets, "Pelne Cechy z DNA 1xLSTM", 'both_with_dna', LSTMFunctionalWithEmbeddingModel)
        
    wyniki['Czas + Tekst + DNA (2xLSTM)'] = run_dl_experiment(
        encoded_tweets, "Pelne Cechy z DNA 2xLSTM", 'both_with_dna', LSTMDoubleFunctionalWithEmbeddingModel)

    print("\n\n" + "#"*60)
    print("PODSUMOWANIE WYNIKÓW (ABLATION STUDIES)")
    print("#"*60)
    for nazwa, acc in wyniki.items():
        print(f"{nazwa.ljust(35)}: {acc*100:.2f}%")
        
if __name__ == "__main__":
    main()