from data_processing.data_loader import TwitterDatasetLoader
from data_processing.preprocessing import DataPreprocessor
from content_analysis.text_encoder import TextEncoder
from data_processing.sequence_builder import SequenceBuilder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("--- URUCHAMIANIE MODELU BAZOWEGO (NAIWNY BAYES) ---")
    base_directory = "datasets_full.csv"
    
    loader = TwitterDatasetLoader(base_path=base_directory)
    users_dataframe, tweets_dataframe = loader.load_data()
    
    if tweets_dataframe.empty:
        print("Brak danych do analizy.")
        return

    # Ten sam proces przygotowania danych co zawsze
    preprocessor = DataPreprocessor(sequence_length=100)
    processed_tweets = preprocessor.preprocess_tweets(tweets_dataframe)
    
    encoder = TextEncoder(max_features=50)
    encoded_tweets = encoder.encode_tweets(processed_tweets)
    
    sequence_builder = SequenceBuilder(sequence_length=100)
    X, y = sequence_builder.build_sequences(encoded_tweets)
    
    print("\nRozpłaszczanie danych trójwymiarowych dla klasycznego algorytmu...")
    # Zmiana kształtu z (liczba_użytkowników, 100, 51) na (liczba_użytkowników, 5100)
    liczba_probek, czas, cechy = X.shape
    X_flat = X.reshape((liczba_probek, czas * cechy))
    
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_flat, y, test_size=0.2, random_state=42
    )
    
    print("\nTrenowanie modelu Naiwnego Bayesa (GaussianNB)...")
    bayes_model = GaussianNB()
    bayes_model.fit(X_train, y_train)
    
    print("\nEwaluacja modelu bazowego:")
    y_pred = bayes_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test Accuracy (Baseline): {accuracy:.4f}")
    print("\nSzczegółowy raport klasyfikacji:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()