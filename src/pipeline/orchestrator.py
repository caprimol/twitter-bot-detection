import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_processing.loader import CSVDataLoader
from src.data_processing.preprocessor import NLPPreprocessor
from src.features.behavior_dna import DigitalDNAExtractor
from src.features.text_patterns import TFIDFTextExtractor
from src.models.lstm_classifier import KerasLSTMClassifier

class MVPBotDetectionPipeline:
    def __init__(self, max_seq_len: int = 100):
        self.max_seq_len = max_seq_len
        self.loader = CSVDataLoader()
        self.preprocessor = NLPPreprocessor()
        self.dna_extractor = DigitalDNAExtractor(max_sequence_length=max_seq_len)
        self.text_extractor = TFIDFTextExtractor(max_features=200)
        self.classifier = None

    def run_pipeline(self, tweets_path: str, labels_path: str):
        raw_data = self.loader.load_data(tweets_path)
        labels_data = self.loader.load_data(labels_path)
        filtered_data = self.loader.preprocess_and_filter(raw_data, min_quantile=0.25)

        cleaned_text_data = self.preprocessor.clean_text(filtered_data)
        aggregated_texts = self.preprocessor.aggregate_texts(cleaned_text_data)
        text_features_matrix = self.text_extractor.extract_features(aggregated_texts, is_train=True)

        df_dna = self.dna_extractor.extract_dna_sequences(filtered_data)
        df_final = pd.merge(df_dna, labels_data, on='user_id', how='inner')

        X_dna, word_index = self.dna_extractor.tokenize_and_pad(df_final, is_train=True)
        y = df_final['label'].values
        vocab_size = len(word_index)

        X_train, X_test, y_train, y_test = train_test_split(X_dna, y, test_size=0.2, random_state=42)

        self.classifier = KerasLSTMClassifier(vocab_size=vocab_size, max_len=self.max_seq_len)
        self.classifier.train(X_train, y_train, epochs=5)

        predictions = self.classifier.predict(X_test)
        metrics = self.classifier.evaluate(predictions, y_test)

        for metric_name, value in metrics.items():
            print(f"{metric_name.upper()}: {value:.4f}")