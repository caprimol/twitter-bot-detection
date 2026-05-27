from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import numpy as np

class KerasLSTMClassifier:
    def __init__(self, vocab_size: int, max_len: int):
        self.model = Sequential([
            Embedding(input_dim=vocab_size + 1, output_dim=16, input_length=max_len),
            LSTM(64, return_sequences=False),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print(self.model.summary())

    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 5, batch_size: int = 32):
        self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2
        )

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        probs = self.model.predict(X_test)
        return (probs > 0.5).astype(int).flatten()

    def evaluate(self, predictions: np.ndarray, true_labels: np.ndarray) -> dict:
        return {
            'accuracy': accuracy_score(true_labels, predictions),
            'f1_score': f1_score(true_labels, predictions),
            'mcc': matthews_corrcoef(true_labels, predictions)
        }