import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class LSTMModel:
    def __init__(self, sequence_length: int, num_features: int):
        self.model = Sequential()
        self.model.add(LSTM(64, input_shape=(sequence_length, num_features)))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.training_history = []
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 5, batch_size: int = 32):
        print("Starting LSTM training...")
        training_history = self.model.fit(
            X_train, 
            y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=0.2
        )

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        print("Evaluating model...")
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {accuracy:.4f}")
        return accuracy