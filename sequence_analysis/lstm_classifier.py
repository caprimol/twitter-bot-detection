import numpy as np
from tensorflow.keras.models import Sequential, Model # Dodano Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input # Dodano Input
from tensorflow.keras.callbacks import EarlyStopping

class LSTMModel:
    def __init__(self, sequence_length: int, num_features: int):
        self.model = Sequential()
        self.model.add(LSTM(64, input_shape=(sequence_length, num_features)))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.training_history = None # Poprawione przypisanie
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    # Zmieniona domyślna liczba epok na 50
    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 50, batch_size: int = 32):
        print("\nArchitektura modelu:")
        self.model.summary() # WYŚWIETLENIE TABELKI DLA PROWADZĄCEGO
        
        print("\nStarting LSTM training...")
        
        # KONFIGURACJA EARLY STOPPING
        early_stop = EarlyStopping(
            monitor='val_loss', 
            patience=5, # Przerwie trening po 5 epokach braku poprawy
            restore_best_weights=True # Przywróci wagi z najlepszej epoki
        )

        # NAPRAWIONY BŁĄD (dodano self. i parametr callbacks)
        self.training_history = self.model.fit(
            X_train, 
            y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=0.2,
            callbacks=[early_stop] 
        )

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        print("Evaluating model...")
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {accuracy:.4f}")
        return accuracy

class LSTMFunctionalModel:
    def __init__(self, sequence_length: int, num_features: int):
        # 1. Zdefiniowanie warstwy wejściowej
        inputs = Input(shape=(sequence_length, num_features), name="Wejscie_Danych")
        
        # 2. Przepływ danych przez kolejne warstwy (jak w fabryce na taśmie produkcyjnej)
        x = LSTM(64, name="Warstwa_LSTM")(inputs)
        x = Dropout(0.2, name="Warstwa_Dropout")(x)
        x = Dense(32, activation='relu', name="Warstwa_Ukryta_Dense")(x)
        
        # 3. Warstwa wyjściowa
        outputs = Dense(1, activation='sigmoid', name="Wyjscie_Klasyfikacji")(x)
        
        # 4. Połączenie wszystkiego w jeden Model Funkcyjny
        self.model = Model(inputs=inputs, outputs=outputs, name="Model_Funkcyjny_Twitter_Bot")
        
        self.training_history = None
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    # Metody train i evaluate są identyczne jak w starym modelu!
    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 50, batch_size: int = 32):
        print("\nArchitektura modelu (API Funkcyjne):")
        self.model.summary() 
        
        print("\nStarting Functional LSTM training...")
        early_stop = EarlyStopping(
            monitor='val_loss', 
            patience=5, 
            restore_best_weights=True
        )

        self.training_history = self.model.fit(
            X_train, 
            y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=0.2,
            callbacks=[early_stop] 
        )

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        print("Evaluating model...")
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {accuracy:.4f}")
        return accuracy