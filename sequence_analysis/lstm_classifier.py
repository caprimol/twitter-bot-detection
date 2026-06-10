import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

class LSTMModel:
    def __init__(self, sequence_length: int, num_features: int):
        self.model = Sequential()
        self.model.add(LSTM(64, input_shape=(sequence_length, num_features)))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.training_history = None
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 50, batch_size: int = 32):
        print("\nArchitektura modelu (Sekwencyjny):")
        self.model.summary()
        
        # Matematyczne wyliczanie wag dla zbalansowania klas boty/ludzie
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, weights))
        print(f"[INFO] Zastosowane zbalansowane wagi klas: {class_weight_dict}")
        
        print("\nStarting LSTM training...")
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
            callbacks=[early_stop],
            class_weight=class_weight_dict
        )

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        print("Evaluating model...")
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {accuracy:.4f}")
        return accuracy


class LSTMFunctionalModel:
    def __init__(self, sequence_length: int, num_features: int):
        inputs = Input(shape=(sequence_length, num_features), name="Wejscie_Danych")
        x = LSTM(64, name="Warstwa_LSTM")(inputs)
        x = Dropout(0.2, name="Warstwa_Dropout")(x)
        x = Dense(32, activation='relu', name="Warstwa_Ukryta_Dense")(x)
        outputs = Dense(1, activation='sigmoid', name="Wyjscie_Klasyfikacji")(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name="Model_Funkcyjny_Twitter_Bot")
        self.training_history = None
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 50, batch_size: int = 32):
        print("\nArchitektura modelu (API Funkcyjne):")
        self.model.summary() 
        
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, weights))
        print(f"[INFO] Zastosowane zbalansowane wagi klas: {class_weight_dict}")
        
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
            callbacks=[early_stop],
            class_weight=class_weight_dict
        )

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        print("Evaluating model...")
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {accuracy:.4f}")
        return accuracy


class LSTMFunctionalWithEmbeddingModel:
    def __init__(self, sequence_length: int, num_features: int, embedding_dim: int = 40):
        inputs = Input(shape=(sequence_length, num_features), name="Wejscie_Danych")
        
        # Feature Embedding kompresujący szerokie wektory (np. z BERTa) przed podaniem do LSTM
        x = Dense(embedding_dim, activation='relu', name=f"Embedding_Cech_{embedding_dim}")(inputs)
        x = LSTM(64, name="Warstwa_LSTM")(x)
        x = Dropout(0.2, name="Warstwa_Dropout")(x)
        x = Dense(32, activation='relu', name="Warstwa_Ukryta_Dense")(x)
        outputs = Dense(1, activation='sigmoid', name="Wyjscie_Klasyfikacji")(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name="Model_Z_Embeddingiem")
        self.training_history = None
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 50, batch_size: int = 32):
        print(f"\nArchitektura modelu z Embeddingiem:")
        self.model.summary() 
        
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, weights))
        print(f"[INFO] Zastosowane zbalansowane wagi klas: {class_weight_dict}")
        
        print("\nStarting LSTM (with Embedding) training...")
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
            callbacks=[early_stop],
            class_weight=class_weight_dict
        )

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        print("Evaluating model...")
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {accuracy:.4f}")
        return accuracy


class LSTMDoubleFunctionalWithEmbeddingModel:
    def __init__(self, sequence_length: int, num_features: int, embedding_dim: int = 40):
        inputs = Input(shape=(sequence_length, num_features), name="Wejscie_Danych")
        
        x = Dense(embedding_dim, activation='relu', name=f"Embedding_Cech_{embedding_dim}")(inputs)
        
        # Pierwsza warstwa LSTM przekazuje sekwencję dalej dzięki return_sequences=True
        x = LSTM(64, return_sequences=True, name="Warstwa_LSTM_1")(x)
        x = LSTM(32, name="Warstwa_LSTM_2")(x)
        
        x = Dropout(0.2, name="Warstwa_Dropout")(x)
        x = Dense(32, activation='relu', name="Warstwa_Ukryta_Dense")(x)
        outputs = Dense(1, activation='sigmoid', name="Wyjscie_Klasyfikacji")(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name="Model_Podwojny_LSTM")
        self.training_history = None
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 50, batch_size: int = 32):
        print(f"\nArchitektura modelu 2xLSTM:")
        self.model.summary() 
        
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, weights))
        print(f"[INFO] Zastosowane zbalansowane wagi klas: {class_weight_dict}")
        
        print("\nStarting 2xLSTM training...")
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
            callbacks=[early_stop],
            class_weight=class_weight_dict
        )

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        print("Evaluating model...")
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {accuracy:.4f}")
        return accuracy