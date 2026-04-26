"""BiLSTM text classification model."""

import io

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, BatchNormalization, Bidirectional, Dense, Dropout, Embedding, SpatialDropout1D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

from config.settings import settings


class BiLSTMModel:
    """BiLSTM architecture for disease classification."""

    def __init__(self, vocab_size: int, num_classes: int, embedding_dim: int, max_length: int):
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.model = None
        self.history = None

    def build(self):
        model = Sequential(
            [
                Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_length),
                SpatialDropout1D(0.2),
                Bidirectional(
                    LSTM(
                        units=settings.BILSTM_UNITS,
                        dropout=settings.BILSTM_DROPOUT,
                        recurrent_dropout=settings.BILSTM_RECURRENT_DROPOUT,
                        return_sequences=True,
                    )
                ),
                Bidirectional(LSTM(units=settings.BILSTM_UNITS // 2, dropout=settings.BILSTM_DROPOUT)),
                Dense(128, activation="relu"),
                BatchNormalization(),
                Dropout(0.4),
                Dense(self.num_classes, activation="softmax"),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=settings.LEARNING_RATE),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        self.model = model
        return model

    def get_callbacks(self):
        return [
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
            ModelCheckpoint(filepath=str(settings.BILSTM_MODEL_PATH), save_best_only=True),
        ]

    def train(self, X_train, y_train, X_val, y_val):
        if self.model is None:
            self.build()
        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=settings.EPOCHS,
            batch_size=settings.BATCH_SIZE,
            callbacks=self.get_callbacks(),
            verbose=1,
        )
        return self.history.history

    def predict(self, X_input) -> np.ndarray:
        return self.model.predict(X_input, verbose=0)

    def get_model_summary(self) -> str:
        if self.model is None:
            self.build()
        stream = io.StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + "\n"))
        return stream.getvalue()

    def save(self):
        self.model.save(settings.BILSTM_MODEL_PATH)

    def load(self):
        self.model = load_model(settings.BILSTM_MODEL_PATH)
        return self.model
