"""CNN text classification model."""

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    Embedding,
    GlobalMaxPooling1D,
    Input,
    SpatialDropout1D,
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

from config.settings import settings


class CNNModel:
    """Parallel Conv1D text classifier."""

    def __init__(self, vocab_size: int, num_classes: int, embedding_dim: int, max_length: int):
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.model = None
        self.history = None

    def build(self):
        inputs = Input(shape=(self.max_length,))
        x = Embedding(self.vocab_size, self.embedding_dim)(inputs)
        x = SpatialDropout1D(0.2)(x)

        branches = []
        for k in settings.CNN_KERNEL_SIZES:
            b = Conv1D(filters=settings.CNN_FILTERS, kernel_size=k, activation="relu", padding="same")(x)
            b = GlobalMaxPooling1D()(b)
            branches.append(b)

        merged = Concatenate()(branches)
        x = Dense(256, activation="relu")(merged)
        x = BatchNormalization()(x)
        x = Dropout(settings.CNN_DROPOUT)(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.num_classes, activation="softmax")(x)

        model = Model(inputs=inputs, outputs=outputs)
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
            ModelCheckpoint(filepath=str(settings.CNN_MODEL_PATH), save_best_only=True),
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

    def save(self):
        self.model.save(settings.CNN_MODEL_PATH)

    def load(self):
        self.model = load_model(settings.CNN_MODEL_PATH)
        return self.model
