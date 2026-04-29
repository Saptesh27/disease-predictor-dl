"""Dataset loading and preprocessing utilities."""

import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from config.settings import settings


class DataLoader:
    """Load and transform disease-symptom dataset for model training."""

    def load_raw_data(self) -> pd.DataFrame:
        df = pd.read_csv(settings.DATASET_PATH)
        print(f"Loaded data shape: {df.shape}")
        print(df.head())
        return df

    def prepare_features(self, df: pd.DataFrame) -> tuple[list[str], list[str]]:
        # Symptoms are spread over many columns, so we flatten each row into one text sample.
        symptom_cols = [c for c in df.columns if "Symptom" in c]
        X = (
            df[symptom_cols]
            .fillna("")
            .astype(str)
            .apply(lambda row: " ".join([v.strip().lower() for v in row if v.strip()]), axis=1)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .tolist()
        )
        y = df["Disease"].astype(str).tolist()
        return X, y

    def deduplicate_samples(self, X: list[str], y: list[str]) -> tuple[list[str], list[str]]:
        """Remove identical symptom-text + disease duplicates to prevent train/test leakage."""
        pairs = pd.DataFrame({"text": X, "label": y}).drop_duplicates(subset=["text", "label"])
        print(f"Deduplicated samples: {len(X)} -> {len(pairs)}")
        return pairs["text"].tolist(), pairs["label"].tolist()

    def encode_labels(self, y: list[str]) -> tuple[list[int], LabelEncoder, int]:
        # LabelEncoder maps disease names to integer ids and lets us decode predictions later.
        encoder = LabelEncoder()
        y_enc = encoder.fit_transform(y)
        with open(settings.LABEL_ENCODER_PATH, "wb") as f:
            pickle.dump(encoder, f)
        return y_enc, encoder, len(encoder.classes_)

    def tokenize_text(self, X: list[str]):
        # Tokenizer maps words to ids; padded sequences create fixed-size model inputs.
        tokenizer = Tokenizer(num_words=settings.MAX_VOCAB_SIZE, oov_token="<OOV>")
        tokenizer.fit_on_texts(X)
        sequences = tokenizer.texts_to_sequences(X)
        X_pad = pad_sequences(sequences, maxlen=settings.MAX_SEQUENCE_LENGTH, padding="post", truncating="post")
        with open(settings.TOKENIZER_PATH, "wb") as f:
            pickle.dump(tokenizer, f)
        return X_pad, tokenizer

    def train_test_split_data(self, X_pad, y_enc):
        n_samples = len(y_enc)
        n_classes = len(set(y_enc))
        # Stratified split requires at least one sample per class in each split.
        min_test_ratio = n_classes / n_samples
        test_ratio = max(settings.TEST_SPLIT, min_test_ratio)
        if test_ratio >= 0.5:
            test_ratio = 0.5
        if test_ratio != settings.TEST_SPLIT:
            print(
                f"Adjusted TEST_SPLIT from {settings.TEST_SPLIT:.2f} to {test_ratio:.3f} "
                f"for stratification across {n_classes} classes."
            )

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_pad,
            y_enc,
            test_size=test_ratio,
            random_state=settings.RANDOM_SEED,
            stratify=y_enc,
        )

        val_ratio = settings.VALIDATION_SPLIT / (1 - test_ratio)
        if val_ratio >= 0.5:
            val_ratio = 0.5

        # Validation split can fail if remaining class counts are too small; fallback to non-stratified split.
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val,
                y_train_val,
                test_size=val_ratio,
                random_state=settings.RANDOM_SEED,
                stratify=y_train_val,
            )
        except ValueError:
            print("Validation stratification not feasible with deduplicated data. Falling back to non-stratified split.")
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val,
                y_train_val,
                test_size=val_ratio,
                random_state=settings.RANDOM_SEED,
                stratify=None,
            )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def prepare_all(self) -> dict:
        df = self.load_raw_data()
        X, y = self.prepare_features(df)
        X, y = self.deduplicate_samples(X, y)
        y_enc, encoder, num_classes = self.encode_labels(y)
        X_pad, tokenizer = self.tokenize_text(X)
        X_train, X_val, X_test, y_train, y_val, y_test = self.train_test_split_data(X_pad, y_enc)
        print(f"Classes: {num_classes}, vocab: {len(tokenizer.word_index)}, train: {len(X_train)}, test: {len(X_test)}")
        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "tokenizer": tokenizer,
            "label_encoder": encoder,
            "num_classes": num_classes,
            "vocab_size": min(settings.MAX_VOCAB_SIZE, len(tokenizer.word_index) + 1),
        }
