"""Project settings for disease predictor."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Application configuration values."""

    BASE_DIR: Path = Path(__file__).resolve().parents[1]

    # Dataset
    DATASET_PATH: Path = BASE_DIR / "data" / "dataset.csv"
    DRUGS_CSV_PATH: Path = BASE_DIR / "data" / "drugs.csv"
    TOKENIZER_PATH: Path = BASE_DIR / "data" / "tokenizer.pkl"
    LABEL_ENCODER_PATH: Path = BASE_DIR / "data" / "label_encoder.pkl"
    TRAINING_HISTORY_PATH: Path = BASE_DIR / "data" / "training_history.pkl"

    # Text Processing
    MAX_SEQUENCE_LENGTH: int = 100
    MAX_VOCAB_SIZE: int = 5000
    EMBEDDING_DIM: int = 128

    # BiLSTM Hyperparameters
    BILSTM_UNITS: int = 64
    BILSTM_DROPOUT: float = 0.3
    BILSTM_RECURRENT_DROPOUT: float = 0.3

    # CNN Hyperparameters
    CNN_FILTERS: int = 128
    CNN_KERNEL_SIZES: list[int] = field(default_factory=lambda: [2, 3, 4])
    CNN_DROPOUT: float = 0.5

    # Training
    EPOCHS: int = 15
    BATCH_SIZE: int = 32
    VALIDATION_SPLIT: float = 0.2
    TEST_SPLIT: float = 0.1
    LEARNING_RATE: float = 0.001
    RANDOM_SEED: int = 42

    # Model Save Paths
    BILSTM_MODEL_PATH: Path = BASE_DIR / "saved_models" / "bilstm" / "model.h5"
    CNN_MODEL_PATH: Path = BASE_DIR / "saved_models" / "cnn" / "model.h5"

    # App
    TOP_N_DISEASES: int = 5
    CONFIDENCE_THRESHOLD: float = 0.1


settings = Settings()
