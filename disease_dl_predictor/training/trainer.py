"""Training orchestration for both DL models."""

from config.settings import settings
from models.bilstm_model import BiLSTMModel
from models.cnn_model import CNNModel
from training.data_loader import DataLoader
from training.evaluator import Evaluator


class Trainer:
    """Prepare data, train models, and evaluate."""

    def __init__(self):
        self.loader = DataLoader()
        self.data = None
        self.bilstm = None
        self.cnn = None

    def prepare_data(self):
        self.data = self.loader.prepare_all()
        return self.data

    def build_models(self):
        self.bilstm = BiLSTMModel(
            vocab_size=self.data["vocab_size"],
            num_classes=self.data["num_classes"],
            embedding_dim=settings.EMBEDDING_DIM,
            max_length=settings.MAX_SEQUENCE_LENGTH,
        )
        self.cnn = CNNModel(
            vocab_size=self.data["vocab_size"],
            num_classes=self.data["num_classes"],
            embedding_dim=settings.EMBEDDING_DIM,
            max_length=settings.MAX_SEQUENCE_LENGTH,
        )
        self.bilstm.build()
        self.cnn.build()
        print(self.bilstm.get_model_summary())
        self.cnn.model.summary()

    def train_bilstm(self):
        hist = self.bilstm.train(
            self.data["X_train"], self.data["y_train"], self.data["X_val"], self.data["y_val"]
        )
        self.bilstm.save()
        return hist

    def train_cnn(self):
        hist = self.cnn.train(self.data["X_train"], self.data["y_train"], self.data["X_val"], self.data["y_val"])
        self.cnn.save()
        return hist

    def train_both(self):
        self.prepare_data()
        self.build_models()
        print("Training BiLSTM...")
        bilstm_history = self.train_bilstm()
        print("Training CNN...")
        cnn_history = self.train_cnn()
        evalr = Evaluator(self.bilstm, self.cnn, self.data)
        comparison = evalr.compare_both(self.data["X_test"], self.data["y_test"])
        comparison["bilstm_history"] = bilstm_history
        comparison["cnn_history"] = cnn_history
        return comparison
