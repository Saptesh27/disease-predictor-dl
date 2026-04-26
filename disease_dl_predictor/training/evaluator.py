"""Model evaluation utilities."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score


class Evaluator:
    """Evaluate and compare two trained models."""

    def __init__(self, bilstm_model, cnn_model, data):
        self.bilstm_model = bilstm_model
        self.cnn_model = cnn_model
        self.data = data

    def evaluate_model(self, model, X_test, y_test) -> dict:
        probs = model.predict(X_test, verbose=0)
        y_pred = np.argmax(probs, axis=1)
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }

    def compare_both(self, X_test, y_test) -> dict:
        bilstm = self.evaluate_model(self.bilstm_model.model, X_test, y_test)
        cnn = self.evaluate_model(self.cnn_model.model, X_test, y_test)
        winner = "BiLSTM" if bilstm["f1"] >= cnn["f1"] else "CNN"
        return {"bilstm": bilstm, "cnn": cnn, "winner": winner}

    def get_per_class_report(self, model, X_test, y_test, label_encoder) -> pd.DataFrame:
        probs = model.predict(X_test, verbose=0)
        y_pred = np.argmax(probs, axis=1)
        report = classification_report(
            y_test, y_pred, target_names=label_encoder.classes_, output_dict=True, zero_division=0
        )
        return pd.DataFrame(report).T.sort_values(by="f1-score", ascending=False)
