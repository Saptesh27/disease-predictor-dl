"""Model evaluation utilities."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class Evaluator:
    """Evaluate and compare two trained models."""

    def __init__(self, bilstm_model, cnn_model, data):
        self.bilstm_model = bilstm_model
        self.cnn_model = cnn_model
        self.data = data

    def compare_both(self, X_test, y_test) -> dict:
        bilstm_probs = self.bilstm_model.model.predict(X_test, verbose=0)
        bilstm_pred = np.argmax(bilstm_probs, axis=1)
        cnn_probs = self.cnn_model.model.predict(X_test, verbose=0)
        cnn_pred = np.argmax(cnn_probs, axis=1)

        bilstm_metrics = {
            "accuracy": round(accuracy_score(y_test, bilstm_pred) * 100, 2),
            "precision": round(precision_score(y_test, bilstm_pred, average="weighted", zero_division=0) * 100, 2),
            "recall": round(recall_score(y_test, bilstm_pred, average="weighted", zero_division=0) * 100, 2),
            "f1": round(f1_score(y_test, bilstm_pred, average="weighted", zero_division=0) * 100, 2),
            "confusion_matrix": confusion_matrix(y_test, bilstm_pred).tolist(),
            "predictions": bilstm_pred.tolist(),
            "probabilities": bilstm_probs.tolist(),
        }
        cnn_metrics = {
            "accuracy": round(accuracy_score(y_test, cnn_pred) * 100, 2),
            "precision": round(precision_score(y_test, cnn_pred, average="weighted", zero_division=0) * 100, 2),
            "recall": round(recall_score(y_test, cnn_pred, average="weighted", zero_division=0) * 100, 2),
            "f1": round(f1_score(y_test, cnn_pred, average="weighted", zero_division=0) * 100, 2),
            "confusion_matrix": confusion_matrix(y_test, cnn_pred).tolist(),
            "predictions": cnn_pred.tolist(),
            "probabilities": cnn_probs.tolist(),
        }

        bilstm_report = classification_report(y_test, bilstm_pred, output_dict=True, zero_division=0)
        cnn_report = classification_report(y_test, cnn_pred, output_dict=True, zero_division=0)
        winner = "BiLSTM" if bilstm_metrics["f1"] > cnn_metrics["f1"] else "CNN"

        return {
            "bilstm": bilstm_metrics,
            "cnn": cnn_metrics,
            "bilstm_report": bilstm_report,
            "cnn_report": cnn_report,
            "y_test": np.asarray(y_test).tolist(),
            "winner": winner,
            "bilstm_history": self.bilstm_model.history.history if self.bilstm_model.history is not None else {},
            "cnn_history": self.cnn_model.history.history if self.cnn_model.history is not None else {},
        }

    @staticmethod
    def get_per_class_df(report: dict, label_encoder) -> pd.DataFrame:
        rows = []
        for label_key, metrics in report.items():
            if label_key in {"accuracy", "macro avg", "weighted avg"}:
                continue
            try:
                label_idx = int(float(label_key))
                disease_name = label_encoder.inverse_transform([label_idx])[0]
            except (ValueError, TypeError):
                disease_name = str(label_key)
            rows.append(
                {
                    "disease": disease_name,
                    "precision": float(metrics.get("precision", 0.0)) * 100,
                    "recall": float(metrics.get("recall", 0.0)) * 100,
                    "f1_score": float(metrics.get("f1-score", 0.0)) * 100,
                    "support": int(metrics.get("support", 0)),
                }
            )

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.round(2).sort_values(by="f1_score", ascending=False).reset_index(drop=True)

    @staticmethod
    def get_top_confused_diseases(cm: list, label_encoder, top_n=10) -> list:
        matrix = np.asarray(cm)
        confused_pairs = []
        for actual_idx in range(matrix.shape[0]):
            for pred_idx in range(matrix.shape[1]):
                if actual_idx == pred_idx:
                    continue
                count = int(matrix[actual_idx, pred_idx])
                if count <= 0:
                    continue
                confused_pairs.append(
                    {
                        "actual": label_encoder.inverse_transform([actual_idx])[0],
                        "predicted": label_encoder.inverse_transform([pred_idx])[0],
                        "count": count,
                    }
                )
        return sorted(confused_pairs, key=lambda x: x["count"], reverse=True)[:top_n]
