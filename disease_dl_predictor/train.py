"""Train both models and save artifacts."""

import pickle
import time

from config.settings import settings
from training.evaluator import Evaluator
from training.trainer import Trainer


def main():
    print("=" * 60)
    print("  Disease Predictor - Model Training")
    print("  Training BiLSTM + CNN on Kaggle Dataset")
    print("=" * 60)

    trainer = Trainer()
    trainer.prepare_data()
    trainer.build_models()

    start = time.time()
    bilstm_history = trainer.train_bilstm()
    bilstm_time = time.time() - start
    print(f"BiLSTM training time: {bilstm_time:.1f} seconds")

    start = time.time()
    cnn_history = trainer.train_cnn()
    cnn_time = time.time() - start
    print(f"CNN training time: {cnn_time:.1f} seconds")

    evaluator = Evaluator(trainer.bilstm, trainer.cnn, trainer.data)
    results = evaluator.compare_both(trainer.data["X_test"], trainer.data["y_test"])

    with open(settings.EVAL_RESULTS_PATH, "wb") as f:
        pickle.dump(results, f)
    print(f"Evaluation results saved to: {settings.EVAL_RESULTS_PATH}")

    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<20} {'BiLSTM':>12} {'CNN':>12} {'Winner':>10}")
    print("-" * 60)
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    for metric, label in zip(metrics, labels):
        b = results["bilstm"][metric]
        c = results["cnn"][metric]
        winner = "BiLSTM" if b >= c else "CNN"
        marker = "*"
        print(f"{label:<20} {b:>10.2f}% {c:>10.2f}% {marker} {winner:>7}")
    print("=" * 60)
    print(f"Overall Winner: {results['winner']}")
    print("=" * 60)

    history_payload = {"bilstm": bilstm_history, "cnn": cnn_history}
    with open(settings.TRAINING_HISTORY_PATH, "wb") as f:
        pickle.dump(history_payload, f)

    print("\nTraining complete!")
    print("Run the app with: streamlit run app.py")


if __name__ == "__main__":
    main()
