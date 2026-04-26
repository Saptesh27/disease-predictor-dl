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

    print("\n" + "=" * 50)
    print("MODEL COMPARISON RESULTS")
    print("=" * 50)
    print(f"{'Metric':<15} {'BiLSTM':>10} {'CNN':>10} {'Winner':>10}")
    print("-" * 50)
    for metric in ["accuracy", "f1"]:
        b = results["bilstm"][metric]
        c = results["cnn"][metric]
        winner = "BiLSTM" if b >= c else "CNN"
        label = "F1-Score" if metric == "f1" else "Accuracy"
        print(f"{label:<15} {b:>9.2%} {c:>9.2%} {winner:>10}")

    history_payload = {
        "bilstm_history": bilstm_history,
        "cnn_history": cnn_history,
        "metrics": results,
        "training_time": {"bilstm_seconds": bilstm_time, "cnn_seconds": cnn_time},
    }
    with open(settings.TRAINING_HISTORY_PATH, "wb") as f:
        pickle.dump(history_payload, f)

    print("\nTraining complete!")
    print("Run the app with: streamlit run app.py")


if __name__ == "__main__":
    main()
