"""Environment setup helper script."""

import subprocess
import sys
from pathlib import Path


def main():
    base_dir = Path(__file__).resolve().parent
    for rel in ["saved_models/bilstm", "saved_models/cnn", "data"]:
        (base_dir / rel).mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    except Exception as exc:  # noqa: BLE001
        print(f"spaCy model download skipped/failed: {exc}")
    if not (base_dir / "data" / "dataset.csv").exists():
        print("Please download dataset.csv from:")
        print("https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset")
        print("Place it in the data/ folder")
    print("Setup complete.")


if __name__ == "__main__":
    main()
