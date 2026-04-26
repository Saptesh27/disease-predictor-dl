# Disease DL Predictor

BiLSTM vs CNN disease prediction from symptom text with a Streamlit UI.

## Quick start

1. Create venv:
   - Windows PowerShell: `python -m venv .venv` then `.venv\Scripts\Activate.ps1`
2. Install deps: `pip install -r requirements.txt`
3. Setup folders + spaCy model: `python setup.py`
4. Put Kaggle `dataset.csv` into `data/dataset.csv`
5. Train: `python train.py`
6. Run app: `streamlit run app.py`
