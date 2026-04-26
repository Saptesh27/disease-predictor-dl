"""Unified model loader and predictor."""

import pickle

import numpy as np
import pandas as pd

from config.settings import settings
from nlp.drug_extractor import DrugExtractor
from nlp.preprocessor import TextPreprocessor
from nlp.symptom_extractor import SymptomExtractor
from tensorflow.keras.models import load_model


class ModelManager:
    """Load artifacts and run prediction through both models."""

    def __init__(self):
        self.bilstm = None
        self.cnn = None
        self.tokenizer = None
        self.label_encoder = None
        self.preprocessor = None
        self.symptom_extractor = None
        self.drug_extractor = DrugExtractor()
        self.is_loaded = False

    def load_all(self):
        with open(settings.TOKENIZER_PATH, "rb") as f:
            self.tokenizer = pickle.load(f)
        with open(settings.LABEL_ENCODER_PATH, "rb") as f:
            self.label_encoder = pickle.load(f)
        self.preprocessor = TextPreprocessor(self.tokenizer, settings.MAX_SEQUENCE_LENGTH)
        self.bilstm = load_model(settings.BILSTM_MODEL_PATH)
        self.cnn = load_model(settings.CNN_MODEL_PATH)
        symptoms_list = []
        if settings.DATASET_PATH.exists():
            df = pd.read_csv(settings.DATASET_PATH)
            symptom_cols = [c for c in df.columns if "Symptom" in c]
            for c in symptom_cols:
                symptoms_list.extend(df[c].dropna().astype(str).str.lower().tolist())
        self.symptom_extractor = SymptomExtractor(symptoms_list)
        self.is_loaded = True

    def _top_n(self, probs):
        top_indices = np.argsort(probs)[::-1][: settings.TOP_N_DISEASES]
        return [
            {
                "disease": self.label_encoder.inverse_transform([idx])[0],
                "probability": float(probs[idx]),
                "percentage": round(float(probs[idx]) * 100, 2),
            }
            for idx in top_indices
        ]

    def predict_both(self, input_text: str, input_mode: str = "symptoms") -> dict:
        if not self.is_loaded:
            self.load_all()
        symptoms = []
        if input_mode == "prescription":
            clean_text = self.drug_extractor.map_drugs_to_text(input_text)
        else:
            symptoms = self.symptom_extractor.extract_symptoms(input_text)
            clean_text = self.symptom_extractor.symptoms_to_text(symptoms) or input_text.lower()

        X_input = self.preprocessor.text_to_sequence(clean_text)
        bilstm_probs = self.bilstm.predict(X_input, verbose=0)[0]
        cnn_probs = self.cnn.predict(X_input, verbose=0)[0]
        bilstm_results = self._top_n(bilstm_probs)
        cnn_results = self._top_n(cnn_probs)
        winner = "BiLSTM" if bilstm_probs.max() > cnn_probs.max() else "CNN"
        return {
            "input_text": clean_text,
            "extracted_symptoms": symptoms,
            "bilstm": {"predictions": bilstm_results, "top_disease": bilstm_results[0]["disease"], "top_confidence": bilstm_results[0]["percentage"], "model_name": "BiLSTM"},
            "cnn": {"predictions": cnn_results, "top_disease": cnn_results[0]["disease"], "top_confidence": cnn_results[0]["percentage"], "model_name": "CNN"},
            "agreement": bilstm_results[0]["disease"] == cnn_results[0]["disease"],
            "winner": winner,
            "mode": input_mode,
        }
