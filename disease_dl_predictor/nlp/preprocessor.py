"""Text preprocessing helpers."""

import re

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


class TextPreprocessor:
    """Prepare raw text for deep learning model input."""

    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def remove_dosage(self, text: str) -> str:
        text = re.sub(r"\b\d+(\.\d+)?\s?(mg|ml|mcg|g|iu|units?)\b", " ", text, flags=re.I)
        text = re.sub(r"\b(twice daily|tds|bd|od|qid)\b", " ", text, flags=re.I)
        return re.sub(r"\s+", " ", text).strip()

    def text_to_sequence(self, text: str) -> np.ndarray:
        cleaned = self.remove_dosage(self.clean_text(text))
        seq = self.tokenizer.texts_to_sequences([cleaned])
        return pad_sequences(seq, maxlen=self.max_length, padding="post", truncating="post")
