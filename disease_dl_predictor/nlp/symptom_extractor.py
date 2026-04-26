"""Symptom extraction using spaCy phrase matching."""

import spacy
from spacy.matcher import PhraseMatcher


class SymptomExtractor:
    """Extract non-negated symptoms from free text."""

    NEGATIONS = ["no", "not", "without", "denies", "absence of", "no history of"]

    def __init__(self, symptoms_list: list[str]):
        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.neg_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.matcher.add("SYMPTOMS", [self.nlp.make_doc(s) for s in set(symptoms_list) if s])
        self.neg_matcher.add("NEG", [self.nlp.make_doc(n) for n in self.NEGATIONS])

    def extract_symptoms(self, text: str) -> list[str]:
        doc = self.nlp(text)
        symptoms = []
        matches = self.matcher(doc)
        for _, start, end in matches:
            window_start = max(0, start - 3)
            window_text = doc[window_start:start].text.lower()
            if any(neg in window_text for neg in self.NEGATIONS):
                continue
            symptoms.append(doc[start:end].text.lower())
        return sorted(set(symptoms))

    def symptoms_to_text(self, symptoms: list[str]) -> str:
        return " ".join(symptoms).strip()
