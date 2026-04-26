"""Drug extraction and disease mapping helpers."""

import pandas as pd

from config.settings import settings


class DrugExtractor:
    """Map mentioned drugs to disease hints."""

    def __init__(self):
        self.df = None
        if settings.DRUGS_CSV_PATH.exists():
            self.df = pd.read_csv(settings.DRUGS_CSV_PATH)

    def extract_drug_tokens(self, text: str) -> list[str]:
        return [t.strip().lower() for t in text.split() if t.strip()]

    def map_drugs_to_text(self, text: str) -> str:
        if self.df is None:
            return text.lower()
        tokens = set(self.extract_drug_tokens(text))
        cols = {c.lower(): c for c in self.df.columns}
        if "drug" not in cols or "disease" not in cols:
            return text.lower()
        matches = self.df[self.df[cols["drug"]].astype(str).str.lower().isin(tokens)]
        return " ".join(matches[cols["disease"]].astype(str).str.lower().tolist()) or text.lower()
