from dataclasses import dataclass
from typing import List, Tuple

try:
    import spacy
except ImportError:  # pragma: no cover - optional dependency
    spacy = None


@dataclass(frozen=True)
class NerResult:
    redacted_text: str
    types: List[str]


BALANCED_LABELS = {"PER", "PERSON", "ORG"}
STRICT_LABELS = {"PER", "PERSON", "ORG", "GPE", "LOC", "DATE", "TIME"}

LABEL_MAP = {
    "PER": "NAME",
    "PERSON": "NAME",
    "ORG": "ORG",
    "GPE": "LOC",
    "LOC": "LOC",
    "DATE": "DATE",
    "TIME": "TIME",
}


class PiiNer:
    def __init__(self, model: str = "fr_core_news_sm", mode: str = "balanced"):
        if spacy is None:
            raise RuntimeError("spaCy is not installed")
        if mode not in {"balanced", "strict"}:
            raise ValueError("mode must be 'balanced' or 'strict'")

        self.mode = mode
        self.labels = BALANCED_LABELS if mode == "balanced" else STRICT_LABELS
        # Keep only the NER pipeline to reduce overhead
        self.nlp = spacy.load(model, disable=["tagger", "parser", "lemmatizer", "attribute_ruler"])

    def redact(self, text: str) -> NerResult:
        doc = self.nlp(text)
        spans: List[Tuple[int, int, str]] = []

        for ent in doc.ents:
            if ent.label_ in self.labels:
                spans.append((ent.start_char, ent.end_char, ent.label_))

        if not spans:
            return NerResult(redacted_text=text, types=[])

        # Replace from the end to keep offsets stable
        redacted = text
        types: List[str] = []
        for start, end, label in sorted(spans, key=lambda s: s[0], reverse=True):
            mapped = LABEL_MAP.get(label, "PII")
            redacted = redacted[:start] + f"[{mapped}_REDACTED]" + redacted[end:]
            types.append(mapped)

        return NerResult(redacted_text=redacted, types=types)
