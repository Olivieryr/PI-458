import json
from dataclasses import dataclass
from typing import Any, Dict, List

from core.llm_client import ChatMessage, OllamaClient


INTENTS: List[str] = [
    "REPORTS_PROBLEM",
    "PROVIDES_CONTEXT",
    "COMPLIES",
    "COMPLIES_WITH_RISK",
    "EXPRESSES_CONCERN",
    "ASKS_WHAT_TO_DO",
    "REFUSES_TO_SHARE",
    "SUSPICIOUS",
    "OTHER",
]


@dataclass(frozen=True)
class TransitionResult:
    intent: str
    confidence: float
    slots: Dict[str, Any]


class TransitionEngine:
    def __init__(self, client: OllamaClient):
        self.client = client

    def classify(self, text: str) -> TransitionResult:
        """
        Classify redacted user input into a fixed intent + slots.
        Must return JSON only. If parsing fails, fallback to OTHER.
        """
        system = (
            "You are a classification engine. Output JSON only, no extra text. "
            "Return: {\"intent\": <ENUM>, \"confidence\": <0-1>, \"slots\": {...}}. "
            f"Valid intents: {', '.join(INTENTS)}. "
            "Slots (optional): target_service (it/hr/finance/other), "
            "target_device (laptop/desktop/phone), urgency_level (low/medium/high), "
            "target_compliance_level (low/medium/high), collected_info (list of labels)."
        )
        messages = [
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=text),
        ]

        raw = self.client.chat(messages=messages, temperature=0.2, top_p=0.9, num_predict=120)
        parsed = self._parse_json(raw)
        return self._normalize(parsed)

    def _parse_json(self, raw: str) -> Dict[str, Any]:
        try:
            return json.loads(raw)
        except Exception:
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return {}
            try:
                return json.loads(raw[start : end + 1])
            except Exception:
                return {}

    def _normalize(self, data: Dict[str, Any]) -> TransitionResult:
        intent = str(data.get("intent", "OTHER")).upper()
        if intent not in INTENTS:
            intent = "OTHER"
        try:
            confidence = float(data.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        slots = data.get("slots", {})
        if not isinstance(slots, dict):
            slots = {}
        return TransitionResult(intent=intent, confidence=confidence, slots=slots)
