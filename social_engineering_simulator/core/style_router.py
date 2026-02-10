import re
from dataclasses import dataclass
from collections import deque
from typing import Deque


@dataclass(frozen=True)
class StyleProfile:
    style: str          # "formal" | "casual" | "neutral" | "tense"
    address: str        # "vous" | "tu"
    verbosity: str      # "short" | "medium"


FORMAL_HINTS = [
    "bonjour", "monsieur", "madame", "cordialement", "veuillez",
    "pourriez-vous", "serait-il possible", "je vous prie", "merci de",
]
POLITE_MARKERS = [
    "svp", "s'il vous plait", "merci", "je vous remercie",
]
CASUAL_HINTS = [
    "salut", "yo", "wesh", "stp", "s'il te plait", "mdr", "ptdr",
    "tkt", "ok", "bg", "frero", "pote", "c'est relou", "grave",
]
TENSE_HINTS = [
    "c'est n'importe quoi", "ca marche pas", "vous etes serieux",
    "arnaque", "scam", "c'est une blague", "j'en ai marre",
    "relou", "inadmissible", "honteux",
]
URGENCY_HINTS = [
    "urgent", "tres urgent", "immediat", "tout de suite",
    "depeche", "vite", "asap", "maintenant",
]

# Gross detection of insults / aggressiveness (V1)
AGGRESSIVE_RE = re.compile(r"(?i)\b(connard|idiot|nul|merde|putain|chiant)\b")


def detect_style(text: str) -> StyleProfile:
    t = text.strip()
    tl = t.lower()

    score = 0

    # Tu / Vous explicit
    if re.search(r"(?i)\b(vous|votre|vos)\b", t):
        score += 1
    if re.search(r"(?i)\b(tu|ton|ta|tes)\b", t):
        score -= 1

    # Lexical hints
    if any(k in tl for k in FORMAL_HINTS):
        score += 2
    if any(k in tl for k in POLITE_MARKERS):
        score += 1
    if any(k in tl for k in CASUAL_HINTS):
        score -= 2

    # Tension + urgency + excessive punctuation
    excessive_punct = bool(re.search(r"[!?]{2,}", t))
    urgency = any(k in tl for k in URGENCY_HINTS)
    tense = (
        any(k in tl for k in TENSE_HINTS)
        or bool(AGGRESSIVE_RE.search(t))
        or excessive_punct
        or urgency
    )

    # Length / verbosity (simple)
    verbosity = "short" if len(t) < 45 else "medium"

    if tense:
        style = "tense"
        address = "vous" if score >= 0 else "tu"
        return StyleProfile(style=style, address=address, verbosity=verbosity)

    if score >= 2:
        return StyleProfile(style="formal", address="vous", verbosity=verbosity)
    if score <= -2:
        return StyleProfile(style="casual", address="tu", verbosity=verbosity)

    # Neutral: default to pro framing
    return StyleProfile(style="neutral", address="vous", verbosity=verbosity)


@dataclass
class StyleMemory:
    window: int = 3
    _history: Deque[str] = None

    def __post_init__(self) -> None:
        self._history = deque(maxlen=self.window)

    def update(self, profile: StyleProfile) -> StyleProfile:
        # Do not override tense signals
        self._history.append(profile.style)
        if profile.style == "tense":
            return profile

        if len(self._history) == self.window and all(s == "casual" for s in self._history):
            return StyleProfile(style="casual", address="tu", verbosity=profile.verbosity)

        return profile


def build_system_prompt(scenario_label: str, scenario_desc: str, profile: StyleProfile) -> str:
    """Deprecated: use build_style_instructions + scenario context in the text generator."""
    _ = scenario_label, scenario_desc
    return build_style_instructions(profile)


def build_style_instructions(profile: StyleProfile) -> str:
    if profile.style == "formal":
        style = (
            "Tone: professional and authoritative but polite. Use formal address (vous). "
            "Be precise and structured."
        )
    elif profile.style == "casual":
        style = (
            "Tone: casual and friendly. Use informal address (tu). "
            "Be direct and relational."
        )
    elif profile.style == "tense":
        style = (
            "User seems tense or suspicious. Stay calm and reassuring, use technical terms "
            f"and keep {profile.address} address for credibility."
        )
    else:
        style = "Tone: neutral and distant. Use formal address (vous)."

    if profile.verbosity == "short":
        style += " Keep it brief (1-3 sentences)."
    else:
        style += " Medium detail (3-6 sentences)."

    return style
