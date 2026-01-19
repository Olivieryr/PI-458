import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class RedactionResult:
    redacted_text: str
    detected: bool
    types: List[str]


# --- Patterns (V1 pragmatique) ---
EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
PHONE_FR_RE = re.compile(r"\b(?:(?:\+|00)33|0)\s*[1-9](?:[\s.-]*\d{2}){4}\b")

# IBAN (approx robuste) : FR + generique IBAN
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}(?:[ -]?[A-Z0-9]){11,30}\b")

# Carte bancaire : 13 a 19 chiffres (avec espaces/tirets)
CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")

# CVV : 3-4 chiffres, souvent "cvv: 123"
CVV_RE = re.compile(r"(?i)\b(?:cvv|cvc)\s*[:=]?\s*\d{3,4}\b")

# OTP / codes : "code: 123456", "otp 123456", "sms 123456"
OTP_RE = re.compile(r"(?i)\b(?:otp|code\s*sms|code|2fa|auth)\s*[:=]?\s*\d{4,8}\b")

# Mot de passe : "mdp: xxx", "password=xxx", "mot de passe : xxx"
PASSWORD_RE = re.compile(
    r"(?i)\b(?:mdp|mot\s*de\s*passe|password|pass)\s*[:=]\s*\S+"
)

# NIR / SSN fr (approx) : 13 a 15 chiffres (attention faux positifs, on reste prudent)
NIR_RE = re.compile(r"\b\d{13,15}\b")


def _apply(pattern: re.Pattern, label: str, text: str) -> Tuple[str, bool]:
    if pattern.search(text):
        return pattern.sub(f"[{label}_REDACTED]", text), True
    return text, False


def redact_pii(text: str) -> RedactionResult:
    """
    Detecte et masque des informations sensibles/PII avant tout traitement.
    V1: heuristique + regex. Objectif: defense en profondeur, pas perfection.
    """
    redacted = text
    types: List[str] = []

    for pattern, label in [
        (EMAIL_RE, "EMAIL"),
        (PHONE_FR_RE, "PHONE"),
        (IBAN_RE, "IBAN"),
        (CVV_RE, "CVV"),
        (OTP_RE, "OTP"),
        (PASSWORD_RE, "PASSWORD"),
        (CARD_RE, "CARD"),
        (NIR_RE, "NIR"),
    ]:
        redacted, hit = _apply(pattern, label, redacted)
        if hit:
            types.append(label)

    return RedactionResult(
        redacted_text=redacted,
        detected=len(types) > 0,
        types=types,
    )
