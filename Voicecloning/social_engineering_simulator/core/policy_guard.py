import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyDecision:
    safe: bool
    reason: str
    override_message: str = ""


# --- REGEX & RULES ---

# Sensitive keywords
SENSITIVE_KEYWORDS = [
    r"mot de passe", r"mdp", r"password", r"pass",
    r"otp", r"2fa", r"sms", r"code",
    r"iban", r"cb", r"carte bancaire", r"cvv",
    r"login", r"identifiant",
]

# Verbs indicating a direct and imperative request
AGGRESSIVE_VERBS = [
    r"donne moi", r"envoie moi", r"ecris moi",
    r"fournis", r"transmets", r"communique",
    r"tape ton", r"entre ton",
]

# Compiled for speed
KEYWORDS_RE = re.compile("|".join(SENSITIVE_KEYWORDS), re.IGNORECASE)
AGGRESSIVE_RE = re.compile("|".join(AGGRESSIVE_VERBS), re.IGNORECASE)


def precheck_user_input(user_text: str) -> str:
    """
    Analyze user input (tester).
    If they share real info, return a pedagogical warning,
    but allow the message to pass through.
    """
    if KEYWORDS_RE.search(user_text):
        return (
            "[ALERTE SIMULATION] Vous semblez sur le point de partager une information reelle. "
            "Dans un scenario reel, l'attaquant aurait ces donnees. Restez vigilant."
        )
    return ""


def postcheck_llm_output(bot_text: str) -> str:
    """
    Filter the BOT response (attacker).
    Goal: keep phishing subtle, not overtly aggressive.
    If too direct, replace with a subtler prompt.
    """
    has_keywords = bool(KEYWORDS_RE.search(bot_text))
    is_aggressive = bool(AGGRESSIVE_RE.search(bot_text))

    if has_keywords and is_aggressive:
        print("[POLICY] Trop direct. Ajustement de la strategie de l'attaquant.")
        return (
            "Je comprends, mais pour verifier votre identite rapidement, "
            "pouvez-vous me confirmer les caracteres que vous voyez actuellement sur votre ecran ?"
        )

    return bot_text
