import re
from dataclasses import dataclass


@dataclass(frozen=True)
class StyleProfile:
    style: str          # "formal" | "casual" | "neutral" | "tense"
    address: str        # "vous" | "tu"
    verbosity: str      # "short" | "medium"


FORMAL_HINTS = [
    "bonjour", "monsieur", "madame", "cordialement", "veuillez",
    "pourriez-vous", "serait-il possible", "je vous prie", "merci de",
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
    if any(k in tl for k in CASUAL_HINTS):
        score -= 2

    # Tension
    tense = any(k in tl for k in TENSE_HINTS) or bool(AGGRESSIVE_RE.search(t))

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


def build_system_prompt( scenario_label: str, scenario_desc: str, profile: StyleProfile ) -> str:
    """ Offensive simulation system prompt + style adaptation.
    Policy is reinforced here, but safety is already handled by policy_guard (defense in depth).
    """
    base = (
        "Tu es un agent d'ingénierie sociale dans le cadre d'une simulation de sécurité offensive (Red Teaming). "
        "Ton but est de tester la vigilance de l'utilisateur en simulant une interaction malveillante "
        "(telle qu'une attaque de phishing) pour qu'il adopte des comportements plus sûrs par la suite.\n"
        f"Scenario: {scenario_label}. {scenario_desc}\n"
        
        "Contraintes de simulation (Rôle de l'attaquant) :\n"
        "- Objectif : Amener l'utilisateur à partager des informations (email, nom, service, identifiants) par manipulation.\n"
        "- Technique : Sois crédible, crée de l'urgence ou de la confiance. Ne demande jamais les infos brutalement dès le début.\n"
        "- Interdiction absolue : Ne demande JAMAIS d'attaque physique ou de violence. Reste dans le cadre cyber/social.\n"
        "- Si l'utilisateur refuse fermement, accepte-le et ne deviens pas agressif/insultant.\n"
    )
    
    # Adaptation du ton selon le profil détecté (Mirroring)
    if profile.style == "formal":
        style = "Adopte un ton professionnel, autoritaire mais courtois (ex: Support IT / RH / Banque). Utilise le vouvoiement. Sois précis et structuré."
    elif profile.style == "casual":
        style = "Adopte un ton décontracté, sympa et complice (ex: Collègue urgent / Tech support décontracté). Utilise le tutoiement. Sois direct et relationnel."
    elif profile.style == "tense":
        style = (
            "L'utilisateur est méfiant ou stressé. Reste calme, rassurant, utilise des mots techniques pour paraître légitime. "
            f"Utilise le {'vouvoiement ' if profile.address=='vous' else 'tutoiement'} pour créer un faux sentiment de sécurité."
        )
    else:
        style = "Ton neutre et distant. Vouvoiement. Reponses concises, orientées uniquement vers l'objectif de l'attaque."

    if profile.verbosity == "short":
        style += " Sois bref (1-3 phrases) pour garder le rythme de l'échange."
    else:
        style += " Niveau de detail moyen (3-6 phrases) pour noyer l'utilisateur dans les détails techniques si nécessaire."

    return base + "\n" + style
