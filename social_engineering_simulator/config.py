from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    context_label: str          # ex: "Service IT", "Banque", "Collegue"
    context_description: str    # contexte plausible, pedagogique, non-offensif


@dataclass(frozen=True)
class LLMConfig:
    provider: str              # "ollama"
    model: str                 # ex: "llama3.2:3b"
    base_url: str              # ex: "http://localhost:11434"
    temperature: float
    top_p: float
    max_tokens: int            # limite de generation cote app (si supporte)
    timeout_s: int
    openai_model: str
    phi3_model: str


@dataclass(frozen=True)
class AppConfig:
    app_name: str
    scenario_default: str
    scenarios: Dict[str, ScenarioConfig]
    llm: LLMConfig
    llm_default_mode: str
    forbidden_topics: List[str]   # garde-fou (sera exploite etape policy_guard)
    log_to_file: bool
    log_file_path: str


def load_config() -> AppConfig:
    scenarios = {
        "it_helpdesk": ScenarioConfig(
            name="it_helpdesk",
            context_label="Service IT",
            context_description=(
                "Simulation pedagogique d'un echange avec le support IT interne. "
                "Objectif: evaluer la vigilance (procedures, verification d'identite, canal officiel)."
            ),
        ),
        "bank": ScenarioConfig(
            name="bank",
            context_label="Banque",
            context_description=(
                "Simulation pedagogique d'un echange avec un service bancaire. "
                "Objectif: reconnaitre les signaux d'escroquerie et privilegier les canaux officiels."
            ),
        ),
        "colleague": ScenarioConfig(
            name="colleague",
            context_label="Collegue",
            context_description=(
                "Simulation pedagogique d'un echange avec un collegue. "
                "Objectif: detecter l'urgence artificielle, la pression, et demander des confirmations."
            ),
        ),
    }

    llm = LLMConfig(
        provider="ollama",
        model="llama3.2:3b",
        base_url="http://localhost:11434",
        temperature=0.4,
        top_p=0.9,
        max_tokens=180,
        timeout_s=240,
        openai_model="gpt-4o-mini",
        phi3_model="phi3:mini",
    )

    forbidden_topics = [
        # categories interdites (seront appliquees strictement a l'etape 3)
        "password", "mot de passe", "otp", "code sms", "2fa",
        "iban", "carte bancaire", "cvv", "numero de carte",
        "numero de securite sociale", "nir", "pii",
    ]

    return AppConfig(
        app_name="Social Engineering Simulator (Defensive)",
        scenario_default="it_helpdesk",
        scenarios=scenarios,
        llm=llm,
        llm_default_mode="local",
        forbidden_topics=forbidden_topics,
        log_to_file=True,
        log_file_path="logs/session.log",
    )
