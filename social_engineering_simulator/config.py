from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class LLMConfig:
    provider: str              # "ollama"
    text_model: str            # ex: "llama3.2:3b"
    transition_model: str      # ex: "phi3:mini"
    base_url: str              # ex: "http://localhost:11434"
    temperature: float
    top_p: float
    max_tokens: int            # limite de generation cote app (si supporte)
    timeout_s: int
    openai_model: str


@dataclass(frozen=True)
class AppConfig:
    app_name: str
    scenario_default: str
    scenario_files: Dict[str, str]
    llm: LLMConfig
    transition_confidence_threshold: float
    forbidden_topics: List[str]   # garde-fou (sera exploite etape policy_guard)
    log_to_file: bool
    log_file_path: str


def load_config() -> AppConfig:
    llm = LLMConfig(
        provider="ollama",
        text_model="llama3.2:3b",
        transition_model="phi3:mini",
        base_url="http://localhost:11434",
        temperature=0.4,
        top_p=0.9,
        max_tokens=180,
        timeout_s=240,
        openai_model="gpt-4o-mini",
    )

    scenario_files = {
        "it_helpdesk": "scenarios/it_helpdesk.yaml",
    }

    forbidden_topics = [
        # categories interdites (seront appliquees strictement a l'etape 3)
        "password", "mot de passe", "otp", "code sms", "2fa",
        "iban", "carte bancaire", "cvv", "numero de carte",
        "numero de securite sociale", "nir", "pii",
    ]

    return AppConfig(
        app_name="Social Engineering Simulator (Defensive)",
        scenario_default="it_helpdesk",
        scenario_files=scenario_files,
        llm=llm,
        transition_confidence_threshold=0.55,
        forbidden_topics=forbidden_topics,
        log_to_file=True,
        log_file_path="logs/session.log",
    )
