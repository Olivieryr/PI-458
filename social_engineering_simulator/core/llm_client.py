import requests
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass(frozen=True)
class ChatMessage:
    role: str   # "system" | "user" | "assistant"
    content: str


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout_s: int = 60):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.session = requests.Session()
        self.session.trust_env = False

    def chat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.6,
        top_p: float = 0.9,
        num_predict: int = 180,
    ) -> str:
        """
        Appelle Ollama /api/chat (non-streaming) et retourne le contenu assistant.
        """
        url = f"{self.base_url}/api/chat"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": num_predict,
            },
        }

        try:
            r = self.session.post(url, json=payload, timeout=self.timeout_s)
            r.raise_for_status()
            data = r.json()
            return data.get("message", {}).get("content", "").strip()
        except requests.RequestException as e:
            return (
                f"Erreur: impossible de contacter Ollama ({type(e).__name__}). "
                "Verifiez l'acces a http://localhost:11434."
            )
