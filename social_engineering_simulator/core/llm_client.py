import requests
from dataclasses import dataclass
from typing import List, Dict, Any
import os


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


class OpenAIClient:
    def __init__(self, model: str, timeout_s: int = 60, api_key: str = "", base_url: str = ""):
        try:
            from openai import OpenAI
        except Exception as e:  # pragma: no cover - optional dependency
            raise RuntimeError("openai sdk is not installed") from e

        self.model = model
        self.timeout_s = timeout_s
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or None,
            timeout=timeout_s,
        )

    def chat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.6,
        top_p: float = 0.9,
        num_predict: int = 180,
    ) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=temperature,
                top_p=top_p,
                max_tokens=num_predict,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            return f"Erreur: appel OpenAI impossible ({type(e).__name__}). Verifiez la cle OPENAI_API_KEY."
