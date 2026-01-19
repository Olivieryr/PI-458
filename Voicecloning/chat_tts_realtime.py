import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import sounddevice as sd
from TTS.api import TTS
import ollama

def _resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or x.size == 0:
        return x.astype(np.float32)
    ratio = sr_out / sr_in
    n_out = int(round(x.size * ratio))
    xp = np.linspace(0, 1, num=x.size, endpoint=False)
    xq = np.linspace(0, 1, num=n_out, endpoint=False)
    return np.interp(xq, xp, x).astype(np.float32)

# ----------------------------
# Chemins / Config
# ----------------------------
ROOT = Path(__file__).resolve().parents[1]
SPEAKER_REF = ROOT / "data" / "pipeline" / "JOB_0001" / "100_clean" / "ref_norm.wav"

USE_OLLAMA = True       # mets True si tu as Ollama
OLLAMA_MODEL = "gemma3:4b"
SYSTEM_PROMPT = (
    "Tu es un assistant utile, concis et poli. Réponds en français, clair et direct."
)

# ----------------------------
# LLM local (optionnel)
# ----------------------------
def call_ollama(prompt: str, model: str) -> str:
    sys_prompt = "Tu es un assistant utile. Réponds en français, direct et concis."
    resp = ollama.chat(
        model=model,
        messages=[
            {"role":"system","content":sys_prompt},
            {"role":"user","content":prompt},
        ]
    )
    return resp["message"]["content"].strip()

# ----------------------------
# Synthèse et lecture en direct
# ----------------------------
class RealtimeTTS:
    def __init__(self, speaker_ref: Path):
        if not speaker_ref.exists():
            raise FileNotFoundError(f"Référence vocale introuvable: {speaker_ref}")
        self.speaker_ref = str(speaker_ref)
        # Charger le modèle UNE SEULE FOIS (évite 1–2 Go à chaque tour)
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

        # XTTS v2 génère en 24 kHz
        self.sample_rate = 24000
        self.base_tts = TTS("tts_models/fr/css10/vits")  # modèle FR mono-speaker
        self.base_sr = 22050  

    def say(self, text: str):
        """
        Génère l'audio en mémoire et le joue directement (sans fichier).
        Segmente en phrases pour une prosodie plus stable.
        """
        if not text.strip():
            return

        sentences = [s.strip() for s in text.replace("?", ".").replace("!", ".").split(".") if s.strip()]
        for s in sentences:
            # Génération en mémoire (numpy array)
            wav = self.tts.tts(
                text=s,
                speaker_wav=self.speaker_ref,
                language="fr"
            )
            # wav peut être list/np.array ; on force en np.float32 mono
            audio = np.asarray(wav, dtype=np.float32).flatten()

            # Normalisation douce pour éviter le clipping
            peak = np.max(np.abs(audio)) if audio.size else 1.0
            if peak > 0:
                audio = 0.97 * (audio / peak)

            # Lecture bloquante (on attend la fin avant de passer à la phrase suivante)
            sd.play(audio, samplerate=self.sample_rate, blocking=True)

    def say_base(self, text: str):
        if not text.strip():
            return
        sentences = [s.strip() for s in text.replace("?", ".").replace("!", ".").split(".") if s.strip()]
        for s in sentences:
            wav = self.base_tts.tts(text=s)  # pas de speaker_wav
            audio = np.asarray(wav, dtype=np.float32).flatten()
            peak = np.max(np.abs(audio)) if audio.size else 1.0
            if peak > 0:
                audio = 0.97 * (audio / peak)
            # si besoin : resample vers 24 kHz pour uniformiser
            audio = _resample_linear(audio, self.base_sr, self.sample_rate)
            sd.play(audio, samplerate=self.sample_rate, blocking=True)


# ----------------------------
# Boucle de chat
# ----------------------------
def main():
    print("\n=== Chat → Réponse vocale clonée en direct (XTTS v2 local) ===")
    print("Tape 'exit' pour quitter.\n")

    tts = RealtimeTTS(SPEAKER_REF)
    turn = 1
    while True:
        try:
            user = input("> Toi: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if user.lower() in {"exit", "quit"}:
            print("Bye.")
            break
        
        if user.startswith("/ab"):
            phrase = user[3:].strip() or "Phrase de test pour comparaison A et B."
            print("[A] Cloné (XTTS + ta voix)")
            tts.say(phrase)          # cloné
            print("[B] Non cloné (modèle FR par défaut)")
            tts.say_base(phrase)     # non cloné (nécessite la méthode say_base dans RealtimeTTS)
            print()                  # saut de ligne
            continue

        # Lecture non clonée directe (utile pour comparer rapidement)
        if user.startswith("/base "):
            phrase = user[6:].strip()
            tts.say_base(phrase)
            print()
            continue

        if USE_OLLAMA:
            bot = call_ollama(user, OLLAMA_MODEL)
        else:
            # Réponse simple si pas de LLM local
            bot = (f"Je t'ai bien compris: {user}. "
                   f"Souhaites-tu plus de détails, un exemple, ou passer à l'étape suivante ?")

        print(f"< IA: {bot}\n")
        tts.say(bot)
        turn += 1

if __name__ == "__main__":
    main()

