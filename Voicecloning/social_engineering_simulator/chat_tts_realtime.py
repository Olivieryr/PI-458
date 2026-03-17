import torch
from TTS.api import TTS
import ollama
import sounddevice as sd
import numpy as np
from pathlib import Path
import re # Pour nettoyer le texte (regex)

# --- CONFIGURATION ---
SPEAKER_REF = "voix_norm2.wav"
OLLAMA_MODEL = "gemma3:4b"

# Définition de la personnalité de l'IA
SYSTEM_PROMPT = {
    'role': 'system', 
    'content': "Tu es une IA conversationnelle utile et concise. Réponds de manière naturelle, comme à l'oral. Évite les listes à puces trop longues."
}
# On désactive la vérification de sécurité qui posait problème
import functools
torch.load = functools.partial(torch.load, weights_only=False)

# Chargement du modèle (GPU si dispo, sinon CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def clean_text(text):
    """Enlève les astérisques (*sourit*) et les émojis pour la lecture."""
    # Enlève le texte entre astérisques (ex: *rit*)
    text = re.sub(r'\*.*?\*', '', text)
    # Enlève les espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def play_audio(text):
    """Génère et joue l'audio pour une phrase donnée."""
    if not text.strip(): return
    
    try:
        # Génération
        wav = tts.tts(text=text, speaker_wav=SPEAKER_REF, language="fr")
        # Lecture
        sd.play(np.array(wav), samplerate=24000)
        sd.wait() # On attend la fin de la phrase pour que ce soit fluide
    except Exception as e:
        print(f"\n[Erreur Audio]: {e}")

def main():
    # Historique de la conversation (Mémoire)
    messages = [SYSTEM_PROMPT]
    
    print("\n✅ Prêt ! (Mode Streaming activé). Tape 'exit' pour quitter.")
    
    while True:
        try:
            user_input = input("\n> Toi: ")
            if user_input.lower() in ["exit", "quit"]:
                break

            # 1. Ajout du message utilisateur à l'historique
            messages.append({'role': 'user', 'content': user_input})
            
            print("IA: ", end="", flush=True)

            # 2. Appel à Ollama en mode STREAMING
            stream = ollama.chat(
                model=OLLAMA_MODEL, 
                messages=messages, 
                stream=True
            )

            full_response = ""
            current_sentence = ""
            
            # 3. Boucle de lecture en direct
            for chunk in stream:
                content = chunk['message']['content']
                print(content, end="", flush=True) # Affichage visuel progressif
                
                full_response += content
                current_sentence += content

                # Détection de fin de phrase (. ? ! ou retour ligne)
                if any(punct in content for punct in [".", "?", "!", "\n", ":"]):
                    # On nettoie et on lit la phrase immédiatement
                    clean_sentence = clean_text(current_sentence)
                    if len(clean_sentence) > 2: # Évite de lire des "." isolés
                        play_audio(clean_sentence)
                    current_sentence = "" # On vide le tampon pour la suite

            # Si il reste un bout de phrase à la fin
            if current_sentence.strip():
                 play_audio(clean_text(current_sentence))
            
            print() # Saut de ligne à la fin

            # 4. Sauvegarde de la réponse complète dans la mémoire
            messages.append({'role': 'assistant', 'content': full_response})

        except KeyboardInterrupt:
            print("\nArrêt en cours...")
            break
        except Exception as e:
            print(f"\n❌ Erreur : {e}")

if __name__ == "__main__":
    main()