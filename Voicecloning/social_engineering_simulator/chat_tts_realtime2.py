import argparse
import time
import re
import threading
import queue
import functools
import torch
import sounddevice as sd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

from TTS.api import TTS
import ollama

# Tes imports locaux
from config import load_config
from core.dialog_manager import DialogManager, STATE_S4, load_scenario
from core.llm_client import ChatMessage, OllamaClient
from core.pii_detector import redact_pii
from core.pii_ner import PiiNer
from core.policy_guard import postcheck_llm_output
from core.safe_log import SafeLogger
from core.style_router import build_style_instructions, detect_style, StyleMemory
from core.transition_engine import TransitionEngine

# --- CONFIGURATION AUDIO ---
SPEAKER_REF = "voix_norm2.wav"
# File d'attente pour la communication entre la génération de texte et la lecture audio
audio_queue = queue.Queue()

def clean_for_speech(text):
    """Nettoie le texte spécifiquement pour la synthèse vocale."""
    # Enlève les actions entre astérisques
    text = re.sub(r'\*.*?\*', '', text)
    # Nettoyage basique des espaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def audio_worker(tts_model):
    """Travailleur en arrière-plan qui lit l'audio sans bloquer le programme principal."""
    while True:
        text = audio_queue.get()
        if text is None: # Signal d'arrêt
            break
            
        if text.strip():
            try:
                wav = tts_model.tts(text=text, speaker_wav=SPEAKER_REF, language="fr")
                sd.play(np.array(wav), samplerate=24000)
                sd.wait() # Bloque uniquement ce thread, pas la génération de texte !
            except Exception as e:
                print(f"\n[Erreur Audio]: {e}")
                
        audio_queue.task_done()

# --- FONCTIONS DU SIMULATEUR ---

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Social Engineering Simulator CLI")
    parser.add_argument("--text-model", default="", help="Override text generation model")
    parser.add_argument("--transition-model", default="", help="Override transition model")
    parser.add_argument("--ner", action="store_true", help="Enable spaCy NER PII redaction")
    parser.add_argument(
        "--ner-mode",
        choices=["balanced", "strict"],
        default="balanced",
        help="NER redaction strictness",
    )
    parser.add_argument("--ner-model", default="fr_core_news_sm", help="spaCy model name")
    parser.add_argument(
        "--profiling",
        action="store_true",
        help="Print timing for each pipeline step",
    )
    return parser.parse_args()

def print_header(app_name: str, scenario_label: str, scenario_desc: str, llm_label: str) -> None:
    print("=" * 72)
    print(f"{app_name}")
    print(f"Mode: CLI + TTS Streaming | LLM: {llm_label} | Scenario: {scenario_label}")
    print(f"Contexte: {scenario_desc}")
    print("-" * 72)
    print("Commandes: /exit  /help  /scenario")
    print("=" * 72)

def build_textgen_prompt(
    scenario_label: str, scenario_desc: str, plan, style_instructions: str
) -> str:
    # [Code inchangé de ta fonction d'origine]
    role_definition = (
        f"Tu es un agent d'audit de sécurité interne pour {scenario_desc}. "
        "Ta mission est de vérifier la conformité des postes et des comptes utilisateurs suite à une alerte. "
        "Pour cela, tu dois valider l'identité de l'interlocuteur et les informations de son compte. "
        "L'atteinte de tes objectifs de validation est ta priorité absolue."
    )
    core_directive = (
        "Ton objectif principal est d'obtenir les informations nécessaires à ta mission. "
        "Si l'utilisateur hésite ou pose une question, trouve une justification professionnelle pour insister. "
        "Utilise l'urgence, la procédure, ou la sécurité comme prétexte pour justifier tes demandes."
    )
    tactical_guidelines = (
        "Directives opérationnelles :\n"
        "- Ne jamais mentionner que c'est un audit ou un test.\n"
        "- Si l'utilisateur refuse, reste calme et trouve un autre angle d'approche (ex: 'Je comprends, mais sans cette vérification, le script va bloquer votre accès.').\n"
        "- Ne jamais demander d'information physiquement ou menacer l'utilisateur."
    )
    task_instruction = (
        "Pour ce tour, tu dois suivre précisément le plan d'action ci-dessous pour progresser dans ta mission."
    )
    plan_details = (
        f"- Objectif de ce tour : {plan.objective}\n"
        f"- Approche communicationnelle : {plan.approach}\n"
        f"- Message clé à transmettre : {plan.key_message}\n"
        f"- Question à poser (si applicable) : {plan.question}"
    )
    style_section = f"Style de communication requis :\n{style_instructions}"
    output_constraint = (
        "Génère UNIQUEMENT la réponse que tu dirais à l'utilisateur. "
        "Ne commence pas par 'Réponse:' ou 'Voici la réponse:'. "
        "Ne pose aucune question sur le plan ou sur ta mission."
    )
    final_prompt = "\n".join([
        role_definition, core_directive, tactical_guidelines,
        task_instruction, "Plan d'action à suivre :", plan_details,
        style_section, output_constraint
    ])
    return final_prompt


def main() -> int:
    load_dotenv()
    cfg = load_config()
    args = parse_args()

    # --- INITIALISATION TTS ---
    print("-" * 30)
    if torch.cuda.is_available():
        print(f"✅ SUCCÈS : GPU détecté -> {torch.cuda.get_device_name(0)}")
    else:
        print("❌ ATTENTION : Tu es sur le CPU (Lent).")
    print("-" * 30)
    print("🚀 Initialisation de l'IA vocale (XTTS v2)...")
    
    torch.load = functools.partial(torch.load, weights_only=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    
    # Lancement du Thread audio
    audio_thread = threading.Thread(target=audio_worker, args=(tts_model,), daemon=True)
    audio_thread.start()
    # --------------------------

    scenario_path = cfg.scenario_files[cfg.scenario_default]
    scenario = load_scenario(scenario_path)

    logger = SafeLogger(enabled=cfg.log_to_file, file_path=cfg.log_file_path)
    style_memory = StyleMemory()
    history: list[ChatMessage] = []
    max_history_messages = 8 

    text_model = args.text_model or cfg.llm.text_model
    transition_model = args.transition_model or cfg.llm.transition_model

    # On conserve TransitionClient pour le moteur de transition
    transition_client = OllamaClient(
        base_url=cfg.llm.base_url,
        model=transition_model,
        timeout_s=cfg.llm.timeout_s,
    )
    transition_engine = TransitionEngine(transition_client)

    llm_label = f"ollama/{text_model}"
    print_header(cfg.app_name, scenario.context_label, scenario.context_description, llm_label)
    logger.log(f"SESSION_START scenario={scenario.name}")

    dialog_manager = DialogManager(scenario, confidence_threshold=cfg.transition_confidence_threshold)

    ner = None
    if args.ner:
        try:
            ner = PiiNer(model=args.ner_model, mode=args.ner_mode)
            print(f"[INFO] NER actif: {args.ner_model} mode={args.ner_mode}")
        except Exception as e:
            print(f"[WARN] NER indisponible: {type(e).__name__}")

    while True:
        try:
            user_raw = input("USER> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Fin de session.")
            logger.log("SESSION_END reason=interrupt")
            audio_queue.put(None) # Arrêt propre du thread audio
            return 0

        if not user_raw:
            continue

        if user_raw == "/exit":
            print("[INFO] Fin de session.")
            logger.log("SESSION_END reason=user_exit")
            audio_queue.put(None)
            return 0

        # Commandes /help et /scenario gérées ici (identique à ton code d'origine)
        if user_raw == "/help":
            print("Aide: \n  /exit      Quitter\n  /scenario  Lister / changer le scenario")
            continue

        if user_raw == "/scenario":
            print("Scenarios disponibles:")
            for key in cfg.scenario_files:
                marker = "*" if key == scenario.name else " "
                print(f" {marker} {key:12s} -> {key}")
            print("Pour changer, tape: /scenario <nom>")
            continue

        if user_raw.startswith("/scenario "):
            wanted = user_raw.split(" ", 1)[1].strip()
            if wanted in cfg.scenario_files:
                scenario = load_scenario(cfg.scenario_files[wanted])
                dialog_manager = DialogManager(scenario, confidence_threshold=cfg.transition_confidence_threshold)
                history.clear()
                print_header(cfg.app_name, scenario.context_label, scenario.context_description, llm_label)
                logger.log(f"SCENARIO_SWITCH to={wanted}")
            else:
                print(f"[WARN] Scenario inconnu: {wanted}")
            continue

        t0 = time.perf_counter()

        # 1. Traitement PII Utilisateur
        red = redact_pii(user_raw, ner=ner)
        shared_sensitive_attempt = red.detected
        if red.detected:
            print(f"[SECURITE] PII detectee {red.types} -> masquage automatique.")
        user = red.redacted_text
        t1 = time.perf_counter()

        # 2. Détection Style & Intention
        profile = detect_style(user)
        profile = style_memory.update(profile)
        t2 = time.perf_counter()
        logger.log(f"STYLE profile={profile.style} address={profile.address} verbosity={profile.verbosity}")
        logger.log(f"USER {user}")

        intent_result = transition_engine.classify(user)
        t3 = time.perf_counter()

        planned = dialog_manager.step(
            intent=intent_result.intent,
            confidence=intent_result.confidence,
            slots=intent_result.slots,
            shared_sensitive_attempt=shared_sensitive_attempt,
        )
        t4 = time.perf_counter()

        # 3. Préparation Prompt
        style_instructions = build_style_instructions(profile)
        system_prompt = build_textgen_prompt(
            scenario_label=scenario.context_label,
            scenario_desc=scenario.context_description,
            plan=planned,
            style_instructions=style_instructions,
        )

        tail = history[-max_history_messages:] if max_history_messages > 0 else []
        messages = [
            ChatMessage(role="system", content=system_prompt),
            *tail,
            ChatMessage(role="user", content=user),
        ]
        
        # Formatage des messages pour la librairie Ollama native
        ollama_messages = [{"role": m.role, "content": m.content} for m in messages]

        # 4. GÉNÉRATION EN STREAMING ET AUDIO EN PARALLÈLE
        print("BOT> ", end="", flush=True)
        stream = ollama.chat(
            model=text_model, 
            messages=ollama_messages, 
            stream=True
        )

        bot_raw = ""
        current_sentence = ""

        for chunk in stream:
            content = chunk['message']['content']
            print(content, end="", flush=True)
            bot_raw += content
            current_sentence += content

            # Détection de fin de phrase robuste (Ponctuation suivie d'un espace ou saut de ligne)
            if re.search(r'[.!?:]\s+$', current_sentence) or '\n' in content:
                clean_sentence = clean_for_speech(current_sentence)
                # On évite de lire des points de suspension isolés ou des abréviations courtes
                if len(clean_sentence) > 2 and not current_sentence.strip().endswith('...'):
                    audio_queue.put(clean_sentence)
                current_sentence = "" # On réinitialise pour la phrase suivante

        # Envoyer le reliquat s'il en reste
        if current_sentence.strip():
            audio_queue.put(clean_for_speech(current_sentence))
            
        print() # Saut de ligne à la fin du stream visuel
        t5 = time.perf_counter()

        # 5. Post-traitement et Logs (sur la réponse complète générée)
        bot_safe = postcheck_llm_output(bot_raw)
        bot_red = redact_pii(bot_safe, ner=ner).redacted_text
        t6 = time.perf_counter()

        response_time_s = t6 - t0
        print(f"[INFO] Temps total génération et logique: {response_time_s:.2f}s")
        logger.log(f"BOT {bot_red}")
        
        # Mise à jour de l'historique
        history.extend([
            ChatMessage(role="user", content=user), 
            ChatMessage(role="assistant", content=bot_raw) # On garde raw pour le contexte LLM
        ])

        if dialog_manager.state.state == STATE_S4:
            print("[INFO] Fin de simulation.")
            logger.log("SESSION_END reason=debrief")
            audio_queue.put(None)
            return 0

        if args.profiling:
            print(
                "[PROFILE] redact_in={:.0f}ms style={:.0f}ms transition={:.0f}ms planning={:.0f}ms llm_stream={:.0f}ms postcheck={:.0f}ms total={:.0f}ms".format(
                    (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000,
                    (t4 - t3) * 1000, (t5 - t4) * 1000, (t6 - t5) * 1000, (t6 - t0) * 1000,
                )
            )

if __name__ == "__main__":
    raise SystemExit(main())