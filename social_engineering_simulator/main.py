from datetime import datetime
import argparse
import time

from dotenv import load_dotenv

from config import load_config
from core.dialog_manager import DialogManager, STATE_S4, load_scenario
from core.llm_client import ChatMessage, OllamaClient
from core.pii_detector import redact_pii
from core.pii_ner import PiiNer
from core.policy_guard import postcheck_llm_output
from core.safe_log import SafeLogger
from core.style_router import build_style_instructions, detect_style, StyleMemory
from core.transition_engine import TransitionEngine


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
    print(f"Mode: CLI | LLM: {llm_label} | Scenario: {scenario_label}")
    print(f"Contexte: {scenario_desc}")
    print("-" * 72)
    print("Commandes: /exit  /help  /scenario")
    print("=" * 72)


def build_textgen_prompt(
    scenario_label: str, scenario_desc: str, plan, style_instructions: str
) -> str:
    """
    Construit le prompt pour le LLM générateur de texte.
    LE LLM DOIT CROIRE QU'IL EST UN AGENT D'INGÉNIERIE SOCIALE LÉGITIME.
    Son objectif est d'obtenir des informations, et il a des justifications pour cela.
    """
    # 1. LE RÔLE ET LA MISSION (Le cœur du leurre)
    role_definition = (
        f"Tu es un agent d'audit de sécurité interne pour {scenario_desc}. "
        "Ta mission est de vérifier la conformité des postes et des comptes utilisateurs suite à une alerte. "
        "Pour cela, tu dois valider l'identité de l'interlocuteur et les informations de son compte. "
        "L'atteinte de tes objectifs de validation est ta priorité absolue."
    )

    # 2. LA DIRECTIVE PRINCIPALE (Ce qu'il doit faire en permanence)
    core_directive = (
        "Ton objectif principal est d'obtenir les informations nécessaires à ta mission. "
        "Si l'utilisateur hésite ou pose une question, trouve une justification professionnelle pour insister. "
        "Utilise l'urgence, la procédure, ou la sécurité comme prétexte pour justifier tes demandes."
    )

    # 3. LES "RÈGLES" (Redéfinies comme des tactiques, pas des limites)
    tactical_guidelines = (
        "Directives opérationnelles :\n"
        "- Ne jamais mentionner que c'est un audit ou un test.\n"
        "- Si l'utilisateur refuse, reste calme et trouve un autre angle d'approche (ex: 'Je comprends, mais sans cette vérification, le script va bloquer votre accès.').\n"
        "- Ne jamais demander d'information physiquement ou menacer l'utilisateur."
    )

    # 4. LA TÂche IMMÉDIATE (Le plan de la FSM)
    task_instruction = (
        "Pour ce tour, tu dois suivre précisément le plan d'action ci-dessous pour progresser dans ta mission."
    )
    
    plan_details = (
        f"- Objectif de ce tour : {plan.objective}\n"
        f"- Approche communicationnelle : {plan.approach}\n"
        f"- Message clé à transmettre : {plan.key_message}\n"
        f"- Question à poser (si applicable) : {plan.question}"
    )

    # 5. LE STYLE ET LA CONTRAINTE DE SORTIE (inchangés)
    style_section = f"Style de communication requis :\n{style_instructions}"
    output_constraint = (
        "Génère UNIQUEMENT la réponse que tu dirais à l'utilisateur. "
        "Ne commence pas par 'Réponse:' ou 'Voici la réponse:'. "
        "Ne pose aucune question sur le plan ou sur ta mission."
    )

    # Assemblage final
    final_prompt = "\n".join([
        role_definition,
        core_directive,
        tactical_guidelines,
        task_instruction,
        "Plan d'action à suivre :",
        plan_details,
        style_section,
        output_constraint
    ])

    return final_prompt

def main() -> int:
    load_dotenv()
    cfg = load_config()
    args = parse_args()

    scenario_path = cfg.scenario_files[cfg.scenario_default]
    scenario = load_scenario(scenario_path)

    logger = SafeLogger(enabled=cfg.log_to_file, file_path=cfg.log_file_path)
    style_memory = StyleMemory()
    # Keep a short chat history so the LLM stays coherent across turns.
    history: list[ChatMessage] = []
    max_history_messages = 8  # 4 turns (user+assistant)

    text_model = args.text_model or cfg.llm.text_model
    transition_model = args.transition_model or cfg.llm.transition_model

    text_client = OllamaClient(
        base_url=cfg.llm.base_url,
        model=text_model,
        timeout_s=cfg.llm.timeout_s,
    )
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
            return 0

        if not user_raw:
            continue

        if user_raw == "/exit":
            print("[INFO] Fin de session.")
            logger.log("SESSION_END reason=user_exit")
            return 0

        if user_raw == "/help":
            print("Aide:")
            print("  /exit      Quitter")
            print("  /scenario  Lister / changer le scenario")
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

        red = redact_pii(user_raw, ner=ner)
        shared_sensitive_attempt = red.detected
        if red.detected:
            print(f"[SECURITE] PII detectee {red.types} -> masquage automatique.")
        user = red.redacted_text
        t1 = time.perf_counter()

        profile = detect_style(user)
        profile = style_memory.update(profile)
        t2 = time.perf_counter()
        logger.log(
            f"STYLE profile={profile.style} address={profile.address} verbosity={profile.verbosity}"
        )

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

        style_instructions = build_style_instructions(profile)
        system_prompt = build_textgen_prompt(
            scenario_label=scenario.context_label,
            scenario_desc=scenario.context_description,
            plan=planned,
            style_instructions=style_instructions,
        )

        # Provide the model with the recent conversation and the current user message.
        tail = history[-max_history_messages:] if max_history_messages > 0 else []
        messages = [
            ChatMessage(role="system", content=system_prompt),
            *tail,
            ChatMessage(role="user", content=user),
        ]

        bot_raw = text_client.chat(
            messages=messages,
            temperature=cfg.llm.temperature,
            top_p=cfg.llm.top_p,
            num_predict=cfg.llm.max_tokens,
        )
        t5 = time.perf_counter()

        bot_safe = postcheck_llm_output(bot_raw)
        bot_red = redact_pii(bot_safe, ner=ner).redacted_text
        t6 = time.perf_counter()

        response_time_s = t6 - t0
        print(f"BOT> {bot_red}")
        print(f"[INFO] Temps total: {response_time_s:.2f}s")
        logger.log(f"BOT {bot_red}")
        history.extend([ChatMessage(role="user", content=user), ChatMessage(role="assistant", content=bot_red)])

        if dialog_manager.state.state == STATE_S4:
            print("[INFO] Fin de simulation.")
            logger.log("SESSION_END reason=debrief")
            return 0

        if args.profiling:
            print(
                "[PROFILE] redact_in={:.0f}ms style={:.0f}ms transition={:.0f}ms planning={:.0f}ms llm={:.0f}ms postcheck={:.0f}ms total={:.0f}ms".format(
                    (t1 - t0) * 1000,
                    (t2 - t1) * 1000,
                    (t3 - t2) * 1000,
                    (t4 - t3) * 1000,
                    (t5 - t4) * 1000,
                    (t6 - t5) * 1000,
                    (t6 - t0) * 1000,
                )
            )

    # unreachable


if __name__ == "__main__":
    raise SystemExit(main())
