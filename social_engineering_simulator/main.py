from datetime import datetime
import argparse
import time
from config import load_config

from dotenv import load_dotenv

from core.llm_client import ChatMessage, OllamaClient, OpenAIClient
from core.pii_detector import redact_pii
from core.pii_ner import PiiNer
from core.policy_guard import postcheck_llm_output, precheck_user_input
from core.safe_log import SafeLogger
from core.style_router import build_system_prompt, detect_style, StyleMemory


def parse_args(default_mode: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Social Engineering Simulator CLI")
    parser.add_argument(
        "--mode",
        choices=["local", "openai", "phi3"],
        default=default_mode,
        help="LLM mode (local=ollama, openai=API, phi3=ollama phi3:mini)",
    )
    parser.add_argument("--model", default="", help="Override model name for the selected mode")
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


def main() -> int:
    load_dotenv()
    cfg = load_config()
    args = parse_args(cfg.llm_default_mode)
    scenario = cfg.scenarios[cfg.scenario_default]
    logger = SafeLogger(enabled=cfg.log_to_file, file_path=cfg.log_file_path)
    style_memory = StyleMemory()

    if args.mode == "openai":
        provider = "openai"
        model = args.model or cfg.llm.openai_model
    elif args.mode == "phi3":
        provider = "ollama"
        model = args.model or cfg.llm.phi3_model
    else:
        provider = cfg.llm.provider
        model = args.model or cfg.llm.model

    if provider == "ollama":
        client = OllamaClient(
            base_url=cfg.llm.base_url,
            model=model,
            timeout_s=cfg.llm.timeout_s,
        )
    elif provider == "openai":
        try:
            client = OpenAIClient(model=model, timeout_s=cfg.llm.timeout_s)
        except Exception as e:
            print(f"[ERROR] OpenAI init failed: {type(e).__name__}")
            return 1
    else:
        print(f"[ERROR] Provider inconnu: {provider}")
        return 1

    _ = client.chat(
        messages=[ChatMessage(role="system", content="Reponds simplement: OK")],
        temperature=0.0,
        top_p=1.0,
        num_predict=8,
    )

    ner = None
    if args.ner:
        try:
            ner = PiiNer(model=args.ner_model, mode=args.ner_mode)
            print(f"[INFO] NER actif: {args.ner_model} mode={args.ner_mode}")
        except Exception as e:
            print(f"[WARN] NER indisponible: {type(e).__name__}")

    llm_label = f"{provider}/{model}"
    print_header(cfg.app_name, scenario.context_label, scenario.context_description, llm_label)
    logger.log(f"SESSION_START scenario={scenario.name}")

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
            for key, sc in cfg.scenarios.items():
                marker = "*" if key == scenario.name else " "
                print(f" {marker} {key:12s} -> {sc.context_label}")
            print("Pour changer, tape: /scenario <nom>")
            continue

        if user_raw.startswith("/scenario "):
            wanted = user_raw.split(" ", 1)[1].strip()
            if wanted in cfg.scenarios:
                scenario = cfg.scenarios[wanted]
                print_header(cfg.app_name, scenario.context_label, scenario.context_description, llm_label)
                logger.log(f"SCENARIO_SWITCH to={wanted}")
            else:
                print(f"[WARN] Scenario inconnu: {wanted}")
            continue

        # 1) Pre-check user (pedagogical warning)
        t0 = time.perf_counter()
        warning = precheck_user_input(user_raw)
        t1 = time.perf_counter()
        if warning:
            print(f"[SYSTEM] {warning}")
            logger.log(f"WARNING {warning}")

        # 2) Redaction input
        red = redact_pii(user_raw, ner=ner)
        t2 = time.perf_counter()
        if red.detected:
            print(f"[SECURITE] PII detectee {red.types} -> masquage automatique.")
        user = red.redacted_text

        # 3) Style routing
        profile = detect_style(user)
        profile = style_memory.update(profile)
        t3 = time.perf_counter()
        logger.log(
            f"STYLE profile={profile.style} address={profile.address} verbosity={profile.verbosity}"
        )
        system_prompt = build_system_prompt(
            scenario.context_label, scenario.context_description, profile
        )
        logger.log("SYSTEM_PROMPT " + system_prompt)

        # Logging securise (jamais brut)
        logger.log(f"USER {user}")

        # 4) Construire messages pour le LLM
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user),
        ]

        # 5) Appel LLM (local)
        bot_raw = client.chat(
            messages=messages,
            temperature=cfg.llm.temperature,
            top_p=cfg.llm.top_p,
            num_predict=cfg.llm.max_tokens,
        )
        t4 = time.perf_counter()

        # 6) Policy post-check
        bot_safe = postcheck_llm_output(bot_raw)
        t5 = time.perf_counter()

        # 7) Defense en profondeur: redaction output
        bot_red = redact_pii(bot_safe, ner=ner).redacted_text
        t6 = time.perf_counter()
        print(f"BOT> {bot_red}")
        logger.log(f"BOT {bot_red}")

        if args.profiling:
            print(
                "[PROFILE] precheck={:.0f}ms redact_in={:.0f}ms style={:.0f}ms llm={:.0f}ms postcheck={:.0f}ms redact_out={:.0f}ms total={:.0f}ms".format(
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
