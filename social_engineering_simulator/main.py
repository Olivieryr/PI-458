from datetime import datetime
from config import load_config

from core.llm_client import ChatMessage, OllamaClient
from core.pii_detector import redact_pii
from core.policy_guard import postcheck_llm_output, precheck_user_input
from core.safe_log import SafeLogger
from core.style_router import build_system_prompt, detect_style


def print_header(app_name: str, scenario_label: str, scenario_desc: str) -> None:
    print("=" * 72)
    print(f"{app_name}")
    print(f"Mode: CLI | Scenario: {scenario_label}")
    print(f"Contexte: {scenario_desc}")
    print("-" * 72)
    print("Commandes: /exit  /help  /scenario")
    print("=" * 72)


def main() -> int:
    cfg = load_config()
    scenario = cfg.scenarios[cfg.scenario_default]
    logger = SafeLogger(enabled=cfg.log_to_file, file_path=cfg.log_file_path)
    client = OllamaClient(
        base_url=cfg.llm.base_url,
        model=cfg.llm.model,
        timeout_s=cfg.llm.timeout_s,
    )
    _ = client.chat(
        messages=[ChatMessage(role="system", content="Reponds simplement: OK")],
        temperature=0.0,
        top_p=1.0,
        num_predict=8,
    )

    print_header(cfg.app_name, scenario.context_label, scenario.context_description)
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
                print_header(cfg.app_name, scenario.context_label, scenario.context_description)
                logger.log(f"SCENARIO_SWITCH to={wanted}")
            else:
                print(f"[WARN] Scenario inconnu: {wanted}")
            continue

        # 1) Pre-check user (pedagogical warning)
        warning = precheck_user_input(user_raw)
        if warning:
            print(f"[SYSTEM] {warning}")
            logger.log(f"WARNING {warning}")

        # 2) Redaction input
        red = redact_pii(user_raw)
        if red.detected:
            print(f"[SECURITE] PII detectee {red.types} -> masquage automatique.")
        user = red.redacted_text

        # 3) Style routing
        profile = detect_style(user)
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

        # 6) Policy post-check
        bot_safe = postcheck_llm_output(bot_raw)

        # 7) Defense en profondeur: redaction output
        bot_red = redact_pii(bot_safe).redacted_text
        print(f"BOT> {bot_red}")
        logger.log(f"BOT {bot_red}")

    # unreachable


if __name__ == "__main__":
    raise SystemExit(main())
