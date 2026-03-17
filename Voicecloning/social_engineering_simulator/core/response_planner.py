from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class PlannedAttackResponse:
    objective: str
    approach: str
    key_message: str
    question: str
    next_state: str


class ResponsePlanner:
    def __init__(self, scenario: Dict[str, Dict[str, str]]):
        self.scenario = scenario

    def plan_for_state(
        self,
        state: str,
        slots: Dict[str, str],
        next_state: str,
    ) -> PlannedAttackResponse:
        template = self.scenario["states"].get(state, {})
        objective = template.get("goal", "Avancer dans la simulation")
        approach = template.get("social_tactic", "Autorite")
        key_message = self._inject_context(template.get("key_message", ""), slots)
        question = self._build_question(state, template.get("question", ""), slots)
        return PlannedAttackResponse(
            objective=objective,
            approach=approach,
            key_message=key_message,
            question=question,
            next_state=next_state,
        )

    def clarification(self, state: str, next_state: str) -> PlannedAttackResponse:
        question = self.scenario.get("clarification_question", "Pouvez-vous preciser ?")
        return PlannedAttackResponse(
            objective="Clarifier la situation",
            approach="Neutralite",
            key_message="Je veux m'assurer de bien comprendre.",
            question=question,
            next_state=next_state,
        )

    def debrief(self, success: bool, score: int) -> PlannedAttackResponse:
        debrief = self.scenario["debrief"]["success" if success else "failure"]
        message = debrief.get("key_message", "Fin de simulation.")
        advice = debrief.get("advice", "")
        if advice:
            message = f"{message} {advice}"
        message = f"{message} Score: {score}/100."
        return PlannedAttackResponse(
            objective="Debrief pedagogique",
            approach="Pedagogie",
            key_message=message,
            question=debrief.get("question", ""),
            next_state="S4_DEBRIEF_END",
        )

    def _inject_context(self, text: str, slots: Dict[str, str]) -> str:
        if not text:
            return text
        service = slots.get("target_service") or "votre service"
        device = slots.get("target_device") or "votre poste"
        return text.replace("{target_service}", service).replace("{target_device}", device)

    def _build_question(self, state: str, default: str, slots: Dict[str, str]) -> str:
        if state == "S1_BUILD_TRUST":
            if slots.get("target_service"):
                return "Vous etes plutot sur un laptop, un desktop, ou un telephone ?"
            return "Pouvez-vous me confirmer votre service et votre type de poste ?"
        if state == "S2_CREATE_URGENCY":
            return default or "Pouvez-vous me dire ce que vous voyez de votre cote ?"
        if state == "S3_ATTEMPT_COLLECT":
            return default or "Pour verification, pouvez-vous me confirmer le code affiche ?"
        return default
