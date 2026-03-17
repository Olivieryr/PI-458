from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from core.response_planner import PlannedAttackResponse, ResponsePlanner
from core.scoring import compute_score


STATE_S0 = "S0_INITIAL_CONTACT"
STATE_S1 = "S1_BUILD_TRUST"
STATE_S2 = "S2_CREATE_URGENCY"
STATE_S3 = "S3_ATTEMPT_COLLECT"
STATE_S4 = "S4_DEBRIEF_END"


@dataclass(frozen=True)
class ScenarioDefinition:
    name: str
    context_label: str
    context_description: str
    states: Dict[str, Dict[str, str]]
    clarification_question: str
    debrief: Dict[str, Dict[str, str]]


@dataclass
class DialogState:
    state: str = STATE_S0
    slots: Dict[str, str] = field(default_factory=dict)
    collected_info: List[str] = field(default_factory=list)
    target_compliance_level: str = "low"


def load_scenario(path: str) -> ScenarioDefinition:
    # Resolve relative paths robustly: allow running from repo root or package dir.
    p = Path(path)
    if not p.is_absolute():
        candidates = [
            Path.cwd() / p,
            Path(__file__).resolve().parents[1] / p,  # social_engineering_simulator/
        ]
        for c in candidates:
            if c.exists():
                p = c
                break

    if not p.exists():
        raise FileNotFoundError(
            f"Scenario file not found: {path}. Tried: "
            + ", ".join(str(x) for x in ([p] if p.is_absolute() else candidates))
        )

    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return ScenarioDefinition(
        name=data.get("name", "scenario"),
        context_label=data.get("context_label", "Contexte"),
        context_description=data.get("context_description", ""),
        states=data.get("states", {}),
        clarification_question=data.get("clarification_question", "Pouvez-vous preciser ?"),
        debrief=data.get("debrief", {}),
    )


class DialogManager:
    def __init__(self, scenario: ScenarioDefinition, confidence_threshold: float = 0.55):
        self.scenario = scenario
        self.planner = ResponsePlanner(
            {
                "states": scenario.states,
                "clarification_question": scenario.clarification_question,
                "debrief": scenario.debrief,
            }
        )
        self.state = DialogState()
        self.confidence_threshold = confidence_threshold

    def step(
        self,
        intent: str,
        confidence: float,
        slots: Dict[str, str],
        shared_sensitive_attempt: bool,
    ) -> PlannedAttackResponse:
        if confidence < self.confidence_threshold:
            intent = "OTHER"

        self._update_slots(slots, shared_sensitive_attempt, intent)

        if intent == "OTHER":
            return self.planner.clarification(self.state.state, self.state.state)

        next_state, success = self._transition(self.state.state, intent)
        if next_state == STATE_S4:
            score = compute_score(self.state.collected_info, self.state.target_compliance_level, success)
            self.state.state = next_state
            return self.planner.debrief(success=success, score=score)

        planned = self.planner.plan_for_state(self.state.state, self.state.slots, next_state)
        self.state.state = next_state
        return planned

    def _transition(self, state: str, intent: str) -> Tuple[str, bool]:
        if intent in {"SUSPICIOUS", "REFUSES_TO_SHARE"}:
            return STATE_S4, False
        if state == STATE_S0 and intent in {"PROVIDES_CONTEXT", "REPORTS_PROBLEM"}:
            return STATE_S1, False
        if state == STATE_S1 and intent == "COMPLIES":
            return STATE_S2, False
        if state == STATE_S2 and intent in {"EXPRESSES_CONCERN", "ASKS_WHAT_TO_DO"}:
            return STATE_S3, False
        if state == STATE_S3 and intent == "COMPLIES_WITH_RISK":
            return STATE_S4, True
        return state, False

    def _update_slots(
        self,
        slots: Dict[str, str],
        shared_sensitive_attempt: bool,
        intent: str,
    ) -> None:
        for key in ("target_service", "target_device", "urgency_level"):
            if key in slots and slots[key]:
                self.state.slots[key] = slots[key]

        if intent in {"COMPLIES", "COMPLIES_WITH_RISK"}:
            self.state.target_compliance_level = "high"
        elif intent in {"EXPRESSES_CONCERN"}:
            self.state.target_compliance_level = "medium"

        if shared_sensitive_attempt and "password_attempt" not in self.state.collected_info:
            self.state.collected_info.append("password_attempt")
