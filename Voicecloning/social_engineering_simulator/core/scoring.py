from typing import List


def compute_score(collected_info: List[str], compliance_level: str, success: bool) -> int:
    score = 60 if not success else 30
    if compliance_level == "high":
        score -= 15
    elif compliance_level == "medium":
        score -= 5

    if "password_attempt" in collected_info:
        score -= 20

    if not success:
        score += 15

    return max(0, min(100, score))
