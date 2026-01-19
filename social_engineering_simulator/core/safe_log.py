import os
from dataclasses import dataclass
from datetime import datetime

from core.pii_detector import redact_pii


@dataclass
class SafeLogger:
    enabled: bool
    file_path: str

    def log(self, message: str) -> None:
        """
        Log securise: redaction systematique avant ecriture.
        """
        if not self.enabled:
            return

        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

        safe = redact_pii(message).redacted_text
        ts = datetime.now().isoformat(timespec="seconds")
        line = f"{ts} | {safe}\n"

        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(line)
