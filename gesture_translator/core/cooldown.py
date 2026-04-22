from __future__ import annotations

import time


class CooldownGate:
    def __init__(self, cooldown_seconds: float = 1.2) -> None:
        self.cooldown_seconds = float(cooldown_seconds)
        self._last_emitted = {}

    def allow(self, label: str) -> bool:
        normalized = str(label or "").strip()
        if not normalized or normalized == "none":
            return False

        now = time.perf_counter()
        last = float(self._last_emitted.get(normalized, 0.0))
        if now - last < self.cooldown_seconds:
            return False

        self._last_emitted[normalized] = now
        return True
