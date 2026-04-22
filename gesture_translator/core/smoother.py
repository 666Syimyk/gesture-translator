from __future__ import annotations

from collections import Counter, deque


class PredictionSmoother:
    def __init__(self, window_size: int = 5, stable_votes_required: int = 3) -> None:
        self.window_size = max(1, int(window_size))
        self.stable_votes_required = max(1, int(stable_votes_required))
        self.labels = deque(maxlen=self.window_size)
        self.payloads = deque(maxlen=self.window_size)

    def update(self, label: str, confidence: float, extra: dict | None = None):
        normalized = str(label or "none").strip() or "none"
        payload = {
            "label": normalized,
            "confidence": float(confidence or 0.0),
            **(extra or {}),
        }
        self.labels.append(normalized)
        self.payloads.append(payload)

        votes = Counter(self.labels)
        best_label, best_votes = votes.most_common(1)[0]
        if best_label == "none" or best_votes < self.stable_votes_required:
            return None

        matching = [item for item in self.payloads if item["label"] == best_label]
        confidence_avg = sum(item["confidence"] for item in matching) / max(len(matching), 1)
        best_payload = max(matching, key=lambda item: item["confidence"])
        return {
            **best_payload,
            "label": best_label,
            "confidence": confidence_avg,
            "votes": best_votes,
        }
