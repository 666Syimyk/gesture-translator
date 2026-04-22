from __future__ import annotations

import json
from pathlib import Path

import torch

from gesture_translator.core.feature_builder import build_sequence_tensor
from gesture_translator.models.gesture_classifier import GestureClassifier
from gesture_translator.rules.spatial_rules import score_spatial_rules
from gesture_translator.rules.temporal_rules import score_temporal_rules


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def load_model_bundle(artifact_dir: Path | str) -> dict | None:
    artifact_path = Path(artifact_dir)
    metadata_path = artifact_path / "metadata.json"
    model_path = artifact_path / "sequence_model.pt"
    if not metadata_path.exists() or not model_path.exists():
        return None

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model = GestureClassifier(
        input_size=int(metadata["input_size"]),
        num_classes=len(metadata["labels"]),
        hidden_size=int(metadata.get("hidden_size", 128)),
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return {"metadata": metadata, "model": model}


RULE_MIN_BY_LABEL = {
    "privet": 0.16,
    "poka": 0.18,
    "ya": 0.2,
    "ty": 0.2,
    "muzhchina": 0.14,
    "zhenshchina": 0.14,
    "spasibo": 0.24,
    "bolshoy": 0.2,
    "est": 0.24,
    "krasivyy": 0.1,
    "malenkiy": 0.1,
}

MODEL_WEIGHT_BY_LABEL = {
    "muzhchina": 0.5,
    "zhenshchina": 0.5,
    "krasivyy": 0.68,
    "malenkiy": 0.68,
}


def _strict_gate_results(spatial_debug: dict, temporal_debug: dict) -> tuple[dict[str, bool], dict[str, str]]:
    zone = spatial_debug.get("zone", {}) if isinstance(spatial_debug, dict) else {}
    motion = temporal_debug.get("motion", {}) if isinstance(temporal_debug, dict) else {}

    gates = {
        "muzhchina": bool(zone.get("spatial_gate_muzhchina")) and bool(motion.get("temporal_gate_muzhchina")),
        "zhenshchina": bool(zone.get("spatial_gate_zhenshchina")) and bool(motion.get("temporal_gate_zhenshchina")),
        "spasibo": bool(zone.get("spatial_gate_spasibo")) and bool(motion.get("temporal_gate_spasibo")),
        "est": bool(zone.get("spatial_gate_est")) and bool(motion.get("temporal_gate_est")),
    }
    reasons = {}
    for label, passed in gates.items():
        if passed:
            continue
        if label == "muzhchina":
            reasons[label] = "rejected: require two_fingertips + strict upper_lip_zone"
        elif label == "zhenshchina":
            reasons[label] = "rejected: require open_palm + cheek_jaw_path"
        elif label == "spasibo":
            reasons[label] = "rejected: require fist + chin->forehead sequence"
        elif label == "est":
            reasons[label] = "rejected: require all_fingertips_together + lips_zone"
    return gates, reasons


class HybridInferenceEngine:
    def __init__(self, artifact_dir: str | None, sequence_length: int = 32, confidence_threshold: float = 0.65):
        self.bundle = load_model_bundle(artifact_dir) if artifact_dir else None
        self.sequence_length = int(sequence_length)
        self.confidence_threshold = float(confidence_threshold)
        self.mode_name = "hybrid" if self.bundle else "rules-only"

    def _model_probabilities(self, frames: list[dict]) -> dict[str, float]:
        if not self.bundle:
            return {}

        tensor = build_sequence_tensor(frames, sequence_length=self.sequence_length)
        inputs = torch.tensor(tensor[None, ...], dtype=torch.float32)
        with torch.no_grad():
            logits = self.bundle["model"](inputs)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

        labels = self.bundle["metadata"]["labels"]
        return {label: float(probabilities[index]) for index, label in enumerate(labels)}

    def predict(self, frames: list[dict]) -> dict:
        spatial_scores, spatial_debug = score_spatial_rules(frames)
        temporal_scores, temporal_debug = score_temporal_rules(frames)
        model_scores = self._model_probabilities(frames)
        strict_gates, reject_reasons = _strict_gate_results(spatial_debug, temporal_debug)

        label_space = set(spatial_scores) | set(temporal_scores) | set(model_scores) | {"none"}
        final_scores = {}
        for label in label_space:
            spatial = float(spatial_scores.get(label, 0.0))
            temporal = float(temporal_scores.get(label, 0.0))
            rules_score = clamp01(spatial * 0.52 + temporal * 0.48)
            model = float(model_scores.get(label, 0.0))

            if self.bundle:
                model_weight = float(MODEL_WEIGHT_BY_LABEL.get(label, 0.6))
                rules_weight = 1.0 - model_weight
                score = clamp01(model * model_weight + rules_score * rules_weight)
                min_rules = float(RULE_MIN_BY_LABEL.get(label, 0.08))
                if label != "none" and rules_score < min_rules:
                    score *= 0.74
            else:
                score = rules_score

            if label in strict_gates and not strict_gates[label]:
                score = 0.0

            final_scores[label] = score

        ranked_final = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)
        label, confidence = ranked_final[0]
        competing_label, competing_score = ranked_final[1] if len(ranked_final) > 1 else ("none", 0.0)
        reason_for_reject = reject_reasons.get(label)
        if label != "none" and confidence < self.confidence_threshold:
            reason_for_reject = reason_for_reject or f"low_confidence:{label}"
            label = "none"
        if label == "none" and ranked_final and ranked_final[0][0] != "none":
            candidate = ranked_final[0][0]
            reason_for_reject = reason_for_reject or reject_reasons.get(candidate) or f"strict_none:{candidate}"

        return {
            "label": label,
            "confidence": float(final_scores.get(label, confidence)),
            "scores": final_scores,
            "debug": {
                "spatial_top": sorted(spatial_scores.items(), key=lambda item: item[1], reverse=True)[:3],
                "temporal_top": sorted(temporal_scores.items(), key=lambda item: item[1], reverse=True)[:3],
                "model_top": sorted(model_scores.items(), key=lambda item: item[1], reverse=True)[:3],
                "strict_gates": strict_gates,
                "reason_for_reject": reason_for_reject,
                "competing_label": competing_label,
                "competing_score": round(float(competing_score), 4),
                "top_3_predictions": [{"label": name, "score": round(float(score), 4)} for name, score in ranked_final[:3]],
                **spatial_debug,
                **temporal_debug,
            },
        }
