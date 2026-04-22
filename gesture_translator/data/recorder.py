from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from gesture_translator.config import DUAL_HAND_LABEL_IDS, FACE_AREA_LABEL_IDS, MAX_SEQUENCE_LENGTH, MIN_SEQUENCE_LENGTH
from gesture_translator.core.feature_builder import build_sequence_observations


def summarize_sequence(frames: list[dict]) -> dict:
    total = max(len(frames), 1)
    left_hand = sum(1 for frame in frames if frame.get("quality", {}).get("has_left_hand"))
    right_hand = sum(1 for frame in frames if frame.get("quality", {}).get("has_right_hand"))
    face = sum(1 for frame in frames if frame.get("quality", {}).get("has_face"))
    pose = sum(1 for frame in frames if frame.get("quality", {}).get("has_pose"))
    hand = sum(
        1
        for frame in frames
        if frame.get("quality", {}).get("has_left_hand") or frame.get("quality", {}).get("has_right_hand")
    )
    two_hands = sum(
        1
        for frame in frames
        if frame.get("quality", {}).get("has_left_hand") and frame.get("quality", {}).get("has_right_hand")
    )
    hand_count_total = sum(
        int(bool(frame.get("quality", {}).get("has_left_hand"))) + int(bool(frame.get("quality", {}).get("has_right_hand")))
        for frame in frames
    )

    quality_scores = []
    for frame in frames:
        quality = frame.get("quality", {})
        hand_score = 1.0 if (quality.get("has_left_hand") or quality.get("has_right_hand")) else 0.0
        face_score = 1.0 if quality.get("has_face") else 0.0
        pose_score = 1.0 if quality.get("has_pose") else 0.0
        quality_scores.append(hand_score * 0.5 + face_score * 0.3 + pose_score * 0.2)

    return {
        "frame_count": len(frames),
        "hand_ratio": round(hand / total, 4),
        "two_hands_ratio": round(two_hands / total, 4),
        "face_ratio": round(face / total, 4),
        "pose_ratio": round(pose / total, 4),
        "left_hand_ratio": round(left_hand / total, 4),
        "right_hand_ratio": round(right_hand / total, 4),
        "average_hands": round(hand_count_total / total, 4),
        "tracking_quality": round(sum(quality_scores) / total, 4),
    }


def infer_dominant_zone(frames: list[dict]) -> str:
    observations = build_sequence_observations(frames)
    if not observations:
        return "none"

    zone_counts = Counter()
    for observation in observations:
        dominant = observation.get("dominant", {})
        candidates = {
            "mustache": float(dominant.get("mustache_distance", 9.0)),
            "forehead": float(dominant.get("forehead_distance", 9.0)),
            "cheek": float(dominant.get("cheek_distance", 9.0)),
            "jaw": float(dominant.get("jaw_distance", 9.0)),
            "chin": float(dominant.get("chin_distance", 9.0)),
            "lips": float(dominant.get("mouth_distance", 9.0)),
            "chest": float(dominant.get("chest_distance", 9.0)),
        }
        zone, score = min(candidates.items(), key=lambda item: item[1])
        zone_counts[zone if score <= 1.2 else "space"] += 1

    return zone_counts.most_common(1)[0][0] if zone_counts else "none"


def _compute_frame_valid_ratio(summary: dict, label: str) -> float:
    if label in DUAL_HAND_LABEL_IDS:
        return min(summary["hand_ratio"], summary["two_hands_ratio"])
    if label in FACE_AREA_LABEL_IDS:
        return min(summary["hand_ratio"], summary["face_ratio"])
    return summary["hand_ratio"]


def validate_sequence(frames: list[dict], label: str) -> dict:
    summary = summarize_sequence(frames)
    normalized_label = str(label).strip()

    if summary["frame_count"] < MIN_SEQUENCE_LENGTH:
        return {"ok": False, "reason": "too few frames", "summary": summary}
    if summary["frame_count"] > MAX_SEQUENCE_LENGTH:
        return {"ok": False, "reason": "too many frames", "summary": summary}
    if summary["hand_ratio"] < 0.6:
        return {"ok": False, "reason": "hand not visible long enough", "summary": summary}
    if normalized_label in FACE_AREA_LABEL_IDS and summary["face_ratio"] < 0.55:
        return {"ok": False, "reason": "face not visible enough", "summary": summary}
    if normalized_label in DUAL_HAND_LABEL_IDS and summary["two_hands_ratio"] < 0.45:
        return {"ok": False, "reason": "two hands not visible enough", "summary": summary}
    if summary["pose_ratio"] < 0.3:
        return {"ok": False, "reason": "upper body pose not visible enough", "summary": summary}

    frame_valid_ratio = _compute_frame_valid_ratio(summary, normalized_label)
    if frame_valid_ratio < 0.55:
        return {"ok": False, "reason": "frame validity is too low", "summary": summary}

    return {"ok": True, "reason": "", "summary": summary}


def next_sample_path(label: str, output_dir: Path) -> Path:
    label_dir = output_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)
    next_index = len(list(label_dir.glob("*.json"))) + 1
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return label_dir / f"{label}_{timestamp}_{next_index:04d}.json"


def save_sequence_sample(label: str, frames: list[dict], output_dir: Path, meta: dict | None = None) -> Path:
    verdict = validate_sequence(frames, label)
    if not verdict["ok"]:
        raise ValueError(verdict["reason"])

    summary = verdict["summary"]
    normalized_label = str(label).strip()
    path = next_sample_path(normalized_label, output_dir)
    payload = {
        "label": normalized_label,
        "meta": {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "source": "gesture_translator.record",
            "hands_count": summary["average_hands"],
            "dominant_zone": infer_dominant_zone(frames),
            "tracking_quality": summary["tracking_quality"],
            "frame_valid_ratio": round(_compute_frame_valid_ratio(summary, normalized_label), 4),
            **(meta or {}),
        },
        "summary": summary,
        "sequence": frames,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
