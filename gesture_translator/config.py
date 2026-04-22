from __future__ import annotations

import json
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = PACKAGE_DIR / "runtime"
DATASET_DIR = RUNTIME_DIR / "datasets" / "default"
ARTIFACTS_DIR = RUNTIME_DIR / "artifacts" / "default"
MODEL_CACHE_DIR = RUNTIME_DIR / "models"
LABELS_PATH = PACKAGE_DIR / "data" / "labels.json"

SEQUENCE_LENGTH = 32
MIN_SEQUENCE_LENGTH = 20
MAX_SEQUENCE_LENGTH = 40
CONFIDENCE_THRESHOLD = 0.6
SMOOTHING_WINDOW = 3
STABLE_VOTES_REQUIRED = 2
COOLDOWN_SECONDS = 1.2

HAND_LANDMARKS = list(range(21))
FACE_LANDMARKS = [1, 33, 61, 152, 199, 263, 291]
POSE_LANDMARKS = [0, 11, 12, 13, 14, 15, 16]

FACE_AREA_LABEL_IDS = {"muzhchina", "zhenshchina", "krasivyy", "spasibo", "est"}
DUAL_HAND_LABEL_IDS = {"bolshoy"}
# Backward-compatible aliases.
FACE_AREA_LABELS = FACE_AREA_LABEL_IDS
DUAL_HAND_LABELS = DUAL_HAND_LABEL_IDS

DEFAULT_CONFIG = {
    "camera_index": 0,
    "mirror_preview": True,
    "sequence_length": SEQUENCE_LENGTH,
    "min_sequence_length": MIN_SEQUENCE_LENGTH,
    "confidence_threshold": CONFIDENCE_THRESHOLD,
    "smoothing_window": SMOOTHING_WINDOW,
    "stable_votes_required": STABLE_VOTES_REQUIRED,
    "cooldown_seconds": COOLDOWN_SECONDS,
    "dataset_dir": str(DATASET_DIR),
    "artifacts_dir": str(ARTIFACTS_DIR),
    "labels_path": str(LABELS_PATH),
}


def ensure_runtime_dirs() -> None:
    for path in (RUNTIME_DIR, DATASET_DIR, ARTIFACTS_DIR, MODEL_CACHE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_label_entries(labels_path: Path | str = LABELS_PATH) -> list[dict]:
    payload = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    labels = payload.get("labels", payload)
    if isinstance(labels, dict):
        labels = [labels]

    entries: list[dict] = []
    for label in labels:
        if isinstance(label, dict):
            entry = {
                "id": str(label.get("id", "")).strip(),
                "display_ru": str(label.get("display_ru", label.get("label_ru", label.get("label", "")))).strip(),
                **{key: value for key, value in label.items() if key not in {"id", "display_ru", "label_ru", "label"}},
            }
            entries.append(entry)
        else:
            label_name = str(label).strip()
            entries.append({"id": label_name, "display_ru": label_name})

    return [entry for entry in entries if entry.get("id") and entry.get("display_ru")]


def load_labels(labels_path: Path | str = LABELS_PATH) -> list[str]:
    return [entry["id"] for entry in load_label_entries(labels_path)]


def normalize_label(label: str | None, labels_path: Path | str = LABELS_PATH) -> str:
    normalized = str(label or "").strip()
    if not normalized:
        return ""

    lowered = normalized.lower()
    for entry in load_label_entries(labels_path):
        entry_id = str(entry.get("id", "")).strip()
        entry_display = str(entry.get("display_ru", "")).strip()
        if lowered == entry_id.lower() or lowered == entry_display.lower():
            return entry_id

    return normalized


def label_display_map(labels_path: Path | str = LABELS_PATH) -> dict[str, str]:
    return {
        str(entry.get("id", "")).strip(): str(entry.get("display_ru", entry.get("id", ""))).strip()
        for entry in load_label_entries(labels_path)
        if str(entry.get("id", "")).strip()
    }


def load_runtime_config(overrides: dict | None = None) -> dict:
    ensure_runtime_dirs()
    config = dict(DEFAULT_CONFIG)
    if overrides:
        config.update({key: value for key, value in overrides.items() if value is not None})
    return config
