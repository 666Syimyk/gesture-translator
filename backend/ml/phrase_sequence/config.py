import copy
import json
from pathlib import Path


PHRASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PHRASE_DIR / "default_config.json"

BUILTIN_DEFAULT_CONFIG = {
    "phrases": [
        "Дом",
        "Мужчина",
        "Женщина",
        "Нет",
        "Да",
        "Солнце",
        "Дружба",
        "Привет",
        "Пока",
        "hello",
        "thanks",
        "yes",
        "no",
        "help",
        "water",
        "repeat",
        "stop",
        "understand",
        "dont_understand",
        "none",
    ],
    "dataset_dir": "uploads/datasets/phrase_sequence/latest",
    "artifacts_dir": "ml/artifacts/phrase_sequence/latest",
    "recognition_level": "phrase",
    "feature_mode": "full",
    "sequence_length": 32,
    "clip_seconds": 2.0,
    "capture_fps": 15,
    "quality": {
        "min_frame_count": 8,
        "min_valid_frame_ratio": 0.2,
        "min_hand_frame_ratio": 0.2,
        "reject_empty_non_idle": True,
    },
    "augmentation": {
        "enabled": True,
        "copies_per_sample": 1,
        "coordinate_noise_std": 0.01,
        "scale_jitter_std": 0.03,
        "shift_jitter_std": 0.015,
        "frame_drop_probability": 0.05,
    },
    "model": {
        "type": "lstm",
        "hidden_size": 128,
        "epochs": 80,
        "batch_size": 16,
        "learning_rate": 0.001,
        "early_stopping_patience": 8,
        "seed": 42,
    },
    "inference": {
        "camera_index": 0,
        "confidence_threshold": 0.65,
        "smoothing_window": 5,
        "stable_votes_required": 3,
        "prediction_log_size": 10,
        "min_window_frames": 20,
        "cooldown_seconds": 1.2,
        "idle_label": "none",
        "speak": True,
    },
}


def deep_merge(base, override):
    result = copy.deepcopy(base)

    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def load_config(config_path=None):
    config = copy.deepcopy(BUILTIN_DEFAULT_CONFIG)
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if path.exists():
        with path.open("r", encoding="utf-8-sig") as input_file:
            config = deep_merge(config, json.load(input_file))

    config["phrases"] = [
        str(label).strip()
        for label in config.get("phrases", [])
        if str(label).strip()
    ]

    if "none" not in config["phrases"]:
        config["phrases"].append("none")

    return config


def resolve_backend_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path

    return (BACKEND_DIR / path).resolve()


def save_default_config(output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(load_config(DEFAULT_CONFIG_PATH), output_file, ensure_ascii=False, indent=2)
    return path
