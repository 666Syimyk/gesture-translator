import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_sequence_model import (
    build_sequence_feature,
    build_sequence_tensors,
    create_torch_model,
    normalize_class_entry,
    predict_probabilities_centroid,
    predict_probabilities_torch,
    read_json,
)


def get_class_entries(metadata):
    if metadata.get("class_entries"):
        return [normalize_class_entry(entry) for entry in metadata["class_entries"]]

    return [
        normalize_class_entry(
            {
                "key": f"phrase::{label}",
                "text": label,
                "recognition_level": "phrase",
                "unit_code": None,
            }
        )
        for label in metadata.get("labels", [])
    ]


def resolve_model_dir(model_dir, member_path):
    path = Path(member_path)
    if path.is_absolute():
        return path

    return (model_dir / path).resolve()


def align_probabilities(probabilities, source_entries, target_entries):
    source_index = {
        entry["key"]: index for index, entry in enumerate(source_entries)
    }
    aligned = np.zeros((len(target_entries),), dtype=np.float32)

    for index, entry in enumerate(target_entries):
        source_position = source_index.get(entry["key"])
        if source_position is not None:
            aligned[index] = float(probabilities[source_position])

    total = float(aligned.sum())
    if total > 0:
        aligned /= total

    return aligned


def load_model_bundle(model_dir, _cache=None):
    resolved_model_dir = Path(model_dir).resolve()
    cache = _cache if _cache is not None else {}
    cache_key = str(resolved_model_dir).lower()
    if cache_key in cache:
        cached_bundle = cache[cache_key]
        if cached_bundle.get("_is_loading"):
            raise SystemExit(
                f"Model metadata contains a recursive reference involving {resolved_model_dir}."
            )
        return cached_bundle

    metadata = read_json(resolved_model_dir / "metadata.json")
    model_type = metadata.get("model_type", "baseline")
    feature_mode = metadata.get("feature_mode", "full")
    class_entries = get_class_entries(metadata)
    bundle = {
        "model_dir": resolved_model_dir,
        "metadata": metadata,
        "model_type": model_type,
        "feature_mode": feature_mode,
        "class_entries": class_entries,
        "_is_loading": True,
    }
    cache[cache_key] = bundle

    if model_type == "ensemble":
        members = metadata.get("members", [])
        if not members:
            raise SystemExit("Ensemble model metadata does not contain members.")

        bundle["members"] = [
            {
                "weight": float(member.get("weight", 1.0) or 1.0),
                "bundle": load_model_bundle(
                    resolve_model_dir(resolved_model_dir, member["model_dir"]),
                    _cache=cache,
                ),
            }
            for member in members
        ]
        bundle.pop("_is_loading", None)
        return bundle

    if model_type == "hybrid_gate":
        primary = metadata.get("primary") or {}
        fallback = metadata.get("fallback") or {}

        primary_model_dir = primary.get("model_dir")
        fallback_model_dir = fallback.get("model_dir")
        if not primary_model_dir or not fallback_model_dir:
            raise SystemExit("Hybrid gate metadata does not contain primary/fallback models.")

        bundle["gated_label_keys"] = {
            str(value).strip()
            for value in metadata.get("gated_label_keys", [])
            if str(value).strip()
        }
        bundle["primary"] = {
            "bundle": load_model_bundle(
                resolve_model_dir(resolved_model_dir, primary_model_dir),
                _cache=cache,
            ),
        }
        bundle["fallback"] = {
            "bundle": load_model_bundle(
                resolve_model_dir(resolved_model_dir, fallback_model_dir),
                _cache=cache,
            ),
        }
        bundle.pop("_is_loading", None)
        return bundle

    if model_type == "hybrid_gate_map":
        primary = metadata.get("primary") or {}
        fallback_map = metadata.get("label_fallback_map") or {}
        fallbacks = metadata.get("fallbacks") or {}

        primary_model_dir = primary.get("model_dir")
        if not primary_model_dir:
            raise SystemExit("Hybrid gate map metadata does not contain a primary model.")
        if not fallback_map or not fallbacks:
            raise SystemExit(
                "Hybrid gate map metadata does not contain fallbacks or label mapping."
            )

        bundle["primary"] = {
            "bundle": load_model_bundle(
                resolve_model_dir(resolved_model_dir, primary_model_dir),
                _cache=cache,
            ),
        }
        bundle["fallbacks"] = {}
        for alias, fallback in fallbacks.items():
            fallback_model_dir = (fallback or {}).get("model_dir")
            if not fallback_model_dir:
                raise SystemExit(
                    f"Hybrid gate map fallback '{alias}' does not contain model_dir."
                )
            bundle["fallbacks"][str(alias)] = {
                "bundle": load_model_bundle(
                    resolve_model_dir(resolved_model_dir, fallback_model_dir),
                    _cache=cache,
                ),
            }

        bundle["label_fallback_map"] = {}
        for label_key, alias in fallback_map.items():
            normalized_label_key = str(label_key).strip()
            normalized_alias = str(alias).strip()
            if not normalized_label_key or not normalized_alias:
                continue
            if normalized_alias not in bundle["fallbacks"]:
                raise SystemExit(
                    f"Hybrid gate map label '{normalized_label_key}' references unknown fallback '{normalized_alias}'."
                )
            bundle["label_fallback_map"][normalized_label_key] = normalized_alias
        bundle.pop("_is_loading", None)
        return bundle

    if model_type == "hybrid_confidence_gate":
        primary = metadata.get("primary") or {}
        fallback = metadata.get("fallback") or {}

        primary_model_dir = primary.get("model_dir")
        fallback_model_dir = fallback.get("model_dir")
        if not primary_model_dir or not fallback_model_dir:
            raise SystemExit(
                "Hybrid confidence gate metadata does not contain primary/fallback models."
            )

        bundle["gated_label_keys"] = {
            str(value).strip()
            for value in metadata.get("gated_label_keys", [])
            if str(value).strip()
        }
        bundle["preferred_fallback_label_keys"] = {
            str(value).strip()
            for value in metadata.get("preferred_fallback_label_keys", [])
            if str(value).strip()
        }
        bundle["primary_confidence_threshold"] = float(
            metadata.get(
                "primary_confidence_threshold",
                metadata.get("confidence_threshold", 0.5),
            )
        )
        fallback_max_sequence_length = metadata.get("fallback_max_sequence_length")
        bundle["fallback_max_sequence_length"] = (
            int(fallback_max_sequence_length)
            if fallback_max_sequence_length is not None
            else None
        )
        fallback_max_both_hands_frames = metadata.get("fallback_max_both_hands_frames")
        bundle["fallback_max_both_hands_frames"] = (
            int(fallback_max_both_hands_frames)
            if fallback_max_both_hands_frames is not None
            else None
        )
        bundle["primary"] = {
            "bundle": load_model_bundle(
                resolve_model_dir(resolved_model_dir, primary_model_dir),
                _cache=cache,
            ),
        }
        bundle["fallback"] = {
            "bundle": load_model_bundle(
                resolve_model_dir(resolved_model_dir, fallback_model_dir),
                _cache=cache,
            ),
        }
        bundle.pop("_is_loading", None)
        return bundle

    if model_type == "baseline":
        bundle["weights"] = np.load(resolved_model_dir / "sequence_baseline_model.npz")
        bundle.pop("_is_loading", None)
        return bundle

    config = metadata.get("config", {})
    checkpoint = torch.load(
        resolved_model_dir / "sequence_model.pt",
        map_location="cpu",
    )
    max_sequence_length = int(
        config.get("max_sequence_length", checkpoint.get("max_sequence_length", 48))
    )
    hidden_size = int(config.get("hidden_size", checkpoint.get("hidden_size", 128)))
    model = create_torch_model(
        model_type,
        int(metadata["input_size"]),
        hidden_size,
        len(class_entries),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    bundle["max_sequence_length"] = max_sequence_length
    bundle["model"] = model
    bundle.pop("_is_loading", None)
    return bundle


def predict_probabilities_for_bundle(bundle, sequence):
    class_entries = bundle["class_entries"]
    model_type = bundle["model_type"]
    feature_mode = bundle["feature_mode"]

    if model_type == "ensemble":
        combined = np.zeros((len(class_entries),), dtype=np.float32)
        total_weight = 0.0

        for member in bundle["members"]:
            member_bundle = member["bundle"]
            member_probabilities = predict_probabilities_for_bundle(
                member_bundle,
                sequence,
            )
            weight = float(member.get("weight", 1.0) or 1.0)
            combined += (
                align_probabilities(
                    member_probabilities,
                    member_bundle["class_entries"],
                    class_entries,
                )
                * weight
            )
            total_weight += weight

        if total_weight > 0:
            combined /= total_weight

        total = float(combined.sum())
        if total > 0:
            combined /= total

        return combined.astype(np.float32)

    if model_type == "hybrid_gate":
        primary_bundle = bundle["primary"]["bundle"]
        primary_probabilities = predict_probabilities_for_bundle(primary_bundle, sequence)
        primary_aligned = align_probabilities(
            primary_probabilities,
            primary_bundle["class_entries"],
            class_entries,
        )

        best_index = int(np.argmax(primary_aligned))
        best_entry = class_entries[best_index]
        if best_entry["key"] not in bundle["gated_label_keys"]:
            return primary_aligned.astype(np.float32)

        fallback_bundle = bundle["fallback"]["bundle"]
        fallback_probabilities = predict_probabilities_for_bundle(
            fallback_bundle,
            sequence,
        )
        fallback_aligned = align_probabilities(
            fallback_probabilities,
            fallback_bundle["class_entries"],
            class_entries,
        )
        return fallback_aligned.astype(np.float32)

    if model_type == "hybrid_gate_map":
        primary_bundle = bundle["primary"]["bundle"]
        primary_probabilities = predict_probabilities_for_bundle(primary_bundle, sequence)
        primary_aligned = align_probabilities(
            primary_probabilities,
            primary_bundle["class_entries"],
            class_entries,
        )

        best_index = int(np.argmax(primary_aligned))
        best_entry = class_entries[best_index]
        fallback_alias = bundle["label_fallback_map"].get(best_entry["key"])
        if not fallback_alias:
            return primary_aligned.astype(np.float32)

        fallback_bundle = bundle["fallbacks"][fallback_alias]["bundle"]
        fallback_probabilities = predict_probabilities_for_bundle(
            fallback_bundle,
            sequence,
        )
        fallback_aligned = align_probabilities(
            fallback_probabilities,
            fallback_bundle["class_entries"],
            class_entries,
        )
        return fallback_aligned.astype(np.float32)

    if model_type == "hybrid_confidence_gate":
        primary_bundle = bundle["primary"]["bundle"]
        primary_probabilities = predict_probabilities_for_bundle(primary_bundle, sequence)
        primary_aligned = align_probabilities(
            primary_probabilities,
            primary_bundle["class_entries"],
            class_entries,
        )

        primary_best_index = int(np.argmax(primary_aligned))
        primary_best_entry = class_entries[primary_best_index]
        primary_best_confidence = float(primary_aligned[primary_best_index])

        if primary_best_entry["key"] not in bundle["gated_label_keys"]:
            return primary_aligned.astype(np.float32)

        max_sequence_length = bundle.get("fallback_max_sequence_length")
        if max_sequence_length is not None and len(sequence or []) > max_sequence_length:
            return primary_aligned.astype(np.float32)

        max_both_hands_frames = bundle.get("fallback_max_both_hands_frames")
        if max_both_hands_frames is not None:
            both_hands_frames = sum(
                1
                for frame in sequence or []
                if frame.get("left_hand") and frame.get("right_hand")
            )
            if both_hands_frames > max_both_hands_frames:
                return primary_aligned.astype(np.float32)

        fallback_bundle = bundle["fallback"]["bundle"]
        fallback_probabilities = predict_probabilities_for_bundle(
            fallback_bundle,
            sequence,
        )
        fallback_aligned = align_probabilities(
            fallback_probabilities,
            fallback_bundle["class_entries"],
            class_entries,
        )
        fallback_best_index = int(np.argmax(fallback_aligned))
        fallback_best_entry = class_entries[fallback_best_index]

        should_use_fallback = (
            primary_best_confidence < bundle["primary_confidence_threshold"]
            or fallback_best_entry["key"] in bundle["preferred_fallback_label_keys"]
        )

        if should_use_fallback:
            return fallback_aligned.astype(np.float32)

        return primary_aligned.astype(np.float32)

    if model_type == "baseline":
        weights = bundle["weights"]
        feature_vector = build_sequence_feature(
            {"sequence": sequence, "phrase_text": ""},
            feature_mode=feature_mode,
        )
        mean = weights["mean"]
        std = np.where(weights["std"] < 1e-6, 1.0, weights["std"])
        centroids = weights["centroids"]
        normalized = (feature_vector - mean) / std
        probabilities = predict_probabilities_centroid(
            normalized[None, :],
            centroids,
        )[0]
        return probabilities.astype(np.float32)

    tensors, _, lengths = build_sequence_tensors(
        [
            {
                "sequence": sequence,
                "phrase_text": class_entries[0]["text"],
                "recognition_level": class_entries[0]["recognition_level"],
                "unit_code": class_entries[0]["unit_code"],
            }
        ],
        {class_entries[0]["key"]: 0},
        bundle["max_sequence_length"],
        feature_mode=feature_mode,
    )
    probabilities = predict_probabilities_torch(bundle["model"], tensors, lengths)[0]
    return probabilities.astype(np.float32)


def predict_probabilities_for_sequence(model_dir, sequence):
    bundle = load_model_bundle(model_dir)
    probabilities = predict_probabilities_for_bundle(bundle, sequence)
    return probabilities, bundle["class_entries"]


def apply_level_filter(probabilities, class_entries, allowed_levels):
    if not probabilities.size or not allowed_levels:
        return probabilities

    normalized_levels = {
        str(value).strip().lower() for value in allowed_levels if str(value).strip()
    }
    if not normalized_levels:
        return probabilities

    mask = np.asarray(
        [
            1.0
            if entry.get("recognition_level", "phrase") in normalized_levels
            else 0.0
            for entry in class_entries
        ],
        dtype=np.float32,
    )

    if not np.any(mask):
        return probabilities

    filtered = probabilities * mask
    total = float(filtered.sum())

    if total <= 0:
        return probabilities

    return filtered / total


def apply_label_key_filter(probabilities, class_entries, allowed_label_keys):
    if not probabilities.size or not allowed_label_keys:
        return probabilities

    normalized_keys = {
        str(value).strip() for value in allowed_label_keys if str(value).strip()
    }
    if not normalized_keys:
        return probabilities

    mask = np.asarray(
        [
            1.0 if entry.get("key") in normalized_keys else 0.0
            for entry in class_entries
        ],
        dtype=np.float32,
    )

    if not np.any(mask):
        return probabilities

    filtered = probabilities * mask
    total = float(filtered.sum())

    if total <= 0:
        return probabilities

    return filtered / total


def extract_sequence(input_path=None, sequence=None):
    if input_path:
        sequence_payload = read_json(Path(input_path))
        return sequence_payload.get("frames", sequence_payload.get("sequence", []))

    if sequence is None:
        raise SystemExit("input path or sequence is required")

    return sequence


def build_prediction_output(
    bundle,
    *,
    input_path=None,
    sequence=None,
    allowed_levels=None,
    allowed_label_keys=None,
):
    class_entries = bundle["class_entries"]
    resolved_sequence = extract_sequence(input_path=input_path, sequence=sequence)
    probabilities = predict_probabilities_for_bundle(bundle, resolved_sequence)
    probabilities = apply_level_filter(probabilities, class_entries, allowed_levels or [])
    probabilities = apply_label_key_filter(
        probabilities,
        class_entries,
        allowed_label_keys or [],
    )
    best_index = int(np.argmax(probabilities))
    best_entry = class_entries[best_index]

    return {
        "ok": True,
        "label": best_entry["text"],
        "labelKey": best_entry["key"],
        "recognitionLevel": best_entry["recognition_level"],
        "unitCode": best_entry["unit_code"],
        "confidence": round(float(probabilities[best_index]), 4),
        "modelType": bundle["metadata"].get("model_type", "baseline"),
        "scores": [
            {
                "label": entry["text"],
                "labelKey": entry["key"],
                "recognitionLevel": entry["recognition_level"],
                "unitCode": entry["unit_code"],
                "confidence": round(float(probability), 4),
            }
            for entry, probability in sorted(
                zip(class_entries, probabilities),
                key=lambda item: item[1],
                reverse=True,
            )
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--allowed-recognition-levels")
    parser.add_argument("--allowed-label-keys")
    args = parser.parse_args()

    bundle = load_model_bundle(Path(args.model_dir))
    allowed_levels = (
        args.allowed_recognition_levels.split(",")
        if args.allowed_recognition_levels
        else []
    )
    allowed_label_keys = (
        args.allowed_label_keys.split(",") if args.allowed_label_keys else []
    )
    result = build_prediction_output(
        bundle,
        input_path=args.input,
        allowed_levels=allowed_levels,
        allowed_label_keys=allowed_label_keys,
    )

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
