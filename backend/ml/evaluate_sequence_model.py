import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from predict_sequence_model import load_model_bundle, predict_probabilities_for_bundle
from train_sequence_model import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    build_class_entry,
    build_dataset,
    build_feature_matrix,
    build_metrics_from_probabilities,
    build_sequence_feature,
    build_sequence_tensors,
    create_torch_model,
    normalize_class_entry,
    predict_probabilities_centroid,
    predict_probabilities_torch,
    read_json,
    standardize,
)


SUPPORTED_EVALUATION_MODEL_TYPES = {
    "baseline",
    "gru",
    "lstm",
    "tcn",
    "ensemble",
    "hybrid_gate",
    "hybrid_gate_map",
    "hybrid_confidence_gate",
}


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
    aligned = np.zeros((probabilities.shape[0], len(target_entries)), dtype=np.float32)

    for index, entry in enumerate(target_entries):
        source_position = source_index.get(entry["key"])
        if source_position is not None:
            aligned[:, index] = probabilities[:, source_position]

    row_sums = aligned.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0] = 1.0
    return aligned / row_sums


def predict_probabilities_for_samples(model_dir, samples, dataset_dir):
    metadata = read_json(model_dir / "metadata.json")
    model_type = metadata.get("model_type", "baseline")
    feature_mode = metadata.get("feature_mode", "full")
    class_entries = get_class_entries(metadata)

    if model_type in {"hybrid_gate", "hybrid_gate_map", "hybrid_confidence_gate"}:
        bundle = load_model_bundle(model_dir)
        probabilities = np.zeros((len(samples), len(class_entries)), dtype=np.float32)

        for index, sample in enumerate(samples):
            probabilities[index] = predict_probabilities_for_bundle(
                bundle,
                sample.get("sequence", []),
            )

        return probabilities, class_entries

    if model_type == "ensemble":
        members = metadata.get("members", [])
        if not members:
            raise SystemExit("Ensemble model metadata does not contain members.")

        combined = np.zeros((len(samples), len(class_entries)), dtype=np.float32)
        total_weight = 0.0

        for member in members:
            member_dir = resolve_model_dir(model_dir, member["model_dir"])
            member_probabilities, member_entries = predict_probabilities_for_samples(
                member_dir,
                samples,
                dataset_dir,
            )
            weight = float(member.get("weight", 1.0) or 1.0)
            combined += (
                align_probabilities(member_probabilities, member_entries, class_entries)
                * weight
            )
            total_weight += weight

        if total_weight > 0:
            combined /= total_weight

        row_sums = combined.sum(axis=1, keepdims=True)
        row_sums[row_sums <= 0] = 1.0
        return combined / row_sums, class_entries

    label_to_index = {
        entry["key"]: index for index, entry in enumerate(class_entries)
    }

    if model_type == "baseline":
        train_samples = read_json(dataset_dir / "train.json")
        train_x, train_y, _, _ = build_dataset(train_samples, feature_mode=feature_mode)
        features = (
            build_feature_matrix(samples, feature_mode=feature_mode)
            if samples
            else np.zeros((0, train_x.shape[1]), dtype=np.float32)
        )
        weights = np.load(model_dir / "sequence_baseline_model.npz")
        _, _, normalized = standardize(train_x, features)
        feature_norm = normalized[1] if len(normalized) > 1 else np.zeros_like(features)
        probabilities = predict_probabilities_centroid(feature_norm, weights["centroids"])
        return probabilities.astype(np.float32), class_entries

    config = metadata.get("config", {})
    checkpoint = torch.load(model_dir / "sequence_model.pt", map_location="cpu")
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
    sequences, _, lengths = build_sequence_tensors(
        samples,
        label_to_index,
        max_sequence_length,
        feature_mode=feature_mode,
    )
    probabilities = predict_probabilities_torch(model, sequences, lengths)
    return probabilities.astype(np.float32), class_entries


def labels_for_samples(samples, class_entries):
    label_to_index = {
        entry["key"]: index for index, entry in enumerate(class_entries)
    }
    if not samples:
        return np.zeros((0,), dtype=np.int64)

    return np.asarray(
        [label_to_index[build_class_entry(sample)["key"]] for sample in samples],
        dtype=np.int64,
    )


def evaluate_split(model_dir, samples, class_entries, confidence_threshold, dataset_dir):
    probabilities, loaded_entries = predict_probabilities_for_samples(
        model_dir,
        samples,
        dataset_dir,
    )
    probabilities = align_probabilities(probabilities, loaded_entries, class_entries)
    labels = labels_for_samples(samples, class_entries)

    return build_metrics_from_probabilities(
        probabilities,
        labels,
        class_entries,
        confidence_threshold=confidence_threshold,
        latency_ms_avg=0.0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    model_dir = Path(args.model_dir)

    metadata = read_json(model_dir / "metadata.json")
    model_type = metadata.get("model_type", "baseline")
    class_entries = get_class_entries(metadata)
    labels = [entry["text"] for entry in class_entries]

    if model_type not in SUPPORTED_EVALUATION_MODEL_TYPES:
        raise SystemExit(f"Unsupported model type in metadata: {model_type}")

    train_samples = read_json(dataset_dir / "train.json")
    val_samples = read_json(dataset_dir / "val.json")
    test_samples = read_json(dataset_dir / "test.json")

    evaluation = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "confidence_threshold": args.confidence_threshold,
        "splits": {
            "train": evaluate_split(
                model_dir,
                train_samples,
                class_entries,
                args.confidence_threshold,
                dataset_dir,
            ),
            "val": evaluate_split(
                model_dir,
                val_samples,
                class_entries,
                args.confidence_threshold,
                dataset_dir,
            ),
            "test": evaluate_split(
                model_dir,
                test_samples,
                class_entries,
                args.confidence_threshold,
                dataset_dir,
            ),
        },
        "labels": labels,
        "label_keys": [entry["key"] for entry in class_entries],
        "class_entries": class_entries,
    }

    with (model_dir / "evaluation.json").open("w", encoding="utf-8") as output_file:
        json.dump(evaluation, output_file, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "ok": True,
                "evaluationPath": str(model_dir / "evaluation.json"),
                "evaluation": evaluation,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
