import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PHRASE_DIR = Path(__file__).resolve().parent
ML_DIR = PHRASE_DIR.parent
if str(PHRASE_DIR) not in sys.path:
    sys.path.insert(0, str(PHRASE_DIR))
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from config import load_config, resolve_backend_path
from data import build_class_entries, build_label_to_index, load_dataset_splits
from predict_sequence_model import (
    align_probabilities,
    load_model_bundle,
    predict_probabilities_for_bundle,
)
from train_sequence_model import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    build_metrics_from_probabilities,
)


def labels_for_samples(samples, label_to_index, recognition_level):
    return np.asarray(
        [
            label_to_index[f"{recognition_level}::{sample['phrase_text']}"]
            for sample in samples
        ],
        dtype=np.int64,
    )


def classification_report_from_metrics(metrics):
    matrix = np.asarray(metrics.get("confusion_matrix", []), dtype=np.int64)
    labels = metrics.get("labels", [])
    report = {}

    if matrix.size == 0:
        return report

    for index, label in enumerate(labels):
        tp = float(matrix[index, index])
        fp = float(matrix[:, index].sum() - matrix[index, index])
        fn = float(matrix[index, :].sum() - matrix[index, index])
        support = int(matrix[index, :].sum())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        report[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "support": support,
        }

    return report


def evaluate_split(bundle, samples, class_entries, label_to_index, recognition_level, threshold):
    if not samples:
        metrics = build_metrics_from_probabilities(
            np.zeros((0, len(class_entries)), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            class_entries,
            confidence_threshold=threshold,
            latency_ms_avg=0.0,
        )
        metrics["classification_report"] = {}
        return metrics

    probabilities = []
    for sample in samples:
        sample_probabilities = predict_probabilities_for_bundle(
            bundle,
            sample.get("sequence", []),
        )
        probabilities.append(
            align_probabilities(
                sample_probabilities,
                bundle["class_entries"],
                class_entries,
            )
        )

    probability_matrix = np.stack(probabilities).astype(np.float32)
    labels = labels_for_samples(samples, label_to_index, recognition_level)
    metrics = build_metrics_from_probabilities(
        probability_matrix,
        labels,
        class_entries,
        confidence_threshold=threshold,
        latency_ms_avg=0.0,
    )
    metrics["classification_report"] = classification_report_from_metrics(metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PHRASE_DIR / "default_config.json"))
    parser.add_argument("--dataset-dir")
    parser.add_argument("--model-dir")
    parser.add_argument("--confidence-threshold", type=float)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_dir = resolve_backend_path(args.dataset_dir or config["dataset_dir"])
    model_dir = resolve_backend_path(args.model_dir or config["artifacts_dir"])
    recognition_level = config.get("recognition_level", "phrase")
    threshold = float(
        args.confidence_threshold
        if args.confidence_threshold is not None
        else config["inference"].get(
            "confidence_threshold",
            DEFAULT_CONFIDENCE_THRESHOLD,
        )
    )

    class_entries = build_class_entries(config["phrases"], recognition_level)
    label_to_index = build_label_to_index(class_entries)
    splits = load_dataset_splits(dataset_dir, config["phrases"], recognition_level)
    bundle = load_model_bundle(model_dir)

    evaluation = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(model_dir),
        "dataset_dir": str(dataset_dir),
        "confidence_threshold": threshold,
        "splits": {
            "train": evaluate_split(
                bundle,
                splits["train"],
                class_entries,
                label_to_index,
                recognition_level,
                threshold,
            ),
            "val": evaluate_split(
                bundle,
                splits["val"],
                class_entries,
                label_to_index,
                recognition_level,
                threshold,
            ),
            "test": evaluate_split(
                bundle,
                splits["test"],
                class_entries,
                label_to_index,
                recognition_level,
                threshold,
            ),
        },
        "labels": [entry["text"] for entry in class_entries],
        "label_keys": [entry["key"] for entry in class_entries],
        "class_entries": class_entries,
    }

    output_path = model_dir / "evaluation.json"
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(evaluation, output_file, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "ok": True,
                "evaluationPath": str(output_path),
                "evaluation": evaluation,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

