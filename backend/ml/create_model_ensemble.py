import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from train_sequence_model import read_json
from evaluate_sequence_model import evaluate_split, get_class_entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--member-dirs", nargs="+", required=True)
    parser.add_argument("--member-weights")
    parser.add_argument("--confidence-threshold", type=float, default=0.2)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    member_dirs = [Path(path).resolve() for path in args.member_dirs]
    weights = (
        [float(value) for value in args.member_weights.split(",")]
        if args.member_weights
        else [1.0] * len(member_dirs)
    )

    if len(weights) != len(member_dirs):
        raise SystemExit("member weight count must match member dir count")

    member_metadatas = [read_json(member_dir / "metadata.json") for member_dir in member_dirs]
    class_entries = get_class_entries(member_metadatas[0])
    reference_keys = [entry["key"] for entry in class_entries]
    reference_level = member_metadatas[0].get("recognition_level", "phrase")
    reference_feature_mode = member_metadatas[0].get("feature_mode", "full")

    for metadata, member_dir in zip(member_metadatas, member_dirs):
        if [entry["key"] for entry in get_class_entries(metadata)] != reference_keys:
            raise SystemExit(f"class entries mismatch in {member_dir}")
        if metadata.get("recognition_level", "phrase") != reference_level:
            raise SystemExit(f"recognition level mismatch in {member_dir}")
        if metadata.get("feature_mode", "full") != reference_feature_mode:
            raise SystemExit(f"feature mode mismatch in {member_dir}")

    train_samples = read_json(dataset_dir / "train.json")
    val_samples = read_json(dataset_dir / "val.json")
    test_samples = read_json(dataset_dir / "test.json")

    metadata = {
        "model_type": "ensemble",
        "feature_version": member_metadatas[0].get("feature_version", 3),
        "labels": [entry["text"] for entry in class_entries],
        "label_keys": reference_keys,
        "class_entries": class_entries,
        "input_size": member_metadatas[0].get("input_size"),
        "recognition_level": reference_level,
        "feature_mode": reference_feature_mode,
        "confidence_threshold": args.confidence_threshold,
        "members": [
            {
                "name": member_dir.name,
                "model_dir": str(member_dir),
                "weight": weight,
                "model_type": member_metadata.get("model_type", "baseline"),
            }
            for member_dir, weight, member_metadata in zip(
                member_dirs,
                weights,
                member_metadatas,
            )
        ],
        "config": {
            "model_type": "ensemble",
            "ensemble_method": "weighted_mean",
            "confidence_threshold": args.confidence_threshold,
            "member_weights": weights,
        },
        "artifacts": {
            "evaluation": "evaluation.json",
        },
    }

    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metrics = {
        "train": evaluate_split(
            output_dir,
            train_samples,
            class_entries,
            args.confidence_threshold,
            dataset_dir,
        ),
        "val": evaluate_split(
            output_dir,
            val_samples,
            class_entries,
            args.confidence_threshold,
            dataset_dir,
        ),
        "test": evaluate_split(
            output_dir,
            test_samples,
            class_entries,
            args.confidence_threshold,
            dataset_dir,
        ),
    }

    evaluation = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "confidence_threshold": args.confidence_threshold,
        "splits": metrics,
        "labels": metadata["labels"],
        "label_keys": metadata["label_keys"],
        "class_entries": class_entries,
    }
    metadata["metrics"] = metrics
    metadata["evaluation"] = evaluation
    metadata["dataset"] = {
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "test_count": len(test_samples),
        "label_count": len(class_entries),
        "recognition_level": reference_level,
        "feature_mode": reference_feature_mode,
    }

    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "evaluation.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "ok": True,
                "outputDir": str(output_dir),
                "metrics": metrics,
                "memberDirs": [str(path) for path in member_dirs],
                "weights": weights,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
