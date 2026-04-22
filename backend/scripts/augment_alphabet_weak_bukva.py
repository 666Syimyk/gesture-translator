import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

from bootstrap_bukva_alphabet import (
    CACHE_DIR as DEFAULT_CACHE_DIR,
    DEFAULT_VAL_USER_RATIO,
    build_hand_landmarker,
    build_sample,
    load_or_extract_sequence,
    load_unit_code_map,
    read_annotations,
    split_rows,
)


DEFAULT_BASE_DATASET_DIR = Path(
    "backend/uploads/datasets/alphabet/alnum_digits_val005_20260404"
)
DEFAULT_OUTPUT_DIR = Path(
    "backend/uploads/datasets/alphabet/alnum_weakfix_bukva_20260404"
)
DEFAULT_TARGET_CODES = (
    "LETTER_SH",
    "LETTER_KH",
    "LETTER_P",
    "LETTER_YA",
    "LETTER_T",
    "LETTER_SHCH",
    "LETTER_HARD_SIGN",
)
DEFAULT_TRAIN_TARGET_PER_CLASS = 36
DEFAULT_VAL_TARGET_PER_CLASS = 8


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_summary(train_samples, val_samples, test_samples):
    all_samples = train_samples + val_samples + test_samples
    coverage = Counter(sample["phrase_text"] for sample in all_samples)
    return {
        "source_dataset": "Current alphabet + targeted Bukva weak-class expansion",
        "recognition_level": "alphabet",
        "feature_mode": "hands_only",
        "sample_count": len(all_samples),
        "split_counts": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        },
        "class_count": len(coverage),
        "signer_count": len({sample["signer_key"] for sample in all_samples}),
        "coverage": dict(sorted(coverage.items(), key=lambda item: item[0])),
    }


def count_by_code(samples):
    return Counter(sample.get("unit_code") for sample in samples)


def load_cached_payload(cache_dir: Path, attachment_id: str):
    cache_path = cache_dir / f"{attachment_id}.json"
    if not cache_path.exists():
        raise FileNotFoundError(f"Bukva cache is missing {cache_path}")
    return read_json(cache_path)


def select_augmented_samples(
    *,
    rows,
    split_name,
    cache_dir,
    unit_code_map,
    target_codes,
    existing_ids,
    current_counts,
    target_per_class,
    fetch_context,
    skip_unavailable,
):
    selected = []
    rows = sorted(rows, key=lambda row: row["attachment_id"])

    for row in rows:
        unit_code = unit_code_map.get(row["text"])
        if unit_code not in target_codes:
            continue

        if current_counts[unit_code] >= target_per_class:
            continue

        sample_id = f"bukva:{row['attachment_id']}"
        if sample_id in existing_ids:
            continue

        try:
            payload = load_cached_payload(cache_dir, row["attachment_id"])
        except FileNotFoundError:
            if fetch_context is None:
                if skip_unavailable:
                    continue
                raise

            try:
                payload, next_timestamp_ms = load_or_extract_sequence(
                    row,
                    fetch_context["hand_landmarker"],
                    cache_dir,
                    fetch_context["temp_dir"],
                    fetch_context["next_timestamp_ms"],
                )
                fetch_context["next_timestamp_ms"] = next_timestamp_ms
            except Exception:
                if skip_unavailable:
                    continue
                raise

        sample = build_sample(row, split_name, payload, unit_code)
        selected.append(sample)
        existing_ids.add(sample["sample_id"])
        current_counts[unit_code] += 1

    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dataset-dir", default=str(DEFAULT_BASE_DATASET_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--val-user-ratio", type=float, default=DEFAULT_VAL_USER_RATIO)
    parser.add_argument(
        "--target-codes",
        default=",".join(DEFAULT_TARGET_CODES),
        help="Comma-separated alphabet unit codes to expand from Bukva cache.",
    )
    parser.add_argument(
        "--train-target-per-class",
        type=int,
        default=DEFAULT_TRAIN_TARGET_PER_CLASS,
    )
    parser.add_argument(
        "--val-target-per-class",
        type=int,
        default=DEFAULT_VAL_TARGET_PER_CLASS,
    )
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--fetch-missing", action="store_true")
    parser.add_argument("--skip-unavailable", action="store_true")
    args = parser.parse_args()

    base_dataset_dir = Path(args.base_dataset_dir)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    target_codes = {
        value.strip()
        for value in str(args.target_codes).split(",")
        if value.strip()
    }

    train_samples = read_json(base_dataset_dir / "train.json")
    val_samples = read_json(base_dataset_dir / "val.json")
    test_samples = read_json(base_dataset_dir / "test.json")

    existing_ids = {
        sample["sample_id"]
        for sample in train_samples + val_samples + test_samples
        if sample.get("sample_id")
    }
    train_counts = count_by_code(train_samples)
    val_counts = count_by_code(val_samples)

    unit_code_map = load_unit_code_map()
    annotations = [
        row
        for row in read_annotations(use_cached_only=args.cache_only)
        if unit_code_map.get(row["text"]) in target_codes
    ]
    train_rows, val_rows, _, _ = split_rows(
        annotations,
        args.val_user_ratio,
        max_train_per_class=0,
        max_val_per_class=0,
        max_test_per_class=0,
    )

    fetch_context = None
    hand_landmarker = None

    try:
        if args.fetch_missing:
            hand_landmarker = build_hand_landmarker()
            temp_dir_context = tempfile.TemporaryDirectory(prefix="bukva-weak-augment-")
            fetch_context = {
                "hand_landmarker": hand_landmarker,
                "temp_dir_context": temp_dir_context,
                "temp_dir": temp_dir_context.name,
                "next_timestamp_ms": 0,
            }

        extra_train = select_augmented_samples(
            rows=train_rows,
            split_name="train",
            cache_dir=cache_dir,
            unit_code_map=unit_code_map,
            target_codes=target_codes,
            existing_ids=existing_ids,
            current_counts=train_counts,
            target_per_class=args.train_target_per_class,
            fetch_context=fetch_context,
            skip_unavailable=args.skip_unavailable,
        )
        extra_val = select_augmented_samples(
            rows=val_rows,
            split_name="val",
            cache_dir=cache_dir,
            unit_code_map=unit_code_map,
            target_codes=target_codes,
            existing_ids=existing_ids,
            current_counts=val_counts,
            target_per_class=args.val_target_per_class,
            fetch_context=fetch_context,
            skip_unavailable=args.skip_unavailable,
        )
    finally:
        if fetch_context is not None:
            fetch_context["temp_dir_context"].cleanup()
        if hand_landmarker is not None:
            hand_landmarker.close()

    merged_train = train_samples + extra_train
    merged_val = val_samples + extra_val
    merged_test = test_samples

    write_json(output_dir / "train.json", merged_train)
    write_json(output_dir / "val.json", merged_val)
    write_json(output_dir / "test.json", merged_test)
    summary = build_summary(merged_train, merged_val, merged_test)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "manifest.json", summary)

    print(
        json.dumps(
            {
                "ok": True,
                "outputDir": str(output_dir),
                "added": {
                    "train": len(extra_train),
                    "val": len(extra_val),
                    "test": 0,
                },
                "trainCounts": {
                    code: train_counts.get(code, 0) for code in sorted(target_codes)
                },
                "valCounts": {
                    code: val_counts.get(code, 0) for code in sorted(target_codes)
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
