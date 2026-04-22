import argparse
import json
import math
import urllib.request
from collections import Counter
from pathlib import Path

import ijson

from bootstrap_slovo_signs import (
    DEFAULT_ANNOTATIONS_URL,
    DEFAULT_CACHE_DIR,
    DEFAULT_CONSTANTS_URL,
    DEFAULT_LANDMARKS_URL,
    choose_validation_signers,
    convert_frames,
    extract_annotations_csv_from_zip,
    load_annotations,
    load_slovo_constants,
)


DEFAULT_BASE_DATASET_DIR = Path("backend/uploads/datasets/alphabet/latest")
DEFAULT_OUTPUT_DIR = Path("backend/uploads/datasets/alphabet/alnum_latest")
DEFAULT_LANDMARKS_CACHE_FILE = DEFAULT_CACHE_DIR / "slovo_mediapipe.json"

DIGIT_WORD_MAP = {
    "ноль": {"digit": "0", "unit_code": "NUMBER_0"},
    "один": {"digit": "1", "unit_code": "NUMBER_1"},
    "два": {"digit": "2", "unit_code": "NUMBER_2"},
    "три": {"digit": "3", "unit_code": "NUMBER_3"},
    "четыре": {"digit": "4", "unit_code": "NUMBER_4"},
    "пять": {"digit": "5", "unit_code": "NUMBER_5"},
    "шесть": {"digit": "6", "unit_code": "NUMBER_6"},
    "семь": {"digit": "7", "unit_code": "NUMBER_7"},
    "восемь": {"digit": "8", "unit_code": "NUMBER_8"},
    "девять": {"digit": "9", "unit_code": "NUMBER_9"},
}


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_quality_summary(sequence):
    frame_count = len(sequence)
    if frame_count == 0:
        return {
            "frame_count": 0,
            "valid_frame_ratio": 0.0,
            "left_hand_ratio": 0.0,
            "right_hand_ratio": 0.0,
            "face_ratio": 0.0,
            "pose_ratio": 0.0,
            "missing_hand_ratio": 1.0,
            "missing_face_ratio": 1.0,
            "missing_pose_ratio": 1.0,
        }

    left_hand_frames = sum(1 for frame in sequence if frame["quality"]["has_left_hand"])
    right_hand_frames = sum(1 for frame in sequence if frame["quality"]["has_right_hand"])
    valid_frames = sum(
        1
        for frame in sequence
        if frame["quality"]["has_left_hand"] or frame["quality"]["has_right_hand"]
    )
    missing_hand_frames = frame_count - valid_frames

    return {
        "frame_count": frame_count,
        "valid_frame_ratio": round(valid_frames / frame_count, 4),
        "left_hand_ratio": round(left_hand_frames / frame_count, 4),
        "right_hand_ratio": round(right_hand_frames / frame_count, 4),
        "face_ratio": 0.0,
        "pose_ratio": 0.0,
        "missing_hand_ratio": round(missing_hand_frames / frame_count, 4),
        "missing_face_ratio": 1.0,
        "missing_pose_ratio": 1.0,
    }


def build_digit_sample(annotation, sequence, split):
    mapping = DIGIT_WORD_MAP[annotation["text"].lower()]
    digit = mapping["digit"]
    unit_code = mapping["unit_code"]
    quality_summary = build_quality_summary(sequence)
    frame_count = len(sequence)

    return {
        "sample_id": f"slovo-digit:{annotation['attachment_id']}",
        "label_id": unit_code,
        "label_type": "alphabet",
        "phrase_id": unit_code,
        "phrase_text": digit,
        "entry_type": "alphabet",
        "recognition_level": "alphabet",
        "unit_code": unit_code,
        "category": "Алфавит",
        "sign_language": "rsl",
        "signer_key": f"slovo:{annotation['user_id']}",
        "user_id": annotation["user_id"],
        "user_email": None,
        "duration_ms": int(math.ceil(float(annotation["length"]) * 1000 / 30.0)),
        "dataset_split": split,
        "review_status": "approved",
        "quality_score": 5,
        "landmark_sequence_id": f"slovo-digit:{annotation['attachment_id']}",
        "landmark_frame_count": frame_count,
        "landmark_valid_frame_ratio": quality_summary["valid_frame_ratio"],
        "landmark_missing_hand_ratio": quality_summary["missing_hand_ratio"],
        "landmark_missing_face_ratio": quality_summary["missing_face_ratio"],
        "landmark_missing_pose_ratio": quality_summary["missing_pose_ratio"],
        "landmark_normalization_version": "slovo_digits_hands_only_v1",
        "landmark_file_path": None,
        "landmark_url": None,
        "video_path": f"slovo://digits/{annotation['attachment_id']}.mp4",
        "video_url": None,
        "created_at": None,
        "sequence_meta": {
            "fps": 30.0,
            "frame_count": frame_count,
            "source_frame_count": frame_count,
            "processed_frame_count": frame_count,
            "extractor": "slovo_mediapipe_landmarks",
            "extractor_type": "hand_landmarks",
            "summary": quality_summary,
            "source_video": f"slovo://digits/{annotation['attachment_id']}.mp4",
            "attachment_id": annotation["attachment_id"],
            "user_id": annotation["user_id"],
            "source_label": annotation["text"],
            "digit": digit,
        },
        "sequence": sequence,
    }


def write_dataset(output_dir, train_samples, val_samples, test_samples):
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "train.json", train_samples)
    write_json(output_dir / "val.json", val_samples)
    write_json(output_dir / "test.json", test_samples)

    all_samples = train_samples + val_samples + test_samples
    coverage = Counter(sample["phrase_text"] for sample in all_samples)
    manifest = {
        "source_dataset": "Bukva + Slovo digits",
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
        "digit_coverage": {
            key: value
            for key, value in sorted(coverage.items(), key=lambda item: item[0])
            if key.isdigit()
        },
    }
    write_json(output_dir / "manifest.json", manifest)

    summary = {
        "source_dataset": "Bukva + Slovo digits",
        "recognition_level": "alphabet",
        "feature_mode": "hands_only",
        "sample_count": len(all_samples),
        "split_counts": manifest["split_counts"],
        "class_count": manifest["class_count"],
        "signer_count": manifest["signer_count"],
        "coverage": manifest["coverage"],
        "digit_coverage": manifest["digit_coverage"],
    }
    write_json(output_dir / "summary.json", summary)


def iterate_landmark_items(landmarks_file: Path | None, landmarks_url: str):
    if landmarks_file and landmarks_file.exists():
        with landmarks_file.open("rb") as input_file:
            yield from ijson.kvitems(input_file, "")
        return

    with urllib.request.urlopen(landmarks_url, timeout=120) as response:
        yield from ijson.kvitems(response, "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dataset-dir", default=str(DEFAULT_BASE_DATASET_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--annotations-zip-url", default=DEFAULT_ANNOTATIONS_URL)
    parser.add_argument("--landmarks-url", default=DEFAULT_LANDMARKS_URL)
    parser.add_argument("--landmarks-file", default=str(DEFAULT_LANDMARKS_CACHE_FILE))
    parser.add_argument("--constants-url", default=DEFAULT_CONSTANTS_URL)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()

    base_dataset_dir = Path(args.base_dataset_dir)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    landmarks_file = Path(args.landmarks_file) if args.landmarks_file else None

    base_train = read_json(base_dataset_dir / "train.json")
    base_val = read_json(base_dataset_dir / "val.json")
    base_test = read_json(base_dataset_dir / "test.json")

    annotations_path = extract_annotations_csv_from_zip(
        args.annotations_zip_url,
        cache_dir / "annotations.csv",
    )
    annotations = load_annotations(annotations_path)
    constants = load_slovo_constants(args.constants_url)
    valid_labels = set(constants.values())

    target_rows = [
        row
        for row in annotations
        if row["text"] in valid_labels and row["text"].lower() in DIGIT_WORD_MAP
    ]
    if not target_rows:
        raise SystemExit("No Slovo digit annotations were found.")

    train_rows = [row for row in target_rows if row["train"] == "True"]
    val_signers = choose_validation_signers(train_rows, args.val_ratio)
    attachment_to_row = {row["attachment_id"]: row for row in target_rows}
    sequences = {}

    for attachment_id, raw_frames in iterate_landmark_items(landmarks_file, args.landmarks_url):
        if attachment_id not in attachment_to_row:
            continue

        annotation = attachment_to_row[attachment_id]
        if annotation["train"] != "True":
            split = "test"
        elif annotation["user_id"] in val_signers:
            split = "val"
        else:
            split = "train"

        sequences[attachment_id] = build_digit_sample(
            annotation=annotation,
            sequence=convert_frames(raw_frames),
            split=split,
        )

        if len(sequences) == len(attachment_to_row):
            break

    missing_ids = sorted(set(attachment_to_row) - set(sequences))
    if missing_ids:
        raise SystemExit(
            f"Missing landmark sequences for {len(missing_ids)} digit samples: {missing_ids[:10]}"
        )

    digit_train = [
        sequences[row["attachment_id"]]
        for row in target_rows
        if sequences[row["attachment_id"]]["dataset_split"] == "train"
    ]
    digit_val = [
        sequences[row["attachment_id"]]
        for row in target_rows
        if sequences[row["attachment_id"]]["dataset_split"] == "val"
    ]
    digit_test = [
        sequences[row["attachment_id"]]
        for row in target_rows
        if sequences[row["attachment_id"]]["dataset_split"] == "test"
    ]

    merged_train = base_train + digit_train
    merged_val = base_val + digit_val
    merged_test = base_test + digit_test
    write_dataset(output_dir, merged_train, merged_val, merged_test)

    payload = {
        "ok": True,
        "outputDir": str(output_dir),
        "baseSampleCount": len(base_train) + len(base_val) + len(base_test),
        "digitSampleCount": len(digit_train) + len(digit_val) + len(digit_test),
        "mergedSampleCount": len(merged_train) + len(merged_val) + len(merged_test),
        "digitSplitCounts": {
            "train": len(digit_train),
            "val": len(digit_val),
            "test": len(digit_test),
        },
        "digitCoverage": dict(
            sorted(
                Counter(sample["phrase_text"] for sample in digit_train + digit_val + digit_test).items()
            )
        ),
    }
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
