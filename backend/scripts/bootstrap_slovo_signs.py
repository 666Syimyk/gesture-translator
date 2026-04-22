import argparse
import ast
import csv
import json
import math
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

import ijson


DEFAULT_ANNOTATIONS_URL = (
    "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/slovo.zip"
)
DEFAULT_LANDMARKS_URL = (
    "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/slovo_mediapipe.json"
)
DEFAULT_CONSTANTS_URL = "https://raw.githubusercontent.com/hukenovs/slovo/main/constants.py"
DEFAULT_OUTPUT_DIR = Path("backend/uploads/datasets/sign/slovo_latest")
DEFAULT_CACHE_DIR = Path("backend/uploads/external/slovo")
SIMPLE_TARGET_LABELS = [
    "Привет!",
    "да",
    "хорошо",
    "ждать",
    "вода",
    "еда",
    "пить",
    "дом",
    "холодный",
    "боль",
    "Плохо",
    "медленный",
    "женский туалет",
    "мужской туалет",
]
COMPACT_TARGET_LABELS = [
    "\u041f\u0440\u0438\u0432\u0435\u0442!",
    "\u043f\u0438\u0442\u044c",
    "\u0432\u043e\u0434\u0430",
    "\u0435\u0434\u0430",
    "\u0431\u043e\u043b\u044c",
    "\u0434\u043e\u043c",
    "\u0436\u0435\u043d\u0441\u043a\u0438\u0439 \u0442\u0443\u0430\u043b\u0435\u0442",
    "\u043c\u0443\u0436\u0441\u043a\u043e\u0439 \u0442\u0443\u0430\u043b\u0435\u0442",
    "\u043c\u0435\u0434\u043b\u0435\u043d\u043d\u044b\u0439",
    "\u041f\u043b\u043e\u0445\u043e",
]

EXTENDED_TARGET_LABELS = [
    "\u041f\u0440\u0438\u0432\u0435\u0442!",
    "\u0434\u0430",
    "\u0436\u0434\u0430\u0442\u044c",
    "\u043f\u0438\u0442\u044c",
    "\u0432\u043e\u0434\u0430",
    "\u043f\u0438\u0442\u044c\u0435\u0432\u0430\u044f \u0432\u043e\u0434\u0430",
    "\u0435\u0434\u0430",
    "\u0434\u043e\u043c",
    "\u0431\u043e\u043b\u044c",
    "\u0431\u043e\u043b\u0438\u0442",
    "\u041f\u043b\u043e\u0445\u043e",
    "\u0435\u0449\u0435 \u043d\u0435\u0442",
    "\u043a\u0443\u043f\u0438\u0442\u044c",
    "\u0436\u0435\u043d\u0441\u043a\u0438\u0439 \u0442\u0443\u0430\u043b\u0435\u0442",
    "\u043c\u0443\u0436\u0441\u043a\u043e\u0439 \u0442\u0443\u0430\u043b\u0435\u0442",
    "\u043c\u0435\u0434\u043b\u0435\u043d\u043d\u044b\u0439",
]

DISPLAY_ALIASES = {
    "Привет!": "Привет",
    "да": "Да",
    "хорошо": "Хорошо",
    "ждать": "Подождите",
    "вода": "Вода",
    "еда": "Еда",
    "пить": "Пить",
    "дом": "Дом",
    "холодный": "Холодно",
    "боль": "Больно",
    "Плохо": "Плохо",
    "медленный": "Медленно",
    "женский туалет": "Женский туалет",
    "мужской туалет": "Мужской туалет",
}

DEFAULT_TARGET_LABELS = SIMPLE_TARGET_LABELS


def download_text(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8")


def load_slovo_constants(url):
    payload = download_text(url)
    return ast.literal_eval(payload.split("=", 1)[1].strip())


def extract_annotations_csv_from_zip(zip_url, cache_path):
    if cache_path.exists():
        return cache_path

    head_request = urllib.request.Request(zip_url, method="HEAD")
    with urllib.request.urlopen(head_request, timeout=30) as response:
        zip_size = int(response.headers["Content-Length"])

    tail_size = min(8_000_000, zip_size)
    range_start = zip_size - tail_size
    tail_request = urllib.request.Request(
        zip_url,
        headers={"Range": f"bytes={range_start}-{zip_size - 1}"},
    )
    with urllib.request.urlopen(tail_request, timeout=120) as response:
        tail = response.read()

    marker = b"annotations.csv"
    marker_index = tail.find(marker)
    if marker_index == -1:
        raise SystemExit("annotations.csv was not found inside slovo.zip")

    header_index = tail.rfind(b"PK\x03\x04", 0, marker_index)
    if header_index == -1:
        raise SystemExit("Failed to locate local zip header for annotations.csv")

    signature, version, flag, compression, mtime, mdate, crc32, compressed_size, uncompressed_size, name_len, extra_len = (
        json.loads("null")
    )
    import struct
    import zlib

    (
        signature,
        version,
        flag,
        compression,
        mtime,
        mdate,
        crc32,
        compressed_size,
        uncompressed_size,
        name_len,
        extra_len,
    ) = struct.unpack("<4s5H3I2H", tail[header_index : header_index + 30])

    if signature != b"PK\x03\x04":
        raise SystemExit("annotations.csv header signature is invalid")

    data_start = header_index + 30 + name_len + extra_len
    compressed = tail[data_start : data_start + compressed_size]

    if compression == 8:
        raw = zlib.decompress(compressed, -15)
    elif compression == 0:
        raw = compressed
    else:
        raise SystemExit(f"Unsupported zip compression for annotations.csv: {compression}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(raw)
    return cache_path


def load_annotations(path):
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file, delimiter="\t"))


def slugify_label(text):
    replacements = {
        " ": "_",
        "!": "",
        "?": "",
        ",": "",
        ";": "",
        ":": "",
        "-": "_",
        "/": "_",
        "(": "",
        ")": "",
    }
    normalized = text.strip().lower()
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)

    return normalized or "unknown"


def build_sample(annotation, class_id, sequence, split):
    frame_count = len(sequence)
    quality_summary = {
        "frame_count": frame_count,
        "valid_frame_ratio": 1.0 if frame_count else 0.0,
        "left_hand_ratio": 0.0,
        "right_hand_ratio": 1.0 if frame_count else 0.0,
        "face_ratio": 0.0,
        "pose_ratio": 0.0,
        "missing_hand_ratio": 0.0 if frame_count else 1.0,
        "missing_face_ratio": 1.0,
        "missing_pose_ratio": 1.0,
    }

    source_text = annotation["text"]
    phrase_text = DISPLAY_ALIASES.get(source_text, source_text)
    unit_code = f"SLOVO_{class_id}_{slugify_label(phrase_text)}".upper()
    signer_key = annotation["user_id"]

    return {
        "sample_id": f"slovo:{annotation['attachment_id']}",
        "label_id": unit_code,
        "label_type": "sign",
        "phrase_id": unit_code,
        "phrase_text": phrase_text,
        "entry_type": "sign",
        "recognition_level": "sign",
        "unit_code": unit_code,
        "category": "Slovo",
        "sign_language": "rsl",
        "signer_key": signer_key,
        "user_id": signer_key,
        "user_email": None,
        "duration_ms": int(math.ceil(float(annotation["length"]) * 1000 / 30.0)),
        "dataset_split": split,
        "review_status": "approved",
        "quality_score": 5,
        "landmark_sequence_id": f"slovo:{annotation['attachment_id']}",
        "landmark_frame_count": frame_count,
        "landmark_valid_frame_ratio": quality_summary["valid_frame_ratio"],
        "landmark_missing_hand_ratio": quality_summary["missing_hand_ratio"],
        "landmark_missing_face_ratio": quality_summary["missing_face_ratio"],
        "landmark_missing_pose_ratio": quality_summary["missing_pose_ratio"],
        "landmark_normalization_version": "slovo_hands_only_v1",
        "landmark_file_path": None,
        "landmark_url": None,
        "video_path": f"slovo://{split}/{annotation['attachment_id']}.mp4",
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
            "source_video": f"slovo://{split}/{annotation['attachment_id']}.mp4",
            "attachment_id": annotation["attachment_id"],
            "user_id": signer_key,
            "text": phrase_text,
            "source_text": source_text,
        },
        "sequence": sequence,
    }


def convert_frames(raw_frames):
    def normalize_points(points):
        normalized = []
        for point in points:
            normalized.append(
                {
                    "x": float(point.get("x", 0.0)),
                    "y": float(point.get("y", 0.0)),
                    "z": float(point.get("z", 0.0)),
                }
            )
        return normalized

    sequence = []
    for frame_index, frame in enumerate(raw_frames):
        hand_1 = normalize_points(frame.get("hand 1", []))
        hand_2 = normalize_points(frame.get("hand 2", []))
        sequence.append(
            {
                "frame_index": frame_index,
                "timestamp_ms": int(round(frame_index * (1000.0 / 30.0))),
                "left_hand": hand_2,
                "right_hand": hand_1,
                "face": [],
                "pose": [],
                "left_hand_world": [],
                "right_hand_world": [],
                "pose_world": [],
                "handedness": [],
                "quality": {
                    "has_left_hand": bool(hand_2),
                    "has_right_hand": bool(hand_1),
                    "has_face": False,
                    "has_pose": False,
                },
            }
        )
    return sequence


def choose_validation_signers(train_rows, val_ratio):
    signers = sorted({row["user_id"] for row in train_rows})
    if len(signers) <= 1:
        return set()

    val_count = max(1, round(len(signers) * val_ratio))
    return set(signers[-val_count:])


def write_dataset(output_dir, train_samples, val_samples, test_samples, target_labels):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train.json").write_text(
        json.dumps(train_samples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "val.json").write_text(
        json.dumps(val_samples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "test.json").write_text(
        json.dumps(test_samples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    all_samples = train_samples + val_samples + test_samples
    coverage = Counter(sample["phrase_text"] for sample in all_samples)
    manifest = {
        "source_dataset": "Slovo",
        "recognition_level": "sign",
        "feature_mode": "hands_only",
        "target_labels": target_labels,
        "sample_count": len(all_samples),
        "split_counts": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        },
        "class_count": len(coverage),
        "signer_count": len({sample["signer_key"] for sample in all_samples}),
        "coverage": dict(sorted(coverage.items())),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "source_dataset": "Slovo",
        "recognition_level": "sign",
        "feature_mode": "hands_only",
        "sample_count": len(all_samples),
        "split_counts": manifest["split_counts"],
        "class_count": manifest["class_count"],
        "signer_count": manifest["signer_count"],
        "coverage": manifest["coverage"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def open_landmarks_stream(cache_dir, landmarks_url):
    cached_landmarks_path = cache_dir / "slovo_mediapipe.json"
    if cached_landmarks_path.exists():
        return cached_landmarks_path.open("rb")

    parsed = urlparse(landmarks_url)
    if parsed.scheme in ("", "file"):
        local_path = Path(parsed.path or landmarks_url)
        if local_path.exists():
            return local_path.open("rb")

    return urllib.request.urlopen(landmarks_url, timeout=120)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--annotations-zip-url", default=DEFAULT_ANNOTATIONS_URL)
    parser.add_argument("--landmarks-url", default=DEFAULT_LANDMARKS_URL)
    parser.add_argument("--constants-url", default=DEFAULT_CONSTANTS_URL)
    parser.add_argument("--labels")
    parser.add_argument("--preset", choices=("simple", "compact", "extended"), default="simple")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    annotations_path = extract_annotations_csv_from_zip(
        args.annotations_zip_url,
        cache_dir / "annotations.csv",
    )
    annotations = load_annotations(annotations_path)
    constants = load_slovo_constants(args.constants_url)
    class_by_text = {value: key for key, value in constants.items()}

    if args.labels:
        target_labels = [item.strip() for item in args.labels.split(",") if item.strip()]
    elif args.preset == "extended":
        target_labels = EXTENDED_TARGET_LABELS
    elif args.preset == "compact":
        target_labels = COMPACT_TARGET_LABELS
    else:
        target_labels = DEFAULT_TARGET_LABELS
    missing_labels = [label for label in target_labels if label not in class_by_text]
    if missing_labels:
        raise SystemExit(
            f"Requested Slovo labels are missing in constants.py: {', '.join(missing_labels)}"
        )

    target_set = set(target_labels)
    target_rows = [row for row in annotations if row["text"] in target_set]
    if not target_rows:
        raise SystemExit("No matching Slovo annotations were found for the requested labels")

    train_rows = [row for row in target_rows if row["train"] == "True"]
    test_rows = [row for row in target_rows if row["train"] != "True"]
    val_signers = choose_validation_signers(train_rows, args.val_ratio)

    attachment_to_row = {row["attachment_id"]: row for row in target_rows}
    sequences = {}

    with open_landmarks_stream(cache_dir, args.landmarks_url) as response:
        for attachment_id, raw_frames in ijson.kvitems(response, ""):
            if attachment_id not in attachment_to_row:
                continue

            annotation = attachment_to_row[attachment_id]
            if annotation["train"] != "True":
                split = "test"
            elif annotation["user_id"] in val_signers:
                split = "val"
            else:
                split = "train"

            sequences[attachment_id] = build_sample(
                annotation=annotation,
                class_id=class_by_text[annotation["text"]],
                sequence=convert_frames(raw_frames),
                split=split,
            )

            if len(sequences) == len(attachment_to_row):
                break

    missing_ids = sorted(set(attachment_to_row) - set(sequences))
    if missing_ids:
        raise SystemExit(
            f"Missing landmark sequences for {len(missing_ids)} Slovo samples: {missing_ids[:10]}"
        )

    train_samples = [
        sequences[row["attachment_id"]]
        for row in target_rows
        if sequences[row["attachment_id"]]["dataset_split"] == "train"
    ]
    val_samples = [
        sequences[row["attachment_id"]]
        for row in target_rows
        if sequences[row["attachment_id"]]["dataset_split"] == "val"
    ]
    test_samples = [
        sequences[row["attachment_id"]]
        for row in target_rows
        if sequences[row["attachment_id"]]["dataset_split"] == "test"
    ]

    if not train_samples or not val_samples or not test_samples:
        raise SystemExit(
            "Slovo bootstrap produced an empty split. Try a different label set or val ratio."
        )

    write_dataset(output_dir, train_samples, val_samples, test_samples, target_labels)

    summary = {
        "ok": True,
        "outputDir": str(output_dir),
        "sampleCount": len(train_samples) + len(val_samples) + len(test_samples),
        "splitCounts": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        },
        "classCount": len(target_labels),
        "signerCount": len({sample["signer_key"] for sample in train_samples + val_samples + test_samples}),
        "targetLabels": target_labels,
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
