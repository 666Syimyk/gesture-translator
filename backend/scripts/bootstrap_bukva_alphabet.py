import argparse
import csv
import io
import json
import os
import random
import tempfile
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
from remotezip import RemoteZip


BUKVA_ZIP_URL = (
    "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/bukva/bukva.zip"
)
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DEFAULT_SIGN_LANGUAGE = "rsl"
DEFAULT_VAL_USER_RATIO = 0.15
DEFAULT_MAX_TRAIN_PER_CLASS = 24
DEFAULT_MAX_VAL_PER_CLASS = 6
DEFAULT_MAX_TEST_PER_CLASS = 12
SEED = 42

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
DATASET_DIR = (
    Path(__file__).resolve().parent.parent / "uploads" / "datasets" / "alphabet" / "latest"
)
CACHE_DIR = (
    Path(__file__).resolve().parent.parent / "uploads" / "external" / "bukva" / "alphabet_cache"
)
ANNOTATIONS_CACHE_PATH = (
    Path(__file__).resolve().parent.parent / "uploads" / "external" / "bukva" / "annotations.tsv"
)
LETTERS_TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "docs"
    / "templates"
    / "letters.rsl.template.json"
)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def ensure_model():
    ensure_dir(MODEL_DIR)
    model_path = MODEL_DIR / "hand_landmarker.task"

    if not model_path.exists() or model_path.stat().st_size == 0:
      urllib.request.urlretrieve(HAND_MODEL_URL, model_path)

    return str(model_path)


def load_unit_code_map():
    raw = json.loads(LETTERS_TEMPLATE_PATH.read_text(encoding="utf-8"))
    entries = raw if isinstance(raw, list) else raw.get("entries", [])
    mapping = {}

    for entry in entries:
        text = str(entry.get("text") or "").strip()
        unit_code = str(entry.get("unitCode") or "").strip()

        if text and unit_code:
            mapping[text] = unit_code

    return mapping


def decode_bukva_text(value):
    if not value:
        return ""

    text = str(value).strip()
    for encoding in ("utf-8", "cp1251"):
        try:
            return text.encode("latin1").decode(encoding).strip()
        except Exception:
            continue

    return text


def normalize_train_flag(value):
    return str(value).strip().lower() == "true"


def read_annotations(use_cached_only=False):
    content = None

    if not use_cached_only:
        try:
            with RemoteZip(BUKVA_ZIP_URL) as archive:
                content = archive.read("annotations.tsv").decode("latin1")
            ensure_dir(ANNOTATIONS_CACHE_PATH.parent)
            ANNOTATIONS_CACHE_PATH.write_text(content, encoding="latin1")
        except Exception:
            content = None

    if content is None:
        if not ANNOTATIONS_CACHE_PATH.exists():
            raise RuntimeError(
                "Failed to load Bukva annotations and local cache is not available."
            )
        content = ANNOTATIONS_CACHE_PATH.read_text(encoding="latin1")

    rows = []
    for row in csv.DictReader(io.StringIO(content), delimiter="\t"):
        rows.append(
            {
                "attachment_id": str(row["attachment_id"]).strip(),
                "user_id": str(row["user_id"]).strip(),
                "text": decode_bukva_text(row["text"]),
                "train": normalize_train_flag(row["train"]),
                "width": int(row.get("width") or 0),
                "height": int(row.get("height") or 0),
                "length": int(row.get("length") or 0),
            }
        )

    return rows


def cap_rows_per_class(rows, max_per_class):
    if max_per_class <= 0:
        return rows

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["text"]].append(row)

    capped = []
    for label in sorted(grouped):
        capped.extend(grouped[label][:max_per_class])

    return capped


def split_rows(rows, val_user_ratio, max_train_per_class, max_val_per_class, max_test_per_class):
    train_source = [row for row in rows if row["train"]]
    test_source = [row for row in rows if not row["train"]]

    train_users = sorted({row["user_id"] for row in train_source})
    rng = random.Random(SEED)
    rng.shuffle(train_users)
    val_user_count = max(1, int(round(len(train_users) * val_user_ratio)))
    val_users = set(train_users[:val_user_count])

    train_rows = [row for row in train_source if row["user_id"] not in val_users]
    val_rows = [row for row in train_source if row["user_id"] in val_users]
    test_rows = list(test_source)

    train_rows = cap_rows_per_class(train_rows, max_train_per_class)
    val_rows = cap_rows_per_class(val_rows, max_val_per_class)
    test_rows = cap_rows_per_class(test_rows, max_test_per_class)

    return train_rows, val_rows, test_rows, val_users


def serialize_points(points):
    if not points:
        return []

    return [
        {
            "x": round(float(point.x), 6),
            "y": round(float(point.y), 6),
            "z": round(float(point.z), 6),
        }
        for point in points
    ]


def serialize_handedness(category):
    if category is None:
        return None

    return {
        "label": getattr(category, "category_name", ""),
        "score": round(float(getattr(category, "score", 0.0)), 6),
    }


def build_hand_landmarker():
    return vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=ensure_model()),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.65,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7,
        )
    )


def extract_frame_payload(hand_result, frame_index, timestamp_ms):
    left_hand = []
    right_hand = []
    handedness = []

    hand_landmarks = getattr(hand_result, "hand_landmarks", []) or []
    hand_handedness = getattr(hand_result, "handedness", []) or []

    for index, landmarks in enumerate(hand_landmarks):
        handedness_category = None
        if index < len(hand_handedness) and hand_handedness[index]:
            handedness_category = hand_handedness[index][0]

        handedness_payload = serialize_handedness(handedness_category)
        if handedness_payload:
            handedness.append(handedness_payload)

        hand_label = (handedness_payload or {}).get("label", "Right").lower()
        if hand_label == "left":
            left_hand = serialize_points(landmarks)
        else:
            right_hand = serialize_points(landmarks)

    quality = {
        "has_left_hand": bool(left_hand),
        "has_right_hand": bool(right_hand),
        "has_face": False,
        "has_pose": False,
    }

    return {
        "frame_index": frame_index,
        "timestamp_ms": timestamp_ms,
        "left_hand": left_hand,
        "right_hand": right_hand,
        "face": [],
        "pose": [],
        "left_hand_world": [],
        "right_hand_world": [],
        "pose_world": [],
        "handedness": handedness,
        "quality": quality,
    }


def build_summary(frames):
    frame_count = len(frames)

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

    left_hand_frames = sum(1 for frame in frames if frame["quality"]["has_left_hand"])
    right_hand_frames = sum(1 for frame in frames if frame["quality"]["has_right_hand"])
    valid_frames = sum(
        1
        for frame in frames
        if frame["quality"]["has_left_hand"] or frame["quality"]["has_right_hand"]
    )
    missing_hand_frames = sum(
        1
        for frame in frames
        if not frame["quality"]["has_left_hand"] and not frame["quality"]["has_right_hand"]
    )

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


def extract_sequence_from_video(video_path: Path, hand_landmarker, start_timestamp_ms=0):
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
    frame_index = 0
    last_timestamp_ms = int(start_timestamp_ms) - 1

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            relative_timestamp_ms = (
                round((frame_index / fps) * 1000) if fps > 0 else frame_index * 33
            )
            timestamp_ms = max(last_timestamp_ms + 1, int(start_timestamp_ms) + relative_timestamp_ms)
            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            frames.append(extract_frame_payload(hand_result, frame_index, timestamp_ms))
            frame_index += 1
            last_timestamp_ms = timestamp_ms
    finally:
        capture.release()

    summary = build_summary(frames)
    payload = {
        "meta": {
            "fps": round(fps, 4),
            "frame_width": frame_width,
            "frame_height": frame_height,
            "source_frame_count": total_frames,
            "processed_frame_count": len(frames),
            "frame_count": len(frames),
            "extractor": "mediapipe_tasks_python_hand_v1",
            "extractor_type": "hand_landmarks",
            "summary": summary,
        },
        "frames": frames,
    }

    return payload, last_timestamp_ms + 33


def download_trimmed_video(attachment_id, temp_dir, retries=3):
    target_path = Path(temp_dir) / f"{attachment_id}.mp4"
    member_name = f"trimmed/{attachment_id}.mp4"

    last_error = None

    for _ in range(retries):
        try:
            with RemoteZip(BUKVA_ZIP_URL) as archive:
                with archive.open(member_name) as input_file, target_path.open(
                    "wb"
                ) as output_file:
                    output_file.write(input_file.read())
            return target_path
        except Exception as error:
            last_error = error

    raise RuntimeError(f"Failed to download {member_name}: {last_error}") from last_error


def load_or_extract_sequence(
    row,
    hand_landmarker,
    cache_dir,
    temp_dir,
    start_timestamp_ms,
):
    attachment_id = row["attachment_id"]
    cache_path = cache_dir / f"{attachment_id}.json"

    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        frame_count = int(payload.get("meta", {}).get("frame_count", 0) or 0)
        fps = float(payload.get("meta", {}).get("fps", 0.0) or 0.0)
        approx_duration_ms = (
            round((frame_count / fps) * 1000) if fps > 0 and frame_count > 0 else frame_count * 33
        )
        return payload, int(start_timestamp_ms) + approx_duration_ms + 33

    video_path = download_trimmed_video(attachment_id, temp_dir)
    payload, next_timestamp_ms = extract_sequence_from_video(
        video_path,
        hand_landmarker,
        start_timestamp_ms=start_timestamp_ms,
    )
    payload["meta"]["source_video"] = f"bukva://trimmed/{attachment_id}.mp4"
    payload["meta"]["attachment_id"] = attachment_id
    payload["meta"]["user_id"] = row["user_id"]
    payload["meta"]["text"] = row["text"]
    cache_path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )

    try:
        video_path.unlink()
    except OSError:
        pass

    return payload, next_timestamp_ms


def build_sample(row, split_name, payload, unit_code):
    summary = payload["meta"]["summary"]
    attachment_id = row["attachment_id"]
    created_at = datetime.now(timezone.utc).isoformat()

    return {
        "sample_id": f"bukva:{attachment_id}",
        "label_id": unit_code,
        "label_type": "alphabet",
        "phrase_id": unit_code,
        "phrase_text": row["text"],
        "entry_type": "alphabet",
        "recognition_level": "alphabet",
        "unit_code": unit_code,
        "category": "Алфавит",
        "sign_language": DEFAULT_SIGN_LANGUAGE,
        "signer_key": f"bukva:{row['user_id']}",
        "user_id": None,
        "user_email": None,
        "duration_ms": round((row["length"] / 30.0) * 1000),
        "dataset_split": split_name,
        "review_status": "approved",
        "quality_score": 5,
        "landmark_sequence_id": f"bukva:{attachment_id}",
        "landmark_frame_count": payload["meta"]["frame_count"],
        "landmark_valid_frame_ratio": summary["valid_frame_ratio"],
        "landmark_missing_hand_ratio": summary["missing_hand_ratio"],
        "landmark_missing_face_ratio": summary["missing_face_ratio"],
        "landmark_missing_pose_ratio": summary["missing_pose_ratio"],
        "landmark_normalization_version": "bukva_hands_only_v1",
        "landmark_file_path": None,
        "landmark_url": None,
        "video_path": f"bukva://trimmed/{attachment_id}.mp4",
        "video_url": None,
        "created_at": created_at,
        "sequence_meta": payload["meta"],
        "sequence": payload["frames"],
    }


def average_metric(samples, key):
    if not samples:
        return 0.0
    return round(sum(float(sample.get(key, 0.0)) for sample in samples) / len(samples), 4)


def build_manifest(train_samples, val_samples, test_samples):
    all_samples = train_samples + val_samples + test_samples
    class_counts = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})

    for sample in train_samples:
        class_counts[sample["phrase_text"]]["train"] += 1
    for sample in val_samples:
        class_counts[sample["phrase_text"]]["val"] += 1
    for sample in test_samples:
        class_counts[sample["phrase_text"]]["test"] += 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": "Bukva",
        "source_url": BUKVA_ZIP_URL,
        "license": "CC BY-SA 4.0",
        "recognition_level": "alphabet",
        "sign_language": DEFAULT_SIGN_LANGUAGE,
        "feature_mode": "hands_only",
        "sample_count": len(all_samples),
        "split_counts": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        },
        "class_count": len(class_counts),
        "classes": [
            {
                "label": label,
                "train": counts["train"],
                "val": counts["val"],
                "test": counts["test"],
            }
            for label, counts in sorted(class_counts.items())
        ],
        "signer_count": len({sample["signer_key"] for sample in all_samples}),
    }


def write_json(path, payload):
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train-per-class", type=int, default=DEFAULT_MAX_TRAIN_PER_CLASS)
    parser.add_argument("--max-val-per-class", type=int, default=DEFAULT_MAX_VAL_PER_CLASS)
    parser.add_argument("--max-test-per-class", type=int, default=DEFAULT_MAX_TEST_PER_CLASS)
    parser.add_argument("--val-user-ratio", type=float, default=DEFAULT_VAL_USER_RATIO)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--skip-unavailable", action="store_true")
    args = parser.parse_args()

    random.seed(SEED)
    ensure_dir(DATASET_DIR)
    ensure_dir(CACHE_DIR)

    unit_code_map = load_unit_code_map()
    annotations = [
        row
        for row in read_annotations(use_cached_only=args.cache_only)
        if row["text"] in unit_code_map
    ]
    train_rows, val_rows, test_rows, val_users = split_rows(
        annotations,
        args.val_user_ratio,
        args.max_train_per_class,
        args.max_val_per_class,
        args.max_test_per_class,
    )

    hand_landmarker = build_hand_landmarker()
    train_samples = []
    val_samples = []
    test_samples = []
    processed = 0
    skipped = 0
    total = len(train_rows) + len(val_rows) + len(test_rows)
    next_timestamp_ms = 0

    try:
        with tempfile.TemporaryDirectory(prefix="bukva-import-") as temp_dir:
            for split_name, rows, target in (
                ("train", train_rows, train_samples),
                ("val", val_rows, val_samples),
                ("test", test_rows, test_samples),
            ):
                for row in rows:
                    try:
                        payload, next_timestamp_ms = load_or_extract_sequence(
                            row,
                            hand_landmarker,
                            CACHE_DIR,
                            temp_dir,
                            next_timestamp_ms,
                        )
                    except Exception as error:
                        if not args.skip_unavailable:
                            raise

                        skipped += 1
                        print(
                            json.dumps(
                                {
                                    "ok": False,
                                    "stage": "skip",
                                    "processed": processed,
                                    "total": total,
                                    "split": split_name,
                                    "label": row["text"],
                                    "attachment_id": row["attachment_id"],
                                    "reason": str(error),
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
                        continue

                    target.append(
                        build_sample(
                            row,
                            split_name,
                            payload,
                            unit_code_map[row["text"]],
                        )
                    )
                    processed += 1

                    if processed % 25 == 0 or processed == total:
                        print(
                            json.dumps(
                                {
                                    "ok": True,
                                    "stage": "processing",
                                    "processed": processed,
                                    "total": total,
                                    "last_label": row["text"],
                                    "split": split_name,
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
    finally:
        hand_landmarker.close()

    manifest = build_manifest(train_samples, val_samples, test_samples)
    summary = {
        "generated_at": manifest["generated_at"],
        "source_dataset": "Bukva",
        "source_url": BUKVA_ZIP_URL,
        "recognition_level": "alphabet",
        "sign_language": DEFAULT_SIGN_LANGUAGE,
        "feature_mode": "hands_only",
        "sample_count": manifest["sample_count"],
        "split_counts": manifest["split_counts"],
        "class_count": manifest["class_count"],
        "signer_count": manifest["signer_count"],
        "quality_summary": {
            "avg_valid_frame_ratio": average_metric(
                train_samples + val_samples + test_samples,
                "landmark_valid_frame_ratio",
            ),
            "avg_missing_hand_ratio": average_metric(
                train_samples + val_samples + test_samples,
                "landmark_missing_hand_ratio",
            ),
        },
        "selection": {
            "max_train_per_class": args.max_train_per_class,
            "max_val_per_class": args.max_val_per_class,
            "max_test_per_class": args.max_test_per_class,
            "val_user_ratio": args.val_user_ratio,
            "val_user_count": len(val_users),
            "cache_only": bool(args.cache_only),
            "skip_unavailable": bool(args.skip_unavailable),
            "skipped_count": skipped,
        },
        "files": {
            "train": "/uploads/datasets/alphabet/latest/train.json",
            "val": "/uploads/datasets/alphabet/latest/val.json",
            "test": "/uploads/datasets/alphabet/latest/test.json",
            "manifest": "/uploads/datasets/alphabet/latest/manifest.json",
            "summary": "/uploads/datasets/alphabet/latest/summary.json",
        },
    }

    write_json(DATASET_DIR / "train.json", train_samples)
    write_json(DATASET_DIR / "val.json", val_samples)
    write_json(DATASET_DIR / "test.json", test_samples)
    write_json(DATASET_DIR / "manifest.json", manifest)
    write_json(DATASET_DIR / "summary.json", summary)

    print(
        json.dumps(
            {
                "ok": True,
                "datasetDir": str(DATASET_DIR),
                "sampleCount": summary["sample_count"],
                "splitCounts": summary["split_counts"],
                "classCount": summary["class_count"],
                "signerCount": summary["signer_count"],
                "skippedCount": skipped,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
