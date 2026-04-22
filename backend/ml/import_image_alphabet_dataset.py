import argparse
import json
import random
from pathlib import Path

import cv2
import mediapipe as mp


DEFAULT_SEED = 42
DEFAULT_SPLIT = (0.8, 0.1, 0.1)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_label_map(path):
    if not path:
        return {}

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    normalized = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            unit_code = value.get("unit_code") or value.get("unitCode")
            text = value.get("text")
        else:
            unit_code = value
            text = None
        normalized[str(key).strip()] = {
            "unit_code": str(unit_code).strip() if unit_code else "",
            "text": str(text).strip() if text else "",
        }

    return normalized


def default_unit_code(label):
    normalized = label.strip().upper()
    if len(normalized) == 1 and normalized.isalpha():
        return f"LETTER_{normalized}"
    return f"LETTER_{normalized}"


def list_images(root):
    items = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            items.append(path)
    return items


def split_items(items, split):
    random.shuffle(items)
    total = len(items)
    train_count = int(total * split[0])
    val_count = int(total * split[1])
    train = items[:train_count]
    val = items[train_count : train_count + val_count]
    test = items[train_count + val_count :]
    return train, val, test


def build_sample(label, unit_code, text, landmarks, handedness_label, handedness_score):
    frame = {
        "frame_index": 0,
        "timestamp_ms": 0,
        "left_hand": [],
        "right_hand": [],
        "handedness": [],
    }

    if handedness_label:
        frame["handedness"].append(
            {
                "label": handedness_label,
                "score": float(handedness_score or 0.0),
            }
        )

    if handedness_label == "Left":
        frame["left_hand"] = landmarks
    else:
        frame["right_hand"] = landmarks

    return {
        "recognition_level": "alphabet",
        "phrase_text": text or label,
        "unit_code": unit_code,
        "sequence": [frame],
    }


def extract_hand_landmarks(image_bgr, hands):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    if not result.multi_hand_landmarks:
        return None

    landmarks = result.multi_hand_landmarks[0].landmark
    handedness = result.multi_handedness[0].classification[0]
    handedness_label = handedness.label
    handedness_score = handedness.score

    return [
        {"x": float(point.x), "y": float(point.y), "z": float(point.z)}
        for point in landmarks
    ], handedness_label, handedness_score


def process_group(label, files, label_map, hands):
    unit_code = label_map.get(label, {}).get("unit_code") or default_unit_code(label)
    text = label_map.get(label, {}).get("text") or label
    samples = []
    skipped = 0

    for file_path in files:
        image_bgr = cv2.imread(str(file_path))
        if image_bgr is None:
            skipped += 1
            continue

        extracted = extract_hand_landmarks(image_bgr, hands)
        if not extracted:
            skipped += 1
            continue

        landmarks, handedness_label, handedness_score = extracted
        samples.append(
            build_sample(label, unit_code, text, landmarks, handedness_label, handedness_score)
        )

    return samples, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-map", default="")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-per-class", type=int, default=0)
    parser.add_argument(
        "--split",
        default="0.8,0.1,0.1",
        help="Train/val/test split ratios, comma separated.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ratios = [float(value) for value in args.split.split(",")]
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise SystemExit("Split must have three ratios that sum to 1.")

    random.seed(args.seed)
    label_map = load_label_map(args.label_map)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    all_train = []
    all_val = []
    all_test = []
    skipped_total = 0
    counts = {}

    for label_dir in sorted([p for p in input_dir.iterdir() if p.is_dir()]):
        label = label_dir.name
        files = list_images(label_dir)
        if args.max_per_class and len(files) > args.max_per_class:
            files = files[: args.max_per_class]

        train_files, val_files, test_files = split_items(files, ratios)
        train_samples, skipped_train = process_group(label, train_files, label_map, hands)
        val_samples, skipped_val = process_group(label, val_files, label_map, hands)
        test_samples, skipped_test = process_group(label, test_files, label_map, hands)

        all_train.extend(train_samples)
        all_val.extend(val_samples)
        all_test.extend(test_samples)

        skipped_total += skipped_train + skipped_val + skipped_test
        counts[label] = {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        }

    hands.close()

    (output_dir / "train.json").write_text(
        json.dumps(all_train, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "val.json").write_text(
        json.dumps(all_val, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "test.json").write_text(
        json.dumps(all_test, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "train_count": len(all_train),
        "val_count": len(all_val),
        "test_count": len(all_test),
        "skipped_total": skipped_total,
        "per_label_counts": counts,
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
