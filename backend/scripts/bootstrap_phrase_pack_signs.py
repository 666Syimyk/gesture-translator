import argparse
import json
import math
import os
import subprocess
import sys
import urllib.request
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

from bootstrap_slovo_signs import (
    DEFAULT_ANNOTATIONS_URL,
    DEFAULT_CACHE_DIR as DEFAULT_SLOVO_CACHE_DIR,
    DEFAULT_CONSTANTS_URL,
    DEFAULT_LANDMARKS_URL,
    choose_validation_signers,
    convert_frames,
    extract_annotations_csv_from_zip,
    load_annotations,
    load_slovo_constants,
    open_landmarks_stream,
)
import ijson


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "backend" / "uploads" / "datasets" / "sign" / "phrase_pack_20260412"
DEFAULT_SIGNFLOW_DIR = REPO_ROOT / "backend" / "uploads" / "external" / "signflow"
EXTRACT_SCRIPT = REPO_ROOT / "backend" / "scripts" / "extract_landmarks.py"

SOURCE_NOTE = (
    "\u0417\u043d\u0430\u043a \u043f\u043e\u0434\u043a\u0440\u0435\u043f\u043b\u0435\u043d \u0440\u0435\u0430\u043b\u044c\u043d\u044b\u043c\u0438 "
    "Slovo-\u0437\u0430\u043f\u0438\u0441\u044f\u043c\u0438 \u0438\u043b\u0438 SignFlow seed-\u0440\u043e\u043b\u0438\u043a\u0430\u043c\u0438. "
    "Seed-\u0432\u0430\u0440\u0438\u0430\u043d\u0442\u044b \u043d\u0435 \u044f\u0432\u043b\u044f\u044e\u0442\u0441\u044f \u043d\u0435\u0437\u0430\u0432\u0438\u0441\u0438\u043c\u044b\u043c\u0438 "
    "\u043f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u0442\u0435\u043b\u044c\u0441\u043a\u0438\u043c\u0438 \u043a\u043b\u0438\u043f\u0430\u043c\u0438."
)

SIGNFLOW_TRAIN_VARIANTS = [
    ("full", 0, 100),
    ("trim_02_98", 2, 98),
    ("trim_04_96", 4, 96),
    ("trim_06_94", 6, 94),
    ("trim_08_92", 8, 92),
    ("trim_10_90", 10, 90),
    ("trim_12_88", 12, 88),
    ("trim_14_86", 14, 86),
    ("trim_16_84", 16, 84),
    ("trim_18_82", 18, 82),
    ("trim_03_97", 3, 97),
    ("trim_05_95", 5, 95),
    ("trim_07_93", 7, 93),
    ("trim_09_91", 9, 91),
    ("trim_11_89", 11, 89),
]

SIGNFLOW_VAL_VARIANTS = [
    ("val_07_93", 7, 93),
    ("val_15_85", 15, 85),
    ("val_22_78", 22, 78),
]

SIGNFLOW_TEST_VARIANTS = [
    ("test_03_97", 3, 97),
    ("test_10_90", 10, 90),
    ("test_18_82", 18, 82),
    ("test_24_76", 24, 76),
    ("test_30_70", 30, 70),
]

LABEL_SPECS = {
    "\u041c\u0443\u0436\u0447\u0438\u043d\u0430": {
        "unit_code": "SIGN_PHRASEPACK_MUZHCHINA",
        "slug": "muzhchina",
        "signflow_page": "https://signflow.ru/muzhchina",
        "signflow_video": "https://static.signflow.ru/p-rsl-data/video/3e8729af-8881-41e3-89bd-36c3a0320c79.mp4",
        "slovo_labels": [],
    },
    "\u0416\u0435\u043d\u0449\u0438\u043d\u0430": {
        "unit_code": "SIGN_PHRASEPACK_ZHENSHCHINA",
        "slug": "zhenshchina",
        "signflow_page": "https://signflow.ru/zhenshchina",
        "signflow_video": "https://static.signflow.ru/p-rsl-data/video/c4b8cd59-9eb2-49e5-9e45-c2d6636e1144.mp4",
        "slovo_labels": ["\u0436\u0435\u043d\u0449\u0438\u043d\u0430"],
    },
    "\u0421\u043e\u043b\u043d\u0446\u0435": {
        "unit_code": "SIGN_PHRASEPACK_SOLNTSE",
        "slug": "solntse",
        "signflow_page": "https://signflow.ru/solntse",
        "signflow_video": "https://static.signflow.ru/p-rsl-data/video/1568b4e6-730c-446b-88ca-ccf12a12aa74.mp4",
        "slovo_labels": ["\u0441\u043e\u043b\u043d\u0446\u0435"],
    },
}


def download_if_missing(url, target_path):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and target_path.stat().st_size > 0:
        return target_path

    urllib.request.urlretrieve(url, target_path)
    return target_path


def extract_landmarks(video_path, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        with output_path.open("r", encoding="utf-8") as input_file:
            return json.load(input_file)

    command = [
        sys.executable,
        str(EXTRACT_SCRIPT),
        "--input",
        str(video_path),
        "--output",
        str(output_path),
    ]
    env = {
        **os.environ,
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    subprocess.run(command, cwd=str(REPO_ROOT), env=env, check=True)

    with output_path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def trim_empty_edges(frames):
    start = 0
    end = len(frames)

    while start < end and not has_visible_content(frames[start]):
        start += 1

    while end > start and not has_visible_content(frames[end - 1]):
        end -= 1

    return frames[start:end] if start < end else frames


def has_visible_content(frame):
    quality = frame.get("quality") or {}
    return any(
        bool(quality.get(key))
        for key in ("has_left_hand", "has_right_hand", "has_face", "has_pose")
    )


def reindex_frames(frames, fps=30.0):
    rebuilt = []
    for index, frame in enumerate(frames):
        next_frame = deepcopy(frame)
        next_frame["frame_index"] = index
        next_frame["timestamp_ms"] = int(round(index * (1000.0 / fps)))
        rebuilt.append(next_frame)
    return rebuilt


def keep_hands_only(frames):
    normalized = []
    for frame in frames:
        next_frame = deepcopy(frame)
        next_frame["face"] = []
        next_frame["pose"] = []
        next_frame["pose_world"] = []
        quality = dict(next_frame.get("quality") or {})
        quality["has_face"] = False
        quality["has_pose"] = False
        next_frame["quality"] = quality
        normalized.append(next_frame)
    return normalized


def slice_frames(frames, start_percent, end_percent, min_frames=48):
    if not frames:
        return []

    frame_count = len(frames)
    start = int(round(frame_count * (start_percent / 100.0)))
    end = int(round(frame_count * (end_percent / 100.0)))
    start = max(0, min(start, frame_count - 1))
    end = max(start + 1, min(end, frame_count))

    sliced = frames[start:end]
    if len(sliced) < min_frames and frame_count >= min_frames:
        deficit = min_frames - len(sliced)
        pad_left = min(start, deficit // 2)
        pad_right = min(frame_count - end, deficit - pad_left)
        start -= pad_left
        end += pad_right
        if end - start < min_frames:
            shortfall = min_frames - (end - start)
            start = max(0, start - shortfall)
            end = min(frame_count, end + max(0, min_frames - (end - start)))
        sliced = frames[start:end]

    return reindex_frames(sliced)


def build_quality_summary(frames):
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

    left_hand_frames = sum(
        1 for frame in frames if (frame.get("quality") or {}).get("has_left_hand")
    )
    right_hand_frames = sum(
        1 for frame in frames if (frame.get("quality") or {}).get("has_right_hand")
    )
    face_frames = sum(
        1 for frame in frames if (frame.get("quality") or {}).get("has_face")
    )
    pose_frames = sum(
        1 for frame in frames if (frame.get("quality") or {}).get("has_pose")
    )
    valid_frames = sum(1 for frame in frames if has_visible_content(frame))
    missing_hand_frames = sum(
        1
        for frame in frames
        if not (frame.get("quality") or {}).get("has_left_hand")
        and not (frame.get("quality") or {}).get("has_right_hand")
    )

    return {
        "frame_count": frame_count,
        "valid_frame_ratio": round(valid_frames / frame_count, 4),
        "left_hand_ratio": round(left_hand_frames / frame_count, 4),
        "right_hand_ratio": round(right_hand_frames / frame_count, 4),
        "face_ratio": round(face_frames / frame_count, 4),
        "pose_ratio": round(pose_frames / frame_count, 4),
        "missing_hand_ratio": round(missing_hand_frames / frame_count, 4),
        "missing_face_ratio": round(1 - (face_frames / frame_count), 4),
        "missing_pose_ratio": round(1 - (pose_frames / frame_count), 4),
    }


def build_signflow_seed_sample(display_text, unit_code, slug, frames, split, variant_name, variant_index, video_url):
    summary = build_quality_summary(frames)
    relative_landmark_path = f"backend/uploads/external/signflow/{slug}_seed_landmarks.json"
    relative_video_path = f"backend/uploads/external/signflow/{slug}_seed.mp4"
    signer_key = f"signflow_seed_{slug}"

    return {
        "sample_id": f"signflow:{slug}:{variant_name}:{variant_index:02d}",
        "label_id": unit_code,
        "label_type": "sign",
        "phrase_id": unit_code,
        "phrase_text": display_text,
        "entry_type": "sign",
        "recognition_level": "sign",
        "unit_code": unit_code,
        "category": "SignFlow",
        "sign_language": "rsl",
        "signer_key": signer_key,
        "user_id": signer_key,
        "user_email": None,
        "duration_ms": int(round(len(frames) * (1000.0 / 30.0))),
        "dataset_split": split,
        "review_status": "approved_seed",
        "quality_score": 4,
        "landmark_sequence_id": f"signflow:{slug}:{variant_name}",
        "landmark_frame_count": len(frames),
        "landmark_valid_frame_ratio": summary["valid_frame_ratio"],
        "landmark_missing_hand_ratio": summary["missing_hand_ratio"],
        "landmark_missing_face_ratio": summary["missing_face_ratio"],
        "landmark_missing_pose_ratio": summary["missing_pose_ratio"],
        "landmark_normalization_version": "signflow_mediapipe_holistic_seed_v2",
        "landmark_file_path": relative_landmark_path,
        "landmark_url": None,
        "video_path": relative_video_path,
        "video_url": video_url,
        "created_at": None,
        "sequence_meta": {
            "fps": 30.0,
            "frame_count": len(frames),
            "source_frame_count": len(frames),
            "processed_frame_count": len(frames),
            "extractor": "mediapipe_tasks_python_holistic_v2",
            "extractor_type": "holistic_landmarks",
            "summary": summary,
            "source_video": video_url,
            "source_page": LABEL_SPECS[display_text]["signflow_page"],
            "source_title": f"SignFlow - {display_text}",
            "source_note": (
                "\u041e\u0434\u0438\u043d \u0441\u043b\u043e\u0432\u0430\u0440\u043d\u044b\u0439 seed-\u0440\u043e\u043b\u0438\u043a; "
                "variant splits are temporal trims and are not fully independent."
            ),
            "variant_name": variant_name,
            "text": display_text,
            "target_text": display_text,
            "is_seed_label": True,
        },
        "sequence": frames,
    }


def build_slovo_phrase_pack_sample(annotation, class_id, frames, split, display_text, unit_code):
    summary = {
        "frame_count": len(frames),
        "valid_frame_ratio": 1.0 if frames else 0.0,
        "left_hand_ratio": round(
            sum(1 for frame in frames if frame.get("left_hand")) / len(frames), 4
        )
        if frames
        else 0.0,
        "right_hand_ratio": round(
            sum(1 for frame in frames if frame.get("right_hand")) / len(frames), 4
        )
        if frames
        else 0.0,
        "face_ratio": 0.0,
        "pose_ratio": 0.0,
        "missing_hand_ratio": 0.0 if frames else 1.0,
        "missing_face_ratio": 1.0,
        "missing_pose_ratio": 1.0,
    }
    signer_key = annotation["user_id"]

    return {
        "sample_id": f"slovo-pack:{annotation['attachment_id']}",
        "label_id": unit_code,
        "label_type": "sign",
        "phrase_id": unit_code,
        "phrase_text": display_text,
        "entry_type": "sign",
        "recognition_level": "sign",
        "unit_code": unit_code,
        "category": "PhrasePack",
        "sign_language": "rsl",
        "signer_key": signer_key,
        "user_id": signer_key,
        "user_email": None,
        "duration_ms": int(math.ceil(float(annotation["length"]) * 1000 / 30.0)),
        "dataset_split": split,
        "review_status": "approved",
        "quality_score": 5,
        "landmark_sequence_id": f"slovo-pack:{annotation['attachment_id']}",
        "landmark_frame_count": len(frames),
        "landmark_valid_frame_ratio": summary["valid_frame_ratio"],
        "landmark_missing_hand_ratio": summary["missing_hand_ratio"],
        "landmark_missing_face_ratio": summary["missing_face_ratio"],
        "landmark_missing_pose_ratio": summary["missing_pose_ratio"],
        "landmark_normalization_version": "slovo_hands_only_v1",
        "landmark_file_path": None,
        "landmark_url": None,
        "video_path": f"slovo://{split}/{annotation['attachment_id']}.mp4",
        "video_url": None,
        "created_at": None,
        "sequence_meta": {
            "fps": 30.0,
            "frame_count": len(frames),
            "source_frame_count": len(frames),
            "processed_frame_count": len(frames),
            "extractor": "slovo_mediapipe_landmarks",
            "extractor_type": "hand_landmarks",
            "summary": summary,
            "source_video": f"slovo://{split}/{annotation['attachment_id']}.mp4",
            "attachment_id": annotation["attachment_id"],
            "user_id": signer_key,
            "text": display_text,
            "source_text": annotation["text"],
            "target_text": display_text,
            "source_dataset": "Slovo",
            "class_id": class_id,
            "is_seed_label": False,
        },
        "sequence": frames,
    }


def build_signflow_variants(spec, signflow_dir):
    slug = spec["slug"]
    video_path = signflow_dir / f"{slug}_seed.mp4"
    landmarks_path = signflow_dir / f"{slug}_seed_landmarks.json"
    download_if_missing(spec["signflow_video"], video_path)
    payload = extract_landmarks(video_path, landmarks_path)
    frames = keep_hands_only(trim_empty_edges(payload.get("frames") or []))
    if not frames:
        raise SystemExit(f"SignFlow seed for {slug} did not produce any frames")

    samples = {"train": [], "val": [], "test": []}
    plans = {
        "train": SIGNFLOW_TRAIN_VARIANTS,
        "val": SIGNFLOW_VAL_VARIANTS,
        "test": SIGNFLOW_TEST_VARIANTS,
    }

    for split, variants in plans.items():
        for index, (variant_name, start_percent, end_percent) in enumerate(variants):
            variant_frames = slice_frames(frames, start_percent, end_percent)
            samples[split].append(
                build_signflow_seed_sample(
                    display_text=spec["display_text"],
                    unit_code=spec["unit_code"],
                    slug=slug,
                    frames=variant_frames,
                    split=split,
                    variant_name=variant_name,
                    variant_index=index,
                    video_url=spec["signflow_video"],
                )
            )

    return samples


def build_slovo_samples(spec, annotations, constants, cache_dir, landmarks_url):
    if not spec["slovo_labels"]:
        return {"train": [], "val": [], "test": []}

    class_by_text = {value: key for key, value in constants.items()}
    requested_labels = set(spec["slovo_labels"])
    target_rows = [row for row in annotations if row["text"] in requested_labels]
    if not target_rows:
        return {"train": [], "val": [], "test": []}

    train_rows = [row for row in target_rows if row["train"] == "True"]
    val_signers = choose_validation_signers(train_rows, 0.2)
    attachment_to_row = {row["attachment_id"]: row for row in target_rows}
    sequences = {}

    with open_landmarks_stream(cache_dir, landmarks_url) as response:
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

            sequences[attachment_id] = build_slovo_phrase_pack_sample(
                annotation=annotation,
                class_id=class_by_text[annotation["text"]],
                frames=convert_frames(raw_frames),
                split=split,
                display_text=spec["display_text"],
                unit_code=spec["unit_code"],
            )

            if len(sequences) == len(attachment_to_row):
                break

    return {
        "train": [
            sample for sample in sequences.values() if sample["dataset_split"] == "train"
        ],
        "val": [
            sample for sample in sequences.values() if sample["dataset_split"] == "val"
        ],
        "test": [
            sample for sample in sequences.values() if sample["dataset_split"] == "test"
        ],
    }


def write_dataset(output_dir, train_samples, val_samples, test_samples):
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, samples in (
        ("train", train_samples),
        ("val", val_samples),
        ("test", test_samples),
    ):
        (output_dir / f"{split}.json").write_text(
            json.dumps(samples, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    all_samples = train_samples + val_samples + test_samples
    coverage = Counter(sample["phrase_text"] for sample in all_samples)
    split_coverage = {}
    for split_name, split_samples in (
        ("train", train_samples),
        ("val", val_samples),
        ("test", test_samples),
    ):
        split_coverage[f"{split_name}_coverage"] = dict(
            sorted(Counter(sample["phrase_text"] for sample in split_samples).items())
        )

    source_breakdown = defaultdict(lambda: defaultdict(int))
    for sample in all_samples:
        phrase_text = sample["phrase_text"]
        source_breakdown[phrase_text][sample["category"]] += 1

    manifest = {
        "source_dataset": "Phrase pack boost: Slovo + SignFlow seed",
        "recognition_level": "sign",
        "feature_mode": "mixed_hands_and_holistic",
        "sample_count": len(all_samples),
        "split_counts": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        },
        "class_count": len(coverage),
        "coverage": dict(sorted(coverage.items())),
        **split_coverage,
        "source_breakdown": {
            label: dict(sorted(counts.items()))
            for label, counts in sorted(source_breakdown.items())
        },
        "external_sources": {
            "slovo": "https://github.com/hukenovs/slovo",
            "signflow": "https://signflow.ru/phrases",
            "signflow_words": {
                label: spec["signflow_page"]
                for label, spec in LABEL_SPECS.items()
            },
        },
        "seed_only_labels": [
            label
            for label, spec in LABEL_SPECS.items()
            if not spec["slovo_labels"]
        ],
        "notes": SOURCE_NOTE,
    }

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--signflow-dir", default=str(DEFAULT_SIGNFLOW_DIR))
    parser.add_argument("--slovo-cache-dir", default=str(DEFAULT_SLOVO_CACHE_DIR))
    parser.add_argument("--annotations-zip-url", default=DEFAULT_ANNOTATIONS_URL)
    parser.add_argument("--landmarks-url", default=DEFAULT_LANDMARKS_URL)
    parser.add_argument("--constants-url", default=DEFAULT_CONSTANTS_URL)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    signflow_dir = Path(args.signflow_dir)
    slovo_cache_dir = Path(args.slovo_cache_dir)

    annotations_path = extract_annotations_csv_from_zip(
        args.annotations_zip_url,
        slovo_cache_dir / "annotations.csv",
    )
    annotations = load_annotations(annotations_path)
    constants = load_slovo_constants(args.constants_url)

    split_buckets = {"train": [], "val": [], "test": []}
    label_stats = {}

    for display_text, raw_spec in LABEL_SPECS.items():
        spec = {**raw_spec, "display_text": display_text}
        signflow_samples = build_signflow_variants(spec, signflow_dir)
        slovo_samples = build_slovo_samples(
            spec,
            annotations=annotations,
            constants=constants,
            cache_dir=slovo_cache_dir,
            landmarks_url=args.landmarks_url,
        )

        for split in ("train", "val", "test"):
            split_buckets[split].extend(slovo_samples[split])
            split_buckets[split].extend(signflow_samples[split])

        label_stats[display_text] = {
            "slovo": {
                split: len(slovo_samples[split]) for split in ("train", "val", "test")
            },
            "signflow": {
                split: len(signflow_samples[split]) for split in ("train", "val", "test")
            },
        }

    write_dataset(
        output_dir,
        train_samples=split_buckets["train"],
        val_samples=split_buckets["val"],
        test_samples=split_buckets["test"],
    )

    print(
        json.dumps(
            {
                "ok": True,
                "outputDir": str(output_dir),
                "splitCounts": {
                    split: len(samples)
                    for split, samples in split_buckets.items()
                },
                "labelStats": label_stats,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
