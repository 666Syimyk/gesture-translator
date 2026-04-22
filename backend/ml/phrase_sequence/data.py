import json
from datetime import datetime, timezone
from pathlib import Path

SPLITS = ("train", "val", "test")


def make_class_entry(label, recognition_level="phrase"):
    normalized_label = str(label).strip()
    return {
        "key": f"{recognition_level}::{normalized_label}",
        "text": normalized_label,
        "recognition_level": recognition_level,
        "unit_code": None,
    }


def read_json(path):
    with Path(path).open("r", encoding="utf-8-sig") as input_file:
        return json.load(input_file)


def write_json(path, payload):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
    return output_path


def payload_to_sample(payload, label, source_path=None, recognition_level="phrase"):
    frames = payload.get("sequence", payload.get("frames", []))
    meta = payload.get("meta", {})
    normalized_label = str(label).strip()

    return {
        "sequence": frames or [],
        "phrase_text": normalized_label,
        "recognition_level": recognition_level,
        "unit_code": None,
        "source_path": str(source_path) if source_path else None,
        "meta": meta,
    }


def load_split(dataset_dir, split, labels, recognition_level="phrase"):
    split_dir = Path(dataset_dir) / split
    samples = []

    for label in labels:
        label_dir = split_dir / label
        if not label_dir.exists():
            continue

        for sample_path in sorted(label_dir.glob("*.json")):
            payload = read_json(sample_path)
            samples.append(
                payload_to_sample(
                    payload,
                    label,
                    source_path=sample_path,
                    recognition_level=recognition_level,
                )
            )

    return samples


def load_dataset_splits(dataset_dir, labels, recognition_level="phrase"):
    return {
        split: load_split(dataset_dir, split, labels, recognition_level)
        for split in SPLITS
    }


def build_class_entries(labels, recognition_level="phrase"):
    return [make_class_entry(label, recognition_level) for label in labels]


def build_label_to_index(class_entries):
    return {entry["key"]: index for index, entry in enumerate(class_entries)}


def write_landmark_sample(
    dataset_dir,
    split,
    label,
    frames,
    *,
    meta=None,
    prefix="sample",
):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_path = Path(dataset_dir) / split / label / f"{prefix}_{timestamp}.json"
    payload = {
        "meta": {
            **(meta or {}),
            "label": label,
            "split": split,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "storage": "landmark_sequence_json",
        },
        "frames": frames,
    }
    return write_json(output_path, payload)


def summarize_frames(frames):
    frame_count = len(frames or [])
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
        1 for frame in frames if frame.get("left_hand") or frame.get("quality", {}).get("has_left_hand")
    )
    right_hand_frames = sum(
        1 for frame in frames if frame.get("right_hand") or frame.get("quality", {}).get("has_right_hand")
    )
    face_frames = sum(
        1 for frame in frames if frame.get("face") or frame.get("quality", {}).get("has_face")
    )
    pose_frames = sum(
        1 for frame in frames if frame.get("pose") or frame.get("quality", {}).get("has_pose")
    )
    valid_frames = sum(
        1
        for frame in frames
        if frame.get("left_hand")
        or frame.get("right_hand")
        or frame.get("face")
        or frame.get("pose")
        or any(frame.get("quality", {}).values())
    )
    missing_hand_frames = frame_count - sum(
        1
        for frame in frames
        if frame.get("left_hand")
        or frame.get("right_hand")
        or frame.get("quality", {}).get("has_left_hand")
        or frame.get("quality", {}).get("has_right_hand")
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


def get_sample_summary(payload):
    meta_summary = (payload.get("meta") or {}).get("summary")
    if isinstance(meta_summary, dict):
        return meta_summary
    return summarize_frames(payload.get("sequence", payload.get("frames", [])))


def is_usable_sample_summary(label, summary, quality_config, idle_label="none"):
    frame_count = int(summary.get("frame_count", 0) or 0)
    min_frame_count = int(quality_config.get("min_frame_count", 1) or 1)
    if frame_count < min_frame_count:
        return False, f"too_short:{frame_count}<{min_frame_count}"

    if str(label).strip() == str(idle_label).strip():
        return True, "ok_idle"

    if not quality_config.get("reject_empty_non_idle", True):
        return True, "ok"

    valid_frame_ratio = float(summary.get("valid_frame_ratio", 0.0) or 0.0)
    left_hand_ratio = float(summary.get("left_hand_ratio", 0.0) or 0.0)
    right_hand_ratio = float(summary.get("right_hand_ratio", 0.0) or 0.0)
    hand_ratio = left_hand_ratio + right_hand_ratio
    min_valid_frame_ratio = float(quality_config.get("min_valid_frame_ratio", 0.0) or 0.0)
    min_hand_frame_ratio = float(quality_config.get("min_hand_frame_ratio", 0.0) or 0.0)

    if valid_frame_ratio < min_valid_frame_ratio:
        return False, f"low_valid_ratio:{valid_frame_ratio:.3f}<{min_valid_frame_ratio:.3f}"
    if hand_ratio < min_hand_frame_ratio:
        return False, f"low_hand_ratio:{hand_ratio:.3f}<{min_hand_frame_ratio:.3f}"

    return True, "ok"


def build_dataset_summary(dataset_dir, labels, quality_config=None, idle_label="none"):
    dataset_path = Path(dataset_dir)
    quality = quality_config or {}
    summary = {
        "dataset_dir": str(dataset_path),
        "labels": labels,
        "splits": {},
        "totals": {split: 0 for split in SPLITS},
        "total": 0,
    }

    for split in SPLITS:
        split_payload = {}
        for label in labels:
            label_dir = dataset_path / split / label
            files = sorted(label_dir.glob("*.json")) if label_dir.exists() else []
            usable_count = 0
            rejected = []

            for sample_path in files:
                try:
                    payload = read_json(sample_path)
                    sample_summary = get_sample_summary(payload)
                    usable, reason = is_usable_sample_summary(
                        label,
                        sample_summary,
                        quality,
                        idle_label=idle_label,
                    )
                    if usable:
                        usable_count += 1
                    else:
                        rejected.append(
                            {
                                "path": str(sample_path),
                                "reason": reason,
                                "summary": sample_summary,
                            }
                        )
                except Exception as error:
                    rejected.append(
                        {
                            "path": str(sample_path),
                            "reason": f"read_error:{error}",
                            "summary": None,
                        }
                    )

            split_payload[label] = {
                "count": len(files),
                "usable_count": usable_count,
                "rejected_count": len(rejected),
                "rejected": rejected,
            }
            summary["totals"][split] += len(files)
            summary["total"] += len(files)

        summary["splits"][split] = split_payload

    return summary
