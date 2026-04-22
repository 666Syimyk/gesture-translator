import argparse
import json
import sys
import time
from pathlib import Path


PHRASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = PHRASE_DIR.parents[1]
if str(PHRASE_DIR) not in sys.path:
    sys.path.insert(0, str(PHRASE_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config import load_config, resolve_backend_path
from data import build_dataset_summary, is_usable_sample_summary, write_landmark_sample


def import_capture_stack():
    import cv2
    import mediapipe as mp
    from scripts.extract_landmarks import (
        build_landmarkers,
        build_summary,
        extract_frame_payload,
    )

    return cv2, mp, build_landmarkers, build_summary, extract_frame_payload


def draw_status(cv2, frame, text, color=(0, 255, 0)):
    cv2.putText(
        frame,
        text,
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )


def wait_with_preview(cv2, capture, seconds, label):
    deadline = time.perf_counter() + max(float(seconds), 0.0)
    while time.perf_counter() < deadline:
        success, frame = capture.read()
        if not success:
            return False
        remaining = max(deadline - time.perf_counter(), 0.0)
        draw_status(cv2, frame, f"Get ready: {label} in {remaining:.1f}s", (0, 220, 255))
        cv2.imshow("phrase dataset recorder", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False
    return True


def parse_csv_labels(value):
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def resolve_labels_to_record(args, config):
    requested = []
    if args.all_labels:
        requested.extend(config["phrases"])
    if args.labels:
        requested.extend(parse_csv_labels(args.labels))
    if args.label:
        requested.append(str(args.label).strip())

    deduped = []
    for label in requested:
        if label and label not in deduped:
            deduped.append(label)

    if not deduped:
        raise SystemExit("Pass --label, --labels, or --all-labels.")

    unknown = [label for label in deduped if label not in config["phrases"]]
    if unknown:
        raise SystemExit(
            f"Unknown label(s): {', '.join(unknown)}. Add them to {args.config} or use one of: {', '.join(config['phrases'])}"
        )

    return deduped


def print_split_counts(dataset_dir, labels, config):
    summary = build_dataset_summary(
        dataset_dir,
        config["phrases"],
        quality_config=config.get("quality", {}),
        idle_label=config["inference"].get("idle_label", "none"),
    )
    parts = []
    for label in labels:
        counts = []
        for split in ("train", "val", "test"):
            item = summary["splits"][split][label]
            counts.append(f"{split}:{item['usable_count']}/{item['count']}")
        parts.append(f"{label} -> " + ", ".join(counts))
    print("dataset counts: " + " | ".join(parts))


def capture_landmark_clip(
    cv2,
    mp,
    capture,
    hand_landmarker,
    face_landmarker,
    pose_landmarker,
    extract_frame_payload,
    *,
    label,
    clip_seconds,
    capture_fps,
    clock_start,
):
    frames = []
    frame_index = 0
    frame_interval = 1.0 / max(float(capture_fps), 1.0)
    next_capture_time = time.perf_counter()
    deadline = time.perf_counter() + float(clip_seconds)

    while time.perf_counter() < deadline:
        success, frame = capture.read()
        if not success:
            break

        now = time.perf_counter()
        remaining = max(deadline - now, 0.0)
        draw_status(cv2, frame, f"Recording {label}: {remaining:.1f}s")
        cv2.imshow("phrase dataset recorder", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if now < next_capture_time:
            continue
        next_capture_time = now + frame_interval

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((now - clock_start) * 1000)
        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
        pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        frames.append(
            extract_frame_payload(
                hand_result,
                face_result,
                pose_result,
                frame_index,
                timestamp_ms,
            )
        )
        frame_index += 1

    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PHRASE_DIR / "default_config.json"))
    parser.add_argument("--label")
    parser.add_argument("--labels")
    parser.add_argument("--all-labels", action="store_true")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--dataset-dir")
    parser.add_argument("--camera-index", type=int)
    parser.add_argument("--clip-seconds", type=float)
    parser.add_argument("--capture-fps", type=float)
    parser.add_argument("--countdown", type=float, default=1.0)
    args = parser.parse_args()

    config = load_config(args.config)
    labels_to_record = resolve_labels_to_record(args, config)
    dataset_dir = resolve_backend_path(args.dataset_dir or config["dataset_dir"])
    camera_index = (
        int(args.camera_index)
        if args.camera_index is not None
        else int(config["inference"].get("camera_index", 0))
    )
    clip_seconds = float(args.clip_seconds or config.get("clip_seconds", 2.0))
    capture_fps = float(args.capture_fps or config.get("capture_fps", 15))

    cv2, mp, build_landmarkers, build_summary, extract_frame_payload = import_capture_stack()
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise SystemExit(f"Could not open camera index {camera_index}.")

    hand_landmarker, face_landmarker, pose_landmarker = build_landmarkers()
    clock_start = time.perf_counter()
    written_paths = []
    skipped = []

    print_split_counts(dataset_dir, labels_to_record, config)

    try:
        for label in labels_to_record:
            print(f"recording label={label} split={args.split} samples={args.samples}")
            for sample_index in range(1, int(args.samples) + 1):
                if not wait_with_preview(cv2, capture, args.countdown, label):
                    break

                frames = capture_landmark_clip(
                    cv2,
                    mp,
                    capture,
                    hand_landmarker,
                    face_landmarker,
                    pose_landmarker,
                    extract_frame_payload,
                    label=label,
                    clip_seconds=clip_seconds,
                    capture_fps=capture_fps,
                    clock_start=clock_start,
                )
                summary = build_summary(frames)
                usable, reason = is_usable_sample_summary(
                    label,
                    summary,
                    config.get("quality", {}),
                    idle_label=config["inference"].get("idle_label", "none"),
                )
                if not usable:
                    skipped.append(
                        {
                            "label": label,
                            "sampleIndex": sample_index,
                            "reason": reason,
                            "summary": summary,
                        }
                    )
                    print(f"skipped {label} sample {sample_index}: {reason}")
                    continue

                output_path = write_landmark_sample(
                    dataset_dir,
                    args.split,
                    label,
                    frames,
                    meta={
                        "sample_index": sample_index,
                        "clip_seconds": clip_seconds,
                        "capture_fps": capture_fps,
                        "extractor": "mediapipe_tasks_python_holistic_v2",
                        "summary": summary,
                    },
                )
                written_paths.append(str(output_path))
                print(f"saved {output_path}")
    finally:
        capture.release()
        hand_landmarker.close()
        face_landmarker.close()
        pose_landmarker.close()
        cv2.destroyAllWindows()

    print_split_counts(dataset_dir, labels_to_record, config)
    print(
        json.dumps(
            {
                "ok": True,
                "labels": labels_to_record,
                "split": args.split,
                "datasetDir": str(dataset_dir),
                "savedCount": len(written_paths),
                "savedPaths": written_paths,
                "skippedCount": len(skipped),
                "skipped": skipped,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
