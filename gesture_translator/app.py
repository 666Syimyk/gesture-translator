from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gesture_translator.config import (
    CONFIDENCE_THRESHOLD,
    COOLDOWN_SECONDS,
    MAX_SEQUENCE_LENGTH,
    MIN_SEQUENCE_LENGTH,
    SEQUENCE_LENGTH,
    STABLE_VOTES_REQUIRED,
    SMOOTHING_WINDOW,
    ensure_runtime_dirs,
    load_labels,
    normalize_label,
    load_runtime_config,
)
from gesture_translator.core.cooldown import CooldownGate
from gesture_translator.core.smoother import PredictionSmoother
from gesture_translator.data.recorder import save_sequence_sample, validate_sequence
from gesture_translator.models.infer import HybridInferenceEngine
from gesture_translator.models.train import train_gesture_model
from gesture_translator.ui.overlay import draw_landmarks_overlay, draw_prediction_panel
from gesture_translator.ui.screens import draw_live_status, draw_record_status
from gesture_translator.utils.logger import get_logger


LOGGER = get_logger("gesture_translator.app")


def run_record_mode(args: argparse.Namespace) -> int:
    from gesture_translator.core.camera import CameraStream
    from gesture_translator.core.mp_extractor import MediaPipeHolisticExtractor

    ensure_runtime_dirs()
    labels = set(load_labels(args.labels_path))
    normalized_label = normalize_label(args.label, args.labels_path)
    if normalized_label not in labels:
        raise SystemExit(f"Unknown label '{args.label}'. Available labels: {', '.join(sorted(labels))}")
    args.label = normalized_label
    if not (MIN_SEQUENCE_LENGTH <= int(args.sequence_length) <= MAX_SEQUENCE_LENGTH):
        raise SystemExit(
            f"Invalid --sequence-length={args.sequence_length}. "
            f"Expected range: {MIN_SEQUENCE_LENGTH}..{MAX_SEQUENCE_LENGTH}"
        )

    camera = CameraStream(camera_index=args.camera_index, mirror=args.mirror_preview)
    extractor = MediaPipeHolisticExtractor()
    samples_saved = 0
    countdown_started_at = None
    recording_frames = []
    frame_index = 0
    status_text = "Press SPACE to start recording"
    target_samples = max(1, int(args.samples))

    try:
        while samples_saved < target_samples:
            success, frame = camera.read()
            if not success:
                break

            timestamp_ms = int(time.perf_counter() * 1000)
            payload = extractor.process(frame, frame_index=frame_index, timestamp_ms=timestamp_ms)
            frame_index += 1
            draw_landmarks_overlay(frame, payload)

            key = camera.wait_key()
            if key == ord("q"):
                break
            if key == ord(" "):
                countdown_started_at = time.perf_counter()
                recording_frames = []
                status_text = "Get ready: neutral -> gesture -> neutral"

            countdown = None
            is_recording = False
            if countdown_started_at is not None:
                elapsed = time.perf_counter() - countdown_started_at
                if elapsed < args.countdown_seconds:
                    countdown = max(1, args.countdown_seconds - int(elapsed))
                else:
                    is_recording = True
                    recording_frames.append(payload)
                    status_text = f"Recording {len(recording_frames)}/{args.sequence_length}"

            if is_recording and len(recording_frames) >= args.sequence_length:
                verdict = validate_sequence(recording_frames, args.label)
                if verdict["ok"]:
                    output_path = save_sequence_sample(
                        label=args.label,
                        frames=recording_frames,
                        output_dir=Path(args.dataset_dir),
                        meta={
                            "mode": "dataset_recording",
                            "camera_index": args.camera_index,
                            "mirror_preview": bool(args.mirror_preview),
                        },
                    )
                    samples_saved += 1
                    status_text = f"Saved {output_path.name}"
                else:
                    status_text = f"Rejected: {verdict['reason']}"

                countdown_started_at = None
                recording_frames = []

            draw_record_status(
                frame,
                label=args.label,
                saved_count=samples_saved,
                target_count=target_samples,
                countdown_value=countdown,
                status_text=status_text,
            )
            camera.show("gesture translator - record", frame)
    finally:
        extractor.close()
        camera.close()

    LOGGER.info("record mode finished: saved=%s", samples_saved)
    return 0


def run_live_mode(args: argparse.Namespace) -> int:
    from gesture_translator.core.camera import CameraStream
    from gesture_translator.core.mp_extractor import MediaPipeHolisticExtractor

    ensure_runtime_dirs()
    camera = CameraStream(camera_index=args.camera_index, mirror=args.mirror_preview)
    extractor = MediaPipeHolisticExtractor()
    engine = HybridInferenceEngine(
        artifact_dir=args.artifact_dir,
        sequence_length=args.sequence_length,
        confidence_threshold=args.confidence_threshold,
    )
    smoother = PredictionSmoother(
        window_size=args.smoothing_window,
        stable_votes_required=args.stable_votes_required,
    )
    cooldown = CooldownGate(args.cooldown_seconds)
    frames = deque(maxlen=args.sequence_length)
    frame_index = 0
    current_prediction = {"label": "none", "confidence": 0.0, "debug": {}}

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            timestamp_ms = int(time.perf_counter() * 1000)
            payload = extractor.process(frame, frame_index=frame_index, timestamp_ms=timestamp_ms)
            frame_index += 1
            frames.append(payload)
            draw_landmarks_overlay(frame, payload)

            if len(frames) >= args.min_sequence_length:
                current_prediction = engine.predict(list(frames))
                stable_prediction = smoother.update(
                    current_prediction["label"],
                    current_prediction["confidence"],
                    extra=current_prediction,
                )
                if stable_prediction and cooldown.allow(stable_prediction["label"]):
                    print(
                        json.dumps(
                            {
                                "event": "gesture",
                                "label": stable_prediction["label"],
                                "confidence": round(stable_prediction["confidence"], 4),
                            },
                            ensure_ascii=False,
                        )
                    )
                    current_prediction = stable_prediction

            draw_prediction_panel(frame, current_prediction)
            draw_live_status(
                frame,
                buffer_size=len(frames),
                sequence_length=args.sequence_length,
                model_status=engine.mode_name,
                debug=current_prediction.get("debug", {}),
            )
            camera.show("gesture translator - live", frame)

            if camera.wait_key() == ord("q"):
                break
    finally:
        extractor.close()
        camera.close()

    return 0


def run_train_mode(args: argparse.Namespace) -> int:
    ensure_runtime_dirs()
    result = train_gesture_model(
        dataset_dir=Path(args.dataset_dir),
        artifact_dir=Path(args.artifact_dir),
        labels_path=Path(args.labels_path),
        sequence_length=args.sequence_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def run_serve_mode(args: argparse.Namespace) -> int:
    from gesture_translator.server import run_serve_mode as run_server_mode

    return int(run_server_mode(args))


def build_parser() -> argparse.ArgumentParser:
    config = load_runtime_config()
    parser = argparse.ArgumentParser(description="Gesture-word MVP on MediaPipe landmarks.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    record = subparsers.add_parser("record", help="Dataset recording mode")
    record.add_argument("--label", required=True)
    record.add_argument("--samples", type=int, default=20)
    record.add_argument("--camera-index", type=int, default=int(config["camera_index"]))
    record.add_argument("--mirror-preview", action="store_true", default=bool(config["mirror_preview"]))
    record.add_argument("--dataset-dir", default=config["dataset_dir"])
    record.add_argument("--labels-path", default=config["labels_path"])
    record.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    record.add_argument("--countdown-seconds", type=int, default=3)
    record.set_defaults(handler=run_record_mode)

    train = subparsers.add_parser("train", help="Training mode")
    train.add_argument("--dataset-dir", default=config["dataset_dir"])
    train.add_argument("--artifact-dir", default=config["artifacts_dir"])
    train.add_argument("--labels-path", default=config["labels_path"])
    train.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    train.add_argument("--epochs", type=int, default=18)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--hidden-size", type=int, default=128)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.set_defaults(handler=run_train_mode)

    live = subparsers.add_parser("live", help="Live inference mode")
    live.add_argument("--camera-index", type=int, default=int(config["camera_index"]))
    live.add_argument("--mirror-preview", action="store_true", default=bool(config["mirror_preview"]))
    live.add_argument("--artifact-dir", default=config["artifacts_dir"])
    live.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    live.add_argument("--min-sequence-length", type=int, default=20)
    live.add_argument("--confidence-threshold", type=float, default=CONFIDENCE_THRESHOLD)
    live.add_argument("--smoothing-window", type=int, default=SMOOTHING_WINDOW)
    live.add_argument("--stable-votes-required", type=int, default=STABLE_VOTES_REQUIRED)
    live.add_argument("--cooldown-seconds", type=float, default=COOLDOWN_SECONDS)
    live.set_defaults(handler=run_live_mode)

    serve = subparsers.add_parser("serve", help="Serve mode for backend bridge")
    serve.add_argument("--artifact-dir", default=config["artifacts_dir"])
    serve.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    serve.add_argument("--min-sequence-length", type=int, default=20)
    serve.add_argument("--confidence-threshold", type=float, default=CONFIDENCE_THRESHOLD)
    serve.add_argument("--smoothing-window", type=int, default=SMOOTHING_WINDOW)
    serve.add_argument("--stable-votes-required", type=int, default=STABLE_VOTES_REQUIRED)
    serve.add_argument("--cooldown-seconds", type=float, default=COOLDOWN_SECONDS)
    serve.set_defaults(handler=run_serve_mode)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
