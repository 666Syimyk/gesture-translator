import argparse
from collections import deque
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


PHRASE_DIR = Path(__file__).resolve().parent
ML_DIR = PHRASE_DIR.parent
BACKEND_DIR = PHRASE_DIR.parents[1]
if str(PHRASE_DIR) not in sys.path:
    sys.path.insert(0, str(PHRASE_DIR))
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config import load_config, resolve_backend_path
from predict_sequence_model import load_model_bundle, predict_probabilities_for_bundle


def import_capture_stack():
    import cv2
    import mediapipe as mp
    from scripts.extract_landmarks import build_landmarkers, extract_frame_payload

    return cv2, mp, build_landmarkers, extract_frame_payload


def speak_text(text):
    if os.name != "nt" or not text:
        return

    safe_text = str(text).replace("'", "''")
    command = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.Speak('{safe_text}')"
    )
    subprocess.Popen(
        ["powershell", "-NoProfile", "-Command", command],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def draw_overlay(cv2, frame, label, confidence, status, prediction_log):
    cv2.putText(
        frame,
        f"{status}: {label} {confidence:.2f}",
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Press q to quit",
        (24, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    y = 112
    for item in list(prediction_log)[-4:]:
        cv2.putText(
            frame,
            f"{item['label']} {item['confidence']:.2f}",
            (24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 220, 200),
            1,
            cv2.LINE_AA,
        )
        y += 24


def smoothed_prediction(probability_window, class_entries, stable_votes_required):
    if not probability_window:
        return "none", 0.0

    matrix = np.stack(probability_window)
    averaged = np.mean(matrix, axis=0)
    best_index = int(np.argmax(averaged))
    best_label = class_entries[best_index]["text"]
    frame_predictions = np.argmax(matrix, axis=1)
    best_votes = int(np.sum(frame_predictions == best_index))

    if best_votes < int(stable_votes_required):
        return "none", float(averaged[best_index])

    return best_label, float(averaged[best_index])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PHRASE_DIR / "default_config.json"))
    parser.add_argument("--model-dir")
    parser.add_argument("--camera-index", type=int)
    parser.add_argument("--confidence-threshold", type=float)
    parser.add_argument("--no-tts", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    inference_config = config["inference"]
    model_dir = resolve_backend_path(args.model_dir or config["artifacts_dir"])
    if not (model_dir / "metadata.json").exists():
        raise SystemExit(
            f"Model not found in {model_dir}. Train it first with phrase:train."
        )

    bundle = load_model_bundle(model_dir)
    class_entries = bundle["class_entries"]
    sequence_length = int(config.get("sequence_length", 32))
    min_window_frames = int(inference_config.get("min_window_frames", 20))
    smoothing_window = int(inference_config.get("smoothing_window", 5))
    stable_votes_required = min(
        int(inference_config.get("stable_votes_required", 3)),
        smoothing_window,
    )
    threshold = float(
        args.confidence_threshold
        if args.confidence_threshold is not None
        else inference_config.get("confidence_threshold", 0.65)
    )
    idle_label = str(inference_config.get("idle_label", "none"))
    cooldown_seconds = float(inference_config.get("cooldown_seconds", 1.2))
    prediction_log_size = int(inference_config.get("prediction_log_size", 10))
    enable_tts = bool(inference_config.get("speak", True)) and not args.no_tts
    camera_index = (
        int(args.camera_index)
        if args.camera_index is not None
        else int(inference_config.get("camera_index", 0))
    )

    cv2, mp, build_landmarkers, extract_frame_payload = import_capture_stack()
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise SystemExit(f"Could not open camera index {camera_index}.")

    hand_landmarker, face_landmarker, pose_landmarker = build_landmarkers()
    frames = deque(maxlen=sequence_length)
    probabilities = deque(maxlen=smoothing_window)
    prediction_log = deque(maxlen=prediction_log_size)
    clock_start = time.perf_counter()
    frame_index = 0
    last_spoken_label = None
    last_spoken_at = 0.0
    current_label = idle_label
    current_confidence = 0.0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            now = time.perf_counter()
            timestamp_ms = int((now - clock_start) * 1000)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
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

            if len(frames) >= min_window_frames:
                probabilities.append(
                    predict_probabilities_for_bundle(bundle, list(frames))
                )
                current_label, current_confidence = smoothed_prediction(
                    probabilities,
                    class_entries,
                    stable_votes_required,
                )
                prediction_log.append(
                    {
                        "label": current_label,
                        "confidence": current_confidence,
                        "time": round(now - clock_start, 3),
                    }
                )

                is_phrase = (
                    current_label != idle_label and current_confidence >= threshold
                )
                if (
                    is_phrase
                    and current_label != last_spoken_label
                    and now - last_spoken_at >= cooldown_seconds
                ):
                    if enable_tts:
                        speak_text(current_label)
                    print(
                        json.dumps(
                            {
                                "event": "phrase",
                                "label": current_label,
                                "confidence": round(current_confidence, 4),
                                "time": round(now - clock_start, 3),
                            },
                            ensure_ascii=False,
                        )
                    )
                    last_spoken_label = current_label
                    last_spoken_at = now
                elif current_label == idle_label or current_confidence < threshold:
                    last_spoken_label = None

            is_active_phrase = (
                current_label != idle_label and current_confidence >= threshold
            )
            status = "phrase" if is_active_phrase else "idle"
            display_label = current_label if is_active_phrase else idle_label
            draw_overlay(
                cv2,
                frame,
                display_label,
                current_confidence,
                status,
                prediction_log,
            )
            cv2.imshow("phrase live inference", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        hand_landmarker.close()
        face_landmarker.close()
        pose_landmarker.close()
        cv2.destroyAllWindows()

    print(
        json.dumps(
            {
                "ok": True,
                "modelDir": str(model_dir),
                "lastLabel": current_label,
                "lastConfidence": round(current_confidence, 4),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
