import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision


MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_URLS = {
    "hand": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "face": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "pose": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}

HOLISTIC_CONFIG = {
    "num_hands": 2,
    "num_faces": 1,
    "num_poses": 1,
    "min_hand_detection_confidence": 0.65,
    "min_hand_presence_confidence": 0.7,
    "min_hand_tracking_confidence": 0.7,
    "min_face_detection_confidence": 0.6,
    "min_face_presence_confidence": 0.6,
    "min_face_tracking_confidence": 0.6,
    "min_pose_detection_confidence": 0.6,
    "min_pose_presence_confidence": 0.6,
    "min_pose_tracking_confidence": 0.6,
}


def ensure_model(name):
    model_path = MODEL_DIR / f"{name}_landmarker.task"

    if not model_path.exists() or model_path.stat().st_size == 0:
        urllib.request.urlretrieve(MODEL_URLS[name], model_path)

    return str(model_path)


def serialize_points(points, include_visibility=False):
    if not points:
        return []

    serialized = []

    for point in points:
        item = {
            "x": round(float(point.x), 6),
            "y": round(float(point.y), 6),
            "z": round(float(point.z), 6),
        }

        if include_visibility:
            item["visibility"] = round(float(getattr(point, "visibility", 0.0)), 6)

        serialized.append(item)

    return serialized


def serialize_handedness(category):
    if category is None:
        return None

    return {
        "label": getattr(category, "category_name", ""),
        "score": round(float(getattr(category, "score", 0.0)), 6),
    }


def build_landmarkers():
    running_mode = vision.RunningMode.VIDEO

    hand = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=ensure_model("hand")),
            running_mode=running_mode,
            num_hands=HOLISTIC_CONFIG["num_hands"],
            min_hand_detection_confidence=HOLISTIC_CONFIG[
                "min_hand_detection_confidence"
            ],
            min_hand_presence_confidence=HOLISTIC_CONFIG[
                "min_hand_presence_confidence"
            ],
            min_tracking_confidence=HOLISTIC_CONFIG["min_hand_tracking_confidence"],
        )
    )

    face = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=ensure_model("face")),
            running_mode=running_mode,
            num_faces=HOLISTIC_CONFIG["num_faces"],
            min_face_detection_confidence=HOLISTIC_CONFIG[
                "min_face_detection_confidence"
            ],
            min_face_presence_confidence=HOLISTIC_CONFIG[
                "min_face_presence_confidence"
            ],
            min_tracking_confidence=HOLISTIC_CONFIG["min_face_tracking_confidence"],
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
    )

    pose = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=ensure_model("pose")),
            running_mode=running_mode,
            num_poses=HOLISTIC_CONFIG["num_poses"],
            min_pose_detection_confidence=HOLISTIC_CONFIG[
                "min_pose_detection_confidence"
            ],
            min_pose_presence_confidence=HOLISTIC_CONFIG[
                "min_pose_presence_confidence"
            ],
            min_tracking_confidence=HOLISTIC_CONFIG["min_pose_tracking_confidence"],
            output_segmentation_masks=False,
        )
    )

    return hand, face, pose


def extract_frame_payload(hand_result, face_result, pose_result, frame_index, timestamp_ms):
    left_hand = []
    right_hand = []
    left_hand_world = []
    right_hand_world = []
    handedness = []

    hand_landmarks = getattr(hand_result, "hand_landmarks", []) or []
    hand_world_landmarks = getattr(hand_result, "hand_world_landmarks", []) or []
    hand_handedness = getattr(hand_result, "handedness", []) or []

    for index, landmarks in enumerate(hand_landmarks):
        handedness_category = None

        if index < len(hand_handedness) and hand_handedness[index]:
            handedness_category = hand_handedness[index][0]

        handedness_payload = serialize_handedness(handedness_category)

        if handedness_payload:
            handedness.append(handedness_payload)

        hand_label = (handedness_payload or {}).get("label", "Right").lower()
        world_points = (
            serialize_points(hand_world_landmarks[index])
            if index < len(hand_world_landmarks)
            else []
        )

        if hand_label == "left":
            left_hand = serialize_points(landmarks)
            left_hand_world = world_points
        else:
            right_hand = serialize_points(landmarks)
            right_hand_world = world_points

    face_points = (
        serialize_points(face_result.face_landmarks[0])
        if getattr(face_result, "face_landmarks", None)
        else []
    )
    pose_points = (
        serialize_points(pose_result.pose_landmarks[0], include_visibility=True)
        if getattr(pose_result, "pose_landmarks", None)
        else []
    )
    pose_world_points = (
        serialize_points(pose_result.pose_world_landmarks[0], include_visibility=True)
        if getattr(pose_result, "pose_world_landmarks", None)
        else []
    )

    quality = {
        "has_left_hand": bool(left_hand),
        "has_right_hand": bool(right_hand),
        "has_face": bool(face_points),
        "has_pose": bool(pose_points),
    }

    return {
        "frame_index": frame_index,
        "timestamp_ms": timestamp_ms,
        "left_hand": left_hand,
        "right_hand": right_hand,
        "face": face_points,
        "pose": pose_points,
        "left_hand_world": left_hand_world,
        "right_hand_world": right_hand_world,
        "pose_world": pose_world_points,
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
    right_hand_frames = sum(
        1 for frame in frames if frame["quality"]["has_right_hand"]
    )
    face_frames = sum(1 for frame in frames if frame["quality"]["has_face"])
    pose_frames = sum(1 for frame in frames if frame["quality"]["has_pose"])
    valid_frames = sum(
        1
        for frame in frames
        if frame["quality"]["has_left_hand"]
        or frame["quality"]["has_right_hand"]
        or frame["quality"]["has_face"]
        or frame["quality"]["has_pose"]
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
        "face_ratio": round(face_frames / frame_count, 4),
        "pose_ratio": round(pose_frames / frame_count, 4),
        "missing_hand_ratio": round(missing_hand_frames / frame_count, 4),
        "missing_face_ratio": round(1 - (face_frames / frame_count), 4),
        "missing_pose_ratio": round(1 - (pose_frames / frame_count), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sign-language", default="rsl")
    parser.add_argument("--training-video-id", type=int, default=0)
    parser.add_argument("--phrase-id", type=int, default=0)
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.input)

    if not capture.isOpened():
        print("Failed to open input video", file=sys.stderr)
        return 1

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    hand_landmarker, face_landmarker, pose_landmarker = build_landmarkers()
    frames = []
    frame_index = 0

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            if fps > 0:
                timestamp_ms = round((frame_index / fps) * 1000)
            else:
                timestamp_ms = frame_index * 33

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
    finally:
        capture.release()
        hand_landmarker.close()
        face_landmarker.close()
        pose_landmarker.close()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    summary = build_summary(frames)

    payload = {
        "meta": {
            "source_video": args.input,
            "training_video_id": args.training_video_id or None,
            "phrase_id": args.phrase_id or None,
            "sign_language": args.sign_language,
            "fps": round(fps, 4),
            "frame_width": frame_width,
            "frame_height": frame_height,
            "source_frame_count": total_frames,
            "processed_frame_count": len(frames),
            "frame_count": len(frames),
            "extractor": "mediapipe_tasks_python_holistic_v2",
            "extractor_type": "holistic_landmarks",
            "config": HOLISTIC_CONFIG,
            "summary": summary,
        },
        "frames": frames,
    }

    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False)

    print(
        json.dumps(
            {
                "ok": True,
                "frameCount": len(frames),
                "fps": round(fps, 4),
                "width": frame_width,
                "height": frame_height,
                "outputPath": args.output,
                "extractorType": "holistic_landmarks",
                "trainingVideoId": args.training_video_id or None,
                "phraseId": args.phrase_id or None,
                "summary": summary,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
