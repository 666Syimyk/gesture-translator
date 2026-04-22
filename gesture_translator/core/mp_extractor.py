from __future__ import annotations

import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from gesture_translator.config import MODEL_CACHE_DIR


MODEL_URLS = {
    "hand": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "face": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "pose": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}


def ensure_model(name: str) -> str:
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_CACHE_DIR / f"{name}_landmarker.task"
    if not path.exists() or path.stat().st_size == 0:
        urllib.request.urlretrieve(MODEL_URLS[name], path)
    return str(path)


def serialize_points(points, include_visibility: bool = False) -> list[dict]:
    result = []
    for point in points or []:
        item = {
            "x": round(float(point.x), 6),
            "y": round(float(point.y), 6),
            "z": round(float(point.z), 6),
        }
        if include_visibility:
            item["visibility"] = round(float(getattr(point, "visibility", 0.0)), 6)
        result.append(item)
    return result


def serialize_handedness(category) -> dict | None:
    if category is None:
        return None
    return {
        "label": getattr(category, "category_name", ""),
        "score": round(float(getattr(category, "score", 0.0)), 6),
    }


class MediaPipeHolisticExtractor:
    def __init__(self) -> None:
        running_mode = vision.RunningMode.VIDEO
        self.hand = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=ensure_model("hand")),
                running_mode=running_mode,
                num_hands=2,
                min_hand_detection_confidence=0.6,
                min_hand_presence_confidence=0.6,
                min_tracking_confidence=0.6,
            )
        )
        self.face = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=ensure_model("face")),
                running_mode=running_mode,
                num_faces=1,
                min_face_detection_confidence=0.55,
                min_face_presence_confidence=0.55,
                min_tracking_confidence=0.55,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
        )
        self.pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=ensure_model("pose")),
                running_mode=running_mode,
                num_poses=1,
                min_pose_detection_confidence=0.55,
                min_pose_presence_confidence=0.55,
                min_tracking_confidence=0.55,
                output_segmentation_masks=False,
            )
        )

    def process(self, frame, frame_index: int, timestamp_ms: int) -> dict:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        hand_result = self.hand.detect_for_video(mp_image, timestamp_ms)
        face_result = self.face.detect_for_video(mp_image, timestamp_ms)
        pose_result = self.pose.detect_for_video(mp_image, timestamp_ms)
        return self._build_payload(hand_result, face_result, pose_result, frame_index, timestamp_ms)

    def _build_payload(self, hand_result, face_result, pose_result, frame_index: int, timestamp_ms: int):
        left_hand = []
        right_hand = []
        left_hand_world = []
        right_hand_world = []
        handedness = []

        for index, landmarks in enumerate(getattr(hand_result, "hand_landmarks", []) or []):
            categories = getattr(hand_result, "handedness", []) or []
            handedness_payload = None
            if index < len(categories) and categories[index]:
                handedness_payload = serialize_handedness(categories[index][0])
            if handedness_payload:
                handedness.append(handedness_payload)

            label = str((handedness_payload or {}).get("label", "Right")).lower()
            world_landmarks = getattr(hand_result, "hand_world_landmarks", []) or []
            world_points = serialize_points(world_landmarks[index]) if index < len(world_landmarks) else []
            if label == "left":
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

        return {
            "frame_index": int(frame_index),
            "timestamp_ms": int(timestamp_ms),
            "left_hand": left_hand,
            "right_hand": right_hand,
            "left_hand_world": left_hand_world,
            "right_hand_world": right_hand_world,
            "face": face_points,
            "pose": pose_points,
            "pose_world": pose_world_points,
            "handedness": handedness,
            "quality": {
                "has_left_hand": bool(left_hand),
                "has_right_hand": bool(right_hand),
                "has_face": bool(face_points),
                "has_pose": bool(pose_points),
            },
        }

    def close(self) -> None:
        self.hand.close()
        self.face.close()
        self.pose.close()
