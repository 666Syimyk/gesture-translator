from __future__ import annotations

import math

from gesture_translator.config import FACE_LANDMARKS, HAND_LANDMARKS, POSE_LANDMARKS


def safe_point(points, index, include_visibility: bool = False):
    if not points or index >= len(points):
        return {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            **({"visibility": 0.0} if include_visibility else {}),
        }
    point = points[index]
    return {
        "x": float(point.get("x", 0.0)),
        "y": float(point.get("y", 0.0)),
        "z": float(point.get("z", 0.0)),
        **({"visibility": float(point.get("visibility", 0.0))} if include_visibility else {}),
    }


def average_points(points: list[dict]) -> dict:
    if not points:
        return {"x": 0.0, "y": 0.0, "z": 0.0}
    count = len(points)
    return {
        "x": sum(point["x"] for point in points) / count,
        "y": sum(point["y"] for point in points) / count,
        "z": sum(point["z"] for point in points) / count,
    }


def point_distance(first: dict | None, second: dict | None) -> float:
    if not first or not second:
        return 0.0
    return math.hypot(float(first["x"]) - float(second["x"]), float(first["y"]) - float(second["y"]))


def select_points(points, indices, include_visibility: bool = False) -> list[dict]:
    return [safe_point(points, index, include_visibility=include_visibility) for index in indices]


def infer_anchor_and_scale(frame: dict) -> tuple[dict, float]:
    pose = frame.get("pose", []) or []
    face = frame.get("face", []) or []
    left_shoulder = safe_point(pose, 11, include_visibility=True)
    right_shoulder = safe_point(pose, 12, include_visibility=True)
    if left_shoulder["visibility"] > 0.2 and right_shoulder["visibility"] > 0.2:
        anchor = {
            "x": (left_shoulder["x"] + right_shoulder["x"]) / 2,
            "y": (left_shoulder["y"] + right_shoulder["y"]) / 2,
            "z": (left_shoulder["z"] + right_shoulder["z"]) / 2,
        }
        scale = max(point_distance(left_shoulder, right_shoulder), 0.08)
        return anchor, scale

    left_eye = safe_point(face, 33)
    right_eye = safe_point(face, 263)
    nose = safe_point(face, 1)
    if point_distance(left_eye, right_eye) > 0:
        scale = max(point_distance(left_eye, right_eye), 0.06)
        return nose, scale

    return {"x": 0.5, "y": 0.5, "z": 0.0}, 0.2


def build_reference_points(frame: dict) -> dict:
    pose = frame.get("pose", []) or []
    face = frame.get("face", []) or []
    left_shoulder = safe_point(pose, 11, include_visibility=True)
    right_shoulder = safe_point(pose, 12, include_visibility=True)
    nose = safe_point(face, 1)
    chin = safe_point(face, 152)
    left_eye = safe_point(face, 33)
    right_eye = safe_point(face, 263)
    mouth_left = safe_point(face, 61)
    mouth_right = safe_point(face, 291)
    upper_lip = safe_point(face, 13)
    lower_lip = safe_point(face, 14)
    nose_base = safe_point(face, 2)

    shoulders_visible = left_shoulder["visibility"] > 0.2 and right_shoulder["visibility"] > 0.2
    chest = (
        {
            "x": (left_shoulder["x"] + right_shoulder["x"]) / 2,
            "y": (left_shoulder["y"] + right_shoulder["y"]) / 2 + 0.05,
            "z": (left_shoulder["z"] + right_shoulder["z"]) / 2,
        }
        if shoulders_visible
        else {"x": nose["x"], "y": nose["y"] + 0.22, "z": nose["z"]}
    )

    temple = {
        "x": right_eye["x"] + 0.02,
        "y": right_eye["y"] - 0.015,
        "z": right_eye["z"],
    }
    cheek = {
        "x": (right_eye["x"] + mouth_right["x"]) / 2,
        "y": (right_eye["y"] + mouth_right["y"]) / 2 + 0.02,
        "z": (right_eye["z"] + mouth_right["z"]) / 2,
    }
    right_cheek = {
        "x": (right_eye["x"] + mouth_right["x"]) / 2,
        "y": (right_eye["y"] + mouth_right["y"]) / 2 + 0.01,
        "z": (right_eye["z"] + mouth_right["z"]) / 2,
    }
    left_cheek = {
        "x": (left_eye["x"] + mouth_left["x"]) / 2,
        "y": (left_eye["y"] + mouth_left["y"]) / 2 + 0.01,
        "z": (left_eye["z"] + mouth_left["z"]) / 2,
    }
    jaw_right = {
        "x": (chin["x"] + mouth_right["x"]) / 2,
        "y": (chin["y"] + mouth_right["y"]) / 2,
        "z": (chin["z"] + mouth_right["z"]) / 2,
    }
    jaw_left = {
        "x": (chin["x"] + mouth_left["x"]) / 2,
        "y": (chin["y"] + mouth_left["y"]) / 2,
        "z": (chin["z"] + mouth_left["z"]) / 2,
    }
    chin_center = {
        "x": (mouth_left["x"] + mouth_right["x"]) / 2,
        "y": (chin["y"] + mouth_left["y"] + mouth_right["y"]) / 3,
        "z": chin["z"],
    }
    mouth_center = average_points([mouth_left, mouth_right])
    lips_center = average_points([upper_lip, lower_lip, mouth_left, mouth_right])
    forehead = {
        "x": (left_eye["x"] + right_eye["x"]) / 2,
        "y": min(left_eye["y"], right_eye["y"]) - 0.06,
        "z": (left_eye["z"] + right_eye["z"]) / 2,
    }
    mustache = {
        "x": upper_lip["x"],
        "y": upper_lip["y"] - 0.01,
        "z": upper_lip["z"],
    }

    return {
        "nose": nose,
        "nose_base": nose_base,
        "chin": chin,
        "chest": chest,
        "temple": temple,
        "cheek": cheek,
        "right_cheek": right_cheek,
        "left_cheek": left_cheek,
        "jaw_right": jaw_right,
        "jaw_left": jaw_left,
        "chin_center": chin_center,
        "mouth_center": mouth_center,
        "upper_lip": upper_lip,
        "lower_lip": lower_lip,
        "lips_center": lips_center,
        "forehead": forehead,
        "mustache": mustache,
        "face_center": average_points([nose, chin, left_eye, right_eye, mouth_left, mouth_right]),
    }


def normalize_point(point: dict, anchor: dict, scale: float, include_visibility: bool = False) -> list[float]:
    normalized = [
        (float(point["x"]) - float(anchor["x"])) / scale,
        (float(point["y"]) - float(anchor["y"])) / scale,
        float(point["z"]) / scale,
    ]
    if include_visibility:
        normalized.append(float(point.get("visibility", 0.0)))
    return normalized


def build_normalized_frame(frame: dict) -> dict:
    anchor, scale = infer_anchor_and_scale(frame)
    refs = build_reference_points(frame)
    return {
        "anchor": anchor,
        "scale": scale,
        "refs": refs,
        "left_hand": [normalize_point(point, anchor, scale) for point in select_points(frame.get("left_hand", []), HAND_LANDMARKS)],
        "right_hand": [normalize_point(point, anchor, scale) for point in select_points(frame.get("right_hand", []), HAND_LANDMARKS)],
        "face": [normalize_point(point, anchor, scale) for point in select_points(frame.get("face", []), FACE_LANDMARKS)],
        "pose": [normalize_point(point, anchor, scale, include_visibility=True) for point in select_points(frame.get("pose", []), POSE_LANDMARKS, include_visibility=True)],
    }
