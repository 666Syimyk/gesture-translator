from __future__ import annotations

import numpy as np

from gesture_translator.core.normalizer import build_normalized_frame, point_distance, safe_point


FINGER_TIP_PIP = [(8, 6), (12, 10), (16, 14), (20, 18)]
FINGERTIP_INDICES = [4, 8, 12, 16, 20]


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def hand_center(points: list[dict]) -> dict:
    indices = [0, 5, 9, 13, 17]
    chosen = [safe_point(points, index) for index in indices if points and index < len(points)]
    if not chosen:
        return {"x": 0.0, "y": 0.0, "z": 0.0}
    count = len(chosen)
    return {
        "x": sum(point["x"] for point in chosen) / count,
        "y": sum(point["y"] for point in chosen) / count,
        "z": sum(point["z"] for point in chosen) / count,
    }


def _tips_center(points: list[dict], tip_indices: list[int]) -> dict:
    chosen = [safe_point(points, index) for index in tip_indices]
    count = len(chosen)
    return {
        "x": sum(point["x"] for point in chosen) / max(count, 1),
        "y": sum(point["y"] for point in chosen) / max(count, 1),
        "z": sum(point["z"] for point in chosen) / max(count, 1),
    }


def count_active_fingers(points: list[dict], handedness: str) -> tuple[int, dict]:
    if not points or len(points) < 21:
        return 0, {"thumb": 0, "index": 0, "middle": 0, "ring": 0, "pinky": 0}

    states = {}
    thumb_tip = safe_point(points, 4)
    thumb_ip = safe_point(points, 3)
    thumb_mcp = safe_point(points, 2)
    thumb_score = abs(thumb_tip["x"] - thumb_mcp["x"]) - abs(thumb_ip["x"] - thumb_mcp["x"])
    states["thumb"] = 1 if thumb_score > 0.015 else 0

    for finger_name, (tip_index, pip_index) in zip(("index", "middle", "ring", "pinky"), FINGER_TIP_PIP):
        tip = safe_point(points, tip_index)
        pip = safe_point(points, pip_index)
        states[finger_name] = 1 if tip["y"] < pip["y"] - 0.01 else 0

    active_count = int(sum(states.values()))
    return active_count, states


def _fingertips_together_score(points: list[dict], scale: float) -> float:
    tips = [safe_point(points, index) for index in FINGERTIP_INDICES]
    if len(tips) < 5:
        return 0.0
    center = _tips_center(points, FINGERTIP_INDICES)
    distances = [point_distance(point, center) / max(scale, 1e-4) for point in tips]
    avg_distance = float(sum(distances) / len(distances))
    return clamp01((0.22 - avg_distance) / 0.22)


def _three_finger_mouth_contact(points: list[dict], mouth_center: dict, scale: float) -> float:
    if not points or len(points) < 21:
        return 0.0
    tips = [safe_point(points, 4), safe_point(points, 8), safe_point(points, 12)]
    close_count = 0
    for tip in tips:
        distance = point_distance(tip, mouth_center) / max(scale, 1e-4)
        if distance <= 0.22:
            close_count += 1
    return close_count / 3.0


def _infer_contact_zone(distances: dict) -> str:
    ranked = sorted(distances.items(), key=lambda item: float(item[1]))
    if not ranked:
        return "space"
    zone, score = ranked[0]
    return zone if float(score) <= 1.2 else "space"


def hand_metrics(points: list[dict], handedness: str, refs: dict, scale: float) -> dict:
    if not points or len(points) < 21:
        return {
            "present": False,
            "center": {"x": 0.0, "y": 0.0, "z": 0.0},
            "active_count": 0,
            "open_palm": 0.0,
            "pointing": 0.0,
            "pinch": 0.0,
            "fist": 0.0,
            "two_fingertips_joined": 0.0,
            "all_fingertips_together": 0.0,
            "yery_shape": 0.0,
            "pointing_outward": 0.0,
            "pointing_chest": 0.0,
            "three_finger_mouth_contact": 0.0,
            "face_distance": 9.0,
            "mouth_distance": 9.0,
            "chest_distance": 9.0,
            "temple_distance": 9.0,
            "cheek_distance": 9.0,
            "jaw_distance": 9.0,
            "chin_distance": 9.0,
            "forehead_distance": 9.0,
            "mustache_distance": 9.0,
            "upper_lip_distance": 9.0,
            "lower_lip_distance": 9.0,
            "lips_distance": 9.0,
            "nose_base_distance": 9.0,
            "right_cheek_distance": 9.0,
            "left_cheek_distance": 9.0,
            "jawline_distance": 9.0,
            "contact_zone": "space",
            "handshape_label": "none",
            "contact_point": {"x": 0.0, "y": 0.0, "z": 0.0},
            "index_tip": {"x": 0.0, "y": 0.0, "z": 0.0},
            "index_tip_z": 0.0,
            "wrist_z": 0.0,
        }

    active_count, states = count_active_fingers(points, handedness)
    center = hand_center(points)
    wrist = safe_point(points, 0)
    index_tip = safe_point(points, 8)
    thumb_tip = safe_point(points, 4)
    middle_tip = safe_point(points, 12)
    ring_tip = safe_point(points, 16)
    pinky_tip = safe_point(points, 20)
    pinch_distance = point_distance(index_tip, thumb_tip) / max(scale, 1e-4)
    other_folded = 1.0 - ((states["middle"] + states["ring"] + states["pinky"]) / 3.0)
    three_finger_contact = _three_finger_mouth_contact(points, refs["mouth_center"], scale)

    open_palm = clamp01((active_count - 3) / 2.0)
    fist = clamp01((2.5 - active_count) / 2.5)
    two_fingertips_joined = clamp01((0.16 - pinch_distance) / 0.16) * clamp01(other_folded + 0.12)
    all_fingertips_together = _fingertips_together_score(points, scale)

    # "Ы" detector reused as explicit handshape class for phrase mode:
    # thumb+index+middle comparatively open, ring+pinky comparatively closed.
    yery_shape = clamp01(
        (
            clamp01(states["thumb"])
            + clamp01(states["index"])
            + clamp01(states["middle"])
            + clamp01(1 - states["ring"])
            + clamp01(1 - states["pinky"])
        )
        / 5.0
    )
    fingertips_point = _tips_center(points, FINGERTIP_INDICES)
    thumb_index_contact = {
        "x": (thumb_tip["x"] + index_tip["x"]) / 2,
        "y": (thumb_tip["y"] + index_tip["y"]) / 2,
        "z": (thumb_tip["z"] + index_tip["z"]) / 2,
    }
    contact_point = (
        thumb_index_contact
        if two_fingertips_joined >= all_fingertips_together
        else fingertips_point
    )

    pointing_outward = clamp01((wrist["z"] - index_tip["z"] - 0.01) / 0.08)
    chest_distance = point_distance(index_tip, refs["chest"]) / max(scale, 1e-4)

    upper_lip_distance = point_distance(contact_point, refs["upper_lip"]) / max(scale, 1e-4)
    lower_lip_distance = point_distance(contact_point, refs["lower_lip"]) / max(scale, 1e-4)
    lips_distance = point_distance(contact_point, refs["lips_center"]) / max(scale, 1e-4)
    nose_base_distance = point_distance(contact_point, refs["nose_base"]) / max(scale, 1e-4)
    chin_distance = point_distance(contact_point, refs["chin_center"]) / max(scale, 1e-4)
    forehead_distance = point_distance(contact_point, refs["forehead"]) / max(scale, 1e-4)
    right_cheek_distance = point_distance(contact_point, refs["right_cheek"]) / max(scale, 1e-4)
    left_cheek_distance = point_distance(contact_point, refs["left_cheek"]) / max(scale, 1e-4)
    jawline_distance = min(
        point_distance(contact_point, refs["jaw_left"]),
        point_distance(contact_point, refs["jaw_right"]),
    ) / max(scale, 1e-4)

    contact_zone = _infer_contact_zone(
        {
            "upper_lip_zone": upper_lip_distance,
            "lower_lip_zone": lower_lip_distance,
            "lips_zone": lips_distance,
            "nose_base_zone": nose_base_distance,
            "chin_zone": chin_distance,
            "forehead_zone": forehead_distance,
            "right_cheek_zone": right_cheek_distance,
            "left_cheek_zone": left_cheek_distance,
            "jawline_path": jawline_distance,
        }
    )

    handshape_scores = {
        "open_palm": open_palm,
        "fist": fist,
        "two_fingertips_joined": two_fingertips_joined,
        "all_fingertips_together": all_fingertips_together,
        "yery_shape": yery_shape,
    }
    handshape_label, handshape_score = max(handshape_scores.items(), key=lambda item: item[1])
    if handshape_score < 0.45:
        handshape_label = "unknown"

    return {
        "present": True,
        "center": center,
        "active_count": active_count,
        "open_palm": open_palm,
        "pointing": clamp01(states["index"] * other_folded),
        "pinch": clamp01((0.16 - pinch_distance) / 0.16),
        "fist": fist,
        "two_fingertips_joined": two_fingertips_joined,
        "all_fingertips_together": all_fingertips_together,
        "yery_shape": yery_shape,
        "pointing_outward": pointing_outward,
        "pointing_chest": clamp01((1.0 - chest_distance) / 1.0),
        "three_finger_mouth_contact": three_finger_contact,
        "face_distance": point_distance(center, refs["face_center"]) / max(scale, 1e-4),
        "mouth_distance": point_distance(index_tip, refs["mouth_center"]) / max(scale, 1e-4),
        "chest_distance": chest_distance,
        "temple_distance": point_distance(center, refs["temple"]) / max(scale, 1e-4),
        "cheek_distance": min(
            point_distance(center, refs["right_cheek"]),
            point_distance(index_tip, refs["right_cheek"]),
            point_distance(center, refs["left_cheek"]),
            point_distance(index_tip, refs["left_cheek"]),
        ) / max(scale, 1e-4),
        "jaw_distance": min(
            point_distance(center, refs["jaw_right"]),
            point_distance(index_tip, refs["jaw_right"]),
            point_distance(center, refs["jaw_left"]),
            point_distance(index_tip, refs["jaw_left"]),
        ) / max(scale, 1e-4),
        "chin_distance": min(
            point_distance(center, refs["chin_center"]),
            point_distance(index_tip, refs["chin_center"]),
        ) / max(scale, 1e-4),
        "forehead_distance": min(
            point_distance(center, refs["forehead"]),
            point_distance(index_tip, refs["forehead"]),
        ) / max(scale, 1e-4),
        "mustache_distance": min(
            point_distance(center, refs["mustache"]),
            point_distance(index_tip, refs["mustache"]),
        ) / max(scale, 1e-4),
        "upper_lip_distance": upper_lip_distance,
        "lower_lip_distance": lower_lip_distance,
        "lips_distance": lips_distance,
        "nose_base_distance": nose_base_distance,
        "right_cheek_distance": right_cheek_distance,
        "left_cheek_distance": left_cheek_distance,
        "jawline_distance": jawline_distance,
        "contact_zone": contact_zone,
        "contact_point": contact_point,
        "handshape_label": handshape_label,
        "index_tip": index_tip,
        "index_tip_z": float(index_tip["z"]),
        "wrist_z": float(wrist["z"]),
        "middle_tip": middle_tip,
        "thumb_tip": thumb_tip,
        "ring_tip": ring_tip,
        "pinky_tip": pinky_tip,
    }


def choose_dominant_hand(left_metrics: dict, right_metrics: dict) -> tuple[str, dict]:
    if right_metrics["present"]:
        return "right", right_metrics
    if left_metrics["present"]:
        return "left", left_metrics
    return "none", right_metrics


def flatten_points(point_groups) -> list[float]:
    flat = []
    for point in point_groups:
        flat.extend(point)
    return flat


def infer_dominant_zone(dominant: dict) -> str:
    candidates = {
        "upper_lip_zone": float(dominant.get("upper_lip_distance", 9.0)),
        "lower_lip_zone": float(dominant.get("lower_lip_distance", 9.0)),
        "lips_zone": float(dominant.get("lips_distance", 9.0)),
        "nose_base_zone": float(dominant.get("nose_base_distance", 9.0)),
        "forehead_zone": float(dominant.get("forehead_distance", 9.0)),
        "right_cheek_zone": float(dominant.get("right_cheek_distance", 9.0)),
        "left_cheek_zone": float(dominant.get("left_cheek_distance", 9.0)),
        "jawline_path": float(dominant.get("jawline_distance", 9.0)),
        "chin_zone": float(dominant.get("chin_distance", 9.0)),
        "chest_zone": float(dominant.get("chest_distance", 9.0)),
    }
    zone, score = min(candidates.items(), key=lambda item: item[1])
    return zone if score <= 1.25 else "space"


def build_frame_observation(frame: dict, previous_frame: dict | None = None) -> dict:
    normalized = build_normalized_frame(frame)
    scale = float(normalized["scale"])
    refs = normalized["refs"]
    left_raw = frame.get("left_hand", []) or []
    right_raw = frame.get("right_hand", []) or []
    left_metrics = hand_metrics(left_raw, "left", refs, scale)
    right_metrics = hand_metrics(right_raw, "right", refs, scale)
    dominant_name, dominant = choose_dominant_hand(left_metrics, right_metrics)
    hand_count = int(left_metrics["present"]) + int(right_metrics["present"])
    hand_separation = (
        point_distance(left_metrics["center"], right_metrics["center"]) / max(scale, 1e-4)
        if left_metrics["present"] and right_metrics["present"]
        else 0.0
    )

    delta_x = 0.0
    delta_y = 0.0
    delta_sep = 0.0
    delta_open_palm = 0.0
    if previous_frame:
        prev_left = previous_frame["left_metrics"]
        prev_right = previous_frame["right_metrics"]
        prev_dom = prev_right if dominant_name == "right" else prev_left
        delta_x = dominant["center"]["x"] - prev_dom["center"]["x"]
        delta_y = dominant["center"]["y"] - prev_dom["center"]["y"]
        delta_sep = hand_separation - float(previous_frame.get("hand_separation", 0.0))
        delta_open_palm = float(dominant.get("open_palm", 0.0)) - float(prev_dom.get("open_palm", 0.0))

    dominant_zone = infer_dominant_zone(dominant)

    scalars = [
        float(hand_count),
        float(left_metrics["active_count"]),
        float(right_metrics["active_count"]),
        float(left_metrics["open_palm"]),
        float(right_metrics["open_palm"]),
        float(left_metrics["pointing"]),
        float(right_metrics["pointing"]),
        float(left_metrics["pinch"]),
        float(right_metrics["pinch"]),
        float(dominant.get("pointing_outward", 0.0)),
        float(dominant.get("pointing_chest", 0.0)),
        float(dominant.get("three_finger_mouth_contact", 0.0)),
        float(dominant.get("face_distance", 9.0)),
        float(dominant.get("mouth_distance", 9.0)),
        float(dominant.get("chest_distance", 9.0)),
        float(dominant.get("temple_distance", 9.0)),
        float(dominant.get("cheek_distance", 9.0)),
        float(dominant.get("jaw_distance", 9.0)),
        float(dominant.get("chin_distance", 9.0)),
        float(dominant.get("forehead_distance", 9.0)),
        float(dominant.get("mustache_distance", 9.0)),
        float(dominant.get("upper_lip_distance", 9.0)),
        float(dominant.get("lower_lip_distance", 9.0)),
        float(dominant.get("lips_distance", 9.0)),
        float(dominant.get("nose_base_distance", 9.0)),
        float(dominant.get("right_cheek_distance", 9.0)),
        float(dominant.get("left_cheek_distance", 9.0)),
        float(dominant.get("jawline_distance", 9.0)),
        float(dominant.get("two_fingertips_joined", 0.0)),
        float(dominant.get("all_fingertips_together", 0.0)),
        float(dominant.get("fist", 0.0)),
        float(dominant.get("yery_shape", 0.0)),
        float(hand_separation),
        float(delta_x),
        float(delta_y),
        float(delta_sep),
        float(delta_open_palm),
        float(dominant["index_tip_z"]),
    ]

    vector = np.asarray(
        flatten_points(normalized["left_hand"])
        + flatten_points(normalized["right_hand"])
        + flatten_points(normalized["face"])
        + flatten_points(normalized["pose"])
        + scalars,
        dtype=np.float32,
    )

    return {
        "vector": vector,
        "left_metrics": left_metrics,
        "right_metrics": right_metrics,
        "dominant_name": dominant_name,
        "dominant": dominant,
        "dominant_zone": dominant_zone,
        "hand_count": hand_count,
        "hand_separation": hand_separation,
        "normalized": normalized,
    }


def build_sequence_observations(frames: list[dict]) -> list[dict]:
    observations = []
    previous = None
    for frame in frames:
        observation = build_frame_observation(frame, previous_frame=previous)
        observations.append(observation)
        previous = observation
    return observations


def resample_or_pad(matrix: np.ndarray, sequence_length: int) -> np.ndarray:
    if matrix.shape[0] == 0:
        return np.zeros((sequence_length, 1), dtype=np.float32)
    if matrix.shape[0] == sequence_length:
        return matrix.astype(np.float32)
    if matrix.shape[0] > sequence_length:
        indices = np.linspace(0, matrix.shape[0] - 1, sequence_length).astype(int)
        return matrix[indices].astype(np.float32)
    pad_count = sequence_length - matrix.shape[0]
    padding = np.repeat(matrix[-1][None, :], pad_count, axis=0)
    return np.concatenate([matrix, padding], axis=0).astype(np.float32)


def build_sequence_tensor(frames: list[dict], sequence_length: int = 32) -> np.ndarray:
    observations = build_sequence_observations(frames)
    if not observations:
        return np.zeros((sequence_length, 1), dtype=np.float32)
    matrix = np.stack([item["vector"] for item in observations], axis=0)
    return resample_or_pad(matrix, sequence_length)


def count_direction_changes(values: list[float], min_delta: float = 0.012) -> int:
    previous_direction = 0
    changes = 0
    for index in range(1, len(values)):
        delta = float(values[index]) - float(values[index - 1])
        if abs(delta) < min_delta:
            continue
        direction = 1 if delta > 0 else -1
        if previous_direction and direction != previous_direction:
            changes += 1
        previous_direction = direction
    return changes
