from __future__ import annotations

import cv2


HAND_CONNECTIONS = [
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17),
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
]
POSE_CONNECTIONS = [(0, 11), (0, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def _draw_point(frame, point, color, radius=4):
    if not point:
        return
    h, w = frame.shape[:2]
    x = int(float(point.get("x", 0.0)) * w)
    y = int(float(point.get("y", 0.0)) * h)
    cv2.circle(frame, (x, y), radius, color, -1)


def _draw_connections(frame, points, connections, color):
    if not points:
        return
    h, w = frame.shape[:2]
    for start_idx, end_idx in connections:
        if start_idx >= len(points) or end_idx >= len(points):
            continue
        start = points[start_idx]
        end = points[end_idx]
        cv2.line(
            frame,
            (int(float(start.get("x", 0.0)) * w), int(float(start.get("y", 0.0)) * h)),
            (int(float(end.get("x", 0.0)) * w), int(float(end.get("y", 0.0)) * h)),
            color,
            2,
        )


def draw_landmarks_overlay(frame, payload: dict) -> None:
    face = payload.get("face", []) or []
    pose = payload.get("pose", []) or []
    left_hand = payload.get("left_hand", []) or []
    right_hand = payload.get("right_hand", []) or []

    for face_index in (1, 33, 61, 152, 263, 291):
        if face_index < len(face):
            _draw_point(frame, face[face_index], (255, 170, 60), radius=5)

    _draw_connections(frame, pose, POSE_CONNECTIONS, (110, 180, 255))
    _draw_connections(frame, left_hand, HAND_CONNECTIONS, (30, 220, 255))
    _draw_connections(frame, right_hand, HAND_CONNECTIONS, (30, 220, 255))
    for points in (left_hand, right_hand):
        for point in points:
            _draw_point(frame, point, (30, 220, 255), radius=4)


def draw_prediction_panel(frame, prediction: dict) -> None:
    label = str(prediction.get("label", "none"))
    confidence = float(prediction.get("confidence", 0.0))
    cv2.rectangle(frame, (16, 16), (520, 126), (22, 28, 36), -1)
    cv2.rectangle(frame, (16, 16), (520, 126), (80, 110, 140), 1)
    cv2.putText(frame, f"Word: {label}", (32, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 245, 245), 2)
    cv2.putText(frame, f"Confidence: {confidence:.2f}", (32, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (190, 220, 255), 2)
