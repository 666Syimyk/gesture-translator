from __future__ import annotations

import cv2


def draw_record_status(frame, label: str, saved_count: int, target_count: int, countdown_value, status_text: str) -> None:
    cv2.rectangle(frame, (16, 140), (560, 280), (15, 20, 26), -1)
    cv2.rectangle(frame, (16, 140), (560, 280), (80, 110, 140), 1)
    cv2.putText(frame, f"Record label: {label}", (32, 176), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Saved: {saved_count}/{target_count}", (32, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 220, 255), 2)
    if countdown_value is not None:
        cv2.putText(frame, f"Countdown: {countdown_value}", (32, 244), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 210, 120), 2)
    cv2.putText(frame, status_text, (32, 268), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (190, 240, 190), 2)


def draw_live_status(frame, buffer_size: int, sequence_length: int, model_status: str, debug: dict) -> None:
    cv2.rectangle(frame, (16, 140), (620, 360), (15, 20, 26), -1)
    cv2.rectangle(frame, (16, 140), (620, 360), (80, 110, 140), 1)
    cv2.putText(frame, f"Mode: {model_status}", (32, 174), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"Buffer: {buffer_size}/{sequence_length}", (32, 204), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (190, 220, 255), 2)

    lines = []
    zone = debug.get("zone", {})
    motion = debug.get("motion", {})
    if zone:
        lines.append(f"zone one/two: {zone.get('one_hand_ratio', 0):.2f}/{zone.get('two_hands_ratio', 0):.2f}")
        lines.append(
            "zone upper/mustache/cheek: "
            f"{zone.get('upper_zone_ratio', zone.get('upper_face_ratio', 0)):.2f}/"
            f"{zone.get('mustache_zone_ratio', 0):.2f}/"
            f"{zone.get('cheek_zone_ratio', zone.get('lower_face_ratio', 0)):.2f}"
        )
        lines.append(f"zone sep: {zone.get('hand_separation_avg', 0):.2f}")
    if motion:
        lines.append(f"motion x/y: {motion.get('x_range', 0):.2f}/{motion.get('y_range', 0):.2f}")
        lines.append(f"repeats: {motion.get('x_direction_changes', 0)} palm: {motion.get('open_close_changes', 0)}")
        lines.append(f"growth/out/down: {motion.get('separation_growth', 0):.2f}/{motion.get('outward_depth', 0):.2f}/{motion.get('downward_delta', 0):.2f}")

    top_sources = (
        ("rule spatial", debug.get("spatial_top", [])),
        ("rule temporal", debug.get("temporal_top", [])),
        ("model", debug.get("model_top", [])),
    )
    for name, items in top_sources:
        if items:
            label, score = items[0]
            lines.append(f"{name}: {label} {score:.2f}")

    y = 234
    for line in lines[:6]:
        cv2.putText(frame, line, (32, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (205, 235, 205), 1)
        y += 26
