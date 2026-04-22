from __future__ import annotations

from gesture_translator.core.feature_builder import build_sequence_observations, count_direction_changes


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _mean(values: list[float], default: float = 0.0) -> float:
    if not values:
        return default
    return sum(values) / len(values)


def _stable_ratio(values: list[float], threshold: float) -> float:
    if not values:
        return 0.0
    return sum(1 for value in values if float(value) >= threshold) / len(values)


def _first_hit_index(values: list[float], threshold: float) -> int | None:
    for index, value in enumerate(values):
        if float(value) <= threshold:
            return index
    return None


def _first_joint_hit_index(
    distances: list[float],
    strengths: list[float],
    distance_threshold: float,
    strength_threshold: float,
) -> int | None:
    for index, (distance, strength) in enumerate(zip(distances, strengths, strict=False)):
        if float(distance) <= float(distance_threshold) and float(strength) >= float(strength_threshold):
            return index
    return None


def score_temporal_rules(frames: list[dict]) -> tuple[dict[str, float], dict]:
    observations = build_sequence_observations(frames)
    if not observations:
        return {"none": 1.0}, {}

    dominant_frames = [item for item in observations if item.get("dominant", {}).get("present")]
    if not dominant_frames:
        return {"none": 1.0}, {}

    dominant_x = [item["dominant"]["center"]["x"] for item in dominant_frames]
    dominant_y = [item["dominant"]["center"]["y"] for item in dominant_frames]
    dominant_z = [item["dominant"]["index_tip_z"] for item in dominant_frames]
    chest_distance = [item["dominant"]["chest_distance"] for item in dominant_frames]
    chin_distance = [item["dominant"]["chin_distance"] for item in dominant_frames]
    forehead_distance = [item["dominant"].get("forehead_distance", 9.0) for item in dominant_frames]
    upper_lip_distance = [item["dominant"].get("upper_lip_distance", 9.0) for item in dominant_frames]
    lower_lip_distance = [item["dominant"].get("lower_lip_distance", 9.0) for item in dominant_frames]
    lips_distance = [item["dominant"].get("lips_distance", 9.0) for item in dominant_frames]
    nose_base_distance = [item["dominant"].get("nose_base_distance", 9.0) for item in dominant_frames]
    right_cheek_distance = [item["dominant"].get("right_cheek_distance", 9.0) for item in dominant_frames]
    left_cheek_distance = [item["dominant"].get("left_cheek_distance", 9.0) for item in dominant_frames]
    jawline_distance = [item["dominant"].get("jawline_distance", 9.0) for item in dominant_frames]

    separations = [item.get("hand_separation", 0.0) for item in dominant_frames]
    open_palm_series = [item["dominant"].get("open_palm", 0.0) for item in dominant_frames]
    two_fingertips_series = [item["dominant"].get("two_fingertips_joined", 0.0) for item in dominant_frames]
    all_fingertips_series = [item["dominant"].get("all_fingertips_together", 0.0) for item in dominant_frames]
    fist_series = [item["dominant"].get("fist", 0.0) for item in dominant_frames]
    outward_series = [item["dominant"].get("pointing_outward", 0.0) for item in dominant_frames]
    chest_point_series = [item["dominant"].get("pointing_chest", 0.0) for item in dominant_frames]
    contact_zones = [str(item["dominant"].get("contact_zone", "space")) for item in dominant_frames]

    x_range = (max(dominant_x) - min(dominant_x)) if len(dominant_x) >= 2 else 0.0
    y_range = (max(dominant_y) - min(dominant_y)) if len(dominant_y) >= 2 else 0.0
    x_changes = count_direction_changes(dominant_x)
    open_close_changes = count_direction_changes(open_palm_series, min_delta=0.14)
    separation_growth = (separations[-1] - separations[0]) if len(separations) >= 2 else 0.0
    outward_depth = (dominant_z[0] - dominant_z[-1]) if len(dominant_z) >= 2 else 0.0
    downward_delta = (dominant_y[-1] - dominant_y[0]) if len(dominant_y) >= 2 else 0.0
    upward_delta = (dominant_y[0] - dominant_y[-1]) if len(dominant_y) >= 2 else 0.0

    # "Спасибо": кулак касается лба (без подбородка).
    # Учитываем форму кулака прямо в момент касания, чтобы убрать ложные срабатывания.
    forehead_idx = _first_joint_hit_index(forehead_distance, fist_series, distance_threshold=0.25, strength_threshold=0.54)
    forehead_touch_flags = [
        (float(dist) <= 0.25 and float(fist) >= 0.54)
        for dist, fist in zip(forehead_distance, fist_series, strict=False)
    ]
    forehead_touch_ratio = _stable_ratio([1.0 if flag else 0.0 for flag in forehead_touch_flags], threshold=0.5)
    transition_detected = forehead_idx is not None
    temporal_order = "forehead" if transition_detected else "invalid"
    sequence_valid = bool(forehead_idx is not None)
    temporal_match = bool(transition_detected)

    right_cheek_hits = [index for index, value in enumerate(right_cheek_distance) if value <= 0.32]
    left_cheek_hits = [index for index, value in enumerate(left_cheek_distance) if value <= 0.32]
    cheek_path_valid = bool(
        right_cheek_hits
        and left_cheek_hits
        and min(left_cheek_hits) > min(right_cheek_hits)
        and _mean(jawline_distance, default=9.0) <= 0.4
    )

    two_fingertips_stability = _stable_ratio(two_fingertips_series, threshold=0.55)
    all_fingertips_stability = _stable_ratio(all_fingertips_series, threshold=0.55)
    fist_stability = _stable_ratio(fist_series, threshold=0.58)

    upper_lip_precision = _stable_ratio(upper_lip_distance, threshold=0.17)
    lower_lip_pollution = _stable_ratio(lower_lip_distance, threshold=0.2)
    nose_pollution = _stable_ratio(nose_base_distance, threshold=0.18)
    chin_pollution = _stable_ratio(chin_distance, threshold=0.2)
    lips_contact = _stable_ratio(lips_distance, threshold=0.22)

    mouth_close_ratio = _stable_ratio(lips_distance, threshold=0.22)
    chest_end = chest_distance[-1] if chest_distance else 9.0
    pointing_outward_avg = _mean(outward_series)
    pointing_chest_avg = _mean(chest_point_series)

    muzhchina_temporal_gate = (
        two_fingertips_stability >= 0.5
        and upper_lip_precision >= 0.3
        and lower_lip_pollution <= 0.24
        and nose_pollution <= 0.16
        and chin_pollution <= 0.14
        and x_range <= 0.2
        and y_range <= 0.2
    )
    est_temporal_gate = all_fingertips_stability >= 0.46 and lips_contact >= 0.34 and not (
        two_fingertips_stability >= 0.5 and upper_lip_precision >= 0.3
    )
    zhenshchina_temporal_gate = (
        _stable_ratio(open_palm_series, 0.55) >= 0.5
        and cheek_path_valid
        and upper_lip_precision <= 0.2
        and temporal_order != "chin->forehead"
    )
    spasibo_temporal_gate = (
        fist_stability >= 0.5
        and transition_detected
        and forehead_touch_ratio >= 0.14
        and x_range <= 0.3
        and y_range <= 0.3
    )

    scores = {
        "privet": clamp01(
            clamp01((x_range - 0.03) / 0.18) * 0.46
            + clamp01((2 - abs(x_changes - 1)) / 2) * 0.42
            + clamp01((0.12 - y_range) / 0.12) * 0.1
        ),
        "poka": clamp01(
            clamp01(open_close_changes / 3) * 0.44
            + clamp01((x_range - 0.01) / 0.16) * 0.2
            + clamp01((0.2 - y_range) / 0.2) * 0.12
        ),
        "ya": clamp01(
            clamp01((1.0 - chest_end) / 1.0) * 0.5 + clamp01(pointing_chest_avg) * 0.34 + clamp01((0.12 - x_range) / 0.12) * 0.12
        ),
        "ty": clamp01(
            clamp01(pointing_outward_avg) * 0.4 + clamp01((chest_end - 0.9) / 1.1) * 0.26 + clamp01((outward_depth + 0.08) / 0.24) * 0.2
        ),
        "muzhchina": (
            clamp01(two_fingertips_stability * 0.38 + upper_lip_precision * 0.36 + clamp01((0.2 - (x_range + y_range) / 2) / 0.2) * 0.18)
            if muzhchina_temporal_gate
            else 0.0
        ),
        "zhenshchina": (
            clamp01(_stable_ratio(open_palm_series, 0.55) * 0.3 + clamp01((x_range + y_range) / 0.34) * 0.22 + (0.42 if cheek_path_valid else 0.0))
            if zhenshchina_temporal_gate
            else 0.0
        ),
        "krasivyy": clamp01(clamp01((x_range + y_range) / 0.35) * 0.34 + _stable_ratio(open_palm_series, 0.45) * 0.2),
        "spasibo": (
            clamp01(
                fist_stability * 0.24
                + (0.62 if transition_detected else 0.0)
                + clamp01((forehead_touch_ratio - 0.14) / 0.76) * 0.1
                + clamp01((0.3 - (x_range + y_range) / 2) / 0.3) * 0.06
            )
            if spasibo_temporal_gate
            else 0.0
        ),
        "bolshoy": clamp01(clamp01(separation_growth / 0.85) * 0.56 + (clamp01((separations[-1] - 0.85) / 1.2) * 0.28 if separations else 0.0)),
        "malenkiy": clamp01(clamp01((1.18 - _mean(separations, default=1.18)) / 1.18) * 0.44 + clamp01((0.18 - y_range) / 0.18) * 0.2),
        "est": (
            clamp01(all_fingertips_stability * 0.44 + mouth_close_ratio * 0.34 + clamp01((0.2 - (x_range + y_range) / 2) / 0.2) * 0.16)
            if est_temporal_gate
            else 0.0
        ),
    }

    scores["none"] = clamp01(1.0 - max(scores.values()))
    start_zone = contact_zones[0] if contact_zones else "none"
    end_zone = contact_zones[-1] if contact_zones else "none"

    return scores, {
        "motion": {
            "x_range": round(x_range, 3),
            "y_range": round(y_range, 3),
            "x_direction_changes": int(x_changes),
            "open_close_changes": int(open_close_changes),
            "separation_growth": round(separation_growth, 3),
            "outward_depth": round(outward_depth, 3),
            "downward_delta": round(downward_delta, 3),
            "upward_delta": round(upward_delta, 3),
            "start_zone": start_zone,
            "end_zone": end_zone,
            "sequence_valid": bool(sequence_valid),
            "temporal_match": bool(temporal_match),
            "temporal_order": temporal_order,
            "transition_detected": bool(transition_detected),
            "forehead_touch_ratio": round(float(forehead_touch_ratio), 3),
            "cheek_jaw_path": bool(cheek_path_valid),
            "two_fingertips_stability": round(two_fingertips_stability, 3),
            "all_fingertips_stability": round(all_fingertips_stability, 3),
            "fist_stability": round(fist_stability, 3),
            "upper_lip_precision": round(upper_lip_precision, 3),
            "lips_contact": round(lips_contact, 3),
            "lower_lip_pollution": round(lower_lip_pollution, 3),
            "nose_pollution": round(nose_pollution, 3),
            "chin_pollution": round(chin_pollution, 3),
            "temporal_gate_muzhchina": bool(muzhchina_temporal_gate),
            "temporal_gate_zhenshchina": bool(zhenshchina_temporal_gate),
            "temporal_gate_spasibo": bool(spasibo_temporal_gate),
            "temporal_gate_est": bool(est_temporal_gate),
        }
    }
