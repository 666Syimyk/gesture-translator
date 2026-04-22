from __future__ import annotations

from collections import Counter

from gesture_translator.core.feature_builder import build_sequence_observations


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _ratio(values: list[bool]) -> float:
    if not values:
        return 0.0
    return sum(1 for value in values if value) / len(values)


def _dominant_value(observations: list[dict], key: str) -> float:
    if not observations:
        return 0.0
    return sum(float(item["dominant"].get(key, 0.0)) for item in observations) / len(observations)


def score_spatial_rules(frames: list[dict]) -> tuple[dict[str, float], dict]:
    observations = build_sequence_observations(frames)
    if not observations:
        return {"none": 1.0}, {}

    count = len(observations)
    one_hand_ratio = sum(1 for item in observations if item["hand_count"] == 1) / count
    two_hands_ratio = sum(1 for item in observations if item["hand_count"] >= 2) / count
    open_palm_ratio = _dominant_value(observations, "open_palm")
    fist_ratio = _dominant_value(observations, "fist")
    pointing_ratio = _dominant_value(observations, "pointing")
    pointing_outward_ratio = _dominant_value(observations, "pointing_outward")
    pointing_chest_ratio = _dominant_value(observations, "pointing_chest")
    pinch_ratio = _dominant_value(observations, "pinch")
    two_fingertips_ratio = _dominant_value(observations, "two_fingertips_joined")
    all_fingertips_ratio = _dominant_value(observations, "all_fingertips_together")
    hand_separation_avg = sum(item["hand_separation"] for item in observations) / count

    upper_lip_hit = _ratio([item["dominant"].get("upper_lip_distance", 9.0) <= 0.16 for item in observations])
    lower_lip_hit = _ratio([item["dominant"].get("lower_lip_distance", 9.0) <= 0.18 for item in observations])
    lips_hit = _ratio([item["dominant"].get("lips_distance", 9.0) <= 0.2 for item in observations])
    nose_hit = _ratio([item["dominant"].get("nose_base_distance", 9.0) <= 0.16 for item in observations])
    chin_hit = _ratio([item["dominant"].get("chin_distance", 9.0) <= 0.18 for item in observations])
    forehead_hit = _ratio([item["dominant"].get("forehead_distance", 9.0) <= 0.2 for item in observations])
    cheek_jaw_hit = _ratio(
        [
            (
                item["dominant"].get("right_cheek_distance", 9.0) <= 0.28
                or item["dominant"].get("left_cheek_distance", 9.0) <= 0.28
                or item["dominant"].get("jawline_distance", 9.0) <= 0.28
            )
            for item in observations
        ]
    )

    # Strict anti-confusion gates.
    muzhchina_spatial_gate = (
        two_fingertips_ratio >= 0.58
        and upper_lip_hit >= 0.3
        and nose_hit <= 0.16
        and lower_lip_hit <= 0.22
        and chin_hit <= 0.12
    )
    est_spatial_gate = (
        all_fingertips_ratio >= 0.54
        and lips_hit >= 0.34
        and not (two_fingertips_ratio >= 0.56 and upper_lip_hit >= 0.3)
    )
    zhenshchina_spatial_gate = (
        open_palm_ratio >= 0.55
        and cheek_jaw_hit >= 0.42
        and upper_lip_hit <= 0.16
        and not (chin_hit >= 0.25 and forehead_hit >= 0.25)
    )
    # "Спасибо": кулак касается лба (без подбородка).
    spasibo_spatial_gate = fist_ratio >= 0.56 and forehead_hit >= 0.22 and chin_hit <= 0.22

    scores = {
        "privet": clamp01(one_hand_ratio * 0.2 + open_palm_ratio * 0.42 + clamp01((0.3 - chin_hit) / 0.3) * 0.16),
        "poka": clamp01(one_hand_ratio * 0.22 + open_palm_ratio * 0.25 + (1.0 - pointing_ratio) * 0.18),
        "ya": clamp01(one_hand_ratio * 0.16 + pointing_ratio * 0.34 + pointing_chest_ratio * 0.44),
        "ty": clamp01(one_hand_ratio * 0.16 + pointing_ratio * 0.31 + pointing_outward_ratio * 0.41),
        "muzhchina": (
            clamp01(
                one_hand_ratio * 0.16
                + two_fingertips_ratio * 0.36
                + upper_lip_hit * 0.34
                + clamp01((0.2 - nose_hit) / 0.2) * 0.08
                + clamp01((0.22 - lower_lip_hit) / 0.22) * 0.06
            )
            if muzhchina_spatial_gate
            else 0.0
        ),
        "zhenshchina": (
            clamp01(one_hand_ratio * 0.14 + open_palm_ratio * 0.34 + cheek_jaw_hit * 0.42)
            if zhenshchina_spatial_gate
            else 0.0
        ),
        "krasivyy": clamp01(one_hand_ratio * 0.2 + cheek_jaw_hit * 0.4 + open_palm_ratio * 0.2),
        "spasibo": (clamp01(one_hand_ratio * 0.12 + fist_ratio * 0.44 + forehead_hit * 0.38) if spasibo_spatial_gate else 0.0),
        "bolshoy": clamp01(two_hands_ratio * 0.45 + clamp01((hand_separation_avg - 0.78) / 1.15) * 0.41),
        "malenkiy": clamp01(one_hand_ratio * 0.28 + clamp01((1.15 - hand_separation_avg) / 1.15) * 0.35 + pinch_ratio * 0.2),
        "est": (
            clamp01(one_hand_ratio * 0.12 + all_fingertips_ratio * 0.42 + lips_hit * 0.32)
            if est_spatial_gate
            else 0.0
        ),
    }
    scores["none"] = clamp01(1.0 - max(scores.values()))

    contact_zone_counts = Counter(
        str(item["dominant"].get("contact_zone", "space")).strip() or "space"
        for item in observations
    )
    handshape_counts = Counter(
        str(item["dominant"].get("handshape_label", "unknown")).strip() or "unknown"
        for item in observations
    )

    return scores, {
        "zone": {
            "one_hand_ratio": round(one_hand_ratio, 3),
            "two_hands_ratio": round(two_hands_ratio, 3),
            "upper_lip_hit_ratio": round(upper_lip_hit, 3),
            "lower_lip_hit_ratio": round(lower_lip_hit, 3),
            "lips_hit_ratio": round(lips_hit, 3),
            "nose_base_hit_ratio": round(nose_hit, 3),
            "chin_hit_ratio": round(chin_hit, 3),
            "forehead_hit_ratio": round(forehead_hit, 3),
            "cheek_jaw_hit_ratio": round(cheek_jaw_hit, 3),
            "two_fingertips_ratio": round(two_fingertips_ratio, 3),
            "all_fingertips_ratio": round(all_fingertips_ratio, 3),
            "fist_ratio": round(fist_ratio, 3),
            "open_palm_ratio": round(open_palm_ratio, 3),
            "hand_separation_avg": round(hand_separation_avg, 3),
            "contact_zone": contact_zone_counts.most_common(1)[0][0] if contact_zone_counts else "space",
            "handshape": handshape_counts.most_common(1)[0][0] if handshape_counts else "unknown",
            "dominant_zone": observations[-1].get("dominant_zone", "space"),
            "spatial_gate_muzhchina": bool(muzhchina_spatial_gate),
            "spatial_gate_zhenshchina": bool(zhenshchina_spatial_gate),
            "spatial_gate_spasibo": bool(spasibo_spatial_gate),
            "spatial_gate_est": bool(est_spatial_gate),
        }
    }
