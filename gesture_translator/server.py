from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from collections import deque
from statistics import StatisticsError, mean

import cv2
import numpy as np

from gesture_translator.config import (
    CONFIDENCE_THRESHOLD,
    COOLDOWN_SECONDS,
    MIN_SEQUENCE_LENGTH,
    SEQUENCE_LENGTH,
    STABLE_VOTES_REQUIRED,
    SMOOTHING_WINDOW,
    ensure_runtime_dirs,
    load_runtime_config,
)
from gesture_translator.core.cooldown import CooldownGate
from gesture_translator.core.feature_builder import build_sequence_observations
from gesture_translator.core.mp_extractor import MediaPipeHolisticExtractor
from gesture_translator.core.smoother import PredictionSmoother
from gesture_translator.models.infer import HybridInferenceEngine
from gesture_translator.utils.logger import get_logger


LOGGER = get_logger("gesture_translator.server")

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

LABEL_RU_BY_ID = {
    "none": "none",
    "privet": "\u041f\u0440\u0438\u0432\u0435\u0442",
    "poka": "\u041f\u043e\u043a\u0430",
    "ya": "\u042f",
    "ty": "\u0422\u044b",
    "muzhchina": "\u041c\u0443\u0436\u0447\u0438\u043d\u0430",
    "zhenshchina": "\u0416\u0435\u043d\u0449\u0438\u043d\u0430",
    "bolshoy": "\u0411\u043e\u043b\u044c\u0448\u043e\u0439",
    "malenkiy": "\u041c\u0430\u043b\u0435\u043d\u044c\u043a\u0438\u0439",
    "krasivyy": "\u041a\u0440\u0430\u0441\u0438\u0432\u044b\u0439",
    "spasibo": "\u0421\u043f\u0430\u0441\u0438\u0431\u043e",
    "est": "\u0415\u0441\u0442\u044c",
}

LABEL_ID_MAP = {value: key for key, value in LABEL_RU_BY_ID.items()}
LABEL_ID_MAP_LOWER = {value.lower(): key for key, value in LABEL_ID_MAP.items()}
LABEL_ID_ALIASES = {
    "krasiviy": "krasivyy",
}


def label_to_id(label: str | None) -> str:
    normalized = str(label or "none").strip() or "none"
    lower_normalized = normalized.lower()
    if normalized in LABEL_RU_BY_ID:
        return normalized
    if lower_normalized in LABEL_RU_BY_ID:
        return lower_normalized
    return (
        LABEL_ID_ALIASES.get(lower_normalized)
        or LABEL_ID_MAP.get(normalized)
        or LABEL_ID_MAP_LOWER.get(lower_normalized)
        or "none"
    )


def label_to_ru(label_id: str | None, fallback: str | None = None) -> str:
    normalized_id = label_to_id(label_id)
    if normalized_id in LABEL_RU_BY_ID:
        return LABEL_RU_BY_ID[normalized_id]
    return str(fallback or "none").strip() or "none"


def serialize_ranked(items: list[tuple[str, float]] | None) -> list[dict]:
    result = []
    for label, score in items or []:
        normalized = str(label or "none").strip() or "none"
        label_id = label_to_id(normalized)
        result.append(
            {
                "label_id": label_id,
                "label_ru": label_to_ru(label_id, normalized),
                "score": round(float(score or 0.0), 4),
            }
        )
    return result


def decode_image(image_b64: str):
    payload = str(image_b64 or "")
    if "," in payload:
        payload = payload.split(",", 1)[1]
    binary = base64.b64decode(payload)
    image = cv2.imdecode(np.frombuffer(binary, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode JPEG frame.")
    return image


def safe_round(value: float | int | None, digits: int = 4) -> float:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return 0.0


def safe_mean(values: list[float], default: float = 0.0) -> float:
    try:
        return mean(values) if values else default
    except StatisticsError:
        return default


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def compute_tracking_quality(quality: dict, hands_count: int) -> float:
    hand_score = 1.0 if hands_count >= 1 else 0.0
    face_score = 1.0 if quality.get("has_face") else 0.0
    pose_score = 1.0 if quality.get("has_pose") else 0.0
    return clamp01(hand_score * 0.5 + face_score * 0.3 + pose_score * 0.2)


class GestureServeWorker:
    def __init__(
        self,
        artifact_dir: str | None,
        sequence_length: int,
        min_sequence_length: int,
        confidence_threshold: float,
        smoothing_window: int,
        stable_votes_required: int,
        cooldown_seconds: float,
    ) -> None:
        ensure_runtime_dirs()
        self.sequence_length = max(8, int(sequence_length))
        self.min_sequence_length = max(4, min(int(min_sequence_length), self.sequence_length))
        self.extractor = MediaPipeHolisticExtractor()
        self.engine = HybridInferenceEngine(
            artifact_dir=artifact_dir,
            sequence_length=self.sequence_length,
            confidence_threshold=confidence_threshold,
        )
        self.smoother = PredictionSmoother(
            window_size=smoothing_window,
            stable_votes_required=stable_votes_required,
        )
        self.cooldown = CooldownGate(cooldown_seconds)
        self.frames = deque(maxlen=self.sequence_length)
        self.frame_index = 0
        self.last_timestamp_ms = 0
        self.process_times_ms = deque(maxlen=40)
        self.inference_times_ms = deque(maxlen=40)
        self.decoding_times_ms = deque(maxlen=40)
        self.frames_received = 0
        self.frames_decoded = 0
        self.decode_failures = 0
        self.landmark_failures = 0
        self.predictions_emitted = 0

    def close(self) -> None:
        self.extractor.close()

    def reset(self) -> None:
        self.frames.clear()
        self.frame_index = 0
        self.last_timestamp_ms = 0
        self.smoother = PredictionSmoother(
            window_size=self.smoother.window_size,
            stable_votes_required=self.smoother.stable_votes_required,
        )
        self.cooldown = CooldownGate(self.cooldown.cooldown_seconds)
        self.process_times_ms.clear()
        self.inference_times_ms.clear()
        self.decoding_times_ms.clear()

    def process_message(self, message: dict) -> tuple[list[dict], bool]:
        message_type = str(message.get("type", "")).strip().lower()
        if message_type == "ping":
            return (
                [
                    {
                        "type": "pong",
                        "ts": int(time.time() * 1000),
                        "worker_uptime_ms": int(time.perf_counter() * 1000),
                        "frames_received": self.frames_received,
                        "predictions_emitted": self.predictions_emitted,
                    }
                ],
                False,
            )
        if message_type == "reset":
            self.reset()
            return (
                [
                    self._build_state_message(
                        tracking_ok=False,
                        hands_count=0,
                        dominant_zone="none",
                        movement_type="none",
                        repeat_count=0,
                        client_frame_id=0,
                        client_sent_at_ms=0,
                        timestamp_ms=int(time.time() * 1000),
                        decode_ms=0.0,
                        processing_ms=0.0,
                        landmarks_found=False,
                        none_reason="reset",
                        quality={},
                        extra={"buffer_cleared": True},
                    )
                ],
                False,
            )
        if message_type == "shutdown":
            return ([{"type": "shutdown", "ok": True}], True)
        if message_type != "frame":
            return ([{"type": "error", "message": f"Unsupported message type: {message_type or 'empty'}"}], False)

        return (self._process_frame_message(message), False)

    def _process_frame_message(self, message: dict) -> list[dict]:
        started_at = time.perf_counter()
        image_b64 = message.get("image_b64")
        if not image_b64:
            return [{"type": "error", "message": "Frame payload is missing image_b64."}]

        client_frame_id = int(message.get("client_frame_id") or 0)
        client_sent_at_ms = message.get("client_sent_at_ms")
        self.frames_received += 1

        decode_started_at = time.perf_counter()
        try:
            frame = decode_image(image_b64)
        except Exception as exc:  # noqa: BLE001
            self.decode_failures += 1
            return [
                {
                    "type": "error",
                    "message": str(exc),
                    "client_frame_id": client_frame_id,
                    "client_sent_at_ms": client_sent_at_ms,
                    "reason": "decode_failed",
                }
            ]
        decode_ms = (time.perf_counter() - decode_started_at) * 1000
        self.frames_decoded += 1
        self.decoding_times_ms.append(decode_ms)

        timestamp_ms = self._next_timestamp(message.get("ts"))
        payload = self.extractor.process(frame, frame_index=self.frame_index, timestamp_ms=timestamp_ms)
        self.frame_index += 1
        self.frames.append(payload)

        observations = build_sequence_observations(list(self.frames))
        last_observation = observations[-1] if observations else None
        hands_count = int(bool(payload["quality"]["has_left_hand"])) + int(bool(payload["quality"]["has_right_hand"]))
        landmarks_found = bool(payload["quality"]["has_face"] or payload["quality"]["has_pose"] or hands_count >= 1)
        tracking_ok = bool((payload["quality"]["has_face"] or payload["quality"]["has_pose"]) and hands_count >= 1)
        if not landmarks_found:
            self.landmark_failures += 1
        dominant_zone = self._infer_zone(last_observation)
        movement_type, repeat_count = ("none", 0)
        none_reason = "buffer_warming" if len(self.frames) < self.min_sequence_length else "tracking_lost"
        total_processing_ms = (time.perf_counter() - started_at) * 1000
        self.process_times_ms.append(total_processing_ms)

        messages = [
            self._build_state_message(
                tracking_ok=tracking_ok,
                hands_count=hands_count,
                dominant_zone=dominant_zone,
                movement_type=movement_type,
                repeat_count=repeat_count,
                client_frame_id=client_frame_id,
                client_sent_at_ms=client_sent_at_ms,
                timestamp_ms=timestamp_ms,
                decode_ms=decode_ms,
                processing_ms=total_processing_ms,
                landmarks_found=landmarks_found,
                none_reason=none_reason,
                quality=payload["quality"],
            )
        ]

        if len(self.frames) < self.min_sequence_length or not tracking_ok:
            return messages

        inference_started_at = time.perf_counter()
        prediction = self.engine.predict(list(self.frames))
        inference_ms = (time.perf_counter() - inference_started_at) * 1000
        self.inference_times_ms.append(inference_ms)
        movement_type, repeat_count = self._infer_motion(prediction.get("debug", {}))
        stable_prediction = self.smoother.update(
            prediction["label"],
            prediction["confidence"],
            extra=prediction,
        )

        emitted_prediction = prediction
        is_stable = False
        none_reason = self._infer_none_reason(prediction)
        cooldown_allowed = (
            self.cooldown.allow(stable_prediction["label"])
            if stable_prediction
            else False
        )
        if stable_prediction and cooldown_allowed:
            emitted_prediction = stable_prediction
            is_stable = True
            none_reason = None
        elif stable_prediction and not cooldown_allowed:
            none_reason = "cooldown"
        elif prediction["label"] != "none" and not stable_prediction:
            none_reason = "awaiting_stability"

        self.predictions_emitted += 1
        messages.append(
            self._build_prediction_message(
                prediction=emitted_prediction,
                tracking_ok=tracking_ok,
                hands_count=hands_count,
                dominant_zone=dominant_zone,
                movement_type=movement_type,
                repeat_count=repeat_count,
                stable=is_stable,
                client_frame_id=client_frame_id,
                client_sent_at_ms=client_sent_at_ms,
                timestamp_ms=timestamp_ms,
                decode_ms=decode_ms,
                processing_ms=(time.perf_counter() - started_at) * 1000,
                inference_ms=inference_ms,
                landmarks_found=landmarks_found,
                none_reason=none_reason,
                quality=payload["quality"],
            )
        )
        return messages

    def _infer_none_reason(self, prediction: dict) -> str:
        label = str(prediction.get("label") or "none").strip() or "none"
        if label != "none":
            return "candidate_detected"

        scores = prediction.get("scores", {}) or {}
        non_none_scores = [
            (candidate_label, float(score))
            for candidate_label, score in scores.items()
            if str(candidate_label or "").strip() not in {"", "none"}
        ]
        if not non_none_scores:
            return "no_signal"

        best_label, best_score = max(non_none_scores, key=lambda item: item[1])
        if best_score < self.engine.confidence_threshold * 0.55:
            return f"low_signal:{best_label}"
        if best_score < self.engine.confidence_threshold:
            return f"low_confidence:{best_label}"
        return f"rejected:{best_label}"

    def _next_timestamp(self, incoming_ts) -> int:
        try:
            candidate = int(float(incoming_ts))
        except (TypeError, ValueError):
            candidate = int(time.time() * 1000)
        if candidate <= self.last_timestamp_ms:
            candidate = self.last_timestamp_ms + 1
        self.last_timestamp_ms = candidate
        return candidate

    def _infer_zone(self, observation: dict | None) -> str:
        if not observation:
            return "none"

        dominant = observation["dominant"]
        candidates = {
            "chest": float(dominant.get("chest_distance", 9.0)),
            "chin": float(dominant.get("chin_distance", 9.0)),
            "cheek": float(dominant.get("cheek_distance", 9.0)),
            "temple": float(dominant.get("temple_distance", 9.0)),
            "face": float(dominant.get("face_distance", 9.0)),
        }
        zone, distance = min(candidates.items(), key=lambda item: item[1])
        if zone == "face":
            return "face" if distance <= 1.25 else "space"
        return zone if distance <= 1.2 else "space"

    def _infer_motion(self, debug: dict | None) -> tuple[str, int]:
        if not debug:
            return ("none", 0)

        motion = debug.get("motion", {})
        x_changes = int(motion.get("x_direction_changes", 0))
        open_close_changes = int(motion.get("open_close_changes", 0))
        separation_growth = float(motion.get("separation_growth", 0.0))
        outward_depth = float(motion.get("outward_depth", 0.0))
        downward_delta = float(motion.get("downward_delta", 0.0))
        x_range = float(motion.get("x_range", 0.0))

        if open_close_changes >= 1:
            return ("open_close", open_close_changes)
        if x_changes >= 2 or x_range >= 0.14:
            return ("side_to_side", x_changes)
        if separation_growth >= 0.18:
            return ("expanding", 1)
        if outward_depth >= 0.05 and downward_delta >= 0.04:
            return ("forward_down", 1)
        if outward_depth >= 0.06:
            return ("forward", 1)
        if downward_delta >= 0.05:
            return ("downward", 1)
        return ("stable", 0)

    def _build_state_message(
        self,
        *,
        tracking_ok: bool,
        hands_count: int,
        dominant_zone: str,
        movement_type: str,
        repeat_count: int,
        client_frame_id: int,
        client_sent_at_ms,
        timestamp_ms: int,
        decode_ms: float,
        processing_ms: float,
        landmarks_found: bool,
        none_reason: str | None,
        quality: dict,
        extra: dict | None = None,
    ) -> dict:
        return {
            "type": "state",
            "client_frame_id": client_frame_id,
            "client_sent_at_ms": client_sent_at_ms,
            "worker_timestamp_ms": int(timestamp_ms),
            "buffer_size": len(self.frames),
            "sequence_length": self.sequence_length,
            "tracking_ok": bool(tracking_ok),
            "hands_count": int(hands_count),
            "dominant_zone": dominant_zone,
            "movement_type": movement_type,
            "repeat_count": int(repeat_count),
            "model_mode": self.engine.mode_name,
            "frame_decoded": True,
            "landmarks_found": bool(landmarks_found),
            "has_face": bool(quality.get("has_face")),
            "has_pose": bool(quality.get("has_pose")),
            "has_left_hand": bool(quality.get("has_left_hand")),
            "has_right_hand": bool(quality.get("has_right_hand")),
            "tracking_quality": safe_round(compute_tracking_quality(quality, hands_count), 3),
            "decode_ms": safe_round(decode_ms, 2),
            "processing_ms": safe_round(processing_ms, 2),
            "avg_processing_ms": safe_round(safe_mean(self.process_times_ms), 2),
            "avg_decode_ms": safe_round(safe_mean(self.decoding_times_ms), 2),
            "none_reason": none_reason,
            "frames_received": self.frames_received,
            "frames_decoded": self.frames_decoded,
            "decode_failures": self.decode_failures,
            "landmark_failures": self.landmark_failures,
            "dropped_frames": int(self.decode_failures + self.landmark_failures),
            **(extra or {}),
        }

    def _build_prediction_message(
        self,
        *,
        prediction: dict,
        tracking_ok: bool,
        hands_count: int,
        dominant_zone: str,
        movement_type: str,
        repeat_count: int,
        stable: bool,
        client_frame_id: int,
        client_sent_at_ms,
        timestamp_ms: int,
        decode_ms: float,
        processing_ms: float,
        inference_ms: float,
        landmarks_found: bool,
        none_reason: str | None,
        quality: dict,
    ) -> dict:
        raw_label = str(prediction.get("label") or "none").strip() or "none"
        label_id = label_to_id(raw_label)
        label_ru = label_to_ru(label_id, raw_label)
        debug = prediction.get("debug", {})
        zone_debug = debug.get("zone", {}) if isinstance(debug.get("zone"), dict) else {}
        motion_debug = debug.get("motion", {}) if isinstance(debug.get("motion"), dict) else {}
        spatial_top = serialize_ranked(debug.get("spatial_top"))
        temporal_top = serialize_ranked(debug.get("temporal_top"))
        model_top = serialize_ranked(debug.get("model_top"))
        top_model_score = model_top[0]["score"] if model_top else 0.0
        final_scores = prediction.get("scores", {}) or {}
        top_final_scores = serialize_ranked(
            sorted(
                final_scores.items(),
                key=lambda item: float(item[1]),
                reverse=True,
            )[:3]
        )
        top_final_label = top_final_scores[0] if top_final_scores else None
        return {
            "type": "prediction",
            "client_frame_id": client_frame_id,
            "client_sent_at_ms": client_sent_at_ms,
            "worker_timestamp_ms": int(timestamp_ms),
            "label_id": label_id,
            "label_ru": label_ru,
            "confidence": round(float(prediction.get("confidence") or 0.0), 4),
            "stable": bool(stable),
            "top_label_id": top_final_label["label_id"] if top_final_label else "none",
            "top_label_ru": top_final_label["label_ru"] if top_final_label else label_to_ru("none"),
            "top_confidence": top_final_label["score"] if top_final_label else 0.0,
            "debug": {
                "tracking_ok": bool(tracking_ok),
                "frame_decoded": True,
                "landmarks_found": bool(landmarks_found),
                "has_face": bool(quality.get("has_face")),
                "has_pose": bool(quality.get("has_pose")),
                "has_left_hand": bool(quality.get("has_left_hand")),
                "has_right_hand": bool(quality.get("has_right_hand")),
                "hands_count": int(hands_count),
                "tracking_quality": safe_round(compute_tracking_quality(quality, hands_count), 3),
                "dominant_zone": dominant_zone,
                "movement_type": movement_type,
                "repeat_count": int(repeat_count),
                "classifier_confidence": round(float(top_model_score), 4),
                "decode_ms": safe_round(decode_ms, 2),
                "processing_ms": safe_round(processing_ms, 2),
                "inference_ms": safe_round(inference_ms, 2),
                "avg_processing_ms": safe_round(safe_mean(self.process_times_ms), 2),
                "avg_inference_ms": safe_round(safe_mean(self.inference_times_ms), 2),
                "none_reason": none_reason,
                "predictions_emitted": self.predictions_emitted,
                "dropped_frames": int(self.decode_failures + self.landmark_failures),
                "rule_score_summary": {
                    "spatial_top": spatial_top[:2],
                    "temporal_top": temporal_top[:2],
                },
                "handshape": zone_debug.get("handshape"),
                "contact_zone": zone_debug.get("contact_zone"),
                "start_zone": motion_debug.get("start_zone"),
                "end_zone": motion_debug.get("end_zone"),
                "temporal_order": motion_debug.get("temporal_order"),
                "sequence_valid": motion_debug.get("sequence_valid"),
                "temporal_match": motion_debug.get("temporal_match"),
                "reason_for_reject": debug.get("reason_for_reject"),
                "competing_label": debug.get("competing_label"),
                "top_3_predictions": debug.get("top_3_predictions", []),
                **zone_debug,
                **motion_debug,
            },
            "top_final": top_final_scores,
            "top_rules": [
                {"kind": "spatial", **item} for item in spatial_top
            ]
            + [{"kind": "temporal", **item} for item in temporal_top],
            "top_model": model_top,
        }


def emit(message: dict) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def run_serve_mode(args: argparse.Namespace | None = None) -> int:
    config = load_runtime_config()
    artifact_dir = getattr(args, "artifact_dir", None) or config["artifacts_dir"]
    sequence_length = int(getattr(args, "sequence_length", SEQUENCE_LENGTH))
    min_sequence_length = int(getattr(args, "min_sequence_length", MIN_SEQUENCE_LENGTH))
    confidence_threshold = float(getattr(args, "confidence_threshold", CONFIDENCE_THRESHOLD))
    smoothing_window = int(getattr(args, "smoothing_window", SMOOTHING_WINDOW))
    stable_votes_required = int(getattr(args, "stable_votes_required", STABLE_VOTES_REQUIRED))
    cooldown_seconds = float(getattr(args, "cooldown_seconds", COOLDOWN_SECONDS))

    worker = GestureServeWorker(
        artifact_dir=artifact_dir,
        sequence_length=sequence_length,
        min_sequence_length=min_sequence_length,
        confidence_threshold=confidence_threshold,
        smoothing_window=smoothing_window,
        stable_votes_required=stable_votes_required,
        cooldown_seconds=cooldown_seconds,
    )

    emit(
        {
            "type": "ready",
            "sequence_length": worker.sequence_length,
            "min_sequence_length": worker.min_sequence_length,
            "model_mode": worker.engine.mode_name,
        }
    )

    should_exit = False
    try:
        for line in sys.stdin:
            raw = line.strip()
            if not raw:
                continue
            try:
                message = json.loads(raw)
                responses, should_exit = worker.process_message(message)
                for response in responses:
                    emit(response)
                if should_exit:
                    break
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("serve loop error")
                emit({"type": "error", "message": str(exc)})
    finally:
        worker.close()

    return 0


def build_parser() -> argparse.ArgumentParser:
    config = load_runtime_config()
    parser = argparse.ArgumentParser(description="Gesture translator serve mode.")
    parser.add_argument("--artifact-dir", default=config["artifacts_dir"])
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--min-sequence-length", type=int, default=MIN_SEQUENCE_LENGTH)
    parser.add_argument("--confidence-threshold", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--smoothing-window", type=int, default=SMOOTHING_WINDOW)
    parser.add_argument("--stable-votes-required", type=int, default=STABLE_VOTES_REQUIRED)
    parser.add_argument("--cooldown-seconds", type=float, default=COOLDOWN_SECONDS)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_serve_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())


