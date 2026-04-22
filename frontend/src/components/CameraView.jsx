import { useEffect, useRef, useState } from "react";
import {
  FilesetResolver,
  HandLandmarker,
  HolisticLandmarker,
} from "@mediapipe/tasks-vision";
import { predictGesture, UNKNOWN_GESTURE } from "../gestureUtils";

const TASKS_VISION_VERSION = "0.10.32";
const STABLE_FRAME_TARGET = 10;
const NO_HAND_GRACE_FRAMES = 10;
const PREDICTION_WINDOW_SIZE = 16;
const MAJORITY_RATIO_TARGET = 0.8;
const WEIGHTED_RATIO_TARGET = 0.74;
const MIN_CONFIRM_CONFIDENCE = 88;
const MIN_CONFIRM_MARGIN = 0.08;
const MIN_STABLE_AVG_MARGIN = 0.07;
const MIN_STABLE_AVG_CONFIDENCE = 0.72;
const MIN_SEQUENCE_TRIGGER_WEIGHTED_RATIO = 0.58;
const MIN_SEQUENCE_TRIGGER_RATIO = 0.64;
const MIN_SEQUENCE_TRIGGER_CONFIDENCE = 0.68;
const MIN_SEQUENCE_TRIGGER_MARGIN = 0.055;
const MIN_ALPHABET_SEQUENCE_TRIGGER_WEIGHTED_RATIO = 0.34;
const MIN_ALPHABET_SEQUENCE_TRIGGER_CONFIDENCE = 0.42;
const LIVE_SEQUENCE_QUALITY_RESET_FRAMES = 5;
const CALIBRATION_DURATION_MS = 3600;
const CALIBRATION_MIN_FRAMES = 40;
const MIN_HAND_PRESENCE_CONFIDENCE = 0.5;
const JITTER_WINDOW_SIZE = 10;
const HAND_SWITCH_HYSTERESIS_RATIO = 1.2;
const DEFAULT_MAX_CONFIRM_JITTER = 0.03;
const DEFAULT_MIN_HAND_COVERAGE = 0.0075;
const ALPHABET_MAX_CONFIRM_JITTER = 0.018;
const ALPHABET_MIN_HAND_COVERAGE = 0.011;
const SIGN_MAX_CONFIRM_JITTER = 0.035;
const SIGN_MIN_HAND_COVERAGE = 0.0085;
const STATUS_UPDATE_INTERVAL_MS = 60;
const PREDICTION_EMIT_INTERVAL_MS = 50;
const STREAM_FRAME_INTERVAL_MS = 105;
const STREAM_FRAME_WIDTH = 272;
const STREAM_FRAME_HEIGHT = 204;
const STREAM_FRAME_JPEG_QUALITY = 0.58;
const OVERLAY_SMOOTHING_SCALE_XY = 0.55;
const OVERLAY_SMOOTHING_SCALE_Z = 0.65;
const OVERLAY_MIN_ALPHA_XY = 0.18;
const OVERLAY_MIN_ALPHA_Z = 0.16;
const SMOOTHING_CONFIGS = {
  alphabet: {
    baseAlphaXY: 0.62,
    fastAlphaXY: 0.9,
    baseAlphaZ: 0.44,
    fastAlphaZ: 0.72,
    fastMotionThreshold: 0.008,
  },
  sign: {
    baseAlphaXY: 0.72,
    fastAlphaXY: 0.93,
    baseAlphaZ: 0.52,
    fastAlphaZ: 0.78,
    fastMotionThreshold: 0.009,
  },
  phrase: {
    baseAlphaXY: 0.78,
    fastAlphaXY: 0.95,
    baseAlphaZ: 0.58,
    fastAlphaZ: 0.82,
    fastMotionThreshold: 0.01,
  },
};

function createCalibrationState(recognitionLevel = "phrase") {
  return {
    isActive: true,
    startedAt: 0,
    frames: 0,
    recognitionLevel,
    jitterSamples: [],
    coverageSamples: [],
  };
}

const HAND_CONNECTIONS = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [5, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [9, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [13, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  [0, 17],
];
const HAND_HELPER_POINTS_PER_SEGMENT = 2;

const FINGERTIP_INDICES = [4, 8, 12, 16, 20];
const FACE_TESSELLATION_CONNECTIONS = Array.isArray(
  HolisticLandmarker?.FACE_LANDMARKS_TESSELATION,
)
  ? HolisticLandmarker.FACE_LANDMARKS_TESSELATION
  : [];
const FACE_OVAL_INDICES = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
  378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162,
  21, 54, 103, 67, 109, 10,
];
const FACE_MESH_LINE_COLOR = "rgba(250, 204, 21, 0.9)";
const FACE_MESH_POINT_COLOR = "rgba(125, 211, 252, 0.98)";
const FACE_OVAL_LINE_COLOR = "rgba(56, 189, 248, 1)";

function roundMetric(value, digits = 2) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return 0;
  }

  return Number(value.toFixed(digits));
}

function getPointDistance(firstPoint, secondPoint) {
  if (!firstPoint || !secondPoint) {
    return 0;
  }

  return Math.hypot(
    Number(firstPoint.x ?? 0) - Number(secondPoint.x ?? 0),
    Number(firstPoint.y ?? 0) - Number(secondPoint.y ?? 0),
  );
}

function serializeLandmarks(landmarks = []) {
  return landmarks.map((point) => ({
    x: roundMetric(point.x, 6),
    y: roundMetric(point.y, 6),
    z: roundMetric(point.z, 6),
  }));
}

function cloneLandmarks(landmarks = []) {
  return landmarks.map((point) => ({
    x: point.x,
    y: point.y,
    z: point.z,
  }));
}

function getSmoothingConfig(recognitionLevel = "phrase") {
  return SMOOTHING_CONFIGS[recognitionLevel] ?? SMOOTHING_CONFIGS.phrase;
}

function computeAverageLandmarkMotion(previousLandmarks, currentLandmarks) {
  if (
    !Array.isArray(previousLandmarks) ||
    !previousLandmarks.length ||
    !Array.isArray(currentLandmarks) ||
    !currentLandmarks.length
  ) {
    return 0;
  }

  let totalDistance = 0;
  let countedPoints = 0;

  for (let index = 0; index < currentLandmarks.length; index += 1) {
    const previousPoint = previousLandmarks[index];
    const currentPoint = currentLandmarks[index];

    if (!previousPoint || !currentPoint) {
      continue;
    }

    totalDistance += Math.hypot(
      currentPoint.x - previousPoint.x,
      currentPoint.y - previousPoint.y,
      (currentPoint.z - previousPoint.z) * 0.35,
    );
    countedPoints += 1;
  }

  return countedPoints ? totalDistance / countedPoints : 0;
}

function smoothLandmarks(
  currentLandmarks,
  previousLandmarks,
  recognitionLevel = "phrase",
) {
  if (!Array.isArray(currentLandmarks) || !currentLandmarks.length) {
    return [];
  }

  if (!Array.isArray(previousLandmarks) || !previousLandmarks.length) {
    return cloneLandmarks(currentLandmarks);
  }

  const smoothingConfig = getSmoothingConfig(recognitionLevel);
  const averageMotion = computeAverageLandmarkMotion(
    previousLandmarks,
    currentLandmarks,
  );
  const motionRatio = Math.min(
    1,
    averageMotion / smoothingConfig.fastMotionThreshold,
  );
  const alphaXY =
    smoothingConfig.baseAlphaXY +
    (smoothingConfig.fastAlphaXY - smoothingConfig.baseAlphaXY) * motionRatio;
  const alphaZ =
    smoothingConfig.baseAlphaZ +
    (smoothingConfig.fastAlphaZ - smoothingConfig.baseAlphaZ) * motionRatio;

  return currentLandmarks.map((point, index) => {
    const previousPoint = previousLandmarks[index] ?? point;

    return {
      x: alphaXY * point.x + (1 - alphaXY) * previousPoint.x,
      y: alphaXY * point.y + (1 - alphaXY) * previousPoint.y,
      z: alphaZ * point.z + (1 - alphaZ) * previousPoint.z,
    };
  });
}

function smoothLandmarksForOverlay(
  currentLandmarks,
  previousLandmarks,
  recognitionLevel = "phrase",
) {
  if (!Array.isArray(currentLandmarks) || !currentLandmarks.length) {
    return [];
  }

  if (!Array.isArray(previousLandmarks) || !previousLandmarks.length) {
    return cloneLandmarks(currentLandmarks);
  }

  const smoothingConfig = getSmoothingConfig(recognitionLevel);
  const averageMotion = computeAverageLandmarkMotion(
    previousLandmarks,
    currentLandmarks,
  );
  const motionRatio = Math.min(
    1,
    averageMotion / smoothingConfig.fastMotionThreshold,
  );
  const alphaXYBase =
    smoothingConfig.baseAlphaXY +
    (smoothingConfig.fastAlphaXY - smoothingConfig.baseAlphaXY) * motionRatio;
  const alphaZBase =
    smoothingConfig.baseAlphaZ +
    (smoothingConfig.fastAlphaZ - smoothingConfig.baseAlphaZ) * motionRatio;
  const alphaXY = Math.max(OVERLAY_MIN_ALPHA_XY, alphaXYBase * OVERLAY_SMOOTHING_SCALE_XY);
  const alphaZ = Math.max(OVERLAY_MIN_ALPHA_Z, alphaZBase * OVERLAY_SMOOTHING_SCALE_Z);

  return currentLandmarks.map((point, index) => {
    const previousPoint = previousLandmarks[index] ?? point;

    return {
      x: alphaXY * point.x + (1 - alphaXY) * previousPoint.x,
      y: alphaXY * point.y + (1 - alphaXY) * previousPoint.y,
      z: alphaZ * point.z + (1 - alphaZ) * previousPoint.z,
    };
  });
}

function getAverage(values = []) {
  if (!values.length) {
    return 0;
  }

  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function computeFingertipJitter(previousLandmarks, currentLandmarks) {
  if (
    !Array.isArray(previousLandmarks) ||
    !previousLandmarks.length ||
    !Array.isArray(currentLandmarks) ||
    !currentLandmarks.length
  ) {
    return 0;
  }

  const distances = FINGERTIP_INDICES.map((index) => {
    const previousPoint = previousLandmarks[index];
    const currentPoint = currentLandmarks[index];

    if (!previousPoint || !currentPoint) {
      return 0;
    }

    return Math.hypot(
      currentPoint.x - previousPoint.x,
      currentPoint.y - previousPoint.y,
      currentPoint.z - previousPoint.z,
    );
  });

  return getAverage(distances);
}

function computeHandCoverage(landmarks = []) {
  if (!landmarks.length) {
    return 0;
  }

  const xValues = landmarks.map((point) => point.x);
  const yValues = landmarks.map((point) => point.y);
  const minX = Math.min(...xValues);
  const maxX = Math.max(...xValues);
  const minY = Math.min(...yValues);
  const maxY = Math.max(...yValues);

  return Math.max(0, maxX - minX) * Math.max(0, maxY - minY);
}

function getQualityThresholds(recognitionLevel = "phrase") {
  if (recognitionLevel === "alphabet") {
    return {
      maxConfirmJitter: ALPHABET_MAX_CONFIRM_JITTER,
      minHandCoverage: ALPHABET_MIN_HAND_COVERAGE,
    };
  }

  if (recognitionLevel === "sign") {
    return {
      maxConfirmJitter: SIGN_MAX_CONFIRM_JITTER,
      minHandCoverage: SIGN_MIN_HAND_COVERAGE,
    };
  }

  return {
    maxConfirmJitter: DEFAULT_MAX_CONFIRM_JITTER,
    minHandCoverage: DEFAULT_MIN_HAND_COVERAGE,
  };
}

function getLiveSequenceSettings(recognitionLevel = "phrase") {
  if (recognitionLevel === "alphabet") {
    return {
      windowSize: 6,
      minFrames: 4,
      requestIntervalMs: 80,
    };
  }

  if (recognitionLevel === "sign") {
    return {
      windowSize: 10,
      minFrames: 5,
      requestIntervalMs: 90,
    };
  }

  return {
    windowSize: 10,
    minFrames: 5,
    requestIntervalMs: 80,
  };
}

function usesBodyGuideOverlay(recognitionLevel = "phrase") {
  return recognitionLevel === "phrase" || recognitionLevel === "sign";
}

function pickPrimaryHand(
  leftHandLandmarks = [],
  rightHandLandmarks = [],
  preferredLabel = "Right",
) {
  const leftCoverage = computeHandCoverage(leftHandLandmarks);
  const rightCoverage = computeHandCoverage(rightHandLandmarks);

  if (rightCoverage > 0 || leftCoverage > 0) {
    if (leftCoverage > 0 && rightCoverage > 0) {
      const preferredCoverage =
        preferredLabel === "Left" ? leftCoverage : rightCoverage;
      const otherCoverage =
        preferredLabel === "Left" ? rightCoverage : leftCoverage;

      if (
        preferredCoverage > 0 &&
        preferredCoverage * HAND_SWITCH_HYSTERESIS_RATIO >= otherCoverage
      ) {
        return preferredLabel === "Left"
          ? {
              landmarks: leftHandLandmarks,
              handednessLabel: "Left",
              score: leftCoverage,
            }
          : {
              landmarks: rightHandLandmarks,
              handednessLabel: "Right",
              score: rightCoverage,
            };
      }
    }

    if (rightCoverage >= leftCoverage) {
      return {
        landmarks: rightHandLandmarks,
        handednessLabel: "Right",
        score: rightCoverage,
      };
    }

    return {
      landmarks: leftHandLandmarks,
      handednessLabel: "Left",
      score: leftCoverage,
    };
  }

  if (rightHandLandmarks.length) {
    return {
      landmarks: rightHandLandmarks,
      handednessLabel: "Right",
      score: 0,
    };
  }

  return {
    landmarks: leftHandLandmarks,
    handednessLabel: "Left",
    score: 0,
  };
}

function buildSequenceFrame({
  leftHandLandmarks = [],
  rightHandLandmarks = [],
  smoothedLeftHandLandmarks = [],
  smoothedRightHandLandmarks = [],
  faceLandmarks = [],
  poseLandmarks = [],
  timestampMs,
}) {
  return {
    frame_index: 0,
    timestamp_ms: Math.round(timestampMs),
    left_hand: serializeLandmarks(
      smoothedLeftHandLandmarks.length
        ? smoothedLeftHandLandmarks
        : leftHandLandmarks,
    ),
    right_hand: serializeLandmarks(
      smoothedRightHandLandmarks.length
        ? smoothedRightHandLandmarks
        : rightHandLandmarks,
    ),
    face: serializeLandmarks(faceLandmarks),
    pose: serializeLandmarks(poseLandmarks),
  };
}

function clamp01(value) {
  if (!Number.isFinite(value)) {
    return 0;
  }

  return Math.max(0, Math.min(1, value));
}

function getSerializedHandCenter(landmarks = []) {
  if (!Array.isArray(landmarks) || !landmarks.length) {
    return null;
  }

  const palmAnchorIndices = [0, 5, 9, 13, 17].filter((index) => landmarks[index]);
  const anchorPoints = palmAnchorIndices.length
    ? palmAnchorIndices.map((index) => landmarks[index])
    : landmarks;

  return {
    x: getAverage(anchorPoints.map((point) => Number(point?.x ?? 0))),
    y: getAverage(anchorPoints.map((point) => Number(point?.y ?? 0))),
  };
}

function getFingertipsCenter(landmarks = []) {
  if (!Array.isArray(landmarks) || landmarks.length < 21) {
    return null;
  }

  const fingertips = FINGERTIP_INDICES
    .map((index) => landmarks?.[index])
    .filter(Boolean);

  if (fingertips.length < 3) {
    return null;
  }

  return {
    x: getAverage(fingertips.map((point) => Number(point?.x ?? 0))),
    y: getAverage(fingertips.map((point) => Number(point?.y ?? 0))),
  };
}

function getHandWristPoint(landmarks = [], poseLandmarks = [], poseIndex) {
  if (Array.isArray(landmarks) && landmarks.length >= 1 && landmarks[0]) {
    return landmarks[0];
  }

  return hasPosePoint(poseLandmarks, poseIndex, 0.08)
    ? poseLandmarks[poseIndex]
    : null;
}

function getShoulderLine(poseLandmarks = []) {
  const shoulders = [poseLandmarks?.[11], poseLandmarks?.[12]].filter(Boolean);

  if (!shoulders.length) {
    return null;
  }

  return {
    x: getAverage(shoulders.map((point) => Number(point?.x ?? 0))),
    y: getAverage(shoulders.map((point) => Number(point?.y ?? 0))),
  };
}

function hasPosePoint(poseLandmarks = [], index, minVisibility = 0.18) {
  const point = poseLandmarks?.[index];

  if (!point) {
    return false;
  }

  const visibility = Number(point.visibility ?? point.presence ?? 1);

  if (!Number.isFinite(visibility)) {
    return true;
  }

  return visibility >= minVisibility;
}

function hasFacePoint(faceLandmarks = [], index) {
  return Boolean(faceLandmarks?.[index]);
}

function getAveragePoint(points = []) {
  const validPoints = points.filter(Boolean);

  if (!validPoints.length) {
    return null;
  }

  return {
    x: getAverage(validPoints.map((point) => Number(point?.x ?? 0))),
    y: getAverage(validPoints.map((point) => Number(point?.y ?? 0))),
  };
}

function getLandmarkBoundsCenter(landmarks = []) {
  if (!Array.isArray(landmarks) || !landmarks.length) {
    return null;
  }

  const xValues = landmarks.map((point) => Number(point?.x ?? 0));
  const yValues = landmarks.map((point) => Number(point?.y ?? 0));

  return {
    x: (Math.min(...xValues) + Math.max(...xValues)) / 2,
    y: (Math.min(...yValues) + Math.max(...yValues)) / 2,
  };
}

function getFaceFeaturePoint(faceLandmarks = [], indices = []) {
  const points = indices
    .map((index) => faceLandmarks?.[index])
    .filter(Boolean);

  return getAveragePoint(points);
}

function getPreferredPosePoint(
  poseLandmarks = [],
  primaryIndex,
  fallbackIndex,
  minVisibility = 0.1,
) {
  if (hasPosePoint(poseLandmarks, primaryIndex, minVisibility)) {
    return poseLandmarks[primaryIndex];
  }

  if (hasPosePoint(poseLandmarks, fallbackIndex, minVisibility)) {
    return poseLandmarks[fallbackIndex];
  }

  return null;
}

function buildBodyGuideMarkers({
  faceLandmarks = [],
  poseLandmarks = [],
  leftHandLandmarks = [],
  rightHandLandmarks = [],
  dominantHand = "right",
}) {
  const normalizedDominantHand = dominantHand === "left" ? "left" : "right";
  const faceCenter =
    getLandmarkBoundsCenter(faceLandmarks) ??
    getPreferredPosePoint(poseLandmarks, 0, 0, 0.08);
  const leftEyePoint = getFaceFeaturePoint(faceLandmarks, [33, 133, 159, 145]);
  const rightEyePoint = getFaceFeaturePoint(faceLandmarks, [362, 263, 386, 374]);
  const nosePoint = getFaceFeaturePoint(faceLandmarks, [1, 4, 5]);
  const lipsPoint = getFaceFeaturePoint(faceLandmarks, [13, 14, 78, 308]);
  const leftEarPoint = getPreferredPosePoint(poseLandmarks, 7, 8, 0.08);
  const rightEarPoint = getPreferredPosePoint(poseLandmarks, 8, 7, 0.08);
  const poseNosePoint = getPreferredPosePoint(poseLandmarks, 0, 0, 0.08);
  const leftShoulderPoint = hasPosePoint(poseLandmarks, 11, 0.08)
    ? poseLandmarks[11]
    : null;
  const rightShoulderPoint = hasPosePoint(poseLandmarks, 12, 0.08)
    ? poseLandmarks[12]
    : null;
  const leftElbowPoint = hasPosePoint(poseLandmarks, 13, 0.08)
    ? poseLandmarks[13]
    : null;
  const rightElbowPoint = hasPosePoint(poseLandmarks, 14, 0.08)
    ? poseLandmarks[14]
    : null;
  const leftHandPoint = getHandWristPoint(leftHandLandmarks, poseLandmarks, 15);
  const rightHandPoint = getHandWristPoint(rightHandLandmarks, poseLandmarks, 16);
  const leftFingersPoint = getFingertipsCenter(leftHandLandmarks);
  const rightFingersPoint = getFingertipsCenter(rightHandLandmarks);
  const shoulderMidPoint = getAveragePoint([leftShoulderPoint, rightShoulderPoint]);
  const headPoint =
    getFaceFeaturePoint(faceLandmarks, [10, 151, 9, 8]) ??
    (
      poseNosePoint
        ? {
            x: Number(poseNosePoint?.x ?? 0),
            y: Math.max(0, Number(poseNosePoint?.y ?? 0) - 0.12),
          }
        : null
    );
  const neckPoint =
    shoulderMidPoint ??
    (
      faceCenter
        ? {
            x: Number(faceCenter?.x ?? 0),
            y: Math.min(1, Number(faceCenter?.y ?? 0) + 0.18),
          }
        : null
    );
  const dominantShoulderPoint =
    normalizedDominantHand === "left"
      ? leftShoulderPoint ?? rightShoulderPoint
      : rightShoulderPoint ?? leftShoulderPoint;
  const dominantElbowPoint =
    normalizedDominantHand === "left"
      ? leftElbowPoint ?? rightElbowPoint
      : rightElbowPoint ?? leftElbowPoint;
  const dominantHandPoint =
    normalizedDominantHand === "left"
      ? leftHandPoint ?? rightHandPoint
      : rightHandPoint ?? leftHandPoint;
  const dominantFingersPoint =
    normalizedDominantHand === "left"
      ? leftFingersPoint ?? rightFingersPoint
      : rightFingersPoint ?? leftFingersPoint;

  return {
    faceLandmarks: Array.isArray(faceLandmarks) ? faceLandmarks : [],
    head: headPoint
      ? { point: headPoint, color: "#93c5fd" }
      : null,
    face: faceCenter
      ? { point: faceCenter, color: "#7dd3fc" }
      : null,
    leftEye: leftEyePoint
      ? { point: leftEyePoint, color: "#60a5fa" }
      : null,
    rightEye: rightEyePoint
      ? { point: rightEyePoint, color: "#60a5fa" }
      : null,
    nose: nosePoint
      ? { point: nosePoint, color: "#f59e0b" }
      : null,
    lips: lipsPoint
      ? { point: lipsPoint, color: "#fb7185" }
      : null,
    leftEar: leftEarPoint
      ? { point: leftEarPoint, color: "#c084fc" }
      : null,
    rightEar: rightEarPoint
      ? { point: rightEarPoint, color: "#c084fc" }
      : null,
    leftShoulder: leftShoulderPoint
      ? { point: leftShoulderPoint, color: "#38bdf8" }
      : null,
    rightShoulder: rightShoulderPoint
      ? { point: rightShoulderPoint, color: "#38bdf8" }
      : null,
    leftElbow: leftElbowPoint
      ? { point: leftElbowPoint, color: "#38bdf8" }
      : null,
    rightElbow: rightElbowPoint
      ? { point: rightElbowPoint, color: "#38bdf8" }
      : null,
    leftHand: leftHandPoint
      ? { point: leftHandPoint, color: "#22d3ee" }
      : null,
    rightHand: rightHandPoint
      ? { point: rightHandPoint, color: "#22d3ee" }
      : null,
    leftFingers: leftFingersPoint
      ? { point: leftFingersPoint, color: "#facc15" }
      : null,
    rightFingers: rightFingersPoint
      ? { point: rightFingersPoint, color: "#facc15" }
      : null,
    neck: neckPoint
      ? { point: neckPoint, color: "#a5b4fc" }
      : null,
    shoulder: dominantShoulderPoint
      ? { point: dominantShoulderPoint, color: "#38bdf8" }
      : null,
    elbow: dominantElbowPoint
      ? { point: dominantElbowPoint, color: "#38bdf8" }
      : null,
    hand: dominantHandPoint
      ? { point: dominantHandPoint, color: "#22d3ee" }
      : null,
    fingers: dominantFingersPoint
      ? { point: dominantFingersPoint, color: "#facc15" }
      : null,
  };
}

function drawGuideMarker(context, marker, width, height, radius = 6) {
  const point = marker?.point;

  if (!point) {
    return;
  }

  const x = point.x * width;
  const y = point.y * height;
  const color = marker?.color ?? "#f8fafc";

  context.save();
  context.beginPath();
  context.arc(x, y, radius + 7, 0, Math.PI * 2);
  context.fillStyle = color;
  context.globalAlpha = 0.28;
  context.fill();
  context.globalAlpha = 1;
  context.beginPath();
  context.arc(x, y, radius, 0, Math.PI * 2);
  context.fillStyle = color;
  context.globalAlpha = 0.98;
  context.fill();
  context.globalAlpha = 1;
  context.lineWidth = 2.8;
  context.strokeStyle = "rgba(7, 10, 16, 0.96)";
  context.stroke();
  context.beginPath();
  context.arc(x, y, Math.max(2.6, radius * 0.34), 0, Math.PI * 2);
  context.fillStyle = "rgba(255, 255, 255, 0.98)";
  context.fill();
  context.restore();
}

function drawGuideConnection(context, startMarker, endMarker, width, height) {
  const startPoint = startMarker?.point;
  const endPoint = endMarker?.point;

  if (!startPoint || !endPoint) {
    return;
  }

  context.save();
  context.beginPath();
  context.moveTo(startPoint.x * width, startPoint.y * height);
  context.lineTo(endPoint.x * width, endPoint.y * height);
  context.strokeStyle = endMarker?.color ?? startMarker?.color ?? "#38bdf8";
  context.lineWidth = 4.5;
  context.globalAlpha = 0.98;
  context.stroke();
  context.restore();
}

function _drawFaceOval(context, faceLandmarks, width, height) {
  if (!Array.isArray(faceLandmarks) || faceLandmarks.length < 40) {
    return false;
  }

  const ovalPoints = FACE_OVAL_INDICES
    .map((index) => faceLandmarks?.[index])
    .filter(Boolean);

  if (ovalPoints.length < 6) {
    return false;
  }

  context.save();
  context.beginPath();
  context.moveTo(ovalPoints[0].x * width, ovalPoints[0].y * height);

  for (let index = 1; index < ovalPoints.length; index += 1) {
    context.lineTo(ovalPoints[index].x * width, ovalPoints[index].y * height);
  }

  context.strokeStyle = FACE_OVAL_LINE_COLOR;
  context.lineWidth = 4.2;
  context.globalAlpha = 1;
  context.stroke();
  context.restore();
  return true;
}

function drawBodyGuideMarkers(context, markers, width, height) {
  if (!markers || typeof markers !== "object") {
    return;
  }

  drawGuideConnection(context, markers.head, markers.leftEar, width, height);
  drawGuideConnection(context, markers.head, markers.rightEar, width, height);
  drawGuideConnection(context, markers.face, markers.leftEye, width, height);
  drawGuideConnection(context, markers.face, markers.rightEye, width, height);
  drawGuideConnection(context, markers.face, markers.nose, width, height);
  drawGuideConnection(context, markers.leftEar, markers.leftEye, width, height);
  drawGuideConnection(context, markers.rightEar, markers.rightEye, width, height);
  drawGuideConnection(context, markers.leftEar, markers.neck, width, height);
  drawGuideConnection(context, markers.rightEar, markers.neck, width, height);
  drawGuideConnection(context, markers.leftEye, markers.rightEye, width, height);
  drawGuideConnection(context, markers.leftEye, markers.nose, width, height);
  drawGuideConnection(context, markers.rightEye, markers.nose, width, height);
  drawGuideConnection(context, markers.leftEye, markers.lips, width, height);
  drawGuideConnection(context, markers.rightEye, markers.lips, width, height);
  drawGuideConnection(context, markers.nose, markers.lips, width, height);
  drawGuideConnection(context, markers.head, markers.face, width, height);
  drawGuideConnection(context, markers.face, markers.neck, width, height);
  drawGuideConnection(
    context,
    markers.neck ?? markers.face,
    markers.leftShoulder,
    width,
    height,
  );
  drawGuideConnection(
    context,
    markers.neck ?? markers.face,
    markers.rightShoulder,
    width,
    height,
  );
  drawGuideConnection(
    context,
    markers.leftShoulder,
    markers.rightShoulder,
    width,
    height,
  );
  drawGuideConnection(
    context,
    markers.neck,
    markers.lips,
    width,
    height,
  );
  drawGuideConnection(
    context,
    markers.leftShoulder,
    markers.leftElbow,
    width,
    height,
  );
  drawGuideConnection(
    context,
    markers.leftElbow,
    markers.leftHand,
    width,
    height,
  );
  drawGuideConnection(
    context,
    markers.leftHand,
    markers.leftFingers,
    width,
    height,
  );
  drawGuideConnection(context, markers.rightShoulder, markers.rightElbow, width, height);
  drawGuideConnection(context, markers.rightElbow, markers.rightHand, width, height);
  drawGuideConnection(context, markers.rightHand, markers.rightFingers, width, height);

  if (markers.head) {
    drawGuideMarker(context, markers.head, width, height, 10);
  }

  if (markers.face) {
    drawGuideMarker(context, markers.face, width, height, 8);
  }

  if (markers.leftEye) {
    drawGuideMarker(context, markers.leftEye, width, height, 7);
  }

  if (markers.rightEye) {
    drawGuideMarker(context, markers.rightEye, width, height, 7);
  }

  if (markers.nose) {
    drawGuideMarker(context, markers.nose, width, height, 7);
  }

  if (markers.lips) {
    drawGuideMarker(context, markers.lips, width, height, 8);
  }

  if (markers.leftEar) {
    drawGuideMarker(context, markers.leftEar, width, height, 7);
  }

  if (markers.rightEar) {
    drawGuideMarker(context, markers.rightEar, width, height, 7);
  }

  if (markers.neck) {
    drawGuideMarker(context, markers.neck, width, height, 9);
  }

  if (markers.leftShoulder) {
    drawGuideMarker(context, markers.leftShoulder, width, height, 10);
  }

  if (markers.rightShoulder) {
    drawGuideMarker(context, markers.rightShoulder, width, height, 10);
  }

  if (markers.leftElbow) {
    drawGuideMarker(context, markers.leftElbow, width, height, 9);
  }

  if (markers.rightElbow) {
    drawGuideMarker(context, markers.rightElbow, width, height, 9);
  }

  if (markers.leftHand) {
    drawGuideMarker(context, markers.leftHand, width, height, 7);
  }

  if (markers.rightHand) {
    drawGuideMarker(context, markers.rightHand, width, height, 7);
  }

  if (markers.leftFingers) {
    drawGuideMarker(context, markers.leftFingers, width, height, 6);
  }

  if (markers.rightFingers) {
    drawGuideMarker(context, markers.rightFingers, width, height, 6);
  }
}

function countDirectionChanges(values = [], minDelta = 0.006) {
  if (!Array.isArray(values) || values.length < 3) {
    return 0;
  }

  let changes = 0;
  let previousDirection = 0;

  for (let index = 1; index < values.length; index += 1) {
    const delta = Number(values[index] ?? 0) - Number(values[index - 1] ?? 0);

    if (Math.abs(delta) < minDelta) {
      continue;
    }

    const currentDirection = delta > 0 ? 1 : -1;

    if (
      previousDirection &&
      currentDirection !== previousDirection
    ) {
      changes += 1;
    }

    previousDirection = currentDirection;
  }

  return changes;
}

function summarizeWaveMotion(sequenceFrames = []) {
  if (!Array.isArray(sequenceFrames) || !sequenceFrames.length) {
    return {
      dominantHand: "",
      handPresenceRatio: 0,
      oneHandRatio: 0,
      bothHandsRatio: 0,
      xRange: 0,
      yRange: 0,
      horizontalTravel: 0,
      verticalTravel: 0,
      horizontalDominance: 0,
      directionChanges: 0,
      waveScore: 0,
      waveLike: false,
    };
  }

  const handPositions = {
    left: [],
    right: [],
  };
  let oneHandFrames = 0;
  let bothHandsFrames = 0;

  sequenceFrames.forEach((frame) => {
    const leftCenter = getSerializedHandCenter(frame?.left_hand ?? []);
    const rightCenter = getSerializedHandCenter(frame?.right_hand ?? []);

    if (leftCenter) {
      handPositions.left.push(leftCenter);
    }

    if (rightCenter) {
      handPositions.right.push(rightCenter);
    }

    if (leftCenter && rightCenter) {
      bothHandsFrames += 1;
    } else if (leftCenter || rightCenter) {
      oneHandFrames += 1;
    }
  });

  const dominantHand =
    handPositions.right.length >= handPositions.left.length ? "right" : "left";
  const dominantPositions = handPositions[dominantHand];
  const handPresenceRatio = dominantPositions.length / sequenceFrames.length;
  const oneHandRatio = oneHandFrames / sequenceFrames.length;
  const bothHandsRatio = bothHandsFrames / sequenceFrames.length;

  if (dominantPositions.length < 5) {
    return {
      dominantHand,
      handPresenceRatio,
      oneHandRatio,
      bothHandsRatio,
      xRange: 0,
      yRange: 0,
      horizontalTravel: 0,
      verticalTravel: 0,
      horizontalDominance: 0,
      directionChanges: 0,
      waveScore: 0,
      waveLike: false,
    };
  }

  const xValues = dominantPositions.map((position) => position.x);
  const yValues = dominantPositions.map((position) => position.y);
  const xRange = Math.max(...xValues) - Math.min(...xValues);
  const yRange = Math.max(...yValues) - Math.min(...yValues);
  let horizontalTravel = 0;
  let verticalTravel = 0;

  for (let index = 1; index < dominantPositions.length; index += 1) {
    horizontalTravel += Math.abs(
      dominantPositions[index].x - dominantPositions[index - 1].x,
    );
    verticalTravel += Math.abs(
      dominantPositions[index].y - dominantPositions[index - 1].y,
    );
  }

  const horizontalDominance =
    horizontalTravel / Math.max(0.0001, verticalTravel + yRange * 0.35);
  const directionChanges = countDirectionChanges(xValues);
  const waveScore = clamp01(
    clamp01((xRange - 0.06) / 0.12) * 0.34 +
      clamp01(directionChanges / 2) * 0.24 +
      clamp01((oneHandRatio - 0.55) / 0.3) * 0.18 +
      clamp01((handPresenceRatio - 0.5) / 0.35) * 0.12 +
      clamp01((horizontalDominance - 1.05) / 1.2) * 0.16 -
      clamp01((bothHandsRatio - 0.18) / 0.35) * 0.16,
  );
  const waveLike =
    dominantPositions.length >= 8 &&
    xRange >= 0.085 &&
    directionChanges >= 1 &&
    oneHandRatio >= 0.62 &&
    horizontalDominance >= 1.08;

  return {
    dominantHand,
    handPresenceRatio,
    oneHandRatio,
    bothHandsRatio,
    xRange,
    yRange,
    horizontalTravel,
    verticalTravel,
    horizontalDominance,
    directionChanges,
    waveScore,
    waveLike,
  };
}

function buildGreetingGuide({
  recognitionLevel = "phrase",
  waveMotionSummary,
  leftHandLandmarks = [],
  rightHandLandmarks = [],
  faceLandmarks = [],
  poseLandmarks = [],
  faceDetected = false,
  poseDetected = false,
}) {
  if (recognitionLevel !== "phrase") {
    return null;
  }

  const handsDetectedCount =
    (leftHandLandmarks.length ? 1 : 0) + (rightHandLandmarks.length ? 1 : 0);
  const dominantHand =
    waveMotionSummary?.dominantHand === "left" ? "left" : "right";
  const dominantHandLandmarks =
    dominantHand === "left" ? leftHandLandmarks : rightHandLandmarks;
  const fallbackHandLandmarks = leftHandLandmarks.length
    ? leftHandLandmarks
    : rightHandLandmarks;
  const activeHandLandmarks =
    dominantHandLandmarks.length >= 21
      ? dominantHandLandmarks
      : fallbackHandLandmarks;
  const handCenter = getSerializedHandCenter(activeHandLandmarks);
  const indexTip = activeHandLandmarks?.[8] ?? null;
  const shoulderLine = getShoulderLine(poseLandmarks);
  const leftEyePoint = getFaceFeaturePoint(faceLandmarks, [33, 133, 159, 145]);
  const rightEyePoint = getFaceFeaturePoint(faceLandmarks, [362, 263, 386, 374]);
  const nosePoint = getFaceFeaturePoint(faceLandmarks, [1, 4, 5]);
  const faceCenterPoint =
    getLandmarkBoundsCenter(faceLandmarks) ??
    getPreferredPosePoint(poseLandmarks, 0, 0, 0.08);
  const faceScale = Math.max(
    getPointDistance(leftEyePoint, rightEyePoint),
    getPointDistance(faceCenterPoint, nosePoint),
    0.08,
  );
  const directionSign = dominantHand === "left" ? -1 : 1;
  const sideEyePoint = dominantHand === "left" ? leftEyePoint : rightEyePoint;
  const templePoint = sideEyePoint
    ? {
        x: Number(sideEyePoint.x ?? 0) + directionSign * faceScale * 0.32,
        y: Number(sideEyePoint.y ?? 0) - faceScale * 0.08,
      }
    : null;
  const targetPoint = indexTip ?? handCenter;
  const templeTouchDone = Boolean(
    targetPoint &&
      templePoint &&
      getPointDistance(targetPoint, templePoint) / faceScale <= 1.05,
  );
  const outwardFromHead = targetPoint
    ? (
      (Number(targetPoint.x ?? 0) - Number(faceCenterPoint?.x ?? 0.5)) *
      directionSign
    ) / faceScale
    : 0;
  const forwardFromHeadDone =
    outwardFromHead >= 0.24 ||
    (
      Number(waveMotionSummary?.xRange ?? 0) >= 0.08 &&
      Number(waveMotionSummary?.horizontalDominance ?? 0) >= 1.06
    );
  const headVisible =
    hasFacePoint(faceLandmarks, 10) ||
    hasFacePoint(faceLandmarks, 151) ||
    hasPosePoint(poseLandmarks, 0, 0.08);
  const faceVisible =
    Boolean(faceDetected) ||
    (Array.isArray(faceLandmarks) && faceLandmarks.length >= 8);
  const eyesVisible =
    hasFacePoint(faceLandmarks, 33) ||
    hasFacePoint(faceLandmarks, 263) ||
    hasFacePoint(faceLandmarks, 133) ||
    hasFacePoint(faceLandmarks, 362);
  const noseVisible =
    hasFacePoint(faceLandmarks, 1) ||
    hasFacePoint(faceLandmarks, 4) ||
    hasFacePoint(faceLandmarks, 5);
  const lipsVisible =
    hasFacePoint(faceLandmarks, 13) ||
    hasFacePoint(faceLandmarks, 14) ||
    hasFacePoint(faceLandmarks, 78) ||
    hasFacePoint(faceLandmarks, 308);
  const earsVisible =
    hasPosePoint(poseLandmarks, 7, 0.08) ||
    hasPosePoint(poseLandmarks, 8, 0.08);
  const neckVisible =
    (
      hasPosePoint(poseLandmarks, 11, 0.08) ||
      hasPosePoint(poseLandmarks, 12, 0.08)
    ) &&
    (
      hasPosePoint(poseLandmarks, 0, 0.08) ||
      faceVisible
    );
  const shoulderVisible =
    Boolean(poseDetected) &&
    (
      hasPosePoint(poseLandmarks, 11, 0.1) ||
      hasPosePoint(poseLandmarks, 12, 0.1)
    );
  const elbowVisible =
    Boolean(poseDetected) &&
    (
      hasPosePoint(poseLandmarks, 13, 0.08) ||
      hasPosePoint(poseLandmarks, 14, 0.08)
    );
  const handVisible = activeHandLandmarks.length >= 21;
  const fingersVisible =
    activeHandLandmarks.length >= 21 &&
    FINGERTIP_INDICES
      .map((index) => activeHandLandmarks?.[index])
      .filter(Boolean).length >= 4;
  const oneHandDone =
    handsDetectedCount === 1 ||
    Number(waveMotionSummary?.oneHandRatio ?? 0) >= 0.7;
  const waveDone = templeTouchDone;
  const shoulderDone = Boolean(
    forwardFromHeadDone &&
      shoulderLine &&
      handCenter &&
      handCenter.y <= shoulderLine.y + 0.22,
  );
  const readyCount = [oneHandDone, waveDone, shoulderDone].filter(Boolean).length;

  return {
    mode: "privet",
    oneHandDone,
    waveDone,
    shoulderDone,
    headVisible,
    faceVisible,
    eyesVisible,
    noseVisible,
    lipsVisible,
    earsVisible,
    neckVisible,
    shoulderVisible,
    elbowVisible,
    handVisible,
    fingersVisible,
    poseDetected: Boolean(poseDetected && shoulderLine),
    handsDetectedCount,
    waveScore: roundMetric(Number(waveMotionSummary?.waveScore ?? 0), 3),
    readyCount,
  };
}

function summarizePredictionWindow(window) {
  if (!window.length) {
    return {
      gesture: "",
      count: 0,
      ratio: 0,
      weightedRatio: 0,
      dominantWeight: 0,
      averageConfidence: 0,
      averageMargin: 0,
    };
  }

  const counts = window.reduce((accumulator, entry) => {
    const gesture = entry?.gesture ?? "";
    if (!gesture) {
      return accumulator;
    }

    const confidence = Number(entry?.confidence ?? 0);
    const margin = Number(entry?.margin ?? 0);
    const confidenceScore = Math.max(0, Math.min(1, confidence));
    const marginScore = Math.max(0, Math.min(1, margin));
    const weight = 0.18 + confidenceScore * 0.6 + marginScore * 2.2;
    const gestureStats = accumulator[gesture] ?? {
      count: 0,
      weight: 0,
    };

    gestureStats.count += 1;
    gestureStats.weight += weight;
    accumulator[gesture] = gestureStats;
    return accumulator;
  }, {});

  const windowConfidence = window
    .map((entry) => Number(entry?.confidence ?? 0))
    .filter((value) => Number.isFinite(value));
  const windowMargin = window
    .map((entry) => Number(entry?.margin ?? 0))
    .filter((value) => Number.isFinite(value));

  const averageConfidence = windowConfidence.length
    ? getAverage(windowConfidence)
    : 0;
  const averageMargin = windowMargin.length ? getAverage(windowMargin) : 0;

  const totalWeight = Object.values(counts).reduce(
    (sum, stats) => sum + stats.weight,
    0,
  );
  const [dominantGesture, dominantStats] = Object.entries(counts).sort(
    (left, right) =>
      right[1].weight - left[1].weight || right[1].count - left[1].count,
  )[0];
  const dominantCount = dominantStats?.count ?? 0;
  const dominantWeight = dominantStats?.weight ?? 0;

  return {
    gesture: dominantGesture,
    count: dominantCount,
    ratio: dominantCount / window.length,
    weightedRatio: totalWeight > 0 ? dominantWeight / totalWeight : 0,
    dominantWeight,
    averageConfidence,
    averageMargin,
  };
}

function buildDebugPayload(prediction, summary, windowSize, quality) {
  const features = prediction.features;

  if (!features) {
    return null;
  }

  return {
    openCount: features.openCount,
    relaxedOpenCount: features.relaxedOpenCount,
    longFingerOpenCount: features.longFingerOpenCount,
    longFingerRelaxedCount: features.longFingerRelaxedCount,
    palmSpread: roundMetric(features.palmSpread),
    fingerExtensionAverage: roundMetric(features.fingerExtensionAverage),
    strictOpenFlags: features.strictOpenFlags,
    relaxedOpenFlags: features.relaxedOpenFlags,
    openScores: Object.fromEntries(
      Object.entries(features.openScores).map(([finger, score]) => [
        finger,
        Math.round(score * 100),
      ]),
    ),
    dominantGesture: summary.gesture,
    dominantRatio: Math.round(summary.ratio * 100),
    dominantWeightedRatio: Math.round(summary.weightedRatio * 100),
    dominantCount: summary.count,
    windowSize,
    handDetected: quality.handDetected,
    handedness: quality.handednessLabel,
    handScore: Math.round(quality.handednessScore * 100),
    handCoverage: roundMetric(quality.handCoverage * 100, 1),
    fingertipJitter: roundMetric(quality.fingertipJitter, 4),
    jitterAverage: roundMetric(quality.jitterAverage, 4),
    trackingStable: quality.trackingStable,
    handVisibleEnough: quality.handVisibleEnough,
    maxConfirmJitter: roundMetric(quality.maxConfirmJitter, 4),
    minHandCoverage: roundMetric(quality.minHandCoverage * 100, 2),
    freezeFramesRemaining: quality.freezeFramesRemaining,
    faceDetected: quality.faceDetected,
    poseDetected: quality.poseDetected,
  };
}

function resolveHandednessLabel(handednessEntry, fallbackLabel = "Right") {
  const label =
    handednessEntry?.[0]?.categoryName ??
    handednessEntry?.[0]?.displayName ??
    fallbackLabel;

  if (typeof label !== "string") {
    return fallbackLabel;
  }

  const normalized = label.trim().toLowerCase();

  if (normalized === "left") {
    return "Left";
  }

  if (normalized === "right") {
    return "Right";
  }

  return fallbackLabel;
}

function normalizeVisionResults(results, useHolisticLandmarker) {
  if (useHolisticLandmarker) {
    const normalizeLandmarkList = (value) => {
      if (!Array.isArray(value) || !value.length) {
        return [];
      }

      if (
        Array.isArray(value[0]) &&
        Array.isArray(value[0][0])
      ) {
        return value[0] ?? [];
      }

      if (Array.isArray(value[0]) && value[0]?.[0] && typeof value[0][0] === "object") {
        return value[0] ?? [];
      }

      if (value[0] && typeof value[0] === "object" && "x" in value[0] && "y" in value[0]) {
        return value;
      }

      return [];
    };

    const leftHandLandmarks = normalizeLandmarkList(results?.leftHandLandmarks);
    const rightHandLandmarks = normalizeLandmarkList(results?.rightHandLandmarks);
    const faceLandmarks = normalizeLandmarkList(results?.faceLandmarks);
    const poseLandmarks = normalizeLandmarkList(results?.poseLandmarks);

    return {
      leftHandLandmarks,
      rightHandLandmarks,
      faceLandmarks,
      poseLandmarks,
      faceDetected: faceLandmarks.length > 0,
      poseDetected: poseLandmarks.length > 0,
    };
  }

  const hands = results?.landmarks ?? [];
  const handedness = results?.handedness ?? results?.handednesses ?? [];
  let leftHandLandmarks = [];
  let rightHandLandmarks = [];

  hands.forEach((landmarks, index) => {
    const label = resolveHandednessLabel(
      handedness[index],
      index === 0 ? "Right" : "Left",
    );

    if (label === "Left" && !leftHandLandmarks.length) {
      leftHandLandmarks = landmarks;
      return;
    }

    if (label === "Right" && !rightHandLandmarks.length) {
      rightHandLandmarks = landmarks;
      return;
    }

    if (!rightHandLandmarks.length) {
      rightHandLandmarks = landmarks;
      return;
    }

    if (!leftHandLandmarks.length) {
      leftHandLandmarks = landmarks;
    }
  });

  return {
    leftHandLandmarks,
    rightHandLandmarks,
    faceLandmarks: [],
    poseLandmarks: [],
    faceDetected: true,
    poseDetected: true,
  };
}

function drawHand(context, landmarks, width, height, color = "#22d3ee") {
  context.strokeStyle = color;
  context.lineWidth = 3;

  for (const [start, end] of HAND_CONNECTIONS) {
    const pointA = landmarks[start];
    const pointB = landmarks[end];

    if (!pointA || !pointB) {
      continue;
    }

    context.beginPath();
    context.moveTo(pointA.x * width, pointA.y * height);
    context.lineTo(pointB.x * width, pointB.y * height);
    context.stroke();

    for (let step = 1; step <= HAND_HELPER_POINTS_PER_SEGMENT; step += 1) {
      const interpolationFactor = step / (HAND_HELPER_POINTS_PER_SEGMENT + 1);
      const helperX =
        pointA.x + (pointB.x - pointA.x) * interpolationFactor;
      const helperY =
        pointA.y + (pointB.y - pointA.y) * interpolationFactor;

      context.beginPath();
      context.arc(helperX * width, helperY * height, 3, 0, Math.PI * 2);
      context.fillStyle = color;
      context.globalAlpha = 0.7;
      context.fill();
      context.globalAlpha = 1;
    }
  }

  for (const point of landmarks) {
    context.beginPath();
    context.arc(point.x * width, point.y * height, 6, 0, Math.PI * 2);
    context.fillStyle = color;
    context.fill();
  }
}

export default function CameraView({
  onGestureDetected,
  onLiveGestureChange,
  onPredictionChange,
  onSequencePredict,
  onStreamFrame,
  enableLiveSequencePredict = true,
  recognitionLevel = "phrase",
  facingMode = "user",
  mirrorPreview = false,
  showLandmarkOverlay = true,
  showStatusOverlay = true,
  startSignal = 0,
}) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const canvasContextRef = useRef(null);
  const animationRef = useRef(null);
  const handLandmarkerRef = useRef(null);
  const holisticLandmarkerRef = useRef(null);
  const lastVideoTimeRef = useRef(-1);
  const lastConfirmedPhraseRef = useRef({
    value: "",
    timestamp: 0,
  });
  const missingFramesRef = useRef(0);
  const predictionWindowRef = useRef([]);
  const liveSequenceRef = useRef([]);
  const motionSequenceRef = useRef([]);
  const jitterWindowRef = useRef([]);
  const lowQualityFramesRef = useRef(0);
  const motionLowQualityFramesRef = useRef(0);
  const smoothedLeftHandRef = useRef(null);
  const smoothedRightHandRef = useRef(null);
  const overlayLeftHandRef = useRef(null);
  const overlayRightHandRef = useRef(null);
  const overlayFaceRef = useRef(null);
  const overlayPoseRef = useRef(null);
  const lastHandednessLabelRef = useRef("Right");
  const lastQualityRef = useRef(null);
  const calibrationRef = useRef(createCalibrationState(recognitionLevel));
  const adaptiveThresholdsRef = useRef(null);
  const isSequenceRequestInFlightRef = useRef(false);
  const lastSequenceRequestAtRef = useRef(0);
  const streamCanvasRef = useRef(null);
  const lastStreamFrameAtRef = useRef(0);
  const enableLiveSequencePredictRef = useRef(enableLiveSequencePredict);
  const onGestureDetectedRef = useRef(onGestureDetected);
  const onLiveGestureChangeRef = useRef(onLiveGestureChange);
  const onPredictionChangeRef = useRef(onPredictionChange);
  const onSequencePredictRef = useRef(onSequencePredict);
  const onStreamFrameRef = useRef(onStreamFrame);
  const recognitionLevelRef = useRef(recognitionLevel);
  const showLandmarkOverlayRef = useRef(showLandmarkOverlay);
  const showStatusOverlayRef = useRef(showStatusOverlay);
  const lastStatusTextRef = useRef("");
  const lastStatusUpdateAtRef = useRef(0);
  const lastLiveGestureRef = useRef("");
  const lastPredictionEmitAtRef = useRef(0);
  const lastPredictionSignatureRef = useRef("");

  const [error, setError] = useState("");
  const [status, setStatus] = useState(
    'Нажмите кнопку "Запустить камеру"',
  );

  useEffect(() => {
    onGestureDetectedRef.current = onGestureDetected;
  }, [onGestureDetected]);

  useEffect(() => {
    onLiveGestureChangeRef.current = onLiveGestureChange;
  }, [onLiveGestureChange]);

  useEffect(() => {
    onPredictionChangeRef.current = onPredictionChange;
  }, [onPredictionChange]);

  useEffect(() => {
    onSequencePredictRef.current = onSequencePredict;
  }, [onSequencePredict]);

  useEffect(() => {
    onStreamFrameRef.current = onStreamFrame;
  }, [onStreamFrame]);

  useEffect(() => {
    recognitionLevelRef.current = recognitionLevel;
    liveSequenceRef.current = [];
    motionSequenceRef.current = [];
    lowQualityFramesRef.current = 0;
    motionLowQualityFramesRef.current = 0;
    calibrationRef.current = createCalibrationState(recognitionLevel);
    adaptiveThresholdsRef.current = null;
    isSequenceRequestInFlightRef.current = false;
    lastSequenceRequestAtRef.current = 0;
    lastStreamFrameAtRef.current = 0;
    smoothedLeftHandRef.current = null;
    smoothedRightHandRef.current = null;
  }, [recognitionLevel]);

  useEffect(() => {
    showLandmarkOverlayRef.current = showLandmarkOverlay;
  }, [showLandmarkOverlay]);

  useEffect(() => {
    showStatusOverlayRef.current = showStatusOverlay;
  }, [showStatusOverlay]);

  useEffect(() => {
    enableLiveSequencePredictRef.current = enableLiveSequencePredict;

    if (!enableLiveSequencePredict) {
      liveSequenceRef.current = [];
      motionSequenceRef.current = [];
      isSequenceRequestInFlightRef.current = false;
    }
  }, [enableLiveSequencePredict]);

  useEffect(() => {
    let stream;
    let isMounted = true;

    function updateStatus(nextStatus, { force = false } = {}) {
      const normalizedStatus = String(nextStatus ?? "").trim();

      if (!normalizedStatus) {
        return;
      }

      const now = performance.now();

      if (
        !force &&
        normalizedStatus !== lastStatusTextRef.current &&
        now - lastStatusUpdateAtRef.current < STATUS_UPDATE_INTERVAL_MS
      ) {
        return;
      }

      if (
        !force &&
        normalizedStatus === lastStatusTextRef.current &&
        now - lastStatusUpdateAtRef.current < STATUS_UPDATE_INTERVAL_MS
      ) {
        return;
      }

      lastStatusTextRef.current = normalizedStatus;
      lastStatusUpdateAtRef.current = now;
      setStatus((previousStatus) =>
        previousStatus === normalizedStatus ? previousStatus : normalizedStatus,
      );
    }

    function emitLiveGesture(nextGesture, { force = false } = {}) {
      if (!onLiveGestureChangeRef.current) {
        return;
      }

      if (!force && nextGesture === lastLiveGestureRef.current) {
        return;
      }

      lastLiveGestureRef.current = nextGesture;
      onLiveGestureChangeRef.current(nextGesture);
    }

    function emitPrediction(payload, { force = false } = {}) {
      if (!onPredictionChangeRef.current) {
        return;
      }

      const signature = [
        payload.gesture,
        payload.confidence,
        payload.dominantGesture,
        payload.dominantRatio,
        payload.windowSize,
        payload.stableCount,
        payload.isConfidentEnough ? 1 : 0,
        payload.isStableMajority ? 1 : 0,
        payload.trackingStable ? 1 : 0,
        payload.handVisibleEnough ? 1 : 0,
      ].join("|");
      const now = performance.now();

      if (
        !force &&
        signature === lastPredictionSignatureRef.current &&
        now - lastPredictionEmitAtRef.current < PREDICTION_EMIT_INTERVAL_MS
      ) {
        return;
      }

      lastPredictionSignatureRef.current = signature;
      lastPredictionEmitAtRef.current = now;
      onPredictionChangeRef.current(payload);
    }

    function emitPredictionReset(guide = null) {
      emitLiveGesture("Жест не распознан", { force: true });
      emitPrediction(
        {
          gesture: "Жест не распознан",
          confidence: 0,
          dominantGesture: "",
          dominantRatio: 0,
          windowSize: 0,
          stableCount: 0,
          isConfidentEnough: false,
          isStableMajority: false,
          trackingStable: false,
          handVisibleEnough: false,
          guide,
          debug: null,
        },
        { force: true },
      );
    }

    function resetGestureState({ guide = null } = {}) {
      lastVideoTimeRef.current = -1;
      predictionWindowRef.current = [];
      liveSequenceRef.current = [];
      motionSequenceRef.current = [];
      jitterWindowRef.current = [];
      lowQualityFramesRef.current = 0;
      motionLowQualityFramesRef.current = 0;
      smoothedLeftHandRef.current = null;
      smoothedRightHandRef.current = null;
      lastQualityRef.current = null;
      lastConfirmedPhraseRef.current = {
        value: "",
        timestamp: 0,
      };
      lastLiveGestureRef.current = "";
      lastPredictionSignatureRef.current = "";
      lastPredictionEmitAtRef.current = 0;
      calibrationRef.current = createCalibrationState(
        recognitionLevelRef.current,
      );
      calibrationRef.current.isActive = false;
      adaptiveThresholdsRef.current = null;
      emitPredictionReset(guide);
    }

    function stopSession() {
      const video = videoRef.current;

      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }

      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
        stream = null;
      }

      if (video) {
        video.pause();
        video.srcObject = null;
      }

      if (holisticLandmarkerRef.current) {
        holisticLandmarkerRef.current.close();
        holisticLandmarkerRef.current = null;
      }

      if (handLandmarkerRef.current) {
        handLandmarkerRef.current.close();
        handLandmarkerRef.current = null;
      }

      canvasContextRef.current = null;

      resetGestureState();
    }

    if (!startSignal) {
      resetGestureState();

      return () => {
        isMounted = false;
        stopSession();
      };
    }

    async function setupCamera() {
      updateStatus("Запуск камеры...", { force: true });

      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("Camera API is unavailable");
      }

      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 960 },
          aspectRatio: { ideal: 4 / 3 },
          facingMode,
        },
        audio: false,
      });

      const video = videoRef.current;

      if (!video || !isMounted) {
        return;
      }

      video.srcObject = stream;

      await new Promise((resolve) => {
        video.onloadedmetadata = () => resolve();
      });

      if (!isMounted) {
        return;
      }

      await video.play();
    }

    async function setupVisionLandmarkers() {
      updateStatus("Загрузка моделей...", { force: true });

      const vision = await FilesetResolver.forVisionTasks(
        `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VISION_VERSION}/wasm`,
      );

      if (!isMounted) {
        return;
      }

      handLandmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.6,
        minHandPresenceConfidence: MIN_HAND_PRESENCE_CONFIDENCE,
        minTrackingConfidence: 0.6,
      });

      holisticLandmarkerRef.current = await HolisticLandmarker.createFromOptions(
        vision,
        {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task",
          },
          runningMode: "VIDEO",
          minFaceDetectionConfidence: 0.5,
          minFacePresenceConfidence: 0.5,
          minPoseDetectionConfidence: 0.5,
          minPosePresenceConfidence: 0.5,
          minHandLandmarksConfidence: MIN_HAND_PRESENCE_CONFIDENCE,
        },
      );
    }

    function beginCalibrationSession() {
      calibrationRef.current = createCalibrationState(
        recognitionLevelRef.current,
      );
      adaptiveThresholdsRef.current = null;
      updateStatus("Подготовка распознавания жестов...", { force: true });
    }

    function maybeRunLiveSequencePrediction(sequenceFrames, metadata = null) {
      const liveSequenceSettings = getLiveSequenceSettings(
        recognitionLevelRef.current,
      );

      if (
        !enableLiveSequencePredictRef.current ||
        !onSequencePredictRef.current ||
        sequenceFrames.length < liveSequenceSettings.minFrames
      ) {
        return;
      }

      const now = Date.now();

      if (
        isSequenceRequestInFlightRef.current ||
        now - lastSequenceRequestAtRef.current <
          liveSequenceSettings.requestIntervalMs
      ) {
        return;
      }

      isSequenceRequestInFlightRef.current = true;
      lastSequenceRequestAtRef.current = now;

      Promise.resolve(
        onSequencePredictRef.current({
          frames: sequenceFrames,
          metadata,
        }),
      ).finally(() => {
        isSequenceRequestInFlightRef.current = false;
      });
    }

    function emitStreamFrame(video) {
      if (
        recognitionLevelRef.current !== "sign" ||
        !onStreamFrameRef.current ||
        document.hidden
      ) {
        return;
      }

      const now = performance.now();
      if (now - lastStreamFrameAtRef.current < STREAM_FRAME_INTERVAL_MS) {
        return;
      }

      const streamCanvas =
        streamCanvasRef.current ?? document.createElement("canvas");
      const streamContext =
        streamCanvas.getContext("2d", { alpha: false }) ??
        streamCanvas.getContext("2d");

      if (!streamContext) {
        return;
      }

      streamCanvasRef.current = streamCanvas;
      streamCanvas.width = STREAM_FRAME_WIDTH;
      streamCanvas.height = STREAM_FRAME_HEIGHT;
      streamContext.drawImage(video, 0, 0, streamCanvas.width, streamCanvas.height);

      const dataUrl = streamCanvas.toDataURL("image/jpeg", STREAM_FRAME_JPEG_QUALITY);
      const imageBase64 = dataUrl.includes(",")
        ? dataUrl.split(",", 2)[1]
        : "";

      if (!imageBase64) {
        return;
      }

      lastStreamFrameAtRef.current = now;
      onStreamFrameRef.current({
        ts: Date.now(),
        imageBase64,
      });
    }

    function renderFrozenFrame(context, width, height) {
      const frozenLeftHand = smoothedLeftHandRef.current ?? [];
      const frozenRightHand = smoothedRightHandRef.current ?? [];

      if (!frozenLeftHand.length && !frozenRightHand.length) {
        return false;
      }

      if (!showLandmarkOverlayRef.current) {
        return true;
      }

      if (frozenLeftHand.length) {
        drawHand(context, frozenLeftHand, width, height, "#22d3ee");
      }

      if (frozenRightHand.length) {
        drawHand(context, frozenRightHand, width, height, "#38bdf8");
      }

      const quality = lastQualityRef.current ?? {
        handDetected: false,
        handednessLabel: lastHandednessLabelRef.current,
        handednessScore: 0,
        handCoverage: 0,
        fingertipJitter: 0,
        jitterAverage: 0,
        trackingStable: false,
        handVisibleEnough: false,
        freezeFramesRemaining: 0,
        faceDetected: false,
        poseDetected: false,
      };

      lastQualityRef.current = {
        ...quality,
        freezeFramesRemaining: Math.max(
          0,
          NO_HAND_GRACE_FRAMES - missingFramesRef.current,
        ),
      };

      updateStatus(
        `Заморозка последнего кадра... осталось ${lastQualityRef.current.freezeFramesRemaining} кадров.`,
      );

      return true;
    }

    function drawResults(results, timestampMs, useHolisticLandmarker) {
      const video = videoRef.current;
      const canvas = canvasRef.current;

      if (!video || !canvas) {
        return;
      }

      const context =
        canvasContextRef.current ??
        canvas.getContext("2d", { alpha: true, desynchronized: true }) ??
        canvas.getContext("2d");
      const width = video.videoWidth || 720;
      const height = video.videoHeight || 960;

      if (!context) {
        return;
      }

      canvasContextRef.current = context;

      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }

      context.clearRect(0, 0, width, height);

      const {
        leftHandLandmarks,
        rightHandLandmarks,
        faceLandmarks,
        poseLandmarks,
        faceDetected,
        poseDetected,
      } = normalizeVisionResults(results, useHolisticLandmarker);

      const overlayFaceLandmarks = faceLandmarks.length
        ? smoothLandmarksForOverlay(
            faceLandmarks,
            overlayFaceRef.current,
            recognitionLevelRef.current,
          )
        : [];
      const overlayPoseLandmarks = poseLandmarks.length
        ? smoothLandmarksForOverlay(
            poseLandmarks,
            overlayPoseRef.current,
            recognitionLevelRef.current,
          )
        : [];

      overlayFaceRef.current = overlayFaceLandmarks.length ? overlayFaceLandmarks : null;
      overlayPoseRef.current = overlayPoseLandmarks.length ? overlayPoseLandmarks : null;
      const hasHands = leftHandLandmarks.length || rightHandLandmarks.length;

      if (!hasHands) {
        missingFramesRef.current += 1;
        const shouldRenderBodyGuide = usesBodyGuideOverlay(
          recognitionLevelRef.current,
        );
        const noHandGuideMarkers = shouldRenderBodyGuide
          ? buildBodyGuideMarkers({
              faceLandmarks: overlayFaceLandmarks,
              poseLandmarks: overlayPoseLandmarks,
              leftHandLandmarks: [],
              rightHandLandmarks: [],
              dominantHand:
                lastHandednessLabelRef.current === "Left" ? "left" : "right",
            })
          : null;

        if (
          missingFramesRef.current < NO_HAND_GRACE_FRAMES &&
          renderFrozenFrame(context, width, height)
        ) {
          if (showLandmarkOverlayRef.current && noHandGuideMarkers) {
            drawBodyGuideMarkers(context, noHandGuideMarkers, width, height);
          }
          return;
        }

        if (showLandmarkOverlayRef.current && noHandGuideMarkers) {
          drawBodyGuideMarkers(context, noHandGuideMarkers, width, height);
        }

        if (calibrationRef.current.isActive) {
          updateStatus("Калибровка активна — покажите руки в кадре");
          emitPredictionReset();
          return;
        }

        const greetingGuide = buildGreetingGuide({
          recognitionLevel: recognitionLevelRef.current,
          waveMotionSummary: summarizeWaveMotion([]),
          leftHandLandmarks: [],
          rightHandLandmarks: [],
          faceLandmarks: overlayFaceLandmarks,
          poseLandmarks: overlayPoseLandmarks,
          faceDetected,
          poseDetected,
        });
        updateStatus("Руки не обнаружены");
        resetGestureState({ guide: greetingGuide });
        return;
      }

      missingFramesRef.current = 0;

      const primaryHand = pickPrimaryHand(
        leftHandLandmarks,
        rightHandLandmarks,
        lastHandednessLabelRef.current,
      );
      const previousSmoothedLeftHand = smoothedLeftHandRef.current;
      const previousSmoothedRightHand = smoothedRightHandRef.current;
      const smoothedLeftHand = leftHandLandmarks.length
        ? smoothLandmarks(
            leftHandLandmarks,
            previousSmoothedLeftHand,
            recognitionLevelRef.current,
          )
        : [];
      const smoothedRightHand = rightHandLandmarks.length
        ? smoothLandmarks(
            rightHandLandmarks,
            previousSmoothedRightHand,
            recognitionLevelRef.current,
          )
        : [];

      smoothedLeftHandRef.current = smoothedLeftHand.length
        ? smoothedLeftHand
        : null;
      smoothedRightHandRef.current = smoothedRightHand.length
        ? smoothedRightHand
        : null;

      const primarySmoothedHand = pickPrimaryHand(
        smoothedLeftHand,
        smoothedRightHand,
        primaryHand.handednessLabel,
      );
      const handednessLabel = primarySmoothedHand.handednessLabel;
      const handednessScore = primarySmoothedHand.score;
      const smoothedLandmarks = primarySmoothedHand.landmarks;
      const previousSmoothedLandmarks =
        handednessLabel === "Left"
          ? previousSmoothedLeftHand
          : previousSmoothedRightHand;
      const fingertipJitter = computeFingertipJitter(
        previousSmoothedLandmarks,
        smoothedLandmarks,
      );

      jitterWindowRef.current = [
        ...jitterWindowRef.current,
        fingertipJitter,
      ].slice(-JITTER_WINDOW_SIZE);

      const jitterAverage = getAverage(jitterWindowRef.current);
      const handCoverage = computeHandCoverage(smoothedLandmarks);
      const baseThresholds = getQualityThresholds(recognitionLevelRef.current);
      const adaptiveThresholds = adaptiveThresholdsRef.current;
      const maxConfirmJitter =
        adaptiveThresholds?.recognitionLevel === recognitionLevelRef.current
          ? adaptiveThresholds.maxConfirmJitter
          : baseThresholds.maxConfirmJitter;
      const minHandCoverage =
        adaptiveThresholds?.recognitionLevel === recognitionLevelRef.current
          ? adaptiveThresholds.minHandCoverage
          : baseThresholds.minHandCoverage;
      const liveSequenceSettings = getLiveSequenceSettings(
        recognitionLevelRef.current,
      );
      const trackingStable = jitterAverage <= maxConfirmJitter;
      const handVisibleEnough =
        handCoverage >= minHandCoverage ||
        (trackingStable && handCoverage >= Math.max(minHandCoverage * 0.68, 0.0055));
      const sequenceFrameQualityOk = trackingStable && handVisibleEnough;
      const motionFrameQualityOk =
        handCoverage >= Math.max(minHandCoverage * 0.68, 0.0055);

      lastHandednessLabelRef.current = handednessLabel;
      lastQualityRef.current = {
        handDetected: true,
        handednessLabel,
        handednessScore,
        handCoverage,
        fingertipJitter,
        jitterAverage,
        trackingStable,
        handVisibleEnough,
        maxConfirmJitter,
        minHandCoverage,
        freezeFramesRemaining: 0,
        faceDetected,
        poseDetected,
      };

      const calibration = calibrationRef.current;

      if (calibration.isActive) {
        if (!calibration.startedAt) {
          calibration.startedAt = timestampMs;
        }

        calibration.frames += 1;
        calibration.jitterSamples = [
          ...calibration.jitterSamples,
          jitterAverage,
        ].slice(-120);
        calibration.coverageSamples = [
          ...calibration.coverageSamples,
          handCoverage,
        ].slice(-120);

        const elapsedMs = timestampMs - calibration.startedAt;
        const remainingMs = Math.max(0, CALIBRATION_DURATION_MS - elapsedMs);
        const readyToFinish =
          elapsedMs >= CALIBRATION_DURATION_MS &&
          calibration.frames >= CALIBRATION_MIN_FRAMES;

        if (showLandmarkOverlayRef.current && smoothedLeftHand.length) {
          drawHand(context, smoothedLeftHand, width, height, "#22d3ee");
        }

        if (showLandmarkOverlayRef.current && smoothedRightHand.length) {
          drawHand(context, smoothedRightHand, width, height, "#38bdf8");
        }

        if (
          showLandmarkOverlayRef.current &&
          usesBodyGuideOverlay(recognitionLevelRef.current)
        ) {
          drawBodyGuideMarkers(
            context,
            buildBodyGuideMarkers({
              faceLandmarks: overlayFaceLandmarks,
              poseLandmarks: overlayPoseLandmarks,
              leftHandLandmarks: smoothedLeftHand,
              rightHandLandmarks: smoothedRightHand,
              dominantHand: handednessLabel === "Left" ? "left" : "right",
            }),
            width,
            height,
          );
        }

        if (!readyToFinish) {
          updateStatus(
            `Калибровка выполняется... ${Math.ceil(remainingMs / 1000)}с`,
          );
          emitPredictionReset();
          return;
        }

        const averageCalibrationJitter = getAverage(calibration.jitterSamples);
        const averageCalibrationCoverage = getAverage(calibration.coverageSamples);

        adaptiveThresholdsRef.current = {
          recognitionLevel: recognitionLevelRef.current,
          maxConfirmJitter: Math.min(
            baseThresholds.maxConfirmJitter * 1.35,
            Math.max(
              baseThresholds.maxConfirmJitter * 0.8,
              averageCalibrationJitter * 1.35,
            ),
          ),
          minHandCoverage: Math.max(
            baseThresholds.minHandCoverage * 0.72,
            Math.min(
              baseThresholds.minHandCoverage * 1.22,
              averageCalibrationCoverage * 0.84,
            ),
          ),
        };
        calibration.isActive = false;
        updateStatus("Калибровка завершена. Можно продолжать.");
        emitPredictionReset();
        return;
      }

      const prediction = predictGesture(smoothedLandmarks, handednessLabel);
      const detectedGesture = prediction.gesture;
      const confidencePercent = Math.round(prediction.confidence * 100);
      const predictionMargin = Number(prediction.margin ?? 0);

      predictionWindowRef.current = [
        ...predictionWindowRef.current,
        {
          gesture: detectedGesture,
          confidence: prediction.confidence,
          margin: prediction.margin ?? 0,
        },
      ].slice(-PREDICTION_WINDOW_SIZE);

      if (enableLiveSequencePredictRef.current && sequenceFrameQualityOk) {
        lowQualityFramesRef.current = 0;
        liveSequenceRef.current = [
          ...liveSequenceRef.current,
          buildSequenceFrame({
            leftHandLandmarks,
            rightHandLandmarks,
            smoothedLeftHandLandmarks: smoothedLeftHand,
            smoothedRightHandLandmarks: smoothedRightHand,
            faceLandmarks,
            poseLandmarks,
            timestampMs,
          }),
        ].slice(-liveSequenceSettings.windowSize);
      } else if (enableLiveSequencePredictRef.current) {
        lowQualityFramesRef.current += 1;

        if (lowQualityFramesRef.current >= LIVE_SEQUENCE_QUALITY_RESET_FRAMES) {
          liveSequenceRef.current = [];
          lowQualityFramesRef.current = 0;
        }
      }

      if (enableLiveSequencePredictRef.current && motionFrameQualityOk) {
        motionLowQualityFramesRef.current = 0;
        motionSequenceRef.current = [
          ...motionSequenceRef.current,
          buildSequenceFrame({
            leftHandLandmarks,
            rightHandLandmarks,
            smoothedLeftHandLandmarks: smoothedLeftHand,
            smoothedRightHandLandmarks: smoothedRightHand,
            faceLandmarks,
            poseLandmarks,
            timestampMs,
          }),
        ].slice(-Math.max(liveSequenceSettings.windowSize, 24));
      } else if (enableLiveSequencePredictRef.current) {
        motionLowQualityFramesRef.current += 1;

        if (
          motionLowQualityFramesRef.current >=
          LIVE_SEQUENCE_QUALITY_RESET_FRAMES + 2
        ) {
          motionSequenceRef.current = [];
          motionLowQualityFramesRef.current = 0;
        }
      }

      const predictionWindow = predictionWindowRef.current;
      const summary = summarizePredictionWindow(predictionWindow);
      const waveMotionSummary = summarizeWaveMotion(motionSequenceRef.current);
      const greetingGuide = buildGreetingGuide({
        recognitionLevel: recognitionLevelRef.current,
        waveMotionSummary,
        leftHandLandmarks: smoothedLeftHand,
        rightHandLandmarks: smoothedRightHand,
        faceLandmarks: overlayFaceLandmarks,
        poseLandmarks: overlayPoseLandmarks,
        faceDetected,
        poseDetected,
      });
      const dominantGesture = summary.gesture;
      const dominantGestureKnown =
        dominantGesture && dominantGesture !== UNKNOWN_GESTURE;
      const isAlphabetMode = recognitionLevelRef.current === "alphabet";
      const isSequenceGestureStable =
        isAlphabetMode
          ? summary.weightedRatio >=
              MIN_ALPHABET_SEQUENCE_TRIGGER_WEIGHTED_RATIO &&
            summary.averageConfidence >=
              MIN_ALPHABET_SEQUENCE_TRIGGER_CONFIDENCE
          : dominantGestureKnown &&
            summary.weightedRatio >= MIN_SEQUENCE_TRIGGER_WEIGHTED_RATIO &&
            summary.ratio >= MIN_SEQUENCE_TRIGGER_RATIO &&
            summary.averageConfidence >= MIN_SEQUENCE_TRIGGER_CONFIDENCE &&
            summary.averageMargin >= MIN_SEQUENCE_TRIGGER_MARGIN;
      const isWaveSequenceStable =
        usesBodyGuideOverlay(recognitionLevelRef.current) &&
        motionSequenceRef.current.length >=
          Math.max(8, liveSequenceSettings.minFrames - 2) &&
        waveMotionSummary.waveLike &&
        waveMotionSummary.waveScore >= 0.62 &&
        waveMotionSummary.oneHandRatio >= 0.64;
      const sequenceFramesForPrediction =
        isWaveSequenceStable && !isSequenceGestureStable
          ? motionSequenceRef.current
          : liveSequenceRef.current;

      if (
        enableLiveSequencePredictRef.current &&
        (
          (sequenceFrameQualityOk && isSequenceGestureStable) ||
          (motionFrameQualityOk && isWaveSequenceStable)
        )
      ) {
        maybeRunLiveSequencePrediction(sequenceFramesForPrediction, {
          dominantGesture,
          dominantRatio: summary.ratio,
          dominantWeightedRatio: summary.weightedRatio,
          averageConfidence: summary.averageConfidence,
          averageMargin: summary.averageMargin,
          sequenceFrameCount: sequenceFramesForPrediction.length,
          sequenceTriggerReason: isWaveSequenceStable && !isSequenceGestureStable
            ? "wave_motion"
            : "stable_handshape",
          waveLike: waveMotionSummary.waveLike,
          waveScore: waveMotionSummary.waveScore,
          waveDirectionChanges: waveMotionSummary.directionChanges,
          waveOneHandRatio: waveMotionSummary.oneHandRatio,
          waveHandPresenceRatio: waveMotionSummary.handPresenceRatio,
          waveHorizontalRange: waveMotionSummary.xRange,
          waveVerticalRange: waveMotionSummary.yRange,
          waveHorizontalDominance: waveMotionSummary.horizontalDominance,
          waveDominantHand: waveMotionSummary.dominantHand,
        });
      }

      const dominantRatioPercent = Math.round(summary.ratio * 100);
      const dominantWeightedRatioPercent = Math.round(
        summary.weightedRatio * 100,
      );
      const currentGestureLabel =
        detectedGesture === UNKNOWN_GESTURE
          ? "Распознаю жест..."
          : detectedGesture;
      const liveGestureLabel =
        dominantGestureKnown && summary.count >= 4
          ? dominantGesture
          : currentGestureLabel;

      const hasEnoughFrames = predictionWindow.length >= STABLE_FRAME_TARGET;
      const isStableMajority =
        dominantGestureKnown &&
        (summary.weightedRatio >= WEIGHTED_RATIO_TARGET ||
          (summary.ratio >= MAJORITY_RATIO_TARGET &&
            summary.weightedRatio >= WEIGHTED_RATIO_TARGET - 0.08));
      const isConfidentEnough = confidencePercent >= MIN_CONFIRM_CONFIDENCE;
      const isConfirmed =
        detectedGesture === dominantGesture &&
        hasEnoughFrames &&
        isStableMajority &&
        isConfidentEnough &&
        predictionMargin >= MIN_CONFIRM_MARGIN &&
        summary.averageMargin >= MIN_STABLE_AVG_MARGIN &&
        summary.averageConfidence >= MIN_STABLE_AVG_CONFIDENCE &&
        trackingStable &&
        handVisibleEnough;

      emitLiveGesture(liveGestureLabel);
      emitPrediction({
        gesture: liveGestureLabel,
        confidence: confidencePercent,
        dominantGesture,
        dominantRatio: dominantRatioPercent,
        windowSize: predictionWindow.length,
        stableCount: summary.count,
        isConfidentEnough,
        isStableMajority,
        trackingStable,
        handVisibleEnough,
        guide: greetingGuide,
        debug: buildDebugPayload(
          prediction,
          summary,
          predictionWindow.length,
          lastQualityRef.current,
        ),
      });

      const confirmedLabel =
        dominantGestureKnown ? dominantGesture : "Жест не определён";

      if (!handVisibleEnough) {
        updateStatus("Поднесите руку ближе к камере");
      } else if (!trackingStable) {
        updateStatus(
          `Слежение нестабильно... jitter ${roundMetric(jitterAverage, 4)}`,
        );
      } else if (isConfirmed) {
        updateStatus(
          `Жест подтверждён: ${dominantGesture} (${confidencePercent}%)`,
        );
      } else if (dominantGestureKnown && hasEnoughFrames) {
        updateStatus(
          `Статистика: ${summary.count}/${predictionWindow.length} • ${dominantRatioPercent}%/${dominantWeightedRatioPercent}% • ${confidencePercent}%`,
        );
      } else if (dominantGestureKnown) {
        updateStatus(
          `Похоже на жест: ${dominantGesture} — ${dominantRatioPercent}% кадров`,
        );
      } else if (!faceDetected) {
        updateStatus("Лицо не найдено — держите лицо в кадре");
      } else if (!poseDetected) {
        updateStatus("Тело не найдено — держите плечи в кадре");
      } else {
        updateStatus("Распознаю жест...");
      }

      const now = Date.now();

      if (
        isConfirmed &&
        onGestureDetectedRef.current &&
        (confirmedLabel !== lastConfirmedPhraseRef.current.value ||
          now - lastConfirmedPhraseRef.current.timestamp >= 1200)
      ) {
        lastConfirmedPhraseRef.current = {
          value: confirmedLabel,
          timestamp: now,
        };
        onGestureDetectedRef.current(confirmedLabel);
      }

      const renderedLeftHand = smoothedLeftHand.length
        ? smoothedLeftHand
        : leftHandLandmarks;
      const renderedRightHand = smoothedRightHand.length
        ? smoothedRightHand
        : rightHandLandmarks;

      const overlayLeftHand = renderedLeftHand.length
        ? smoothLandmarksForOverlay(
            renderedLeftHand,
            overlayLeftHandRef.current,
            recognitionLevelRef.current,
          )
        : [];
      const overlayRightHand = renderedRightHand.length
        ? smoothLandmarksForOverlay(
            renderedRightHand,
            overlayRightHandRef.current,
            recognitionLevelRef.current,
          )
        : [];

      overlayLeftHandRef.current = overlayLeftHand.length ? overlayLeftHand : null;
      overlayRightHandRef.current = overlayRightHand.length ? overlayRightHand : null;

      if (showLandmarkOverlayRef.current && overlayLeftHand.length) {
        drawHand(context, overlayLeftHand, width, height, "#22d3ee");
      }

      if (showLandmarkOverlayRef.current && overlayRightHand.length) {
        drawHand(context, overlayRightHand, width, height, "#38bdf8");
      }

      if (
        showLandmarkOverlayRef.current &&
        usesBodyGuideOverlay(recognitionLevelRef.current)
      ) {
        drawBodyGuideMarkers(
          context,
          buildBodyGuideMarkers({
            faceLandmarks: overlayFaceLandmarks,
            poseLandmarks: overlayPoseLandmarks,
            leftHandLandmarks: overlayLeftHand.length ? overlayLeftHand : renderedLeftHand,
            rightHandLandmarks: overlayRightHand.length ? overlayRightHand : renderedRightHand,
            dominantHand: handednessLabel === "Left" ? "left" : "right",
          }),
          width,
          height,
        );
      }
    }

    function predictLoop() {
      const video = videoRef.current;
      const useHolisticLandmarker = usesBodyGuideOverlay(
        recognitionLevelRef.current,
      );
      const activeLandmarker = useHolisticLandmarker
        ? holisticLandmarkerRef.current
        : handLandmarkerRef.current;

      if (!isMounted) {
        return;
      }

      if (!video || !activeLandmarker) {
        animationRef.current = requestAnimationFrame(predictLoop);
        return;
      }

      if (video.readyState >= 2 && video.currentTime !== lastVideoTimeRef.current) {
        lastVideoTimeRef.current = video.currentTime;
        const timestampMs = performance.now();
        const results = activeLandmarker.detectForVideo(video, timestampMs);
        drawResults(results, timestampMs, useHolisticLandmarker);
        emitStreamFrame(video);
      }

      animationRef.current = requestAnimationFrame(predictLoop);
    }

    async function init() {
      try {
        setError("");
        resetGestureState();
        await setupCamera();
        await setupVisionLandmarkers();
        beginCalibrationSession();

        if (!isMounted) {
          return;
        }

        updateStatus("Камера готова", { force: true });
        predictLoop();
      } catch (caughtError) {
        console.error(caughtError);

        if (!isMounted) {
          return;
        }

        setError("Ошибка инициализации Holistic");
      }
    }

    init();

    return () => {
      isMounted = false;
      stopSession();
    };
  }, [startSignal, facingMode]);

  if (error) {
    return (
      <div className="camera-live-error">
        {error}
      </div>
    );
  }

  return (
    <div
      className={`camera-live ${
        facingMode === "user" && !mirrorPreview ? "camera-live-unmirrored" : ""
      }`}
    >
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="camera-live-video"
      />
      <canvas ref={canvasRef} className="camera-live-canvas" />
      {showStatusOverlay ? (
        <div className="camera-live-status">
          {status}
        </div>
      ) : null}
    </div>
  );
}
