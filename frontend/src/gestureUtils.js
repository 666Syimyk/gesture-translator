export const UNKNOWN_GESTURE = "Неизвестно";

function clamp(value, min = 0, max = 1) {
  return Math.min(max, Math.max(min, value));
}

function average(values) {
  if (!Array.isArray(values) || !values.length) {
    return 0;
  }

  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function distance(pointA, pointB) {
  return Math.hypot(pointA.x - pointB.x, pointA.y - pointB.y);
}

function safeRatio(value, divisor, fallback = 0) {
  if (!Number.isFinite(value) || !Number.isFinite(divisor) || divisor <= 0) {
    return fallback;
  }

  return value / divisor;
}

function fingerOpenScore(tip, pip, mcp, palmSize) {
  const tipToPip = safeRatio(pip.y - tip.y, palmSize);
  const pipToMcp = safeRatio(mcp.y - pip.y, palmSize);
  const tipToMcp = safeRatio(mcp.y - tip.y, palmSize);

  return clamp((tipToPip * 0.35 + pipToMcp * 0.25 + tipToMcp * 0.4 - 0.12) / 0.92);
}

function thumbOpenScore(
  thumbTip,
  thumbIp,
  thumbMcp,
  indexMcp,
  palmSize,
  handedness,
) {
  const direction = handedness === "Left" ? 1 : -1;
  const sideSpread = safeRatio((thumbTip.x - thumbMcp.x) * direction, palmSize);
  const diagonalSpread = safeRatio((thumbTip.x - indexMcp.x) * direction, palmSize);
  const thumbRaised = safeRatio(thumbMcp.y - thumbTip.y, palmSize);
  const thumbStack = safeRatio(thumbIp.y - thumbTip.y, palmSize);

  return clamp(
    Math.max(
      sideSpread * 0.78,
      diagonalSpread * 0.6,
      thumbRaised * 0.62,
      thumbStack * 0.42,
    ),
  );
}

function profileScore(openScores, profile) {
  const fingerWeights = {
    thumb: 0.8,
    index: 1,
    middle: 1,
    ring: 0.9,
    pinky: 0.9,
  };

  const scores = Object.entries(profile).flatMap(([finger, expectedState]) => {
    if (expectedState === "any") {
      return [];
    }

    const openScore = openScores[finger];
    const targetScore = expectedState === "open" ? openScore : 1 - openScore;

    return [clamp(targetScore) * fingerWeights[finger]];
  });

  return scores.length > 0 ? average(scores) : 0;
}

export function extractHandFeatures(landmarks, handedness = "Right") {
  if (!landmarks || landmarks.length < 21) {
    return null;
  }

  const wrist = landmarks[0];
  const thumbMcp = landmarks[2];
  const thumbIp = landmarks[3];
  const thumbTip = landmarks[4];
  const indexMcp = landmarks[5];
  const indexPip = landmarks[6];
  const indexTip = landmarks[8];
  const middleMcp = landmarks[9];
  const middlePip = landmarks[10];
  const middleTip = landmarks[12];
  const ringMcp = landmarks[13];
  const ringPip = landmarks[14];
  const ringTip = landmarks[16];
  const pinkyMcp = landmarks[17];
  const pinkyPip = landmarks[18];
  const pinkyTip = landmarks[20];

  const palmSize = Math.max(distance(wrist, middleMcp), 0.12);
  const palmWidth = Math.max(distance(indexMcp, pinkyMcp), 0.08);

  const openScores = {
    thumb: thumbOpenScore(
      thumbTip,
      thumbIp,
      thumbMcp,
      indexMcp,
      palmSize,
      handedness,
    ),
    index: fingerOpenScore(indexTip, indexPip, indexMcp, palmSize),
    middle: fingerOpenScore(middleTip, middlePip, middleMcp, palmSize),
    ring: fingerOpenScore(ringTip, ringPip, ringMcp, palmSize),
    pinky: fingerOpenScore(pinkyTip, pinkyPip, pinkyMcp, palmSize),
  };

  const strictOpenFlags = Object.fromEntries(
    Object.entries(openScores).map(([finger, score]) => [finger, score >= 0.56]),
  );

  const relaxedOpenFlags = Object.fromEntries(
    Object.entries(openScores).map(([finger, score]) => [finger, score >= 0.36]),
  );

  const openCount = Object.values(strictOpenFlags).filter(Boolean).length;
  const relaxedOpenCount = Object.values(relaxedOpenFlags).filter(Boolean).length;
  const longFingerOpenCount = [
    strictOpenFlags.index,
    strictOpenFlags.middle,
    strictOpenFlags.ring,
    strictOpenFlags.pinky,
  ].filter(Boolean).length;
  const longFingerRelaxedCount = [
    relaxedOpenFlags.index,
    relaxedOpenFlags.middle,
    relaxedOpenFlags.ring,
    relaxedOpenFlags.pinky,
  ].filter(Boolean).length;

  const palmSpread = clamp((safeRatio(palmWidth, palmSize) - 0.42) / 0.7);
  const fingerExtensionAverage = average([
    openScores.index,
    openScores.middle,
    openScores.ring,
    openScores.pinky,
  ]);

  const indexMiddleGap = clamp(safeRatio(distance(indexTip, middleTip), palmWidth));
  const middleRingGap = clamp(safeRatio(distance(middleTip, ringTip), palmWidth));
  const ringPinkyGap = clamp(safeRatio(distance(ringTip, pinkyTip), palmWidth));
  const thumbToIndexTip = clamp(safeRatio(distance(thumbTip, indexTip), palmWidth));
  const thumbAbovePalm = thumbTip.y < indexMcp.y;
  const curledLongFingerCount = [
    openScores.index,
    openScores.middle,
    openScores.ring,
    openScores.pinky,
  ].filter((score) => score <= 0.3).length;

  return {
    handedness,
    palmSize,
    palmWidth,
    palmSpread,
    openScores,
    strictOpenFlags,
    relaxedOpenFlags,
    openCount,
    relaxedOpenCount,
    longFingerOpenCount,
    longFingerRelaxedCount,
    curledLongFingerCount,
    fingerExtensionAverage,
    thumbAbovePalm,
    indexMiddleGap,
    middleRingGap,
    ringPinkyGap,
    thumbToIndexTip,
  };
}

export function predictGesture(landmarks, handedness = "Right") {
  const features = extractHandFeatures(landmarks, handedness);

  if (!features) {
    return {
      gesture: UNKNOWN_GESTURE,
      confidence: 0,
      margin: 0,
      features: null,
      scores: {},
    };
  }

  const {
    openScores,
    openCount,
    relaxedOpenCount,
    longFingerOpenCount,
    longFingerRelaxedCount,
    curledLongFingerCount,
    fingerExtensionAverage,
    thumbAbovePalm,
    palmSpread,
    indexMiddleGap,
    middleRingGap,
    ringPinkyGap,
    thumbToIndexTip,
  } = features;

  const indexIsolation = clamp(
    openScores.index - Math.max(openScores.middle, openScores.ring, openScores.pinky) + 0.46,
  );
  const twoFingerBalance = clamp(1 - Math.abs(openScores.index - openScores.middle));
  const twoFingerClosedOthers = clamp(1 - average([openScores.ring, openScores.pinky]) + 0.2);
  const widePalmGaps = clamp((indexMiddleGap + middleRingGap + ringPinkyGap) / 1.8);

  const scores = {
    "Открытая ладонь": average([
      profileScore(openScores, {
        thumb: "open",
        index: "open",
        middle: "open",
        ring: "open",
        pinky: "open",
      }),
      clamp((longFingerRelaxedCount / 4) * 0.82 + fingerExtensionAverage * 0.28),
      clamp(palmSpread * 0.6 + widePalmGaps * 0.45),
    ]),

    Кулак: average([
      profileScore(openScores, {
        thumb: "closed",
        index: "closed",
        middle: "closed",
        ring: "closed",
        pinky: "closed",
      }),
      clamp((5 - openCount) / 5 + curledLongFingerCount * 0.05),
      clamp(1 - fingerExtensionAverage + 0.18),
    ]),

    "Палец вверх": average([
      profileScore(openScores, {
        thumb: "open",
        index: "closed",
        middle: "closed",
        ring: "closed",
        pinky: "closed",
      }),
      thumbAbovePalm ? 1 : 0.35,
      clamp(1 - fingerExtensionAverage + 0.2),
      clamp(1 - openScores.index * 0.5),
    ]),

    Указательный: average([
      profileScore(openScores, {
        thumb: "any",
        index: "open",
        middle: "closed",
        ring: "closed",
        pinky: "closed",
      }),
      indexIsolation,
      clamp(1 - openScores.middle + 0.16),
      clamp(1 - indexMiddleGap * 0.7),
    ]),

    "Два пальца": average([
      profileScore(openScores, {
        thumb: "any",
        index: "open",
        middle: "open",
        ring: "closed",
        pinky: "closed",
      }),
      twoFingerBalance,
      twoFingerClosedOthers,
      clamp(indexMiddleGap * 0.85 + (1 - openScores.ring) * 0.15),
    ]),

    "Три пальца": average([
      profileScore(openScores, {
        thumb: "closed",
        index: "open",
        middle: "open",
        ring: "open",
        pinky: "closed",
      }),
      clamp(1 - Math.abs(openScores.index - openScores.middle)),
      clamp(1 - Math.abs(openScores.middle - openScores.ring)),
      clamp(1 - openScores.pinky + 0.2),
      clamp(middleRingGap * 0.7 + 0.24),
    ]),

    "Четыре пальца": average([
      profileScore(openScores, {
        thumb: "closed",
        index: "open",
        middle: "open",
        ring: "open",
        pinky: "open",
      }),
      clamp(1 - openScores.thumb + 0.2),
      clamp(longFingerOpenCount / 4 + 0.14),
      clamp(ringPinkyGap * 0.6 + 0.3),
    ]),
  };

  if (longFingerRelaxedCount >= 4 && palmSpread >= 0.24) {
    scores["Открытая ладонь"] = Math.max(
      scores["Открытая ладонь"],
      clamp(0.72 + fingerExtensionAverage * 0.16 + palmSpread * 0.12),
    );
  }

  if (longFingerOpenCount >= 4) {
    scores["Открытая ладонь"] = Math.max(
      scores["Открытая ладонь"],
      clamp(0.82 + palmSpread * 0.1 + widePalmGaps * 0.08),
    );
    scores["Два пальца"] *= 0.45;
    scores.Указательный *= 0.55;
  }

  if (openScores.index >= 0.72 && openScores.middle >= 0.65) {
    scores.Указательный *= 0.48;
  }

  if (openScores.index >= 0.74 && openScores.middle <= 0.4 && openScores.ring <= 0.38) {
    scores["Два пальца"] *= 0.58;
    scores.Указательный = Math.max(scores.Указательный, clamp(0.75 + indexIsolation * 0.12));
  }

  if (twoFingerBalance >= 0.86 && indexMiddleGap >= 0.22 && twoFingerClosedOthers >= 0.72) {
    scores["Два пальца"] = Math.max(
      scores["Два пальца"],
      clamp(0.76 + twoFingerBalance * 0.12),
    );
  }

  if (thumbAbovePalm && openScores.thumb >= 0.6 && thumbToIndexTip >= 0.35) {
    scores["Палец вверх"] = Math.max(
      scores["Палец вверх"],
      clamp(0.78 + (1 - fingerExtensionAverage) * 0.16),
    );
  }

  if (relaxedOpenCount <= 1) {
    scores["Открытая ладонь"] *= 0.38;
    scores["Два пальца"] *= 0.7;
  }

  const rankedGestures = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  const [bestGesture, bestScore] = rankedGestures[0];
  const secondScore = rankedGestures[1]?.[1] ?? 0;
  const margin = bestScore - secondScore;
  const confidence = clamp(bestScore * 0.74 + margin * 0.65);

  const dynamicMinMargin = bestScore >= 0.84 ? 0.05 : 0.07;

  if (confidence < 0.65 || margin < dynamicMinMargin) {
    return {
      gesture: UNKNOWN_GESTURE,
      confidence,
      margin,
      features,
      scores,
    };
  }

  return {
    gesture: bestGesture,
    confidence,
    margin,
    features,
    scores,
  };
}

export function detectSimpleGesture(landmarks, handedness = "Right") {
  return predictGesture(landmarks, handedness).gesture;
}
