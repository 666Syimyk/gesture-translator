export const GESTURE_LABEL_RU_BY_ID = Object.freeze({
  none: "none",
  privet: "\u041f\u0440\u0438\u0432\u0435\u0442",
  poka: "\u041f\u043e\u043a\u0430",
  ya: "\u042f",
  ty: "\u0422\u044b",
  muzhchina: "\u041c\u0443\u0436\u0447\u0438\u043d\u0430",
  zhenshchina: "\u0416\u0435\u043d\u0449\u0438\u043d\u0430",
  bolshoy: "\u0411\u043e\u043b\u044c\u0448\u043e\u0439",
  malenkiy: "\u041c\u0430\u043b\u0435\u043d\u044c\u043a\u0438\u0439",
  krasivyy: "\u041a\u0440\u0430\u0441\u0438\u0432\u044b\u0439",
  spasibo: "\u0421\u043f\u0430\u0441\u0438\u0431\u043e",
  est: "\u0415\u0441\u0442\u044c",
});

const GESTURE_LABEL_ID_BY_RU = Object.freeze(
  Object.fromEntries(Object.entries(GESTURE_LABEL_RU_BY_ID).map(([labelId, labelRu]) => [labelRu, labelId])),
);
const GESTURE_LABEL_ID_BY_RU_LOWER = Object.freeze(
  Object.fromEntries(
    Object.entries(GESTURE_LABEL_ID_BY_RU).map(([labelRu, labelId]) => [labelRu.toLowerCase(), labelId]),
  ),
);
const GESTURE_LABEL_ALIASES = Object.freeze({
  krasiviy: "krasivyy",
});

export function resolveGestureLabelId(value) {
  const normalized = String(value ?? "").trim();
  if (!normalized) {
    return "none";
  }
  const lowerNormalized = normalized.toLowerCase();
  if (normalized in GESTURE_LABEL_RU_BY_ID) {
    return normalized;
  }
  if (lowerNormalized in GESTURE_LABEL_RU_BY_ID) {
    return lowerNormalized;
  }
  return (
    GESTURE_LABEL_ALIASES[lowerNormalized] ??
    GESTURE_LABEL_ID_BY_RU[normalized] ??
    GESTURE_LABEL_ID_BY_RU_LOWER[lowerNormalized] ??
    "none"
  );
}

export function resolveGestureLabelRu(labelId, fallback = "") {
  const normalized = resolveGestureLabelId(labelId);
  if (normalized && normalized in GESTURE_LABEL_RU_BY_ID) {
    return GESTURE_LABEL_RU_BY_ID[normalized];
  }
  const rawFallback = String(fallback ?? "").trim();
  return rawFallback || "none";
}

function normalizeGestureRankedItem(item) {
  if (!item || typeof item !== "object") {
    return item;
  }

  const labelId = resolveGestureLabelId(item.label_id ?? item.labelId ?? item.label_ru ?? item.labelRu);
  return {
    ...item,
    label_id: labelId,
    label_ru: resolveGestureLabelRu(labelId, item.label_ru ?? item.labelRu),
  };
}

export function normalizeGesturePredictionMessage(message) {
  if (!message || typeof message !== "object" || message.type !== "prediction") {
    return message;
  }

  const labelId = resolveGestureLabelId(message.label_id ?? message.labelId ?? message.label_ru ?? message.labelRu);
  const topLabelId = resolveGestureLabelId(
    message.top_label_id ?? message.topLabelId ?? message.top_label_ru ?? message.topLabelRu,
  );

  return {
    ...message,
    label_id: labelId,
    label_ru: resolveGestureLabelRu(labelId, message.label_ru ?? message.labelRu),
    top_label_id: topLabelId,
    top_label_ru: resolveGestureLabelRu(topLabelId, message.top_label_ru ?? message.topLabelRu),
    top_final: Array.isArray(message.top_final) ? message.top_final.map(normalizeGestureRankedItem) : message.top_final,
    top_rules: Array.isArray(message.top_rules) ? message.top_rules.map(normalizeGestureRankedItem) : message.top_rules,
    top_model: Array.isArray(message.top_model) ? message.top_model.map(normalizeGestureRankedItem) : message.top_model,
    debug:
      message.debug && typeof message.debug === "object"
        ? {
            ...message.debug,
            rule_score_summary:
              message.debug.rule_score_summary && typeof message.debug.rule_score_summary === "object"
                ? {
                    ...message.debug.rule_score_summary,
                    spatial_top: Array.isArray(message.debug.rule_score_summary.spatial_top)
                      ? message.debug.rule_score_summary.spatial_top.map(normalizeGestureRankedItem)
                      : message.debug.rule_score_summary.spatial_top,
                    temporal_top: Array.isArray(message.debug.rule_score_summary.temporal_top)
                      ? message.debug.rule_score_summary.temporal_top.map(normalizeGestureRankedItem)
                      : message.debug.rule_score_summary.temporal_top,
                  }
                : message.debug.rule_score_summary,
          }
        : message.debug,
  };
}
