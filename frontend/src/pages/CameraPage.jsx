import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import CameraView from "../components/CameraView";
import {
  GlassCard,
  PrimaryButton,
  SecondaryButton,
} from "../components/PremiumUI";
import { fetchLatestModel, predictLatestModel } from "../api/mlApi";
import { createRecognitionRun } from "../api/recognitionApi";
import {
  cancelSpeech,
  getSpeakableText,
  isSpeakableText,
  speakWithSettings,
} from "../utils/speech";
import {
  DEFAULT_RECOGNITION_LEVEL,
  getRecognitionLevelLabel,
  RECOGNITION_LEVEL_OPTIONS,
} from "../recognitionLevels";

const ALL_CATEGORY = "Все";
const DEFAULT_ML_COMMIT_THRESHOLD = 0.85;
const DEFAULT_ML_STABILITY_WINDOW_SIZE = 5;
const DEFAULT_ML_STABILITY_RATIO = 0.75;
const DEFAULT_ML_MIN_STABLE_COUNT = 4;
const ALPHABET_ML_COMMIT_THRESHOLD = 0.55;
const ALPHABET_ML_STABILITY_WINDOW_SIZE = 4;
const ALPHABET_ML_STABILITY_RATIO = 0.7;
const ALPHABET_ML_MIN_STABLE_COUNT = 3;
const ALPHABET_ML_MIN_CONFIDENCE_MARGIN = 0.12;
const SIGN_ML_COMMIT_THRESHOLD = 0.5;
const SIGN_ML_STABILITY_WINDOW_SIZE = 5;
const SIGN_ML_STABILITY_RATIO = 0.75;
const SIGN_ML_MIN_STABLE_COUNT = 4;
const RELIABLE_SIGN_MIN_TEST_ACCURACY = 0.75;
const USER_WORD_SIGN_MIN_TEST_ACCURACY = 0.35;
const RELIABLE_SIGN_MIN_TEST_SAMPLES = 5;
const RECOGNITION_SPEECH_COOLDOWN_MS = 1400;
const RECOGNITION_REPEAT_COOLDOWN_MS = 1200;
const LIVE_PREDICTION_HOLD_MS = 1600;
const ALPHABET_REPEAT_RELEASE_MISSES = 2;
const EMPTY_RECOGNIZED_TEXT = "...";
const ML_STABILITY_MISS_TOLERANCE = 3;
const ML_QUALITY_REJECT_MESSAGE = "Сигнал нестабилен, повторите жест";
const ML_STABILITY_REJECT_MESSAGE = "Жду стабильного подтверждения";
const ML_REJECT_MESSAGE = "Не уверен, повторите жест";

const CAMERA_OUTPUT_MODE_OPTIONS = RECOGNITION_LEVEL_OPTIONS;

const FINGER_LABELS = {
  thumb: "Б",
  index: "У",
  middle: "С",
  ring: "Без",
  pinky: "М",
};

function getMlThresholds(recognitionLevel, modelMetadata = null) {
  const metadataThreshold = Number(
    modelMetadata?.config?.confidence_threshold,
  );
  const resolvedMetadataThreshold =
    Number.isFinite(metadataThreshold) && metadataThreshold > 0
      ? metadataThreshold
      : null;

  if (recognitionLevel === "alphabet") {
    return {
      commitThreshold: Math.max(
        resolvedMetadataThreshold ?? 0,
        ALPHABET_ML_COMMIT_THRESHOLD,
      ),
      stabilityWindowSize: ALPHABET_ML_STABILITY_WINDOW_SIZE,
      stabilityRatio: ALPHABET_ML_STABILITY_RATIO,
      minStableCount: ALPHABET_ML_MIN_STABLE_COUNT,
      minConfidenceMargin: ALPHABET_ML_MIN_CONFIDENCE_MARGIN,
    };
  }

  if (recognitionLevel === "sign") {
    return {
      commitThreshold: Math.max(
        resolvedMetadataThreshold ?? 0,
        SIGN_ML_COMMIT_THRESHOLD,
      ),
      stabilityWindowSize: SIGN_ML_STABILITY_WINDOW_SIZE,
      stabilityRatio: SIGN_ML_STABILITY_RATIO,
      minStableCount: SIGN_ML_MIN_STABLE_COUNT,
      minConfidenceMargin: 0,
    };
  }

  return {
    commitThreshold: resolvedMetadataThreshold ?? DEFAULT_ML_COMMIT_THRESHOLD,
    stabilityWindowSize: DEFAULT_ML_STABILITY_WINDOW_SIZE,
    stabilityRatio: DEFAULT_ML_STABILITY_RATIO,
    minStableCount: DEFAULT_ML_MIN_STABLE_COUNT,
    minConfidenceMargin: 0,
  };
}

function isRecognizedPhrase(text) {
  return text && text !== "..." && text !== "Жест не распознан";
}

function normalizeAlphabetText(text) {
  return isRecognizedPhrase(text) ? String(text) : "";
}

function appendAlphabetCharacter(currentText, character) {
  const normalizedCharacter = String(character ?? "").trim();

  if (!normalizedCharacter) {
    return normalizeAlphabetText(currentText) || EMPTY_RECOGNIZED_TEXT;
  }

  return `${normalizeAlphabetText(currentText)}${normalizedCharacter}`;
}

function appendAlphabetSpace(currentText) {
  const normalizedText = normalizeAlphabetText(currentText).trimEnd();

  if (!normalizedText) {
    return EMPTY_RECOGNIZED_TEXT;
  }

  return `${normalizedText} `;
}

function removeAlphabetCharacter(currentText) {
  const normalizedText = normalizeAlphabetText(currentText);

  if (!normalizedText) {
    return EMPTY_RECOGNIZED_TEXT;
  }

  const nextText = normalizedText.slice(0, -1);
  return nextText || EMPTY_RECOGNIZED_TEXT;
}

function createAlphabetCommitLock() {
  return {
    symbol: "",
    requiresRelease: false,
  };
}

function summarizeMlPredictionWindow(items = []) {
  if (!items.length) {
    return {
      label: "",
      count: 0,
      ratio: 0,
      averageConfidence: 0,
    };
  }

  const grouped = items.reduce((accumulator, item) => {
    const key = item.label;

    if (!accumulator[key]) {
      accumulator[key] = {
        count: 0,
        confidence: 0,
      };
    }

    accumulator[key].count += 1;
    accumulator[key].confidence += Number(item.confidence ?? 0);
    return accumulator;
  }, {});

  const [label, stats] = Object.entries(grouped).sort((left, right) => {
    if (right[1].count !== left[1].count) {
      return right[1].count - left[1].count;
    }

    return right[1].confidence - left[1].confidence;
  })[0];

  return {
    label,
    count: stats.count,
    ratio: stats.count / items.length,
    averageConfidence: stats.confidence / stats.count,
  };
}

function getReliableSignEntries(modelMetadata = null) {
  const perClassAccuracy = modelMetadata?.metrics?.test?.per_class_accuracy ?? {};
  const minAccuracy = modelMetadata?.trained_user_words
    ? USER_WORD_SIGN_MIN_TEST_ACCURACY
    : RELIABLE_SIGN_MIN_TEST_ACCURACY;

  return Object.entries(perClassAccuracy)
    .map(([labelKey, value]) => ({
      key: labelKey,
      label: value?.label ?? "",
      accuracy: Number(value?.accuracy ?? 0),
      sampleCount: Number(value?.sample_count ?? 0),
    }))
    .filter(
      (entry) =>
        entry.label &&
        Number.isFinite(entry.accuracy) &&
        Number.isFinite(entry.sampleCount) &&
        entry.accuracy >= minAccuracy &&
        entry.sampleCount >= RELIABLE_SIGN_MIN_TEST_SAMPLES,
    )
    .sort((left, right) => {
      if (right.accuracy !== left.accuracy) {
        return right.accuracy - left.accuracy;
      }

      return right.sampleCount - left.sampleCount;
    });
}

function buildLiveQualityHints({
  gestureDebug,
  recognitionMode,
  liveModelRejectReason,
  liveModelError,
}) {
  const hints = [];

  if (!gestureDebug) {
    return hints;
  }

  if (!gestureDebug.handVisibleEnough) {
    hints.push("Поднесите руку ближе к камере и держите кисть полностью в кадре.");
  }

  if (!gestureDebug.trackingStable) {
    hints.push("Стабилизируйте руку на секунду и не меняйте жест слишком резко.");
  }

  if ((gestureDebug.freezeFramesRemaining ?? 0) > 0) {
    hints.push("Камера кратко потеряла руку. Повторите жест без резкого движения.");
  }

  if (
    typeof gestureDebug.jitterAverage === "number" &&
    gestureDebug.jitterAverage > 0.025
  ) {
    hints.push("Жест слишком дрожит. Покажите его медленнее и спокойнее.");
  }

  if (
    recognitionMode !== "rules" &&
    liveModelRejectReason &&
    !hints.includes(liveModelRejectReason)
  ) {
    hints.push(liveModelRejectReason);
  }

  if (
    recognitionMode !== "rules" &&
    liveModelError &&
    !hints.includes(liveModelError)
  ) {
    hints.push(liveModelError);
  }

  return hints.slice(0, 3);
}

export default function CameraPage({
  currentUser,
  recognizedText,
  setRecognizedText,
  addToHistory,
  liveGesture,
  setLiveGesture,
  settings,
  phraseCategories,
  phraseCards,
  isContentLoading,
  contentError,
}) {
  const lastSpokenRecognitionRef = useRef({
    key: "",
    timestamp: 0,
  });
  const lastRecognitionKeyRef = useRef({
    key: "",
    timestamp: 0,
  });
  const recognizedTextRef = useRef(recognizedText);
  const previousRecognitionLevelRef = useRef(DEFAULT_RECOGNITION_LEVEL);
  const alphabetCommitLockRef = useRef(createAlphabetCommitLock());
  const gestureDebugRef = useRef(null);
  const mlPredictionWindowRef = useRef([]);
  const mlPredictionMissesRef = useRef(0);
  const lastStableLivePredictionRef = useRef(null);
  const [selectedCategory, setSelectedCategory] = useState(ALL_CATEGORY);
  const [selectedRecognitionLevel, setSelectedRecognitionLevel] = useState(
    DEFAULT_RECOGNITION_LEVEL,
  );
  const [recognitionMode, setRecognitionMode] = useState("hybrid");
  const [gestureConfidence, setGestureConfidence] = useState(0);
  const [gestureDebug, setGestureDebug] = useState(null);
  const [liveModelPrediction, setLiveModelPrediction] = useState(null);
  const [isLiveModelReady, setIsLiveModelReady] = useState(false);
  const [, setIsLiveModelPredicting] = useState(false);
  const [liveModelError, setLiveModelError] = useState("");
  const [liveModelRejectReason, setLiveModelRejectReason] = useState("");
  const [liveModelMetadata, setLiveModelMetadata] = useState(null);

  useEffect(() => {
    recognizedTextRef.current = recognizedText;
  }, [recognizedText]);

  useEffect(() => {
    if (previousRecognitionLevelRef.current === selectedRecognitionLevel) {
      return;
    }

    previousRecognitionLevelRef.current = selectedRecognitionLevel;
    recognizedTextRef.current = EMPTY_RECOGNIZED_TEXT;
    setRecognizedText(EMPTY_RECOGNIZED_TEXT);
    lastRecognitionKeyRef.current = {
      key: "",
      timestamp: 0,
    };
    alphabetCommitLockRef.current = createAlphabetCommitLock();
  }, [selectedRecognitionLevel, setRecognizedText]);

  useEffect(() => {
    let isMounted = true;

    async function loadModelStatus() {
      try {
        const resolvedMetadata = await fetchLatestModel(undefined, {
          allowedRecognitionLevels: [selectedRecognitionLevel],
          profile: "fast",
        });

        if (!isMounted) {
          return;
        }

        setLiveModelMetadata(resolvedMetadata);
        setIsLiveModelReady(true);
        setLiveModelPrediction(null);
        setLiveModelError("");
        setLiveModelRejectReason("");
      } catch {
        if (!isMounted) {
          return;
        }

        setLiveModelMetadata(null);
        setIsLiveModelReady(false);
        setLiveModelPrediction(null);
        setLiveModelError("ML-модель ещё не обучена");
        setLiveModelRejectReason("");
      }
    }

    loadModelStatus();

    return () => {
      isMounted = false;
    };
  }, [selectedRecognitionLevel]);

  useEffect(() => {
    if (selectedRecognitionLevel !== "phrase" && recognitionMode !== "ml") {
      setRecognitionMode("ml");
    }
  }, [recognitionMode, selectedRecognitionLevel]);

  const categories = useMemo(() => {
    const preferredCategories = settings.preferredCategories ?? [];
    const sortedCategories = [...phraseCategories].sort((left, right) => {
      const leftPreferred = preferredCategories.includes(left.name) ? 0 : 1;
      const rightPreferred = preferredCategories.includes(right.name) ? 0 : 1;

      if (leftPreferred !== rightPreferred) {
        return leftPreferred - rightPreferred;
      }

      return left.sortOrder - right.sortOrder;
    });

    return [ALL_CATEGORY, ...sortedCategories.map((category) => category.name)];
  }, [phraseCategories, settings.preferredCategories]);

  const filteredCards = useMemo(() => {
    const preferredCategories = settings.preferredCategories ?? [];
    const scopedCards =
      selectedCategory === ALL_CATEGORY
        ? phraseCards
        : phraseCards.filter((item) => item.category === selectedCategory);

    const scopedByLevel = scopedCards.filter(
      (item) => item.recognitionLevel === selectedRecognitionLevel,
    );

    return [...scopedByLevel].sort((left, right) => {
      const leftPreferred = preferredCategories.includes(left.category) ? 0 : 1;
      const rightPreferred = preferredCategories.includes(right.category) ? 0 : 1;

      if (leftPreferred !== rightPreferred) {
        return leftPreferred - rightPreferred;
      }

      if (left.category !== right.category) {
        return left.category.localeCompare(right.category, "ru");
      }

      return left.sortOrder - right.sortOrder;
    });
  }, [
    phraseCards,
    selectedCategory,
    selectedRecognitionLevel,
    settings.preferredCategories,
  ]);

  const speakRecognition = useCallback(
    (textToSpeak = recognizedText, options = {}) => {
      const recognitionLevel =
        options.recognitionLevel ?? selectedRecognitionLevel;

      if (!isSpeakableText(textToSpeak, { recognitionLevel })) {
        return false;
      }

      const speechText = getSpeakableText(textToSpeak, {
        recognitionLevel,
      });
      const speechKey = `${recognitionLevel}:${speechText}`;
      const now = Date.now();

      if (
        !options.force &&
        lastSpokenRecognitionRef.current.key === speechKey &&
        now - lastSpokenRecognitionRef.current.timestamp <
          RECOGNITION_SPEECH_COOLDOWN_MS
      ) {
        return false;
      }

      const didSpeak = speakWithSettings(textToSpeak, settings, {
        recognitionLevel,
      });

      if (didSpeak) {
        lastSpokenRecognitionRef.current = {
          key: speechKey,
          timestamp: now,
        };
      }

      return didSpeak;
    },
    [recognizedText, selectedRecognitionLevel, settings],
  );

  const commitRecognition = useCallback(
    (text, options = {}) => {
      if (!isRecognizedPhrase(text)) {
        return;
      }

      const {
        sourceMode = "rules",
        confidence = null,
        rawOutput = {},
        phraseId,
        recognitionLevel = selectedRecognitionLevel,
        autoSpeak = settings.autoSpeakEnabled,
        forceSpeak = false,
      } = options;
      const recognitionKey = `${recognitionLevel}:${sourceMode}:${text}`;
      const now = Date.now();
      const normalizedAlphabetSymbol =
        recognitionLevel === "alphabet" ? String(text ?? "").trim() : "";
      const isRepeatedRecognition =
        recognitionLevel !== "alphabet" &&
        lastRecognitionKeyRef.current.key === recognitionKey &&
        now - lastRecognitionKeyRef.current.timestamp <
          RECOGNITION_REPEAT_COOLDOWN_MS;

      if (recognitionLevel === "alphabet") {
        const alphabetCommitLock = alphabetCommitLockRef.current;

        if (
          alphabetCommitLock.requiresRelease &&
          alphabetCommitLock.symbol === normalizedAlphabetSymbol &&
          !forceSpeak
        ) {
          return;
        }
      }

      const committedText =
        recognitionLevel === "alphabet"
          ? appendAlphabetCharacter(recognizedTextRef.current, text)
          : text;

      recognizedTextRef.current = committedText;
      setRecognizedText(committedText);

      if (recognitionLevel === "alphabet") {
        alphabetCommitLockRef.current = {
          symbol: normalizedAlphabetSymbol,
          requiresRelease: true,
        };
      }

      if (recognitionLevel !== "alphabet") {
        addToHistory("gesture", text);
      }

      const speechTextToSpeak =
        recognitionLevel === "alphabet" ? text : committedText;

      if (forceSpeak || autoSpeak) {
        speakRecognition(speechTextToSpeak, {
          recognitionLevel,
          force: forceSpeak,
        });
      }

      if (isRepeatedRecognition) {
        return;
      }

      lastRecognitionKeyRef.current = {
        key: recognitionKey,
        timestamp: now,
      };

      createRecognitionRun({
        userEmail: currentUser?.email,
        phraseId,
        recognizedText: recognitionLevel === "alphabet" ? text : committedText,
        sourceMode,
        signLanguage: settings.signLanguage,
        recognitionLevel,
        confidence,
        rawOutput,
      }).catch(() => {});
    },
    [
      addToHistory,
      currentUser?.email,
      selectedRecognitionLevel,
      setRecognizedText,
      settings.autoSpeakEnabled,
      settings.signLanguage,
      speakRecognition,
    ],
  );

  const handleGestureDetected = useCallback(
    (text) => {
      if (recognitionMode === "ml" || selectedRecognitionLevel !== "phrase") {
        return;
      }

      commitRecognition(text, {
        sourceMode: recognitionMode === "hybrid" ? "hybrid_rules" : "rules",
        confidence: gestureConfidence / 100,
        rawOutput: {
          gestureDebug,
          liveGesture,
        },
      });
    },
    [
      commitRecognition,
      gestureConfidence,
      gestureDebug,
      liveGesture,
      recognitionMode,
      selectedRecognitionLevel,
    ],
  );

  const handlePredictionChange = useCallback((prediction) => {
    setGestureConfidence(prediction.confidence ?? 0);
    setGestureDebug(prediction.debug ?? null);
  }, []);

  useEffect(() => {
    gestureDebugRef.current = gestureDebug;
  }, [gestureDebug]);

  const hasAlphabetText = useMemo(
    () => normalizeAlphabetText(recognizedText).length > 0,
    [recognizedText],
  );

  const updateAlphabetText = useCallback(
    (updater) => {
      const nextText = updater(recognizedTextRef.current);
      recognizedTextRef.current = nextText;
      setRecognizedText(nextText);
      return nextText;
    },
    [setRecognizedText],
  );

  const handleAlphabetSpace = useCallback(() => {
    updateAlphabetText((currentText) => appendAlphabetSpace(currentText));
    lastRecognitionKeyRef.current = {
      key: "",
      timestamp: 0,
    };
    alphabetCommitLockRef.current = createAlphabetCommitLock();
  }, [updateAlphabetText]);

  const handleAlphabetDelete = useCallback(() => {
    updateAlphabetText((currentText) => removeAlphabetCharacter(currentText));
    lastRecognitionKeyRef.current = {
      key: "",
      timestamp: 0,
    };
    alphabetCommitLockRef.current = createAlphabetCommitLock();
  }, [updateAlphabetText]);

  const reliableSignEntries = useMemo(
    () =>
      selectedRecognitionLevel === "sign"
        ? getReliableSignEntries(liveModelMetadata)
        : [],
    [liveModelMetadata, selectedRecognitionLevel],
  );
  const reliableSignLabelKeys = useMemo(
    () => reliableSignEntries.map((entry) => entry.key),
    [reliableSignEntries],
  );

  const handleSequencePredict = useCallback(
    async (payload) => {
      if (!isLiveModelReady || recognitionMode === "rules") {
        return;
      }

      const {
        commitThreshold,
        stabilityWindowSize,
        stabilityRatio,
        minStableCount,
        minConfidenceMargin,
      } = getMlThresholds(selectedRecognitionLevel, liveModelMetadata);

      setIsLiveModelPredicting(true);

      try {
        const now = Date.now();
        const prediction = await predictLatestModel({
          profile: "fast",
          allowedRecognitionLevels: [selectedRecognitionLevel],
          allowedLabelKeys:
            selectedRecognitionLevel === "sign" && reliableSignLabelKeys.length
              ? reliableSignLabelKeys
              : undefined,
          sequence: payload.frames,
        });

        const confidence = prediction?.confidence ?? 0;
        const allCandidates = prediction?.scores ?? [];
        const topCandidates = allCandidates.slice(0, 3);
        const confidenceMargin =
          topCandidates.length >= 2
            ? Number(topCandidates[0]?.confidence ?? 0) -
              Number(topCandidates[1]?.confidence ?? 0)
            : Number(topCandidates[0]?.confidence ?? 0);
        const quality = gestureDebugRef.current;
        const qualityRejected =
          !!quality &&
          (!quality.trackingStable ||
            !quality.handVisibleEnough ||
            Number(quality.jitterAverage ?? 0) > 0.025);
        const marginRejected = confidenceMargin < minConfidenceMargin;
        const heldPredictionState = lastStableLivePredictionRef.current;
        const heldPredictionLabel = heldPredictionState?.prediction?.label;
        const relaxedConfidenceThreshold = commitThreshold * 0.85;
        const relaxedMarginThreshold =
          minConfidenceMargin > 0 ? minConfidenceMargin * 0.5 : 0;
        const softAcceptPrediction =
          prediction?.label &&
          prediction.label === heldPredictionLabel &&
          confidence >= relaxedConfidenceThreshold &&
          !qualityRejected &&
          confidenceMargin >= relaxedMarginThreshold;

        if (
          (prediction?.label &&
            confidence >= commitThreshold &&
            !qualityRejected &&
            !marginRejected) ||
          softAcceptPrediction
        ) {
          mlPredictionMissesRef.current = 0;
          mlPredictionWindowRef.current = [
            ...mlPredictionWindowRef.current,
            {
              label: prediction.label,
              confidence,
            },
          ].slice(-stabilityWindowSize);
        } else {
          mlPredictionMissesRef.current += 1;

          if (mlPredictionMissesRef.current > ML_STABILITY_MISS_TOLERANCE) {
            mlPredictionWindowRef.current = [];
            mlPredictionMissesRef.current = 0;
            lastStableLivePredictionRef.current = null;
          }
        }

        const stability = summarizeMlPredictionWindow(mlPredictionWindowRef.current);
        const isStablePrediction =
          stability.label === prediction?.label &&
          stability.count >= minStableCount &&
          stability.ratio >= stabilityRatio;
        const rejectReason =
          confidence < commitThreshold
            ? ML_REJECT_MESSAGE
            : qualityRejected
              ? ML_QUALITY_REJECT_MESSAGE
              : marginRejected
                ? "Похожие буквы слишком близки, покажите жест ещё раз."
              : !isStablePrediction
                ? ML_STABILITY_REJECT_MESSAGE
                : "";
        const isRejected = !!rejectReason;

        const candidatePrediction = {
          ...prediction,
          allCandidates,
          topCandidates,
          confidenceMargin,
          stability,
          qualityRejected,
          marginRejected,
          isRejected,
          isHolding: false,
        };
        const canHoldStablePrediction =
          isRejected &&
          heldPredictionState &&
          now - heldPredictionState.timestamp <= LIVE_PREDICTION_HOLD_MS;
        const resolvedPrediction = canHoldStablePrediction
          ? {
              ...heldPredictionState.prediction,
              topCandidates:
                heldPredictionState.prediction.topCandidates?.length
                  ? heldPredictionState.prediction.topCandidates
                  : topCandidates,
              allCandidates:
                heldPredictionState.prediction.allCandidates?.length
                  ? heldPredictionState.prediction.allCandidates
                  : allCandidates,
              stability: heldPredictionState.prediction.stability ?? stability,
              qualityRejected,
              marginRejected,
              isRejected: false,
              isHolding: true,
            }
          : candidatePrediction;
        const stabilityOverridePrediction =
          stability.label &&
          stability.label !== prediction?.label &&
          stability.count >= Math.min(stabilityWindowSize, minStableCount + 1) &&
          stability.ratio >= stabilityRatio &&
          stability.averageConfidence >= commitThreshold * 0.92 &&
          !qualityRejected;
        const effectiveTopCandidates = stabilityOverridePrediction
          ? [
              {
                label: stability.label,
                confidence: stability.averageConfidence,
              },
              ...topCandidates.filter((candidate) => candidate.label !== stability.label),
            ].slice(0, 3)
          : resolvedPrediction.topCandidates ?? topCandidates;
        const effectiveResolvedPrediction = stabilityOverridePrediction
          ? {
              ...resolvedPrediction,
              label: stability.label,
              confidence: stability.averageConfidence,
              topCandidates: effectiveTopCandidates,
              allCandidates:
                resolvedPrediction.allCandidates ??
                heldPredictionState?.prediction?.allCandidates ??
                allCandidates,
              marginRejected: false,
              isRejected: false,
              isHolding: false,
              rawLabel: prediction?.label,
              rawConfidence: confidence,
            }
          : {
              ...resolvedPrediction,
              topCandidates: effectiveTopCandidates,
              allCandidates:
                resolvedPrediction.allCandidates ??
                heldPredictionState?.prediction?.allCandidates ??
                allCandidates,
            };
        const effectiveIsRejected = stabilityOverridePrediction ? false : isRejected;
        const effectiveRejectReason = stabilityOverridePrediction ? "" : rejectReason;

        if (
          selectedRecognitionLevel === "alphabet" &&
          (effectiveIsRejected || effectiveResolvedPrediction.isHolding) &&
          mlPredictionMissesRef.current >= ALPHABET_REPEAT_RELEASE_MISSES
        ) {
          alphabetCommitLockRef.current = createAlphabetCommitLock();
        }

        if (!effectiveIsRejected && effectiveResolvedPrediction.label) {
          lastStableLivePredictionRef.current = {
            prediction: effectiveResolvedPrediction,
            timestamp: now,
          };
        }

        setLiveModelPrediction(effectiveResolvedPrediction);
        setLiveModelError("");
        setLiveModelRejectReason(
          canHoldStablePrediction || stabilityOverridePrediction
            ? ""
            : effectiveRejectReason,
        );

        if (
          effectiveResolvedPrediction.label &&
          !effectiveIsRejected &&
          !(selectedRecognitionLevel === "alphabet" && effectiveResolvedPrediction.isHolding) &&
          (recognitionMode === "ml" ||
            !isRecognizedPhrase(recognizedText) ||
            effectiveResolvedPrediction.label === recognizedText)
        ) {
          commitRecognition(effectiveResolvedPrediction.label, {
            sourceMode: recognitionMode === "hybrid" ? "hybrid_ml" : "ml",
            confidence: effectiveResolvedPrediction.confidence,
            rawOutput: {
              ...prediction,
              topCandidates: effectiveTopCandidates,
              stability,
              quality,
              effectiveLabel: effectiveResolvedPrediction.label,
              effectiveConfidence: effectiveResolvedPrediction.confidence,
            },
          });
        }
      } catch (error) {
        mlPredictionWindowRef.current = [];
        mlPredictionMissesRef.current = 0;
        alphabetCommitLockRef.current = createAlphabetCommitLock();
        setLiveModelRejectReason("");
        setLiveModelError(
          error.message || "Не удалось получить live ML prediction",
        );
      } finally {
        setIsLiveModelPredicting(false);
      }
    },
    [
      commitRecognition,
      isLiveModelReady,
      liveModelMetadata,
      reliableSignLabelKeys,
      recognitionMode,
      recognizedText,
      selectedRecognitionLevel,
    ],
  );

  const speakText = useCallback(
    (textToSpeak = recognizedText) => {
      speakRecognition(textToSpeak, { force: true });
    },
    [recognizedText, speakRecognition],
  );

  const handlePhraseCardClick = useCallback(
    (phrase) => {
      commitRecognition(phrase.label, {
        sourceMode: "phrase_card",
        phraseId: phrase.id,
        forceSpeak: true,
      });
    },
    [commitRecognition],
  );

  useEffect(() => {
    mlPredictionWindowRef.current = [];
    mlPredictionMissesRef.current = 0;
    lastStableLivePredictionRef.current = null;
    alphabetCommitLockRef.current = createAlphabetCommitLock();
    setLiveModelPrediction(null);
    setLiveModelRejectReason("");
    setLiveModelError((currentError) => {
      if (!isLiveModelReady) {
        return "ML-модель ещё не обучена";
      }

      if (recognitionMode === "rules") {
        return "Live ML остановлен в режиме правил";
      }

      return currentError === "Live ML остановлен в режиме правил"
        ? ""
        : currentError;
    });
  }, [isLiveModelReady, recognitionMode]);

  const liveQualityHints = useMemo(
    () =>
      buildLiveQualityHints({
        gestureDebug,
        recognitionMode,
        liveModelRejectReason,
        liveModelError,
      }),
    [gestureDebug, liveModelError, liveModelRejectReason, recognitionMode],
  );
  const livePreviewTitle =
    selectedRecognitionLevel === "alphabet"
      ? "Текущий символ"
      : selectedRecognitionLevel === "sign"
        ? "Текущий знак"
        : "Текущий жест";
  const confirmedResultTitle =
    selectedRecognitionLevel === "alphabet"
      ? "Подтверждённый символ"
      : selectedRecognitionLevel === "sign"
        ? "Подтверждённый знак"
        : "Подтверждённая фраза";
  const usesRulePreview =
    selectedRecognitionLevel === "phrase" && recognitionMode === "rules";
  const confirmedResultDisplayTitle =
    selectedRecognitionLevel === "alphabet"
      ? "Собранное слово"
      : confirmedResultTitle;
  const livePreviewLabel = usesRulePreview
    ? liveGesture
    : liveModelPrediction?.label ||
      (selectedRecognitionLevel === "alphabet"
        ? "Покажите букву или цифру"
        : selectedRecognitionLevel === "sign"
          ? "Покажите знак"
          : "Ожидание распознавания...");

  return (
    <div className="space-y-4">
      <div className="app-surface glow-ring overflow-hidden rounded-[32px]">
        <div className="relative">
          <CameraView
            onGestureDetected={handleGestureDetected}
            onLiveGestureChange={setLiveGesture}
            onPredictionChange={handlePredictionChange}
            onSequencePredict={handleSequencePredict}
            recognitionLevel={selectedRecognitionLevel}
            enableLiveSequencePredict={
              recognitionMode !== "rules" && isLiveModelReady
            }
          />

          <div className="pointer-events-none absolute inset-x-4 bottom-24 rounded-[22px] bg-black/55 p-3 ring-1 ring-white/10">
            <div className="text-[11px] uppercase tracking-[0.24em] text-slate-300">
              {livePreviewTitle}
            </div>
            <div
              className={`mt-2 font-bold text-white ${
                settings.largeTextEnabled ? "text-xl" : "text-lg"
              }`}
            >
              {livePreviewLabel}
            </div>
          </div>

          <div className="pointer-events-none absolute inset-x-4 bottom-4 rounded-[26px] bg-slate-950/85 p-4 ring-1 ring-white/10">
            <div className="text-[11px] uppercase tracking-[0.24em] text-cyan-300/70">
              {confirmedResultDisplayTitle}
            </div>
            <div
              className={`mt-2 font-black leading-tight text-white ${
                settings.largeTextEnabled ? "text-4xl" : "text-3xl"
              }`}
            >
              {recognizedText}
            </div>
          </div>
        </div>
      </div>

      <GlassCard>
        <div className="text-sm font-semibold text-white">Что распознавать</div>
        <div className="mt-3 flex flex-wrap gap-2">
          {CAMERA_OUTPUT_MODE_OPTIONS.map((option) => (
            <button
              key={option.id}
              type="button"
              onClick={() => setSelectedRecognitionLevel(option.id)}
              className={`rounded-full px-4 py-2 text-sm font-bold transition ${
                selectedRecognitionLevel === option.id
                  ? "bg-cyan-400 text-slate-950"
                  : "border border-white/10 bg-white/5 text-slate-300 hover:bg-white/10"
              }`}
            >
              {option.label}
            </button>
          ))}
        </div>

        <div className="mt-3 text-sm leading-6 text-slate-300">
          Сейчас камера работает в режиме{" "}
          <span className="font-semibold text-white">
            {getRecognitionLevelLabel(selectedRecognitionLevel)}
          </span>
          .
        </div>
      </GlassCard>

      <div className="grid grid-cols-2 gap-3">
        <PrimaryButton
          onClick={() => speakText()}
          disabled={
            !isSpeakableText(recognizedText, {
              recognitionLevel: selectedRecognitionLevel,
            })
          }
        >
          Озвучить
        </PrimaryButton>
        <SecondaryButton
          onClick={() => {
            cancelSpeech();
            recognizedTextRef.current = EMPTY_RECOGNIZED_TEXT;
            setRecognizedText(EMPTY_RECOGNIZED_TEXT);
            setLiveGesture("Жест не найден");
            setGestureConfidence(0);
            setGestureDebug(null);
            setLiveModelPrediction(null);
            setLiveModelRejectReason("");
            mlPredictionWindowRef.current = [];
            mlPredictionMissesRef.current = 0;
            lastStableLivePredictionRef.current = null;
            alphabetCommitLockRef.current = createAlphabetCommitLock();
            setLiveModelError(
              !isLiveModelReady
                ? "ML-модель ещё не обучена"
                : recognitionMode === "rules"
                  ? "Live ML остановлен в режиме правил"
                  : "",
            );
            lastSpokenRecognitionRef.current = {
              key: "",
              timestamp: 0,
            };
            lastRecognitionKeyRef.current = {
              key: "",
              timestamp: 0,
            };
          }}
        >
          Очистить
        </SecondaryButton>
      </div>

      {!settings.autoSpeakEnabled ? (
        <div className="rounded-[18px] border border-amber-400/20 bg-amber-500/10 px-4 py-3 text-sm leading-6 text-amber-100">
          В приложении открой Настройки и проверь, что включено “Авто-озвучка”.
        </div>
      ) : null}

      {selectedRecognitionLevel === "alphabet" ? (
        <GlassCard>
          <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
            <div>
              <div className="text-sm font-semibold text-white">Сборка слова</div>
              <div className="mt-1 text-sm leading-6 text-slate-300">
                Каждая подтверждённая буква добавляется в строку. Если нужно,
                можно поставить пробел или удалить последний символ.
              </div>
            </div>
            <div className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-semibold text-cyan-200">
              live ввод
            </div>
          </div>

          <div className="mt-4 rounded-[22px] bg-black/20 p-4 ring-1 ring-white/10">
            <div className="text-[11px] uppercase tracking-[0.24em] text-cyan-300/70">
              Текущий текст
            </div>
            <div
              className={`mt-2 break-words font-black text-white ${
                settings.largeTextEnabled ? "text-3xl" : "text-2xl"
              }`}
            >
              {recognizedText}
            </div>
          </div>

          <div className="mt-4 grid grid-cols-2 gap-3">
            <button
              type="button"
              onClick={handleAlphabetSpace}
              disabled={!hasAlphabetText}
              className="rounded-[18px] border border-white/10 bg-white/5 px-4 py-3 text-sm font-bold text-white transition hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-40"
            >
              Пробел
            </button>
            <button
              type="button"
              onClick={handleAlphabetDelete}
              disabled={!hasAlphabetText}
              className="rounded-[18px] border border-white/10 bg-white/5 px-4 py-3 text-sm font-bold text-white transition hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-40"
            >
              Удалить
            </button>
          </div>
        </GlassCard>
      ) : null}

      {(liveModelRejectReason || liveModelError || liveQualityHints.length) ? (
        <GlassCard className="bg-amber-500/10 text-amber-100 ring-1 ring-amber-400/20">
          <div className="text-sm font-semibold text-white">Подсказка</div>
          <div className="mt-2 text-sm leading-6 text-amber-100/90">
            {liveModelRejectReason || liveModelError || liveQualityHints[0]}
          </div>
          {liveQualityHints.length > 1 ? (
            <div className="mt-3 space-y-2 text-sm leading-6">
              {liveQualityHints.slice(1).map((hint) => (
                <div key={hint} className="rounded-[16px] bg-black/20 px-3 py-2">
                  {hint}
                </div>
              ))}
            </div>
          ) : null}
        </GlassCard>
      ) : null}

      {selectedRecognitionLevel === "phrase" ? (
        <GlassCard>
          <div>
            <div className="text-sm font-semibold text-white">Быстрые фразы</div>
            <div className="mt-1 text-sm text-slate-300">
              Если сейчас удобнее выбрать готовую фразу, нажмите её ниже.
            </div>
          </div>

          {contentError ? (
            <div className="mt-4 rounded-[20px] border border-red-400/20 bg-red-500/10 p-4 text-sm text-red-200">
              {contentError}
            </div>
          ) : isContentLoading ? (
            <div className="mt-4 rounded-[20px] bg-white/5 p-4 text-sm text-slate-300">
              Загружаю фразы...
            </div>
          ) : (
            <>
              <div className="mt-3 flex flex-wrap gap-2">
                {categories.map((category) => (
                  <button
                    key={category}
                    type="button"
                    onClick={() => setSelectedCategory(category)}
                    className={`rounded-full px-4 py-2 text-sm font-bold transition ${
                      selectedCategory === category
                        ? "bg-cyan-400 text-slate-950"
                        : "border border-white/10 bg-white/5 text-slate-300 hover:bg-white/10"
                    }`}
                  >
                    {category}
                  </button>
                ))}
              </div>

              <div className="mt-4 grid grid-cols-2 gap-3">
                {filteredCards.map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => handlePhraseCardClick(item)}
                    className="rounded-[22px] border border-white/10 bg-black/20 p-4 text-left transition hover:bg-black/30"
                  >
                    <div className="text-xs uppercase tracking-[0.18em] text-cyan-300/70">
                      {item.category}
                    </div>
                    <div className="mt-2 text-base font-bold leading-snug text-white">
                      {item.label}
                    </div>
                  </button>
                ))}
              </div>
            </>
          )}
        </GlassCard>
      ) : null}
    </div>
  );
}
