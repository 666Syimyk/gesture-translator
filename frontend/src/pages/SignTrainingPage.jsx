import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { FilesetResolver, HolisticLandmarker } from "@mediapipe/tasks-vision";
import {
  createTrainingVideo,
  extractTrainingVideoLandmarks,
  updateTrainingVideoReview,
} from "../api/trainingVideosApi";
import { prepareDatasetExport } from "../api/datasetApi";
import { trainLatestModel } from "../api/mlApi";
import {
  GlassCard,
  PrimaryButton,
  SecondaryButton,
  StatusBadge,
  TitleBlock,
} from "../components/PremiumUI";
import {
  DEFAULT_RECOGNITION_LEVEL,
  RECOGNITION_LEVEL_OPTIONS,
} from "../recognitionLevels";

const TASKS_VISION_VERSION = "0.10.32";
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

const AUTO_REVIEW_NOTES =
  "Автоодобрено после записи через упрощённый экран обучения.";
const ALPHABET_PRIORITY_LABEL_KEYS = [
  "alphabet::LETTER_D",
  "alphabet::LETTER_E",
  "alphabet::LETTER_Z",
  "alphabet::LETTER_I_SHORT",
  "alphabet::LETTER_P",
  "alphabet::LETTER_F",
  "alphabet::LETTER_CH",
  "alphabet::LETTER_SHCH",
  "alphabet::LETTER_SOFT_SIGN",
  "alphabet::LETTER_EH",
  "alphabet::LETTER_YU",
  "alphabet::LETTER_YA",
  "alphabet::LETTER_YO",
];
const ALPHABET_PRIORITY_ENTRIES = [
  "Д",
  "Е",
  "З",
  "Й",
  "П",
  "Ф",
  "Ч",
  "Щ",
  "Ь",
  "Э",
  "Ю",
  "Я",
  "Ё",
];
const PHRASE_PRIORITY_ENTRIES = [
  "Мужчина",
  "Женщина",
  "Нет",
  "Да",
  "Солнце",
  "Дружба",
  "Привет",
  "Пока",
  "Дом",
  "Здравствуйте",
  "Подождите",
  "Повторите",
  "Мне нужна помощь",
  "Мне нужна вода",
  "Мне нужна еда",
  "Мне плохо",
  "Позвоните врачу",
  "Я не понимаю",
  "Говорите медленнее",
  "Спасибо",
  "Стоп",
];
const PHRASE_PRIORITY_LABEL_KEYS = PHRASE_PRIORITY_ENTRIES.map(
  (label) => `phrase::${label}`,
);
const AUTO_MODEL_CONFIGS = {
  alphabet: {
    recognitionLevel: "alphabet",
    modelType: "tcn",
    epochs: 36,
    maxSequenceLength: 56,
    hiddenSize: 160,
    classBalance: "both",
    classWeightPower: 0.5,
    focusLabelKeys: ALPHABET_PRIORITY_LABEL_KEYS,
    focusWeightMultiplier: 2.4,
    confidenceThreshold: 0.48,
  },
  sign: {
    recognitionLevel: "sign",
    modelType: "tcn",
    epochs: 32,
    maxSequenceLength: 48,
    hiddenSize: 192,
    confidenceThreshold: 0.35,
  },
  phrase: {
    recognitionLevel: "phrase",
    modelType: "tcn",
    epochs: 48,
    maxSequenceLength: 72,
    hiddenSize: 224,
    classBalance: "both",
    classWeightPower: 0.45,
    focusLabelKeys: PHRASE_PRIORITY_LABEL_KEYS,
    focusWeightMultiplier: 2.2,
    confidenceThreshold: 0.56,
  },
};
const AUTO_TRAIN_REQUIREMENTS = {
  alphabet: { minSampleCount: 120, minClassCount: 10 },
  sign: { minSampleCount: 80, minClassCount: 6 },
  phrase: { minSampleCount: 60, minClassCount: 5 },
};
const ALPHABET_FOCUS_ENTRIES = [
  ...ALPHABET_PRIORITY_ENTRIES,
  "5",
  "8",
  "4",
  "7",
  "К",
  "0",
  "6",
];
const PHRASE_FOCUS_ENTRIES = [
  ...PHRASE_PRIORITY_ENTRIES,
  "Где туалет?",
  "Подойдите сюда",
  "Больно",
  "Я хочу домой",
  "Мне холодно",
  "Мне жарко",
];
const MOTION_ALPHABET_UNIT_CODES = new Set([
  "LETTER_YO",
  "LETTER_Z",
  "LETTER_I_SHORT",
  "LETTER_SHCH",
  "LETTER_HARD_SIGN",
  "LETTER_EH",
]);
const URL_PATTERN = /(https?:\/\/[^\s]+)/g;

function renderTextWithLinks(text) {
  const normalizedText = String(text ?? "");

  return normalizedText.split(URL_PATTERN).map((part, index) => {
    if (!part) {
      return null;
    }

    if (URL_PATTERN.test(part)) {
      URL_PATTERN.lastIndex = 0;
      return (
        <a
          key={`${part}-${index}`}
          href={part}
          target="_blank"
          rel="noreferrer"
          className="font-semibold text-cyan-200 underline decoration-cyan-300/50 underline-offset-4 hover:text-cyan-100"
        >
          {part}
        </a>
      );
    }

    URL_PATTERN.lastIndex = 0;
    return part;
  });
}

function canAutoRetrain(datasetSummary, recognitionLevel) {
  const requirements =
    AUTO_TRAIN_REQUIREMENTS[recognitionLevel] ?? AUTO_TRAIN_REQUIREMENTS.alphabet;

  return (
    Number(datasetSummary?.sample_count ?? 0) >= requirements.minSampleCount &&
    Number(datasetSummary?.class_count ?? 0) >= requirements.minClassCount
  );
}

function getLevelUiCopy(recognitionLevel) {
  if (recognitionLevel === "phrase") {
    return {
      eyebrow: "Фразы",
      title: "Покажите фразу и сохраните её",
      subtitle:
        "Выберите фразу, покажите её в камеру и сохраните запись. Так мы будем усиливать модель фраз на ваших реальных примерах.",
      entryTitle: "Фразы",
      entryEmpty: "В библиотеке пока нет фраз.",
      promptTitle: "Какая это фраза?",
      promptHint:
        "Выберите фразу из списка выше. При желании можно ввести её вручную.",
      placeholder: "Например: Мне нужна помощь",
      statusText:
        "Покажите фразу в камеру, запишите её и сохраните пример для phrase-модели.",
      saveIdle: "Сохранить фразу",
      saveBusy: "Сохраняю и обновляю фразы...",
      unitLabel: "фразу",
      unitGenitivePlural: "фраз",
    };
  }

  if (recognitionLevel === "sign") {
    return {
      eyebrow: "Знаки",
      title: "Покажите знак и сохраните его",
      subtitle:
        "Выберите знак, покажите его в камеру и сохраните пример. Так мы будем усиливать модель знаков на ваших реальных жестах.",
      entryTitle: "Знаки",
      entryEmpty: "В библиотеке пока нет знаков.",
      promptTitle: "Какой это знак?",
      promptHint:
        "Выберите знак из списка выше. При желании можно ввести его вручную.",
      placeholder: "Например: вода",
      statusText:
        "Покажите знак в камеру, запишите его и сохраните пример для sign-модели.",
      saveIdle: "Сохранить знак",
      saveBusy: "Сохраняю и обновляю знаки...",
      unitLabel: "знак",
      unitGenitivePlural: "знаков",
    };
  }

  return {
    eyebrow: "Буквы и цифры",
    title: "Покажите символ и сохраните его",
    subtitle:
      "Вы показываете букву или цифру в камеру, выбираете символ и нажимаете сохранить. Когда данных станет достаточно, программа безопасно обновит alphabet-модель.",
    entryTitle: "Буквы и цифры",
    entryEmpty: "В библиотеке пока нет символов.",
    promptTitle: "Какой это символ?",
    promptHint: "Введите букву или цифру, которую вы показываете жестом.",
    placeholder: "Например: А или 1",
    statusText:
      "Покажите букву или цифру в камеру, запишите символ и нажмите сохранить.",
    saveIdle: "Сохранить символ",
    saveBusy: "Сохраняю и обновляю символы...",
    unitLabel: "символ",
    unitGenitivePlural: "символов",
  };
}

function normalizeEntryValue(value, recognitionLevel) {
  const trimmed = String(value ?? "").trim().replace(/\s+/g, " ");

  if (recognitionLevel === "alphabet") {
    return trimmed.toUpperCase().slice(0, 1);
  }

  return trimmed;
}

function normalizeEntryForMatch(value, recognitionLevel) {
  const normalized = normalizeEntryValue(value, recognitionLevel);
  return recognitionLevel === "alphabet"
    ? normalized
    : normalized.toLocaleLowerCase("ru-RU");
}

function createFocusEntryOrder(entries, recognitionLevel) {
  return new Map(
    entries.map((entry, index) => [
      normalizeEntryForMatch(entry, recognitionLevel),
      index,
    ]),
  );
}

const ALPHABET_FOCUS_ENTRY_ORDER = createFocusEntryOrder(
  ALPHABET_FOCUS_ENTRIES,
  "alphabet",
);
const PHRASE_FOCUS_ENTRY_ORDER = createFocusEntryOrder(
  PHRASE_FOCUS_ENTRIES,
  "phrase",
);

function slugifyEntryLabel(label) {
  return String(label ?? "")
    .trim()
    .toLowerCase()
    .replace(/\s+/g, "-")
    .replace(/[^a-z0-9а-яё_-]+/gi, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "") || "sample";
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
      context.arc(helperX * width, helperY * height, 2.5, 0, Math.PI * 2);
      context.fillStyle = color;
      context.globalAlpha = 0.7;
      context.fill();
      context.globalAlpha = 1;
    }
  }

  for (const point of landmarks) {
    context.beginPath();
    context.arc(point.x * width, point.y * height, 5, 0, Math.PI * 2);
    context.fillStyle = color;
    context.fill();
  }
}

function pickRecorderMimeType() {
  if (typeof MediaRecorder === "undefined") {
    return "";
  }

  const mimeTypes = [
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
  ];

  return (
    mimeTypes.find((mimeType) => {
      if (typeof MediaRecorder.isTypeSupported !== "function") {
        return mimeType === "video/webm";
      }

      return MediaRecorder.isTypeSupported(mimeType);
    }) ?? ""
  );
}

function formatDuration(durationMs) {
  const totalSeconds = Math.max(0, Math.round(durationMs / 1000));
  const minutes = String(Math.floor(totalSeconds / 60)).padStart(2, "0");
  const seconds = String(totalSeconds % 60).padStart(2, "0");
  return `${minutes}:${seconds}`;
}

function buildTrackingStatus({ hasLeftHand, hasRightHand, hasFace, hasPose }) {
  if (!hasLeftHand && !hasRightHand) {
    return "Покажите руку ближе к камере, чтобы видеть точки.";
  }

  const handsLabel =
    hasLeftHand && hasRightHand
      ? "Обе руки"
      : hasRightHand
        ? "Правая рука"
        : "Левая рука";

  return `${handsLabel} видна. Лицо: ${hasFace ? "да" : "нет"}. Корпус: ${
    hasPose ? "да" : "нет"
  }.`;
}

function getEntryFocusRank(entry, recognitionLevel) {
  const normalizedLabel = normalizeEntryForMatch(entry?.label, recognitionLevel);

  if (recognitionLevel === "alphabet") {
    return (
      ALPHABET_FOCUS_ENTRY_ORDER.get(normalizedLabel) ?? Number.POSITIVE_INFINITY
    );
  }

  if (recognitionLevel === "phrase") {
    return PHRASE_FOCUS_ENTRY_ORDER.get(normalizedLabel) ?? Number.POSITIVE_INFINITY;
  }

  return Number.POSITIVE_INFINITY;
}

function sortEntries(left, right, recognitionLevel = DEFAULT_RECOGNITION_LEVEL) {
  const leftFocusRank = getEntryFocusRank(left, recognitionLevel);
  const rightFocusRank = getEntryFocusRank(right, recognitionLevel);

  if (leftFocusRank !== rightFocusRank) {
    return leftFocusRank - rightFocusRank;
  }

  if ((left.sortOrder ?? 0) !== (right.sortOrder ?? 0)) {
    return (left.sortOrder ?? 0) - (right.sortOrder ?? 0);
  }

  return left.label.localeCompare(right.label, "ru");
}

function isMotionAlphabetEntry(entry) {
  return MOTION_ALPHABET_UNIT_CODES.has(String(entry?.unitCode ?? "").trim());
}

export default function SignTrainingPage({
  currentUser,
  settings,
  phraseLibrary,
  isContentLoading,
  contentError,
}) {
  const liveVideoRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const streamRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const holisticLandmarkerRef = useRef(null);
  const animationRef = useRef(null);
  const lastVideoTimeRef = useRef(-1);
  const chunksRef = useRef([]);
  const recordingStartedAtRef = useRef(0);
  const recordingTimerRef = useRef(null);

  const [selectedRecognitionLevel, setSelectedRecognitionLevel] = useState(
    DEFAULT_RECOGNITION_LEVEL,
  );
  const [entryInput, setEntryInput] = useState("");
  const [cameraError, setCameraError] = useState("");
  const [trackingStatus, setTrackingStatus] = useState(
    "Покажите руку в кадре, чтобы видеть точки.",
  );
  const [recorderStatus, setRecorderStatus] = useState(
    getLevelUiCopy(DEFAULT_RECOGNITION_LEVEL).statusText,
  );
  const [isRecording, setIsRecording] = useState(false);
  const [recordingDurationMs, setRecordingDurationMs] = useState(0);
  const [recordedSample, setRecordedSample] = useState(null);
  const [isSaving, setIsSaving] = useState(false);
  const [saveError, setSaveError] = useState("");
  const [saveMessage, setSaveMessage] = useState("");

  const levelUiCopy = useMemo(
    () => getLevelUiCopy(selectedRecognitionLevel),
    [selectedRecognitionLevel],
  );

  const activeEntries = useMemo(() => {
    return [...phraseLibrary]
      .filter((item) => item.recognitionLevel === selectedRecognitionLevel)
      .sort((left, right) =>
        sortEntries(left, right, selectedRecognitionLevel),
      );
  }, [phraseLibrary, selectedRecognitionLevel]);

  const normalizedEntryValue = useMemo(
    () => normalizeEntryValue(entryInput, selectedRecognitionLevel),
    [entryInput, selectedRecognitionLevel],
  );

  const selectedEntry = useMemo(() => {
    if (!normalizedEntryValue) {
      return null;
    }

    const matchValue = normalizeEntryForMatch(
      normalizedEntryValue,
      selectedRecognitionLevel,
    );

    return (
      activeEntries.find(
        (item) =>
          normalizeEntryForMatch(item.label, selectedRecognitionLevel) ===
          matchValue,
      ) ?? null
    );
  }, [activeEntries, normalizedEntryValue, selectedRecognitionLevel]);
  const selectedEntryIsMotion = useMemo(
    () =>
      selectedRecognitionLevel === "alphabet" && isMotionAlphabetEntry(selectedEntry),
    [selectedEntry, selectedRecognitionLevel],
  );

  useEffect(() => {
    setRecordedSample(null);
    setRecordingDurationMs(0);
    setSaveError("");
    setSaveMessage("");
    setRecorderStatus(levelUiCopy.statusText);
  }, [levelUiCopy.statusText, selectedRecognitionLevel]);

  useEffect(() => {
    const hasCurrentSelection =
      entryInput &&
      activeEntries.some(
        (item) =>
          normalizeEntryForMatch(item.label, selectedRecognitionLevel) ===
          normalizeEntryForMatch(entryInput, selectedRecognitionLevel),
      );

    if (!hasCurrentSelection) {
      setEntryInput(activeEntries[0]?.label ?? "");
    }
  }, [activeEntries, entryInput, selectedRecognitionLevel]);

  const stopRecordingTimer = useCallback(() => {
    if (recordingTimerRef.current) {
      window.clearInterval(recordingTimerRef.current);
      recordingTimerRef.current = null;
    }
  }, []);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }
  }, []);

  useEffect(() => {
    let isMounted = true;

    function clearOverlay() {
      const canvas = overlayCanvasRef.current;

      if (!canvas) {
        return;
      }

      const context = canvas.getContext("2d");

      if (!context) {
        return;
      }

      context.clearRect(0, 0, canvas.width, canvas.height);
    }

    function drawResults(results) {
      const video = liveVideoRef.current;
      const canvas = overlayCanvasRef.current;

      if (!video || !canvas) {
        return;
      }

      const width = video.videoWidth || 720;
      const height = video.videoHeight || 960;
      const context = canvas.getContext("2d");

      if (!context) {
        return;
      }

      canvas.width = width;
      canvas.height = height;
      context.clearRect(0, 0, width, height);

      const leftHand = results?.leftHandLandmarks?.[0] ?? [];
      const rightHand = results?.rightHandLandmarks?.[0] ?? [];
      const hasLeftHand = leftHand.length > 0;
      const hasRightHand = rightHand.length > 0;
      const hasFace = (results?.faceLandmarks?.[0] ?? []).length > 0;
      const hasPose = (results?.poseLandmarks?.[0] ?? []).length > 0;

      if (!hasLeftHand && !hasRightHand) {
        setTrackingStatus("Точки не видны. Покажите руку ближе к камере.");
        return;
      }

      if (hasLeftHand) {
        drawHand(context, leftHand, width, height, "#22d3ee");
      }

      if (hasRightHand) {
        drawHand(context, rightHand, width, height, "#38bdf8");
      }

      setTrackingStatus(
        buildTrackingStatus({
          hasLeftHand,
          hasRightHand,
          hasFace,
          hasPose,
        }),
      );
    }

    function startTrackingLoop() {
      const video = liveVideoRef.current;
      const holisticLandmarker = holisticLandmarkerRef.current;

      if (!isMounted) {
        return;
      }

      if (!video || !holisticLandmarker) {
        animationRef.current = requestAnimationFrame(startTrackingLoop);
        return;
      }

      if (video.readyState >= 2 && video.currentTime !== lastVideoTimeRef.current) {
        lastVideoTimeRef.current = video.currentTime;
        const results = holisticLandmarker.detectForVideo(video, performance.now());
        drawResults(results);
      }

      animationRef.current = requestAnimationFrame(startTrackingLoop);
    }

    async function startCamera() {
      if (
        typeof navigator === "undefined" ||
        !navigator.mediaDevices?.getUserMedia
      ) {
        setCameraError("Браузер не даёт доступ к камере.");
        return;
      }

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: "user",
            width: { ideal: 720 },
            height: { ideal: 960 },
          },
          audio: false,
        });

        if (!isMounted) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }

        streamRef.current = stream;

        if (liveVideoRef.current) {
          liveVideoRef.current.srcObject = stream;
          await liveVideoRef.current.play().catch(() => {});
        }

        setCameraError("");
      } catch {
        setCameraError(
          "Не удалось открыть камеру. Разрешите доступ в браузере.",
        );
      }
    }

    async function setupHolisticLandmarker() {
      const vision = await FilesetResolver.forVisionTasks(
        `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VISION_VERSION}/wasm`,
      );

      if (!isMounted) {
        return;
      }

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
          minHandLandmarksConfidence: 0.7,
        },
      );
    }

    async function init() {
      try {
        await startCamera();
        await setupHolisticLandmarker();

        if (!isMounted) {
          return;
        }

        startTrackingLoop();
      } catch {
        if (!isMounted) {
          return;
        }

        setCameraError("Не удалось включить Holistic поверх камеры.");
        setTrackingStatus("Точки недоступны.");
      }
    }

    init();

    return () => {
      isMounted = false;
      stopRecordingTimer();
      clearOverlay();

      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
        mediaRecorderRef.current.stop();
      }

      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }

      if (holisticLandmarkerRef.current) {
        holisticLandmarkerRef.current.close();
        holisticLandmarkerRef.current = null;
      }

      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, [stopRecordingTimer]);

  const startRecording = useCallback(() => {
    if (!streamRef.current) {
      setSaveMessage("");
      setSaveError("Сначала дайте доступ к камере.");
      return;
    }

    if (!selectedEntry) {
      setSaveMessage("");
      setSaveError(`Сначала выберите ${levelUiCopy.unitLabel}.`);
      return;
    }

    const mimeType = pickRecorderMimeType();

    try {
      chunksRef.current = [];
      const recorder = mimeType
        ? new MediaRecorder(streamRef.current, { mimeType })
        : new MediaRecorder(streamRef.current);

      mediaRecorderRef.current = recorder;
      recordingStartedAtRef.current = Date.now();
      setRecordingDurationMs(0);
      setRecordedSample(null);
      setSaveError("");
      setSaveMessage("");

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.onstop = () => {
        stopRecordingTimer();
        const blob = new Blob(chunksRef.current, {
          type: recorder.mimeType || "video/webm",
        });
        const durationMs = Date.now() - recordingStartedAtRef.current;

        if (blob.size > 0) {
          setRecordedSample({ blob, durationMs });
          setRecorderStatus("Пример записан. Теперь нажмите сохранить.");
        } else {
          setRecorderStatus("Видео не получилось. Попробуйте ещё раз.");
        }

        setIsRecording(false);
      };

      recorder.start();
      setIsRecording(true);
      setRecorderStatus(`Идёт запись: ${selectedEntry.label}`);

      recordingTimerRef.current = window.setInterval(() => {
        setRecordingDurationMs(Date.now() - recordingStartedAtRef.current);
      }, 250);
    } catch {
      setSaveError("Не удалось начать запись.");
    }
  }, [levelUiCopy.unitLabel, selectedEntry, stopRecordingTimer]);

  const saveSample = useCallback(async () => {
    if (!selectedEntry) {
      setSaveMessage("");
      setSaveError(`Сначала выберите ${levelUiCopy.unitLabel}.`);
      return;
    }

    if (!recordedSample?.blob) {
      setSaveMessage("");
      setSaveError(
        `Сначала запишите ${levelUiCopy.unitLabel}, потом сохраняйте.`,
      );
      return;
    }

    setIsSaving(true);
    setSaveError("");
    setSaveMessage("");

    try {
      const formData = new FormData();
      formData.append(
        "video",
        recordedSample.blob,
        `${selectedRecognitionLevel}-${slugifyEntryLabel(selectedEntry.label)}-${Date.now()}.webm`,
      );
      formData.append("phraseId", String(selectedEntry.id));
      formData.append("userEmail", currentUser?.email ?? "");
      formData.append("signLanguage", settings.signLanguage ?? "rsl");
      formData.append("durationMs", String(recordedSample.durationMs));
      formData.append(
        "signerLabel",
        currentUser?.displayName || currentUser?.email || "default-signer",
      );
      formData.append("status", "draft");

      const createdVideo = await createTrainingVideo(formData);
      let landmarksReady = false;
      let reviewApproved = false;
      let datasetPrepared = false;
      let datasetSummary = null;
      let modelUpdated = false;

      try {
        await extractTrainingVideoLandmarks(createdVideo.id);
        landmarksReady = true;
      } catch {
        landmarksReady = false;
      }

      if (landmarksReady) {
        try {
          await updateTrainingVideoReview(createdVideo.id, {
            reviewStatus: "approved",
            qualityScore: 3,
            reviewNotes: AUTO_REVIEW_NOTES,
          });
          reviewApproved = true;
        } catch {
          reviewApproved = false;
        }
      }

      if (landmarksReady && reviewApproved) {
        try {
          datasetSummary = await prepareDatasetExport({
            signLanguage: settings.signLanguage ?? "rsl",
            recognitionLevel: selectedRecognitionLevel,
          });
          datasetPrepared = true;
        } catch {
          datasetPrepared = false;
        }
      }

      if (datasetPrepared && canAutoRetrain(datasetSummary, selectedRecognitionLevel)) {
        try {
          await trainLatestModel(
            AUTO_MODEL_CONFIGS[selectedRecognitionLevel] ??
              AUTO_MODEL_CONFIGS.alphabet,
          );
          modelUpdated = true;
        } catch {
          modelUpdated = false;
        }
      }

      const requirements =
        AUTO_TRAIN_REQUIREMENTS[selectedRecognitionLevel] ??
        AUTO_TRAIN_REQUIREMENTS.alphabet;

      if (modelUpdated) {
        setSaveMessage(
          `${selectedEntry.label} сохранено. Landmarks готовы, пример одобрен, модель обновлена.`,
        );
      } else if (datasetPrepared) {
        setSaveMessage(
          `${selectedEntry.label} сохранено. Датасет обновлён, но модель пока не переобучилась автоматически: нужно минимум ${requirements.minSampleCount} примеров и ${requirements.minClassCount} разных ${levelUiCopy.unitGenitivePlural}.`,
        );
      } else if (reviewApproved) {
        setSaveMessage(
          `${selectedEntry.label} сохранено и одобрено. Модель можно обновить позже.`,
        );
      } else if (landmarksReady) {
        setSaveMessage(
          `${selectedEntry.label} сохранено. Landmarks готовы, но автоподготовка завершилась не полностью.`,
        );
      } else {
        setSaveMessage(
          `${selectedEntry.label} сохранено. Landmarks можно извлечь позже.`,
        );
      }

      setRecordedSample(null);
      setRecordingDurationMs(0);
      setRecorderStatus(levelUiCopy.statusText);
    } catch (error) {
      setSaveError(error.message || `Не удалось сохранить ${levelUiCopy.unitLabel}.`);
    } finally {
      setIsSaving(false);
    }
  }, [
    currentUser?.displayName,
    currentUser?.email,
    levelUiCopy.statusText,
    levelUiCopy.unitGenitivePlural,
    levelUiCopy.unitLabel,
    recordedSample,
    selectedEntry,
    selectedRecognitionLevel,
    settings.signLanguage,
  ]);

  return (
    <div className="space-y-4">
      <GlassCard className="bg-[linear-gradient(135deg,rgba(34,211,238,0.14),rgba(14,165,233,0.08),rgba(255,255,255,0.04))]">
        <TitleBlock
          eyebrow={levelUiCopy.eyebrow}
          title={levelUiCopy.title}
          subtitle={levelUiCopy.subtitle}
          action={<StatusBadge className="text-cyan-300">{(settings.signLanguage ?? "rsl").toUpperCase()}</StatusBadge>}
        />
      </GlassCard>

      <GlassCard>
        <div className="text-sm font-bold text-white">Что вы хотите учить сейчас</div>
        <div className="mt-3 flex flex-wrap gap-2">
          {RECOGNITION_LEVEL_OPTIONS.map((option) => {
            const isActive = option.id === selectedRecognitionLevel;

            return (
              <button
                key={option.id}
                type="button"
                onClick={() => setSelectedRecognitionLevel(option.id)}
                className={`rounded-full px-4 py-2 text-sm font-bold transition ${
                  isActive
                    ? "bg-cyan-400 text-slate-950"
                    : "bg-white/10 text-white hover:bg-white/15"
                }`}
              >
                {option.label}
              </button>
            );
          })}
        </div>
      </GlassCard>

      <GlassCard>
        <div className="text-sm font-bold text-white">{levelUiCopy.entryTitle}</div>
        <div className="mt-3 flex flex-wrap gap-2">
          {isContentLoading ? (
            <div className="text-sm text-slate-300">Загружаю элементы...</div>
          ) : contentError ? (
            <div className="text-sm text-red-200">{contentError}</div>
          ) : activeEntries.length === 0 ? (
            <div className="text-sm text-slate-300">{levelUiCopy.entryEmpty}</div>
          ) : (
            activeEntries.map((item) => {
              const isActive =
                selectedEntry?.id === item.id ||
                normalizeEntryForMatch(item.label, selectedRecognitionLevel) ===
                  normalizeEntryForMatch(normalizedEntryValue, selectedRecognitionLevel);

              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => {
                    setEntryInput(item.label);
                    setSaveError("");
                    setSaveMessage("");
                  }}
                  className={`rounded-full px-4 py-2 text-sm font-bold transition ${
                    isActive
                      ? "bg-cyan-400 text-slate-950"
                      : "bg-white/10 text-white hover:bg-white/15"
                  }`}
                >
                  {item.label}
                </button>
              );
            })
          )}
        </div>
      </GlassCard>

      <GlassCard>
        <div className="relative overflow-hidden rounded-[24px] bg-slate-950 ring-1 ring-white/10">
          {cameraError ? (
            <div className="flex aspect-[3/4] items-center justify-center px-6 text-center text-sm text-red-200">
              {cameraError}
            </div>
          ) : (
            <>
              <video
                ref={liveVideoRef}
                autoPlay
                muted
                playsInline
                className="aspect-[3/4] w-full object-cover"
              />
              <canvas
                ref={overlayCanvasRef}
                className="pointer-events-none absolute inset-0 h-full w-full"
              />
              <div className="absolute left-3 top-3 rounded-full bg-black/55 px-3 py-1 text-xs font-bold text-cyan-200">
                {trackingStatus}
              </div>
            </>
          )}
        </div>

        <div className="mt-4 rounded-[20px] bg-white/5 px-4 py-3 text-sm text-slate-200">
          <div className="text-xs uppercase tracking-[0.16em] text-cyan-300/70">
            Статус
          </div>
          <div className="mt-2 font-semibold text-white">{recorderStatus}</div>
          <div className="mt-2 text-slate-400">
            Длительность записи: {formatDuration(recordingDurationMs)}
          </div>
        </div>

        <div className="mt-4 grid grid-cols-2 gap-3">
          {isRecording ? (
            <SecondaryButton onClick={stopRecording} className="col-span-2">
              Остановить запись
            </SecondaryButton>
          ) : (
            <PrimaryButton onClick={startRecording} className="col-span-2">
              Начать запись
            </PrimaryButton>
          )}
        </div>

        <div className="mt-4">
          <div className="text-sm font-bold text-white">{levelUiCopy.promptTitle}</div>
          {selectedRecognitionLevel === "alphabet" ? (
            <input
              type="text"
              inputMode="text"
              maxLength={1}
              value={entryInput}
              onChange={(event) => {
                setEntryInput(
                  normalizeEntryValue(event.target.value, selectedRecognitionLevel),
                );
                setSaveError("");
                setSaveMessage("");
              }}
              placeholder={levelUiCopy.placeholder}
              className="mt-3 w-full rounded-[20px] border border-white/10 bg-slate-950 px-4 py-4 text-2xl font-black uppercase text-white outline-none placeholder:text-slate-500"
            />
          ) : (
            <textarea
              rows={3}
              value={entryInput}
              onChange={(event) => {
                setEntryInput(
                  normalizeEntryValue(event.target.value, selectedRecognitionLevel),
                );
                setSaveError("");
                setSaveMessage("");
              }}
              placeholder={levelUiCopy.placeholder}
              className="mt-3 w-full rounded-[20px] border border-white/10 bg-slate-950 px-4 py-4 text-base font-semibold text-white outline-none placeholder:text-slate-500"
            />
          )}
          <div className="mt-2 text-sm text-slate-400">
            {levelUiCopy.promptHint}
          </div>
          {selectedEntry ? (
            <div className="mt-3 rounded-[18px] bg-cyan-500/10 px-4 py-3 text-sm text-cyan-100 ring-1 ring-cyan-400/20">
              Выбрано: <span className="font-bold">{selectedEntry.label}</span>
              {selectedEntryIsMotion ? (
                <span className="ml-2 inline-flex rounded-full bg-cyan-400/15 px-2 py-1 text-[11px] font-semibold uppercase tracking-[0.14em] text-cyan-200">
                  С движением
                </span>
              ) : null}
            </div>
          ) : normalizedEntryValue ? (
            <div className="mt-3 rounded-[18px] bg-amber-500/10 px-4 py-3 text-sm text-amber-100 ring-1 ring-amber-400/20">
              Такого элемента нет в текущем списке. Выберите его из набора выше или введите правильное значение.
            </div>
          ) : null}
          {selectedEntry &&
          (selectedEntry.description || selectedEntry.referenceNotes) ? (
            <div className="mt-3 rounded-[18px] bg-white/5 px-4 py-3 ring-1 ring-white/10">
              <div className="text-xs uppercase tracking-[0.16em] text-cyan-300/70">
                {selectedRecognitionLevel === "alphabet" ? "Как показать" : "Подсказка"}
              </div>
              {selectedEntry.description ? (
                <div className="mt-2 text-sm leading-6 text-white">
                  {renderTextWithLinks(selectedEntry.description)}
                </div>
              ) : null}
              {selectedEntry.referenceNotes ? (
                <div className="mt-2 text-xs leading-5 text-slate-400">
                  {renderTextWithLinks(selectedEntry.referenceNotes)}
                </div>
              ) : null}
            </div>
          ) : null}
        </div>

        <div className="mt-4 grid grid-cols-1 gap-3">
          <PrimaryButton
            onClick={saveSample}
            disabled={isSaving || !recordedSample || !selectedEntry}
          >
            {isSaving ? levelUiCopy.saveBusy : levelUiCopy.saveIdle}
          </PrimaryButton>
        </div>

        {saveError ? (
          <div className="mt-4 rounded-[20px] bg-red-500/10 p-4 text-sm text-red-200 ring-1 ring-red-400/20">
            {saveError}
          </div>
        ) : null}

        {saveMessage ? (
          <div className="mt-4 rounded-[20px] bg-emerald-500/10 p-4 text-sm text-emerald-200 ring-1 ring-emerald-400/20">
            {saveMessage}
          </div>
        ) : null}
      </GlassCard>
    </div>
  );
}
