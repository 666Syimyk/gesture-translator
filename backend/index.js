import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import multer from "multer";
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import {
  cp,
  mkdir,
  readFile,
  rm,
  stat,
  unlink,
  writeFile,
} from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { WebSocketServer } from "ws";
import { pool } from "./db.js";
import {
  DEFAULT_PHRASE_CATEGORIES,
  DEFAULT_PHRASES,
  DEFAULT_SIGN_LANGUAGES,
  DEFAULT_USER,
  DEFAULT_USER_SETTINGS,
  FIRST_WAVE_DATASET_PHRASES,
} from "./data/defaultData.js";
import { GestureBridgeService } from "./services/gestureBridgeService.js";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PROJECT_ROOT_DIR = path.resolve(__dirname, "..");
const PYTHON_BIN = process.env.PYTHON_BIN || "python";
const UPLOADS_ROOT = path.join(__dirname, "uploads");
const TRAINING_UPLOADS_DIR = path.join(UPLOADS_ROOT, "training-videos");
const LANDMARK_UPLOADS_DIR = path.join(UPLOADS_ROOT, "landmarks");
const DATASET_EXPORTS_DIR = path.join(UPLOADS_ROOT, "datasets");
const TEMP_UPLOADS_DIR = path.join(UPLOADS_ROOT, "temp");
const LANDMARK_EXTRACT_SCRIPT = path.join(
  __dirname,
  "scripts",
  "extract_landmarks.py",
);
const ML_ROOT_DIR = path.join(__dirname, "ml");
const ML_ARTIFACTS_DIR = path.join(ML_ROOT_DIR, "artifacts");
const ML_CANDIDATES_DIR = path.join(ML_ARTIFACTS_DIR, "candidates");
const TRAIN_SEQUENCE_MODEL_SCRIPT = path.join(
  ML_ROOT_DIR,
  "train_sequence_model.py",
);
const EVALUATE_SEQUENCE_MODEL_SCRIPT = path.join(
  ML_ROOT_DIR,
  "evaluate_sequence_model.py",
);
const PREDICT_SEQUENCE_WORKER_SCRIPT = path.join(
  ML_ROOT_DIR,
  "predict_sequence_worker.py",
);
const gestureBridgeService = new GestureBridgeService({
  pythonBin: PYTHON_BIN,
  projectRoot: PROJECT_ROOT_DIR,
});
const predictionWorkers = new Map();
let predictionWorkerRequestSequence = 0;
const DATASET_QUALITY_THRESHOLDS = {
  minValidFrameRatio: 0.7,
  maxMissingHandRatio: 0.55,
  maxMissingFaceRatio: 0.4,
  maxMissingPoseRatio: 0.4,
};
const DATASET_COLLECTION_TARGETS = {
  minApprovedReadyTotal: 500,
  minSigners: 5,
  targetApprovedReadyPerPhrase: 20,
};
const MODEL_BENCHMARK_TYPES = ["baseline", "gru", "lstm", "tcn"];
const SUPPORTED_TRAINING_VIDEO_STATUSES = [
  "draft",
  "processing",
  "ready",
  "archived",
];
const SUPPORTED_DATASET_SPLITS = ["unassigned", "train", "val", "test"];
const MODEL_TRAINING_DEFAULTS = {
  alphabet: {
    baseline: {
      epochs: 72,
      maxSequenceLength: 72,
      hiddenSize: 256,
      confidenceThreshold: 0.5,
    },
    gru: {
      epochs: 120,
      maxSequenceLength: 72,
      hiddenSize: 384,
      confidenceThreshold: 0.5,
    },
    lstm: {
      epochs: 120,
      maxSequenceLength: 72,
      hiddenSize: 384,
      confidenceThreshold: 0.5,
    },
    tcn: {
      epochs: 120,
      maxSequenceLength: 72,
      hiddenSize: 384,
      confidenceThreshold: 0.5,
    },
  },
  sign: {
    baseline: {
      epochs: 24,
      maxSequenceLength: 56,
      hiddenSize: 128,
      confidenceThreshold: 0.35,
    },
    gru: {
      epochs: 40,
      maxSequenceLength: 48,
      hiddenSize: 192,
      confidenceThreshold: 0.35,
    },
    lstm: {
      epochs: 40,
      maxSequenceLength: 48,
      hiddenSize: 192,
      confidenceThreshold: 0.35,
    },
    tcn: {
      epochs: 40,
      maxSequenceLength: 48,
      hiddenSize: 192,
      confidenceThreshold: 0.35,
    },
  },
  phrase: {
    baseline: {
      epochs: 18,
      maxSequenceLength: 48,
      hiddenSize: 128,
      confidenceThreshold: 0.85,
    },
    gru: {
      epochs: 24,
      maxSequenceLength: 48,
      hiddenSize: 128,
      confidenceThreshold: 0.85,
    },
    lstm: {
      epochs: 24,
      maxSequenceLength: 48,
      hiddenSize: 128,
      confidenceThreshold: 0.85,
    },
    tcn: {
      epochs: 24,
      maxSequenceLength: 48,
      hiddenSize: 128,
      confidenceThreshold: 0.85,
    },
  },
};
const DEFAULT_RECOGNITION_LEVEL = "phrase";
const SUPPORTED_RECOGNITION_LEVELS = ["alphabet", "sign", "phrase"];
const DEFAULT_MODEL_SCOPE = "unified";
const DEFAULT_MODEL_PROFILE = "auto";
const SUPPORTED_MODEL_PROFILES = [
  "fast",
  "accurate",
  "dom_fallback",
  "phrase_pack",
  "phrase_pack_plus",
  "phrase_pack_target9",
  "phrase_pack_target9_smart",
];
const CUSTOM_MODEL_PROFILE_PATTERN = /^[a-z0-9][a-z0-9_-]{1,63}$/;
const SUPPORTED_MODEL_SCOPES = [
  ...SUPPORTED_RECOGNITION_LEVELS,
  DEFAULT_MODEL_SCOPE,
];
const TRAINING_VIDEO_SELECT = `
  SELECT
    training_videos.id,
    training_videos.user_id,
    app_users.email AS user_email,
    training_videos.phrase_id AS label_id,
    training_videos.phrase_id,
    training_videos.label_type,
    phrase_library.text AS phrase_text,
    phrase_library.entry_type,
    phrase_library.recognition_level,
    phrase_library.unit_code,
    phrase_categories.name AS category_name,
    training_videos.sign_language,
    training_videos.signer_label,
    training_videos.video_path,
    training_videos.duration_ms,
    training_videos.dataset_split,
    training_videos.status,
    training_videos.quality_score,
    training_videos.review_status,
    training_videos.review_notes,
    training_videos.reviewed_by,
    reviewers.email AS reviewer_email,
    training_videos.reviewed_at,
    training_videos.created_at,
    landmark_sequences.id AS landmark_sequence_id,
    landmark_sequences.file_path AS landmark_file_path,
    landmark_sequences.frame_count AS landmark_frame_count,
    landmark_sequences.status AS landmark_status,
    landmark_sequences.valid_frame_ratio AS landmark_valid_frame_ratio,
    landmark_sequences.missing_hand_ratio AS landmark_missing_hand_ratio,
    landmark_sequences.missing_face_ratio AS landmark_missing_face_ratio,
    landmark_sequences.missing_pose_ratio AS landmark_missing_pose_ratio,
    landmark_sequences.normalization_version AS landmark_normalization_version,
    landmark_sequences.error_message AS landmark_error_message,
    landmark_sequences.updated_at AS landmark_updated_at
  FROM training_videos
  LEFT JOIN app_users
    ON app_users.id = training_videos.user_id
  LEFT JOIN app_users AS reviewers
    ON reviewers.id = training_videos.reviewed_by
  JOIN phrase_library
    ON phrase_library.id = training_videos.phrase_id
  JOIN phrase_categories
    ON phrase_categories.id = phrase_library.category_id
  LEFT JOIN landmark_sequences
    ON landmark_sequences.training_video_id = training_videos.id
`;

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors());
app.use(express.json({ limit: "15mb" }));
app.use("/uploads", express.static(UPLOADS_ROOT));
app.use("/ml-artifacts", express.static(ML_ARTIFACTS_DIR));

function normalizePreferredCategories(value) {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .filter((item) => typeof item === "string" && item.trim())
    .map((item) => item.trim());
}

function normalizeTrainingVideoStatus(value, fallback = "draft") {
  if (typeof value !== "string") {
    return fallback;
  }

  const normalized = value.trim().toLowerCase();
  return SUPPORTED_TRAINING_VIDEO_STATUSES.includes(normalized)
    ? normalized
    : fallback;
}

function normalizeDatasetSplit(value, fallback = "unassigned") {
  if (typeof value !== "string") {
    return fallback;
  }

  const normalized = value.trim().toLowerCase();
  return SUPPORTED_DATASET_SPLITS.includes(normalized)
    ? normalized
    : fallback;
}

function stripJsonBom(value) {
  if (typeof value !== "string") {
    return value;
  }

  return value.charCodeAt(0) === 0xfeff ? value.slice(1) : value;
}

function parseJsonText(value) {
  return JSON.parse(stripJsonBom(value));
}

function normalizeRecognitionLevel(value, fallback = DEFAULT_RECOGNITION_LEVEL) {
  if (typeof value !== "string") {
    return fallback;
  }

  const normalized = value.trim().toLowerCase();
  return SUPPORTED_RECOGNITION_LEVELS.includes(normalized) ? normalized : fallback;
}

function normalizeModelScope(value, fallback = DEFAULT_MODEL_SCOPE) {
  if (typeof value !== "string") {
    return fallback;
  }

  const normalized = value.trim().toLowerCase();
  return SUPPORTED_MODEL_SCOPES.includes(normalized) ? normalized : fallback;
}

function normalizeModelProfile(value, fallback = DEFAULT_MODEL_PROFILE) {
  if (typeof value !== "string") {
    return fallback;
  }

  const normalized = value.trim().toLowerCase();
  return SUPPORTED_MODEL_PROFILES.includes(normalized) ||
    CUSTOM_MODEL_PROFILE_PATTERN.test(normalized)
    ? normalized
    : fallback;
}

function normalizeRecognitionLevelList(value) {
  const sourceValues = Array.isArray(value)
    ? value
    : typeof value === "string"
      ? value.split(",")
      : [];

  return Array.from(
    new Set(
      sourceValues
        .map((item) =>
          typeof item === "string" ? item.trim().toLowerCase() : "",
        )
        .filter((item) => SUPPORTED_RECOGNITION_LEVELS.includes(item)),
    ),
  );
}

function getModelTrainingDefaults(recognitionLevel, modelType) {
  const safeRecognitionLevel = SUPPORTED_RECOGNITION_LEVELS.includes(recognitionLevel)
    ? recognitionLevel
    : DEFAULT_RECOGNITION_LEVEL;
  const safeModelType = MODEL_BENCHMARK_TYPES.includes(modelType)
    ? modelType
    : "baseline";

  return (
    MODEL_TRAINING_DEFAULTS[safeRecognitionLevel]?.[safeModelType] ??
    MODEL_TRAINING_DEFAULTS[DEFAULT_RECOGNITION_LEVEL].baseline
  );
}

function isUnifiedModelScope(value) {
  return normalizeModelScope(value) === DEFAULT_MODEL_SCOPE;
}

function getDatasetExportDir(recognitionLevel, exportName = "latest") {
  return path.join(
    DATASET_EXPORTS_DIR,
    normalizeModelScope(recognitionLevel),
    exportName,
  );
}

function buildDatasetExportUrl(recognitionLevel, fileName, exportName = "latest") {
  return `/uploads/datasets/${normalizeModelScope(recognitionLevel)}/${exportName}/${fileName}`;
}

function getMlRecognitionLevelDir(recognitionLevel) {
  return path.join(ML_ARTIFACTS_DIR, normalizeModelScope(recognitionLevel));
}

function getMlLatestDir(recognitionLevel) {
  return path.join(getMlRecognitionLevelDir(recognitionLevel), "latest");
}

function getMlProfilesDir(recognitionLevel) {
  return path.join(getMlRecognitionLevelDir(recognitionLevel), "profiles");
}

function getMlProfileDir(
  recognitionLevel,
  modelProfile = DEFAULT_MODEL_PROFILE,
) {
  return path.join(
    getMlProfilesDir(recognitionLevel),
    normalizeModelProfile(modelProfile),
  );
}

function getMlModelDir(
  recognitionLevel,
  modelProfile = DEFAULT_MODEL_PROFILE,
) {
  const normalizedProfile = normalizeModelProfile(modelProfile);

  if (normalizedProfile !== DEFAULT_MODEL_PROFILE) {
    return getMlProfileDir(recognitionLevel, normalizedProfile);
  }

  return getMlLatestDir(recognitionLevel);
}

function getMlCandidatesDir(recognitionLevel) {
  return path.join(getMlRecognitionLevelDir(recognitionLevel), "candidates");
}

function getMlBenchmarkLatestPath(recognitionLevel) {
  return path.join(
    getMlRecognitionLevelDir(recognitionLevel),
    "benchmark-latest.json",
  );
}

function formatSettingsRow(row) {
  if (!row) {
    return {
      autoSpeakEnabled: DEFAULT_USER_SETTINGS.autoSpeakEnabled,
      speechRate: DEFAULT_USER_SETTINGS.speechRate,
      speechPitch: DEFAULT_USER_SETTINGS.speechPitch,
      voiceName: DEFAULT_USER_SETTINGS.voiceName,
      uiLanguage: DEFAULT_USER_SETTINGS.uiLanguage,
      signLanguage: DEFAULT_USER_SETTINGS.signLanguage,
      preferredCategories: DEFAULT_USER_SETTINGS.preferredCategories,
      largeTextEnabled: DEFAULT_USER_SETTINGS.largeTextEnabled,
      developerModeEnabled: DEFAULT_USER_SETTINGS.developerModeEnabled,
    };
  }

  return {
    autoSpeakEnabled: row.auto_speak_enabled,
    speechRate: Number(row.speech_rate),
    speechPitch: Number(row.speech_pitch),
    voiceName: row.voice_name ?? "",
    uiLanguage: row.interface_language,
    signLanguage: row.sign_language,
    preferredCategories: row.preferred_categories ?? [],
    largeTextEnabled: row.large_text_enabled,
    developerModeEnabled: row.developer_mode_enabled ?? false,
  };
}

function normalizeStoredFilePath(value = "") {
  if (value === null || value === undefined) {
    return "";
  }

  return String(value)
    .replace(/\\/g, "/")
    .replace(/^\/+/, "")
    .replace(/^uploads\//, "");
}

function buildPublicFileUrl(storedPath) {
  if (!storedPath) {
    return "";
  }

  const normalizedPath = normalizeStoredFilePath(storedPath);
  return normalizedPath ? `/uploads/${normalizedPath}` : "";
}

function resolveUploadFilePath(storedPath) {
  const normalizedPath = normalizeStoredFilePath(storedPath);

  if (!normalizedPath) {
    return null;
  }

  return path.join(UPLOADS_ROOT, ...normalizedPath.split("/"));
}

function formatTrainingVideoRow(row) {
  return {
    id: row.id,
    user_id: row.user_id,
    user_email: row.user_email,
    label_id: row.label_id ?? row.phrase_id,
    phrase_id: row.phrase_id,
    phrase_text: row.phrase_text,
    entry_type:
      row.entry_type ??
      row.recognition_level ??
      DEFAULT_RECOGNITION_LEVEL,
    label_type:
      row.label_type ??
      row.entry_type ??
      row.recognition_level ??
      DEFAULT_RECOGNITION_LEVEL,
    recognition_level: row.recognition_level ?? DEFAULT_RECOGNITION_LEVEL,
    unit_code: row.unit_code ?? null,
    category_name: row.category_name,
    sign_language: row.sign_language,
    signer_label: row.signer_label ?? "",
    video_path: row.video_path,
    video_url: buildPublicFileUrl(row.video_path),
    duration_ms: row.duration_ms,
    dataset_split: row.dataset_split,
    status: row.status,
    quality_score: row.quality_score,
    review_status: row.review_status,
    review_notes: row.review_notes,
    reviewed_by: row.reviewed_by,
    reviewer_email: row.reviewer_email,
    reviewed_at: row.reviewed_at,
    created_at: row.created_at,
    landmark_sequence_id: row.landmark_sequence_id,
    landmark_file_path: row.landmark_file_path,
    landmark_url: buildPublicFileUrl(row.landmark_file_path),
    landmark_frame_count: row.landmark_frame_count,
    landmark_status: row.landmark_status,
    landmark_valid_frame_ratio:
      row.landmark_valid_frame_ratio === null ||
      row.landmark_valid_frame_ratio === undefined
        ? null
        : Number(row.landmark_valid_frame_ratio),
    landmark_missing_hand_ratio:
      row.landmark_missing_hand_ratio === null ||
      row.landmark_missing_hand_ratio === undefined
        ? null
        : Number(row.landmark_missing_hand_ratio),
    landmark_missing_face_ratio:
      row.landmark_missing_face_ratio === null ||
      row.landmark_missing_face_ratio === undefined
        ? null
        : Number(row.landmark_missing_face_ratio),
    landmark_missing_pose_ratio:
      row.landmark_missing_pose_ratio === null ||
      row.landmark_missing_pose_ratio === undefined
        ? null
        : Number(row.landmark_missing_pose_ratio),
    landmark_normalization_version: row.landmark_normalization_version ?? "none",
    landmark_error_message: row.landmark_error_message,
    landmark_updated_at: row.landmark_updated_at,
  };
}

function formatRecognitionRunRow(row) {
  return {
    id: row.id,
    user_id: row.user_id,
    user_email: row.user_email,
    phrase_id: row.phrase_id,
    phrase_text: row.phrase_text,
    sign_language: row.sign_language,
    recognition_level: row.recognition_level ?? DEFAULT_RECOGNITION_LEVEL,
    source_mode: row.source_mode,
    recognized_text: row.recognized_text,
    confidence:
      row.confidence === null || row.confidence === undefined
        ? null
        : Number(row.confidence),
    raw_output_json: row.raw_output_json ?? {},
    created_at: row.created_at,
  };
}

function averageMetric(rows, accessor) {
  if (!rows.length) {
    return 0;
  }

  const values = rows
    .map(accessor)
    .filter((value) => typeof value === "number" && Number.isFinite(value));

  if (!values.length) {
    return 0;
  }

  return Number((values.reduce((sum, value) => sum + value, 0) / values.length).toFixed(4));
}

async function buildDatasetAnalytics(
  signLanguage,
  recognitionLevel = DEFAULT_RECOGNITION_LEVEL,
) {
  const values = [];
  const conditions = [];
  const normalizedRecognitionLevel = normalizeModelScope(recognitionLevel);
  const libraryValues = [];
  const libraryConditions = [];

  if (signLanguage?.trim()) {
    values.push(signLanguage.trim());
    conditions.push(`training_videos.sign_language = $${values.length}`);
    libraryValues.push(signLanguage.trim());
    libraryConditions.push(`phrase_library.sign_language = $${libraryValues.length}`);
  }

  if (!isUnifiedModelScope(normalizedRecognitionLevel)) {
    values.push(normalizedRecognitionLevel);
    conditions.push(`phrase_library.recognition_level = $${values.length}`);
    libraryValues.push(normalizedRecognitionLevel);
    libraryConditions.push(
      `phrase_library.recognition_level = $${libraryValues.length}`,
    );
  }

  const whereClause = conditions.length ? `WHERE ${conditions.join(" AND ")}` : "";
  const libraryWhereClause = libraryConditions.length
    ? `WHERE ${libraryConditions.join(" AND ")}`
    : "";

  const rowsResult = await pool.query(
    `
      ${TRAINING_VIDEO_SELECT}
      ${whereClause}
      ORDER BY training_videos.created_at DESC
    `,
    values,
  );

  const rows = rowsResult.rows;
  const libraryEntriesResult = await pool.query(
    `
      SELECT
        phrase_library.id,
        phrase_library.text,
        phrase_library.entry_type,
        phrase_library.recognition_level,
        phrase_library.unit_code,
        phrase_categories.name AS category_name
      FROM phrase_library
      JOIN phrase_categories
        ON phrase_categories.id = phrase_library.category_id
      ${libraryWhereClause}
      ORDER BY phrase_library.sort_order ASC, phrase_library.id ASC
    `,
    libraryValues,
  );
  const libraryEntries = libraryEntriesResult.rows;
  const review_counts = rows.reduce(
    (accumulator, row) => {
      const status = row.review_status || "pending";
      accumulator.total += 1;
      accumulator[status] = (accumulator[status] ?? 0) + 1;
      return accumulator;
    },
    { total: 0, pending: 0, approved: 0, rejected: 0, needs_retake: 0 },
  );

  const approvedReadyRows = rows.filter(
    (row) => row.review_status === "approved" && row.landmark_status === "ready",
  );
  const categorySlugToName = new Map(
    DEFAULT_PHRASE_CATEGORIES.map((category) => [category.slug, category.name]),
  );
  const labelCoverageMap = new Map();

  for (const entry of libraryEntries) {
    const entryKey = `${
      entry.recognition_level ?? DEFAULT_RECOGNITION_LEVEL
    }:${entry.unit_code ?? entry.id}`;

    labelCoverageMap.set(entryKey, {
      entry_key: entryKey,
      label_id: entry.id,
      label_type:
        entry.entry_type ?? entry.recognition_level ?? DEFAULT_RECOGNITION_LEVEL,
      phrase_id: entry.id,
      phrase_text: entry.text,
      entry_type: entry.entry_type ?? entry.recognition_level ?? DEFAULT_RECOGNITION_LEVEL,
      recognition_level: entry.recognition_level ?? DEFAULT_RECOGNITION_LEVEL,
      unit_code: entry.unit_code ?? null,
      category_name: entry.category_name,
      total_count: 0,
      approved_ready_count: 0,
      remaining_to_target: DATASET_COLLECTION_TARGETS.targetApprovedReadyPerPhrase,
    });
  }

  for (const row of rows) {
    const entryKey = `${row.recognition_level ?? DEFAULT_RECOGNITION_LEVEL}:${
      row.unit_code ?? row.phrase_id
    }`;
    const existing =
      labelCoverageMap.get(entryKey) ??
      {
        entry_key: entryKey,
        label_id: row.phrase_id,
        label_type:
          row.label_type ??
          row.entry_type ??
          row.recognition_level ??
          DEFAULT_RECOGNITION_LEVEL,
        phrase_id: row.phrase_id,
        phrase_text: row.phrase_text,
        entry_type:
          row.entry_type ?? row.recognition_level ?? DEFAULT_RECOGNITION_LEVEL,
        recognition_level: row.recognition_level ?? DEFAULT_RECOGNITION_LEVEL,
        unit_code: row.unit_code ?? null,
        category_name: row.category_name,
        total_count: 0,
        approved_ready_count: 0,
        remaining_to_target: DATASET_COLLECTION_TARGETS.targetApprovedReadyPerPhrase,
      };

    existing.total_count += 1;

    if (row.review_status === "approved" && row.landmark_status === "ready") {
      existing.approved_ready_count += 1;
    }

    existing.remaining_to_target = Math.max(
      0,
      DATASET_COLLECTION_TARGETS.targetApprovedReadyPerPhrase -
        existing.approved_ready_count,
    );

    labelCoverageMap.set(entryKey, existing);
  }

  const per_phrase = [...labelCoverageMap.values()].sort((left, right) => {
    if (left.approved_ready_count !== right.approved_ready_count) {
      return left.approved_ready_count - right.approved_ready_count;
    }

    if (left.total_count !== right.total_count) {
      return left.total_count - right.total_count;
    }

    return left.phrase_text.localeCompare(right.phrase_text, "ru");
  });

  const signerKeys = new Set(approvedReadyRows.map((row) => buildSignerKey(row)));
  const phraseCoverageMap = new Map(
    per_phrase.map((item) => [item.phrase_text, item]),
  );
  const firstWaveProgress =
    normalizedRecognitionLevel === "phrase"
      ? FIRST_WAVE_DATASET_PHRASES.map((phrase) => {
          const existing = phraseCoverageMap.get(phrase.text);
          const approvedReadyCount = existing?.approved_ready_count ?? 0;
          const totalCount = existing?.total_count ?? 0;

          return {
            phrase_text: phrase.text,
            category_name:
              categorySlugToName.get(phrase.categorySlug) ?? phrase.categorySlug,
            total_count: totalCount,
            approved_ready_count: approvedReadyCount,
            remaining_to_target: Math.max(
              0,
              DATASET_COLLECTION_TARGETS.targetApprovedReadyPerPhrase -
                approvedReadyCount,
            ),
          };
        })
      : [];
  const genericLevelProgress = per_phrase.map((item) => ({
    phrase_id: item.phrase_id,
    phrase_text: item.phrase_text,
    category_name: item.category_name,
    total_count: item.total_count,
    approved_ready_count: item.approved_ready_count,
    remaining_to_target: item.remaining_to_target,
    recognition_level: item.recognition_level,
    entry_type: item.entry_type,
  }));
  const coverageProgress =
    normalizedRecognitionLevel === "phrase"
      ? firstWaveProgress
      : genericLevelProgress;
  const coveredLabelCount = coverageProgress.filter(
    (item) => item.remaining_to_target === 0,
  ).length;
  const firstWaveCoveredCount = firstWaveProgress.filter(
    (item) => item.remaining_to_target === 0,
  ).length;
  const benchmarkReadiness = {
    ready:
      approvedReadyRows.length >= DATASET_COLLECTION_TARGETS.minApprovedReadyTotal &&
      signerKeys.size >= DATASET_COLLECTION_TARGETS.minSigners,
    remaining_approved_ready: Math.max(
      0,
      DATASET_COLLECTION_TARGETS.minApprovedReadyTotal - approvedReadyRows.length,
    ),
    remaining_signers: Math.max(
      0,
      DATASET_COLLECTION_TARGETS.minSigners - signerKeys.size,
    ),
    covered_label_count: coveredLabelCount,
    total_label_count: coverageProgress.length,
    labels_below_target_count: coverageProgress.filter(
      (item) => item.remaining_to_target > 0,
    ).length,
    covered_phrase_count: firstWaveCoveredCount,
    total_phrase_count:
      normalizedRecognitionLevel === "phrase"
        ? firstWaveProgress.length
        : coverageProgress.length,
    phrases_below_target_count:
      normalizedRecognitionLevel === "phrase"
        ? firstWaveProgress.filter((item) => item.remaining_to_target > 0).length
        : coverageProgress.filter((item) => item.remaining_to_target > 0).length,
  };

  return {
    sign_language: signLanguage ?? null,
    recognition_level: normalizedRecognitionLevel,
    generated_at: new Date().toISOString(),
    targets: DATASET_COLLECTION_TARGETS,
    review_counts,
    approved_ready_count: approvedReadyRows.length,
    signer_count: signerKeys.size,
    catalog_entry_count: libraryEntries.length,
    benchmark_readiness: benchmarkReadiness,
    low_data_labels: per_phrase.slice(0, 10),
    low_data_phrases: per_phrase.slice(0, 10),
    first_wave_progress: firstWaveProgress,
    level_progress: coverageProgress,
    collection_backlog: coverageProgress
      .filter((item) => item.remaining_to_target > 0)
      .sort((left, right) => right.remaining_to_target - left.remaining_to_target)
      .slice(0, 10),
    extraction_quality: {
      average_valid_frame_ratio: averageMetric(
        approvedReadyRows,
        (row) => Number(row.landmark_valid_frame_ratio ?? 0),
      ),
      average_missing_hand_ratio: averageMetric(
        approvedReadyRows,
        (row) => Number(row.landmark_missing_hand_ratio ?? 0),
      ),
      average_missing_face_ratio: averageMetric(
        approvedReadyRows,
        (row) => Number(row.landmark_missing_face_ratio ?? 0),
      ),
      average_missing_pose_ratio: averageMetric(
        approvedReadyRows,
        (row) => Number(row.landmark_missing_pose_ratio ?? 0),
      ),
    },
  };
}

async function ensureUploadsDirectories() {
  await mkdir(TRAINING_UPLOADS_DIR, { recursive: true });
  await mkdir(LANDMARK_UPLOADS_DIR, { recursive: true });
  await mkdir(DATASET_EXPORTS_DIR, { recursive: true });
  await mkdir(TEMP_UPLOADS_DIR, { recursive: true });
  await mkdir(ML_ARTIFACTS_DIR, { recursive: true });
  await mkdir(ML_CANDIDATES_DIR, { recursive: true });
  for (const recognitionLevel of SUPPORTED_MODEL_SCOPES) {
    await mkdir(getDatasetExportDir(recognitionLevel), { recursive: true });
    await mkdir(getMlLatestDir(recognitionLevel), { recursive: true });
    await mkdir(getMlProfilesDir(recognitionLevel), { recursive: true });
    for (const modelProfile of SUPPORTED_MODEL_PROFILES) {
      await mkdir(getMlProfileDir(recognitionLevel, modelProfile), {
        recursive: true,
      });
    }
    await mkdir(getMlCandidatesDir(recognitionLevel), { recursive: true });
  }
}

async function ensureDatabaseSchema() {
  const schemaPath = path.join(__dirname, "schema.sql");
  const schemaSql = await readFile(schemaPath, "utf8");
  await pool.query(schemaSql);
}

async function findUserByEmail(userEmail = DEFAULT_USER.email) {
  const result = await pool.query(
    `
      SELECT id, email, display_name, created_at
      FROM app_users
      WHERE email = $1
      LIMIT 1
    `,
    [userEmail],
  );

  return result.rows[0] ?? null;
}

async function findPhraseByIdOrText({
  phraseId,
  text,
  signLanguage,
  recognitionLevel,
}) {
  if (Number.isFinite(phraseId) && phraseId > 0) {
    const result = await pool.query(
      `
        SELECT id, text, entry_type, recognition_level, sign_language
        FROM phrase_library
        WHERE id = $1
        LIMIT 1
      `,
      [phraseId],
    );

    return result.rows[0] ?? null;
  }

  if (typeof text === "string" && text.trim()) {
    const values = [text.trim()];
    const conditions = [
      `text = $1`,
      `is_active = TRUE`,
    ];

    if (signLanguage?.trim()) {
      values.push(signLanguage.trim());
      conditions.push(`sign_language = $${values.length}`);
    }

    if (typeof recognitionLevel === "string" && recognitionLevel.trim()) {
      values.push(normalizeRecognitionLevel(recognitionLevel));
      conditions.push(`recognition_level = $${values.length}`);
    }

    const result = await pool.query(
      `
        SELECT id, text, entry_type, recognition_level, sign_language
        FROM phrase_library
        WHERE ${conditions.join(" AND ")}
        ORDER BY id ASC
        LIMIT 1
      `,
      values,
    );

    return result.rows[0] ?? null;
  }

  return null;
}

async function ensureDefaultUser() {
  const result = await pool.query(
    `
      INSERT INTO app_users (email, password_hash, display_name)
      VALUES ($1, $2, $3)
      ON CONFLICT (email)
      DO UPDATE SET display_name = EXCLUDED.display_name
      RETURNING id, email, display_name, created_at
    `,
    [
      DEFAULT_USER.email,
      DEFAULT_USER.passwordHash,
      DEFAULT_USER.displayName,
    ],
  );

  const defaultUser = result.rows[0];

  await pool.query(
    `
      INSERT INTO user_settings (
        user_id,
        interface_language,
        speech_rate,
        speech_pitch,
        voice_name,
        auto_speak_enabled,
        sign_language,
        preferred_categories,
        large_text_enabled,
        developer_mode_enabled
      )
      VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10)
      ON CONFLICT (user_id) DO NOTHING
    `,
    [
      defaultUser.id,
      DEFAULT_USER_SETTINGS.uiLanguage,
      DEFAULT_USER_SETTINGS.speechRate,
      DEFAULT_USER_SETTINGS.speechPitch,
      DEFAULT_USER_SETTINGS.voiceName,
      DEFAULT_USER_SETTINGS.autoSpeakEnabled,
      DEFAULT_USER_SETTINGS.signLanguage,
      JSON.stringify(DEFAULT_USER_SETTINGS.preferredCategories),
      DEFAULT_USER_SETTINGS.largeTextEnabled,
      DEFAULT_USER_SETTINGS.developerModeEnabled,
    ],
  );

  return defaultUser;
}

async function getOrCreateUserSettings(userEmail = DEFAULT_USER.email) {
  const user = await findUserByEmail(userEmail);

  if (!user) {
    return null;
  }

  await pool.query(
    `
      INSERT INTO user_settings (
        user_id,
        interface_language,
        speech_rate,
        speech_pitch,
        voice_name,
        auto_speak_enabled,
        sign_language,
        preferred_categories,
        large_text_enabled,
        developer_mode_enabled
      )
      VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10)
      ON CONFLICT (user_id) DO NOTHING
    `,
    [
      user.id,
      DEFAULT_USER_SETTINGS.uiLanguage,
      DEFAULT_USER_SETTINGS.speechRate,
      DEFAULT_USER_SETTINGS.speechPitch,
      DEFAULT_USER_SETTINGS.voiceName,
      DEFAULT_USER_SETTINGS.autoSpeakEnabled,
      DEFAULT_USER_SETTINGS.signLanguage,
      JSON.stringify(DEFAULT_USER_SETTINGS.preferredCategories),
      DEFAULT_USER_SETTINGS.largeTextEnabled,
      DEFAULT_USER_SETTINGS.developerModeEnabled,
    ],
  );

  const settingsResult = await pool.query(
    `
      SELECT
        id,
        user_id,
        interface_language,
        speech_rate,
        speech_pitch,
        voice_name,
        auto_speak_enabled,
        sign_language,
        preferred_categories,
        large_text_enabled,
        developer_mode_enabled,
        created_at,
        updated_at
      FROM user_settings
      WHERE user_id = $1
      LIMIT 1
    `,
    [user.id],
  );

  return {
    user,
    settings: settingsResult.rows[0] ?? null,
  };
}

async function ensurePhraseCategories() {
  const categoryMap = new Map();

  for (const category of DEFAULT_PHRASE_CATEGORIES) {
    const result = await pool.query(
      `
        INSERT INTO phrase_categories (slug, name, sort_order)
        VALUES ($1, $2, $3)
        ON CONFLICT (slug)
        DO UPDATE SET
          name = EXCLUDED.name,
          sort_order = EXCLUDED.sort_order
        RETURNING id, slug
      `,
      [category.slug, category.name, category.sortOrder],
    );

    categoryMap.set(result.rows[0].slug, result.rows[0].id);
  }

  return categoryMap;
}

async function ensureSignLanguages() {
  for (const language of DEFAULT_SIGN_LANGUAGES) {
    await pool.query(
      `
        INSERT INTO sign_languages (code, name, is_active, is_default)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (code)
        DO UPDATE SET
          name = EXCLUDED.name,
          is_active = EXCLUDED.is_active,
          is_default = EXCLUDED.is_default
      `,
      [
        language.code,
        language.name,
        language.isActive,
        language.isDefault,
      ],
    );
  }
}

async function ensurePhraseLibrary(categoryMap) {
  for (const phrase of DEFAULT_PHRASES) {
    const categoryId = categoryMap.get(phrase.categorySlug);

    if (!categoryId) {
      continue;
    }

    await pool.query(
      `
        INSERT INTO phrase_library (
          category_id,
          text,
          sign_language,
          entry_type,
          recognition_level,
          unit_code,
          description,
          reference_notes,
          is_v1,
          is_locked,
          is_featured,
          is_active,
          sort_order
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, TRUE, $12)
        ON CONFLICT (category_id, text, entry_type)
        DO UPDATE SET
          sign_language = EXCLUDED.sign_language,
          entry_type = EXCLUDED.entry_type,
          recognition_level = EXCLUDED.recognition_level,
          unit_code = EXCLUDED.unit_code,
          description = EXCLUDED.description,
          reference_notes = EXCLUDED.reference_notes,
          is_v1 = EXCLUDED.is_v1,
          is_locked = EXCLUDED.is_locked,
          is_featured = EXCLUDED.is_featured,
          is_active = TRUE,
          sort_order = EXCLUDED.sort_order
      `,
      [
        categoryId,
        phrase.text,
        phrase.signLanguage ?? DEFAULT_USER_SETTINGS.signLanguage,
        normalizeRecognitionLevel(phrase.entryType ?? phrase.recognitionLevel),
        normalizeRecognitionLevel(phrase.recognitionLevel),
        phrase.unitCode?.trim() || null,
        phrase.description ?? "",
        phrase.referenceNotes ?? "",
        Boolean(phrase.isV1),
        Boolean(phrase.isLocked),
        phrase.isFeatured,
        phrase.sortOrder,
      ],
    );
  }
}

async function bootstrapDatabase() {
  await ensureUploadsDirectories();
  await ensureDatabaseSchema();

  const defaultUser = await ensureDefaultUser();
  await ensureSignLanguages();
  const categoryMap = await ensurePhraseCategories();
  await ensurePhraseLibrary(categoryMap);

  console.log("Database schema is ready");
  console.log(`Default user is ready: ${defaultUser.email}`);
}

async function getTrainingVideoById(trainingVideoId) {
  const result = await pool.query(
    `
      ${TRAINING_VIDEO_SELECT}
      WHERE training_videos.id = $1
      LIMIT 1
    `,
    [trainingVideoId],
  );

  return result.rows[0] ? formatTrainingVideoRow(result.rows[0]) : null;
}

async function upsertLandmarkSequence({
  trainingVideoId,
  filePath,
  frameCount = 0,
  status,
  validFrameRatio = null,
  missingHandRatio = null,
  missingFaceRatio = null,
  missingPoseRatio = null,
  normalizationVersion = "none",
  errorMessage = null,
}) {
  const result = await pool.query(
    `
      INSERT INTO landmark_sequences (
        training_video_id,
        file_path,
        frame_count,
        status,
        valid_frame_ratio,
        missing_hand_ratio,
        missing_face_ratio,
        missing_pose_ratio,
        normalization_version,
        error_message,
        extractor
      )
      VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, 'mediapipe_holistic_py')
      ON CONFLICT (training_video_id)
      DO UPDATE SET
        file_path = EXCLUDED.file_path,
        frame_count = EXCLUDED.frame_count,
        status = EXCLUDED.status,
        valid_frame_ratio = EXCLUDED.valid_frame_ratio,
        missing_hand_ratio = EXCLUDED.missing_hand_ratio,
        missing_face_ratio = EXCLUDED.missing_face_ratio,
        missing_pose_ratio = EXCLUDED.missing_pose_ratio,
        normalization_version = EXCLUDED.normalization_version,
        error_message = EXCLUDED.error_message,
        extractor = EXCLUDED.extractor,
        updated_at = NOW()
      RETURNING
        id,
        training_video_id,
        file_path,
        frame_count,
        status,
        valid_frame_ratio,
        missing_hand_ratio,
        missing_face_ratio,
        missing_pose_ratio,
        normalization_version,
        error_message,
        updated_at
    `,
    [
      trainingVideoId,
      filePath,
      frameCount,
      status,
      validFrameRatio,
      missingHandRatio,
      missingFaceRatio,
      missingPoseRatio,
      normalizationVersion,
      errorMessage,
    ],
  );

  return result.rows[0];
}

function runLandmarkExtraction({
  inputPath,
  outputPath,
  signLanguage,
  trainingVideoId,
  phraseId,
}) {
  return new Promise((resolve, reject) => {
    const child = spawn(
      PYTHON_BIN,
      [
        LANDMARK_EXTRACT_SCRIPT,
        "--input",
        inputPath,
        "--output",
        outputPath,
        "--sign-language",
        signLanguage,
        "--training-video-id",
        String(Number(trainingVideoId) || 0),
        "--phrase-id",
        String(Number(phraseId) || 0),
      ],
      {
        cwd: __dirname,
        env: {
          ...process.env,
          PYTHONUTF8: "1",
          PYTHONIOENCODING: "utf-8",
        },
        stdio: ["ignore", "pipe", "pipe"],
      },
    );

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (data) => {
      stdout += data.toString();
    });

    child.stderr.on("data", (data) => {
      stderr += data.toString();
    });

    child.on("error", (error) => {
      reject(error);
    });

    child.on("close", (code) => {
      if (code !== 0) {
        reject(
          new Error(
            stderr.trim() ||
              stdout.trim() ||
              `Landmark extraction failed with code ${code}`,
          ),
        );
        return;
      }

      const lines = stdout
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter(Boolean);
      const lastLine = lines.at(-1);

      if (!lastLine) {
        reject(new Error("Landmark extractor returned empty output"));
        return;
      }

      try {
        resolve(JSON.parse(lastLine));
      } catch {
        reject(new Error("Failed to parse landmark extractor output"));
      }
    });
  });
}

function runPythonJsonScript(scriptPath, argumentsList) {
  return new Promise((resolve, reject) => {
    const child = spawn(PYTHON_BIN, [scriptPath, ...argumentsList], {
      cwd: __dirname,
      env: {
        ...process.env,
        PYTHONUTF8: "1",
        PYTHONIOENCODING: "utf-8",
      },
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (data) => {
      stdout += data.toString();
    });

    child.stderr.on("data", (data) => {
      stderr += data.toString();
    });

    child.on("error", (error) => {
      reject(error);
    });

    child.on("close", (code) => {
      if (code !== 0) {
        reject(
          new Error(
            stderr.trim() ||
              stdout.trim() ||
              `Python script failed with code ${code}`,
          ),
        );
        return;
      }

      const lines = stdout
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter(Boolean);
      const lastLine = lines.at(-1);

      if (!lastLine) {
        reject(new Error("Python script returned empty output"));
        return;
      }

      try {
        resolve(JSON.parse(lastLine));
      } catch {
        reject(new Error("Failed to parse Python script output"));
      }
    });
  });
}

function rejectPredictionWorkerPending(workerState, error) {
  for (const pending of workerState.pending.values()) {
    pending.reject(error);
  }

  workerState.pending.clear();
}

function buildPredictionWorkerKey(
  recognitionLevel,
  modelProfile = DEFAULT_MODEL_PROFILE,
) {
  const normalizedRecognitionLevel = normalizeModelScope(recognitionLevel);
  const normalizedProfile = normalizeModelProfile(modelProfile);

  return normalizedProfile === DEFAULT_MODEL_PROFILE
    ? `${normalizedRecognitionLevel}:latest`
    : `${normalizedRecognitionLevel}:${normalizedProfile}`;
}

function disposePredictionWorker(workerKey, reason) {
  const workerState = predictionWorkers.get(workerKey);

  if (!workerState) {
    return;
  }

  predictionWorkers.delete(workerKey);
  workerState.closed = true;
  rejectPredictionWorkerPending(workerState, new Error(reason));

  try {
    if (!workerState.child.stdin.destroyed) {
      workerState.child.stdin.end();
    }
  } catch {}

  try {
    workerState.child.kill();
  } catch {}
}

function handlePredictionWorkerLine(workerState, line) {
  let payload;

  try {
    payload = JSON.parse(line);
  } catch {
    return;
  }

  if (!payload?.id) {
    return;
  }

  const pending = workerState.pending.get(payload.id);

  if (!pending) {
    return;
  }

  workerState.pending.delete(payload.id);

  if (payload.ok === false) {
    pending.reject(new Error(payload.error || "Prediction worker failed"));
    return;
  }

  pending.resolve(payload.result ?? payload);
}

function createPredictionWorker(
  workerKey,
  recognitionLevel,
  modelProfile,
  modelDir,
  metadataMtimeMs,
) {
  const child = spawn(
    PYTHON_BIN,
    [PREDICT_SEQUENCE_WORKER_SCRIPT, "--model-dir", modelDir],
    {
      cwd: __dirname,
      env: {
        ...process.env,
        PYTHONUTF8: "1",
        PYTHONIOENCODING: "utf-8",
      },
      stdio: ["pipe", "pipe", "pipe"],
    },
  );

  const workerState = {
    workerKey,
    recognitionLevel,
    modelProfile,
    modelDir,
    metadataMtimeMs,
    child,
    pending: new Map(),
    stdoutBuffer: "",
    stderrBuffer: "",
    closed: false,
  };

  child.stdout.on("data", (data) => {
    workerState.stdoutBuffer += data.toString();
    let newlineIndex = workerState.stdoutBuffer.indexOf("\n");

    while (newlineIndex >= 0) {
      const line = workerState.stdoutBuffer.slice(0, newlineIndex).trim();
      workerState.stdoutBuffer = workerState.stdoutBuffer.slice(newlineIndex + 1);

      if (line) {
        handlePredictionWorkerLine(workerState, line);
      }

      newlineIndex = workerState.stdoutBuffer.indexOf("\n");
    }
  });

  child.stderr.on("data", (data) => {
    workerState.stderrBuffer += data.toString();
  });

  child.on("error", (error) => {
    workerState.closed = true;
    rejectPredictionWorkerPending(workerState, error);

    if (predictionWorkers.get(workerKey) === workerState) {
      predictionWorkers.delete(workerKey);
    }
  });

  child.on("close", (code) => {
    workerState.closed = true;
    const details =
      workerState.stderrBuffer.trim() ||
      workerState.stdoutBuffer.trim() ||
      `Prediction worker exited with code ${code}`;
    rejectPredictionWorkerPending(workerState, new Error(details));

    if (predictionWorkers.get(workerKey) === workerState) {
      predictionWorkers.delete(workerKey);
    }
  });

  return workerState;
}

async function getPredictionWorker(
  recognitionLevel,
  modelProfile = DEFAULT_MODEL_PROFILE,
) {
  const workerKey = buildPredictionWorkerKey(recognitionLevel, modelProfile);
  const modelDir = getMlModelDir(recognitionLevel, modelProfile);
  const metadataPath = path.join(modelDir, "metadata.json");
  const metadataStats = await stat(metadataPath);
  const metadataMtimeMs = metadataStats.mtimeMs;
  const existingWorker = predictionWorkers.get(workerKey);

  if (
    existingWorker &&
    !existingWorker.closed &&
    existingWorker.modelDir === modelDir &&
    existingWorker.metadataMtimeMs === metadataMtimeMs
  ) {
    return existingWorker;
  }

  if (existingWorker) {
    disposePredictionWorker(
      workerKey,
      "Prediction worker restarted after model update",
    );
  }

  const workerState = createPredictionWorker(
    workerKey,
    recognitionLevel,
    modelProfile,
    modelDir,
    metadataMtimeMs,
  );
  predictionWorkers.set(workerKey, workerState);
  return workerState;
}

async function runPersistentPrediction({
  recognitionLevel,
  modelProfile,
  inputPath,
  sequence,
  allowedRecognitionLevels,
  allowedLabelKeys,
}) {
  const workerState = await getPredictionWorker(recognitionLevel, modelProfile);

  if (workerState.closed) {
    throw new Error("Prediction worker is not available");
  }

  return new Promise((resolve, reject) => {
    predictionWorkerRequestSequence += 1;
    const requestId = `${recognitionLevel}:${Date.now()}:${predictionWorkerRequestSequence}`;

    workerState.pending.set(requestId, {
      resolve,
      reject,
    });

    const payload = JSON.stringify({
      id: requestId,
      inputPath,
      sequence,
      allowedRecognitionLevels,
      allowedLabelKeys,
    });

    workerState.child.stdin.write(`${payload}\n`, "utf8", (error) => {
      if (!error) {
        return;
      }

      workerState.pending.delete(requestId);
      reject(error);
    });
  });
}

function clampRatio(value, fallback) {
  const normalized = Number(value);

  if (!Number.isFinite(normalized) || normalized <= 0) {
    return fallback;
  }

  return normalized;
}

function buildSignerKey(row) {
  if (row.signer_label?.trim()) {
    return `signer:${row.signer_label.trim().toLowerCase()}`;
  }

  if (row.user_id) {
    return `user:${row.user_id}`;
  }

  if (row.user_email?.trim()) {
    return `email:${row.user_email.trim().toLowerCase()}`;
  }

  return `video:${row.id}`;
}

function assignDatasetSplits(rows, ratios) {
  const signerGroups = rows.reduce((accumulator, row) => {
    const signerKey = buildSignerKey(row);

    if (!accumulator.has(signerKey)) {
      accumulator.set(signerKey, {
        signerKey,
        rows: [],
        sampleCount: 0,
        firstSeen: new Date(row.created_at).getTime(),
      });
    }

    const group = accumulator.get(signerKey);
    group.rows.push(row);
    group.sampleCount += 1;
    group.firstSeen = Math.min(group.firstSeen, new Date(row.created_at).getTime());
    return accumulator;
  }, new Map());

  const groups = [...signerGroups.values()].sort((left, right) => {
    if (right.sampleCount !== left.sampleCount) {
      return right.sampleCount - left.sampleCount;
    }

    return left.firstSeen - right.firstSeen;
  });

  const totalSamples = rows.length;
  const groupCount = groups.length;
  const rawTargets = {
    train: Math.round(totalSamples * ratios.train),
    val: Math.round(totalSamples * ratios.val),
    test: Math.round(totalSamples * ratios.test),
  };
  const targetTotal = rawTargets.train + rawTargets.val + rawTargets.test;

  if (targetTotal !== totalSamples) {
    rawTargets.train += totalSamples - targetTotal;
  }

  const remainingTargets = {
    train: Math.max(rawTargets.train, 1),
    val: groupCount >= 2 ? Math.max(rawTargets.val, 1) : 0,
    test: groupCount >= 3 ? Math.max(rawTargets.test, 1) : 0,
  };

  while (
    remainingTargets.train + remainingTargets.val + remainingTargets.test >
    totalSamples
  ) {
    if (remainingTargets.train > remainingTargets.val && remainingTargets.train > 1) {
      remainingTargets.train -= 1;
    } else if (remainingTargets.val > remainingTargets.test && remainingTargets.val > 0) {
      remainingTargets.val -= 1;
    } else if (remainingTargets.test > 0) {
      remainingTargets.test -= 1;
    } else {
      break;
    }
  }

  const assignments = new Map();
  const splitSignerCounts = { train: 0, val: 0, test: 0 };

  groups.forEach((group, index) => {
    const groupsLeft = groupCount - index;
    const minReservations = {
      train: 1,
      val: remainingTargets.val > 0 ? 1 : 0,
      test: remainingTargets.test > 0 ? 1 : 0,
    };

    const candidateSplits = ["train", "val", "test"].filter((split) => {
      if (remainingTargets[split] <= 0) {
        return false;
      }

      const reservedGroupsForOthers = ["train", "val", "test"]
        .filter((candidate) => candidate !== split && remainingTargets[candidate] > 0)
        .reduce((sum, candidate) => sum + minReservations[candidate], 0);

      return groupsLeft - 1 >= reservedGroupsForOthers;
    });

    const targetSplit =
      candidateSplits.sort((left, right) => {
        if (remainingTargets[right] !== remainingTargets[left]) {
          return remainingTargets[right] - remainingTargets[left];
        }

        return splitSignerCounts[left] - splitSignerCounts[right];
      })[0] ?? "train";

    group.rows.forEach((row) => {
      assignments.set(row.id, targetSplit);
    });

    remainingTargets[targetSplit] = Math.max(
      remainingTargets[targetSplit] - group.sampleCount,
      0,
    );
    splitSignerCounts[targetSplit] += 1;
  });

  return {
    assignments,
    signerCount: groups.length,
    splitSignerCounts,
  };
}

async function fetchReadyDatasetRows(
  signLanguage,
  recognitionLevel = DEFAULT_RECOGNITION_LEVEL,
) {
  const values = [];
  const conditions = [
    `landmark_sequences.status = 'ready'`,
    `phrase_library.is_active = TRUE`,
    `training_videos.review_status = 'approved'`,
  ];
  const normalizedRecognitionLevel = normalizeModelScope(recognitionLevel);

  if (signLanguage?.trim()) {
    values.push(signLanguage.trim());
    conditions.push(`training_videos.sign_language = $${values.length}`);
  }

  if (!isUnifiedModelScope(normalizedRecognitionLevel)) {
    values.push(normalizedRecognitionLevel);
    conditions.push(`phrase_library.recognition_level = $${values.length}`);
  }

  const result = await pool.query(
    `
      ${TRAINING_VIDEO_SELECT}
      WHERE ${conditions.join(" AND ")}
      ORDER BY phrase_library.id ASC, training_videos.created_at ASC, training_videos.id ASC
    `,
    values,
  );

  const candidateRows = result.rows;
  const rows = candidateRows.filter((row) => {
    const validFrameRatio = Number(row.landmark_valid_frame_ratio ?? 0);
    const missingHandRatio = Number(row.landmark_missing_hand_ratio ?? 1);
    const missingFaceRatio = Number(row.landmark_missing_face_ratio ?? 1);
    const missingPoseRatio = Number(row.landmark_missing_pose_ratio ?? 1);

    return (
      validFrameRatio >= DATASET_QUALITY_THRESHOLDS.minValidFrameRatio &&
      missingHandRatio <= DATASET_QUALITY_THRESHOLDS.maxMissingHandRatio &&
      missingFaceRatio <= DATASET_QUALITY_THRESHOLDS.maxMissingFaceRatio &&
      missingPoseRatio <= DATASET_QUALITY_THRESHOLDS.maxMissingPoseRatio
    );
  });

  return {
    rows,
    candidateCount: candidateRows.length,
    filteredOutCount: candidateRows.length - rows.length,
    qualityThresholds: DATASET_QUALITY_THRESHOLDS,
    qualitySummary: {
      average_valid_frame_ratio: averageMetric(
        rows,
        (row) => Number(row.landmark_valid_frame_ratio ?? 0),
      ),
      average_missing_hand_ratio: averageMetric(
        rows,
        (row) => Number(row.landmark_missing_hand_ratio ?? 0),
      ),
      average_missing_face_ratio: averageMetric(
        rows,
        (row) => Number(row.landmark_missing_face_ratio ?? 0),
      ),
      average_missing_pose_ratio: averageMetric(
        rows,
        (row) => Number(row.landmark_missing_pose_ratio ?? 0),
      ),
    },
  };
}

async function writeDatasetExport(
  rows,
  assignmentPayload,
  exportName = "latest",
  datasetMeta = {},
  recognitionLevel = DEFAULT_RECOGNITION_LEVEL,
) {
  const normalizedRecognitionLevel = normalizeModelScope(recognitionLevel);
  const exportDir = getDatasetExportDir(normalizedRecognitionLevel, exportName);
  await mkdir(exportDir, { recursive: true });
  const assignments =
    assignmentPayload instanceof Map
      ? assignmentPayload
      : assignmentPayload.assignments;
  const signerCount =
    assignmentPayload instanceof Map ? null : assignmentPayload.signerCount;
  const splitSignerCounts =
    assignmentPayload instanceof Map
      ? null
      : assignmentPayload.splitSignerCounts;

  const splitBuckets = {
    train: [],
    val: [],
    test: [],
  };

  for (const row of rows) {
    const split = assignments.get(row.id) ?? "train";
    const landmarkFilePath = resolveUploadFilePath(row.landmark_file_path);

    if (!landmarkFilePath) {
      continue;
    }

    const landmarkPayload = JSON.parse(await readFile(landmarkFilePath, "utf8"));

    splitBuckets[split].push({
      sample_id: row.id,
      label_id: row.phrase_id,
      label_type:
        row.label_type ??
        row.entry_type ??
        row.recognition_level ??
        normalizedRecognitionLevel,
      phrase_id: row.phrase_id,
      phrase_text: row.phrase_text,
      entry_type:
        row.entry_type ??
        row.recognition_level ??
        normalizedRecognitionLevel,
      recognition_level: row.recognition_level ?? normalizedRecognitionLevel,
      unit_code: row.unit_code ?? null,
      category: row.category_name,
      sign_language: row.sign_language,
      signer_key: buildSignerKey(row),
      user_id: row.user_id,
      user_email: row.user_email,
      duration_ms: row.duration_ms,
      dataset_split: split,
      review_status: row.review_status,
      quality_score: row.quality_score,
      landmark_sequence_id: row.landmark_sequence_id,
      landmark_frame_count: row.landmark_frame_count,
      landmark_valid_frame_ratio:
        row.landmark_valid_frame_ratio === null ||
        row.landmark_valid_frame_ratio === undefined
          ? null
          : Number(row.landmark_valid_frame_ratio),
      landmark_missing_hand_ratio:
        row.landmark_missing_hand_ratio === null ||
        row.landmark_missing_hand_ratio === undefined
          ? null
          : Number(row.landmark_missing_hand_ratio),
      landmark_missing_face_ratio:
        row.landmark_missing_face_ratio === null ||
        row.landmark_missing_face_ratio === undefined
          ? null
          : Number(row.landmark_missing_face_ratio),
      landmark_missing_pose_ratio:
        row.landmark_missing_pose_ratio === null ||
        row.landmark_missing_pose_ratio === undefined
          ? null
          : Number(row.landmark_missing_pose_ratio),
      landmark_normalization_version: row.landmark_normalization_version ?? "none",
      landmark_file_path: row.landmark_file_path,
      landmark_url: buildPublicFileUrl(row.landmark_file_path),
      video_path: row.video_path,
      video_url: buildPublicFileUrl(row.video_path),
      created_at: row.created_at,
      sequence_meta: landmarkPayload.meta ?? null,
      sequence: landmarkPayload.frames ?? [],
    });
  }

  const summary = {
    export_name: exportName,
    exported_at: new Date().toISOString(),
    sign_language: rows[0]?.sign_language ?? null,
    recognition_level: normalizedRecognitionLevel,
    approved_only: true,
    quality_filtered: true,
    split_strategy: "by_signer",
    signer_count: signerCount,
    split_signer_counts: splitSignerCounts,
    candidate_count: datasetMeta.candidateCount ?? rows.length,
    filtered_out_count: datasetMeta.filteredOutCount ?? 0,
    quality_thresholds: datasetMeta.qualityThresholds ?? DATASET_QUALITY_THRESHOLDS,
    quality_summary: datasetMeta.qualitySummary ?? null,
    sample_count:
      splitBuckets.train.length + splitBuckets.val.length + splitBuckets.test.length,
    split_counts: {
      train: splitBuckets.train.length,
      val: splitBuckets.val.length,
      test: splitBuckets.test.length,
    },
    phrase_counts: rows.reduce((accumulator, row) => {
      accumulator[row.phrase_text] = (accumulator[row.phrase_text] ?? 0) + 1;
      return accumulator;
    }, {}),
    files: {
      manifest: buildDatasetExportUrl(normalizedRecognitionLevel, "manifest.json", exportName),
      train: buildDatasetExportUrl(normalizedRecognitionLevel, "train.json", exportName),
      val: buildDatasetExportUrl(normalizedRecognitionLevel, "val.json", exportName),
      test: buildDatasetExportUrl(normalizedRecognitionLevel, "test.json", exportName),
      summary: buildDatasetExportUrl(normalizedRecognitionLevel, "summary.json", exportName),
    },
  };

  const manifest = {
    summary,
    samples: [...splitBuckets.train, ...splitBuckets.val, ...splitBuckets.test].map(
      (sample) => ({
      sample_id: sample.sample_id,
      label_id: sample.label_id,
      label_type: sample.label_type,
      phrase_id: sample.phrase_id,
      phrase_text: sample.phrase_text,
      entry_type: sample.entry_type,
      recognition_level: sample.recognition_level,
        unit_code: sample.unit_code,
        category: sample.category,
        sign_language: sample.sign_language,
        signer_key: sample.signer_key,
        dataset_split: sample.dataset_split,
        review_status: sample.review_status,
        quality_score: sample.quality_score,
        landmark_frame_count: sample.landmark_frame_count,
        landmark_valid_frame_ratio: sample.landmark_valid_frame_ratio,
        landmark_missing_hand_ratio: sample.landmark_missing_hand_ratio,
        landmark_missing_face_ratio: sample.landmark_missing_face_ratio,
        landmark_missing_pose_ratio: sample.landmark_missing_pose_ratio,
        landmark_normalization_version: sample.landmark_normalization_version,
        created_at: sample.created_at,
      }),
    ),
  };

  await writeFile(
    path.join(exportDir, "train.json"),
    JSON.stringify(splitBuckets.train, null, 2),
    "utf8",
  );
  await writeFile(
    path.join(exportDir, "val.json"),
    JSON.stringify(splitBuckets.val, null, 2),
    "utf8",
  );
  await writeFile(
    path.join(exportDir, "test.json"),
    JSON.stringify(splitBuckets.test, null, 2),
    "utf8",
  );
  await writeFile(
    path.join(exportDir, "manifest.json"),
    JSON.stringify(manifest, null, 2),
    "utf8",
  );
  await writeFile(
    path.join(exportDir, "summary.json"),
    JSON.stringify(summary, null, 2),
    "utf8",
  );

  return summary;
}

async function readLatestModelMetadata(
  recognitionLevel = DEFAULT_RECOGNITION_LEVEL,
  modelProfile = DEFAULT_MODEL_PROFILE,
) {
  const target = await resolveModelArtifactTarget({
    recognitionLevel,
    modelProfile,
  });

  if (!target) {
    const error = new Error("Model has not been trained yet");
    error.code = "ENOENT";
    throw error;
  }

  const metadataPath = path.join(target.modelDir, "metadata.json");
  const metadata = parseJsonText(await readFile(metadataPath, "utf8"));
  let evaluation = null;

  try {
    evaluation = parseJsonText(
      await readFile(path.join(target.modelDir, "evaluation.json"), "utf8"),
    );
  } catch {
    evaluation = metadata.evaluation ?? null;
  }

  return {
    ...metadata,
    recognition_level: target.recognitionLevel,
    model_profile: target.modelProfile ?? null,
    evaluation,
    files: {
      metadata: `${target.modelBaseUrl}/metadata.json`,
      weights: metadata.artifacts?.weights
        ? `${target.modelBaseUrl}/${path.basename(metadata.artifacts.weights)}`
        : null,
      evaluation: `${target.modelBaseUrl}/evaluation.json`,
    },
  };
}

async function readLatestModelEvaluation(
  recognitionLevel = DEFAULT_RECOGNITION_LEVEL,
) {
  const evaluationPath = path.join(
    getMlLatestDir(recognitionLevel),
    "evaluation.json",
  );
  return parseJsonText(await readFile(evaluationPath, "utf8"));
}

async function readLatestBenchmarkSummary(
  recognitionLevel = DEFAULT_RECOGNITION_LEVEL,
) {
  return parseJsonText(
    await readFile(getMlBenchmarkLatestPath(recognitionLevel), "utf8"),
  );
}

async function pathExists(targetPath) {
  try {
    await stat(targetPath);
    return true;
  } catch {
    return false;
  }
}

async function hasReadyDatasetExport(
  recognitionLevel = DEFAULT_RECOGNITION_LEVEL,
) {
  const datasetDir = getDatasetExportDir(recognitionLevel);
  const requiredFiles = ["train.json", "val.json", "test.json"];
  const checks = await Promise.all(
    requiredFiles.map((fileName) => pathExists(path.join(datasetDir, fileName))),
  );

  return checks.every(Boolean);
}

async function hasLatestModelArtifacts(
  recognitionLevel = DEFAULT_RECOGNITION_LEVEL,
) {
  return pathExists(path.join(getMlLatestDir(recognitionLevel), "metadata.json"));
}

async function hasModelArtifacts(
  recognitionLevel = DEFAULT_RECOGNITION_LEVEL,
  modelProfile = DEFAULT_MODEL_PROFILE,
) {
  return pathExists(
    path.join(getMlModelDir(recognitionLevel, modelProfile), "metadata.json"),
  );
}

async function resolveModelArtifactTarget({
  recognitionLevel = DEFAULT_RECOGNITION_LEVEL,
  modelProfile = DEFAULT_MODEL_PROFILE,
} = {}) {
  const normalizedRecognitionLevel = normalizeModelScope(recognitionLevel);
  const normalizedProfile = normalizeModelProfile(modelProfile);
  const candidateProfiles =
    normalizedProfile === DEFAULT_MODEL_PROFILE
      ? [DEFAULT_MODEL_PROFILE]
      : [normalizedProfile, DEFAULT_MODEL_PROFILE];

  for (const profile of candidateProfiles) {
    if (!(await hasModelArtifacts(normalizedRecognitionLevel, profile))) {
      continue;
    }

    const isProfileDir = profile !== DEFAULT_MODEL_PROFILE;
    return {
      recognitionLevel: normalizedRecognitionLevel,
      modelProfile: isProfileDir ? profile : null,
      modelDir: getMlModelDir(normalizedRecognitionLevel, profile),
      modelBaseUrl: isProfileDir
        ? `/ml-artifacts/${normalizedRecognitionLevel}/profiles/${profile}`
        : `/ml-artifacts/${normalizedRecognitionLevel}/latest`,
    };
  }

  return null;
}

async function resolvePublicModelScope({
  recognitionLevel = DEFAULT_MODEL_SCOPE,
  allowedRecognitionLevels = [],
} = {}) {
  const requestedScope = normalizeModelScope(recognitionLevel);
  const candidateScopes = [];

  const pushScope = (scope) => {
    if (
      typeof scope !== "string" ||
      !SUPPORTED_MODEL_SCOPES.includes(scope) ||
      candidateScopes.includes(scope)
    ) {
      return;
    }

    candidateScopes.push(scope);
  };

  pushScope(requestedScope);

  if (requestedScope === DEFAULT_MODEL_SCOPE) {
    normalizeRecognitionLevelList(allowedRecognitionLevels).forEach(pushScope);
  } else {
    pushScope(DEFAULT_MODEL_SCOPE);
  }

  for (const scope of candidateScopes) {
    if (await hasLatestModelArtifacts(scope)) {
      return scope;
    }
  }

  return null;
}

function formatDatasetExportRow(row) {
  return {
    id: row.id,
    sign_language: row.sign_language,
    recognition_level: row.recognition_level ?? DEFAULT_RECOGNITION_LEVEL,
    export_name: row.export_name,
    sample_count: row.sample_count,
    split_counts: row.split_counts ?? {},
    quality_summary: row.quality_summary ?? {},
    manifest_path: row.manifest_path,
    summary_path: row.summary_path,
    created_at: row.created_at,
  };
}

function formatModelRunRow(row) {
  return {
    id: row.id,
    run_type: row.run_type,
    model_type: row.model_type,
    sign_language: row.sign_language,
    recognition_level: row.recognition_level ?? DEFAULT_RECOGNITION_LEVEL,
    dataset_export_id: row.dataset_export_id,
    artifact_dir: row.artifact_dir,
    metrics_json: row.metrics_json ?? {},
    config_json: row.config_json ?? {},
    is_winner: row.is_winner,
    created_at: row.created_at,
  };
}

async function saveDatasetExportRecord(summary) {
  const result = await pool.query(
    `
      INSERT INTO dataset_exports (
        sign_language,
        recognition_level,
        export_name,
        sample_count,
        split_counts,
        quality_summary,
        manifest_path,
        summary_path
      )
      VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, $7, $8)
      RETURNING *
    `,
    [
      summary.sign_language ?? DEFAULT_USER_SETTINGS.signLanguage,
      summary.recognition_level ?? DEFAULT_RECOGNITION_LEVEL,
      "latest",
      Number(summary.sample_count ?? 0),
      JSON.stringify(summary.split_counts ?? {}),
      JSON.stringify(summary.quality_summary ?? {}),
      summary.files?.manifest ?? null,
      summary.files?.summary ?? null,
    ],
  );

  return formatDatasetExportRow(result.rows[0]);
}

async function findLatestDatasetExportId(
  signLanguage,
  recognitionLevel = DEFAULT_RECOGNITION_LEVEL,
) {
  const result = await pool.query(
    `
      SELECT id
      FROM dataset_exports
      WHERE sign_language = $1
        AND recognition_level = $2
      ORDER BY created_at DESC, id DESC
      LIMIT 1
    `,
    [signLanguage, normalizeModelScope(recognitionLevel)],
  );

  return result.rows[0]?.id ?? null;
}

async function saveModelRunRecord({
  runType,
  modelType,
  signLanguage,
  recognitionLevel = DEFAULT_RECOGNITION_LEVEL,
  datasetExportId = null,
  artifactDir = null,
  metrics = {},
  config = {},
  isWinner = false,
}) {
  const result = await pool.query(
    `
      INSERT INTO model_runs (
        run_type,
        model_type,
        sign_language,
        recognition_level,
        dataset_export_id,
        artifact_dir,
        metrics_json,
        config_json,
        is_winner
      )
      VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8::jsonb, $9)
      RETURNING *
    `,
    [
      runType,
      modelType,
      signLanguage ?? DEFAULT_USER_SETTINGS.signLanguage,
      normalizeModelScope(recognitionLevel),
      datasetExportId,
      artifactDir,
      JSON.stringify(metrics ?? {}),
      JSON.stringify(config ?? {}),
      isWinner,
    ],
  );

  return formatModelRunRow(result.rows[0]);
}

function buildWeakPhraseSummary(metrics = {}) {
  const perClass = metrics.per_class_accuracy ?? {};

  return Object.entries(perClass)
    .map(([label, value]) => ({
      label_key: label,
      label: value.label ?? label,
      recognition_level: value.recognition_level ?? null,
      unit_code: value.unit_code ?? null,
      sample_count: value.sample_count ?? 0,
      accuracy: value.accuracy ?? 0,
    }))
    .sort((left, right) => {
      if (left.accuracy !== right.accuracy) {
        return left.accuracy - right.accuracy;
      }

      return left.sample_count - right.sample_count;
    })
    .slice(0, 10);
}

function buildConfusionPairs(metrics = {}) {
  const matrix = metrics.confusion_matrix ?? [];
  const labels = metrics.labels ?? [];
  const labelKeys = metrics.label_keys ?? [];
  const classEntries = metrics.class_entries ?? [];
  const pairs = [];

  for (let actualIndex = 0; actualIndex < matrix.length; actualIndex += 1) {
    const row = matrix[actualIndex] ?? [];

    for (let predictedIndex = 0; predictedIndex < row.length; predictedIndex += 1) {
      if (actualIndex === predictedIndex) {
        continue;
      }

      const count = Number(row[predictedIndex] ?? 0);

      if (count <= 0) {
        continue;
      }

      pairs.push({
        actual_label_key: labelKeys[actualIndex] ?? `class_${actualIndex}`,
        actual_label: labels[actualIndex] ?? `class_${actualIndex}`,
        actual_recognition_level:
          classEntries[actualIndex]?.recognition_level ?? null,
        predicted_label_key:
          labelKeys[predictedIndex] ?? `class_${predictedIndex}`,
        predicted_label: labels[predictedIndex] ?? `class_${predictedIndex}`,
        predicted_recognition_level:
          classEntries[predictedIndex]?.recognition_level ?? null,
        count,
      });
    }
  }

  return pairs.sort((left, right) => right.count - left.count).slice(0, 10);
}

function buildModelComparisonRow(modelType, payload = {}) {
  const evaluation = payload.evaluation ?? {};
  const testMetrics = evaluation.splits?.test ?? {};
  const valMetrics = evaluation.splits?.val ?? {};

  return {
    model_type: modelType,
    label_count: payload.labelCount ?? payload.labels?.length ?? 0,
    input_size: payload.inputSize ?? payload.input_size ?? 0,
    metrics: {
      train: evaluation.splits?.train ?? null,
      val: valMetrics,
      test: testMetrics,
    },
    weak_phrases: buildWeakPhraseSummary(testMetrics),
    confusion_pairs: buildConfusionPairs(testMetrics),
    files: {
      metadata: payload.metadataPath ?? null,
      weights: payload.weightsPath ?? null,
      evaluation: payload.evaluationPath ?? null,
    },
  };
}

function scoreBenchmarkModel(row) {
  const testMetrics = row.metrics?.test ?? {};
  const top1 = Number(testMetrics.top1_accuracy ?? testMetrics.accuracy ?? 0);
  const top3 = Number(testMetrics.top3_accuracy ?? 0);
  const lowConfidenceRate = Number(testMetrics.low_confidence_rate ?? 1);
  const latency = Number(testMetrics.latency_ms_avg ?? Number.MAX_SAFE_INTEGER);

  return [
    top1,
    top3,
    -lowConfidenceRate,
    -latency,
  ];
}

function compareBenchmarkScores(left, right) {
  const leftScore = scoreBenchmarkModel(left);
  const rightScore = scoreBenchmarkModel(right);

  for (let index = 0; index < leftScore.length; index += 1) {
    if (leftScore[index] !== rightScore[index]) {
      return rightScore[index] - leftScore[index];
    }
  }

  return 0;
}

const trainingVideoStorage = multer.diskStorage({
  destination(req, file, callback) {
    callback(null, TRAINING_UPLOADS_DIR);
  },
  filename(req, file, callback) {
    const extension = path.extname(file.originalname) || ".webm";
    callback(
      null,
      `${Date.now()}-${Math.round(Math.random() * 1e9)}${extension.slice(0, 10)}`,
    );
  },
});

const trainingVideoUpload = multer({
  storage: trainingVideoStorage,
  limits: {
    fileSize: 50 * 1024 * 1024,
  },
  fileFilter(req, file, callback) {
    if (!file.mimetype || file.mimetype.startsWith("video/")) {
      callback(null, true);
      return;
    }

    callback(new Error("Only video files are allowed"));
  },
});

app.get("/api/health", async (req, res) => {
  try {
    const result = await pool.query("SELECT NOW() as now");
    res.json({
      ok: true,
      db: "connected",
      time: result.rows[0].now,
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({
      ok: false,
      error: "Database connection failed",
    });
  }
});

app.get("/api/gestures/status", (req, res) => {
  res.json({
    ok: true,
    ...gestureBridgeService.getStatus(),
  });
});

app.post("/api/gestures/start", async (req, res) => {
  try {
    const status = await gestureBridgeService.start();
    res.json({
      ok: true,
      ...status,
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({
      ok: false,
      error: "Failed to start gesture bridge worker",
      details: error.message,
    });
  }
});

app.post("/api/gestures/stop", async (req, res) => {
  try {
    const status = await gestureBridgeService.stop();
    res.json({
      ok: true,
      ...status,
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({
      ok: false,
      error: "Failed to stop gesture bridge worker",
      details: error.message,
    });
  }
});

app.post("/api/gestures/reset", async (req, res) => {
  try {
    const status = await gestureBridgeService.reset();
    res.json({
      ok: true,
      ...status,
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({
      ok: false,
      error: "Failed to reset gesture bridge worker",
      details: error.message,
    });
  }
});

app.get("/api/history", async (req, res) => {
  try {
    const result = await pool.query(
      `
        SELECT id, type, text, created_at
        FROM message_history
        ORDER BY created_at DESC
      `,
    );

    res.json(result.rows);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Failed to fetch history" });
  }
});

app.post("/api/history", async (req, res) => {
  try {
    const type = req.body.type?.trim();
    const text = req.body.text?.trim();
    const userEmail = req.body.userEmail?.trim() || DEFAULT_USER.email;

    if (!type || !text) {
      return res.status(400).json({ error: "type and text are required" });
    }

    const userResult = await pool.query(
      `
        SELECT id
        FROM app_users
        WHERE email = $1
        LIMIT 1
      `,
      [userEmail],
    );

    const phraseResult = await pool.query(
      `
        SELECT id
        FROM phrase_library
        WHERE text = $1
        LIMIT 1
      `,
      [text],
    );

    const userId = userResult.rows[0]?.id ?? null;
    const phraseId = phraseResult.rows[0]?.id ?? null;

    const result = await pool.query(
      `
        INSERT INTO message_history (type, text, user_id, phrase_id)
        VALUES ($1, $2, $3, $4)
        RETURNING id, type, text, created_at
      `,
      [type, text, userId, phraseId],
    );

    res.status(201).json(result.rows[0]);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Failed to save history" });
  }
});

app.delete("/api/history", async (req, res) => {
  try {
    await pool.query("DELETE FROM message_history");
    res.json({ ok: true });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Failed to clear history" });
  }
});

app.get("/api/recognitions", async (req, res) => {
  try {
    const values = [];
    const conditions = [];

    if (req.query.userEmail?.trim()) {
      values.push(req.query.userEmail.trim());
      conditions.push(`app_users.email = $${values.length}`);
    }

    if (req.query.sourceMode?.trim()) {
      values.push(req.query.sourceMode.trim());
      conditions.push(`recognition_runs.source_mode = $${values.length}`);
    }

    if (req.query.signLanguage?.trim()) {
      values.push(req.query.signLanguage.trim());
      conditions.push(`recognition_runs.sign_language = $${values.length}`);
    }

    if (req.query.recognitionLevel?.trim()) {
      values.push(normalizeRecognitionLevel(req.query.recognitionLevel.trim()));
      conditions.push(`recognition_runs.recognition_level = $${values.length}`);
    }

    const whereClause = conditions.length
      ? `WHERE ${conditions.join(" AND ")}`
      : "";

    const result = await pool.query(
      `
        SELECT
          recognition_runs.id,
          recognition_runs.user_id,
          app_users.email AS user_email,
          recognition_runs.phrase_id,
          phrase_library.text AS phrase_text,
          recognition_runs.sign_language,
          recognition_runs.recognition_level,
          recognition_runs.source_mode,
          recognition_runs.recognized_text,
          recognition_runs.confidence,
          recognition_runs.raw_output_json,
          recognition_runs.created_at
        FROM recognition_runs
        LEFT JOIN app_users
          ON app_users.id = recognition_runs.user_id
        LEFT JOIN phrase_library
          ON phrase_library.id = recognition_runs.phrase_id
        ${whereClause}
        ORDER BY recognition_runs.created_at DESC, recognition_runs.id DESC
        LIMIT 100
      `,
      values,
    );

    res.json(result.rows.map(formatRecognitionRunRow));
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Failed to fetch recognition runs" });
  }
});

app.post("/api/recognitions", async (req, res) => {
  try {
    const recognizedText = req.body.recognizedText?.trim();
    const sourceMode = req.body.sourceMode?.trim() || "rules";
    const signLanguage =
      req.body.signLanguage?.trim() || DEFAULT_USER_SETTINGS.signLanguage;
    const confidence =
      typeof req.body.confidence === "number" ? req.body.confidence : null;
    const userEmail = req.body.userEmail?.trim() || DEFAULT_USER.email;
    const phraseId = Number(req.body.phraseId);
    const rawOutput =
      req.body.rawOutput && typeof req.body.rawOutput === "object"
        ? req.body.rawOutput
        : {};

    if (!recognizedText) {
      return res.status(400).json({ error: "recognizedText is required" });
    }

    const requestedRecognitionLevel =
      typeof req.body.recognitionLevel === "string" && req.body.recognitionLevel.trim()
        ? normalizeRecognitionLevel(req.body.recognitionLevel)
        : null;
    const [user, phrase] = await Promise.all([
      findUserByEmail(userEmail),
      findPhraseByIdOrText({
        phraseId,
        text: recognizedText,
        signLanguage,
        recognitionLevel: requestedRecognitionLevel,
      }),
    ]);
    const recognitionLevel = normalizeRecognitionLevel(
      requestedRecognitionLevel ?? phrase?.recognition_level,
    );

    const result = await pool.query(
      `
        INSERT INTO recognition_runs (
          user_id,
          phrase_id,
          sign_language,
          recognition_level,
          source_mode,
          recognized_text,
          confidence,
          raw_output_json
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
        RETURNING
          id,
          user_id,
          phrase_id,
          sign_language,
          recognition_level,
          source_mode,
          recognized_text,
          confidence,
          raw_output_json,
          created_at
      `,
      [
        user?.id ?? null,
        phrase?.id ?? null,
        signLanguage,
        phrase?.recognition_level ?? recognitionLevel,
        sourceMode,
        recognizedText,
        confidence,
        JSON.stringify(rawOutput),
      ],
    );

    res.status(201).json(
      formatRecognitionRunRow({
        ...result.rows[0],
        user_email: user?.email ?? null,
        phrase_text: phrase?.text ?? null,
      }),
    );
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Failed to save recognition run" });
  }
});

app.get("/api/training-videos", async (req, res) => {
  try {
    const values = [];
    const conditions = [];

    if (req.query.userEmail?.trim()) {
      values.push(req.query.userEmail.trim());
      conditions.push(`app_users.email = $${values.length}`);
    }

    if (req.query.phraseId?.trim()) {
      values.push(Number(req.query.phraseId));
      conditions.push(`training_videos.phrase_id = $${values.length}`);
    }

    if (req.query.signLanguage?.trim()) {
      values.push(req.query.signLanguage.trim());
      conditions.push(`training_videos.sign_language = $${values.length}`);
    }

    if (req.query.status?.trim()) {
      values.push(req.query.status.trim());
      conditions.push(`training_videos.status = $${values.length}`);
    }

    if (req.query.datasetSplit?.trim()) {
      values.push(req.query.datasetSplit.trim());
      conditions.push(`training_videos.dataset_split = $${values.length}`);
    }

    if (req.query.reviewStatus?.trim()) {
      values.push(req.query.reviewStatus.trim());
      conditions.push(`training_videos.review_status = $${values.length}`);
    }

    if (req.query.recognitionLevel?.trim()) {
      values.push(normalizeRecognitionLevel(req.query.recognitionLevel.trim()));
      conditions.push(`phrase_library.recognition_level = $${values.length}`);
    }

    const whereClause = conditions.length
      ? `WHERE ${conditions.join(" AND ")}`
      : "";

    const result = await pool.query(
      `
        ${TRAINING_VIDEO_SELECT}
        ${whereClause}
        ORDER BY training_videos.created_at DESC, training_videos.id DESC
      `,
      values,
    );

    res.json(result.rows.map(formatTrainingVideoRow));
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Failed to fetch training videos" });
  }
});

app.post("/api/training-videos", (req, res) => {
  trainingVideoUpload.single("video")(req, res, async (uploadError) => {
    if (uploadError) {
      return res.status(400).json({ error: uploadError.message });
    }

    async function cleanupUploadedFile() {
      if (req.file?.path) {
        await unlink(req.file.path).catch(() => {});
      }
    }

    try {
      const phraseId = Number(req.body.phraseId);
      const signLanguage =
        req.body.signLanguage?.trim() || DEFAULT_USER_SETTINGS.signLanguage;
      const userEmail = req.body.userEmail?.trim() || DEFAULT_USER.email;
      const durationMs = Math.max(0, Number(req.body.durationMs) || 0);
      const rawStatus = req.body.status;
      const rawDatasetSplit = req.body.datasetSplit;
      const status = normalizeTrainingVideoStatus(rawStatus, "draft");
      const datasetSplit = normalizeDatasetSplit(rawDatasetSplit, "unassigned");

      if (!req.file) {
        return res.status(400).json({ error: "video is required" });
      }

      if (!Number.isFinite(phraseId) || phraseId <= 0) {
        await cleanupUploadedFile();
        return res.status(400).json({ error: "phraseId is required" });
      }

      if (
        typeof rawStatus === "string" &&
        rawStatus.trim() &&
        status !== rawStatus.trim().toLowerCase()
      ) {
        await cleanupUploadedFile();
        return res.status(400).json({
          error: `Invalid status. Supported values: ${SUPPORTED_TRAINING_VIDEO_STATUSES.join(", ")}`,
        });
      }

      if (
        typeof rawDatasetSplit === "string" &&
        rawDatasetSplit.trim() &&
        datasetSplit !== rawDatasetSplit.trim().toLowerCase()
      ) {
        await cleanupUploadedFile();
        return res.status(400).json({
          error: `Invalid datasetSplit. Supported values: ${SUPPORTED_DATASET_SPLITS.join(", ")}`,
        });
      }

      const phraseResult = await pool.query(
        `
          SELECT id, entry_type, recognition_level
          FROM phrase_library
          WHERE id = $1
          LIMIT 1
        `,
        [phraseId],
      );

      if (phraseResult.rows.length === 0) {
        await cleanupUploadedFile();
        return res.status(404).json({ error: "Phrase not found" });
      }

      const user = await findUserByEmail(userEmail);
      const storedVideoPath = path.posix.join(
        "training-videos",
        req.file.filename,
      );

      const createdResult = await pool.query(
        `
          INSERT INTO training_videos (
            user_id,
            phrase_id,
            label_type,
            sign_language,
            signer_label,
            video_path,
            duration_ms,
            dataset_split,
            status
          )
          VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
          RETURNING id
        `,
        [
          user?.id ?? null,
          phraseId,
          phraseResult.rows[0].entry_type ??
            phraseResult.rows[0].recognition_level ??
            DEFAULT_RECOGNITION_LEVEL,
          signLanguage,
          req.body.signerLabel?.trim() || user?.display_name || user?.email || null,
          storedVideoPath,
          durationMs,
          datasetSplit,
          status,
        ],
      );

      const trainingVideo = await getTrainingVideoById(createdResult.rows[0].id);
      res.status(201).json(trainingVideo);
    } catch (error) {
      await cleanupUploadedFile();
      console.error(error);
      res.status(500).json({ error: "Failed to save training video" });
    }
  });
});

app.post("/api/training-videos/:id/extract-landmarks", async (req, res) => {
  try {
    const trainingVideoId = Number(req.params.id);

    if (!Number.isFinite(trainingVideoId) || trainingVideoId <= 0) {
      return res.status(400).json({ error: "Invalid training video id" });
    }

    const trainingVideo = await getTrainingVideoById(trainingVideoId);

    if (!trainingVideo) {
      return res.status(404).json({ error: "Training video not found" });
    }

    const videoFilePath = resolveUploadFilePath(trainingVideo.video_path);

    if (!videoFilePath) {
      return res.status(404).json({ error: "Training video file not found" });
    }

    const storedLandmarkPath = path.posix.join(
      "landmarks",
      `training-video-${trainingVideoId}.json`,
    );
    const outputFilePath = resolveUploadFilePath(storedLandmarkPath);

    await upsertLandmarkSequence({
      trainingVideoId,
      filePath: storedLandmarkPath,
      frameCount: 0,
      status: "processing",
      normalizationVersion: "holistic_v2",
      errorMessage: null,
    });

    try {
      const extractionResult = await runLandmarkExtraction({
        inputPath: videoFilePath,
        outputPath: outputFilePath,
        signLanguage: trainingVideo.sign_language,
        trainingVideoId,
        phraseId: trainingVideo.phrase_id,
      });

      await upsertLandmarkSequence({
        trainingVideoId,
        filePath: storedLandmarkPath,
        frameCount: Number(extractionResult.frameCount) || 0,
        status: "ready",
        validFrameRatio: extractionResult.summary?.valid_frame_ratio ?? null,
        missingHandRatio: extractionResult.summary?.missing_hand_ratio ?? null,
        missingFaceRatio: extractionResult.summary?.missing_face_ratio ?? null,
        missingPoseRatio: extractionResult.summary?.missing_pose_ratio ?? null,
        normalizationVersion: "holistic_v2",
        errorMessage: null,
      });
    } catch (error) {
      await upsertLandmarkSequence({
        trainingVideoId,
        filePath: storedLandmarkPath,
        frameCount: 0,
        status: "failed",
        normalizationVersion: "holistic_v2",
        errorMessage: error.message,
      });

      console.error(error);
      return res.status(500).json({
        error: "Failed to extract landmarks",
        details: error.message,
      });
    }

    const updatedTrainingVideo = await getTrainingVideoById(trainingVideoId);
    res.json(updatedTrainingVideo);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Failed to extract landmarks" });
  }
});

app.put("/api/training-videos/:id/review", async (req, res) => {
  try {
    const trainingVideoId = Number(req.params.id);

    if (!Number.isFinite(trainingVideoId) || trainingVideoId <= 0) {
      return res.status(400).json({ error: "Invalid training video id" });
    }

    const reviewStatus =
      typeof req.body.reviewStatus === "string" && req.body.reviewStatus.trim()
        ? req.body.reviewStatus.trim()
        : "pending";
    const reviewNotes =
      typeof req.body.reviewNotes === "string" ? req.body.reviewNotes.trim() : "";
    const reviewerEmail = req.body.reviewerEmail?.trim() || DEFAULT_USER.email;
    const qualityScoreRaw =
      req.body.qualityScore === null || req.body.qualityScore === undefined
        ? null
        : Number(req.body.qualityScore);
    const qualityScore =
      qualityScoreRaw === null
        ? null
        : Number.isFinite(qualityScoreRaw) && qualityScoreRaw >= 1 && qualityScoreRaw <= 5
          ? Math.round(qualityScoreRaw)
          : NaN;

    if (!["pending", "approved", "rejected", "needs_retake"].includes(reviewStatus)) {
      return res.status(400).json({ error: "Invalid reviewStatus" });
    }

    if (Number.isNaN(qualityScore)) {
      return res.status(400).json({ error: "qualityScore must be between 1 and 5" });
    }

    const existingVideo = await getTrainingVideoById(trainingVideoId);

    if (!existingVideo) {
      return res.status(404).json({ error: "Training video not found" });
    }

    const reviewer = await findUserByEmail(reviewerEmail);

    await pool.query(
      `
        UPDATE training_videos
        SET
          quality_score = $1,
          review_status = $2,
          review_notes = $3,
          reviewed_by = $4,
          reviewed_at = NOW()
        WHERE id = $5
      `,
      [
        qualityScore,
        reviewStatus,
        reviewNotes || null,
        reviewer?.id ?? null,
        trainingVideoId,
      ],
    );

    const updatedVideo = await getTrainingVideoById(trainingVideoId);
    res.json(updatedVideo);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Failed to update training video review" });
  }
});

app.delete("/api/training-videos/:id", async (req, res) => {
  try {
    const trainingVideoId = Number(req.params.id);

    if (!Number.isFinite(trainingVideoId) || trainingVideoId <= 0) {
      return res.status(400).json({ error: "Invalid training video id" });
    }

    const existingResult = await pool.query(
      `
        SELECT
          training_videos.id,
          training_videos.video_path,
          landmark_sequences.file_path AS landmark_file_path
        FROM training_videos
        LEFT JOIN landmark_sequences
          ON landmark_sequences.training_video_id = training_videos.id
        WHERE training_videos.id = $1
        LIMIT 1
      `,
      [trainingVideoId],
    );

    const existingVideo = existingResult.rows[0];

    if (!existingVideo) {
      return res.status(404).json({ error: "Training video not found" });
    }

    await pool.query("DELETE FROM training_videos WHERE id = $1", [
      trainingVideoId,
    ]);

    const videoFilePath = resolveUploadFilePath(existingVideo.video_path);
    const landmarkFilePath = resolveUploadFilePath(existingVideo.landmark_file_path);

    if (videoFilePath) {
      await unlink(videoFilePath).catch(() => {});
    }

    if (landmarkFilePath) {
      await unlink(landmarkFilePath).catch(() => {});
    }

    res.status(204).send();
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Failed to delete training video" });
  }
});

app.get("/api/dataset/export/latest", async (req, res) => {
  try {
    const recognitionLevel = normalizeModelScope(
      req.query.recognitionLevel,
    );
    const summaryPath = path.join(
      getDatasetExportDir(recognitionLevel),
      "summary.json",
    );
    const summary = JSON.parse(await readFile(summaryPath, "utf8"));
    res.json(summary);
  } catch (error) {
    if (error?.code === "ENOENT") {
      return res.status(404).json({ error: "Dataset export has not been prepared yet" });
    }

    console.error(error);
    res.status(500).json({ error: "Failed to read latest dataset export" });
  }
});

app.get("/api/dataset/exports", async (req, res) => {
  try {
    const signLanguage =
      req.query.signLanguage?.trim() || DEFAULT_USER_SETTINGS.signLanguage;
    const recognitionLevel = normalizeModelScope(
      req.query.recognitionLevel,
    );
    const limit = Math.min(20, Math.max(1, Number(req.query.limit) || 10));

    const result = await pool.query(
      `
        SELECT *
        FROM dataset_exports
        WHERE sign_language = $1
          AND recognition_level = $2
        ORDER BY created_at DESC, id DESC
        LIMIT $3
      `,
      [signLanguage, recognitionLevel, limit],
    );

    res.json(result.rows.map(formatDatasetExportRow));
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Failed to fetch dataset exports" });
  }
});

app.get("/api/dataset/analytics", async (req, res) => {
  try {
    const signLanguage =
      req.query.signLanguage?.trim() || DEFAULT_USER_SETTINGS.signLanguage;
    const recognitionLevel = normalizeModelScope(
      req.query.recognitionLevel,
    );
    const analytics = await buildDatasetAnalytics(
      signLanguage,
      recognitionLevel,
    );
    res.json(analytics);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Failed to build dataset analytics" });
  }
});

app.post("/api/dataset/export", async (req, res) => {
  try {
    const signLanguage =
      req.body.signLanguage?.trim() || DEFAULT_USER_SETTINGS.signLanguage;
    const recognitionLevel = normalizeModelScope(
      req.body.recognitionLevel,
    );
    const ratios = {
      train: clampRatio(req.body.trainRatio, 0.7),
      val: clampRatio(req.body.valRatio, 0.15),
      test: clampRatio(req.body.testRatio, 0.15),
    };

    const ratioTotal = ratios.train + ratios.val + ratios.test;
    ratios.train /= ratioTotal;
    ratios.val /= ratioTotal;
    ratios.test /= ratioTotal;

    const datasetPayload = await fetchReadyDatasetRows(
      signLanguage,
      recognitionLevel,
    );
    const rows = datasetPayload.rows;

    if (rows.length === 0) {
      return res.status(400).json({
        error: "No approved landmark sequences passed dataset quality thresholds",
      });
    }

    const assignmentPayload = assignDatasetSplits(rows, ratios);

    if (isUnifiedModelScope(recognitionLevel)) {
      await pool.query(
        `
          UPDATE training_videos
          SET dataset_split = 'unassigned'
          WHERE sign_language = $1
        `,
        [signLanguage],
      );
    } else {
      await pool.query(
        `
          UPDATE training_videos
          SET dataset_split = 'unassigned'
          WHERE sign_language = $1
            AND phrase_id IN (
              SELECT id
              FROM phrase_library
              WHERE recognition_level = $2
            )
        `,
        [signLanguage, recognitionLevel],
      );
    }

    const updates = rows.map((row) =>
      pool.query(
        `
          UPDATE training_videos
          SET dataset_split = $1
          WHERE id = $2
        `,
        [assignmentPayload.assignments.get(row.id) ?? "train", row.id],
      ),
    );

    await Promise.all(updates);

    const summary = await writeDatasetExport(
      rows,
      assignmentPayload,
      "latest",
      datasetPayload,
      recognitionLevel,
    );
    await saveDatasetExportRecord(summary);
    res.json(summary);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Failed to prepare dataset export" });
  }
});

app.get("/api/ml/model/latest", async (req, res) => {
  try {
    const allowedRecognitionLevels = normalizeRecognitionLevelList(
      req.query.allowedRecognitionLevels,
    );
    const modelProfile = normalizeModelProfile(req.query.profile);
    const recognitionLevel = await resolvePublicModelScope({
      recognitionLevel: req.query.recognitionLevel,
      allowedRecognitionLevels,
    });

    if (!recognitionLevel) {
      return res.status(404).json({ error: "Model has not been trained yet" });
    }

    const metadata = await readLatestModelMetadata(
      recognitionLevel,
      modelProfile,
    );
    getPredictionWorker(recognitionLevel, metadata.model_profile ?? undefined).catch(
      () => {},
    );
    res.json(metadata);
  } catch (error) {
    if (error?.code === "ENOENT") {
      return res.status(404).json({ error: "Model has not been trained yet" });
    }

    console.error(error);
    res.status(500).json({ error: "Failed to read latest model metadata" });
  }
});

app.get("/api/ml/evaluation/latest", async (req, res) => {
  try {
    const recognitionLevel = normalizeModelScope(
      req.query.recognitionLevel,
    );
    const evaluation = await readLatestModelEvaluation(recognitionLevel);
    res.json(evaluation);
  } catch (error) {
    if (error?.code === "ENOENT") {
      return res.status(404).json({ error: "Model evaluation is not available yet" });
    }

    console.error(error);
    res.status(500).json({ error: "Failed to read latest model evaluation" });
  }
});

app.get("/api/ml/benchmark/latest", async (req, res) => {
  try {
    const recognitionLevel = normalizeModelScope(
      req.query.recognitionLevel,
    );
    const summary = await readLatestBenchmarkSummary(recognitionLevel);
    res.json(summary);
  } catch (error) {
    if (error?.code === "ENOENT") {
      return res.status(404).json({ error: "Model benchmark is not available yet" });
    }

    console.error(error);
    res.status(500).json({ error: "Failed to read latest benchmark summary" });
  }
});

app.get("/api/model-runs", async (req, res) => {
  try {
    const signLanguage =
      req.query.signLanguage?.trim() || DEFAULT_USER_SETTINGS.signLanguage;
    const recognitionLevel = normalizeModelScope(
      req.query.recognitionLevel,
    );
    const limit = Math.min(30, Math.max(1, Number(req.query.limit) || 15));

    const result = await pool.query(
      `
        SELECT *
        FROM model_runs
        WHERE sign_language = $1
          AND recognition_level = $2
        ORDER BY created_at DESC, id DESC
        LIMIT $3
      `,
      [signLanguage, recognitionLevel, limit],
    );

    res.json(result.rows.map(formatModelRunRow));
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Failed to fetch model runs" });
  }
});

app.post("/api/ml/train", async (req, res) => {
  try {
    const signLanguage =
      req.body.signLanguage?.trim() || DEFAULT_USER_SETTINGS.signLanguage;
    const recognitionLevel = normalizeModelScope(
      req.body.recognitionLevel,
    );
    const datasetReady = await hasReadyDatasetExport(recognitionLevel);

    if (!datasetReady) {
      return res.status(409).json({
        error: "Dataset export is not ready",
        details:
          "Prepare a dataset export from approved landmark sequences before training the model.",
      });
    }

    const datasetDir = getDatasetExportDir(recognitionLevel);
    const outputDir = getMlLatestDir(recognitionLevel);
    const modelType = req.body.modelType?.trim() || "baseline";
    const defaults = getModelTrainingDefaults(recognitionLevel, modelType);
    const epochs = Number(req.body.epochs) || defaults.epochs || undefined;
    const maxSequenceLength =
      Number(req.body.maxSequenceLength) || defaults.maxSequenceLength || undefined;
    const hiddenSize = Number(req.body.hiddenSize) || defaults.hiddenSize || undefined;
    const classBalance = req.body.classBalance?.trim() || undefined;
    const classWeightPower = Number(req.body.classWeightPower) || undefined;
    const focusLabelKeys = Array.isArray(req.body.focusLabelKeys)
      ? req.body.focusLabelKeys
          .map((value) => String(value || "").trim())
          .filter(Boolean)
      : [];
    const focusWeightMultiplier =
      Number(req.body.focusWeightMultiplier) || undefined;
    const confidenceThreshold =
      Number(req.body.confidenceThreshold) || defaults.confidenceThreshold || 0.85;
    const datasetExportId = await findLatestDatasetExportId(
      signLanguage,
      recognitionLevel,
    );

    const args = [
      "--dataset-dir",
      datasetDir,
      "--output-dir",
      outputDir,
      "--confidence-threshold",
      String(confidenceThreshold),
      "--model-type",
      modelType,
    ];

    if (epochs) {
      args.push("--epochs", String(epochs));
    }

    if (maxSequenceLength) {
      args.push("--max-sequence-length", String(maxSequenceLength));
    }

    if (hiddenSize) {
      args.push("--hidden-size", String(hiddenSize));
    }

    if (classBalance) {
      args.push("--class-balance", String(classBalance));
    }

    if (classWeightPower) {
      args.push("--class-weight-power", String(classWeightPower));
    }

    if (focusLabelKeys.length) {
      args.push("--focus-labels", focusLabelKeys.join(","));
    }

    if (focusWeightMultiplier) {
      args.push("--focus-weight-multiplier", String(focusWeightMultiplier));
    }

    const result = await runPythonJsonScript(TRAIN_SEQUENCE_MODEL_SCRIPT, args);
    await saveModelRunRecord({
      runType: "train",
      modelType,
      signLanguage,
      recognitionLevel,
      datasetExportId,
      artifactDir: outputDir,
      metrics: result.evaluation ?? result.metrics ?? {},
      config: {
        epochs,
        maxSequenceLength,
        hiddenSize,
        classBalance,
        classWeightPower,
        focusLabelKeys,
        focusWeightMultiplier,
        confidenceThreshold,
      },
      isWinner: false,
    });

    res.json(result);
  } catch (error) {
    console.error(error);
    res.status(500).json({
      error: "Failed to train sequence model",
      details: error.message,
    });
  }
});

app.post("/api/ml/benchmark", async (req, res) => {
  try {
    const signLanguage =
      req.body.signLanguage?.trim() || DEFAULT_USER_SETTINGS.signLanguage;
    const recognitionLevel = normalizeModelScope(
      req.body.recognitionLevel,
    );
    const datasetReady = await hasReadyDatasetExport(recognitionLevel);

    if (!datasetReady) {
      return res.status(409).json({
        error: "Dataset export is not ready",
        details:
          "Prepare a dataset export from approved landmark sequences before running the benchmark.",
      });
    }

    const datasetDir = getDatasetExportDir(recognitionLevel);
    const benchmarkDefaults = getModelTrainingDefaults(recognitionLevel, "tcn");
    const confidenceThresholdValue =
      Number(req.body.confidenceThreshold) ||
      benchmarkDefaults.confidenceThreshold ||
      0.85;
    const confidenceThreshold = String(confidenceThresholdValue);
    const datasetExportId = await findLatestDatasetExportId(
      signLanguage,
      recognitionLevel,
    );
    const requestedModelTypes = Array.isArray(req.body.modelTypes)
      ? req.body.modelTypes
      : MODEL_BENCHMARK_TYPES;
    const modelTypes = requestedModelTypes.filter((modelType) =>
      MODEL_BENCHMARK_TYPES.includes(String(modelType).trim()),
    );

    if (!modelTypes.length) {
      return res.status(400).json({ error: "At least one valid model type is required" });
    }

    const results = [];

    for (const modelType of modelTypes) {
      const outputDir = path.join(getMlCandidatesDir(recognitionLevel), modelType);
      await rm(outputDir, { recursive: true, force: true });
      await mkdir(outputDir, { recursive: true });

      const args = [
        "--dataset-dir",
        datasetDir,
        "--output-dir",
        outputDir,
        "--confidence-threshold",
        confidenceThreshold,
        "--model-type",
        modelType,
      ];

      const defaults = getModelTrainingDefaults(recognitionLevel, modelType);

      if (defaults.epochs) {
        args.push("--epochs", String(defaults.epochs));
      }

      if (defaults.maxSequenceLength) {
        args.push("--max-sequence-length", String(defaults.maxSequenceLength));
      }

      if (defaults.hiddenSize) {
        args.push("--hidden-size", String(defaults.hiddenSize));
      }

      const result = await runPythonJsonScript(TRAIN_SEQUENCE_MODEL_SCRIPT, args);
      results.push(buildModelComparisonRow(modelType, result));
    }

    const rankedModels = [...results].sort(compareBenchmarkScores);
    const winner = rankedModels[0];
    const winnerDir = path.join(
      getMlCandidatesDir(recognitionLevel),
      winner.model_type,
    );
    const latestDir = getMlLatestDir(recognitionLevel);

    await rm(latestDir, { recursive: true, force: true });
    await cp(winnerDir, latestDir, { recursive: true });

    const summary = {
      generated_at: new Date().toISOString(),
      recognition_level: recognitionLevel,
      confidence_threshold: Number(confidenceThreshold),
      evaluated_models: rankedModels.map((row) => ({
        model_type: row.model_type,
        metrics: row.metrics,
        weak_phrases: row.weak_phrases,
        confusion_pairs: row.confusion_pairs,
      })),
      winner: {
        model_type: winner.model_type,
        metrics: winner.metrics,
        weak_phrases: winner.weak_phrases,
        confusion_pairs: winner.confusion_pairs,
      },
      files: {
        latest_model_metadata: `/ml-artifacts/${recognitionLevel}/latest/metadata.json`,
        latest_model_evaluation: `/ml-artifacts/${recognitionLevel}/latest/evaluation.json`,
      },
    };

    await writeFile(
      getMlBenchmarkLatestPath(recognitionLevel),
      JSON.stringify(summary, null, 2),
      "utf8",
    );

    for (const model of rankedModels) {
      await saveModelRunRecord({
        runType: "benchmark",
        modelType: model.model_type,
        signLanguage,
        recognitionLevel,
        datasetExportId,
        artifactDir: path.join(getMlCandidatesDir(recognitionLevel), model.model_type),
        metrics: model.metrics,
        config: {
          confidenceThreshold: confidenceThresholdValue,
        },
        isWinner: model.model_type === winner.model_type,
      });
    }

    res.json(summary);
  } catch (error) {
    console.error(error);
    res.status(500).json({
      error: "Failed to benchmark sequence models",
      details: error.message,
    });
  }
});

app.post("/api/ml/evaluate", async (req, res) => {
  try {
    const signLanguage =
      req.body.signLanguage?.trim() || DEFAULT_USER_SETTINGS.signLanguage;
    const recognitionLevel = normalizeModelScope(
      req.body.recognitionLevel,
    );
    const datasetReady = await hasReadyDatasetExport(recognitionLevel);
    const modelReady = await hasLatestModelArtifacts(recognitionLevel);

    if (!datasetReady) {
      return res.status(409).json({
        error: "Dataset export is not ready",
        details:
          "Prepare a dataset export from approved landmark sequences before evaluating the model.",
      });
    }

    if (!modelReady) {
      return res.status(409).json({
        error: "Model is not trained yet",
        details:
          "Train a model first, then run evaluation on the latest export.",
      });
    }

    const datasetDir = getDatasetExportDir(recognitionLevel);
    const modelDir = getMlLatestDir(recognitionLevel);
    const evaluationDefaults = getModelTrainingDefaults(recognitionLevel, "tcn");
    const confidenceThreshold =
      Number(req.body.confidenceThreshold) ||
      evaluationDefaults.confidenceThreshold ||
      0.85;
    const datasetExportId = await findLatestDatasetExportId(
      signLanguage,
      recognitionLevel,
    );

    const result = await runPythonJsonScript(EVALUATE_SEQUENCE_MODEL_SCRIPT, [
      "--dataset-dir",
      datasetDir,
      "--model-dir",
      modelDir,
      "--confidence-threshold",
      String(confidenceThreshold),
    ]);
    const latestMetadata = await readLatestModelMetadata(recognitionLevel);
    await saveModelRunRecord({
      runType: "evaluate",
      modelType: latestMetadata.model_type ?? "baseline",
      signLanguage,
      recognitionLevel,
      datasetExportId,
      artifactDir: modelDir,
      metrics: result.evaluation ?? {},
      config: {
        confidenceThreshold,
      },
      isWinner: false,
    });

    res.json(result);
  } catch (error) {
    console.error(error);
    res.status(500).json({
      error: "Failed to evaluate sequence model",
      details: error.message,
    });
  }
});

app.post("/api/ml/predict", async (req, res) => {
  try {
    const allowedRecognitionLevels = normalizeRecognitionLevelList(
      req.body.allowedRecognitionLevels,
    );
    const requestedModelProfile = normalizeModelProfile(
      req.body.profile ?? req.body.modelProfile,
    );
    const recognitionLevel = await resolvePublicModelScope({
      recognitionLevel: req.body.recognitionLevel,
      allowedRecognitionLevels,
    });
    const allowedLabelKeys = Array.isArray(req.body.allowedLabelKeys)
      ? req.body.allowedLabelKeys
          .map((value) => String(value ?? "").trim())
          .filter(Boolean)
      : [];
    const landmarkPath = req.body.landmarkPath?.trim();
    const trainingVideoId = Number(req.body.trainingVideoId);
    const sequence = Array.isArray(req.body.sequence)
      ? req.body.sequence
      : Array.isArray(req.body.frames)
        ? req.body.frames
        : null;
    let inputFilePath = null;
    let inputSequence = null;

    if (landmarkPath) {
      inputFilePath = resolveUploadFilePath(landmarkPath);
    } else if (Number.isFinite(trainingVideoId) && trainingVideoId > 0) {
      const trainingVideo = await getTrainingVideoById(trainingVideoId);
      inputFilePath = resolveUploadFilePath(trainingVideo?.landmark_file_path);
    } else if (sequence?.length) {
      inputSequence = sequence;
    }

    if (!inputFilePath && !inputSequence?.length) {
      return res.status(400).json({
        error: "landmarkPath, trainingVideoId or live sequence is required",
      });
    }

    const inferredModelProfile =
      requestedModelProfile !== DEFAULT_MODEL_PROFILE
        ? requestedModelProfile
        : inputSequence?.length
          ? "fast"
          : "accurate";
    const modelTarget = recognitionLevel
      ? await resolveModelArtifactTarget({
          recognitionLevel,
          modelProfile: inferredModelProfile,
        })
      : null;

    if (!modelTarget) {
      return res.status(409).json({
        error: "Model is not trained yet",
        details:
          "Train the public model or one of its internal specializations before requesting live or offline prediction.",
      });
    }

    const output = await runPersistentPrediction({
      recognitionLevel: modelTarget.recognitionLevel,
      modelProfile: modelTarget.modelProfile ?? undefined,
      inputPath: inputFilePath,
      sequence: inputSequence,
      allowedRecognitionLevels,
      allowedLabelKeys,
    });

    res.json(output);
  } catch (error) {
    console.error(error);
    res.status(500).json({
      error: "Failed to run model prediction",
      details: error.message,
    });
  }
});

app.get("/api/meta/users/default", async (req, res) => {
  try {
    const user = await findUserByEmail(DEFAULT_USER.email);
    res.json(user);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Failed to fetch default user" });
  }
});

app.get("/api/settings", async (req, res) => {
  try {
    const userEmail = req.query.userEmail?.trim() || DEFAULT_USER.email;
    const result = await getOrCreateUserSettings(userEmail);

    if (!result) {
      return res.status(404).json({ error: "User not found" });
    }

    res.json({
      user: result.user,
      settings: formatSettingsRow(result.settings),
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Failed to fetch settings" });
  }
});

app.put("/api/settings", async (req, res) => {
  try {
    const userEmail = req.body.userEmail?.trim() || DEFAULT_USER.email;
    const current = await getOrCreateUserSettings(userEmail);

    if (!current) {
      return res.status(404).json({ error: "User not found" });
    }

    const currentSettings = formatSettingsRow(current.settings);
    const nextSettings = {
      autoSpeakEnabled:
        typeof req.body.autoSpeakEnabled === "boolean"
          ? req.body.autoSpeakEnabled
          : currentSettings.autoSpeakEnabled,
      speechRate:
        typeof req.body.speechRate === "number"
          ? req.body.speechRate
          : currentSettings.speechRate,
      speechPitch:
        typeof req.body.speechPitch === "number"
          ? req.body.speechPitch
          : currentSettings.speechPitch,
      voiceName:
        typeof req.body.voiceName === "string"
          ? req.body.voiceName.trim()
          : currentSettings.voiceName,
      uiLanguage:
        typeof req.body.uiLanguage === "string" && req.body.uiLanguage.trim()
          ? req.body.uiLanguage.trim()
          : currentSettings.uiLanguage,
      signLanguage:
        typeof req.body.signLanguage === "string" && req.body.signLanguage.trim()
          ? req.body.signLanguage.trim()
          : currentSettings.signLanguage,
      preferredCategories:
        req.body.preferredCategories !== undefined
          ? normalizePreferredCategories(req.body.preferredCategories)
          : currentSettings.preferredCategories,
      largeTextEnabled:
        typeof req.body.largeTextEnabled === "boolean"
          ? req.body.largeTextEnabled
          : currentSettings.largeTextEnabled,
      developerModeEnabled:
        typeof req.body.developerModeEnabled === "boolean"
          ? req.body.developerModeEnabled
          : currentSettings.developerModeEnabled,
    };

    const updatedResult = await pool.query(
      `
        UPDATE user_settings
        SET
          auto_speak_enabled = $1,
          speech_rate = $2,
          speech_pitch = $3,
          voice_name = $4,
          interface_language = $5,
          sign_language = $6,
          preferred_categories = $7::jsonb,
          large_text_enabled = $8,
          developer_mode_enabled = $9,
          updated_at = NOW()
        WHERE user_id = $10
        RETURNING
          id,
          user_id,
          interface_language,
          speech_rate,
          speech_pitch,
          voice_name,
          auto_speak_enabled,
          sign_language,
          preferred_categories,
          large_text_enabled,
          developer_mode_enabled,
          created_at,
          updated_at
      `,
      [
        nextSettings.autoSpeakEnabled,
        nextSettings.speechRate,
        nextSettings.speechPitch,
        nextSettings.voiceName,
        nextSettings.uiLanguage,
        nextSettings.signLanguage,
        JSON.stringify(nextSettings.preferredCategories),
        nextSettings.largeTextEnabled,
        nextSettings.developerModeEnabled,
        current.user.id,
      ],
    );

    res.json({
      user: current.user,
      settings: formatSettingsRow(updatedResult.rows[0]),
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Failed to update settings" });
  }
});

app.get("/api/meta/categories", async (req, res) => {
  try {
    const result = await pool.query(
      `
        SELECT id, slug, name, sort_order
        FROM phrase_categories
        ORDER BY sort_order ASC, id ASC
      `,
    );

    res.json(result.rows);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Failed to fetch categories" });
  }
});

app.get("/api/meta/sign-languages", async (req, res) => {
  try {
    const result = await pool.query(
      `
        SELECT id, code, name, is_active, is_default
        FROM sign_languages
        ORDER BY is_default DESC, code ASC
      `,
    );

    res.json(result.rows);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Failed to fetch sign languages" });
  }
});

app.get("/api/meta/phrases", async (req, res) => {
  try {
    const featuredOnly = req.query.featured === "true";
    const category = req.query.category?.trim();
    const recognitionLevel = req.query.recognitionLevel?.trim();
    const signLanguage =
      req.query.signLanguage?.trim() || DEFAULT_USER_SETTINGS.signLanguage;

    const conditions = ["phrase_library.is_active = TRUE"];
    const values = [];

    if (signLanguage) {
      values.push(signLanguage);
      conditions.push(`phrase_library.sign_language = $${values.length}`);
    }

    if (featuredOnly) {
      values.push(true);
      conditions.push(`phrase_library.is_featured = $${values.length}`);
    }

    if (category) {
      values.push(category);
      conditions.push(
        `(phrase_categories.slug = $${values.length} OR phrase_categories.name = $${values.length})`,
      );
    }

    if (recognitionLevel) {
      values.push(normalizeRecognitionLevel(recognitionLevel));
      conditions.push(`phrase_library.recognition_level = $${values.length}`);
    }

    const result = await pool.query(
      `
        SELECT
          phrase_library.id,
          phrase_library.text,
          phrase_library.sign_language,
          phrase_library.entry_type,
          phrase_library.recognition_level,
          phrase_library.unit_code,
          phrase_library.description,
          phrase_library.reference_notes,
          phrase_library.is_v1,
          phrase_library.is_locked,
          phrase_library.is_featured,
          phrase_library.is_active,
          phrase_library.sort_order,
          phrase_categories.id AS category_id,
          phrase_categories.slug AS category_slug,
          phrase_categories.name AS category_name
        FROM phrase_library
        JOIN phrase_categories
          ON phrase_categories.id = phrase_library.category_id
        WHERE ${conditions.join(" AND ")}
        ORDER BY phrase_categories.sort_order ASC, phrase_library.sort_order ASC, phrase_library.id ASC
      `,
      values,
    );

    res.json(result.rows);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Failed to fetch phrases" });
  }
});

app.get("/api/meta/bootstrap", async (req, res) => {
  try {
    const bootstrapUser = await getOrCreateUserSettings(DEFAULT_USER.email);
    const bootstrapSignLanguage =
      req.query.signLanguage?.trim() ||
      formatSettingsRow(bootstrapUser?.settings).signLanguage ||
      DEFAULT_USER_SETTINGS.signLanguage;
    const [categoriesResult, phrasesResult, signLanguagesResult] = await Promise.all([
      pool.query(
        `
          SELECT id, slug, name, sort_order
          FROM phrase_categories
          ORDER BY sort_order ASC, id ASC
        `,
      ),
      pool.query(
        `
          SELECT
          phrase_library.id,
          phrase_library.text,
          phrase_library.sign_language,
          phrase_library.entry_type,
          phrase_library.recognition_level,
          phrase_library.unit_code,
          phrase_library.description,
            phrase_library.reference_notes,
            phrase_library.is_v1,
            phrase_library.is_locked,
            phrase_library.is_featured,
            phrase_library.is_active,
            phrase_library.sort_order,
            phrase_categories.slug AS category_slug,
            phrase_categories.name AS category_name
          FROM phrase_library
          JOIN phrase_categories
            ON phrase_categories.id = phrase_library.category_id
          WHERE phrase_library.is_active = TRUE
            AND phrase_library.sign_language = $1
          ORDER BY phrase_categories.sort_order ASC, phrase_library.sort_order ASC, phrase_library.id ASC
        `,
        [bootstrapSignLanguage],
      ),
      pool.query(
        `
          SELECT id, code, name, is_active, is_default
          FROM sign_languages
          ORDER BY is_default DESC, code ASC
        `,
      ),
    ]);

    res.json({
      defaultUser: bootstrapUser?.user ?? null,
      settings: formatSettingsRow(bootstrapUser?.settings),
      categories: categoriesResult.rows,
      phrases: phrasesResult.rows,
      signLanguages: signLanguagesResult.rows,
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Failed to fetch bootstrap data" });
  }
});

const httpServer = createServer(app);
const gestureWss = new WebSocketServer({
  server: httpServer,
  path: "/api/gestures/ws",
});

function handleFatalServerError(error) {
  if (error?.code === "EADDRINUSE") {
    console.error(`PORT ${PORT} is already in use. Stop the other process or set PORT to a different value.`);
  } else {
    console.error("Server error:", error);
  }
  process.exit(1);
}

httpServer.on("error", handleFatalServerError);
gestureWss.on("error", handleFatalServerError);

function broadcastGestureMessage(message) {
  const payload = JSON.stringify(message);
  gestureWss.clients.forEach((client) => {
    if (client.readyState === 1) {
      client.send(payload);
    }
  });
}

gestureBridgeService.on("message", (message) => {
  broadcastGestureMessage(message);
});

gestureBridgeService.on("status", (status) => {
  broadcastGestureMessage({
    type: "status",
    ...status,
  });
});

gestureWss.on("connection", (socket) => {
  socket.send(
    JSON.stringify({
      type: "status",
      ...gestureBridgeService.getStatus(),
    }),
  );

  socket.on("message", async (raw) => {
    try {
      const message = JSON.parse(raw.toString());
      const messageType = String(message.type || "").trim().toLowerCase();

      if (messageType === "start") {
        const status = await gestureBridgeService.start();
        socket.send(JSON.stringify({ type: "status", ...status }));
        return;
      }

      if (messageType === "stop") {
        const status = await gestureBridgeService.stop();
        socket.send(JSON.stringify({ type: "status", ...status }));
        return;
      }

      if (messageType === "reset") {
        const status = await gestureBridgeService.reset();
        socket.send(JSON.stringify({ type: "status", ...status }));
        return;
      }

      if (messageType === "ping") {
        socket.send(
          JSON.stringify({
            type: "pong",
            ts: Date.now(),
            client_ping_id: message.client_ping_id ?? null,
            client_sent_at_ms: message.client_sent_at_ms ?? null,
          }),
        );
        return;
      }

      if (messageType === "frame") {
        const result = gestureBridgeService.sendFrame({
          ts: message.ts,
          imageB64: message.image_b64 ?? message.imageBase64,
          clientFrameId: message.client_frame_id ?? message.clientFrameId,
          clientSentAtMs: message.client_sent_at_ms ?? message.clientSentAtMs,
        });
        if (!result?.accepted && result?.reason) {
          socket.send(
            JSON.stringify({
              type: "frame_drop",
              reason: result.reason,
              client_frame_id: result.clientFrameId ?? message.client_frame_id ?? null,
              ...gestureBridgeService.getStatus(),
            }),
          );
        }
        return;
      }

      socket.send(
        JSON.stringify({
          type: "error",
          message: `Unsupported gesture socket message: ${messageType || "empty"}`,
        }),
      );
    } catch (error) {
      socket.send(
        JSON.stringify({
          type: "error",
          message: error.message,
        }),
      );
    }
  });
});

bootstrapDatabase()
  .then(() => {
    httpServer.listen(PORT, () => {
      console.log(`Server running on http://localhost:${PORT}`);
    });
  })
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
