import path from "node:path";
import { mkdir, rm } from "node:fs/promises";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import dotenv from "dotenv";
import { pool } from "../db.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const SERVER_ROOT = path.resolve(__dirname, "..");
const UPLOADS_ROOT = path.join(SERVER_ROOT, "uploads");
const PYTHON_BIN = process.env.PYTHON_BIN || "python";
const LANDMARK_EXTRACT_SCRIPT = path.join(__dirname, "extract_landmarks.py");

dotenv.config({ path: path.join(SERVER_ROOT, ".env") });

function resolveUploadFilePath(relativePath) {
  if (!relativePath || typeof relativePath !== "string") {
    return null;
  }

  return path.join(UPLOADS_ROOT, ...relativePath.split("/"));
}

function parseArgs(argv) {
  const options = {
    recognitionLevel: null,
    force: false,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index];

    if (value === "--recognition-level") {
      options.recognitionLevel = argv[index + 1] ?? null;
      index += 1;
      continue;
    }

    if (value === "--force") {
      options.force = true;
    }
  }

  return options;
}

function runExtraction(args) {
  return new Promise((resolve, reject) => {
    const child = spawn(PYTHON_BIN, [LANDMARK_EXTRACT_SCRIPT, ...args], {
      cwd: SERVER_ROOT,
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

    child.on("error", reject);
    child.on("close", (code) => {
      if (code !== 0) {
        reject(
          new Error(
            stderr.trim() ||
              stdout.trim() ||
              `extract_landmarks.py failed with code ${code}`,
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
        reject(new Error("extract_landmarks.py returned empty output"));
        return;
      }

      try {
        resolve(JSON.parse(lastLine));
      } catch {
        reject(new Error("extract_landmarks.py returned invalid JSON"));
      }
    });
  });
}

async function updateLandmarkSequence(trainingVideoId, payload) {
  await pool.query(
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
    `,
    [
      trainingVideoId,
      payload.filePath,
      payload.frameCount ?? 0,
      payload.status,
      payload.validFrameRatio ?? null,
      payload.missingHandRatio ?? null,
      payload.missingFaceRatio ?? null,
      payload.missingPoseRatio ?? null,
      payload.normalizationVersion ?? "holistic_v2",
      payload.errorMessage ?? null,
    ],
  );
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const values = [];
  const conditions = [];

  if (options.recognitionLevel) {
    values.push(options.recognitionLevel);
    conditions.push(`phrase_library.recognition_level = $${values.length}`);
  }

  const whereClause = conditions.length ? `WHERE ${conditions.join(" AND ")}` : "";
  const result = await pool.query(
    `
      SELECT
        training_videos.id,
        training_videos.phrase_id,
        training_videos.sign_language,
        training_videos.video_path,
        landmark_sequences.status AS landmark_status
      FROM training_videos
      JOIN phrase_library
        ON phrase_library.id = training_videos.phrase_id
      LEFT JOIN landmark_sequences
        ON landmark_sequences.training_video_id = training_videos.id
      ${whereClause}
      ORDER BY training_videos.id ASC
    `,
    values,
  );

  const rows = result.rows;
  const summary = {
    total: rows.length,
    rebuilt: 0,
    failed: 0,
    skipped: 0,
  };

  for (const row of rows) {
    if (!options.force && row.landmark_status === "ready") {
      summary.skipped += 1;
      continue;
    }

    const inputPath = resolveUploadFilePath(row.video_path);
    const storedLandmarkPath = path.posix.join(
      "landmarks",
      `training-video-${row.id}.json`,
    );
    const outputPath = resolveUploadFilePath(storedLandmarkPath);

    if (!inputPath || !outputPath) {
      summary.failed += 1;
      await updateLandmarkSequence(row.id, {
        filePath: storedLandmarkPath,
        status: "failed",
        errorMessage: "Training video path is missing",
      });
      continue;
    }

    await mkdir(path.dirname(outputPath), { recursive: true });
    await rm(outputPath, { force: true }).catch(() => {});

    await updateLandmarkSequence(row.id, {
      filePath: storedLandmarkPath,
      status: "processing",
      normalizationVersion: "holistic_v2",
    });

    try {
      const extractionResult = await runExtraction([
        "--input",
        inputPath,
        "--output",
        outputPath,
        "--sign-language",
        row.sign_language || "rsl",
        "--training-video-id",
        String(row.id),
        "--phrase-id",
        String(row.phrase_id),
      ]);

      await updateLandmarkSequence(row.id, {
        filePath: storedLandmarkPath,
        frameCount: Number(extractionResult.frameCount) || 0,
        status: "ready",
        validFrameRatio: extractionResult.summary?.valid_frame_ratio ?? null,
        missingHandRatio: extractionResult.summary?.missing_hand_ratio ?? null,
        missingFaceRatio: extractionResult.summary?.missing_face_ratio ?? null,
        missingPoseRatio: extractionResult.summary?.missing_pose_ratio ?? null,
        normalizationVersion: "holistic_v2",
      });
      summary.rebuilt += 1;
    } catch (error) {
      await updateLandmarkSequence(row.id, {
        filePath: storedLandmarkPath,
        status: "failed",
        normalizationVersion: "holistic_v2",
        errorMessage: error.message,
      });
      summary.failed += 1;
    }
  }

  console.log(JSON.stringify({ ok: true, summary }, null, 2));
}

main()
  .catch((error) => {
    console.error(
      JSON.stringify({
        ok: false,
        error: error.message,
      }),
    );
    process.exitCode = 1;
  })
  .finally(async () => {
    await pool.end().catch(() => {});
  });
