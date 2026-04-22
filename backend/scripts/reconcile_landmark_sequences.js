import path from "node:path";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import dotenv from "dotenv";
import { pool } from "../db.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const SERVER_ROOT = path.resolve(__dirname, "..");

dotenv.config({ path: path.join(SERVER_ROOT, ".env") });

function resolveUploadFilePath(relativePath) {
  if (!relativePath || typeof relativePath !== "string") {
    return null;
  }

  return path.join(SERVER_ROOT, "uploads", ...relativePath.split("/"));
}

function readSummary(payload) {
  return (
    payload?.meta?.summary ??
    payload?.summary ??
    null
  );
}

function normalizeMetric(value, fallback = null) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

async function reconcileLandmarkSequence(row) {
  const filePath = resolveUploadFilePath(row.file_path);

  if (!filePath) {
    return {
      trainingVideoId: row.training_video_id,
      status: "skipped",
      reason: "missing_path",
    };
  }

  let payload;

  try {
    payload = JSON.parse(await readFile(filePath, "utf8"));
  } catch (error) {
    await pool.query(
      `
        UPDATE landmark_sequences
        SET
          status = 'failed',
          error_message = $1,
          updated_at = NOW()
        WHERE training_video_id = $2
      `,
      [error.message, row.training_video_id],
    );

    return {
      trainingVideoId: row.training_video_id,
      status: "failed",
      reason: error.message,
    };
  }

  const summary = readSummary(payload);
  const frameCount =
    normalizeMetric(payload?.meta?.frame_count) ??
    normalizeMetric(summary?.frame_count) ??
    (Array.isArray(payload?.frames) ? payload.frames.length : 0);

  await pool.query(
    `
      UPDATE landmark_sequences
      SET
        frame_count = $1,
        status = 'ready',
        valid_frame_ratio = $2,
        missing_hand_ratio = $3,
        missing_face_ratio = $4,
        missing_pose_ratio = $5,
        normalization_version = COALESCE(NULLIF($6, ''), normalization_version, 'holistic_v2'),
        error_message = NULL,
        updated_at = NOW()
      WHERE training_video_id = $7
    `,
    [
      frameCount,
      normalizeMetric(summary?.valid_frame_ratio),
      normalizeMetric(summary?.missing_hand_ratio),
      normalizeMetric(summary?.missing_face_ratio),
      normalizeMetric(summary?.missing_pose_ratio),
      String(payload?.meta?.extractor_type || payload?.meta?.extractor || "holistic_v2"),
      row.training_video_id,
    ],
  );

  return {
    trainingVideoId: row.training_video_id,
    status: "ready",
    frameCount,
    validFrameRatio: normalizeMetric(summary?.valid_frame_ratio),
    missingHandRatio: normalizeMetric(summary?.missing_hand_ratio),
  };
}

async function main() {
  const result = await pool.query(`
    SELECT training_video_id, file_path, status
    FROM landmark_sequences
    ORDER BY training_video_id ASC
  `);

  const rows = result.rows;
  const output = [];

  for (const row of rows) {
    output.push(await reconcileLandmarkSequence(row));
  }

  const summary = {
    total: output.length,
    ready: output.filter((item) => item.status === "ready").length,
    failed: output.filter((item) => item.status === "failed").length,
    skipped: output.filter((item) => item.status === "skipped").length,
  };

  console.log(
    JSON.stringify(
      {
        ok: true,
        summary,
      },
      null,
      2,
    ),
  );
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
