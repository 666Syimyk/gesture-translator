import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { pool } from "../db.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const SUPPORTED_LEVELS = new Set(["alphabet", "sign", "phrase"]);

function parseArgs(argv) {
  const result = {};

  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];

    if (!token.startsWith("--")) {
      continue;
    }

    const key = token.slice(2);
    const value = argv[index + 1];

    if (!value || value.startsWith("--")) {
      result[key] = true;
      continue;
    }

    result[key] = value;
    index += 1;
  }

  return result;
}

function normalizeRecognitionLevel(value) {
  if (typeof value !== "string") {
    return "phrase";
  }

  const normalized = value.trim().toLowerCase();
  return SUPPORTED_LEVELS.has(normalized) ? normalized : "phrase";
}

function normalizeBoolean(value, fallback = false) {
  if (typeof value === "boolean") {
    return value;
  }

  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase();

    if (["true", "1", "yes", "y"].includes(normalized)) {
      return true;
    }

    if (["false", "0", "no", "n"].includes(normalized)) {
      return false;
    }
  }

  return fallback;
}

function normalizeInteger(value, fallback = 0) {
  const nextValue = Number(value);
  return Number.isFinite(nextValue) ? nextValue : fallback;
}

function assertEntry(entry, index) {
  if (!entry || typeof entry !== "object") {
    throw new Error(`Entry #${index + 1} must be an object`);
  }

  if (!entry.categorySlug || typeof entry.categorySlug !== "string") {
    throw new Error(`Entry #${index + 1} is missing categorySlug`);
  }

  if (!entry.text || typeof entry.text !== "string") {
    throw new Error(`Entry #${index + 1} is missing text`);
  }
}

async function ensureCategory(client, slug, name, sortOrder = 0) {
  const safeSlug = slug.trim();
  const safeName = (name || slug).trim();

  const result = await client.query(
    `
      INSERT INTO phrase_categories (slug, name, sort_order)
      VALUES ($1, $2, $3)
      ON CONFLICT (slug) DO UPDATE
      SET
        name = EXCLUDED.name,
        sort_order = EXCLUDED.sort_order
      RETURNING id
    `,
    [safeSlug, safeName, sortOrder],
  );

  return result.rows[0].id;
}

async function upsertEntry(client, categoryId, entry, defaultSignLanguage) {
  const result = await client.query(
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
      VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
      ON CONFLICT (category_id, text, entry_type) DO UPDATE
      SET
        sign_language = EXCLUDED.sign_language,
        entry_type = EXCLUDED.entry_type,
        recognition_level = EXCLUDED.recognition_level,
        unit_code = EXCLUDED.unit_code,
        description = EXCLUDED.description,
        reference_notes = EXCLUDED.reference_notes,
        is_v1 = EXCLUDED.is_v1,
        is_locked = EXCLUDED.is_locked,
        is_featured = EXCLUDED.is_featured,
        is_active = EXCLUDED.is_active,
        sort_order = EXCLUDED.sort_order
      RETURNING id
    `,
    [
      categoryId,
      entry.text.trim(),
      (entry.signLanguage || defaultSignLanguage || "rsl").trim().toLowerCase(),
      normalizeRecognitionLevel(entry.entryType || entry.recognitionLevel),
      normalizeRecognitionLevel(entry.recognitionLevel),
      entry.unitCode?.trim() || null,
      entry.description?.trim() || "",
      entry.referenceNotes?.trim() || "",
      normalizeBoolean(entry.isV1, false),
      normalizeBoolean(entry.isLocked, false),
      normalizeBoolean(entry.isFeatured, false),
      normalizeBoolean(entry.isActive, true),
      normalizeInteger(entry.sortOrder, 0),
    ],
  );

  return result.rows[0].id;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const inputPath = args.file
    ? path.resolve(process.cwd(), args.file)
    : path.resolve(__dirname, "..", "..", "docs", "templates", "phrases.rsl.template.json");
  const defaultSignLanguage = (args["sign-language"] || "rsl").trim().toLowerCase();

  const raw = await fs.readFile(inputPath, "utf8");
  const parsed = JSON.parse(raw);
  const entries = Array.isArray(parsed) ? parsed : parsed.entries;

  if (!Array.isArray(entries) || !entries.length) {
    throw new Error("Input file must contain a non-empty array or { entries: [] }");
  }

  entries.forEach(assertEntry);

  const client = await pool.connect();
  const summary = {
    inputPath,
    totalEntries: entries.length,
    insertedOrUpdated: 0,
    signLanguage: defaultSignLanguage,
  };

  try {
    await client.query("BEGIN");

    for (let index = 0; index < entries.length; index += 1) {
      const entry = entries[index];
      const categoryId = await ensureCategory(
        client,
        entry.categorySlug,
        entry.categoryName || entry.categorySlug,
        normalizeInteger(entry.categorySortOrder, 0),
      );

      await upsertEntry(client, categoryId, entry, defaultSignLanguage);
      summary.insertedOrUpdated += 1;
    }

    await client.query("COMMIT");
  } catch (error) {
    await client.query("ROLLBACK");
    throw error;
  } finally {
    client.release();
    await pool.end().catch(() => {});
  }

  console.log(JSON.stringify({ ok: true, ...summary }, null, 2));
}

main().catch((error) => {
  console.error(
    JSON.stringify(
      {
        ok: false,
        error: error.message,
      },
      null,
      2,
    ),
  );
  process.exit(1);
});
