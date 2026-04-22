CREATE TABLE IF NOT EXISTS app_users (
  id BIGSERIAL PRIMARY KEY,
  email TEXT NOT NULL UNIQUE,
  password_hash TEXT NOT NULL,
  display_name TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS phrase_categories (
  id BIGSERIAL PRIMARY KEY,
  slug TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL UNIQUE,
  sort_order INTEGER NOT NULL DEFAULT 0,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sign_languages (
  id BIGSERIAL PRIMARY KEY,
  code VARCHAR(20) NOT NULL UNIQUE,
  name TEXT NOT NULL,
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  is_default BOOLEAN NOT NULL DEFAULT FALSE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS phrase_library (
  id BIGSERIAL PRIMARY KEY,
  category_id BIGINT NOT NULL REFERENCES phrase_categories(id) ON DELETE CASCADE,
  text TEXT NOT NULL,
  sign_language VARCHAR(20) NOT NULL DEFAULT 'rsl',
  entry_type VARCHAR(20) NOT NULL DEFAULT 'phrase',
  recognition_level VARCHAR(20) NOT NULL DEFAULT 'phrase',
  unit_code VARCHAR(50),
  description TEXT,
  reference_notes TEXT,
  is_v1 BOOLEAN NOT NULL DEFAULT FALSE,
  is_locked BOOLEAN NOT NULL DEFAULT FALSE,
  is_featured BOOLEAN NOT NULL DEFAULT FALSE,
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  sort_order INTEGER NOT NULL DEFAULT 0,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (category_id, text, entry_type)
);

ALTER TABLE phrase_library
  DROP CONSTRAINT IF EXISTS phrase_library_category_id_text_key;

ALTER TABLE phrase_library
  DROP CONSTRAINT IF EXISTS phrase_library_category_text_entry_type_key;

ALTER TABLE phrase_library
  ADD CONSTRAINT phrase_library_category_text_entry_type_key
  UNIQUE (category_id, text, entry_type);

ALTER TABLE phrase_library
  ADD COLUMN IF NOT EXISTS sign_language VARCHAR(20) NOT NULL DEFAULT 'rsl';

ALTER TABLE phrase_library
  ADD COLUMN IF NOT EXISTS entry_type VARCHAR(20) NOT NULL DEFAULT 'phrase';

ALTER TABLE phrase_library
  ADD COLUMN IF NOT EXISTS recognition_level VARCHAR(20) NOT NULL DEFAULT 'phrase';

ALTER TABLE phrase_library
  ADD COLUMN IF NOT EXISTS unit_code VARCHAR(50);

ALTER TABLE phrase_library
  ADD COLUMN IF NOT EXISTS description TEXT;

ALTER TABLE phrase_library
  ADD COLUMN IF NOT EXISTS reference_notes TEXT;

ALTER TABLE phrase_library
  ADD COLUMN IF NOT EXISTS is_v1 BOOLEAN NOT NULL DEFAULT FALSE;

ALTER TABLE phrase_library
  ADD COLUMN IF NOT EXISTS is_locked BOOLEAN NOT NULL DEFAULT FALSE;

CREATE TABLE IF NOT EXISTS user_settings (
  id BIGSERIAL PRIMARY KEY,
  user_id BIGINT NOT NULL UNIQUE REFERENCES app_users(id) ON DELETE CASCADE,
  interface_language VARCHAR(10) NOT NULL DEFAULT 'ru',
  speech_rate NUMERIC(3,2) NOT NULL DEFAULT 1.00,
  speech_pitch NUMERIC(3,2) NOT NULL DEFAULT 1.00,
  voice_name TEXT NOT NULL DEFAULT '',
  auto_speak_enabled BOOLEAN NOT NULL DEFAULT TRUE,
  sign_language VARCHAR(20) NOT NULL DEFAULT 'rsl',
  preferred_categories JSONB NOT NULL DEFAULT '[]'::jsonb,
  large_text_enabled BOOLEAN NOT NULL DEFAULT FALSE,
  developer_mode_enabled BOOLEAN NOT NULL DEFAULT FALSE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE user_settings
  ADD COLUMN IF NOT EXISTS speech_pitch NUMERIC(3,2) NOT NULL DEFAULT 1.00;

ALTER TABLE user_settings
  ADD COLUMN IF NOT EXISTS voice_name TEXT NOT NULL DEFAULT '';

ALTER TABLE user_settings
  ADD COLUMN IF NOT EXISTS sign_language VARCHAR(20) NOT NULL DEFAULT 'rsl';

ALTER TABLE user_settings
  ADD COLUMN IF NOT EXISTS preferred_categories JSONB NOT NULL DEFAULT '[]'::jsonb;

ALTER TABLE user_settings
  ADD COLUMN IF NOT EXISTS large_text_enabled BOOLEAN NOT NULL DEFAULT FALSE;

ALTER TABLE user_settings
  ADD COLUMN IF NOT EXISTS developer_mode_enabled BOOLEAN NOT NULL DEFAULT FALSE;

CREATE TABLE IF NOT EXISTS message_history (
  id BIGSERIAL PRIMARY KEY,
  type VARCHAR(20) NOT NULL,
  text TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS training_videos (
  id BIGSERIAL PRIMARY KEY,
  user_id BIGINT REFERENCES app_users(id) ON DELETE SET NULL,
  phrase_id BIGINT NOT NULL REFERENCES phrase_library(id) ON DELETE CASCADE,
  label_type VARCHAR(20) NOT NULL DEFAULT 'phrase',
  sign_language VARCHAR(20) NOT NULL DEFAULT 'rsl',
  signer_label TEXT,
  video_path TEXT NOT NULL,
  duration_ms INTEGER NOT NULL DEFAULT 0,
  dataset_split VARCHAR(20) NOT NULL DEFAULT 'unassigned',
  status VARCHAR(20) NOT NULL DEFAULT 'draft',
  quality_score INTEGER,
  review_status VARCHAR(20) NOT NULL DEFAULT 'pending',
  review_notes TEXT,
  reviewed_by BIGINT REFERENCES app_users(id) ON DELETE SET NULL,
  reviewed_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE training_videos
  ADD COLUMN IF NOT EXISTS label_type VARCHAR(20) NOT NULL DEFAULT 'phrase';

ALTER TABLE training_videos
  ADD COLUMN IF NOT EXISTS dataset_split VARCHAR(20) NOT NULL DEFAULT 'unassigned';

ALTER TABLE training_videos
  ADD COLUMN IF NOT EXISTS quality_score INTEGER;

ALTER TABLE training_videos
  ADD COLUMN IF NOT EXISTS review_status VARCHAR(20) NOT NULL DEFAULT 'pending';

ALTER TABLE training_videos
  ADD COLUMN IF NOT EXISTS review_notes TEXT;

ALTER TABLE training_videos
  ADD COLUMN IF NOT EXISTS reviewed_by BIGINT REFERENCES app_users(id) ON DELETE SET NULL;

ALTER TABLE training_videos
  ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMPTZ;

ALTER TABLE training_videos
  ADD COLUMN IF NOT EXISTS signer_label TEXT;

ALTER TABLE training_videos
  DROP CONSTRAINT IF EXISTS training_videos_status_check;

ALTER TABLE training_videos
  ADD CONSTRAINT training_videos_status_check
  CHECK (status IN ('draft', 'processing', 'ready', 'archived')) NOT VALID;

ALTER TABLE training_videos
  DROP CONSTRAINT IF EXISTS training_videos_dataset_split_check;

ALTER TABLE training_videos
  ADD CONSTRAINT training_videos_dataset_split_check
  CHECK (dataset_split IN ('unassigned', 'train', 'val', 'test')) NOT VALID;

CREATE TABLE IF NOT EXISTS landmark_sequences (
  id BIGSERIAL PRIMARY KEY,
  training_video_id BIGINT NOT NULL UNIQUE REFERENCES training_videos(id) ON DELETE CASCADE,
  file_path TEXT NOT NULL,
  frame_count INTEGER NOT NULL DEFAULT 0,
  status VARCHAR(20) NOT NULL DEFAULT 'pending',
  extractor VARCHAR(50) NOT NULL DEFAULT 'mediapipe_holistic_py',
  valid_frame_ratio NUMERIC(5,4),
  missing_hand_ratio NUMERIC(5,4),
  missing_face_ratio NUMERIC(5,4),
  missing_pose_ratio NUMERIC(5,4),
  normalization_version VARCHAR(30) NOT NULL DEFAULT 'none',
  error_message TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE landmark_sequences
  ADD COLUMN IF NOT EXISTS valid_frame_ratio NUMERIC(5,4);

ALTER TABLE landmark_sequences
  ADD COLUMN IF NOT EXISTS missing_hand_ratio NUMERIC(5,4);

ALTER TABLE landmark_sequences
  ADD COLUMN IF NOT EXISTS missing_face_ratio NUMERIC(5,4);

ALTER TABLE landmark_sequences
  ADD COLUMN IF NOT EXISTS missing_pose_ratio NUMERIC(5,4);

ALTER TABLE landmark_sequences
  ADD COLUMN IF NOT EXISTS normalization_version VARCHAR(30) NOT NULL DEFAULT 'none';

CREATE TABLE IF NOT EXISTS recognition_runs (
  id BIGSERIAL PRIMARY KEY,
  user_id BIGINT REFERENCES app_users(id) ON DELETE SET NULL,
  phrase_id BIGINT REFERENCES phrase_library(id) ON DELETE SET NULL,
  sign_language VARCHAR(20) NOT NULL DEFAULT 'rsl',
  recognition_level VARCHAR(20) NOT NULL DEFAULT 'phrase',
  source_mode VARCHAR(20) NOT NULL DEFAULT 'rules',
  recognized_text TEXT NOT NULL,
  confidence NUMERIC(6,4),
  raw_output_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS dataset_exports (
  id BIGSERIAL PRIMARY KEY,
  sign_language VARCHAR(20) NOT NULL DEFAULT 'rsl',
  recognition_level VARCHAR(20) NOT NULL DEFAULT 'phrase',
  export_name VARCHAR(50) NOT NULL DEFAULT 'latest',
  sample_count INTEGER NOT NULL DEFAULT 0,
  split_counts JSONB NOT NULL DEFAULT '{}'::jsonb,
  quality_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
  manifest_path TEXT,
  summary_path TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS model_runs (
  id BIGSERIAL PRIMARY KEY,
  run_type VARCHAR(30) NOT NULL,
  model_type VARCHAR(30) NOT NULL,
  sign_language VARCHAR(20) NOT NULL DEFAULT 'rsl',
  recognition_level VARCHAR(20) NOT NULL DEFAULT 'phrase',
  dataset_export_id BIGINT REFERENCES dataset_exports(id) ON DELETE SET NULL,
  artifact_dir TEXT,
  metrics_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  config_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  is_winner BOOLEAN NOT NULL DEFAULT FALSE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE message_history
  ADD COLUMN IF NOT EXISTS user_id BIGINT REFERENCES app_users(id) ON DELETE SET NULL;

ALTER TABLE message_history
  ADD COLUMN IF NOT EXISTS phrase_id BIGINT REFERENCES phrase_library(id) ON DELETE SET NULL;

ALTER TABLE recognition_runs
  ADD COLUMN IF NOT EXISTS recognition_level VARCHAR(20) NOT NULL DEFAULT 'phrase';

ALTER TABLE dataset_exports
  ADD COLUMN IF NOT EXISTS recognition_level VARCHAR(20) NOT NULL DEFAULT 'phrase';

ALTER TABLE model_runs
  ADD COLUMN IF NOT EXISTS recognition_level VARCHAR(20) NOT NULL DEFAULT 'phrase';

CREATE INDEX IF NOT EXISTS idx_message_history_created_at
  ON message_history (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_message_history_user_id
  ON message_history (user_id);

CREATE INDEX IF NOT EXISTS idx_phrase_library_category_id
  ON phrase_library (category_id, sort_order, id);

CREATE INDEX IF NOT EXISTS idx_phrase_library_sign_language
  ON phrase_library (sign_language, recognition_level, is_v1, sort_order, id);

CREATE INDEX IF NOT EXISTS idx_phrase_library_entry_type
  ON phrase_library (sign_language, entry_type, sort_order, id);

CREATE INDEX IF NOT EXISTS idx_training_videos_phrase_id
  ON training_videos (phrase_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_training_videos_user_id
  ON training_videos (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_training_videos_dataset_split
  ON training_videos (dataset_split, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_training_videos_review_status
  ON training_videos (review_status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_training_videos_label_type
  ON training_videos (label_type, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_landmark_sequences_status
  ON landmark_sequences (status, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_recognition_runs_created_at
  ON recognition_runs (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_recognition_runs_user_id
  ON recognition_runs (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_dataset_exports_created_at
  ON dataset_exports (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_dataset_exports_level
  ON dataset_exports (sign_language, recognition_level, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_model_runs_created_at
  ON model_runs (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_model_runs_model_type
  ON model_runs (model_type, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_model_runs_level
  ON model_runs (sign_language, recognition_level, created_at DESC);
