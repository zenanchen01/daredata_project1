-- === Minimal schema for your project ===

-- 1) Track curated dataset versions stored in S3
CREATE TABLE IF NOT EXISTS dataset_versions (
  id SERIAL PRIMARY KEY,
  s3_key      TEXT NOT NULL,
  etag        TEXT,
  row_count   INT,
  created_at  TIMESTAMPTZ DEFAULT now()
);

-- 2) Track model artifacts stored in S3 + their metrics
CREATE TABLE IF NOT EXISTS model_versions (
  id SERIAL PRIMARY KEY,
  s3_key      TEXT NOT NULL,
  etag        TEXT,
  metrics     JSONB,        -- e.g. {"auc":0.91,"f1":0.52}
  created_at  TIMESTAMPTZ DEFAULT now()
);

-- 3) Small sample of curated data for ad-hoc SQL checks
CREATE TABLE IF NOT EXISTS bank_sample (
  id SERIAL PRIMARY KEY,
  age INT,
  job TEXT,
  marital TEXT,
  education TEXT,
  "default" TEXT,  -- reserved word -> quoted
  balance INT,
  housing TEXT,
  loan TEXT,
  contact TEXT,
  day INT,
  month TEXT,
  campaign INT,
  pdays INT,
  previous INT,
  poutcome TEXT,
  y TEXT
);

-- 4) Lightweight prediction logging for observability
CREATE TABLE IF NOT EXISTS prediction_logs (
  id BIGSERIAL PRIMARY KEY,
  ts TIMESTAMPTZ DEFAULT now(),
  request_json JSONB NOT NULL,
  prediction INT NOT NULL,
  proba DOUBLE PRECISION NOT NULL,
  model_s3_key TEXT,
  model_etag TEXT
);

-- === Helpful indexes (optional but recommended) ===
-- Speed up time-based queries
CREATE INDEX IF NOT EXISTS idx_prediction_logs_ts ON prediction_logs (ts);
-- Simple GIN index to query into JSON payloads (e.g., request_json->>'job')
CREATE INDEX IF NOT EXISTS idx_prediction_logs_req_json ON prediction_logs USING GIN (request_json);
-- Quick lookup by model etag/key
CREATE INDEX IF NOT EXISTS idx_prediction_logs_model ON prediction_logs (model_etag, model_s3_key);