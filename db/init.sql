CREATE TABLE IF NOT EXISTS fraud_alerts (
  id             BIGSERIAL PRIMARY KEY,
  transaction_id TEXT UNIQUE,                  -- idempotency
  user_id        TEXT NOT NULL,
  fraud_score    DOUBLE PRECISION NOT NULL,
  ts_ingested    TIMESTAMPTZ NOT NULL DEFAULT now(),
  ts_event       TIMESTAMPTZ NULL
);

-- Optional: store ALL scored events (not only alerts)
CREATE TABLE IF NOT EXISTS scored_events (
  id             BIGSERIAL PRIMARY KEY,
  transaction_id TEXT UNIQUE,
  user_id        TEXT,
  fraud_score    DOUBLE PRECISION,
  is_fraud       BOOLEAN,
  ts_event       TIMESTAMPTZ,
  ts_ingested    TIMESTAMPTZ NOT NULL DEFAULT now()
);
