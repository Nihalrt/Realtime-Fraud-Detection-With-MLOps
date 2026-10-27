# Real-Time Fraud Detection with MLOps

A portfolio prototype for transaction feature engineering, supervised and
unsupervised fraud scoring, Kafka alerts, and a Streamlit dashboard. It includes
an experimental Google Cloud Pub/Sub transaction generator. The checked-in
streaming job uses **Kafka**, not a Pub/Sub Spark connector.

## What is implemented

- `models/train_model.py`: Parquet loading, time and user-history features,
  RobustScaler, XGBoost (or Logistic Regression when XGBoost is unavailable),
  Isolation Forest, validation metrics, and saved threshold/calibration metadata.
- `spark-streaming/main.py`: Kafka `transactions` ingestion, micro-batch scoring,
  console output, and flagged records published to Kafka `fraud_alerts`.
- `services/alert_sink/main.py`: Kafka alerts persisted to PostgreSQL with a
  transaction-id upsert. Database commits precede Kafka offset commits, so
  a restart can replay unacknowledged records without duplicating rows.
- `streamlit_app/app.py`: Kafka alert inspection and session-level charts.
- `data-generator/main.py`: synthetic labeled transactions published to **GCP
  Pub/Sub**. There is currently no Pub/Sub-to-Kafka bridge in this repository.

```text
Kafka transactions -> Spark scoring -> Kafka fraud_alerts -> PostgreSQL sink
                                                 \------> Streamlit dashboard
Parquet training data -> model + scaler + features + threshold -> Spark scoring

Experimental generator -> GCP Pub/Sub transactions (separate ingestion path)
```

## Local Kafka workflow

Use Python 3.12, Docker Compose, and a Java runtime compatible with Spark 4.
Run commands from the repository root unless shown otherwise.

```bash
git clone https://github.com/Nihalrt/Realtime-Fraud-Detection-With-MLOps.git
cd Realtime-Fraud-Detection-With-MLOps
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` for your environment. Python services read process environment
variables; the presence of `.env` alone does not load them. Export variables in
your shell, or use your process manager's environment-file support. Compose's
local database defaults are `app` / `app`; these are demonstration credentials.
The sink and dashboard use `KAFKA_BOOTSTRAP` and Kafka `fraud_alerts` by default.

```bash
docker compose up -d
# Once PostgreSQL accepts connections, initialize its schema:
docker compose exec -T postgres psql -U app -d fraud < db/init.sql
```

Place trusted training data in `artifacts/enriched_transactions/*.parquet`.
Required columns: `transaction_id`, `user_id`, `amount`, `location`, `timestamp`.
Optional binary `label` selects supervised training when positive examples are
present. Supervised training needs both classes in the training and validation
periods. Existing sample data may be used for a demonstration; no measured
production accuracy is claimed.

```bash
python -m models.train_model
PYTHONPATH=. python spark-streaming/main.py
```

The second command starts a continuous job. In separate terminals, with the
same virtual environment and configuration:

```bash
python services/alert_sink/main.py
streamlit run streamlit_app/app.py
```

Publish transactions to Kafka topic `transactions` with the five required
columns above as JSON. Start the scorer before publishing: it subscribes from
`latest` on a fresh checkpoint. The GCP generator does **not** feed this Kafka
job without an external bridge. The current scorer prints results and publishes
alerts; it does not write fresh training Parquet data.

## Optional Pub/Sub generator

```bash
pip install google-cloud-pubsub
gcloud auth application-default login
export GCP_PROJECT=your-project-id
gcloud pubsub topics create transactions --project="$GCP_PROJECT"
python data-generator/main.py
```

This path requires a GCP project and credentials. It is independent of the local
Kafka workflow; do not expect its messages to appear in the dashboard.

## Tests

For the isolated Python regression suite (no broker, cloud account, Spark JVM,
or database server required):

```bash
pip install -r requirements-test.txt
python -m unittest discover -s tests -v
```

Tests cover label retention and alignment, trailing-window expiry, fractional
training features, F1 threshold selection, and alert persistence/acknowledgment
ordering. They do not establish end-to-end streaming correctness or model
quality on real financial data.

## Security and limitations

- `.env` is local configuration; commit only `.env.example` with placeholders.
  A `.env` was previously tracked. Removing it from the current tree does not
  erase earlier commits. Rotate any real credentials ever stored there; history
  cleanup requires a separate coordinated operation. Never paste credentials
  into issues or PR descriptions.
- Load only trusted model artifacts: joblib/pickle files can execute code.
- Compose exposes unauthenticated development services; use a trusted local
  environment. Production needs authentication, encrypted transport, secret
  management, backups, and monitoring.
- Velocity and frequency features are recomputed inside each micro-batch, so
  user history does not span batches and global frequencies can use later rows
  in the same input. Offline metrics are not leakage-free production estimates.
- Retrain artifacts after changing feature calculations. Do not compare new
  training metrics directly with artifacts produced by the previous code.
- The sink retries a failed batch without acknowledging it. Malformed records
  can therefore block progress until repaired; a dead-letter queue is future work.
- Kafka writes from `foreachBatch` can be repeated after a retry. PostgreSQL
  upserts deduplicate by transaction ID; other consumers need their own policy.
- The repository includes historical generated artifacts and a bundled Cloud
  SDK under `sai/`. They are not needed to run the application. New local copies
  are ignored; removing the historical payload is separate repository cleanup.
