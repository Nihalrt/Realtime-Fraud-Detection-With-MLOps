# Engineering walkthrough

This prototype connects fraud feature engineering, model scoring, Kafka alerts,
and durable storage. It demonstrates component behavior without claiming
production detection accuracy or throughput.

## Training contract

[Training](../models/train_model.py) reads transaction Parquet data, builds
features, and selects supervised training when suitable labels exist; otherwise
it uses Isolation Forest. Saved artifacts carry feature order, preprocessing,
threshold, and score-calibration metadata into inference.

Key checks in [the training suite](../tests/test_training.py) cover label
alignment after sorting/filtering, fractional feature preservation, trailing
window expiry, and the correspondence between F1 points and thresholds.

## Streaming and persistence

[The Spark job](../spark-streaming/main.py) consumes Kafka transactions in
micro-batches, aligns feature columns with training metadata, applies the
saved scaler/model, and publishes flagged records to Kafka.

[The alert sink](../services/alert_sink/main.py) writes a polled batch in a
PostgreSQL transaction, then commits Kafka offsets. If the database write fails,
the batch is rolled back and offsets remain unacknowledged. Transaction-ID
upserts handle duplicate delivery after a restart.

This is replay-aware persistence, not a distributed exactly-once transaction.
Kafka publishing from a retried Spark batch can also emit duplicate alerts.

## What the tests establish

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-test.txt
.venv/bin/python -m unittest discover -s tests -v
```

[Sink tests](../tests/test_alert_sink.py) use mocks to verify commit ordering,
rollback, and disabled auto-commit. These tests do not establish real broker
replay, PostgreSQL durability across a server restart, or sustained load.

Database restart/replay behavior and sustained-write measurements need a
separate integration harness and captured results before they can be treated
as benchmarks. Model quality must be evaluated separately on suitable,
leakage-controlled data.

## Deliberate limits

- Feature history is recomputed within each batch. Global frequencies can use
  later input rows; offline validation is not a leakage-free production estimate.
- [The generator](../data-generator/main.py) publishes to Pub/Sub, whereas Spark
  reads Kafka. The repository does not include the bridge between them.
- The scorer prints results and publishes alerts; it does not create new
  training Parquet output.
- Malformed sink records need repair or a future dead-letter policy.
- Retrain artifacts after feature changes. Load only trusted joblib/pickle
  files, and keep actual credentials out of Git and demonstration data.

See [the README](../README.md) for setup, architecture, and deployment limits.
