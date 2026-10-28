import os, json, time
from dateutil import parser as dtparser
from kafka import KafkaConsumer
import psycopg2

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC     = os.getenv("KAFKA_ALERTS_TOPIC", "fraud_alerts")
GROUP_ID  = os.getenv("KAFKA_GROUP_ID", "fraud_sink")

PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DB   = os.getenv("PG_DB", "fraud")
PG_USER = os.getenv("PG_USER", "app")
PG_PASS = os.getenv("PG_PASS", "app")

SQL_ALERT = """

INSERT INTO fraud_alerts (transaction_id, user_id, fraud_score, ts_event)
VALUES (%s, %s, %s, %s)
ON CONFLICT (transaction_id) DO UPDATE
SET fraud_score = EXCLUDED.fraud_score,
    ts_event = EXCLUDED.ts_event;
    
"""

def get_consumer():
    servers = [s.strip() for s in BOOTSTRAP.split(",") if s.strip()]

    return KafkaConsumer(
        TOPIC,
        bootstrap_servers=servers,
        group_id=GROUP_ID,
        auto_offset_reset="latest",
        enable_auto_commit=False,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        consumer_timeout_ms=1000,
        max_poll_records=500,
    )

def get_conn():
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASS)

def persist_records(conn, records):
    """Persist the entire polled batch before its Kafka offsets are committed.

    A failed insert rolls back the batch and propagates the error. Restarting
    the consumer then replays it; the transaction-id upsert handles duplicates.
    """
    try:
        with conn.cursor() as cur:
            for msgs in records.values():
                for message in msgs:
                    payload = message.value
                    ts_str = payload.get("timestamp")
                    ts_event = dtparser.parse(ts_str) if ts_str else None
                    cur.execute(SQL_ALERT, (
                        payload["transaction_id"], payload["user_id"],
                        float(payload["fraud_score"]), ts_event,
                    ))
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def main():
    consumer = get_consumer()
    conn = None
    try:
        conn = get_conn()
        conn.autocommit = False
        while True:
            records = consumer.poll(timeout_ms=1000)
            if not records:
                continue
            persist_records(conn, records)
            consumer.commit()
    finally:
        if conn is not None:
            conn.close()
        consumer.close()


if __name__ == "__main__":
    while True:
        try:
            main()
        except Exception as e:
            print("sink crashed, retrying in 5s:", e)
            time.sleep(5)




