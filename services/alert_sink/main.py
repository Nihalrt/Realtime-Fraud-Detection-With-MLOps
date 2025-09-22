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
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        consumer_timeout_ms=1000,
        max_poll_records=500,
    )

def get_conn():
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASS)

def main():
    consumer = get_consumer()
    conn = get_conn()
    conn.autocommit = True

    while True:
        try:
            records = consumer.poll(timeout_ms=1000)
            if not records:
                continue

            with conn.cursor() as cur:
                for _, msgs in records.items():
                    for m in msgs:
                        p = m.value
                        txid = p.get("transaction_id")
                        uid = p.get("user_id")
                        score = float(p.get("fraud_score", 0.0))
                        ts_event = None
                        ts_str = p.get("timestamp")
                        try:
                            ts_event = dtparser.parse(ts_str)
                        except:
                            pass

                        try:
                            cur.execute(SQL_ALERT, (txid, uid, score, ts_event))
                        except Exception as e:
                            print("query failed", e)
        except Exception as e:
            print("loop error", e)

if __name__ == "__main__":
    while True:
        try:
            main()
        except Exception as e:
            print("sink crashed, retrying in 5s:", e)
            time.sleep(5)




