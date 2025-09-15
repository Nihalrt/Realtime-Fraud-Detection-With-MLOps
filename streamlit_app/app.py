import os
import json
import time
from datetime import datetime
from dateutil import parser as dtparser
from typing import List, Dict, Any

import streamlit as st
import pandas as pd
from kafka import KafkaConsumer

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC      = os.getenv("KAFKA_ALERT_TOPIC", "fraud_alerts")
GROUP_ID   = os.getenv("KAFKA_GROUP_ID", "fraud_dashboard")
start_from_earliest = st.sidebar.toggle("Start from earliest", value=False)




# -------------- Streamlit page setup --------------
st.set_page_config(page_title="Fraud Alerts", page_icon="⚠️", layout="wide")
st.title("⚠️ Real-Time Fraud Alerts")

with st.sidebar:
    st.header("Settings")

    bootstrap_servers = st.text_input(
        "Kafka Bootstrap Servers",
        value=BOOTSTRAP,
        help="Comma-separated host:port, entries",
    )

    topic = st.text_input(
        "Kafka Topic",
        value=TOPIC,
        help="Alert topic produced by my spark stream",
    )

    start_from_earliest = st.toggle(
        'start_from_earliest',
        value=False,
        help="If ON, consumer starts from the earliest available messages"
    )

    max_records = st.slider(
        "Max records per refresh",
        min_value=10, max_value=2000, value=200, step=10
    )

    refresh_sec = st.slider(
        "Auto-refresh (seconds)",
        min_value=2, max_value=30, value=5
    )

    st.caption("Tip: Turn OFF *Start from earliest* to only read new alerts.")

# --------------- Create a single KafkaConsumer for the app session ------------------
@st.cache_resource(show_spinner=False)
def get_consumer(bootstrap: str, topic: str, group_id: str, earliest: bool=False) -> KafkaConsumer:
    # Normalize whatever we get (string or list) into ["host:port", ...]
    if isinstance(bootstrap, (list, tuple)):
        servers = [str(s).strip() for s in bootstrap if str(s).strip()]
    else:
        servers = [s.strip() for s in str(bootstrap).split(",") if s.strip()]  # <-- strip()

    auto_offset = "earliest" if earliest else "latest"

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=servers,
        group_id=group_id,
        auto_offset_reset=auto_offset,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        consumer_timeout_ms=1000,
        max_poll_records=1000,
    )
    return consumer

consumer = get_consumer(
    bootstrap=bootstrap_servers,
    topic=topic,
    group_id=GROUP_ID,
    earliest=start_from_earliest,
)

# ------------- Reading a batch of alerts ----------------

def poll_alerts(consumer: KafkaConsumer, limit: int, timeout_sec: int = 2) -> List[Dict[str, Any]]:

    end_by = time.time() + timeout_sec # basically, here we are setting a deadline that records the time the function should stop trying to fetch messages
    acc = [] # We are using this list to store the alert messages that we receive

    # We are checking if we have collected the limited amount of alert messages and also if it's the end of time
    while len(acc) < limit and time.time() < end_by:
        records = consumer.poll(timeout_ms=300)
        if not records:
            continue

        for topic, message in records.items():
            for m in message:
                try:
                    payload = m.value

                    alert = {
                        "transaction_id": payload.get("transaction_id"),
                        "user_id": payload.get("user_id"),
                        "fraud_score": float(payload.get("fraud_score", 0.0)),
                        "timestamp": payload.get("timestamp")
                    }
                    acc.append(alert)
                    if len(acc) >= limit:
                        break
                except Exception as e:
                    st.warning(f"Skipped malformed message: {e}")
            if len(acc) >= limit:
                break
    return acc

if "alerts_df" not in st.session_state:
    st.session_state.alerts_df = pd.DataFrame(columns=["transaction_id","user_id","fraud_score","timestamp"])


new_alerts = poll_alerts(consumer, limit=max_records, timeout_sec=2)

if new_alerts:
    df_new = pd.DataFrame(new_alerts)
    with pd.option_context('mode.chained_assignment', None):
        try:
            df_new["ts"] = df_new["timestamp"].map( lambda x: dtparser.parse(x) if isinstance(x, str) else pd.NaT)
        except Exception:
            df_new["ts"] = pd.NaT
    st.session_state.alerts_df = pd.concat([st.session_state.alerts_df, df_new], ignore_index=True).drop_duplicates(subset="transaction_id", keep="last")

df = st.session_state.alerts_df.copy()

# Top KPIs - (Key Performance Indicators) : Basically, a measureable to show how a system is achieving its main objectives, instead of showing all the data
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total alerts (session)", f"{len(df):,}")

if len(df):
    col2.metric("Latest score", f"{df.iloc[-1]['fraud_score']:.3f}")
    col3.metric("90th %ile score", f"{df['fraud_score'].quantile(0.9):.3f}")
    col4.metric("Unique users flagged", df["user_id"].nunique())

left, right = st.columns([2,1], gap="large")

with left:
    st.subheader("Alerts over time")
    if len(df) and "ts" in df.columns:
        df_plot = df.dropna(subset=["ts"]).sort_values("ts")
        st.line_chart(df_plot.set_index("ts")["fraud_score"])
    else:
        st.info("No timestamps parsed yet")

with right:
    st.subheader("Top users by alert count")
    if len(df):
        counts = df["user_id"].value_counts().head(10).rename_axis("user_id").reset_index(name="alerts")
        st.dataframe(counts, hide_index=True, use_container_width=True)
    else:
        st.info("No alerts")


st.subheader("Most recent alerts")
if len(df):
    st.dataframe(
        df.sort_index(ascending=False).head(200),
        hide_index=True,
        use_container_width=True
    )
else:
    st.info("Waiting for alerts from kafka")

# Controls row

c1, c2, c3 = st.columns(3)
with c1:
    if st.button("Clear session table"):
        st.session_state.alerts_df = st.session_state.alerts_df.iloc[0:0]
        st.success("Cleared in-memory session alerts.")
with c2:
    if st.button("Replay from earliest"):
        st.cache_resource.clear()
        get_consumer(bootstrap_servers, topic, GROUP_ID + "_replay", True)
        st.info("Consumer reset to earliest.")

with c3:
    st.write("")

# (optional) tweak query params to avoid browser caching
try:
    st.query_params.update({"_": str(int(time.time()))})
except Exception:
    pass

st.caption(f"Auto-refreshing every {refresh_sec}s…")
time.sleep(refresh_sec)
st.rerun()   # <- new API



















