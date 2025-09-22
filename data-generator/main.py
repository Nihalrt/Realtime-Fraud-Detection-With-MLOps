import os
import random
import uuid
import json
import time
from collections import deque, defaultdict
from datetime import datetime, timezone, timedelta

from google.cloud import pubsub_v1  # pip install google-cloud-pubsub

# ---------------- Config ----------------
PROJECT_ID = os.getenv("GCP_PROJECT", os.getenv("GOOGLE_CLOUD_PROJECT"))
if not PROJECT_ID:
    raise RuntimeError("Set GCP_PROJECT or run `gcloud config set project <ID>`")

TOPIC = os.getenv("PUBSUB_TOPIC", "transactions")  # match what you created

publisher = pubsub_v1.PublisherClient()
TOPIC_PATH = publisher.topic_path(PROJECT_ID, TOPIC)

# ---------------- Domain Data ----------------
US_STATES = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA",
    "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
    "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
    "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY",
]

FAR_STATE_PAIRS = {
    ("CA","NY"),("NY","CA"),("FL","WA"),("WA","FL"),
    ("ME","AZ"),("AZ","ME"),("TX","VT"),("VT","TX"),
}

def draw_normal_amount():
    base = random.lognormvariate(mu=3.2, sigma=0.6)
    return round(min(max(base, 1.0), 500.0), 2)

def draw_state():
    return random.choice(US_STATES)

def now_utc():
    return datetime.now(timezone.utc).isoformat()

class UserState:
    __slots__ = ("recent_ts", "recent_amts", "last_state", "seen_count",
                 "burst_remaining", "replay_remaining", "replay_amount")

    def __init__(self):
        self.recent_ts = deque(maxlen=50)
        self.recent_amts = deque(maxlen=50)
        self.last_state = None
        self.seen_count = 0
        self.burst_remaining = 0
        self.replay_remaining = 0
        self.replay_amount = None

USER = defaultdict(UserState)

# ---------- Generation toggles ----------
FRAUD_RATE = 0.10
P_BURST = 0.50
P_SPIKE = 0.30
P_REPLAY = 0.20
P_TRAVEL = 0.15
P_NEW_USER = 0.05

def draw_spike_amount():
    return round(random.uniform(2000.0, 10000.0), 2)

def pick_far_state(current):
    # Prefer a far pair; fallback: any different state
    for s in US_STATES:
        if s != current and (current, s) in FAR_STATE_PAIRS:
            return s
    return random.choice([st for st in US_STATES if st != current])

def make_base_txn(user_id: str, amount: float, state: str, label=0, reason=None, ts=None):
    ts_iso = ts.isoformat() if isinstance(ts, datetime) else now_utc()
    return {
        "transaction_id": str(uuid.uuid4()),
        "user_id": user_id,
        "amount": float(amount),
        "location": state,
        "timestamp": ts_iso,   # keep this name; downstream expects it
        "label": int(label),
        "fraud_reason": reason,
    }

def normal_txn(u: UserState, user_id: str):
    amt = draw_normal_amount()
    state = u.last_state or draw_state()
    return make_base_txn(user_id, amt, state, label=0, reason=None)

def fraud_amount_spike(u: UserState, user_id: str):
    amt = draw_spike_amount()
    state = u.last_state or draw_state()
    return make_base_txn(user_id, amt, state, label=1, reason="amount_spike")

def fraud_replay(u: UserState, user_id: str):
    if u.replay_remaining <= 0:
        u.replay_amount = round(random.uniform(80.0, 150.0), 2)
        u.replay_remaining = random.randint(5, 9)
    u.replay_remaining -= 1
    state = u.last_state or draw_state()
    return make_base_txn(user_id, u.replay_amount, state, label=1, reason="multiple_txns_of_same_amount")

def fraud_burst(u: UserState, user_id: str):
    if u.burst_remaining <= 0:
        u.burst_remaining = random.randint(5, 9)
    u.burst_remaining -= 1
    amt = round(random.uniform(20.0, 500.0), 2)
    state = u.last_state or draw_state()
    return make_base_txn(user_id, amt, state, label=1, reason="large_no_of_txns_in_small_timeframe")

def fraud_impossible(u: UserState, user_id: str):
    prev = u.last_state or draw_state()
    travel_state = pick_far_state(prev)
    amt = draw_normal_amount()
    return make_base_txn(user_id, amt, travel_state, label=1, reason="impossible_travel_txn")

def fraud_new_user_high_activity():
    user_id = f"user_{random.randint(10001, 20000)}"
    u = USER[user_id]
    u.burst_remaining = random.randint(5, 10)
    return fraud_amount_spike(u, user_id), user_id

# ---------- Pub/Sub publish ----------
def send_transaction(txn: dict):
    data = json.dumps(txn).encode("utf-8")
    future = publisher.publish(TOPIC_PATH, data)
    # optionally wait: result = future.result(timeout=10)
    print("Published:", txn["transaction_id"], txn["user_id"], txn["amount"], txn["location"])

def step_once():
    if USER and random.random() < 0.8:
        user_id = random.choice(list(USER.keys()))
    else:
        user_id = f"user_{random.randint(10001, 20000)}"

    u = USER[user_id]
    is_fraud = (random.random() < FRAUD_RATE)

    if u.burst_remaining > 0:
        txn = fraud_burst(u, user_id); delay = 0.05
    elif u.replay_remaining > 0:
        txn = fraud_replay(u, user_id); delay = 0.10
    elif is_fraud:
        r = random.random()
        if r < P_BURST:
            txn = fraud_burst(u, user_id); delay = 0.05
        elif r < P_BURST + P_SPIKE:
            txn = fraud_amount_spike(u, user_id); delay = 0.30
        elif r < P_BURST + P_SPIKE + P_REPLAY:
            txn = fraud_replay(u, user_id); delay = 0.10
        elif r < P_BURST + P_SPIKE + P_REPLAY + P_TRAVEL:
            txn = fraud_impossible(u, user_id); delay = 0.50
        else:
            txn, user_id = fraud_new_user_high_activity()
            u = USER[user_id]; delay = 0.05
    else:
        txn = normal_txn(u, user_id); delay = 0.50

    u.seen_count += 1
    u.last_state = txn["location"]
    u.recent_ts.append(datetime.now(timezone.utc))
    u.recent_amts.append(txn["amount"])

    send_transaction(txn)
    return delay

def main():
    try:
        while True:
            delay = step_once()
            time.sleep(delay)
    finally:
        # Pub/Sub publisher is managed; no explicit close required
        pass

if __name__ == "__main__":
    main()
