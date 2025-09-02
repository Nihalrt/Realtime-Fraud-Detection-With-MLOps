import random
import uuid
import json
import time
from collections import deque, defaultdict
from datetime import datetime, timezone, timedelta
from kafka import KafkaProducer
from networkx.drawing import draw_networkx_nodes

producer = KafkaProducer(bootstrap_servers=['localhost:9092'], value_serializer=lambda v: json.dumps(v).encode('utf-8'))


TOPIC = "transactions"


US_STATES = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA",
    "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
    "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
    "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY",
]

# A set of state pairs that are far apart, intented for an "impossible travel" scenario.
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
    __slots__ = ("recent_ts", "recent_amts", "last_state", "seen_count", "burst_remaining", "replay_remaining", "replay_amount")

    def __init__(self):
        self.recent_ts = deque(maxlen=50)
        self.recent_amts = deque(maxlen=50)
        self.last_state = None
        self.seen_count = 0
        self.burst_remaining = 0
        self.replay_remaining = 0
        self.replay_amount = None

USER = defaultdict(UserState)

#----------------------------------
# Condition toggles (tuned during generation)
#----------------------------------

FRAUD_RATE = 0.10    # 10% of the events would be fraudulent patterns
P_BURST = 0.30       # among frauds, chance to start/ continue a burst
P_SPIKE = 0.30       # amount spike (if an amt increases quickly, it's fraud)
P_REPLAY = 0.20      # repeated identical amounts
P_TRAVEL = 0.15      # impossible travel scenarios of txns to travel
P_NEW_USER = 0.05    # New users immediately become active

# Amount bands for anomalies
def draw_spike_amount():
    return round(random.uniform(2000.0, 10000.0), 2)

def pick_far_state(current):

    for s in US_STATES:
        if s!=current and (current, s) not in FAR_STATE_PAIRS:
            return s
    return random.choice([st for st in US_STATES if st != current])

def make_base_txn(user_id: str, amount: float, state: str, label=0, reason=None, ts=None):
    ts_iso = ts.isoformat() if isinstance(ts, datetime) else now_utc()
    return {
        "transaction_id": str(uuid.uuid4()),
        "user_id": user_id,
        "amount": float(amount),
        "location": state,
        "timestamp": ts_iso,
        "label": int(label),
        "fraud_reason": reason,
    }

def normal_txn(u: UserState, user_id: str):
    amt = draw_normal_amount()
    state = u.last_state or draw_state()
    txn = make_base_txn(user_id, amt, state, label=0, reason=None)
    return txn

def fraud_amount_spike(u: UserState, user_id: str):
    amt = draw_spike_amount()
    state = u.last_state or draw_state()
    txn = make_base_txn(user_id, amt, state, label=1, reason="amount_spike")
    return txn

def fraud_replay(u: UserState, user_id: str):
    if u.replay_remaining <= 0:
        u.replay_amount = round(random.uniform(80.0, 150.0), 2)
        u.replay_remaining = random.randint(5,9)
    u.replay_remaining -=1
    state = u.last_state or draw_state()
    return make_base_txn(user_id, u.replay_amount, state, label=1, reason="multiple_txns_of_same_amount")

def fraud_burst(u: UserState, user_id: str):
    if u.burst_remaining <= 0:
        u.burst_remaining = random.randint(5,9)
    u.burst_remaining -=1
    amt = round(random.uniform(20.0, 500.0), 2)
    state = u.last_state or draw_state()
    return make_base_txn(user_id, amt, state, label=1, reason="large_no_of_txns_in_small_timeframe")

def fraud_Impossible(u: UserState, user_id: str):
    prev = u.last_state or draw_state()
    travel_state = pick_far_state(prev)
    amt = draw_normal_amount()
    return make_base_txn(user_id, amt, travel_state, label=1, reason="Impossible_travel_txn")

def fraud_new_user_high_activity():
    user_id = f"user_{random.randint(10001, 20000)}"
    u = USER[user_id]
    u.burst_remaining = random.randint(5,10)
    return fraud_amount_spike(u, user_id), user_id

#---------------------------------------
# Send Txn To The Created Topic
#---------------------------------------

def send_transaction(txn):
    producer.send(TOPIC, value=txn)
    print("Sent transaction: ", txn)


def step_once():

    if USER and random.random() < 0.8:
        user_id = random.choice(list(USER.keys()))
    else:
        user_id = f"user_{random.randint(10001, 20000)}"

    u = USER[user_id]

    is_fraud = (random.random() < FRAUD_RATE)

    if u.burst_remaining > 0:
        txn = fraud_burst(u, user_id)
        delay = 0.05
    elif u.replay_remaining > 0:
        txn = fraud_replay(u, user_id)
        delay = 0.1
    elif is_fraud:
        r = random.random()
        if r < P_BURST:
            txn = fraud_burst(u, user_id)
            delay = 0.05
        elif r < P_BURST + P_SPIKE:
            txn = fraud_amount_spike(u, user_id)
            delay = 0.3
        elif r < P_BURST + P_SPIKE + P_REPLAY:
            txn = fraud_replay(u, user_id)
            delay = 0.1
        elif r < P_BURST + P_SPIKE + P_REPLAY + P_TRAVEL:
            txn = fraud_Impossible(u, user_id)
            delay = 0.5
        else:
            txn, user_id = fraud_new_user_high_activity()
            u = USER[user_id]
            delay = 0.05
    else:
        txn = normal_txn(u, user_id)
        delay = 0.5

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
        producer.flush()
        producer.close()

if __name__ == '__main__':
    main()


