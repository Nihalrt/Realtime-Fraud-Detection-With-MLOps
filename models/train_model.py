import os, glob, json, warnings
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.ensemble import IsolationForest
from sklearn.utils.class_weight import compute_class_weight
import joblib

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except:
    HAS_XGBOOST = False
    from sklearn.linear_model import LogisticRegression


DATA_DIR = "./artifacts/enriched_transactions"
ARTIFACT_DIR = "./models"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def load_parquet_folder(path:str, limit_files: int=200) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(path, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {path}")

    files = files[-limit_files:]
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    return df

def parse_times(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        raise ValueError("Expected column 'timestamp' not found in dataframe")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"])
    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df['hour'] = df["timestamp"].dt.hour
    df['dow'] = df["timestamp"].dt.dayofweek

    #cyclic encoding for hours
    df['hour_sin'] = np.sin(2 * np.pi * df["hour"] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df["hour"] / 24)
    return df

def add_global_frequency_features(df: pd.DataFrame) -> pd.DataFrame:
    loc_freq = df["location"].value_counts(normalize=True)
    df["location_freq_global"] = df["location"].map(loc_freq).fillna(0.0)
    user_freq = df["user_id"].value_counts()
    df["user_txn_count_global"] = df["user_id"].map(user_freq).fillna(0).astype(int)
    return df

def add_user_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    # sort per user/time
    df = df.sort_values(["user_id", "timestamp"]).copy()

    # seconds since previous txn for the user
    df["secs_since_prev_txn"] = (
        df.groupby("user_id")["timestamp"].diff().dt.total_seconds().fillna(1e9)
    )

    def _user_roll(g: pd.DataFrame) -> pd.DataFrame:
        # index by time for time-based windows
        g = g.set_index("timestamp").copy()

        # use prior-only values to avoid leakage
        amt_prev = g["amount"].shift(1)

        # 1h window
        g["user_cnt_1h"]  = amt_prev.rolling("1h").count()
        g["user_sum_1h"]  = amt_prev.rolling("1h").sum()
        g["user_mean_1h"] = amt_prev.rolling("1h").mean()

        # 24h window
        g["user_cnt_24h"]  = amt_prev.rolling("24h").count()
        g["user_sum_24h"]  = amt_prev.rolling("24h").sum()
        g["user_mean_24h"] = amt_prev.rolling("24h").mean()

        # 7d window
        g["user_cnt_7d"]  = amt_prev.rolling("7d").count()
        g["user_sum_7d"]  = amt_prev.rolling("7d").sum()
        g["user_mean_7d"] = amt_prev.rolling("7d").mean()
        g["user_std_7d"]  = amt_prev.rolling("7d").std()

        # z-score vs trailing 7d (avoid div-by-zero)
        mu7 = g["user_mean_7d"]
        sd7 = g["user_std_7d"].replace(0, np.nan)
        g["user_amt_z_7d"] = ((g["amount"] - mu7) / sd7).replace([np.inf, -np.inf], np.nan)

        # same-as-previous-location flag (robust boolean→float conversion)
        g["same_as_prev_loc"] = (
            g["location"]
            .eq(g["location"].shift(1))
            .fillna(False)
            .astype(np.float32)
        )

        return g.reset_index()

    out = (
        df.groupby("user_id", group_keys=False)
          .apply(_user_roll)
    )

    # Fill NaNs from early history
    fill0 = [
        "user_cnt_1h","user_sum_1h","user_mean_1h",
        "user_cnt_24h","user_sum_24h","user_mean_24h",
        "user_cnt_7d","user_sum_7d","user_mean_7d",
        "secs_since_prev_txn","same_as_prev_loc"
    ]
    for c in fill0:
        if c in out.columns:
            out[c] = out[c].fillna(0.0)

    out["user_std_7d"]   = out["user_std_7d"].fillna(0.0)
    out["user_amt_z_7d"] = out["user_amt_z_7d"].fillna(0.0)

    return out


def add_amount_transforms(df: pd.DataFrame) -> pd.DataFrame:
    df["amount_log1p"] = np.log1p(df["amount"].clip(lower=0))
    return df

def build_feature_frame(df: pd.DataFrame):
    df = df[["transaction_id", "user_id", "amount", "location", "timestamp"]].copy()
    # normalize names for safety
    df.columns = df.columns.str.lower()

    df = parse_times(df)
    df = add_time_features(df)
    df = add_global_frequency_features(df)
    df = add_user_velocity_features(df)
    df = add_amount_transforms(df)
    df = pd.get_dummies(df, columns=["location"], prefix="loc")


    base_features = [
        "amount", "amount_log1p",
        "hour_sin", "hour_cos", "dow",
        "location_freq_global", "user_txn_count_global",
        "secs_since_prev_txn",
        "user_cnt_1h", "user_sum_1h", "user_mean_1h",
        "user_cnt_24h", "user_sum_24h", "user_mean_24h",
        "user_cnt_7d", "user_sum_7d", "user_mean_7d", "user_std_7d",
        "user_amt_z_7d",
        "same_as_prev_loc",
    ]
    # Dynamically include all dummy columns for location
    loc_dummy_cols = [c for c in df.columns if c.startswith("loc_")]

    features = base_features + loc_dummy_cols

    # 5) Ensure every feature exists (if streaming hasn’t produced some columns yet)
    missing_feats = [c for c in features if c not in df.columns]
    for c in missing_feats:
        # create neutral columns if absent (e.g., unseen dummy category)
        df[c] = 0.0

    X = df[features].astype(float).copy()
    y = df["label"].astype(int) if "label" in df.columns else None
    feature_df = pd.concat([df[["transaction_id", "user_id", "timestamp"]], X], axis=1)
    return feature_df, features, y

def time_split(df: pd.DataFrame, time_col="timestamp", holdout_frac=0.2):
    df = df.sort_values(time_col)
    n = len(df)
    split = int(n * (1-holdout_frac))
    return df.iloc[:split], df.iloc[split:]


def train_supervised_model(df_features: pd.DataFrame, features: List[str], y: pd.Series):
    mask = y.notna()
    Xy = df_features.loc[mask, features + ["timestamp"]].copy()
    y = y.loc[mask].astype(int)

    #Time-based split to avoid data leakage
    train_df, val_df = time_split(pd.concat([Xy, y.rename("label")], axis=1), time_col="timestamp", holdout_frac=0.2)
    x_train, y_train = train_df[features], train_df["label"]
    X_val, y_val = val_df[features], val_df["label"]

    # Scale robustly
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(x_train)
    X_val_s = scaler.transform(X_val)

    # Class imbalance handling
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    if n_pos == 0:
        raise ValueError("No positive transactions found")
    scale_pos_weight = max(n_neg/ max(n_pos, 1), 1.0)

    if HAS_XGBOOST:
        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            random_state=42,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=4,
            scale_pos_weight=scale_pos_weight,

        )
    else:
        model = LogisticRegression(
            class_weight="balanced", max_iter=500, n_jobs=None
        )
    model.fit(X_train_s, y_train)



    if hasattr(model, "predict_proba"):
        val_scores = model.predict_proba(X_val_s)[:, 1]
        score_scale = "proba"
    else:
        raw = model.decision_function(X_val_s)
        val_scores = 1.0 / (1.0 + np.exp(-raw))   # logistic squashing
        score_scale = "logit"   # or "raw+logistic", your choice

    roc = roc_auc_score(y_val, val_scores)
    pr  = average_precision_score(y_val, val_scores)
    prec, rec, thr = precision_recall_curve(y_val, val_scores)
    f1 = (2 * prec * rec) / (prec + rec + 1e-9)
    best_idx = int(np.nanargmax(f1))
    best_thr = float(thr[max(best_idx-1, 0)]) if best_idx < len(thr) else 0.5

    artifacts = {
        "scaler": scaler,
        "model": model,
        "threshold": best_thr,
        "features": features,
        "metrics": {"val_roc_auc": float(roc), "val_pr_auc": float(pr)},
        "mode": "supervised",
        "score_scale": score_scale   # <-- add this
    }
    return artifacts

def train_unsupervised_model(df_features: pd.DataFrame, features: List[str]):
    X = df_features[features].astype(int)
    scaler = RobustScaler()
    Xs = scaler.fit_transform(X)
    iso = IsolationForest(
        n_estimators=300,
        contamination=0.03,
        max_samples="auto",
        random_state=42,
        n_jobs=-1
    )
    iso.fit(Xs)

    raw = -iso.decision_function(Xs)
    minv, maxv = float(np.min(raw)), float(np.max(raw))
    proba = (raw - minv) / (maxv - minv + 1e-9)

    thr = float(np.quantile(proba, 0.97))

    artifacts = {
        "scaler": scaler,
        "model": iso,
        "threshold": thr,
        "features": features,
        "metrics": {},
        "mode": "unsupervised",
        "score_scale": "proba",
        "score_calibration": {
            "type": "minmax",
            "min": minv,
            "max": maxv,
        }
    }
    return artifacts

def main():
    df_raw = load_parquet_folder(DATA_DIR)


    df_raw = df_raw.drop_duplicates(subset=["transaction_id"]).reset_index(drop=True)

    df_feat, features, y = build_feature_frame(df_raw)

    if y is not None and y.dropna().sum() > 0:
        artifacts = train_supervised_model(df_feat, features, y)
        mode = "supervised"
    else:
        warnings.warn("No labels found")
        mode = "unsupervised"
        artifacts = train_unsupervised_model(df_feat, features)

    # Save artifacts
    joblib.dump(artifacts["scaler"], os.path.join(ARTIFACT_DIR, "preprocess_scaler.pkl"))
    joblib.dump(artifacts["model"],  os.path.join(ARTIFACT_DIR, "model.pkl"))
    with open(os.path.join(ARTIFACT_DIR, "features.json"), "w") as f:
        json.dump(artifacts["features"], f)
    with open(os.path.join(ARTIFACT_DIR, "threshold.json"), "w") as f:
        json.dump({"threshold": artifacts["threshold"], "mode": mode, "metrics": artifacts.get("metrics", {}), "score_scale": artifacts.get("score_scale"), "score_calibration": artifacts.get("score_calibration")}, f)

    print(f"[OK] Trained ({mode}). Saved to {ARTIFACT_DIR}/ "
          f"\n - model.pkl\n - preprocess_scaler.pkl\n - features.json\n - threshold.json")
    if artifacts.get("metrics"):
        print("Validation metrics:", artifacts["metrics"])

if __name__ == "__main__":
    main()











