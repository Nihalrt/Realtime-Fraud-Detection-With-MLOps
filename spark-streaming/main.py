import json

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, when, lit
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
)
import numpy as np
from joblib import load

from models.train_model import build_feature_frame

def to_iso(ts_col):
    from pyspark.sql.functions import col
    return col(ts_col)

def build_spark_session() -> SparkSession:
    spark = (
        SparkSession.builder
        .appName("SparkStreaming")
        .master("local[*]")
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.0")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    return spark

def load_artifacts(spark: SparkSession):

    model = load("./models/model.pkl")
    scaler = load("./models/preprocess_scaler.pkl")

    with open("./models/features.json") as f:
        features = json.load(f)
    with open("./models/threshold.json") as f:
        meta = json.load(f)

    return (
        spark.sparkContext.broadcast(model), spark.sparkContext.broadcast(scaler), spark.sparkContext.broadcast(features), spark.sparkContext.broadcast(meta)
    )

def make_kafka_schema():

    transaction_schema = StructType([
        StructField("transaction_id", StringType(), nullable=False),
        StructField("user_id",        StringType(), nullable=False),
        StructField("amount",         DoubleType(), nullable=False),
        StructField("location",       StringType(), nullable=False),
        StructField("timestamp",      StringType(), nullable=False),
    ])

    return transaction_schema

def score_batch_factory(spark, bc_model, bc_scaler, bc_features, bc_meta):
    def score_batch(batch_df, batch_id: int):
        if batch_df.rdd.isEmpty():
            return

        pdf = batch_df.toPandas()
        feat_df, _, _ = build_feature_frame(pdf)

        needed = list(bc_meta.value.get("features", bc_features.value))
        for c in needed:
            if c not in feat_df.columns:
                feat_df[c] = 0.0
        X  = feat_df[needed].astype(float)
        Xs = bc_scaler.value.transform(X)

        model   = bc_model.value
        meta    = bc_meta.value
        thresh  = float(meta["threshold"])
        scale   = meta.get("score_scale")             # e.g. "proba" or None
        calib   = meta.get("score_calibration", {})   # has min_raw / max_raw if provided

        # ---- produce scores on the intended scale ----
        if hasattr(model, "predict_proba"):           # supervised path
            scores = model.predict_proba(Xs)[:, 1]
        else:
            raw = -model.decision_function(Xs)
            if scale == "proba" and calib.get("type") == "minmax":
                mn = float(calib["min"]); mx = float(calib["max"])
                scores = (raw - mn) / (mx - mn + 1e-9)   # same calibration as training
            else:
                scores = raw  # threshold must then be on the raw scale

        flags = (scores >= thresh)

        out = feat_df[["transaction_id", "user_id", "timestamp"]].copy()
        out["fraud_score"] = scores.astype(float)
        out["is_fraud"]    = flags.astype(bool)

        out_sdf = spark.createDataFrame(out)
        out_sdf.show(20, False)

        # 3) Publish real-time alerts (only flagged rows) to Kafka
        alerts = (
            out_sdf
            .filter("is_fraud = true")
            .selectExpr(
                "transaction_id",
                "user_id",
                "CAST(fraud_score AS DOUBLE) AS fraud_score",
                "CAST(timestamp AS STRING)  AS timestamp"
            )
        )

        if alerts.limit(1).count() > 0:
            alerts_json = alerts.selectExpr(
                "to_json(named_struct("
                "'transaction_id', transaction_id, "
                "'user_id',        user_id, "
                "'fraud_score',    fraud_score, "
                "'timestamp',      timestamp"
                ")) AS value"
            )


            alerts_json.write \
                .format("kafka") \
                .option("kafka.bootstrap.servers", "localhost:9092") \
                .option("topic", "fraud_alerts") \
                .mode("append") \
                .save()

    return score_batch

def main():
    spark = build_spark_session()
    bc_model, bc_scaler, bc_features, bc_meta = load_artifacts(spark)

    kafka_schema = make_kafka_schema()
    raw_df = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", "localhost:9092")
        .option("subscribe", "transactions")
        .option("startingOffsets", "latest")     # dev: follow new data
        .option("failOnDataLoss", "false")       # skip gaps if broker purged data
        .load()
    )

    # Parse the JSON in the value section
    value_df = raw_df.selectExpr("CAST(value AS STRING) as json_str")

    parsed_df = (
        value_df
        .select(from_json(col("json_str"), kafka_schema).alias("data"))
        .select("data.*")
    )

    query = (
        parsed_df.writeStream
        .foreachBatch(
            score_batch_factory(spark, bc_model, bc_scaler, bc_features, bc_meta)
        )
        .option("checkpointLocation", "./spark-streaming/_chkpt_scoring")
        .start()
    )
    query.awaitTermination()


if __name__ == "__main__":
    main()




