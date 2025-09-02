# Real-Time Fraud Detection with MLOps

## Overview

This project implements a real-time fraud detection system using a scalable data engineering
and ML pipeline. It simulates continuous financial transactions, streams them through Apache
Kafka, processes them with PySpark Structured Streaming, enriches features, and applies
both supervised and unsupervised machine learning models for fraud scoring.
The system also provides real-time alerting via Kafka topics (fraud_alerts) and persists
enriched data to Parquet storage for offline analysis or model retraining.

## Architecture

1. **Data Generator** - Produces synthetic transactions (user ID, amount, location,
    timestamp).
       ○ Sends messages continuously to Kafka topic transactions.
2. **Kafka Broker (Dockerized)** - Handles real-time ingestion of transaction streams.
    ○ Topics:
       ■ transactions → incoming raw transactions
       ■ fraud_alerts → alerts for suspicious transactions
3. **Spark Structured Streaming** - Reads from Kafka.
    ○ Parses and transforms raw transactions.
    ○ Builds **feature-rich data frames** (time features, velocity features, user behavior,
       etc.).
    ○ Scores transactions using trained ML models.
    ○ Writes results to:
       ■ Console (for visibility)
       ■ Parquet (./artifacts/enriched_transactions)
       ■ Kafka (fraud_alerts) for real-time alerting.
4. **Model Training** - Implemented in train_model.py.
    ○ Two modes:
       ■ **Supervised** (Logistic Regression / XGBoost): requires labeled transactions.
       ■ **Unsupervised** (IsolationForest): anomaly detection without labels.
    ○ Features extracted include velocity, rolling window stats, log-transforms, and
       location frequency.
    ○ Artifacts saved:
       ■ model.pkl → trained model
       ■ preprocess_scaler.pkl → feature scaler
       ■ features.json → exact feature order
       ■ threshold.json → threshold, mode, calibration metadata


5. **Real-Time Scoring (Spark)** - Loads artifacts.
    ○ Normalizes new batch scores consistently (using training min/max).
    ○ Compares against threshold.
    ○ Publishes fraud alerts to Kafka.

## Project Structure

#### 

    ├── data-generator/         # Kafka producer generating random transactions
    ├── spark_streaming/        # Spark Structured Streaming jobs
    │   └── main.py             # Main streaming job with alerting
    ├── models/
    │   ├── train_model.py      # Training pipeline (supervised & unsupervised)
    │   ├── model.pkl           # Saved trained model
    │   ├── preprocess_scaler.pkl
    │   ├── features.json
    │   └── threshold.json
    ├── artifacts/              # Output (Parquet, alerts, scored results)
    └── docker-compose.yml      # Kafka, Zookeeper setup

## Setup & Installation

### 1. Clone Repo

    git clone <your-repo-url>
    cd <your-repo-name>

### 2. Start Kafka with Docker

    docker compose up -d
    Verify broker:
    nc -zv localhost 9092

### 3. Create Kafka Topics

    docker exec -it rtfd-mlops-kafka-1 kafka-topics \
    --create --topic transactions --bootstrap-server localhost:9092 \
    --partitions 1 --replication-factor 1


    docker exec -it rtfd-mlops-kafka-1 kafka-topics \
    --create --topic fraud_alerts --bootstrap-server localhost:9092 \
    --partitions 1 --replication-factor 1

### 4. Start Data Generator

    cd data-generator
    python3 main.py

### 5. Run Spark Streaming (with Kafka package)

    PYSPARK_SUBMIT_ARGS="--packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.
    pyspark-shell" \
    python3 -m spark_streaming.main
    You’ll see fraud scores and flags in the console.

### 6. Tail Alerts

    docker exec -it rtfd-mlops-kafka-1 kafka-console-consumer \
    --bootstrap-server localhost:9092 \
    --topic fraud_alerts \
    --from-beginning

## Alerting

    Suspicious transactions (is_fraud = true) are written to the fraud_alerts Kafka topic.
    Each alert message contains:
    {
    "transaction_id": "...",
    "user_id": "user_123",
    "fraud_score": 0.87,
    "timestamp": "2025-08-29T11:42:18.123Z"
    }


