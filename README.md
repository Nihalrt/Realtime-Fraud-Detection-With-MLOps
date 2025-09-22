# Real-Time Fraud Detection with MLOps

## Overview

This project implements a real-time fraud detection system using a scalable data engineering
and ML pipeline. It simulates continuous financial transactions, streams them through **Google Cloud Pub/Sub (GCP)**, processes them with PySpark Structured Streaming, enriches features, and applies
both supervised and unsupervised machine learning models for fraud scoring.
The system also provides real-time alerting via Pub/Sub topics (`fraud_alerts`) and persists
enriched data to Parquet storage for offline analysis or model retraining.

## Architecture

1. **Data Generator** – Produces synthetic transactions (user ID, amount, location, timestamp).  
   - Publishes messages continuously to Pub/Sub topic `transactions`.

2. **Google Cloud Pub/Sub** – Handles real-time ingestion of transaction streams.  
   - Topics:  
     - `transactions` → incoming raw transactions  
     - `fraud_alerts` → alerts for suspicious transactions  

3. **Spark Structured Streaming** – Reads from Pub/Sub.  
   - Parses and transforms raw transactions.  
   - Builds **feature-rich data frames** (time features, velocity features, user behavior, etc.).  
   - Scores transactions using trained ML models.  
   - Writes results to:  
     - Console (for visibility)  
     - Parquet (`./artifacts/enriched_transactions`)  
     - Pub/Sub (`fraud_alerts`) for real-time alerting.  

4. **Model Training** – Implemented in `train_model.py`.  
   - Two modes:  
     - **Supervised** (Logistic Regression / XGBoost): requires labeled transactions.  
     - **Unsupervised** (IsolationForest): anomaly detection without labels.  
   - Features extracted include velocity, rolling window stats, log-transforms, and location frequency.  
   - Artifacts saved:  
     - `model.pkl` → trained model  
     - `preprocess_scaler.pkl` → feature scaler  
     - `features.json` → exact feature order  
     - `threshold.json` → threshold, mode, calibration metadata  

5. **Real-Time Scoring (Spark)** – Loads artifacts.  
   - Normalizes new batch scores consistently (using training min/max).  
   - Compares against threshold.  
   - Publishes fraud alerts to Pub/Sub.

## Project Structure

    ├── data-generator/ # Pub/Sub publisher generating random transactions
    ├── spark_streaming/ # Spark Structured Streaming jobs
    │ └── main.py # Main streaming job with alerting
    ├── models/
    │ ├── train_model.py # Training pipeline (supervised & unsupervised)
    │ ├── model.pkl # Saved trained model
    │ ├── preprocess_scaler.pkl
    │ ├── features.json
    │ └── threshold.json
    ├── artifacts/ # Output (Parquet, alerts, scored results)
    └── streamlit_app/ # Streamlit dashboard for visualization


## Setup & Installation

### 1. Clone Repo
```bash
git clone <your-repo-url>
cd <your-repo-name>
```


### 2. Authenticate GCP and Set Project
```bash
gcloud auth application-default login
gcloud config set project <your-gcp-project-id>
```


### 3. Create Pub/Sub Topics
```bash
gcloud pubsub topics create transactions
gcloud pubsub topics create fraud_alerts
```


### 4. Start Data Generator
```bash
cd data-generator
python3 main.py
```


### 5. Run Spark Streaming (with Pub/Sub connector)
```bash
python3 -m spark_streaming.main
```

### 6. Pull Alerts from Pub/Sub
```bash
gcloud pubsub subscriptions create fraud_alerts-sub --topic=fraud_alerts
gcloud pubsub subscriptions pull fraud_alerts-sub --auto-ack --limit=5
```





    
