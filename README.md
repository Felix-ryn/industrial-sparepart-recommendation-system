# Industrial Spare Part Recommendation System

## Overview

This project implements a Hybrid Recommendation System for industrial spare part recommendations based on Predictive Maintenance (PdM) data.

The system combines:

* Content-Based Filtering (CBF)
* Collaborative Filtering (CF)
* Knowledge-Based Recommendation (KB)

to identify spare parts that should be prioritized for maintenance activities and reduce unexpected machine downtime.

---

## Features

### Machine Condition Monitoring

Visualizes machine sensor data:

* Voltage
* Rotation Speed
* Pressure
* Vibration

using real-time historical telemetry data.

### Hybrid Recommendation Engine

Generates Top-K spare part recommendations using:

* Content-Based Filtering
* Collaborative Filtering
* Knowledge-Based Rules

### Explainable Recommendation

Displays similar machines used during recommendation generation and explains why a component is recommended.

### Performance Evaluation

Provides:

* Precision
* Recall
* F1 Score

for recommendation performance analysis.

---

## Dataset

This project uses the Microsoft Predictive Maintenance Dataset consisting of:

* PdM_telemetry.csv
* PdM_errors.csv
* PdM_failures.csv
* PdM_maint.csv
* PdM_machines.csv

---

## System Architecture

Raw Data

↓

Preprocessing & Feature Engineering

↓

Interaction Matrix Construction

↓

Content-Based Filtering

↓

Collaborative Filtering

↓

Knowledge-Based Recommendation

↓

Hybrid Recommendation

↓

Dashboard Visualization

---

## Technologies Used

### Backend & Data Processing

* Python
* Pandas
* NumPy
* Scikit-learn

### Dashboard

* Streamlit

### Data Storage

* Parquet
* NumPy

---

## Installation

Clone repository:

```bash
git clone https://github.com/yourusername/industrial-sparepart-recommendation-system.git
cd industrial-sparepart-recommendation-system
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Preprocessing

Generate processed data and similarity models:

```bash
python scripts/run_preprocessing.py
```

Generated files:

```text
data/processed/merged_features.parquet
data/processed/sensor_normalized.parquet
data/processed/interaction_matrix.parquet

models/similarity_matrix.npy
models/cf_similarity_matrix.npy
models/norm_stats.json
```

---

## Run Dashboard

```bash
streamlit run dashboard/app.py
```

---

## Evaluation Result

| K | Precision | Recall | F1 Score |
| - | --------- | ------ | -------- |
| 1 | 0.82      | 0.3167 | 0.4477   |
| 3 | 0.7533    | 0.8683 | 0.7863   |
| 5 | 0.52      | 0.98   | 0.6646   |

Best performance achieved at:

```text
K = 3
F1 Score = 0.7863
```

---

## Author

Felix Ryan Agusta

NRP: 3324600031

Politeknik Elektronika Negeri Surabaya (PENS)

---

## Supervisor

Rony Susetyoko, S.Si., M.Si.
