# ⚙️ Industrial Spare Part Recommendation System

[![Live Demo](https://img.shields.io/badge/🚀-Live_Demo-success?style=for-the-badge)](https://industrial-sparepart-recommendation-system-uxaeqjqhwaunvq9ml7z.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

## 🌐 Live Demo

🔗 https://industrial-sparepart-recommendation-system-uxaeqjqhwaunvq9ml7z.streamlit.app/

---

## 📖 Overview

This project implements a **Hybrid Recommendation System** for industrial spare part recommendations based on **Predictive Maintenance (PdM)** data.

The system combines:

- Content-Based Filtering (CBF)
- Collaborative Filtering (CF)
- Knowledge-Based Recommendation (KB)

to identify spare parts that should be prioritized for maintenance activities and reduce unexpected machine downtime.

---

## ✨ Features

### 📊 Machine Condition Monitoring

Visualizes machine sensor data including:

- Voltage
- Rotation Speed
- Pressure
- Vibration

using historical telemetry data.

### 🔧 Hybrid Recommendation Engine

Generates Top-K spare part recommendations by combining:

- Content-Based Filtering (CBF)
- Collaborative Filtering (CF)
- Knowledge-Based Rules (KB)

### 🔍 Explainable Recommendation

Displays similar machines used during recommendation generation and explains why a component is recommended.

### 📈 Performance Evaluation

Provides evaluation metrics:

- Precision
- Recall
- F1 Score

for recommendation performance analysis.

---

## 📂 Dataset

This project uses the **Microsoft Predictive Maintenance Dataset** consisting of:

- PdM_telemetry.csv
- PdM_errors.csv
- PdM_failures.csv
- PdM_maint.csv
- PdM_machines.csv

---

## 🏗️ System Architecture

```text
Raw Data
    │
    ▼
Preprocessing & Feature Engineering
    │
    ▼
Interaction Matrix Construction
    │
    ├──► Content-Based Filtering (CBF)
    │
    ├──► Collaborative Filtering (CF)
    │
    └──► Knowledge-Based Recommendation (KB)
                │
                ▼
      Hybrid Recommendation
                │
                ▼
      Dashboard Visualization
```

---

## 🛠️ Technologies Used

### Backend & Data Processing

- Python
- Pandas
- NumPy
- Scikit-Learn

### Dashboard

- Streamlit
- Plotly

### Data Storage

- Parquet
- NumPy

---

## 📁 Project Structure

```text
pdm_recommendation/
│
├── dashboard/
│   ├── app.py
│   └── components/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── scripts/
│   └── run_preprocessing.py
│
├── src/
│   ├── preprocessing.py
│   ├── cbf.py
│   ├── cf.py
│   ├── kb.py
│   └── hybrid.py
│
├── config.py
└── requirements.txt
```

---

## ⚙️ Installation

Clone repository:

```bash
git clone https://github.com/Felix-ryn/industrial-sparepart-recommendation-system.git
cd industrial-sparepart-recommendation-system
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔄 Run Preprocessing

Generate processed datasets and similarity matrices:

```bash
python scripts/run_preprocessing.py
```

Generated artifacts:

```text
data/processed/merged_features.parquet
data/processed/sensor_normalized.parquet
data/processed/interaction_matrix.parquet

models/similarity_matrix.npy
models/cf_similarity_matrix.npy
models/norm_stats.json
```

---

## 🚀 Run Dashboard

```bash
streamlit run dashboard/app.py
```

---

## 📊 Evaluation Results

| K | Precision | Recall | F1 Score |
|---|-----------|----------|----------|
| 1 | 0.8200 | 0.3167 | 0.4477 |
| 3 | 0.7533 | 0.8683 | 0.7863 |
| 5 | 0.5200 | 0.9800 | 0.6646 |

### Best Performance

```text
K = 3
Precision = 0.7533
Recall    = 0.8683
F1 Score  = 0.7863
```

---

## 👨‍💻 Author

**Felix Ryan Agusta**  
NRP: **3324600031**  
Politeknik Elektronika Negeri Surabaya (PENS)

---

## 👨‍🏫 Supervisor

**Rony Susetyoko, S.Si., M.Si.**

---

## 📜 License

This project is developed for academic and educational purposes.
