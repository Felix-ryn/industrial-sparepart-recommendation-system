# Sistem Rekomendasi Suku Cadang Mesin Industri

Hybrid Recommendation System (CBF + CF + Knowledge-Based) berbasis dataset Microsoft Azure Predictive Maintenance.

## Setup (Windows, VS Code)

### 1. Buat virtual environment
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Letakkan data raw
Salin 5 file CSV ke `data/raw/`:
```
data/raw/PdM_telemetry.csv
data/raw/PdM_errors.csv
data/raw/PdM_failures.csv
data/raw/PdM_maint.csv
data/raw/PdM_machines.csv
```

### 3. Jalankan preprocessing (SEKALI SAJA)
```bash
python scripts/run_preprocessing.py
```
Akan menghasilkan artefak di `data/processed/` dan `models/`.

### 4. Buka dashboard
```bash
streamlit run dashboard/app.py
```

### 5. EDA & eksperimen (opsional)
```bash
jupyter notebook
```
Buka `notebooks/01_eda.ipynb`

---

## Struktur file

```
pdm_recommendation/
├── data/
│   ├── raw/            ← 5 CSV asli (tidak di-commit ke git)
│   └── processed/      ← output preprocessing (auto-generated)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing_exploration.ipynb
│   └── 03_model_experiment.ipynb
├── src/
│   ├── preprocessing.py  ← merge, feature engineering, normalisasi
│   ├── cbf.py            ← Content-Based Filtering (cosine similarity sensor)
│   ├── cf.py             ← Collaborative Filtering (user-user memory-based)
│   ├── kb.py             ← Knowledge-Based (threshold rules)
│   ├── hybrid.py         ← Weighted combination + weight tuning
│   └── evaluation.py     ← Precision@K, Recall@K
├── dashboard/
│   ├── app.py            ← Streamlit entry point
│   └── components/       ← sensor_chart, risk_gauge, recommendation_table
├── scripts/
│   └── run_preprocessing.py
├── models/               ← similarity matrices + weights (auto-generated)
├── config.py             ← semua konstanta & path
└── requirements.txt
```

## Urutan pengerjaan yang disarankan

1. `01_eda.ipynb` — eksplorasi distribusi sensor, kalibrasi threshold KB
2. `02_preprocessing_exploration.ipynb` — validasi hasil merge & feature
3. `scripts/run_preprocessing.py` — generate semua artefak
4. `03_model_experiment.ipynb` — tuning bobot hybrid, evaluasi
5. `dashboard/app.py` — demo final
