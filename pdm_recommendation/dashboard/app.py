"""
app.py — Streamlit Dashboard Entry Point
──────────────────────────────────────────
Jalankan dengan:
  streamlit run dashboard/app.py

Dashboard ini HANYA membaca dari data/processed/ dan models/
— tidak pernah menjalankan preprocessing lagi.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path

# Root path supaya import src.* jalan
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    APP_TITLE, PROC_FILES, MODEL_FILES, SENSOR_COLS,
    COMPONENT_COLS, DEFAULT_MACHINE_ID
)
from src.cbf    import load_similarity_matrix
from src.cf     import cf_recommend, compute_cf_similarity, get_cf_explanation
from src.kb     import kb_recommend, get_kb_alerts
from src.hybrid import hybrid_recommend, load_weights
from dashboard.components.sensor_chart        import render_sensor_chart
from dashboard.components.risk_gauge          import render_risk_gauge
from dashboard.components.recommendation_table import render_recommendation_table


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="⚙️",
    layout="wide",
)


# ── Load artefak (cached — hanya sekali per session) ──────────────────────────

@st.cache_data(show_spinner="Memuat data sensor...")
def load_data():
    """Load semua artefak preprocessed. Cached supaya tidak reload tiap interaksi."""
    if not PROC_FILES["merged"].exists():
        st.error(
            "❌ File processed tidak ditemukan.\n\n"
            "Jalankan terlebih dahulu:\n"
            "```\npython scripts/run_preprocessing.py\n```"
        )
        st.stop()

    merged      = pd.read_parquet(PROC_FILES["merged"])
    interaction = pd.read_parquet(PROC_FILES["interaction"])
    sensor_norm = pd.read_parquet(PROC_FILES["sensor_norm"])
    return merged, interaction, sensor_norm


@st.cache_resource(show_spinner="Memuat model similarity...")
def load_models():
    """Load similarity matrices. cache_resource karena objek besar."""
    sim_cbf = load_similarity_matrix()

    cf_path     = MODEL_FILES["sim_matrix"].parent / "cf_similarity_matrix.npy"
    cf_idx_path = cf_path.with_suffix(".index.npy")
    cf_arr = np.load(cf_path)
    cf_idx = np.load(cf_idx_path)
    sim_cf  = pd.DataFrame(cf_arr, index=cf_idx, columns=cf_idx)

    stats_path = MODEL_FILES["weights"].parent / "norm_stats.json"
    norm_stats = json.load(open(stats_path)) if stats_path.exists() else {}

    weights = load_weights()
    return sim_cbf, sim_cf, norm_stats, weights


# ── Main layout ────────────────────────────────────────────────────────────────

def main():
    st.title(f"⚙️ {APP_TITLE}")
    st.caption("Sistem rekomendasi suku cadang berbasis Hybrid CBF + CF + Knowledge-Based")

    merged, interaction, sensor_norm = load_data()
    sim_cbf, sim_cf, norm_stats, weights = load_models()

    all_machine_ids = sorted(merged["machineID"].unique().tolist())

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Pilih Mesin")
        machine_id = st.selectbox(
            "Machine ID",
            options=all_machine_ids,
            index=all_machine_ids.index(DEFAULT_MACHINE_ID)
                  if DEFAULT_MACHINE_ID in all_machine_ids else 0,
        )

        st.divider()
        st.subheader("Konfigurasi Hybrid")
        st.caption("Bobot saat ini (dari weights_config.json):")
        st.write(f"- CBF: **{weights['cbf']:.2f}**")
        st.write(f"- CF:  **{weights['cf']:.2f}**")
        st.write(f"- KB:  **{weights['kb']:.2f}**")

        top_k = st.slider("Top-K rekomendasi", min_value=1, max_value=4, value=3)

        st.divider()
        st.caption("💡 Jalankan `scripts/run_preprocessing.py` untuk refresh data.")

    # ── Data terkini mesin ─────────────────────────────────────────────────────
    machine_data = merged[merged["machineID"] == machine_id].sort_values("datetime")
    if machine_data.empty:
        st.error(f"Tidak ada data untuk machineID {machine_id}")
        return

    latest     = machine_data.iloc[-1]
    model_info = machine_data[["model", "age"]].iloc[0] if "model" in machine_data.columns else None

    # Header info mesin
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Machine ID", machine_id)
    if model_info is not None:
        col2.metric("Model",  model_info.get("model", "-"))
        col3.metric("Usia Mesin", f"{model_info.get('age', '-')} tahun")
    col4.metric("Data Terakhir", str(latest["datetime"])[:16])

    st.divider()

    # ── Tab layout ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📊 Kondisi Sensor", "🔧 Rekomendasi", "📈 Evaluasi"])

    # ── TAB 1: Kondisi Sensor ──────────────────────────────────────────────────
    with tab1:
        render_sensor_chart(machine_data, SENSOR_COLS)

    # ── TAB 2: Rekomendasi ─────────────────────────────────────────────────────
    with tab2:
        # Ambil nilai sensor & days_since terkini
        latest_sensor = {col: float(latest[col]) for col in SENSOR_COLS if col in latest}
        days_since    = {
            comp: float(latest.get(f"days_since_{comp}", 9999))
            for comp in COMPONENT_COLS
        }

        # KB alerts dulu (safety layer paling atas)
        alerts = get_kb_alerts(machine_id, latest_sensor, days_since)
        if alerts:
            st.subheader("⚠️ Alert Kondisi Kritis")
            for alert in alerts:
                severity_icon = "🔴" if alert["severity"] == "critical" else \
                                "🟠" if alert["severity"] == "high" else "🟡"
                st.warning(f"{severity_icon} {alert['detail']}")

        st.subheader(f"Rekomendasi Top-{top_k} Suku Cadang")

        # Hybrid recommendation
        recs = hybrid_recommend(
            machine_id, interaction, sim_cbf, sim_cf,
            latest_sensor, days_since, top_k=top_k, weights=weights
        )

        # Risk gauge per komponen
        render_risk_gauge(recs)

        # Tabel rekomendasi detail
        render_recommendation_table(recs)

        # Penjelasan CF
        st.subheader("🔍 Mengapa direkomendasikan? (CF Neighbors)")
        explanations = get_cf_explanation(machine_id, interaction, sim_cf, top_n_neighbors=3)
        for exp in explanations:
            st.write(
                f"- Mesin **{exp['neighbor_machine']}** "
                f"(similarity: {exp['similarity']:.3f}) "
                f"— sering ganti **{exp['most_replaced']}**"
            )

    # ── TAB 3: Evaluasi ────────────────────────────────────────────────────────
    with tab3:
        st.info(
            "Evaluasi penuh dijalankan dari notebook `03_model_experiment.ipynb`. "
            "Tab ini menampilkan hasil yang sudah disimpan."
        )
        eval_path = ROOT / "models" / "evaluation_results.csv"
        if eval_path.exists():
            eval_df = pd.read_csv(eval_path)
            summary = eval_df.groupby("k")[["precision", "recall", "f1"]].mean().round(4)
            st.dataframe(summary, use_container_width=True)
        else:
            st.warning("Belum ada hasil evaluasi. Jalankan notebook 03 terlebih dahulu.")


if __name__ == "__main__":
    main()
