"""Komponen gauge/progress bar risiko per komponen."""
import streamlit as st
import pandas as pd

def render_risk_gauge(recs: pd.DataFrame):
    """
    Tampilkan final_score tiap komponen sebagai progress bar berwarna.
    
    Args:
        recs: DataFrame dari hybrid_recommend() dengan kolom
              [comp, final_score, cbf_score, cf_score, kb_score, rank]
    """
    cols = st.columns(len(recs))
    for i, (_, row) in enumerate(recs.iterrows()):
        score = float(row["final_score"])
        color = "🔴" if score >= 0.7 else "🟠" if score >= 0.4 else "🟢"
        with cols[i]:
            st.metric(
                label=f"{color} {row['comp'].upper()}",
                value=f"#{int(row['rank'])} — {score:.3f}",
            )
            st.progress(score)
            st.caption(
                f"CBF: {row['cbf_score']:.2f} | "
                f"CF: {row['cf_score']:.2f} | "
                f"KB: {row['kb_score']:.2f}"
            )
