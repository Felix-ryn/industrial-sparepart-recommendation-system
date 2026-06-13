"""Komponen tabel rekomendasi detail."""
import streamlit as st
import pandas as pd

def render_recommendation_table(recs: pd.DataFrame):
    """
    Tampilkan tabel rekomendasi dengan formatting skor.
    
    Args:
        recs: DataFrame dari hybrid_recommend()
    """
    display = recs[["rank", "comp", "final_score", "cbf_score", "cf_score", "kb_score"]].copy()
    display.columns = ["Rank", "Komponen", "Skor Final", "CBF", "CF", "KB"]
    display = display.set_index("Rank")

    st.dataframe(
        display.style.format({
            "Skor Final": "{:.4f}",
            "CBF"       : "{:.4f}",
            "CF"        : "{:.4f}",
            "KB"        : "{:.4f}",
        }).background_gradient(subset=["Skor Final"], cmap="RdYlGn"),
        use_container_width=True,
    )
