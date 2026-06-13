"""
cf.py — Collaborative Filtering (Memory-Based, User-User)
──────────────────────────────────────────────────────────
Mesin = "user", Komponen = "item"
Rekomendasi komponen berdasarkan pola penggantian mesin-mesin dengan
histori maintenance serupa.

Kenapa memory-based, bukan model-based (SVD, NMF)?
  - Dataset ini hanya 100 mesin × 4 komponen → matrix terlalu kecil
    untuk model-based yang butuh data banyak supaya meaningful
  - Memory-based lebih interpretable, mudah dijelaskan ke dosen
  - Model-based baru worth it kalau > 1000 users/items
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import COMPONENT_COLS


def compute_cf_similarity(interaction_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Hitung cosine similarity antar mesin berdasarkan interaction matrix
    (frekuensi penggantian komponen).

    Berbeda dari CBF similarity — ini berbasis HISTORI PENGGANTIAN,
    bukan profil sensor. Dua mesin bisa sensor-nya berbeda tapi punya
    pola ganti komponen yang sama.

    Returns:
        DataFrame (machineID × machineID)
    """
    sim_array = cosine_similarity(interaction_matrix.values)
    sim_df = pd.DataFrame(
        sim_array,
        index=interaction_matrix.index,
        columns=interaction_matrix.index
    )
    return sim_df


def cf_recommend(
    machine_id: int,
    interaction_matrix: pd.DataFrame,
    cf_sim_df: pd.DataFrame = None,
    top_n_neighbors: int = 5,
) -> pd.Series:
    """
    Rekomendasi komponen via user-user collaborative filtering.

    Algoritma:
      1. Cari top-N mesin dengan histori maintenance paling mirip
      2. Hitung weighted average frekuensi penggantian mereka
      3. Normalisasi → CF score per komponen

    Args:
        machine_id:       ID mesin target
        interaction_matrix: DataFrame mesin × komponen (frekuensi)
        cf_sim_df:        similarity matrix; dihitung otomatis jika None
        top_n_neighbors:  jumlah neighbors yang dipakai

    Returns:
        Series index=comp_name, values=cf_score [0,1]
    """
    if machine_id not in interaction_matrix.index:
        raise ValueError(f"machineID {machine_id} tidak ada di interaction matrix.")

    if cf_sim_df is None:
        cf_sim_df = compute_cf_similarity(interaction_matrix)

    # Ambil neighbors (exclude self)
    sim_row = cf_sim_df.loc[machine_id].drop(machine_id, errors="ignore")
    neighbors = sim_row.sort_values(ascending=False).head(top_n_neighbors)

    scores = pd.Series(0.0, index=COMPONENT_COLS)
    total_weight = neighbors.sum()

    if total_weight == 0:
        # Tidak ada mesin serupa → fallback ke frekuensi global
        global_freq = interaction_matrix[COMPONENT_COLS].mean()
        max_freq = global_freq.max()
        return (global_freq / max_freq) if max_freq > 0 else scores

    for neighbor_id, sim_score in neighbors.items():
        if neighbor_id not in interaction_matrix.index:
            continue
        freq = interaction_matrix.loc[neighbor_id, COMPONENT_COLS].astype(float)
        scores += freq * sim_score

    # Normalisasi ke [0, 1]
    max_score = scores.max()
    if max_score > 0:
        scores = scores / max_score

    return scores


def get_cf_explanation(
    machine_id: int,
    interaction_matrix: pd.DataFrame,
    cf_sim_df: pd.DataFrame,
    top_n_neighbors: int = 3,
) -> list[dict]:
    """
    Kembalikan penjelasan mengapa komponen direkomendasikan
    (untuk ditampilkan di dashboard).

    Returns:
        List of dict: [{"neighbor": id, "similarity": float, "top_comp": str}]
    """
    sim_row = cf_sim_df.loc[machine_id].drop(machine_id, errors="ignore")
    neighbors = sim_row.sort_values(ascending=False).head(top_n_neighbors)

    explanations = []
    for neighbor_id, sim_score in neighbors.items():
        if neighbor_id not in interaction_matrix.index:
            continue
        freq = interaction_matrix.loc[neighbor_id, COMPONENT_COLS]
        top_comp = freq.idxmax() if freq.max() > 0 else "none"
        explanations.append({
            "neighbor_machine": int(neighbor_id),
            "similarity"      : round(float(sim_score), 3),
            "most_replaced"   : top_comp,
        })
    return explanations
