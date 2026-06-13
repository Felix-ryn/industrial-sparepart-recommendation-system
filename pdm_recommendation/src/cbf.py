"""
cbf.py — Content-Based Filtering
─────────────────────────────────
Mencari mesin-mesin dengan profil sensor serupa menggunakan cosine similarity.
Output: top-N mesin paling mirip dengan mesin target.

Kenapa cosine similarity, bukan euclidean?
  - Cosine tidak sensitif terhadap magnitude, hanya arah vektor
  - Mesin besar vs kecil yang punya pola sensor sama tetap dianggap mirip
  - Lebih robust untuk fitur yang sudah dinormalisasi
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SENSOR_COLS, PROC_FILES, MODEL_FILES, COMPONENT_COLS


# Fitur yang dipakai untuk profil mesin (rolling mean lebih stabil dari raw)
CBF_FEATURE_COLS = [f"{c}_roll_mean" for c in SENSOR_COLS] + \
                   [f"{c}_roll_std"  for c in SENSOR_COLS]


def build_machine_profiles(sensor_norm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ambil snapshot profil terkini tiap mesin = baris TERAKHIR per machineID.

    Logika: baris terakhir sudah berisi rolling_mean 24h terakhir,
    jadi ini merepresentasikan kondisi terkini mesin.

    Returns:
        DataFrame dengan index=machineID, kolom=CBF_FEATURE_COLS
    """
    available_cols = [c for c in CBF_FEATURE_COLS if c in sensor_norm_df.columns]
    if not available_cols:
        raise ValueError(
            "Kolom rolling features tidak ditemukan. "
            "Pastikan run_preprocessing.py sudah dijalankan."
        )

    profiles = (
        sensor_norm_df
        .sort_values(["machineID", "datetime"])
        .groupby("machineID")[available_cols]
        .last()
    )
    return profiles


def compute_similarity_matrix(profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Hitung cosine similarity matrix antar semua mesin.

    Returns:
        DataFrame (machineID × machineID) berisi nilai similarity 0–1
    """
    sim_array = cosine_similarity(profiles.values)
    sim_df = pd.DataFrame(
        sim_array,
        index=profiles.index,
        columns=profiles.index
    )
    return sim_df


def get_similar_machines(
    machine_id: int,
    sim_df: pd.DataFrame,
    top_n: int = 5,
    exclude_self: bool = True
) -> pd.Series:
    """
    Kembalikan top-N mesin paling mirip dengan machine_id.

    Args:
        machine_id:   ID mesin target
        sim_df:       similarity matrix dari compute_similarity_matrix()
        top_n:        jumlah mesin similar yang dikembalikan
        exclude_self: hilangkan mesin itu sendiri dari hasil

    Returns:
        Series dengan index=machineID, values=similarity score, sorted desc
    """
    if machine_id not in sim_df.index:
        raise ValueError(f"machineID {machine_id} tidak ada di similarity matrix.")

    row = sim_df.loc[machine_id].copy()
    if exclude_self:
        row = row.drop(machine_id, errors="ignore")

    return row.sort_values(ascending=False).head(top_n)


def cbf_recommend(
    machine_id: int,
    interaction_matrix: pd.DataFrame,
    sim_df: pd.DataFrame,
    top_n_machines: int = 5,
) -> pd.Series:
    """
    Rekomendasi komponen berdasarkan pola penggantian mesin-mesin serupa.

    Algoritma:
      1. Cari top-N mesin paling mirip (cosine sim)
      2. Ambil frekuensi penggantian komponen mereka
      3. Bobot dengan similarity score
      4. Normalisasi ke [0, 1] sebagai CBF score

    Args:
        machine_id:        ID mesin yang mau direkomendasikan
        interaction_matrix: DataFrame (machineID × comp) frekuensi penggantian
        sim_df:            similarity matrix
        top_n_machines:    berapa mesin serupa yang dipertimbangkan

    Returns:
        Series dengan index=comp_name, values=cbf_score [0,1]
    """
    similar = get_similar_machines(machine_id, sim_df, top_n=top_n_machines)

    scores = pd.Series(0.0, index=COMPONENT_COLS)
    total_weight = similar.sum()

    if total_weight == 0:
        return scores

    for sim_machine, sim_score in similar.items():
        if sim_machine not in interaction_matrix.index:
            continue
        comp_counts = interaction_matrix.loc[sim_machine, COMPONENT_COLS].astype(float)
        scores += comp_counts * sim_score

    # Normalisasi ke [0, 1]
    max_score = scores.max()
    if max_score > 0:
        scores = scores / max_score

    return scores


def save_similarity_matrix(sim_df: pd.DataFrame) -> None:
    """Simpan similarity matrix ke models/ sebagai .npy untuk load cepat."""
    MODEL_FILES["sim_matrix"].parent.mkdir(parents=True, exist_ok=True)
    np.save(MODEL_FILES["sim_matrix"], sim_df.values)
    # Simpan index (machineID list) sebagai companion file
    idx_path = MODEL_FILES["sim_matrix"].with_suffix(".index.npy")
    np.save(idx_path, np.array(sim_df.index))
    print(f"[cbf] Similarity matrix disimpan: {MODEL_FILES['sim_matrix']}")


def load_similarity_matrix() -> pd.DataFrame:
    """Load similarity matrix dari disk."""
    arr = np.load(MODEL_FILES["sim_matrix"])
    idx_path = MODEL_FILES["sim_matrix"].with_suffix(".index.npy")
    idx = np.load(idx_path)
    return pd.DataFrame(arr, index=idx, columns=idx)
