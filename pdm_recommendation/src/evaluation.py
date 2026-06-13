"""
evaluation.py — Precision@K dan Recall@K
──────────────────────────────────────────
Ground truth: failures.csv — komponen yang BENAR-BENAR gagal
setelah timestamp tertentu.

Logika evaluasi:
  - Untuk tiap mesin, ambil histori failures sebagai ground truth
  - Bandingkan dengan top-K rekomendasi sistem
  - Hitung Precision@K dan Recall@K
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EVAL_K_VALUES, COMPONENT_COLS, RAW_FILES


def load_ground_truth() -> dict:
    """
    Load failures.csv dan buat dict ground truth per mesin.
    
    Returns:
        {machineID: set of comp yang pernah gagal}
    """
    failures = pd.read_csv(RAW_FILES["failures"])
    ground_truth = (
        failures.groupby("machineID")["failure"]
        .apply(set)
        .to_dict()
    )
    return ground_truth


def precision_at_k(
    machine_id     : int,
    recommended    : list,
    ground_truth   : dict,
    k              : int = 3,
) -> float:
    """
    Precision@K = |recommended ∩ relevant| / K

    Args:
        machine_id:   ID mesin yang dievaluasi
        recommended:  list komponen hasil rekomendasi (urutan = prioritas)
        ground_truth: dict {machineID: set(comp)} dari load_ground_truth()
        k:            cutoff

    Returns:
        float [0, 1]
    """
    relevant = ground_truth.get(machine_id, set())
    if not relevant:
        return 0.0

    top_k = recommended[:k]
    hits  = sum(1 for comp in top_k if comp in relevant)
    return hits / k


def recall_at_k(
    machine_id     : int,
    recommended    : list,
    ground_truth   : dict,
    k              : int = 3,
) -> float:
    """
    Recall@K = |recommended ∩ relevant| / |relevant|

    Args:
        machine_id:   ID mesin yang dievaluasi
        recommended:  list komponen hasil rekomendasi
        ground_truth: dict dari load_ground_truth()
        k:            cutoff

    Returns:
        float [0, 1]
    """
    relevant = ground_truth.get(machine_id, set())
    if not relevant:
        return 0.0

    top_k = recommended[:k]
    hits  = sum(1 for comp in top_k if comp in relevant)
    return hits / len(relevant)


def evaluate_system(
    recommendations : dict,
    ground_truth    : dict,
    k_values        : list = None,
) -> pd.DataFrame:
    """
    Evaluasi sistem secara menyeluruh untuk semua mesin dan semua nilai K.

    Args:
        recommendations: {machineID: [comp_list]} hasil hybrid_recommend
        ground_truth:    dari load_ground_truth()
        k_values:        list nilai K; default dari config

    Returns:
        DataFrame dengan kolom: [machineID, k, precision, recall, f1]
    """
    if k_values is None:
        k_values = EVAL_K_VALUES

    rows = []
    for machine_id, rec_list in recommendations.items():
        for k in k_values:
            p = precision_at_k(machine_id, rec_list, ground_truth, k)
            r = recall_at_k(machine_id, rec_list, ground_truth, k)
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            rows.append({
                "machineID" : machine_id,
                "k"         : k,
                "precision" : round(p, 4),
                "recall"    : round(r, 4),
                "f1"        : round(f1, 4),
            })

    return pd.DataFrame(rows)


def print_evaluation_summary(eval_df: pd.DataFrame) -> None:
    """Print ringkasan evaluasi per nilai K."""
    print("\n" + "=" * 50)
    print("RINGKASAN EVALUASI SISTEM")
    print("=" * 50)
    summary = (
        eval_df.groupby("k")[["precision", "recall", "f1"]]
        .mean()
        .round(4)
    )
    print(summary.to_string())
    print("=" * 50)
