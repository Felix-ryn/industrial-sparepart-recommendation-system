"""
hybrid.py — Hybrid Weighted Combination
─────────────────────────────────────────
Gabungkan skor CBF, CF, dan KB dengan weighted combination.
Bobot dimuat dari models/weights_config.json (hasil tuning evaluasi).

Formula:
  final_score(comp) = w1·cbf(comp) + w2·cf(comp) + w3·kb(comp)
  di mana w1 + w2 + w3 = 1.0
"""

import json
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import COMPONENT_COLS, DEFAULT_WEIGHTS, MODEL_FILES
from src.cbf import cbf_recommend
from src.cf  import cf_recommend
from src.kb  import kb_recommend


def load_weights() -> dict:
    """
    Load bobot dari weights_config.json.
    Fallback ke DEFAULT_WEIGHTS jika file belum ada.
    """
    wpath = MODEL_FILES["weights"]
    if wpath.exists():
        with open(wpath) as f:
            w = json.load(f)
        # Validasi: jumlah bobot harus = 1.0
        total = sum(w.get(k, 0) for k in ["cbf", "cf", "kb"])
        if abs(total - 1.0) > 0.01:
            print(f"[hybrid] WARNING: bobot tidak sum ke 1.0 ({total:.3f}), pakai default.")
            return DEFAULT_WEIGHTS
        return w
    return DEFAULT_WEIGHTS


def save_weights(weights: dict) -> None:
    """Simpan bobot hasil tuning ke disk."""
    MODEL_FILES["weights"].parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_FILES["weights"], "w") as f:
        json.dump(weights, f, indent=2)
    print(f"[hybrid] Weights disimpan: {MODEL_FILES['weights']}")


def hybrid_recommend(
    machine_id      : int,
    interaction_matrix: pd.DataFrame,
    sim_df_cbf      : pd.DataFrame,    # CBF similarity (sensor-based)
    cf_sim_df       : pd.DataFrame,    # CF similarity (maintenance-based)
    latest_sensor   : dict,
    days_since      : dict,
    top_k           : int = 3,
    weights         : dict = None,
) -> pd.DataFrame:
    """
    Hasilkan rekomendasi suku cadang dengan skor hybrid.

    Args:
        machine_id:         ID mesin target
        interaction_matrix: mesin × komponen (frekuensi penggantian)
        sim_df_cbf:         CBF similarity matrix (dari cbf.py)
        cf_sim_df:          CF similarity matrix (dari cf.py)
        latest_sensor:      dict nilai sensor terkini mesin
        days_since:         dict usia komponen (hari sejak penggantian)
        top_k:              jumlah rekomendasi teratas yang dikembalikan
        weights:            override bobot; None = load dari file/default

    Returns:
        DataFrame kolom: [comp, cbf_score, cf_score, kb_score, final_score, rank]
        Sorted by final_score descending
    """
    if weights is None:
        weights = load_weights()

    w_cbf = weights.get("cbf", DEFAULT_WEIGHTS["cbf"])
    w_cf  = weights.get("cf",  DEFAULT_WEIGHTS["cf"])
    w_kb  = weights.get("kb",  DEFAULT_WEIGHTS["kb"])

    # Hitung skor masing-masing method
    cbf_scores = cbf_recommend(machine_id, interaction_matrix, sim_df_cbf)
    cf_scores  = cf_recommend(machine_id, interaction_matrix, cf_sim_df)
    kb_scores  = kb_recommend(machine_id, latest_sensor, days_since)

    # Gabungkan
    result = pd.DataFrame({
        "comp"       : COMPONENT_COLS,
        "cbf_score"  : [cbf_scores[c] for c in COMPONENT_COLS],
        "cf_score"   : [cf_scores[c]  for c in COMPONENT_COLS],
        "kb_score"   : [kb_scores[c]  for c in COMPONENT_COLS],
    })

    result["final_score"] = (
        w_cbf * result["cbf_score"] +
        w_cf  * result["cf_score"]  +
        w_kb  * result["kb_score"]
    )

    result = result.sort_values("final_score", ascending=False).reset_index(drop=True)
    result["rank"] = result.index + 1

    return result.head(top_k)


def tune_weights(
    all_machine_ids : list,
    interaction_matrix: pd.DataFrame,
    sim_df_cbf      : pd.DataFrame,
    cf_sim_df       : pd.DataFrame,
    merged_df       : pd.DataFrame,
    eval_fn,
    k               : int = 3,
) -> dict:
    """
    Grid search sederhana untuk mencari bobot optimal.
    Evaluasi menggunakan Precision@K dari evaluation.py

    CATATAN: Ini TIDAK dipanggil saat dashboard dibuka.
    Panggil manual dari notebook 03_model_experiment.ipynb
    setelah preprocessing selesai.

    Args:
        eval_fn: fungsi precision_at_k dari evaluation.py
    
    Returns:
        dict bobot terbaik {"cbf": float, "cf": float, "kb": float}
    """
    import numpy as np

    best_score  = -1
    best_weights = DEFAULT_WEIGHTS.copy()

    # Grid: langkah 0.1, semua kombinasi yang sum = 1.0
    candidates = []
    for w1 in np.arange(0.1, 0.9, 0.1):
        for w2 in np.arange(0.1, 0.9 - w1, 0.1):
            w3 = round(1.0 - w1 - w2, 2)
            if 0.05 <= w3 <= 0.85:
                candidates.append({"cbf": round(w1, 2), "cf": round(w2, 2), "kb": w3})

    print(f"[tune] Mencoba {len(candidates)} kombinasi bobot...")
    for i, w in enumerate(candidates):
        scores = []
        for mid in all_machine_ids:
            try:
                # Ambil sensor & days_since dari baris terakhir merged_df
                latest = (
                    merged_df[merged_df["machineID"] == mid]
                    .sort_values("datetime")
                    .iloc[-1]
                )
                sensor = {col: latest[col] for col in ["volt", "rotate", "pressure", "vibration"] if col in latest}
                days   = {col: latest.get(f"days_since_{col}", 9999) for col in ["comp1", "comp2", "comp3", "comp4"]}

                recs = hybrid_recommend(
                    mid, interaction_matrix, sim_df_cbf, cf_sim_df,
                    sensor, days, top_k=k, weights=w
                )
                score = eval_fn(mid, recs["comp"].tolist(), interaction_matrix, k=k)
                scores.append(score)
            except Exception:
                continue

        avg = sum(scores) / len(scores) if scores else 0
        if avg > best_score:
            best_score   = avg
            best_weights = w
            print(f"  [tune] Update: w={w} → avg Precision@{k}={avg:.4f}")

    print(f"\n[tune] Bobot terbaik: {best_weights} (Precision@{k}={best_score:.4f})")
    save_weights(best_weights)
    return best_weights
