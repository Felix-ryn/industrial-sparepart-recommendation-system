"""
run_preprocessing.py
─────────────────────
Entry point untuk menjalankan seluruh pipeline preprocessing.

Jalankan SEKALI sebelum membuka dashboard:
  python scripts/run_preprocessing.py

Akan menghasilkan:
  data/processed/merged_features.parquet
  data/processed/sensor_normalized.parquet
  data/processed/interaction_matrix.parquet
  models/similarity_matrix.npy
  models/similarity_matrix.index.npy
"""

import sys
import time
from pathlib import Path

# Tambahkan root project ke path supaya import src.* bisa jalan
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import run_full_pipeline
from src.cbf import build_machine_profiles, compute_similarity_matrix, save_similarity_matrix
from src.cf  import compute_cf_similarity
from config  import PROC_FILES, MODEL_FILES

import numpy as np
import pandas as pd
import json


def main():
    start = time.time()

    print("\n" + "█" * 55)
    print("  PdM RECOMMENDATION — PREPROCESSING PIPELINE")
    print("█" * 55 + "\n")

    # ── 1. Pipeline preprocessing ─────────────────────────────────────────────
    normalized_df, interaction_matrix, norm_stats = run_full_pipeline()

    # ── 2. Build & simpan CBF similarity matrix ───────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 6/7 — CBF Similarity Matrix")
    print("=" * 55)
    profiles   = build_machine_profiles(normalized_df)
    sim_df_cbf = compute_similarity_matrix(profiles)
    save_similarity_matrix(sim_df_cbf)
    print(f"  → CBF Similarity matrix: {sim_df_cbf.shape}")

    # ── 3. Build & simpan CF similarity matrix ────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 7/7 — CF Similarity Matrix")
    print("=" * 55)
    cf_sim_df = compute_cf_similarity(interaction_matrix)
    cf_path   = MODEL_FILES["sim_matrix"].parent / "cf_similarity_matrix.npy"
    cf_idx_path = cf_path.with_suffix(".index.npy")
    np.save(cf_path, cf_sim_df.values)
    np.save(cf_idx_path, np.array(cf_sim_df.index))
    print(f"  → CF Similarity matrix disimpan: {cf_path}")

    # ── 4. Simpan norm_stats untuk inverse transform di dashboard ────────────
    stats_path = MODEL_FILES["weights"].parent / "norm_stats.json"
    with open(stats_path, "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"  → Norm stats disimpan: {stats_path}")

    elapsed = time.time() - start
    print(f"\n✓ Semua selesai dalam {elapsed:.1f} detik")
    print("\nFile yang dihasilkan:")
    for key, path in PROC_FILES.items():
        size = path.stat().st_size / 1024 / 1024 if path.exists() else 0
        print(f"  {path.name:<40} {size:.1f} MB")


if __name__ == "__main__":
    main()
