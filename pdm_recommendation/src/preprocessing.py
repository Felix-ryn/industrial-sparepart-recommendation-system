"""
preprocessing.py
────────────────
Tugas:
  1. Load 5 CSV raw
  2. Merge berdasarkan machineID + timestamp
  3. Feature engineering (rolling stats 24h)
  4. Normalisasi sensor
  5. Build interaction matrix mesin × komponen
  6. Simpan semua ke data/processed/ sebagai Parquet

Jalankan SEKALI via: python scripts/run_preprocessing.py
Dashboard hanya membaca hasil Parquet — tidak pernah jalankan ini lagi.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    RAW_FILES, PROC_FILES, SENSOR_COLS,
    COMPONENT_COLS, ROLLING_WINDOW_HOURS
)


# ── 1. Loader ──────────────────────────────────────────────────────────────────

def load_raw_data() -> dict:
    """Load semua file CSV raw, parse datetime, sort."""
    dfs = {}
    for name, path in RAW_FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"File tidak ditemukan: {path}\n"
                f"Pastikan semua CSV ada di data/raw/"
            )
        df = pd.read_csv(path)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values(["machineID", "datetime"]).reset_index(drop=True)
        dfs[name] = df
    print(f"[load] Telemetry: {len(dfs['telemetry']):,} baris")
    print(f"[load] Errors:    {len(dfs['errors']):,} baris")
    print(f"[load] Failures:  {len(dfs['failures']):,} baris")
    print(f"[load] Maint:     {len(dfs['maint']):,} baris")
    print(f"[load] Machines:  {len(dfs['machines']):,} baris")
    return dfs


# ── 2. Merge ───────────────────────────────────────────────────────────────────

def merge_datasets(dfs: dict) -> pd.DataFrame:
    """
    Merge telemetry + machines + error count + maint history.

    Strategi merge error:
      - Hitung jumlah error per mesin per jam (resample hourly)
      - Lebih informatif daripada one-hot raw per kejadian

    Strategi merge maint:
      - Untuk tiap baris telemetry, hitung berapa hari sejak
        komponen terakhir diganti (days_since_compX)
      - Nilai 9999 = belum pernah tercatat diganti (flag kritis di KB)
    """
    telemetry = dfs["telemetry"].copy()
    machines  = dfs["machines"]
    errors    = dfs["errors"].copy()
    maint     = dfs["maint"].copy()

    # Merge metadata mesin
    merged = telemetry.merge(machines, on="machineID", how="left")

    # Error count per jam per mesin
    error_dummies = pd.get_dummies(errors["errorID"])
    errors_expanded = pd.concat([errors[["datetime", "machineID"]], error_dummies], axis=1)
    errors_hourly = (
        errors_expanded
        .groupby(["machineID", pd.Grouper(key="datetime", freq="h")])
        .sum()
        .reset_index()
    )
    merged = merged.merge(errors_hourly, on=["machineID", "datetime"], how="left")
    for col in ["error1", "error2", "error3", "error4", "error5"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0).astype(int)

    # Days since last maintenance per komponen
    for comp in COMPONENT_COLS:
        comp_maint = maint[maint["comp"] == comp][["machineID", "datetime"]].copy()
        comp_maint = comp_maint.rename(columns={"datetime": f"last_{comp}"})
        comp_maint = comp_maint.sort_values(["machineID", f"last_{comp}"])

        merged = pd.merge_asof(
            merged.sort_values("datetime"),
            comp_maint.sort_values(f"last_{comp}"),
            left_on="datetime",
            right_on=f"last_{comp}",
            by="machineID",
            direction="backward",
        )
        merged[f"days_since_{comp}"] = (
            (merged["datetime"] - merged[f"last_{comp}"]).dt.total_seconds() / 86400
        ).fillna(9999)
        merged = merged.drop(columns=[f"last_{comp}"])

    merged = merged.sort_values(["machineID", "datetime"]).reset_index(drop=True)
    print(f"[merge] Shape setelah merge: {merged.shape}")
    return merged


# ── 3. Feature engineering (rolling) ──────────────────────────────────────────

def engineer_features(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Tambahkan rolling stats (mean, std) per mesin untuk window 24h.

    CATATAN DESAIN:
      Profil mesin untuk CBF pakai rolling 24h terakhir, bukan
      rata-rata keseluruhan — supaya rekomendasi responsif terhadap
      kondisi terkini, bukan historis.
    """
    df = merged.copy()
    for col in SENSOR_COLS:
        grp = df.groupby("machineID")[col]
        df[f"{col}_roll_mean"] = grp.transform(
            lambda x: x.rolling(ROLLING_WINDOW_HOURS, min_periods=1).mean()
        )
        df[f"{col}_roll_std"] = grp.transform(
            lambda x: x.rolling(ROLLING_WINDOW_HOURS, min_periods=1).std().fillna(0)
        )
    print(f"[features] Kolom setelah feature engineering: {df.shape[1]}")
    return df


# ── 4. Normalisasi ─────────────────────────────────────────────────────────────

def normalize_sensors(df: pd.DataFrame) -> tuple:
    """
    Min-max normalisasi pada SENSOR_COLS + rolling features.
    Kembalikan (df_normalized, stats_dict) — stats dipakai saat
    inverse transform untuk display nilai asli di dashboard.

    PENTING: stats dihitung dari seluruh dataset, bukan per mesin,
    supaya cosine similarity antar mesin bermakna.
    """
    norm_cols = SENSOR_COLS + [
        f"{c}_{s}" for c in SENSOR_COLS for s in ["roll_mean", "roll_std"]
    ]
    stats = {}
    df_norm = df.copy()
    for col in norm_cols:
        if col not in df_norm.columns:
            continue
        cmin = df_norm[col].min()
        cmax = df_norm[col].max()
        stats[col] = {"min": float(cmin), "max": float(cmax)}
        df_norm[col] = (df_norm[col] - cmin) / (cmax - cmin + 1e-9)
    print(f"[normalize] Normalisasi selesai untuk {len(stats)} kolom")
    return df_norm, stats


# ── 5. Interaction matrix ──────────────────────────────────────────────────────

def build_interaction_matrix(dfs: dict) -> pd.DataFrame:
    """
    Bangun matriks mesin × komponen dari failures + maint.
    Nilai = frekuensi penggantian (bukan binary 0/1).

    Ini adalah input utama Collaborative Filtering.
    Semakin sering mesin X ganti comp2, semakin kuat sinyalnya.
    """
    failures = dfs["failures"].copy()
    maint    = dfs["maint"].copy()

    failures = failures.rename(columns={"failure": "comp"})
    events = pd.concat([
        failures[["machineID", "comp"]],
        maint[["machineID", "comp"]],
    ])

    matrix = (
        events.groupby(["machineID", "comp"])
        .size()
        .unstack(fill_value=0)
    )

    # Pastikan semua 4 komponen ada sebagai kolom
    for comp in COMPONENT_COLS:
        if comp not in matrix.columns:
            matrix[comp] = 0

    matrix = matrix[COMPONENT_COLS]
    print(f"[interaction] Matrix shape: {matrix.shape}")
    sparsity = (matrix == 0).sum().sum() / matrix.size
    print(f"[interaction] Sparsity: {sparsity:.1%} zeros")
    return matrix


# ── 6. Pipeline utama ──────────────────────────────────────────────────────────

def run_full_pipeline():
    """
    Jalankan seluruh pipeline dan simpan semua artefak.
    Dipanggil oleh scripts/run_preprocessing.py
    """
    PROC_FILES["merged"].parent.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("STEP 1/5 — Loading raw data")
    print("=" * 55)
    dfs = load_raw_data()

    print("\n" + "=" * 55)
    print("STEP 2/5 — Merging datasets")
    print("=" * 55)
    merged = merge_datasets(dfs)

    print("\n" + "=" * 55)
    print("STEP 3/5 — Feature engineering")
    print("=" * 55)
    featured = engineer_features(merged)

    print("\n" + "=" * 55)
    print("STEP 4/5 — Normalisasi sensor")
    print("=" * 55)
    normalized, norm_stats = normalize_sensors(featured)

    # Simpan dua versi: raw featured (untuk CF) dan normalized (untuk CBF)
    featured.to_parquet(PROC_FILES["merged"], index=False)
    print(f"  → Saved: {PROC_FILES['merged']}")
    normalized.to_parquet(PROC_FILES["sensor_norm"], index=False)
    print(f"  → Saved: {PROC_FILES['sensor_norm']}")

    print("\n" + "=" * 55)
    print("STEP 5/5 — Interaction matrix")
    print("=" * 55)
    interaction = build_interaction_matrix(dfs)
    interaction.to_parquet(PROC_FILES["interaction"])
    print(f"  → Saved: {PROC_FILES['interaction']}")

    print("\n✓ Pipeline selesai. Semua artefak tersimpan di data/processed/")
    return normalized, interaction, norm_stats
