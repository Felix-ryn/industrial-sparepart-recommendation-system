"""
kb.py — Knowledge-Based Recommendation
────────────────────────────────────────
Rule-based layer berbasis threshold sensor dan usia komponen.
Berfungsi sebagai "safety net" — memastikan kondisi kritis tidak
terlewatkan oleh algoritma statistik.

PENTING: threshold di config.py HARUS dikalibrasi setelah EDA.
Jangan deploy dengan nilai placeholder sebelum kamu lihat distribusi
aktual sensor di notebook 01_eda.ipynb.
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SENSOR_COLS, COMPONENT_COLS, KB_THRESHOLDS


# Batas usia komponen (hari) sebelum dianggap berisiko tinggi
# Nilai ini juga HARUS dikalibrasi dari data failures historis
COMP_AGE_THRESHOLDS = {
    "comp1": 120,  # hari
    "comp2": 90,
    "comp3": 150,
    "comp4": 100,
}


def kb_recommend(
    machine_id: int,
    latest_sensor: dict,
    days_since: dict,
) -> pd.Series:
    """
    Hitung KB score per komponen berdasarkan:
      A) Nilai sensor yang melewati threshold (abnormal)
      B) Usia komponen yang melebihi batas (overdue maintenance)

    Args:
        machine_id:    ID mesin (untuk logging/debug)
        latest_sensor: dict berisi nilai sensor terkini
                       {"volt": 185.2, "rotate": 430.1, ...}
        days_since:    dict berisi usia komponen sejak penggantian terakhir
                       {"comp1": 45.0, "comp2": 130.5, ...}

    Returns:
        Series index=comp_name, values=kb_score [0,1]
        Score 1.0 = kondisi kritis, 0.0 = normal
    """
    scores = {comp: 0.0 for comp in COMPONENT_COLS}

    # ── Rule A: Sensor threshold ───────────────────────────────────────────────
    for sensor, rules in KB_THRESHOLDS.items():
        if sensor not in latest_sensor:
            continue
        val = latest_sensor[sensor]
        triggered = False

        if "min" in rules and val < rules["min"]:
            triggered = True
        if "max" in rules and val > rules["max"]:
            triggered = True

        if triggered:
            for comp in rules.get("comp", []):
                if comp in scores:
                    scores[comp] = max(scores[comp], 0.8)

    # ── Rule B: Komponen overdue ───────────────────────────────────────────────
    for comp, threshold_days in COMP_AGE_THRESHOLDS.items():
        age = days_since.get(comp, 0)
        if age >= 9999:
            # Belum pernah tercatat diganti — prioritas sangat tinggi
            scores[comp] = 1.0
        elif age > threshold_days:
            # Overdue — score proporsional dengan seberapa lama lewat threshold
            overdue_ratio = min((age - threshold_days) / threshold_days, 1.0)
            scores[comp] = max(scores[comp], 0.6 + 0.4 * overdue_ratio)

    return pd.Series(scores)


def get_kb_alerts(
    machine_id: int,
    latest_sensor: dict,
    days_since: dict,
) -> list[dict]:
    """
    Kembalikan daftar alert yang aktif (untuk ditampilkan di dashboard).

    Returns:
        List of dict: [{"type": "sensor"|"age", "detail": str, "severity": str}]
    """
    alerts = []

    for sensor, rules in KB_THRESHOLDS.items():
        if sensor not in latest_sensor:
            continue
        val = latest_sensor[sensor]

        if "min" in rules and val < rules["min"]:
            alerts.append({
                "type"    : "sensor",
                "sensor"  : sensor,
                "value"   : round(val, 2),
                "detail"  : f"{sensor} = {val:.2f} (batas bawah: {rules['min']})",
                "severity": "high",
                "comp"    : rules.get("comp", []),
            })
        if "max" in rules and val > rules["max"]:
            alerts.append({
                "type"    : "sensor",
                "sensor"  : sensor,
                "value"   : round(val, 2),
                "detail"  : f"{sensor} = {val:.2f} (batas atas: {rules['max']})",
                "severity": "high",
                "comp"    : rules.get("comp", []),
            })

    for comp, threshold_days in COMP_AGE_THRESHOLDS.items():
        age = days_since.get(comp, 0)
        if age >= 9999:
            alerts.append({
                "type"    : "age",
                "comp"    : comp,
                "detail"  : f"{comp} belum pernah tercatat diganti",
                "severity": "critical",
            })
        elif age > threshold_days:
            alerts.append({
                "type"    : "age",
                "comp"    : comp,
                "detail"  : f"{comp} sudah {age:.0f} hari (threshold: {threshold_days} hari)",
                "severity": "medium",
            })

    return alerts
