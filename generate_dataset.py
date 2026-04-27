"""
==============================================================================
 generate_dataset.py — Synthetic Energy Consumption Dataset Generator
==============================================================================
 PURPOSE  (Viva Explanation)
 ─────────────────────────────
 Real smart-meter data is hard to obtain, so we *simulate* it using
 domain knowledge about how households consume electricity.

 SIMULATION LAYERS
 ─────────────────
   Layer 1 – Base load      : Always-on devices (fridge, Wi-Fi router, standby).
   Layer 2 – Time-of-day    : Mornings & evenings see peaks (cooking, lights).
   Layer 3 – Seasonal trend : Winter heating / summer cooling raises usage.
   Layer 4 – Weekend effect : People stay home → higher daytime consumption.
   Layer 5 – Random noise   : Real meters are never perfectly smooth.

 APPLIANCE-LEVEL BREAKDOWN  (Advanced Feature ✨)
 ─────────────────────────────────────────────────
 The total energy_kwh is decomposed into six appliance categories
 so the dashboard can display a per-appliance breakdown:
   • AC / Heating      — temperature-driven, seasonal
   • Lighting          — active during dark hours, higher in winter
   • Kitchen           — peaks at breakfast, lunch, dinner
   • Fan               — moderate in warm months
   • Entertainment     — TV, gaming; evenings & weekends
   • Other / Standby   — washing machine, iron, miscellaneous

 ANOMALY INJECTION  (Advanced Feature ✨)
 ────────────────────────────────────────
 ~1.5 % of records are artificially spiked (2×–4× normal) to
 simulate real-world anomalies like parties, faulty appliances,
 or extreme weather events.  This lets us demonstrate anomaly
 detection in the ML pipeline.

 OUTPUT COLUMNS (17 total)
 ─────────────────────────
   datetime, hour, day_of_week, month, is_weekend,
   temperature_c, humidity_pct,
   ac_kwh, lighting_kwh, kitchen_kwh, fan_kwh,
   entertainment_kwh, other_kwh,
   energy_kwh   (= sum of all appliance columns)
==============================================================================
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ── Reproducibility — same seed = same dataset every time ────────────────
np.random.seed(42)

# ── Configuration ────────────────────────────────────────────────────────
START_DATE  = datetime(2025, 1, 1)
DAYS        = 365
HOURS       = 24
TOTAL_ROWS  = DAYS * HOURS          # 8 760 hourly readings


def generate_dataset(filepath: str = "dataset.csv") -> pd.DataFrame:
    """
    Generate and save a synthetic energy consumption dataset with
    appliance-level breakdowns and occasional anomaly spikes.

    Parameters
    ----------
    filepath : str
        Where to save the CSV file.

    Returns
    -------
    pd.DataFrame
        The generated dataset.
    """

    # ─── 1. Datetime index ───────────────────────────────────────────────
    timestamps = [START_DATE + timedelta(hours=i) for i in range(TOTAL_ROWS)]
    df = pd.DataFrame({"datetime": timestamps})

    # ─── 2. Time features ────────────────────────────────────────────────
    df["hour"]        = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek           # 0=Mon … 6=Sun
    df["month"]       = df["datetime"].dt.month
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)  # 1 if Sat / Sun

    day_of_year = df["datetime"].dt.dayofyear

    # ─── 3. Outdoor temperature (°C) ────────────────────────────────────
    #   Sinusoidal yearly cycle + daily cycle + Gaussian noise
    yearly_temp = 15 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    daily_temp  = 3  * np.sin(2 * np.pi * (df["hour"] - 6) / 24)
    noise_temp  = np.random.normal(0, 2, TOTAL_ROWS)
    df["temperature_c"] = np.round(yearly_temp + daily_temp + noise_temp, 1)

    # ─── 4. Humidity (%) ─────────────────────────────────────────────────
    df["humidity_pct"] = np.clip(
        60 - 0.5 * df["temperature_c"] + np.random.normal(0, 8, TOTAL_ROWS),
        20, 95
    ).round(1)

    # ─── 5. Multipliers ─────────────────────────────────────────────────
    seasonal = 1.0 + 0.35 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
    weekend_boost = np.where(
        (df["is_weekend"] == 1) & (df["hour"].between(8, 20)), 1.12, 1.0
    )

    # ═════════════════════════════════════════════════════════════════════
    #  APPLIANCE-LEVEL SIMULATION
    # ═════════════════════════════════════════════════════════════════════

    # ── AC / Heating ─────────────────────────────────────────────────────
    ac_base   = 0.04 * np.abs(df["temperature_c"] - 22)
    ac_active = np.where(df["hour"].between(7, 23), 1.0, 0.3)
    df["ac_kwh"] = np.round(np.clip(
        ac_base * ac_active * seasonal + np.random.normal(0, 0.03, TOTAL_ROWS),
        0, None), 3)

    # ── Lighting ─────────────────────────────────────────────────────────
    light_pattern = np.array([
        0.03, 0.02, 0.02, 0.02, 0.02, 0.03,   # 00-05  (minimal)
        0.10, 0.12, 0.08, 0.05, 0.04, 0.04,   # 06-11  (morning)
        0.04, 0.04, 0.04, 0.05, 0.08, 0.15,   # 12-17  (afternoon)
        0.22, 0.25, 0.23, 0.18, 0.10, 0.05    # 18-23  (evening peak)
    ])
    winter_light = 1.0 + 0.3 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
    df["lighting_kwh"] = np.round(np.clip(
        np.array([light_pattern[h] for h in df["hour"]]) * winter_light
        + np.random.normal(0, 0.015, TOTAL_ROWS), 0, None), 3)

    # ── Kitchen ──────────────────────────────────────────────────────────
    kitchen_pattern = np.array([
        0.02, 0.01, 0.01, 0.01, 0.01, 0.02,   # 00-05
        0.05, 0.25, 0.20, 0.08, 0.05, 0.06,   # 06-11  (breakfast)
        0.22, 0.18, 0.08, 0.05, 0.06, 0.15,   # 12-17  (lunch)
        0.30, 0.35, 0.25, 0.12, 0.05, 0.03    # 18-23  (dinner peak)
    ])
    df["kitchen_kwh"] = np.round(np.clip(
        np.array([kitchen_pattern[h] for h in df["hour"]]) * weekend_boost
        + np.random.normal(0, 0.02, TOTAL_ROWS), 0, None), 3)

    # ── Fan ──────────────────────────────────────────────────────────────
    fan_temp  = np.clip((df["temperature_c"] - 20) / 15, 0, 1)
    fan_hours = np.where(df["hour"].between(8, 23), 1.0, 0.3)
    df["fan_kwh"] = np.round(np.clip(
        0.15 * fan_temp * fan_hours
        + np.random.normal(0, 0.01, TOTAL_ROWS), 0, None), 3)

    # ── Entertainment ────────────────────────────────────────────────────
    ent_pattern = np.array([
        0.03, 0.02, 0.01, 0.01, 0.01, 0.01,   # 00-05
        0.02, 0.04, 0.05, 0.06, 0.08, 0.08,   # 06-11
        0.10, 0.10, 0.08, 0.08, 0.10, 0.15,   # 12-17
        0.20, 0.25, 0.28, 0.22, 0.12, 0.06    # 18-23  (prime time)
    ])
    df["entertainment_kwh"] = np.round(np.clip(
        np.array([ent_pattern[h] for h in df["hour"]]) * weekend_boost
        + np.random.normal(0, 0.015, TOTAL_ROWS), 0, None), 3)

    # ── Other / Standby ──────────────────────────────────────────────────
    df["other_kwh"] = np.round(np.clip(
        0.08 + np.random.normal(0, 0.025, TOTAL_ROWS), 0.02, None), 3)

    # ═════════════════════════════════════════════════════════════════════
    #  TOTAL ENERGY = sum of all appliances
    # ═════════════════════════════════════════════════════════════════════
    appliance_cols = [
        "ac_kwh", "lighting_kwh", "kitchen_kwh",
        "fan_kwh", "entertainment_kwh", "other_kwh"
    ]
    df["energy_kwh"] = df[appliance_cols].sum(axis=1).round(3)

    # ═════════════════════════════════════════════════════════════════════
    #  INJECT REALISTIC ANOMALIES  (~1.5 % of data)
    # ═════════════════════════════════════════════════════════════════════
    #  Simulates: a house party, faulty appliance, heatwave AC overload.
    n_anomalies = int(0.015 * TOTAL_ROWS)           # ~131 anomaly hours
    anomaly_idx = np.random.choice(TOTAL_ROWS, n_anomalies, replace=False)
    spike_factor = np.random.uniform(2.0, 4.0, n_anomalies)
    df.loc[anomaly_idx, "energy_kwh"] = (
        df.loc[anomaly_idx, "energy_kwh"] * spike_factor
    ).round(3)

    # ─── 6. Save to CSV ─────────────────────────────────────────────────
    df.to_csv(filepath, index=False)
    print(f"✅  Dataset saved → {filepath}  ({len(df)} records, "
          f"{n_anomalies} anomalies injected)")
    return df


# ── CLI entry point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    generate_dataset()
