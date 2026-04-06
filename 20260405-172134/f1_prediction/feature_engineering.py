"""
feature_engineering.py
-----------------------
Builds feature matrices for:
  1. Historical training data (from master CSV + priors)
  2. 2026 race-by-race prediction dataset

2026 update:
  Incorporates actual 2026 lap-time data (testing + practice + qualifying)
  as the PRIMARY pace signal, with priors as the supporting baseline.

Data weighting:
  AUS Qualifying  → weight 3.0  (direct pace signal from real qualifying)
  AUS Practice 1  → weight 1.5
  Other GP P1     → weight 1.5  (likely Chinese GP)
  Pre-season Test → weight 0.5
  Prior-based score → weight 1.0 (always used as anchor / fallback)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from data_loader import SESSION_WEIGHTS, DRIVER_CODE_MAP
# Canonical driver name used in all priors (new spelling)
_HAD_NAME = "Isak Hadjar"

# ── Circuit mapping ───────────────────────────────────────────────────────────
TRACK_TO_CIRCUIT_REF = {
    "Australia":     "albert_park",
    "China":         "shanghai",
    "Japan":         "suzuka",
    "Miami":         "miami",
    "Canada":        "villeneuve",
    "Monaco":        "monaco",
    "Spain":         "catalunya",
    "Austria":       "red_bull_ring",
    "Great Britain": "silverstone",
    "Belgium":       "spa",
    "Hungary":       "hungaroring",
    "Netherlands":   "zandvoort",
    "Italy":         "monza",
    "Madrid":        "madrid",
    "Azerbaijan":    "baku",
    "Singapore":     "marina_bay",
    "USA Austin":    "americas",
    "Mexico":        "rodriguez",
    "Brazil":        "interlagos",
    "Las Vegas":     "las_vegas",
    "Qatar":         "losail",
    "Abu Dhabi":     "yas_marina",
}

DEFAULT_TRACK = {
    "power_sensitivity":     0.65,
    "downforce_sensitivity": 0.65,
    "overtaking_ease":       0.50,
    "tyre_stress":           0.60,
    "brake_stress":          0.55,
    "safety_car_rate":       0.40,
    "dnf_stress":            0.50,
    "street_factor":         0.20,
    "weather_variability":   0.40,
}

# Features shared between training and 2026 prediction datasets
FEATURE_COLS = [
    "driver_strength",
    "wet_strength",
    "team_strength",
    "engine_power",
    "engine_driveability",
    "aero_quality",
    "tyre_mgmt",
    "start_skill",
    "overtake_skill",
    "pace_2026_quali",       # NEW: 2026 normalized qualifying pace
    "pace_2026_race",        # NEW: 2026 normalized race pace
    "power_sensitivity",
    "downforce_sensitivity",
    "overtaking_ease",
    "tyre_stress",
    "street_factor",
    "safety_car_rate",
    "rain_flag",
    "temp_norm",
    "pace_x_power",
    "pace_x_downforce",
    "wet_x_rain",
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _rank_norm(series: pd.Series, ascending=True) -> pd.Series:
    r = series.rank(method="average", ascending=ascending)
    n = series.notna().sum()
    return 1.0 - (r - 1) / max(n - 1, 1)


# ── 1. Process 2026 live session data into pace signals ───────────────────────
def compute_2026_pace_signals(laps_df: pd.DataFrame) -> pd.DataFrame:
    """
    From raw 2026 lap data, compute one pace signal per driver per session type.

    Returns a DataFrame with columns:
      driver, session_tag,
      best_time, top20pct_avg, pace_ratio,
      s1_ratio, s2_ratio, s3_ratio, consistency

    pace_ratio = fastest_valid_time_in_session / driver_best_time
      → 1.0 = fastest driver, <1.0 = slower

    Outlier filtering:
      - Remove laps > 107% of session median (pit laps, slow laps, yellows)
      - VER's AUS Q time of 102s is automatically excluded
    """
    if laps_df is None or laps_df.empty:
        return pd.DataFrame()

    results = []
    for session_tag in laps_df["session_tag"].unique():
        sdf = laps_df[laps_df["session_tag"] == session_tag].copy()

        # Use Q3 laps for qualifying, all laps for practice
        is_qualifying = session_tag == "aus_qualifying"
        if is_qualifying and "qs" in sdf.columns:
            # Build best per driver per Q session, then take overall best
            pass

        # Compute session-level outlier threshold
        all_valid = pd.to_numeric(sdf["time"], errors="coerce").dropna()
        if all_valid.empty:
            continue
        session_median = all_valid.median()
        threshold = session_median * 1.07

        per_driver = []
        for drv, grp in sdf.groupby("driver"):
            grp = grp.copy()
            grp["time"] = pd.to_numeric(grp["time"], errors="coerce")

            if is_qualifying and "qs" in grp.columns:
                # Best time per Q segment, then overall best
                best_times = {}
                for qs in ["Q3", "Q2", "Q1"]:
                    sub = grp[grp["qs"] == qs]["time"].dropna()
                    sub = sub[sub < threshold]
                    if not sub.empty:
                        best_times[qs] = sub.min()

                if not best_times:
                    continue

                # Use best lap across all Q segments
                best_time = min(best_times.values())
                # Race pace from Q1 laps (more representative of raw pace)
                q1_laps = grp[(grp["qs"] == "Q1")]["time"].dropna()
                q1_laps = q1_laps[q1_laps < threshold]
                race_proxy = q1_laps.nsmallest(max(1, len(q1_laps) // 3)).mean() \
                    if not q1_laps.empty else best_time * 1.01
            else:
                # Practice: filter outliers
                valid = grp["time"].dropna()
                valid = valid[valid < threshold]
                if len(valid) < 2:
                    continue
                best_time  = valid.min()
                top20      = valid.nsmallest(max(1, len(valid) // 5))
                race_proxy = top20.mean()

            # Sector ratios (for AUS qualifying specifically)
            s1_best, s2_best, s3_best = np.nan, np.nan, np.nan
            if all(c in grp.columns for c in ["s1", "s2", "s3"]):
                for sc in ["s1", "s2", "s3"]:
                    grp[sc] = pd.to_numeric(grp[sc], errors="coerce")
                best_row = grp.loc[grp["time"].dropna().idxmin()] if grp["time"].notna().any() else None
                if best_row is not None:
                    s1_best = best_row.get("s1", np.nan)
                    s2_best = best_row.get("s2", np.nan)
                    s3_best = best_row.get("s3", np.nan)

            per_driver.append({
                "driver":       drv,
                "session_tag":  session_tag,
                "best_time":    best_time,
                "race_proxy":   race_proxy,
                "s1":           s1_best,
                "s2":           s2_best,
                "s3":           s3_best,
            })

        if not per_driver:
            continue

        pdf = pd.DataFrame(per_driver)
        # Normalize: fastest in session = 1.0
        fastest_q = pdf["best_time"].min()
        fastest_r = pdf["race_proxy"].min()
        pdf["pace_ratio_quali"] = fastest_q / pdf["best_time"]
        pdf["pace_ratio_race"]  = fastest_r / pdf["race_proxy"]

        # Sector ratios relative to session best sector
        for sec in ["s1", "s2", "s3"]:
            sec_min = pdf[sec].min()
            pdf[f"{sec}_ratio"] = np.where(
                pdf[sec].notna() & (sec_min > 0), sec_min / pdf[sec], np.nan
            )

        results.append(pdf)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


def aggregate_2026_pace(pace_per_session: pd.DataFrame) -> pd.DataFrame:
    """
    Weight-average 2026 pace ratios across sessions for each driver.

    Returns DataFrame: driver, pace_2026_quali, pace_2026_race
    where 1.0 = fastest, lower = slower.
    """
    if pace_per_session.empty:
        return pd.DataFrame(columns=["driver", "pace_2026_quali", "pace_2026_race"])

    drivers = pace_per_session["driver"].unique()
    rows = []
    for driver in drivers:
        ddf = pace_per_session[pace_per_session["driver"] == driver]
        wtd_q = 0.0
        wtd_r = 0.0
        tot_w = 0.0
        for _, row in ddf.iterrows():
            w = SESSION_WEIGHTS.get(row["session_tag"], 1.0)
            if not np.isnan(row.get("pace_ratio_quali", np.nan)):
                wtd_q += w * row["pace_ratio_quali"]
                wtd_r += w * row.get("pace_ratio_race", row["pace_ratio_quali"])
                tot_w += w

        if tot_w > 0:
            rows.append({
                "driver":         driver,
                "pace_2026_quali": round(wtd_q / tot_w, 6),
                "pace_2026_race":  round(wtd_r / tot_w, 6),
            })

    return pd.DataFrame(rows)


# ── 2. Historical training features ───────────────────────────────────────────
def build_training_features(
    master: pd.DataFrame,
    track_priors: pd.DataFrame,
    pace_2026: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Build training feature table from historical data.
    Optionally includes 2026 pace signal rows for additional calibration.
    """
    df = master.copy()
    if df.empty:
        return pd.DataFrame()

    # Track prior lookup
    track_lookup = {}
    if not track_priors.empty:
        for _, row in track_priors.iterrows():
            cref = TRACK_TO_CIRCUIT_REF.get(row["circuit"])
            if cref:
                track_lookup[cref] = row.to_dict()

    def _get_track(cref):
        return track_lookup.get(cref, DEFAULT_TRACK)

    df = df.sort_values(["year", "round"]).copy()

    # Driver rolling qualifying percentile
    df["driver_strength"] = (
        df.groupby(["year", "driver_name"])["quali_position"]
          .transform(lambda s: s.expanding().mean())
    )
    df["driver_strength"] = df.groupby("year")["driver_strength"].transform(
        lambda s: _rank_norm(s, ascending=True)
    ).fillna(0.5)

    df["wet_strength"] = df["driver_strength"]
    df["team_strength"] = (
        df.groupby(["year", "team_name"])["quali_position"]
          .transform(lambda s: s.expanding().mean())
    )
    df["team_strength"] = df.groupby("year")["team_strength"].transform(
        lambda s: _rank_norm(s, ascending=True)
    ).fillna(0.5)

    df["engine_power"]        = df["team_strength"]
    df["engine_driveability"] = df["team_strength"]
    df["aero_quality"]        = df["team_strength"]
    df["tyre_mgmt"]           = df["driver_strength"]
    df["start_skill"]         = df["driver_strength"]
    df["overtake_skill"]      = df["driver_strength"]
    df["pace_2026_quali"]     = 0.5   # unknown for historical rows
    df["pace_2026_race"]      = 0.5

    for feat in DEFAULT_TRACK:
        df[feat] = df["circuitRef"].map(
            {k: v[feat] for k, v in track_lookup.items()}
        ).fillna(DEFAULT_TRACK[feat])

    df["rain_flag"] = df["rain_flag"].fillna(0).clip(0, 1)
    df["temp_norm"] = df["avg_temp"].fillna(25) / 40.0

    df["pace_x_power"]     = df["team_strength"] * df["power_sensitivity"]
    df["pace_x_downforce"] = df["team_strength"] * df["downforce_sensitivity"]
    df["wet_x_rain"]       = df["wet_strength"]  * df["rain_flag"]

    n_drivers = df.groupby(["year", "round"])["driver_name"].transform("count")
    df["quali_norm"] = 1.0 - (df["quali_position"] - 1) / (n_drivers - 1).clip(lower=1)
    df["race_norm"]  = 1.0 - (df["finish_pos"] - 1)     / (n_drivers - 1).clip(lower=1)

    df = df.dropna(subset=["quali_position", "quali_norm"] + FEATURE_COLS)
    return df


# ── 3. Build 2026 prediction feature dataset ──────────────────────────────────
def build_2026_features(
    driver_priors:     pd.DataFrame,
    team_priors:       pd.DataFrame,
    engine_priors:     pd.DataFrame,
    track_priors:      pd.DataFrame,
    reliability_priors: pd.DataFrame,
    upgrade_priors:    pd.DataFrame,
    team_engine_map:   pd.DataFrame,
    driver_team_map:   pd.DataFrame,
    weather:           pd.DataFrame,
    pace_2026:         pd.DataFrame = None,   # NEW: aggregated 2026 pace signal
) -> pd.DataFrame:
    """
    Build a full 2026 feature dataset: one row per (driver, round).
    Merges all priors + 2026 pace signal.
    """
    # ── Merge driver → team → engine ────────────────────────────────────────
    drivers = driver_priors.copy()
    if not driver_team_map.empty:
        missing_mask = ~driver_team_map["driver"].isin(drivers["driver"])
        if missing_mask.any():
            drivers = pd.concat([drivers, driver_team_map[missing_mask]], ignore_index=True)

    drivers = drivers.merge(team_priors,   on="team",   how="left", suffixes=("", "_tp"))

    # Resolve engine_supplier
    if "engine_supplier" not in drivers.columns and "engine_supplier_tp" in drivers.columns:
        drivers["engine_supplier"] = drivers["engine_supplier_tp"]
    elif "engine_supplier_tp" in drivers.columns:
        drivers["engine_supplier"] = drivers["engine_supplier"].fillna(
            drivers["engine_supplier_tp"]
        )

    drivers = drivers.merge(
        engine_priors.rename(columns={"manufacturer": "engine_supplier"}),
        on="engine_supplier", how="left", suffixes=("", "_eng")
    )
    drivers = drivers.merge(
        reliability_priors[["team", "overall_reliability", "engine_reliability",
                             "gearbox_reliability", "electrical_reliability"]],
        on="team", how="left"
    )
    drivers = drivers.merge(
        upgrade_priors[["team", "upgrade_gain_potential", "development_speed",
                         "correlation_confidence"]],
        on="team", how="left"
    )

    # ── Merge 2026 pace signal ────────────────────────────────────────────────
    if pace_2026 is not None and not pace_2026.empty:
        drivers = drivers.merge(pace_2026[["driver", "pace_2026_quali", "pace_2026_race"]],
                                on="driver", how="left")
        # Fill in drivers without 2026 data using prior-based estimate
        # (normalize prior quali_skill to same [0.93–1.0] range as pace_ratio)
        if "pace_2026_quali" in drivers.columns:
            missing_mask = drivers["pace_2026_quali"].isna()
            if missing_mask.any():
                # Scale quali_skill (0.76–0.99) to pace_ratio range (0.93–1.0)
                qs = drivers.loc[missing_mask, "quali_skill"].fillna(0.85)
                drivers.loc[missing_mask, "pace_2026_quali"] = (
                    0.93 + (qs - 0.76) / (0.99 - 0.76) * (1.0 - 0.93)
                )
                rs = drivers.loc[missing_mask, "race_skill"].fillna(0.85)
                drivers.loc[missing_mask, "pace_2026_race"] = (
                    0.93 + (rs - 0.76) / (0.99 - 0.76) * (1.0 - 0.93)
                )
    else:
        # No 2026 data: derive from priors
        qs = drivers["quali_skill"].fillna(0.85)
        drivers["pace_2026_quali"] = 0.93 + (qs - 0.76) / (0.99 - 0.76) * 0.07
        rs = drivers["race_skill"].fillna(0.85)
        drivers["pace_2026_race"] = 0.93 + (rs - 0.76) / (0.99 - 0.76) * 0.07

    # ── Weather lookup ────────────────────────────────────────────────────────
    weather_lookup = {}
    if not weather.empty:
        for _, row in weather.iterrows():
            t = str(row.get("track", "")).strip()
            weather_lookup[t] = {
                "avg_temp":  float(row.get("avg_temp", 25)),
                "rain_flag": float(row.get("rain_flag", 0)),
            }

    # ── Cross-join drivers × rounds ───────────────────────────────────────────
    rows = []
    for _, track_row in track_priors.iterrows():
        rnd     = int(track_row["round"])
        circuit = str(track_row["circuit"])
        w       = weather_lookup.get(circuit, {"avg_temp": 25.0, "rain_flag": 0.0})

        for _, drv in drivers.iterrows():
            row = {
                "round":   rnd,
                "circuit": circuit,
                "driver":  drv["driver"],
                "team":    drv["team"],
                "engine_supplier": drv.get("engine_supplier", "Unknown"),

                # Driver skill priors
                "driver_strength":  drv.get("quali_skill",     0.85),
                "wet_strength":     drv.get("wet_skill",        0.85),
                "race_skill":       drv.get("race_skill",       0.85),
                "tyre_mgmt":        drv.get("tyre_management",  0.85),
                "start_skill":      drv.get("start_skill",      0.85),
                "overtake_skill":   drv.get("overtake_skill",   0.85),
                "defense_skill":    drv.get("defense_skill",    0.85),
                "consistency":      drv.get("consistency",      0.85),

                # Team priors
                "team_strength":    drv.get("base_quali_pace",  0.85),
                "base_race_pace":   drv.get("base_race_pace",   0.85),
                "drag_efficiency":  drv.get("drag_efficiency",  0.85),
                "aero_quality":     drv.get("downforce_quality", 0.85),
                "strategy_quality": drv.get("strategy_quality", 0.82),
                "pit_stop_quality": drv.get("pit_stop_quality", 0.84),
                "upgrade_potential": drv.get("upgrade_potential", 0.85),

                # Engine priors
                "engine_power":        drv.get("peak_power",        0.90),
                "engine_driveability": drv.get("driveability",      0.90),
                "engine_recovery":     drv.get("energy_recovery",   0.90),
                "engine_deployment":   drv.get("energy_deployment", 0.90),

                # Reliability
                "overall_reliability": drv.get("overall_reliability",  0.88),
                "engine_reliability":  drv.get("engine_reliability",   0.88),
                "gearbox_reliability": drv.get("gearbox_reliability",  0.88),

                # Upgrade trajectory
                "upgrade_gain_potential": drv.get("upgrade_gain_potential", 0.85),
                "development_speed":      drv.get("development_speed",      0.85),

                # 2026 pace signals (weighted primary signal)
                "pace_2026_quali": float(drv.get("pace_2026_quali", 0.96)),
                "pace_2026_race":  float(drv.get("pace_2026_race",  0.96)),

                # Track characteristics
                "power_sensitivity":     float(track_row.get("power_sensitivity",    0.65)),
                "downforce_sensitivity": float(track_row.get("downforce_sensitivity", 0.65)),
                "overtaking_ease":       float(track_row.get("overtaking_ease",       0.50)),
                "tyre_stress":           float(track_row.get("tyre_stress",           0.60)),
                "brake_stress":          float(track_row.get("brake_stress",          0.55)),
                "safety_car_rate":       float(track_row.get("safety_car_rate",       0.40)),
                "dnf_stress":            float(track_row.get("dnf_stress",            0.50)),
                "street_factor":         float(track_row.get("street_factor",         0.20)),
                "weather_variability":   float(track_row.get("weather_variability",   0.40)),
                "laps":                  int(track_row.get("laps", 60)),

                # Weather
                "avg_temp":  w["avg_temp"],
                "rain_flag": w["rain_flag"],
                "temp_norm": w["avg_temp"] / 40.0,
            }
            row["pace_x_power"]     = row["team_strength"] * row["power_sensitivity"]
            row["pace_x_downforce"] = row["aero_quality"]  * row["downforce_sensitivity"]
            row["wet_x_rain"]       = row["wet_strength"]  * row["rain_flag"]
            rows.append(row)

    return pd.DataFrame(rows)


# ── Upgrade boost (unchanged) ─────────────────────────────────────────────────
def compute_upgrade_boost(
    round_num: int,
    upgrade_gain_potential: float,
    development_speed: float,
    total_rounds: int = 22,
) -> float:
    progress = (round_num - 1) / max(total_rounds - 1, 1)
    max_boost = upgrade_gain_potential * development_speed * 0.04
    return max_boost * progress
