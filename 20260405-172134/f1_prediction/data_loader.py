"""
data_loader.py
--------------
Loads all F1 data:
  - Historical master dataset (2024-2025 via converted_json fallback)
  - 2026 priors  ← NOW from ~/Desktop/priors/ (updated set, overrides old ones)
  - 2026 live session data (Pre-Season Testing, AUS GP, China GP practice)

Prior directory priority:
  1. ~/Desktop/priors/               ← UPDATED priors (primary)
  2. ~/Desktop/f1_cleaned_output/priors/  ← fallback for any missing file
"""

import os
import json
import glob
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE           = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.abspath(os.path.join(_HERE, "..", "f1_cleaned_output"))
DATA_2026_DIR   = os.path.abspath(os.path.join(_HERE, "..", "2026 cleaned"))
# Updated priors take priority; fall back to old priors dir for any missing file
NEW_PRIORS_DIR  = os.path.abspath(os.path.join(_HERE, "..", "priors"))
OLD_PRIORS_DIR  = os.path.join(DATA_DIR, "priors")
PRIORS_DIR      = NEW_PRIORS_DIR   # used by load_all_priors
MASTER_DIR      = os.path.join(DATA_DIR, "master")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _safe_read(path: str, **kwargs) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as exc:
        print(f"[WARNING] Could not read {path}: {exc}")
        return pd.DataFrame()


# ── Historical master data ─────────────────────────────────────────────────────
# circuit_name → circuitRef mapping for converted_json data
_RACE_NAME_TO_CIRCUIT_REF = {
    "Bahrain":         "bahrain",
    "Saudi Arabia":    "jeddah",
    "Australian":      "albert_park",
    "Japan":           "suzuka",
    "Chinese":         "shanghai",
    "Miami":           "miami",
    "Emilia Romagna":  "imola",
    "Monaco":          "monaco",
    "Canadian":        "villeneuve",
    "Spain":           "catalunya",
    "Spanish":         "catalunya",
    "Austrian":        "red_bull_ring",
    "British":         "silverstone",
    "Hungarian":       "hungaroring",
    "Belgian":         "spa",
    "Dutch":           "zandvoort",
    "Italian":         "monza",
    "Azerbaijan":      "baku",
    "Singapore":       "marina_bay",
    "United States":   "americas",
    "Mexico":          "rodriguez",
    "Mexican":         "rodriguez",
    "São Paulo":       "interlagos",
    "Brazilian":       "interlagos",
    "Las Vegas":       "las_vegas",
    "Qatar":           "losail",
    "Abu Dhabi":       "yas_marina",
}

def _race_name_to_ref(race_name: str) -> str:
    for kw, ref in _RACE_NAME_TO_CIRCUIT_REF.items():
        if kw.lower() in str(race_name).lower():
            return ref
    return "unknown"


def load_master_data(min_year: int = 2015) -> pd.DataFrame:
    """
    Try to load the full master dataset; fall back to building from
    converted_json CSVs (results + qualifying for 2024-2025).
    """
    # Primary path (may not exist if user has moved/deleted)
    primary_paths = [
        os.path.join(MASTER_DIR, "model_ready_combined_2009_2025.csv"),
        os.path.join(DATA_DIR, "master", "model_ready_combined_2009_2025.csv"),
        os.path.join(DATA_DIR, "model_ready_combined_2009_2025.csv"),
    ]
    for path in primary_paths:
        df = _safe_read(path, low_memory=False)
        if not df.empty:
            return _prepare_master(df, min_year)

    # Fallback: build from converted_json
    print("[INFO] Master dataset not found – building from converted_json CSVs.")
    return _build_master_from_json(min_year)


def _prepare_master(df: pd.DataFrame, min_year: int) -> pd.DataFrame:
    df = df[df["season"] >= min_year].copy()
    df.rename(columns={"season": "year", "constructor_name_norm": "team_name"}, inplace=True)
    if "team_name" not in df.columns and "constructor_name" in df.columns:
        df.rename(columns={"constructor_name": "team_name"}, inplace=True)
    for col in ["grid", "quali_position", "position", "positionOrder",
                "avg_finish_last5", "avg_grid_last5", "points_last5",
                "avg_temp", "rain_flag", "wind", "best_quali_ms"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["finish_pos"] = pd.to_numeric(
        df.get("positionOrder", df.get("position")), errors="coerce"
    )
    if "status_text" in df.columns:
        dnf_kw = (r"(Retired|Accident|Collision|Engine|Gearbox|Hydraulic|"
                  r"Electrical|Suspension|Mechanical|Power|Brake|Wheel|"
                  r"Overheating|Fire|Transmission|Disqualified|Did not|Safety)")
        df["dnf"] = df["status_text"].str.contains(dnf_kw, case=False, na=False).astype(int)
    else:
        df["dnf"] = 0
    return df


def _build_master_from_json(min_year: int = 2015) -> pd.DataFrame:
    """
    Build a training DataFrame from results_20XX.csv + qualifying_20XX.csv
    in the converted_json directory.
    """
    json_dir = os.path.join(DATA_DIR, "converted_json")
    if not os.path.isdir(json_dir):
        print("[WARNING] converted_json not found either – no historical training data.")
        return pd.DataFrame()

    result_dfs = []
    quali_dfs  = []
    for yr in range(min_year, 2026):
        r = _safe_read(os.path.join(json_dir, f"results_{yr}.csv"))
        q = _safe_read(os.path.join(json_dir, f"qualifying_{yr}.csv"))
        if not r.empty:
            result_dfs.append(r)
        if not q.empty:
            quali_dfs.append(q)

    if not result_dfs:
        return pd.DataFrame()

    results = pd.concat(result_dfs, ignore_index=True)
    if quali_dfs:
        quali = pd.concat(quali_dfs, ignore_index=True)
    else:
        quali = pd.DataFrame()

    # Normalise column names
    results.rename(columns={
        "season":           "year",
        "constructor_name": "team_name",
        "position_text":    "positionText",
    }, inplace=True)

    results["year"]  = pd.to_numeric(results.get("year", results.get("season")), errors="coerce")
    results["grid"]  = pd.to_numeric(results.get("grid"), errors="coerce")
    results["position"] = pd.to_numeric(results.get("position"), errors="coerce")
    results["finish_pos"] = results["position"]

    # DNF detection from status
    if "status" in results.columns:
        dnf_kw = r"(Retired|Accident|Collision|Engine|Gearbox|Hydraulic|Electrical|Mechanical|Power|Brake)"
        results["dnf"] = results["status"].str.contains(dnf_kw, case=False, na=False).astype(int)
    else:
        results["dnf"] = 0

    # Merge qualifying
    if not quali.empty:
        quali.rename(columns={"season": "year", "constructor_name": "team_name"}, inplace=True)
        quali["year"] = pd.to_numeric(quali.get("year"), errors="coerce")
        quali["quali_position"] = pd.to_numeric(quali.get("quali_position"), errors="coerce")
        quali["best_quali_ms"]  = pd.to_numeric(quali.get("best_quali_ms"),  errors="coerce")
        merge_cols = ["year", "round", "driver_name", "quali_position", "best_quali_ms"]
        merge_cols = [c for c in merge_cols if c in quali.columns]
        results = results.merge(quali[merge_cols], on=["year", "round", "driver_name"], how="left")

    # Add circuitRef for track-prior lookup
    if "race_name" in results.columns:
        results["circuitRef"] = results["race_name"].apply(_race_name_to_ref)
    elif "circuit_name" in results.columns:
        results["circuitRef"] = results["circuit_name"].apply(_race_name_to_ref)
    else:
        results["circuitRef"] = "unknown"

    # Compute rolling avg finish (proxy for avg_finish_last5)
    results = results.sort_values(["year", "round"])
    results["avg_finish_last5"] = (
        results.groupby("driver_name")["finish_pos"]
               .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )
    # Fill missing columns with defaults
    for col, val in [("avg_temp", 25.0), ("rain_flag", 0.0), ("wind", 0.0)]:
        if col not in results.columns:
            results[col] = val

    return results


# ── 2026 Prior CSV files ───────────────────────────────────────────────────────
def _load_prior(filename: str) -> pd.DataFrame:
    """Load a prior CSV, preferring the new priors dir, falling back to old."""
    for d in [NEW_PRIORS_DIR, OLD_PRIORS_DIR]:
        path = os.path.join(d, filename)
        df = _safe_read(path)
        if not df.empty:
            return df
    return pd.DataFrame()


def load_all_priors() -> dict:
    """
    Load all 2026 priors.
    New ~/Desktop/priors/ files take priority over the old f1_cleaned_output/priors/.
    """
    priors = {
        "driver_priors":      _load_prior("driver_priors_2026.csv"),
        "team_priors":        _load_prior("team_priors_2026.csv"),
        "engine_priors":      _load_prior("engine_priors_2026.csv"),
        "track_priors":       _load_prior("track_priors_2026.csv"),
        "reliability_priors": _load_prior("reliability_priors_2026.csv"),
        "upgrade_priors":     _load_prior("upgrade_priors_2026.csv"),
        "team_engine_map":    _load_prior("team_engine_map_2026.csv"),
        "driver_team_map":    _load_prior("driver_team_map_2026.csv"),
        "weather":            _safe_read(os.path.join(DATA_DIR, "weather_all_tracks_cleaned.csv")),
    }
    priors = _normalise_priors(priors)
    return priors


def _normalise_priors(priors: dict) -> dict:
    """
    Clean up name inconsistencies across all prior DataFrames:
      - "Haas F1 Team" → "Haas"
      - "Isak Hadjar"  → canonical (already correct in new priors)
      - Remove Yuki Tsunoda if still present (replaced by Isak Hadjar)
    """
    for key, df in priors.items():
        if not isinstance(df, pd.DataFrame):
            continue
        # Haas name normalisation
        if "team" in df.columns:
            df["team"] = df["team"].str.replace("Haas F1 Team", "Haas", regex=False)
        # Remove legacy Tsunoda entry if Isak Hadjar already present
        if "driver" in df.columns:
            has_isak = df["driver"].str.lower().str.contains("hadjar").any()
            if has_isak:
                df = df[~df["driver"].str.lower().str.contains("tsunoda")]
        priors[key] = df
    return priors


# ── 2026 Live Session Data ─────────────────────────────────────────────────────

# Maps 3-letter driver code → full name used in priors
DRIVER_CODE_MAP = {
    "VER": "Max Verstappen",
    "NOR": "Lando Norris",
    "PIA": "Oscar Piastri",
    "LEC": "Charles Leclerc",
    "HAM": "Lewis Hamilton",
    "RUS": "George Russell",
    "ANT": "Kimi Antonelli",
    "ALO": "Fernando Alonso",
    "STR": "Lance Stroll",
    "HUL": "Nico Hulkenberg",
    "BOR": "Gabriel Bortoleto",
    "ALB": "Alex Albon",
    "SAI": "Carlos Sainz",
    "OCO": "Esteban Ocon",
    "BEA": "Oliver Bearman",
    "LAW": "Liam Lawson",
    "LIN": "Arvid Lindblad",
    "GAS": "Pierre Gasly",
    "COL": "Franco Colapinto",
    "PER": "Sergio Perez",
    "BOT": "Valtteri Bottas",
    "HAD": "Isak Hadjar",     # replaces Tsunoda at Red Bull (new priors: "Isak")
}

# Team name normalisation (from session data → priors name)
TEAM_NAME_MAP = {
    "Haas F1 Team": "Haas",
    "Red Bull": "Red Bull Racing",
}

# Session type weights for pace signal aggregation
SESSION_WEIGHTS = {
    "aus_qualifying":  3.0,   # AUS GP Q1/Q2/Q3 – best 2026 signal
    "aus_p1":          1.5,   # AUS GP Practice 1
    "practice_1":      1.5,   # Other GP Practice 1 (likely China)
    "preseason_p1":    0.5,   # Pre-season testing (different tyres/fuel)
}


def _read_laptimes_dir(path: str, session_tag: str,
                       event: str = "", session: str = "") -> pd.DataFrame:
    """
    Scan a session directory (containing per-driver subdirs with laptimes.csv).
    Return unified DataFrame with columns:
      driver_code, driver, team, session_tag, event, session,
      time, lap, compound, s1, s2, s3, qs (if present)
    """
    rows = []
    if not os.path.isdir(path):
        return pd.DataFrame()

    for drv_code in os.listdir(path):
        drv_dir = os.path.join(path, drv_code)
        if not os.path.isdir(drv_dir):
            continue
        lt_path = os.path.join(drv_dir, "laptimes.csv")
        if not os.path.exists(lt_path):
            continue

        try:
            df = pd.read_csv(lt_path, low_memory=False)
        except Exception:
            continue

        df["time"] = pd.to_numeric(df.get("time"), errors="coerce")
        for col in ["s1", "s2", "s3"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        team_raw = str(df["team"].iloc[0]) if "team" in df.columns else "Unknown"
        team = TEAM_NAME_MAP.get(team_raw, team_raw)
        driver = DRIVER_CODE_MAP.get(drv_code, drv_code)

        df["driver_code"]  = drv_code
        df["driver"]       = driver
        df["team_raw"]     = team_raw
        df["team"]         = team
        df["session_tag"]  = session_tag
        df["event"]        = event
        df["session"]      = session

        keep = ["driver_code", "driver", "team", "session_tag", "event", "session",
                "time", "lap", "compound", "s1", "s2", "s3"]
        if "qs" in df.columns:
            keep.append("qs")

        rows.append(df[[c for c in keep if c in df.columns]])

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_2026_all_laptimes() -> pd.DataFrame:
    """
    Load all 2026 session lap times from the 2026 cleaned directory.
    Returns a unified DataFrame across all sessions with a session_tag column.
    """
    if not os.path.isdir(DATA_2026_DIR):
        print(f"[WARNING] 2026 data directory not found: {DATA_2026_DIR}")
        return pd.DataFrame()

    all_sessions = []

    # 1. Pre-Season Testing Practice 1 (Bahrain)
    test_p1 = os.path.join(DATA_2026_DIR, "Pre-Season Testing", "Practice 1")
    df = _read_laptimes_dir(test_p1, session_tag="preseason_p1",
                            event="Pre-Season Testing", session="Practice 1")
    if not df.empty:
        all_sessions.append(df)
        print(f"  [2026] Pre-Season Testing P1: {df['driver_code'].nunique()} drivers, "
              f"{len(df)} laps")

    # 2. Australian GP Practice 1
    aus_p1 = os.path.join(DATA_2026_DIR, "Australian Grand Prix", "Practice 1")
    df = _read_laptimes_dir(aus_p1, session_tag="aus_p1",
                            event="Australian Grand Prix", session="Practice 1")
    if not df.empty:
        all_sessions.append(df)
        print(f"  [2026] AUS GP Practice 1   : {df['driver_code'].nunique()} drivers, "
              f"{len(df)} laps")

    # 3. Australian GP Qualifying (highest quality signal)
    aus_q = os.path.join(DATA_2026_DIR, "Australian Grand Prix", "Qualifying")
    df = _read_laptimes_dir(aus_q, session_tag="aus_qualifying",
                            event="Australian Grand Prix", session="Qualifying")
    if not df.empty:
        all_sessions.append(df)
        print(f"  [2026] AUS GP Qualifying   : {df['driver_code'].nunique()} drivers, "
              f"{len(df)} laps")

    # 4. Top-level Practice 1 (Chinese GP practice, based on lap times)
    other_p1 = os.path.join(DATA_2026_DIR, "Practice 1")
    df = _read_laptimes_dir(other_p1, session_tag="practice_1",
                            event="Other GP", session="Practice 1")
    if not df.empty:
        all_sessions.append(df)
        print(f"  [2026] Other GP Practice 1 : {df['driver_code'].nunique()} drivers, "
              f"{len(df)} laps")

    if not all_sessions:
        return pd.DataFrame()

    combined = pd.concat(all_sessions, ignore_index=True)
    return combined


def load_2026_qualifying_results() -> pd.DataFrame:
    """
    Extract actual AUS 2026 qualifying order from session data.
    Returns DataFrame with columns: driver, team, q3_time, q2_time, q1_time, best_time, quali_pos
    """
    aus_q_path = os.path.join(DATA_2026_DIR, "Australian Grand Prix", "Qualifying")
    if not os.path.isdir(aus_q_path):
        return pd.DataFrame()

    rows = []
    for drv_code in os.listdir(aus_q_path):
        drv_dir = os.path.join(aus_q_path, drv_code)
        if not os.path.isdir(drv_dir):
            continue
        lt_path = os.path.join(drv_dir, "laptimes.csv")
        if not os.path.exists(lt_path):
            continue

        try:
            df = pd.read_csv(lt_path)
        except Exception:
            continue

        df["time"] = pd.to_numeric(df["time"], errors="coerce")
        team_raw = df["team"].iloc[0] if "team" in df.columns else "Unknown"
        team = TEAM_NAME_MAP.get(str(team_raw), str(team_raw))

        def _best(qs_val):
            if "qs" not in df.columns:
                return np.nan
            t = df[df["qs"] == qs_val]["time"].dropna()
            return t.min() if not t.empty else np.nan

        q1 = _best("Q1")
        q2 = _best("Q2")
        q3 = _best("Q3")
        best = min(x for x in [q1, q2, q3] if not np.isnan(x)) if not all(
            np.isnan(x) for x in [q1, q2, q3]
        ) else np.nan

        rows.append({
            "driver_code": drv_code,
            "driver":      DRIVER_CODE_MAP.get(drv_code, drv_code),  # "Isak Hadjar" for HAD
            "team":        team,
            "q1_time":     q1,
            "q2_time":     q2,
            "q3_time":     q3,
            "best_time":   best,
        })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    # For VER: his Q1 was 102.025s (mechanical failure) – replace with NaN
    ver_mask = result["driver_code"] == "VER"
    if ver_mask.any():
        ver_best = result.loc[ver_mask, "best_time"].values[0]
        session_median = result.loc[~ver_mask, "best_time"].dropna().median()
        if ver_best > session_median * 1.10:
            result.loc[ver_mask, "best_time"] = np.nan
            result.loc[ver_mask, "q1_time"]   = np.nan

    # Sort: Q3 runners first (by Q3 time), then Q2 only, then Q1 only
    def _sort_key(row):
        if not np.isnan(row["q3_time"]):
            return (0, row["q3_time"])
        elif not np.isnan(row["q2_time"]):
            return (1, row.get("best_time", 999))
        elif not np.isnan(row["q1_time"]):
            return (2, row.get("best_time", 999))
        else:
            return (3, 999)

    result["_sort"] = result.apply(_sort_key, axis=1)
    result = result.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)
    result["quali_pos"] = np.arange(1, len(result) + 1)

    return result
