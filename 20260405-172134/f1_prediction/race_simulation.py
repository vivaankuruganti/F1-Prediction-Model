"""
race_simulation.py
------------------
Monte Carlo race simulator for the 2026 F1 season.

v3 update (new priors):
  Updated priors reflect a major shift:
    - Mercedes = dominant car (pace 1.0, engine 1.0)
    - Ferrari   = strong + very reliable (pace 0.98, reliability 0.98)
    - McLaren   = competitive midfront (pace 0.90)
    - Red Bull  = mid-pack, unreliable engine (pace 0.85, engine rel 0.60)
    - Aston Martin = backmarker (Honda engine 0.4 power, team pace 0.64)
    - Haas, Alpine = improved vs 2025

  Weight design (per instruction: priors strongly weighted):
    pace_2026_race     : 0.35  ← 2026 live signal
    team + engine prior: 0.38  ← NEW: increased prior influence
    driver components  : 0.22  ← driver skill, tyre mgmt, consistency
    aero contribution  : 0.05
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from collections import defaultdict

from feature_engineering import compute_upgrade_boost

F1_POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
             6: 8,  7: 6,  8: 4,  9: 2,  10: 1}
FASTEST_LAP_POINT = 1


# ── DNF probability model ─────────────────────────────────────────────────────
def _dnf_probability(overall_rel: float, engine_rel: float, dnf_stress: float) -> float:
    combined_rel = overall_rel * 0.6 + engine_rel * 0.4
    base_dnf = (1.0 - combined_rel) * 0.8
    stress_mult = 1.0 + dnf_stress * 0.5
    return float(np.clip(base_dnf * stress_mult, 0.01, 0.40))


# ── Base race pace score ──────────────────────────────────────────────────────
def _base_pace(row: pd.Series, round_num: int) -> float:
    """
    Compute driver base race pace for a round.

    2026 update: pace_2026_race is the anchor (weight 0.45).
    Still combines driver skill, team pace, engine, aero.
    """
    # 2026 live pace signal (normalized ratio: fastest driver = 1.0)
    pace_2026 = float(row.get("pace_2026_race", 0.96))

    # Driver components
    drv_score = (
        0.55 * row.get("race_skill",  0.85) +
        0.25 * row.get("tyre_mgmt",   0.85) +
        0.20 * row.get("consistency", 0.85)
    )

    # Team base pace (prior)
    team_score = (
        0.65 * row.get("base_race_pace",   0.85) +
        0.35 * row.get("strategy_quality", 0.82)
    )

    # Engine contribution (track-type weighted)
    ps = row.get("power_sensitivity", 0.65)
    engine_score = (
        row.get("engine_power",        0.90) * ps +
        row.get("engine_driveability", 0.90) * (1 - ps)
    )

    # Aero contribution
    ds = row.get("downforce_sensitivity", 0.65)
    aero_score = (
        row.get("aero_quality",    0.85) * ds +
        row.get("drag_efficiency", 0.85) * (1 - ds)
    )

    # Tyre management
    tyre_score = row.get("tyre_mgmt", 0.85) * (1 - row.get("tyre_stress", 0.6) * 0.5)

    pace = (
        0.35 * pace_2026   +   # 2026 live data
        0.22 * drv_score   +   # driver skill
        0.25 * team_score  +   # team pace prior (↑ increased per new priors)
        0.13 * engine_score +  # engine × track (↑ more weight, engine diffs are bigger)
        0.05 * aero_score
    )

    # Upgrade boost
    boost = compute_upgrade_boost(
        round_num=round_num,
        upgrade_gain_potential=row.get("upgrade_gain_potential", 0.85),
        development_speed=row.get("development_speed", 0.85),
    )
    return pace + boost


# ── Single race simulation ────────────────────────────────────────────────────
def _simulate_one_race(
    race_df: pd.DataFrame,
    round_num: int,
    rng: np.random.Generator,
) -> list:
    n   = len(race_df)
    rain = float(race_df["rain_flag"].iloc[0])
    wv   = float(race_df["weather_variability"].iloc[0])
    oe   = float(race_df["overtaking_ease"].iloc[0])
    sc_rate = float(race_df["safety_car_rate"].iloc[0])
    ts   = float(race_df["tyre_stress"].iloc[0])

    # ── DNF ──────────────────────────────────────────────────────────────────
    dnf_flags = np.array([
        rng.random() < _dnf_probability(
            float(row.get("overall_reliability", 0.88)),
            float(row.get("engine_reliability",  0.88)),
            float(row.get("dnf_stress",          0.50)),
        )
        for _, row in race_df.iterrows()
    ])

    # ── Base pace ─────────────────────────────────────────────────────────────
    base_paces = np.array([
        _base_pace(row, round_num) for _, row in race_df.iterrows()
    ])

    # ── Grid / track-position advantage ──────────────────────────────────────
    grid_scale = (1.0 - oe) * 0.06
    grid_bonus = grid_scale * (1 - (race_df["quali_pos"].values - 1) / max(n - 1, 1))

    # ── Start performance ─────────────────────────────────────────────────────
    start_skills = race_df["start_skill"].fillna(0.85).values
    start_bonus  = (start_skills - 0.85) * 0.04 + rng.normal(0, 0.015, size=n)

    # ── Wet weather ───────────────────────────────────────────────────────────
    wet_bonus = np.zeros(n)
    if rain > 0.3:
        wet_skills = race_df["wet_strength"].fillna(0.85).values
        wet_bonus  = (wet_skills - 0.85) * 0.08 * rain

    # ── Tyre degradation ──────────────────────────────────────────────────────
    tyre_mgmt  = race_df["tyre_mgmt"].fillna(0.85).values
    tyre_bonus = (tyre_mgmt - 0.85) * ts * 0.04

    # ── Safety car ────────────────────────────────────────────────────────────
    sc_chaos = rng.normal(0, 0.02 * int(rng.random() < sc_rate), size=n)

    # ── Random race noise ─────────────────────────────────────────────────────
    noise_sigma = 0.018 * (1 + rain * wv * 0.5)
    noise = rng.normal(0, noise_sigma, size=n)

    final_paces = base_paces + grid_bonus + start_bonus + wet_bonus + tyre_bonus + sc_chaos + noise

    # Sort: non-DNF by pace (desc), then DNFs in random order
    driver_names = race_df["driver"].values
    teams        = race_df["team"].values

    classified_idx = np.where(~dnf_flags)[0]
    dnf_idx        = np.where(dnf_flags)[0]
    classified_ord = classified_idx[np.argsort(final_paces[classified_idx])[::-1]]
    dnf_ord        = dnf_idx[rng.permutation(len(dnf_idx))] if len(dnf_idx) > 0 else dnf_idx
    final_order    = np.concatenate([classified_ord, dnf_ord]).astype(int)

    results = []
    for pos, idx in enumerate(final_order, start=1):
        results.append({
            "driver":     driver_names[idx],
            "team":       teams[idx],
            "finish_pos": pos,
            "dnf":        bool(dnf_flags[idx]),
            "pace_score": float(final_paces[idx]),
        })

    # Fastest lap
    non_dnf = [r for r in results if not r["dnf"]]
    if non_dnf:
        top10 = [r for r in non_dnf if r["finish_pos"] <= 10]
        fl_driver = max(top10 or non_dnf, key=lambda r: r["pace_score"])["driver"]
        for r in results:
            r["fastest_lap"] = (r["driver"] == fl_driver)
    else:
        for r in results:
            r["fastest_lap"] = False

    return results


# ── Monte Carlo race aggregation ──────────────────────────────────────────────
def simulate_race(
    features_2026: pd.DataFrame,
    round_num: int,
    n_simulations: int = 5000,
    seed: int = None,
    qualifying_grid: pd.DataFrame = None,
) -> dict:
    race_df = features_2026[features_2026["round"] == round_num].copy()
    if race_df.empty:
        print(f"[RaceSim] No data for round {round_num}.")
        return {}

    circuit = race_df["circuit"].iloc[0]

    if qualifying_grid is not None:
        qg = qualifying_grid[qualifying_grid["round"] == round_num][["driver", "quali_pos"]]
        race_df = race_df.merge(qg, on="driver", how="left")
        race_df["quali_pos"] = race_df["quali_pos"].fillna(len(race_df))
    else:
        race_df = race_df.sort_values("pace_2026_race", ascending=False).reset_index(drop=True)
        race_df["quali_pos"] = np.arange(1, len(race_df) + 1)

    race_df = race_df.reset_index(drop=True)
    n_drivers = len(race_df)

    rng = np.random.default_rng(seed)

    position_counts = defaultdict(lambda: defaultdict(int))
    dnf_counts      = defaultdict(int)
    points_counts   = defaultdict(int)
    fastest_counts  = defaultdict(int)
    pos_sum         = defaultdict(float)
    pace_sum        = defaultdict(float)

    for _ in range(n_simulations):
        sim_results = _simulate_one_race(race_df, round_num, rng)
        for r in sim_results:
            d = r["driver"]
            p = r["finish_pos"]
            position_counts[d][p] += 1
            pos_sum[d]  += p
            pace_sum[d] += r["pace_score"]
            if r["dnf"]:
                dnf_counts[d] += 1
            if F1_POINTS.get(p, 0) > 0 and not r["dnf"]:
                points_counts[d] += 1
            if r.get("fastest_lap") and not r["dnf"] and p <= 10:
                fastest_counts[d] += 1

    drivers = race_df["driver"].tolist()
    teams   = race_df.set_index("driver")["team"].to_dict()

    rows = []
    for d in drivers:
        rows.append({
            "driver":       d,
            "team":         teams.get(d, "Unknown"),
            "win_prob":     round(position_counts[d].get(1, 0) / n_simulations, 4),
            "podium_prob":  round(sum(position_counts[d].get(p, 0)
                                      for p in [1, 2, 3]) / n_simulations, 4),
            "top5_prob":    round(sum(position_counts[d].get(p, 0)
                                      for p in range(1, 6)) / n_simulations, 4),
            "points_prob":  round(points_counts[d] / n_simulations, 4),
            "dnf_prob":     round(dnf_counts[d] / n_simulations, 4),
            "fl_prob":      round(fastest_counts[d] / n_simulations, 4),
            "expected_pos": round(pos_sum[d] / n_simulations, 2),
            "avg_pace":     round(pace_sum[d] / n_simulations, 6),
        })

    probs_df = pd.DataFrame(rows).sort_values("win_prob", ascending=False).reset_index(drop=True)
    probs_df["predicted_pos"] = np.arange(1, len(probs_df) + 1)

    summary = probs_df[["predicted_pos", "driver", "team",
                         "win_prob", "podium_prob", "points_prob",
                         "dnf_prob", "expected_pos"]].copy()

    return {
        "round":           round_num,
        "circuit":         circuit,
        "position_counts": dict(position_counts),
        "probabilities":   probs_df,
        "summary":         summary,
    }


def simulate_season(
    features_2026: pd.DataFrame,
    n_simulations: int = 5000,
    qualifying_grids: pd.DataFrame = None,
    seed: int = 2026,
    verbose: bool = True,
) -> dict:
    rounds = sorted(features_2026["round"].unique())
    season_results = {}
    all_rows = []

    for rnd in rounds:
        circuit = features_2026[features_2026["round"] == rnd]["circuit"].iloc[0]
        if verbose:
            print(f"  Simulating Round {rnd:2d}: {circuit}...")

        qg = qualifying_grids[qualifying_grids["round"] == rnd] \
            if qualifying_grids is not None else None

        result = simulate_race(
            features_2026=features_2026, round_num=rnd,
            n_simulations=n_simulations,
            seed=seed + rnd * 100,
            qualifying_grid=qg,
        )
        season_results[rnd] = result

        if result:
            probs = result["probabilities"].copy()
            probs["round"]   = rnd
            probs["circuit"] = circuit
            all_rows.append(probs)

    season_results["all_race_results"] = (
        pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    )
    return season_results


def expected_points_from_simulation(
    race_result: dict,
    n_simulations: int = 5000,
) -> pd.DataFrame:
    pos_counts = race_result.get("position_counts", {})
    if not pos_counts:
        return pd.DataFrame()

    rows = []
    for driver, pos_dict in pos_counts.items():
        fl_bonus = race_result["probabilities"].set_index("driver")["fl_prob"].get(driver, 0)
        exp_pts  = sum(cnt / n_simulations * F1_POINTS.get(pos, 0)
                       for pos, cnt in pos_dict.items())
        exp_pts += fl_bonus * FASTEST_LAP_POINT
        rows.append({"driver": driver, "expected_points": round(exp_pts, 3)})

    return pd.DataFrame(rows).sort_values("expected_points", ascending=False).reset_index(drop=True)
