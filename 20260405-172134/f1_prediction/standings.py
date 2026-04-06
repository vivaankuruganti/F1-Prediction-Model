"""
standings.py
------------
Computes expected driver and constructor championship standings
from Monte Carlo simulation results.

F1 points: 25-18-15-12-10-8-6-4-2-1 for positions 1–10
Fastest lap bonus: +1 if finisher is in P1–P10
"""

import numpy as np
import pandas as pd
from race_simulation import F1_POINTS, FASTEST_LAP_POINT


def _is_race_round(k) -> bool:
    """True if key is a round number (int or numpy integer), not a string key."""
    return isinstance(k, (int, np.integer)) and not isinstance(k, bool)


def compute_expected_standings(
    season_results: dict,
    n_simulations: int = 5000,
) -> tuple:
    driver_points    = {}
    driver_wins      = {}
    driver_podiums   = {}
    constructor_pts  = {}
    constructor_wins = {}

    rounds = [k for k in season_results if _is_race_round(k)]

    # Build driver→team map from last round
    driver_team_map = {}
    for rnd in sorted(rounds, reverse=True):
        res = season_results[rnd]
        if res:
            df = res.get("probabilities", pd.DataFrame())
            if not df.empty:
                driver_team_map = df.set_index("driver")["team"].to_dict()
                break

    for rnd in rounds:
        result = season_results[rnd]
        if not result:
            continue

        pos_counts = result.get("position_counts", {})
        probs_df   = result.get("probabilities", pd.DataFrame())
        if probs_df.empty:
            continue

        fl_probs = probs_df.set_index("driver")["fl_prob"].to_dict()
        teams    = probs_df.set_index("driver")["team"].to_dict()

        for driver, pos_dict in pos_counts.items():
            team = teams.get(driver, "Unknown")
            driver_points.setdefault(driver,   0.0)
            driver_wins.setdefault(driver,     0.0)
            driver_podiums.setdefault(driver,  0.0)
            constructor_pts.setdefault(team,   0.0)
            constructor_wins.setdefault(team,  0.0)

            for pos, cnt in pos_dict.items():
                prob    = cnt / n_simulations
                pts     = F1_POINTS.get(pos, 0)
                exp_pts = prob * pts
                driver_points[driver]  += exp_pts
                constructor_pts[team]  += exp_pts
                if pos == 1:
                    driver_wins[driver]    += prob
                    constructor_wins[team] += prob
                if pos <= 3:
                    driver_podiums[driver] += prob

            fl_p = fl_probs.get(driver, 0.0)
            driver_points[driver]  += fl_p * FASTEST_LAP_POINT
            constructor_pts[team]  += fl_p * FASTEST_LAP_POINT

    # ── Driver standings ──────────────────────────────────────────────────────
    driver_rows = [
        {
            "driver":  d,
            "team":    driver_team_map.get(d, "Unknown"),
            "points":  round(pts, 1),
            "wins":    round(driver_wins.get(d, 0), 2),
            "podiums": round(driver_podiums.get(d, 0), 2),
        }
        for d, pts in driver_points.items()
    ]
    if not driver_rows:
        driver_df = pd.DataFrame(columns=["position", "driver", "team", "points", "wins", "podiums"])
    else:
        driver_df = (
            pd.DataFrame(driver_rows)
              .sort_values(["points", "wins", "podiums"], ascending=False)
              .reset_index(drop=True)
        )
        driver_df.insert(0, "position", np.arange(1, len(driver_df) + 1))

    # ── Constructor standings ─────────────────────────────────────────────────
    con_rows = [
        {"team": t, "points": round(pts, 1), "wins": round(constructor_wins.get(t, 0), 2)}
        for t, pts in constructor_pts.items()
    ]
    if not con_rows:
        con_df = pd.DataFrame(columns=["position", "team", "points", "wins"])
    else:
        con_df = (
            pd.DataFrame(con_rows)
              .sort_values(["points", "wins"], ascending=False)
              .reset_index(drop=True)
        )
        con_df.insert(0, "position", np.arange(1, len(con_df) + 1))

    return driver_df, con_df


# ── Pretty-print helpers ──────────────────────────────────────────────────────
def print_driver_standings(driver_df: pd.DataFrame, top_n: int = 22) -> None:
    print("\n" + "=" * 72)
    print("  2026 F1 DRIVER CHAMPIONSHIP  (Expected, Monte Carlo)")
    print("=" * 72)
    print(f"{'Pos':<4} {'Driver':<22} {'Team':<22} {'Points':>8}  {'Wins':>6}  {'Podiums':>8}")
    print("-" * 72)
    for _, row in driver_df.head(top_n).iterrows():
        print(f"{int(row['position']):<4} {row['driver']:<22} {row['team']:<22} "
              f"{row['points']:>8.1f}  {row['wins']:>6.2f}  {row['podiums']:>8.2f}")
    print("=" * 72)


def print_constructor_standings(con_df: pd.DataFrame) -> None:
    print("\n" + "=" * 52)
    print("  2026 F1 CONSTRUCTOR CHAMPIONSHIP  (Expected)")
    print("=" * 52)
    print(f"{'Pos':<4} {'Team':<26} {'Points':>8}  {'Wins':>6}")
    print("-" * 46)
    for _, row in con_df.iterrows():
        print(f"{int(row['position']):<4} {row['team']:<26} "
              f"{row['points']:>8.1f}  {row['wins']:>6.2f}")
    print("=" * 52)


def print_race_summary(result: dict, top_n: int = 22) -> None:
    rnd     = result.get("round", "?")
    circuit = result.get("circuit", "?")
    probs   = result.get("probabilities", pd.DataFrame())
    if probs.empty:
        return

    print("\n" + "=" * 82)
    print(f"  2026 ROUND {rnd}: {circuit.upper()} – Race Prediction")
    print("=" * 82)
    print(f"{'Pos':<4} {'Driver':<22} {'Team':<22} "
          f"{'Win%':>6} {'Pod%':>6} {'Pts%':>6} {'DNF%':>6} {'E.Pos':>6}")
    print("-" * 82)
    for _, row in probs.head(top_n).iterrows():
        print(
            f"{int(row['predicted_pos']):<4} "
            f"{row['driver']:<22} "
            f"{row['team']:<22} "
            f"{row['win_prob']*100:>5.1f}% "
            f"{row['podium_prob']*100:>5.1f}% "
            f"{row['points_prob']*100:>5.1f}% "
            f"{row['dnf_prob']*100:>5.1f}% "
            f"{row['expected_pos']:>6.1f}"
        )
    print("=" * 82)
