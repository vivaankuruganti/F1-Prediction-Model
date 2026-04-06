"""
main.py
-------
Orchestrates the 2026 F1 Prediction System (v3 – updated priors).

What's new vs v2:
  - Updated priors from ~/Desktop/priors/ override old ones:
      * Mercedes = dominant car  (pace 1.0, engine 1.0)
      * Ferrari  = strong/reliable (pace 0.98, rel 0.98)
      * McLaren  = competitive midfront (pace 0.90)
      * Red Bull = mid-pack + unreliable engine (pace 0.85, engine_rel 0.60)
      * Aston Martin = backmarker  (Honda engine 0.4 power, pace 0.64)
      * Haas + Alpine = improved
  - Prior weights increased across quali_model and race_simulation
  - "Isak Hadjar" canonical spelling (single 'k') throughout
  - All 2026 live session data retained from v2

Usage
-----
  python3 main.py                      # full season (5000 sims)
  python3 main.py --race 1             # Round 1 only (actual AUS Q grid)
  python3 main.py --race 4 --sims 3000 # Miami
  python3 main.py --no-train           # skip ML
  python3 main.py --save               # write CSVs to ./output/
"""

import argparse
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from data_loader        import (load_master_data, load_all_priors,
                                 load_2026_all_laptimes, load_2026_qualifying_results)
from feature_engineering import (build_training_features, build_2026_features,
                                  compute_2026_pace_signals, aggregate_2026_pace)
from quali_model        import QualiModel, predict_all_grids
from race_simulation    import simulate_race, simulate_season, expected_points_from_simulation
from standings          import (compute_expected_standings, print_driver_standings,
                                 print_constructor_standings, print_race_summary)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="F1 2026 Prediction System (v2)")
    p.add_argument("--race",     type=int, default=None,
                   help="Predict a single race round (1–22)")
    p.add_argument("--sims",     type=int, default=5000,
                   help="Monte Carlo simulations per race (default: 5000)")
    p.add_argument("--no-train", action="store_true",
                   help="Skip ML training; use prior + 2026 scoring only")
    p.add_argument("--save",     action="store_true",
                   help="Save outputs to ./output/")
    p.add_argument("--seed",     type=int, default=2026)
    p.add_argument("--min-year", type=int, default=2018,
                   help="Minimum year for historical training data")
    return p.parse_args()


# ── Save helpers ──────────────────────────────────────────────────────────────
def save_outputs(driver_df, con_df, all_quali, season_results, output_dir="./output"):
    os.makedirs(output_dir, exist_ok=True)
    driver_df.to_csv(os.path.join(output_dir, "driver_standings_2026.csv"), index=False)
    con_df.to_csv(os.path.join(output_dir, "constructor_standings_2026.csv"), index=False)
    all_quali.to_csv(os.path.join(output_dir, "qualifying_grids_2026.csv"), index=False)
    all_race = season_results.get("all_race_results", pd.DataFrame())
    if not all_race.empty:
        all_race.to_csv(os.path.join(output_dir, "race_probabilities_2026.csv"), index=False)
    print(f"\n[Saved] Results → {os.path.abspath(output_dir)}/")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    t0   = time.time()

    print("=" * 68)
    print("  F1 2026 RACE PREDICTION SYSTEM  (v3 – updated priors)")
    print("  Regulation-reset | Monte Carlo | 2026 live data + new priors")
    print("=" * 68)

    # ── Step 1: Load priors + historical data ─────────────────────────────────
    from data_loader import NEW_PRIORS_DIR
    print("\n[1/7] Loading priors & historical data...")
    print(f"  Priors source   : {NEW_PRIORS_DIR}")
    master = load_master_data(min_year=args.min_year)
    priors = load_all_priors()
    if not master.empty and "year" in master.columns:
        print(f"  Historical rows : {len(master):,}  ({master['year'].min()}–{master['year'].max()})")
    else:
        print(f"  Historical rows : {len(master):,}")
    print(f"  2026 drivers    : {len(priors['driver_priors'])}")
    print(f"  2026 rounds     : {len(priors['track_priors'])}")

    # ── Step 2: Load 2026 live session data ───────────────────────────────────
    print("\n[2/7] Loading 2026 live session data...")
    laps_2026 = load_2026_all_laptimes()
    if not laps_2026.empty:
        print(f"  Sessions loaded : {laps_2026['session_tag'].nunique()}")
        for tag in sorted(laps_2026["session_tag"].unique()):
            sub = laps_2026[laps_2026["session_tag"] == tag]
            print(f"    {tag:<22} : {sub['driver'].nunique():2d} drivers, "
                  f"{len(sub):4d} laps")
    else:
        print("  [WARNING] No 2026 session data found – falling back to priors only.")

    # ── Step 3: Compute 2026 pace signal ─────────────────────────────────────
    print("\n[3/7] Computing 2026 pace signals...")
    pace_per_session = compute_2026_pace_signals(laps_2026)
    pace_2026        = aggregate_2026_pace(pace_per_session)

    if not pace_2026.empty:
        print(f"  Pace signal for {len(pace_2026)} drivers (weighted avg across sessions)")
        print()
        print(f"  {'Driver':<24} {'Quali Pace':>12} {'Race Pace':>11}")
        print(f"  {'─'*48}")
        for _, row in pace_2026.sort_values("pace_2026_quali", ascending=False).iterrows():
            print(f"  {row['driver']:<24} {row['pace_2026_quali']:>12.4f} {row['pace_2026_race']:>11.4f}")
    else:
        print("  No pace signal computed – using prior-derived estimates.")

    # ── Step 4: Build 2026 feature table ──────────────────────────────────────
    print("\n[4/7] Building 2026 feature table...")
    features_2026 = build_2026_features(
        driver_priors      = priors["driver_priors"],
        team_priors        = priors["team_priors"],
        engine_priors      = priors["engine_priors"],
        track_priors       = priors["track_priors"],
        reliability_priors = priors["reliability_priors"],
        upgrade_priors     = priors["upgrade_priors"],
        team_engine_map    = priors["team_engine_map"],
        driver_team_map    = priors["driver_team_map"],
        weather            = priors["weather"],
        pace_2026          = pace_2026,
    )
    print(f"  Feature rows: {len(features_2026):,} "
          f"({features_2026['driver'].nunique()} drivers × "
          f"{features_2026['round'].nunique()} rounds)")

    # ── Step 5: Load actual AUS qualifying result ─────────────────────────────
    print("\n[5/7] Loading actual 2026 qualifying results...")
    aus_q_results = load_2026_qualifying_results()
    known_grids = {}
    if not aus_q_results.empty:
        print(f"  AUS Qualifying actual grid ({len(aus_q_results)} drivers):")
        print()
        print(f"  {'Pos':<4} {'Driver':<24} {'Team':<22} {'Q3':>8} {'Best':>8}")
        print(f"  {'─'*68}")
        for _, row in aus_q_results.iterrows():
            q3  = f"{row['q3_time']:.3f}" if not pd.isna(row.get('q3_time', np.nan)) else "  —  "
            bst = f"{row['best_time']:.3f}" if not pd.isna(row.get('best_time', np.nan)) else "  —  "
            print(f"  P{int(row['quali_pos']):<3} {row['driver']:<24} {row['team']:<22} {q3:>8} {bst:>8}")

        known_grids[1] = aus_q_results   # Round 1 = Australian GP
    else:
        print("  AUS qualifying data not found.")

    # ── Step 6: Train qualifying model ────────────────────────────────────────
    qm = QualiModel()
    if not args.no_train and not master.empty:
        print("\n[6/7] Training qualifying model...")
        train_df = build_training_features(
            master=master,
            track_priors=priors["track_priors"],
            pace_2026=pace_2026,
        )
        print(f"  Training rows: {len(train_df):,}")
        qm.train(train_df, verbose=True)
    else:
        print("\n[6/7] Skipping ML training – using prior + 2026 scoring.")

    # ── Predict qualifying grids ──────────────────────────────────────────────
    all_quali = predict_all_grids(qm, features_2026, seed=args.seed,
                                   known_grids=known_grids)

    # ── Step 7: Simulate ──────────────────────────────────────────────────────
    print(f"\n[7/7] Running Monte Carlo simulations ({args.sims:,}/race)...")

    if args.race is not None:
        rnd = args.race
        print(f"\n  Simulating Round {rnd}...")
        result = simulate_race(
            features_2026=features_2026, round_num=rnd,
            n_simulations=args.sims, seed=args.seed,
            qualifying_grid=all_quali,
        )
        season_results = {rnd: result,
                          "all_race_results": result.get("probabilities", pd.DataFrame())}

        # Print qualifying grid
        race_quali = all_quali[all_quali["round"] == rnd].copy()
        print(f"\n  {'─'*54}")
        print(f"  Round {rnd} Qualifying Grid – {result.get('circuit', '?')}")
        if rnd == 1:
            print("  (Actual 2026 result)")
        print(f"  {'─'*54}")
        for _, row in race_quali.iterrows():
            print(f"  P{int(row['quali_pos']):<3} {row['driver']:<24} {row['team']}")

        print_race_summary(result, top_n=22)

        exp_pts = expected_points_from_simulation(result, n_simulations=args.sims)
        print("\n  Expected Points:")
        print(f"  {'Driver':<24} {'E.Pts':>8}")
        print(f"  {'─'*34}")
        for _, row in exp_pts.head(15).iterrows():
            print(f"  {row['driver']:<24} {row['expected_points']:>8.2f}")

        driver_df, con_df = compute_expected_standings(season_results, n_simulations=args.sims)
        print_driver_standings(driver_df)
        print_constructor_standings(con_df)

    else:
        season_results = simulate_season(
            features_2026=features_2026, n_simulations=args.sims,
            qualifying_grids=all_quali, seed=args.seed, verbose=True,
        )
        driver_df, con_df = compute_expected_standings(season_results, n_simulations=args.sims)
        print_driver_standings(driver_df)
        print_constructor_standings(con_df)

        # Print Round 1 (AUS – actual result used as anchor)
        if 1 in season_results:
            print_race_summary(season_results[1], top_n=22)

    if args.save:
        d_df, c_df = (driver_df, con_df) if "driver_df" in dir() else (
            compute_expected_standings(season_results, n_simulations=args.sims)
        )
        save_outputs(d_df, c_df, all_quali, season_results)

    print(f"\n[Done] Runtime: {time.time() - t0:.1f}s\n")


if __name__ == "__main__":
    main()
