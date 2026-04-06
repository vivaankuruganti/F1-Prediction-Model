"""
quali_model.py
--------------
Qualifying prediction model – updated to incorporate 2026 live data.

Training strategy:
  1. Primary signal: actual 2026 qualifying data (AUS GP Q1/Q2/Q3 times)
  2. Supporting signal: historical pattern learning (track-type interactions,
     weather effects) from 2018-2025 data
  3. Prior-based score: anchor for drivers/tracks not yet seen in 2026

2026 pace_2026_quali feature is the dominant predictor because:
  - It is derived from real 2026 lap times at real 2026 race conditions
  - Regulations have reset, so historical patterns only inform *interactions*

Model: GradientBoosting (sklearn) as default; LightGBM/XGBoost when available
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor

_LGB_AVAILABLE = False
_XGB_AVAILABLE = False

try:
    import lightgbm as lgb          # noqa: F401
    _LGB_AVAILABLE = True
except Exception:
    pass

if not _LGB_AVAILABLE:
    try:
        import xgboost as xgb       # noqa: F401
        _XGB_AVAILABLE = True
    except Exception:
        pass

if not _LGB_AVAILABLE and not _XGB_AVAILABLE:
    print("[INFO] LightGBM/XGBoost unavailable. Using scikit-learn GradientBoosting.")

from feature_engineering import FEATURE_COLS, compute_upgrade_boost


class QualiModel:
    """Trains and predicts qualifying performance scores."""

    def __init__(self):
        self.model       = None
        self.fitted      = False
        self.feature_cols_ = None

    # ── Training ──────────────────────────────────────────────────────────────
    def train(self, train_df: pd.DataFrame, verbose: bool = True) -> None:
        if train_df.empty:
            print("[WARNING] No training data – will use prior + 2026 scoring.")
            return

        available = [c for c in FEATURE_COLS if c in train_df.columns]
        X = train_df[available].astype(float)
        y = train_df["quali_norm"].astype(float)

        train_df = train_df.copy()
        train_df["race_group"] = (
            train_df["year"].astype(str) + "_" + train_df["round"].astype(str)
        )
        groups   = train_df["race_group"]
        n_groups = groups.nunique()
        n_splits = min(5, max(2, n_groups // 10))

        if verbose:
            print(f"[QualiModel] Training on {len(X):,} rows, "
                  f"{len(available)} features, {n_groups} races.")

        if _LGB_AVAILABLE:
            self._train_lgb(X, y, groups, n_splits, verbose)
        elif _XGB_AVAILABLE:
            self._train_xgb(X, y, groups, n_splits, verbose)
        else:
            self._train_sklearn(X, y, groups, n_splits, verbose)

    def _train_lgb(self, X, y, groups, n_splits, verbose):
        from lightgbm import LGBMRegressor, early_stopping, log_evaluation
        params = dict(n_estimators=800, learning_rate=0.03, num_leaves=63,
                      max_depth=6, min_child_samples=20, subsample=0.8,
                      colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
                      random_state=42, n_jobs=-1, verbose=-1)
        gkf = GroupKFold(n_splits=n_splits)
        best_model, best_mae, val_maes = None, float("inf"), []
        for _, (tr, va) in enumerate(gkf.split(X, y, groups)):
            m = LGBMRegressor(**params)
            m.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])],
                  callbacks=[early_stopping(50, verbose=False), log_evaluation(-1)])
            mae = mean_absolute_error(y.iloc[va], m.predict(X.iloc[va]))
            val_maes.append(mae)
            if mae < best_mae:
                best_mae, best_model = mae, m
        self.model, self.fitted = best_model, True
        self.feature_cols_ = list(X.columns)
        if verbose:
            print(f"[QualiModel] LGB CV MAE = {np.mean(val_maes):.4f}  (best {best_mae:.4f})")

    def _train_xgb(self, X, y, groups, n_splits, verbose):
        from xgboost import XGBRegressor
        params = dict(n_estimators=800, learning_rate=0.03, max_depth=6,
                      subsample=0.8, colsample_bytree=0.8,
                      reg_alpha=0.1, reg_lambda=0.1,
                      random_state=42, n_jobs=-1, verbosity=0)
        gkf = GroupKFold(n_splits=n_splits)
        best_model, best_mae, val_maes = None, float("inf"), []
        for _, (tr, va) in enumerate(gkf.split(X, y, groups)):
            m = XGBRegressor(**params)
            m.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])],
                  early_stopping_rounds=50, verbose=False)
            mae = mean_absolute_error(y.iloc[va], m.predict(X.iloc[va]))
            val_maes.append(mae)
            if mae < best_mae:
                best_mae, best_model = mae, m
        self.model, self.fitted = best_model, True
        self.feature_cols_ = list(X.columns)
        if verbose:
            print(f"[QualiModel] XGB CV MAE = {np.mean(val_maes):.4f}  (best {best_mae:.4f})")

    def _train_sklearn(self, X, y, groups, n_splits, verbose):
        params = dict(n_estimators=400, learning_rate=0.05, max_depth=5,
                      subsample=0.8, min_samples_leaf=10, random_state=42)
        gkf = GroupKFold(n_splits=n_splits)
        best_model, best_mae, val_maes = None, float("inf"), []
        for _, (tr, va) in enumerate(gkf.split(X, y, groups)):
            m = GradientBoostingRegressor(**params)
            m.fit(X.iloc[tr], y.iloc[tr])
            mae = mean_absolute_error(y.iloc[va], m.predict(X.iloc[va]))
            val_maes.append(mae)
            if mae < best_mae:
                best_mae, best_model = mae, m
        self.model, self.fitted = best_model, True
        self.feature_cols_ = list(X.columns)
        if verbose:
            print(f"[QualiModel] sklearn GB CV MAE = {np.mean(val_maes):.4f}  "
                  f"(best {best_mae:.4f})")

    # ── Scoring ───────────────────────────────────────────────────────────────
    @staticmethod
    def _prior_score(row: pd.Series) -> float:
        """
        Hybrid prior + 2026 pace score for qualifying.

        v3 weights (updated priors now carry more weight):
          pace_2026_quali: 0.35  ← 2026 live signal
          team_strength:   0.22  ← team quali pace (↑ priors now strongly differentiated)
          driver_strength: 0.18  ← driver skill prior
          engine contrib:  0.16  ← engine × power sensitivity (↑ bigger engine gaps)
          aero contrib:    0.09  ← aero × downforce sensitivity
        """
        ps = row.get("power_sensitivity",    0.65)
        ds = row.get("downforce_sensitivity", 0.65)

        engine_contrib = (
            row.get("engine_power", 0.90) * ps +
            row.get("engine_driveability", 0.90) * (1 - ps)
        )
        aero_contrib = (
            row.get("aero_quality", 0.85) * ds +
            row.get("drag_efficiency", 0.85) * (1 - ds)
        )
        rain  = row.get("rain_flag", 0)
        wet_b = row.get("wet_strength", 0.85) * rain * 0.05

        return (
            0.35 * row.get("pace_2026_quali", 0.96) +
            0.22 * row.get("team_strength",   0.85) +
            0.18 * row.get("driver_strength", 0.85) +
            0.16 * engine_contrib +
            0.09 * aero_contrib +
            wet_b
        )

    # ── Predict score for a race ──────────────────────────────────────────────
    def predict_score(self, race_df: pd.DataFrame) -> pd.Series:
        prior_scores = race_df.apply(self._prior_score, axis=1)

        if self.fitted and self.model is not None:
            available = [c for c in self.feature_cols_ if c in race_df.columns]
            X = race_df[available].astype(float).fillna(0.5)
            missing = [c for c in self.feature_cols_ if c not in available]
            for mc in missing:
                X[mc] = 0.5
            X = X[self.feature_cols_]
            ml_scores = pd.Series(self.model.predict(X), index=race_df.index)
            # v3: Prior score carries updated 2026 priors strongly.
            # Use 70% prior (updated-priors-aware) + 30% ML interaction patterns
            return 0.70 * prior_scores + 0.30 * ml_scores
        else:
            return prior_scores

    # ── Predict full qualifying grid for one round ────────────────────────────
    def predict_grid(
        self,
        features_2026: pd.DataFrame,
        round_num: int,
        add_noise: bool = True,
        noise_sigma: float = 0.020,
        seed: int = None,
        # AUS Q actuals: override Round 1 with real results
        known_grids: dict = None,
    ) -> pd.DataFrame:
        """
        Returns predicted qualifying grid for round_num.
        If known_grids[round_num] is provided (actual 2026 results), uses those directly.
        """
        race = features_2026[features_2026["round"] == round_num].copy()
        if race.empty:
            return pd.DataFrame()

        # ── Use actual 2026 qualifying result if available ────────────────────
        if known_grids and round_num in known_grids:
            actual = known_grids[round_num].copy()
            # Merge to get all driver data; use actual grid order
            merged = race.merge(actual[["driver", "quali_pos", "best_time"]],
                                on="driver", how="left")
            # For drivers not in actual (e.g. VER with no time), estimate from score
            missing_pos = merged["quali_pos"].isna()
            if missing_pos.any():
                scores = self.predict_score(merged[missing_pos].copy())
                # Place missing drivers after last known position
                max_pos = merged["quali_pos"].max()
                if np.isnan(max_pos):
                    max_pos = 0
                ranked = scores.rank(method="first", ascending=False)
                for idx in merged[missing_pos].index:
                    merged.loc[idx, "quali_pos"] = max_pos + ranked.loc[idx]

            merged["quali_score"] = 1.0 / merged["quali_pos"]
            merged = merged.sort_values("quali_pos").reset_index(drop=True)
            return merged[["driver", "team", "engine_supplier", "quali_score", "quali_pos"]
                          + [c for c in ["circuit", "round", "rain_flag", "avg_temp",
                                         "weather_variability"] if c in merged.columns]]

        # ── Predict future round ──────────────────────────────────────────────
        if seed is not None:
            np.random.seed(seed)

        scores = self.predict_score(race)

        if add_noise:
            wv = race["weather_variability"].fillna(0.4).values
            noise = np.random.normal(0, noise_sigma, size=len(scores)) * (1 + wv * 0.5)
            scores = scores + noise

        race = race.copy()
        race["quali_score"] = scores.values
        race = race.sort_values("quali_score", ascending=False).reset_index(drop=True)
        race["quali_pos"] = np.arange(1, len(race) + 1)

        return race[["driver", "team", "engine_supplier", "quali_score", "quali_pos"]
                    + [c for c in ["circuit", "round", "rain_flag", "avg_temp",
                                   "weather_variability"] if c in race.columns]]


def predict_all_grids(
    model: QualiModel,
    features_2026: pd.DataFrame,
    seed: int = 2026,
    known_grids: dict = None,
) -> pd.DataFrame:
    """Predict qualifying grid for every round of the 2026 season."""
    results = []
    for rnd in sorted(features_2026["round"].unique()):
        grid = model.predict_grid(
            features_2026, round_num=rnd,
            add_noise=True, seed=seed + rnd,
            known_grids=known_grids,
        )
        results.append(grid)
    return pd.concat(results, ignore_index=True)
