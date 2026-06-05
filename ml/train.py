import json
import os
import pickle
import sys
import threading
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent.parent))
from ml.features import (
    _load_session_safe, _race_results,
    extract_prerace_features, extract_live_features,
)

_ML_DIR = Path(__file__).parent
_PRERACE_MODEL = _ML_DIR / "model_prerace.pkl"
_LIVE_MODEL = _ML_DIR / "model_live.pkl"
_FEATURE_COLS = _ML_DIR / "feature_cols.json"

_SEASON_WEIGHTS = {0: 1.0, 1: 0.7, 2: 0.5}
_DEFAULT_WEIGHT = 0.3
_LAP_INTERVAL = 5


def _weight(race_year: int, current_year: int) -> float:
    return _SEASON_WEIGHTS.get(current_year - race_year, _DEFAULT_WEIGHT)


def _encode_teams(dfs: list) -> tuple[list, dict]:
    all_teams: set[str] = set()
    for df in dfs:
        if "team" in df.columns:
            all_teams.update(df["team"].astype(str).unique())
    team_map = {t: i for i, t in enumerate(sorted(all_teams))}
    encoded = []
    for df in dfs:
        df = df.copy()
        if "team" in df.columns:
            df["team"] = df["team"].map(lambda t: team_map.get(str(t), -1))
        encoded.append(df)
    return encoded, team_map


def _training_races(current_year: int, current_round: int) -> list[tuple[int, int]]:
    races = []
    for delta in range(4):
        year = current_year - delta
        try:
            sched = fastf1.get_event_schedule(year, include_testing=False)
            for _, ev in sched.iterrows():
                rnd = int(ev["RoundNumber"])
                if year == current_year and rnd >= current_round:
                    continue
                races.append((year, rnd))
        except Exception:
            pass
    return races


def _save_feature_cols(key: str, cols: list) -> None:
    data = {}
    if _FEATURE_COLS.exists():
        with open(_FEATURE_COLS) as f:
            try:
                data = json.load(f)
            except Exception:
                data = {}
    data[key] = cols
    with open(_FEATURE_COLS, "w") as f:
        json.dump(data, f, indent=2)


def _fit_and_save(
    dfs: list,
    weights: list,
    path: Path,
    key: str,
    extra_bundle: dict | None = None,
    **model_kwargs,
) -> None:
    if not dfs:
        print(f"[ml.train] No training data for {key}.")
        return
    dfs, team_map = _encode_teams(dfs)
    combined = pd.concat(dfs, ignore_index=True)
    feature_cols = [c for c in combined.columns if c not in ("driver", "final_position")]
    X = combined[feature_cols]
    medians = X.median(numeric_only=True).to_dict()
    X = X.fillna(medians)
    y = combined["final_position"].values

    model = XGBRegressor(**model_kwargs)
    model.fit(X, y, sample_weight=np.array(weights))

    bundle = {
        "model": model,
        "feature_cols": feature_cols,
        "team_map": team_map,
        "feature_medians": medians,
    }
    if extra_bundle:
        bundle.update(extra_bundle)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    _save_feature_cols(key, feature_cols)

    top = sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1])[:5]
    print(f"[ml.train] {key} model saved. Top features: {top}")


def train_prerace(current_year: int, current_round: int) -> None:
    races = _training_races(current_year, current_round)
    n_current = sum(1 for yr, _ in races if yr == current_year)
    low_confidence = n_current < 4

    dfs, weights = [], []
    for year, rnd in races:
        df = extract_prerace_features(year, rnd, include_target=True)
        if df is None or df.empty or "final_position" not in df.columns:
            continue
        df = df.dropna(subset=["final_position"])
        if df.empty:
            continue
        dfs.append(df)
        weights.extend([_weight(year, current_year)] * len(df))

    _fit_and_save(
        dfs, weights, _PRERACE_MODEL, "prerace",
        extra_bundle={"low_confidence": low_confidence},
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    print(f"[ml.train] Low confidence: {low_confidence} ({n_current} current-season races)")


def train_live(current_year: int, current_round: int) -> None:
    races = _training_races(current_year, current_round)
    dfs, weights = [], []

    for year, rnd in races:
        race = _load_session_safe(year, rnd, "R")
        if race is None:
            continue
        total_laps = int(race.laps["LapNumber"].max()) if not race.laps.empty else 0
        if total_laps < 10:
            continue
        final_pos = _race_results(race)

        for lap_n in range(5, total_laps + 1, _LAP_INTERVAL):
            snap = _lap_snapshot(race, lap_n, total_laps)
            if snap is None:
                continue
            df = extract_live_features(snap, lap_n)
            if df is None or df.empty:
                continue
            df["final_position"] = df["driver"].map(final_pos)
            df = df.dropna(subset=["final_position"])
            if df.empty:
                continue
            dfs.append(df)
            weights.extend([_weight(year, current_year)] * len(df))

    _fit_and_save(
        dfs, weights, _LIVE_MODEL, "live",
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )


def _lap_snapshot(race, lap_n: int, total_laps: int) -> dict | None:
    try:
        up_to = race.laps[race.laps["LapNumber"] <= lap_n]
        if up_to.empty:
            return None
        drivers = {}
        for abbr in up_to["Driver"].unique():
            d = up_to[up_to["Driver"] == abbr].sort_values("LapNumber").iloc[-1]
            try:
                pos = int(float(d["Position"])) if not pd.isna(d.get("Position")) else None
            except (TypeError, ValueError):
                pos = None
            stint = int(d["Stint"]) if not pd.isna(d.get("Stint")) else 1
            drivers[abbr] = {
                "position": pos,
                "gap_to_leader": None,
                "interval": None,
                "compound": str(d["Compound"]) if not pd.isna(d.get("Compound")) else None,
                "tyre_life": int(d["TyreLife"]) if not pd.isna(d.get("TyreLife")) else 0,
                "pit_stops": max(0, stint - 1),
                "stint": stint,
                "delta_to_fastest": None,
                "team": None,
            }
        ts_row = race.laps[race.laps["LapNumber"] == lap_n]
        track_status = "1"
        if not ts_row.empty and "TrackStatus" in ts_row.columns:
            ts = ts_row.iloc[0].get("TrackStatus")
            track_status = str(ts) if not pd.isna(ts) else "1"
        return {
            "drivers": drivers,
            "session": {"total_laps": total_laps, "current_lap": lap_n},
            "track_status": track_status,
        }
    except Exception as e:
        print(f"[ml.train] Snapshot failed at lap {lap_n}: {e}")
        return None


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    import importlib
    fastf1_loader = importlib.import_module("fastf1_loader")
    fastf1_loader.enable_cache(os.getenv("CACHE_DIR", "f1_cache"))

    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
    rnd = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    print(f"Training models using data before {year} R{rnd}...")

    t1 = threading.Thread(target=train_prerace, args=(year, rnd))
    t2 = threading.Thread(target=train_live, args=(year, rnd))
    t1.start(); t2.start()
    t1.join(); t2.join()
