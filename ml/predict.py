import pickle
from pathlib import Path
from typing import Optional

import pandas as pd

from ml.features import extract_prerace_features, extract_live_features

_ML_DIR = Path(__file__).parent
_PRERACE_MODEL = _ML_DIR / "model_prerace.pkl"
_LIVE_MODEL = _ML_DIR / "model_live.pkl"

_prerace_bundle: dict | None = None
_live_bundle: dict | None = None


def load_models() -> None:
    global _prerace_bundle, _live_bundle
    if _PRERACE_MODEL.exists():
        with open(_PRERACE_MODEL, "rb") as f:
            _prerace_bundle = pickle.load(f)
        print("[ml.predict] PRE-RACE model loaded.")
    else:
        print("[ml.predict] No PRE-RACE model — predictions disabled.")
    if _LIVE_MODEL.exists():
        with open(_LIVE_MODEL, "rb") as f:
            _live_bundle = pickle.load(f)
        print("[ml.predict] LIVE model loaded.")
    else:
        print("[ml.predict] No LIVE model — live predictions disabled.")


def is_low_confidence() -> bool:
    return bool((_prerace_bundle or {}).get("low_confidence", False))


def predict_prerace(year: int, round_num: int) -> Optional[list[dict]]:
    if _prerace_bundle is None:
        return None
    df = extract_prerace_features(year, round_num, include_target=False)
    if df is None or df.empty:
        return None
    results = _infer(_prerace_bundle, df)
    if results is None:
        return None
    grid_map = df.set_index("driver")["grid_position"].to_dict()
    for r in results:
        r["grid_position"] = int(grid_map[r["driver"]]) if r["driver"] in grid_map else None
    return results


def predict_live(state: dict) -> Optional[list[dict]]:
    if _live_bundle is None:
        return None
    current_lap = (state.get("session") or {}).get("current_lap") or 0
    if current_lap < 5:
        return None
    df = extract_live_features(state, current_lap)
    if df is None or df.empty:
        return None
    return _infer(_live_bundle, df)


def _infer(bundle: dict, df: pd.DataFrame) -> Optional[list[dict]]:
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    team_map = bundle["team_map"]
    medians = bundle.get("feature_medians", {})

    df = df.copy()
    if "team" in df.columns:
        df["team"] = df["team"].map(lambda t: team_map.get(str(t), -1))

    for col in feature_cols:
        if col not in df.columns:
            df[col] = float("nan")

    X = df[feature_cols].fillna(value=medians)

    scores = model.predict(X.values.astype(float))
    results = [{"driver": abbr, "_s": float(s)}
               for abbr, s in zip(df["driver"].tolist(), scores)]
    results.sort(key=lambda x: x["_s"])
    for rank, r in enumerate(results, 1):
        r["predicted_position"] = rank
        del r["_s"]
    return results
