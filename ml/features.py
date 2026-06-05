import fastf1
import pandas as pd

from fastf1_loader import load_session as _load_session

_TRACK_STATUS_MAP = {"1": 1, "2": 2, "4": 3, "6": 4, "7": 5}
_DNF_POSITION = 21
_LONG_RUN_MIN_LAPS = 5
_SEASON_LOOKBACK = 8
_CIRCUIT_LOOKBACK = 3
_CIRCUIT_YEAR_WINDOW = 4
_COMPOUND_MAP = {"SOFT": 1, "MEDIUM": 2, "HARD": 3, "INTERMEDIATE": 4, "WET": 5}
_POINTS_MAP = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}


def _load_session_safe(year: int, round_num: int, session_type: str):
    return _load_session(year, round_num, session_type, weather=False, messages=False)


def _fp2_long_run_pace(fp2_session) -> dict[str, float]:
    if fp2_session is None:
        return {}
    laps = fp2_session.laps.copy()
    laps = laps[laps["IsAccurate"] == True].copy()
    laps = laps.dropna(subset=["LapTime", "Compound", "Driver"])
    laps["_secs"] = laps["LapTime"].dt.total_seconds()

    result = {}
    for driver, d_laps in laps.groupby("Driver"):
        best_mean = None
        for _, c_laps in d_laps.groupby("Compound"):
            c_laps = c_laps.sort_values("LapNumber").reset_index(drop=True)
            nums = c_laps["LapNumber"].values
            runs, run = [], [0]
            for i in range(1, len(nums)):
                if nums[i] == nums[i - 1] + 1:
                    run.append(i)
                else:
                    if len(run) >= _LONG_RUN_MIN_LAPS:
                        runs.append(run)
                    run = [i]
            if len(run) >= _LONG_RUN_MIN_LAPS:
                runs.append(run)
            for r in runs:
                mean = c_laps.iloc[r]["_secs"].mean()
                if best_mean is None or mean < best_mean:
                    best_mean = mean
        if best_mean is not None:
            result[driver] = best_mean
    return result


def _race_results(session) -> dict[str, int]:
    out = {}
    for _, row in session.results.iterrows():
        abbr = str(row["Abbreviation"])
        try:
            p = int(float(row["Position"]))
            out[abbr] = p if 1 <= p <= 20 else _DNF_POSITION
        except (TypeError, ValueError):
            out[abbr] = _DNF_POSITION
    return out


def _season_results_before(year: int, round_num: int) -> list[dict]:
    results = []
    try:
        sched = fastf1.get_event_schedule(year, include_testing=False)
        prior = sorted(int(ev["RoundNumber"]) for _, ev in sched.iterrows()
                       if int(ev["RoundNumber"]) < round_num)
    except Exception:
        return results
    for rnd in prior:
        s = _load_session_safe(year, rnd, "R")
        if s is not None:
            results.append({"round": rnd, "positions": _race_results(s)})
    return results


def _circuit_history(year: int, round_num: int) -> list[dict]:
    try:
        country = fastf1.get_event(year, round_num)["Country"]
    except Exception:
        return []
    history = []
    for hist_year in range(year - 1, year - 1 - _CIRCUIT_YEAR_WINDOW, -1):
        if len(history) >= _CIRCUIT_LOOKBACK:
            break
        try:
            sched = fastf1.get_event_schedule(hist_year, include_testing=False)
            match = sched[sched["Country"] == country]
            if match.empty:
                continue
            hist_round = int(match.iloc[0]["RoundNumber"])
        except Exception:
            continue
        s = _load_session_safe(hist_year, hist_round, "R")
        if s is not None:
            history.append({"year": hist_year, "positions": _race_results(s)})
    return history


def extract_prerace_features(year: int, round_num: int, include_target: bool = True):
    qual = _load_session_safe(year, round_num, "Q")
    if qual is None:
        return None

    fp2 = _load_session_safe(year, round_num, "FP2")
    long_run = _fp2_long_run_pace(fp2)
    field_best = min(long_run.values()) if long_run else None

    race = _load_session_safe(year, round_num, "R") if include_target else None
    season_hist = _season_results_before(year, round_num)
    circuit_hist = _circuit_history(year, round_num)

    rows = []
    for _, qrow in qual.results.iterrows():
        abbr = str(qrow["Abbreviation"])
        team = str(qrow.get("TeamName") or "Unknown")

        try:
            grid = int(float(qrow["Position"]))
        except (TypeError, ValueError):
            grid = 20

        pace = long_run.get(abbr)
        fp2_avg = pace if pace is not None else float("nan")
        fp2_gap = (pace - field_best) if (pace is not None and field_best is not None) else float("nan")

        tail = season_hist[-_SEASON_LOOKBACK:]
        prev_s = [r["positions"].get(abbr, _DNF_POSITION) for r in tail]
        prev_s += [float("nan")] * (_SEASON_LOOKBACK - len(prev_s))

        prev_c = [r["positions"].get(abbr, _DNF_POSITION) for r in circuit_hist]
        prev_c += [float("nan")] * (_CIRCUIT_LOOKBACK - len(prev_c))

        points = sum(_POINTS_MAP.get(r["positions"].get(abbr, _DNF_POSITION), 0) for r in season_hist)

        final_pos = float("nan")
        if race is not None:
            rrow = race.results[race.results["Abbreviation"] == abbr]
            if not rrow.empty:
                rrow = rrow.iloc[0]
                try:
                    p = int(float(rrow["Position"]))
                    final_pos = p if 1 <= p <= 20 else _DNF_POSITION
                except (TypeError, ValueError):
                    final_pos = _DNF_POSITION

        row = {
            "driver": abbr, "team": team,
            "grid_position": grid,
            "fp2_long_run_avg": fp2_avg,
            "fp2_long_run_gap": fp2_gap,
            "season_points": points,
        }
        for i, p in enumerate(prev_s):
            row[f"prev_season_{i + 1}"] = p
        for i, p in enumerate(prev_c):
            row[f"prev_circuit_{i + 1}"] = p
        if include_target:
            row["final_position"] = final_pos
        rows.append(row)

    return pd.DataFrame(rows) if rows else None


def extract_live_features(state: dict, lap_n: int):
    drivers = state.get("drivers", {})
    session = state.get("session", {})
    total_laps = session.get("total_laps") or 0
    if not drivers or total_laps == 0:
        return None

    status_enc = _TRACK_STATUS_MAP.get(str(state.get("track_status", "1")), 1)
    laps_remaining = max(0.0, (total_laps - lap_n) / total_laps)

    rows = []
    for abbr, ds in drivers.items():
        rows.append({
            "driver": abbr,
            "team": ds.get("team") or "Unknown",
            "position": ds.get("position") or 20,
            "gap_to_leader": _parse_gap(ds.get("gap_to_leader")),
            "interval": _parse_gap(ds.get("interval")),
            "compound": _COMPOUND_MAP.get(str(ds.get("compound") or "").upper(), 0),
            "tyre_life": ds.get("tyre_life") or 0,
            "pit_stops": ds.get("pit_stops") or 0,
            "laps_remaining_ratio": laps_remaining,
            "stint": ds.get("stint") or 1,
            "delta_to_fastest": _parse_delta(ds.get("delta_to_fastest")),
            "track_status": status_enc,
        })
    return pd.DataFrame(rows) if rows else None


def _parse_float(val, extra_nan=()) -> float:
    if val is None or val == "" or val in extra_nan:
        return float("nan")
    try:
        return float(str(val).replace("+", "").strip())
    except (ValueError, TypeError):
        return float("nan")


def _parse_gap(val) -> float:
    return _parse_float(val, extra_nan=("leader", "—"))


def _parse_delta(val) -> float:
    return _parse_float(val)
