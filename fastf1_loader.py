import fastf1
import pandas as pd
from datetime import datetime, timezone

fastf1.set_log_level('WARNING')


def enable_cache(cache_dir: str = "f1_cache") -> None:
    fastf1.Cache.enable_cache(cache_dir)


def get_last_completed_race() -> tuple[int, int] | None:
    now = datetime.now(timezone.utc)
    year = now.year

    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
    except Exception as e:
        print(f"[loader] Could not load {year} schedule: {e}")
        return None

    completed = []
    for _, event in schedule.iterrows():
        try:
            race_date = event["Session5DateUtc"]
            if pd.isna(race_date):
                continue
            if race_date.tzinfo is None:
                race_date = race_date.replace(tzinfo=timezone.utc)
            if race_date < now:
                completed.append((year, int(event["RoundNumber"])))
        except Exception:
            continue

    if not completed:
        try:
            schedule = fastf1.get_event_schedule(year - 1, include_testing=False)
            for _, event in schedule.iterrows():
                completed.append((year - 1, int(event["RoundNumber"])))
        except Exception as e:
            print(f"[loader] Could not load {year - 1} schedule: {e}")
            return None

    if not completed:
        return None

    return completed[-1]


def get_completed_races(year: int) -> list[dict]:
    now = datetime.now(timezone.utc)
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
    except Exception as e:
        print(f"[loader] Could not load {year} schedule: {e}")
        return []

    races = []
    for _, event in schedule.iterrows():
        try:
            race_date = event["Session5DateUtc"]
            if pd.isna(race_date):
                continue
            if race_date.tzinfo is None:
                race_date = race_date.replace(tzinfo=timezone.utc)
            if race_date < now:
                races.append({
                    "year":     year,
                    "round":    int(event["RoundNumber"]),
                    "name":     str(event["EventName"]),
                    "country":  str(event["Country"]),
                    "location": str(event["Location"]),
                })
        except Exception:
            continue
    return races


def load_session(
    year: int,
    round_number: int,
    session_type: str = "R",
    weather: bool = True,
    messages: bool = True,
) -> fastf1.core.Session | None:
    try:
        session = fastf1.get_session(year, round_number, session_type)
        session.load(laps=True, telemetry=False, weather=weather, messages=messages)
        print(f"[loader] Loaded: {session.event['EventName']} {year} — {session.name}")
        return session
    except Exception as e:
        print(f"[loader] Failed to load session {year} R{round_number}: {e}")
        return None


def extract_driver_info(session: fastf1.core.Session) -> dict:
    drivers = {}
    for _, row in session.results.iterrows():
        abbr = row["Abbreviation"]
        drivers[abbr] = {
            "team": row["TeamName"],
            "team_colour": "#" + row["TeamColor"] if row["TeamColor"] and not str(row["TeamColor"]).startswith("#") else (row["TeamColor"] or "#ffffff"),
        }
    return drivers


def extract_laps(session: fastf1.core.Session) -> list[dict]:
    """
    Returns all laps as a list of dicts sorted by session time.
    Includes all laps — accurate and inaccurate — so the replay
    has full coverage. Inaccurate laps (e.g. out laps, in laps)
    are still useful for position and tyre tracking.
    """
    laps = session.laps.copy()

    # Drop rows with no session time at all — they can't be replayed
    laps = laps.dropna(subset=["Time"])

    records = []
    for _, lap in laps.iterrows():
        try:
            records.append({
                "driver":           lap["Driver"],
                "lap_number":       int(lap["LapNumber"]) if not pd.isna(lap["LapNumber"]) else None,
                "lap_time":         _fmt_timedelta(lap["LapTime"]),
                "session_time":     lap["Time"].total_seconds(),
                "lap_start_time":   lap["LapStartTime"].total_seconds() if not pd.isna(lap["LapStartTime"]) else None,
                "pit_in_time":      lap["PitInTime"].total_seconds() if not pd.isna(lap["PitInTime"]) else None,
                "pit_out_time":     lap["PitOutTime"].total_seconds() if not pd.isna(lap["PitOutTime"]) else None,
                "compound":         lap["Compound"] if not pd.isna(lap["Compound"]) else "UNKNOWN",
                "tyre_life":        int(lap["TyreLife"]) if not pd.isna(lap["TyreLife"]) else 0,
                "fresh_tyre":       bool(lap["FreshTyre"]) if not pd.isna(lap["FreshTyre"]) else False,
                "stint":            int(lap["Stint"]) if not pd.isna(lap["Stint"]) else 1,
                "sector_1":         _fmt_timedelta(lap["Sector1Time"]),
                "sector_2":         _fmt_timedelta(lap["Sector2Time"]),
                "sector_3":         _fmt_timedelta(lap["Sector3Time"]),
                "position":         int(lap["Position"]) if not pd.isna(lap["Position"]) else None,
                "is_personal_best": bool(lap["IsPersonalBest"]) if not pd.isna(lap["IsPersonalBest"]) else False,
                "track_status":     str(lap["TrackStatus"]) if not pd.isna(lap["TrackStatus"]) else "1",
                "is_accurate":      bool(lap["IsAccurate"]) if not pd.isna(lap["IsAccurate"]) else False,
            })
        except Exception:
            continue

    records.sort(key=lambda x: x["session_time"] or 0)
    return records


def extract_weather(session: fastf1.core.Session) -> list[dict]:
    snapshots = []
    for _, row in session.weather_data.iterrows():
        try:
            snapshots.append({
                "session_time": row["Time"].total_seconds(),
                "air_temp":     float(row["AirTemp"]),
                "track_temp":   float(row["TrackTemp"]),
                "humidity":     float(row["Humidity"]),
                "wind_speed":   float(row["WindSpeed"]),
                "rainfall":     bool(row["Rainfall"]),
            })
        except Exception:
            continue
    return snapshots


def extract_race_control(session: fastf1.core.Session) -> list[dict]:
    messages = []
    rc = session.race_control_messages.copy()

    # Time column may be wall-clock Timestamps or timedeltas depending on FastF1 version.
    # Normalise to session-relative seconds using the session start time.
    session_start = session.date
    if session_start is None:
        return messages

    # Make session_start timezone-aware if needed
    import pytz
    if hasattr(session_start, 'tzinfo') and session_start.tzinfo is None:
        session_start = session_start.replace(tzinfo=pytz.utc)

    for _, row in rc.iterrows():
        try:
            t = row["Time"]
            if pd.isna(t):
                session_time = 0.0
            elif hasattr(t, 'total_seconds'):
                # Already a timedelta
                session_time = t.total_seconds()
            else:
                # Wall clock Timestamp — convert to session-relative seconds
                if hasattr(t, 'tzinfo') and t.tzinfo is None:
                    t = t.replace(tzinfo=pytz.utc)
                session_time = (t - session_start).total_seconds()

            messages.append({
                "session_time": max(session_time, 0.0),
                "message":      str(row["Message"]),
                "flag":         str(row["Flag"]) if not pd.isna(row["Flag"]) else "",
                "lap":          int(row["Lap"]) if not pd.isna(row["Lap"]) else None,
            })
        except Exception:
            continue
    return messages


def extract_total_laps(session: fastf1.core.Session) -> int:
    try:
        return int(session.laps["LapNumber"].max())
    except Exception:
        return 0


def _fmt_timedelta(td) -> str | None:
    if pd.isna(td):
        return None
    try:
        total_seconds = td.total_seconds()
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60
        return f"{minutes}:{seconds:06.3f}"
    except Exception:
        return None


def extract_schedule(year: int) -> list[dict]:
    """
    Returns the full season schedule for a given year.
    Each event includes all session names and dates.
    """
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
    except Exception as e:
        print(f"[loader] Could not load schedule for {year}: {e}")
        return []

    events = []
    for _, event in schedule.iterrows():
        try:
            sessions = []
            for i in range(1, 6):
                name = event.get(f"Session{i}")
                date = event.get(f"Session{i}DateUtc")
                if name and not pd.isna(name) and date and not pd.isna(date):
                    sessions.append({
                        "name": str(name),
                        "date": str(date),
                    })

            events.append({
                "round":       int(event["RoundNumber"]),
                "name":        str(event["EventName"]),
                "country":     str(event["Country"]),
                "location":    str(event["Location"]),
                "date":        str(event["EventDate"]),
                "sessions":    sessions,
            })
        except Exception:
            continue

    return events
