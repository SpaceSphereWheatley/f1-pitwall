import threading
import time
import os
import requests
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

from fastf1_loader import enable_cache, get_last_completed_race, load_session
from replay_engine import ReplayEngine
from live_timing import LiveTimingClient
from state import update_state, get_state
from ml.predict import load_models, predict_prerace, is_low_confidence

load_dotenv()

CACHE_DIR    = os.getenv("CACHE_DIR", "f1_cache")
REPLAY_SPEED = float(os.getenv("REPLAY_SPEED", "1.0"))

# How often to poll Jolpica for live session detection (seconds)
LIVE_POLL_INTERVAL = 60

# Buffer around session start/end times (minutes)
SESSION_START_BUFFER = 10
SESSION_END_BUFFER   = 10

# Session durations in minutes (used when no end time is available)
SESSION_DURATION_MAP = {
    "Practice 1":   60,
    "Practice 2":   60,
    "Practice 3":   60,
    "Sprint Shootout": 60,
    "Sprint":       30,
    "Qualifying":   60,
    "Race":        120,
}


def _fetch_jolpica_schedule(year: int) -> list[dict]:
    """
    Fetch race schedule from Jolpica API.
    Returns list of sessions with start/end UTC datetimes.
    """
    url = f"https://api.jolpi.ca/ergast/f1/{year}.json"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        races = data["MRData"]["RaceTable"]["Races"]
    except Exception as e:
        print(f"[session] Jolpica fetch failed: {e}")
        return []

    sessions = []
    for race in races:
        # Map of session name → date/time keys in Jolpica response
        session_keys = [
            ("Practice 1",      "FirstPractice"),
            ("Practice 2",      "SecondPractice"),
            ("Practice 3",      "ThirdPractice"),
            ("Sprint Shootout", "SprintQualifying"),
            ("Sprint",          "Sprint"),
            ("Qualifying",      "Qualifying"),
            ("Race",            "race"),
        ]

        for session_name, key in session_keys:
            if key == "race":
                date_str = race.get("date")
                time_str = race.get("time", "00:00:00Z")
            else:
                block = race.get(key)
                if not block:
                    continue
                date_str = block.get("date")
                time_str = block.get("time", "00:00:00Z")

            if not date_str:
                continue

            try:
                dt_str = f"{date_str}T{time_str}"
                if not dt_str.endswith("Z"):
                    dt_str += "Z"
                start = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                duration = SESSION_DURATION_MAP.get(session_name, 60)
                end = start + timedelta(minutes=duration)

                sessions.append({
                    "round":        int(race.get("round", 0)),
                    "name":         session_name,
                    "circuit":      race.get("Circuit", {}).get("circuitName", ""),
                    "country":      race.get("Circuit", {}).get("Location", {}).get("country", ""),
                    "start":        start,
                    "end":          end,
                })
            except Exception:
                continue

    return sessions


def _find_active_session(sessions: list[dict]) -> dict | None:
    """
    Returns the session that is currently active (within buffer window),
    or None if no session is active.
    """
    now = datetime.now(timezone.utc)
    for s in sessions:
        window_start = s["start"] - timedelta(minutes=SESSION_START_BUFFER)
        window_end   = s["end"]   + timedelta(minutes=SESSION_END_BUFFER)
        if window_start <= now <= window_end:
            return s
    return None


class SessionManager:
    def __init__(self):
        self._stop_event    = threading.Event()
        self._replay_stop   = threading.Event()
        self._thread        = None
        self._replay_thread = None
        self._live_client: LiveTimingClient | None = None
        self.mode           = "IDLE"
        self.engine: ReplayEngine | None = None
        self._schedule_cache: list[dict] = []
        self._schedule_fetched_at: datetime | None = None
        self._last_live_session: dict | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self):
        enable_cache(CACHE_DIR)
        load_models()
        self._thread = threading.Thread(target=self._run, daemon=True, name="session-manager")
        self._thread.start()
        print("[session] Session manager started")

    def stop(self):
        self._stop_event.set()
        self._replay_stop.set()
        self._stop_live()
        print("[session] Session manager stopped")

    def seek(self, lap_number: int) -> bool:
        if self.engine is None:
            return False
        return self.engine.seek(lap_number)

    def load_race(self, year: int, round_number: int) -> bool:
        """Stop the current replay and start loading a different race in the background.
        Returns False if a live session is active. Returns immediately — app goes IDLE
        while FastF1 loads, then auto-transitions to REPLAY."""
        if self.mode == "LIVE":
            return False
        threading.Thread(
            target=self._load_race_bg,
            args=(year, round_number),
            daemon=True,
            name="race-loader",
        ).start()
        return True

    def _load_race_bg(self, year: int, round_number: int) -> None:
        self._stop_replay()
        update_state({"mode": "IDLE"})
        self.mode = "IDLE"
        print(f"[session] Loading {year} R{round_number} in background...")
        session = load_session(year, round_number)
        if session is None:
            print(f"[session] Failed to load {year} R{round_number} — staying IDLE")
            return
        if self.mode == "LIVE":
            print("[session] Live session started during race load — aborting replay")
            return
        self._start_replay(session)

    # ------------------------------------------------------------------
    # Internal: main run loop
    # ------------------------------------------------------------------

    def _run(self):
        update_state({"mode": "IDLE"})

        # Initial replay load
        self._load_and_start_replay()

        # Poll loop — check for live session every 60 seconds
        while not self._stop_event.is_set():
            self._check_for_live_session()
            self._stop_event.wait(LIVE_POLL_INTERVAL)

    def _load_and_start_replay(self):
        result = get_last_completed_race()
        if result is None:
            print("[session] No completed race found — staying IDLE")
            return

        year, round_number = result
        print(f"[session] Loading last completed race: {year} R{round_number}")

        session = load_session(year, round_number)
        if session is None:
            print("[session] Session load failed — staying IDLE")
            return

        self._start_replay(session)

    # ------------------------------------------------------------------
    # Internal: live session detection
    # ------------------------------------------------------------------

    def _get_schedule(self) -> list[dict]:
        """Return cached schedule, refreshing if older than 6 hours."""
        now = datetime.now(timezone.utc)
        if (
            not self._schedule_cache
            or self._schedule_fetched_at is None
            or (now - self._schedule_fetched_at).total_seconds() > 6 * 3600
        ):
            year = now.year
            print(f"[session] Fetching {year} schedule from Jolpica...")
            self._schedule_cache = _fetch_jolpica_schedule(year)
            self._schedule_fetched_at = now
            print(f"[session] Fetched {len(self._schedule_cache)} session windows")
        return self._schedule_cache

    def _check_for_live_session(self):
        try:
            schedule = self._get_schedule()
            active = _find_active_session(schedule)

            if active and self.mode != "LIVE":
                print(f"[session] Live session detected: {active['name']} — switching to LIVE")
                self._switch_to_live(active)

            elif not active and self.mode == "LIVE":
                print("[session] Live session ended — switching to REPLAY")
                self._switch_to_replay_after_live()

        except Exception as e:
            print(f"[session] Error in live session check: {e}")

    # ------------------------------------------------------------------
    # Internal: mode switching
    # ------------------------------------------------------------------

    def _switch_to_live(self, session_info: dict):
        self._last_live_session = session_info
        # Stop replay engine
        self._stop_replay()

        # Update state with session info and LIVE mode
        state = get_state()
        current_session = dict(state.get("session", {}))
        current_session.update({
            "name":    session_info["name"],
            "circuit": session_info["circuit"],
            "country": session_info["country"],
            "round":   session_info["round"],
            "status":  "Live",
        })
        update_state({"mode": "LIVE", "session": current_session})
        self.mode = "LIVE"

        # Start live timing client
        self._live_client = LiveTimingClient()
        self._live_client.start()
        print("[session] LiveTimingClient started")

    def _switch_to_replay_after_live(self):
        # Stop live client
        self._stop_live()

        # Brief IDLE state while we load FastF1 data
        update_state({"mode": "IDLE"})
        self.mode = "IDLE"
        print("[session] IDLE — loading completed session via FastF1...")

        # Give FastF1 a moment to process the completed session
        time.sleep(30)

        session_type = (self._last_live_session or {}).get("name", "")
        session_round = (self._last_live_session or {}).get("round")
        session_year = datetime.now(timezone.utc).year

        if session_type == "Qualifying" and session_round:
            threading.Thread(
                target=self._run_prerace_predictions,
                args=(session_year, session_round),
                daemon=True,
                name="ml-predict-prerace",
            ).start()
        elif session_type == "Race" and session_round:
            threading.Thread(
                target=self._run_training,
                args=(session_year, session_round),
                daemon=True,
                name="ml-train",
            ).start()

        # Load the just-completed session
        self._load_and_start_replay()

    def _run_prerace_predictions(self, year: int, round_num: int) -> None:
        print(f"[session] Running PRE-RACE predictions for {year} R{round_num}...")
        try:
            results = predict_prerace(year, round_num)
            if results is None:
                print("[session] PRE-RACE predictions unavailable (no model).")
                return
            label = "PRE-RACE (LOW CONFIDENCE)" if is_low_confidence() else "PRE-RACE"
            drivers_patch = {
                r["driver"]: {
                    "predicted_finish": r["predicted_position"],
                    "grid_position": r.get("grid_position"),
                }
                for r in results
            }
            update_state({"session": {"prediction_model": label}, "drivers": drivers_patch})
            print(f"[session] PRE-RACE predictions stored. Label: {label}")
        except Exception as e:
            print(f"[session] PRE-RACE prediction failed: {e}")

    def _run_training(self, year: int, round_num: int) -> None:
        print(f"[session] Triggering ML training after race {year} R{round_num}...")
        from ml.train import train_prerace, train_live

        def safe(fn, *args):
            try:
                fn(*args)
            except Exception as e:
                print(f"[session] {fn.__name__} failed: {e}")

        t1 = threading.Thread(target=safe, args=(train_prerace, year, round_num + 1), daemon=True)
        t2 = threading.Thread(target=safe, args=(train_live, year, round_num + 1), daemon=True)
        t1.start(); t2.start()
        t1.join(); t2.join()
        load_models()

    def _start_replay(self, session):
        self._replay_stop.clear()
        self.mode = "REPLAY"

        self.engine = ReplayEngine(session, speed=REPLAY_SPEED)

        self._replay_thread = threading.Thread(
            target=self.engine.run,
            args=(self._replay_stop,),
            daemon=True,
            name="replay-engine",
        )
        self._replay_thread.start()
        print("[session] Replay engine running")

    def _stop_replay(self):
        self._replay_stop.set()
        if self._replay_thread and self._replay_thread.is_alive():
            self._replay_thread.join(timeout=5)
        self._replay_stop.clear()
        self.engine = None
        print("[session] Replay engine stopped")

    def _stop_live(self):
        if self._live_client and self._live_client.is_running:
            self._live_client.stop()
            self._live_client = None
            print("[session] LiveTimingClient stopped")
