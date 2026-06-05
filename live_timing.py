import logging
import threading
import time
from datetime import datetime, timezone

import fastf1
from fastf1.livetiming.client import SignalRClient

from state import update_state, get_state

logger = logging.getLogger(__name__)


# Map F1 compound codes to display names
COMPOUND_MAP = {
    "SOFT": "S",
    "MEDIUM": "M",
    "HARD": "H",
    "INTERMEDIATE": "I",
    "WET": "W",
    "UNKNOWN": "?",
    "TEST_UNKNOWN": "?",
}

# Map F1 track status codes to display strings
TRACK_STATUS_MAP = {
    "1": "AllClear",
    "2": "Yellow",
    "3": "SCDeployed",
    "4": "SCDeployed",
    "5": "RedFlag",
    "6": "VSCDeployed",
    "7": "VSCEnding",
}


class LiveTimingClient:
    """
    Wraps FastF1 SignalRClient to stream live F1 timing data
    and write it into the shared state dict.
    """

    def __init__(self):
        self._client = None
        self._thread = None
        self._stop_event = threading.Event()
        self._running = False
        self._session_start_time = None

        # Per-driver working data (built up incrementally from live messages)
        self._drivers = {}
        self._position_data = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self):
        """Start the live timing client in a background thread."""
        if self._running:
            logger.warning("LiveTimingClient already running")
            return

        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="live-timing")
        self._thread.start()
        logger.info("LiveTimingClient started")

    def stop(self):
        """Stop the live timing client."""
        self._stop_event.set()
        self._running = False
        if self._client:
            try:
                self._client.stop()
            except Exception:
                pass
        logger.info("LiveTimingClient stopped")

    @property
    def is_running(self):
        return self._running

    # ------------------------------------------------------------------
    # Internal: run loop with reconnection
    # ------------------------------------------------------------------

    def _run(self):
        while not self._stop_event.is_set():
            try:
                logger.info("Connecting to F1 live timing stream...")
                self._connect()
            except Exception as e:
                if self._stop_event.is_set():
                    break
                logger.error(f"Live timing connection error: {e}. Reconnecting in 10s...")
                time.sleep(10)

        self._running = False
        logger.info("LiveTimingClient run loop exited")

    def _connect(self):
        self._client = SignalRClient(filename=None, verbose=False)

        # Register message handlers
        self._client.on("TimingData", self._handle_timing_data)
        self._client.on("TimingAppData", self._handle_timing_app_data)
        self._client.on("Position.z", self._handle_position)
        self._client.on("RaceControlMessages", self._handle_race_control)
        self._client.on("WeatherData", self._handle_weather)
        self._client.on("TrackStatus", self._handle_track_status)
        self._client.on("SessionInfo", self._handle_session_info)
        self._client.on("LapCount", self._handle_lap_count)
        self._client.on("DriverList", self._handle_driver_list)

        self._client.start()

    # ------------------------------------------------------------------
    # Message handlers
    # ------------------------------------------------------------------

    def _handle_session_info(self, data):
        try:
            meeting = data.get("Meeting", {})
            update_state({
                "session": {
                    "name": data.get("Name", ""),
                    "type": data.get("Type", ""),
                    "circuit": meeting.get("Circuit", {}).get("ShortName", ""),
                    "country": meeting.get("Country", {}).get("Name", ""),
                    "round": meeting.get("Number", 0),
                    "year": datetime.now(timezone.utc).year,
                    "status": "Live",
                    "total_laps": data.get("TotalLaps", 0),
                    "current_lap": 0,
                },
                "mode": "LIVE",
            })
            logger.info(f"Session info received: {data.get('Name', '')}")
        except Exception as e:
            logger.error(f"Error handling SessionInfo: {e}")

    def _handle_lap_count(self, data):
        try:
            current = data.get("CurrentLap")
            total = data.get("TotalLaps")
            state = get_state()
            session = dict(state.get("session", {}))
            if current is not None:
                session["current_lap"] = int(current)
            if total is not None:
                session["total_laps"] = int(total)
            update_state({"session": session})
        except Exception as e:
            logger.error(f"Error handling LapCount: {e}")

    def _handle_driver_list(self, data):
        try:
            state = get_state()
            drivers = dict(state.get("drivers", {}))
            for number, info in data.items():
                if not isinstance(info, dict):
                    continue
                abbr = info.get("Tla", str(number))
                team = info.get("TeamName", "")
                colour = "#" + info.get("TeamColour", "FFFFFF") if info.get("TeamColour") else "#FFFFFF"
                full_name = info.get("FullName", abbr)
                if abbr not in drivers:
                    drivers[abbr] = {
                        "driver": abbr,
                        "full_name": full_name,
                        "team": team,
                        "team_colour": colour,
                        "position": 0,
                        "gap_to_leader": "",
                        "interval": "",
                        "last_lap": "",
                        "best_lap": "",
                        "sector_1": "",
                        "sector_2": "",
                        "sector_3": "",
                        "compound": "?",
                        "tyre_age": 0,
                        "pit_stops": 0,
                        "in_pit": False,
                        "lap_fraction": 0.0,
                        "number": str(number),
                    }
                else:
                    drivers[abbr]["team"] = team
                    drivers[abbr]["team_colour"] = colour
                    drivers[abbr]["full_name"] = full_name

                self._drivers[str(number)] = abbr

            update_state({"drivers": drivers})
        except Exception as e:
            logger.error(f"Error handling DriverList: {e}")

    def _handle_timing_data(self, data):
        try:
            lines = data.get("Lines", {})
            state = get_state()
            drivers = dict(state.get("drivers", {}))

            for number, info in lines.items():
                if not isinstance(info, dict):
                    continue
                abbr = self._drivers.get(str(number))
                if not abbr or abbr not in drivers:
                    continue

                d = dict(drivers[abbr])

                pos = info.get("Position")
                if pos is not None:
                    try:
                        d["position"] = int(pos)
                    except (ValueError, TypeError):
                        pass

                gap = info.get("GapToLeader")
                if gap is not None:
                    d["gap_to_leader"] = str(gap)

                interval = info.get("IntervalToPositionAhead", {})
                if isinstance(interval, dict):
                    val = interval.get("Value")
                    if val is not None:
                        d["interval"] = str(val)
                elif interval is not None:
                    d["interval"] = str(interval)

                last_lap = info.get("LastLapTime", {})
                if isinstance(last_lap, dict):
                    val = last_lap.get("Value")
                    if val:
                        d["last_lap"] = str(val)

                best_lap = info.get("BestLapTime", {})
                if isinstance(best_lap, dict):
                    val = best_lap.get("Value")
                    if val:
                        d["best_lap"] = str(val)

                sectors = info.get("Sectors", {})
                if isinstance(sectors, dict):
                    for sector_key, state_key in [("0", "sector_1"), ("1", "sector_2"), ("2", "sector_3")]:
                        s = sectors.get(sector_key, {})
                        if isinstance(s, dict):
                            val = s.get("Value")
                            if val is not None:
                                d[state_key] = str(val)

                in_pit = info.get("InPit")
                if in_pit is not None:
                    d["in_pit"] = bool(in_pit)

                pit_stops = info.get("NumberOfPitStops")
                if pit_stops is not None:
                    try:
                        d["pit_stops"] = int(pit_stops)
                    except (ValueError, TypeError):
                        pass

                drivers[abbr] = d

            update_state({"drivers": drivers})
        except Exception as e:
            logger.error(f"Error handling TimingData: {e}")

    def _handle_timing_app_data(self, data):
        try:
            lines = data.get("Lines", {})
            state = get_state()
            drivers = dict(state.get("drivers", {}))

            for number, info in lines.items():
                if not isinstance(info, dict):
                    continue
                abbr = self._drivers.get(str(number))
                if not abbr or abbr not in drivers:
                    continue

                d = dict(drivers[abbr])

                stints = info.get("Stints", {})
                if isinstance(stints, dict) and stints:
                    # Get the last stint (highest key)
                    latest_key = max(stints.keys(), key=lambda x: int(x) if x.isdigit() else 0)
                    stint = stints[latest_key]
                    if isinstance(stint, dict):
                        compound = stint.get("Compound", "")
                        if compound:
                            d["compound"] = COMPOUND_MAP.get(compound.upper(), "?")
                        age = stint.get("TyreAge")
                        if age is not None:
                            try:
                                d["tyre_age"] = int(age)
                            except (ValueError, TypeError):
                                pass

                drivers[abbr] = d

            update_state({"drivers": drivers})
        except Exception as e:
            logger.error(f"Error handling TimingAppData: {e}")

    def _handle_position(self, data):
        try:
            entries = data.get("Position", [])
            if not entries:
                return

            state = get_state()
            drivers = dict(state.get("drivers", {}))

            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                entries_inner = entry.get("Entries", {})
                for number, pos_data in entries_inner.items():
                    if not isinstance(pos_data, dict):
                        continue
                    abbr = self._drivers.get(str(number))
                    if not abbr or abbr not in drivers:
                        continue
                    # Position data gives X/Y/Z — we don't use it directly
                    # lap_fraction is computed by the replay engine; for live
                    # timing we leave it at 0.0 (circle map will be approximate)
                    # A future enhancement could compute from X/Y coordinates

            update_state({"drivers": drivers})
        except Exception as e:
            logger.error(f"Error handling Position: {e}")

    def _handle_race_control(self, data):
        try:
            messages = data.get("Messages", {})
            if not messages:
                return

            state = get_state()
            existing = list(state.get("race_control", []))

            for key, msg in messages.items():
                if not isinstance(msg, dict):
                    continue
                utc_str = msg.get("Utc", "")
                text = msg.get("Message", "")
                flag = msg.get("Flag", "")
                lap = msg.get("Lap", 0)

                entry = {
                    "time": utc_str,
                    "lap": lap,
                    "flag": flag,
                    "message": text,
                }
                if entry not in existing:
                    existing.append(entry)

            # Keep last 50 messages
            existing = existing[-50:]
            update_state({"race_control": existing})
        except Exception as e:
            logger.error(f"Error handling RaceControlMessages: {e}")

    def _handle_weather(self, data):
        try:
            update_state({
                "weather": {
                    "air_temp": float(data.get("AirTemp", 0)),
                    "track_temp": float(data.get("TrackTemp", 0)),
                    "humidity": float(data.get("Humidity", 0)),
                    "wind_speed": float(data.get("WindSpeed", 0)),
                    "wind_direction": float(data.get("WindDirection", 0)),
                    "rainfall": data.get("Rainfall", "0") not in ("0", "", False, None),
                }
            })
        except Exception as e:
            logger.error(f"Error handling WeatherData: {e}")

    def _handle_track_status(self, data):
        try:
            code = str(data.get("Status", "1"))
            status = TRACK_STATUS_MAP.get(code, "AllClear")
            state = get_state()
            session = dict(state.get("session", {}))
            session["status"] = status
            update_state({"session": session})
        except Exception as e:
            logger.error(f"Error handling TrackStatus: {e}")
