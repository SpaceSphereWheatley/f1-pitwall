"""
Dev tool: replays a recorded raw F1 live-timing feed through LiveTimingClient's
message handlers, so the LIVE parsing -> update_state() path can be exercised and
visually verified without a real session running.

This bypasses LiveTimingClient's SignalRClient transport entirely (it dispatches
straight to the _handle_* methods), since the parsing logic — not the transport —
is what actually needs proving out against real message shapes.

Run as a standalone process (it touches the module-level state singleton in
state.py, so don't import this into the running app).

Producing a recording (uses FastF1's own working SignalRClient as a recorder,
independent of LiveTimingClient):

    python -m fastf1.livetiming save recordings/<session_name>.txt

Replaying it through the real handler logic:

    python live_playback.py recordings/<session_name>.txt [speed]
"""
import sys
import threading
from datetime import timedelta

from fastf1.livetiming.data import LiveTimingData

from live_timing import LiveTimingClient
from state import get_state

# Mirrors the .on(topic, handler) registrations in LiveTimingClient._connect
# (live_timing.py:107-115). Kept local rather than shared so this module stays
# fully decoupled from the SignalRClient transport it's meant to bypass.
TOPIC_HANDLERS = {
    "TimingData": "_handle_timing_data",
    "TimingAppData": "_handle_timing_app_data",
    "Position.z": "_handle_position",
    "RaceControlMessages": "_handle_race_control",
    "WeatherData": "_handle_weather",
    "TrackStatus": "_handle_track_status",
    "SessionInfo": "_handle_session_info",
    "LapCount": "_handle_lap_count",
    "DriverList": "_handle_driver_list",
}


class LiveTimingPlayback:
    """Feeds a recorded raw live-timing feed into a LiveTimingClient's message
    handlers in chronological order, imitating a live session at an accelerated
    pace."""

    def __init__(self, client: LiveTimingClient, filepath: str, speed: float = 20.0):
        self._client = client
        self._filepath = filepath
        self._speed = speed
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self, loop: bool = False):
        timeline = self._build_timeline()
        while not self._stop_event.is_set():
            self._replay(timeline)
            if not loop:
                return

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_timeline(self):
        data = LiveTimingData(self._filepath)
        timeline = []
        for category in data.list_categories():
            if category not in TOPIC_HANDLERS:
                continue
            for offset, payload in data.get(category):
                timeline.append((offset, category, payload))
        timeline.sort(key=lambda entry: entry[0])
        return timeline

    def _replay(self, timeline):
        previous_offset = timedelta(0)
        for offset, category, payload in timeline:
            if self._stop_event.is_set():
                return

            wait = (offset - previous_offset).total_seconds() / self._speed
            if wait > 0:
                self._stop_event.wait(wait)
            previous_offset = offset

            handler = getattr(self._client, TOPIC_HANDLERS[category])
            handler(payload)


# ----------------------------------------------------------------------
# Standalone runner — prints periodic state snapshots for visual verification
# ----------------------------------------------------------------------

def _print_snapshot():
    state = get_state()
    session = state.get("session", {})
    drivers = state.get("drivers", {})

    print(
        f"\n--- mode={state.get('mode')} session={session.get('name')!r} "
        f"lap={session.get('current_lap')}/{session.get('total_laps')} "
        f"track_status={session.get('status')} ---"
    )

    def sort_key(item):
        pos = item[1].get("position")
        return pos if isinstance(pos, int) and pos > 0 else 99

    for abbr, d in sorted(drivers.items(), key=sort_key):
        print(
            f"  P{d.get('position')} {abbr}  "
            f"gap={d.get('gap_to_leader')!r} interval={d.get('interval')!r} "
            f"last_lap={d.get('last_lap')!r} "
            f"sectors=({d.get('sector_1')!r}, {d.get('sector_2')!r}, {d.get('sector_3')!r}) "
            f"compound={d.get('compound')} tyre_age={d.get('tyre_age')} "
            f"in_pit={d.get('in_pit')} pit_stops={d.get('pit_stops')}"
        )


def main():
    if len(sys.argv) < 2:
        print("Usage: python live_playback.py <recorded_feed_file> [speed]")
        sys.exit(1)

    filepath = sys.argv[1]
    speed = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0

    client = LiveTimingClient()
    playback = LiveTimingPlayback(client, filepath, speed=speed)

    thread = threading.Thread(target=playback.run, daemon=True)
    thread.start()

    try:
        while thread.is_alive():
            _print_snapshot()
            thread.join(timeout=5)
    except KeyboardInterrupt:
        playback.stop()
        thread.join()

    _print_snapshot()
    print("\nPlayback finished.")


if __name__ == "__main__":
    main()
