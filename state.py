import threading

_lock = threading.Lock()

_state = {
    "mode": "IDLE",
    "session": {
        "name": None,
        "circuit": None,
        "country": None,
        "round": None,
        "year": None,
        "simulated_time": None,
        "total_laps": None,
        "current_lap": None,
        "prediction_model": None,
    },
    "weather": {
        "air_temp": None,
        "track_temp": None,
        "humidity": None,
        "wind_speed": None,
        "rainfall": False,
    },
    "track_status": "1",
    "drivers": {},
    "race_control": [],
    "schedule": [],
}


def get_state() -> dict:
    with _lock:
        import copy
        return copy.deepcopy(_state)


def update_state(patch: dict) -> None:
    with _lock:
        _deep_merge(_state, patch)


def reset_state() -> None:
    with _lock:
        _state["mode"] = "IDLE"
        _state["session"] = {
            "name": None,
            "circuit": None,
            "country": None,
            "round": None,
            "year": None,
            "simulated_time": None,
            "total_laps": None,
            "current_lap": None,
            "prediction_model": None,
        }
        _state["weather"] = {
            "air_temp": None,
            "track_temp": None,
            "humidity": None,
            "wind_speed": None,
            "rainfall": False,
        }
        _state["track_status"] = "1"
        _state["drivers"] = {}
        _state["race_control"] = []


def _deep_merge(base: dict, patch: dict) -> None:
    for key, value in patch.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = value
