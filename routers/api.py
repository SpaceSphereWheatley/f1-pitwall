import asyncio
import os
import time
from datetime import datetime, timezone

import fastf1
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from auth import get_current_user, require_admin
from state import get_state
from fastf1_loader import extract_schedule, get_completed_races

router = APIRouter()

_start_time = time.time()
_schedule_cache: list[dict] = []
_schedule_year: int | None = None

_session_manager = None

def set_session_manager(sm):
    global _session_manager
    _session_manager = sm


def _get_schedule() -> list[dict]:
    global _schedule_cache, _schedule_year
    year = datetime.now(timezone.utc).year
    if _schedule_year == year and _schedule_cache:
        return _schedule_cache
    print(f"[api] Loading schedule for {year}")
    _schedule_cache = extract_schedule(year)
    _schedule_year = year
    return _schedule_cache


def _cache_size_mb(cache_dir: str = "f1_cache") -> float:
    total = 0
    if os.path.isdir(cache_dir):
        for dirpath, _, filenames in os.walk(cache_dir):
            for f in filenames:
                try:
                    total += os.path.getsize(os.path.join(dirpath, f))
                except OSError:
                    pass
    return round(total / (1024 * 1024), 2)


@router.get("/state")
async def get_state_endpoint(current_user=Depends(get_current_user)):
    state = get_state()
    if state.get("mode") == "IDLE" and not state.get("drivers"):
        raise HTTPException(
            status_code=503,
            detail="Session data not yet available. Server is still loading.",
        )
    return JSONResponse(content=state)


@router.get("/schedule")
async def get_schedule(current_user=Depends(get_current_user)):
    schedule = _get_schedule()
    return {"schedule": schedule}


@router.get("/health")
async def health():
    state = get_state()
    uptime_seconds = int(time.time() - _start_time)
    hours, remainder = divmod(uptime_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return {
        "status": "ok",
        "mode": state.get("mode", "IDLE"),
        "uptime": f"{hours:02}:{minutes:02}:{seconds:02}",
        "uptime_seconds": uptime_seconds,
        "last_update": datetime.now(timezone.utc).isoformat(),
        "cache_size_mb": _cache_size_mb(),
    }


@router.get("/admin/fetch-pit-duration")
async def fetch_pit_duration(request: Request, current_user=Depends(require_admin)):
    """Use Anthropic API with web search to find the pit lane loss time for the current round."""
    import os
    import anthropic
    from dotenv import load_dotenv
    load_dotenv()

    state = get_state()
    session_info = state.get("session", {})
    year = session_info.get("year")
    circuit = session_info.get("circuit")

    if not year or not circuit:
        raise HTTPException(status_code=400, detail="No active session found.")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set in .env")

    client = anthropic.Anthropic(api_key=api_key)

    prompt = (
        f'Search for exactly this: "F1 {circuit} {year} tyre strategy pit lane loss seconds". '
        f"Do a single search, read the first relevant result, and extract the pit lane loss time in seconds. "
        f"It is always written as a number followed by the word 'second' near the words 'pit lane loss'. "
        f"Do not do more than one search. Do not verify. "
        f"Return ONLY a JSON object, no other text: "
        f'{{"pit_loss_seconds": 24, "source": "site name"}}'
    )

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=256,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": prompt}]
        )

        # Extract text from response
        result_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                result_text += block.text

        # Parse JSON from response
        import json
        import re
        match = re.search(r'\{.*?\}', result_text, re.DOTALL)
        if not match:
            raise HTTPException(status_code=500, detail=f"Could not parse response: {result_text}")

        data = json.loads(match.group())
        pit_loss = int(data.get("pit_loss_seconds", 0))
        source = data.get("source", "Pirelli strategy notes")

        if pit_loss < 10 or pit_loss > 60:
            raise HTTPException(status_code=500, detail=f"Implausible value returned: {pit_loss}s. Check manually.")

        return {
            "estimated_pit_duration": pit_loss,
            "source": source,
            "note": f"From Pirelli strategy notes: {pit_loss}s pit lane loss ({source}). Adjust if needed."
        }

    except anthropic.APIError as e:
        raise HTTPException(status_code=500, detail=f"Anthropic API error: {str(e)}")


@router.get("/admin/calculate-pit-duration")
async def calculate_pit_duration(request: Request, current_user=Depends(require_admin)):
    """Load FP1/FP2 for the current round and estimate pit lane travel time."""
    state = get_state()
    session_info = state.get("session", {})
    year = session_info.get("year")
    round_number = session_info.get("round")

    if not year or not round_number:
        raise HTTPException(status_code=400, detail="No active session found.")

    errors = []
    min_pit_time = None

    for session_type in ["FP1", "FP2", "FP3"]:
        try:
            import pandas as pd
            session = fastf1.get_session(year, round_number, session_type)
            session.load(laps=True, telemetry=False, weather=False, messages=False)
            laps = session.laps

            # Use consecutive lap approach: PitOutTime of next lap - PitInTime of current lap
            pit_durations = []
            for driver in laps["Driver"].unique():
                drv_laps = laps[laps["Driver"] == driver].sort_values("LapNumber").reset_index(drop=True)
                for i in range(len(drv_laps) - 1):
                    pit_in  = drv_laps.iloc[i]["PitInTime"]
                    pit_out = drv_laps.iloc[i+1]["PitOutTime"]
                    if pd.isna(pit_in) or pd.isna(pit_out):
                        continue
                    duration = (pit_out - pit_in).total_seconds()
                    # Only keep realistic pit stops (not garage sits)
                    # Min 20s excludes drive-throughs / no-work pit entries
                    if 20 < duration < 60:
                        pit_durations.append(duration)

            if not pit_durations:
                continue
            session_min = min(pit_durations)
            if min_pit_time is None or session_min < min_pit_time:
                min_pit_time = session_min
        except Exception as e:
            errors.append(f"{session_type}: {str(e)}")
            continue

    if min_pit_time is None:
        raise HTTPException(status_code=404, detail=f"No pit data found. {' '.join(errors)}")

    estimated = max(15, round(min_pit_time))

    return {
        "estimated_pit_duration": int(estimated),
        "min_raw_pit_time": round(min_pit_time, 1),
        "note": f"Based on fastest pit stop in practice (raw: {min_pit_time:.1f}s). Adjust if needed."
    }


@router.get("/admin/races")
async def get_races_for_year(year: int, current_user=Depends(require_admin)):
    current_year = datetime.now(timezone.utc).year
    if year < current_year - 5 or year > current_year:
        raise HTTPException(status_code=400, detail=f"Year must be between {current_year - 5} and {current_year}")
    loop = asyncio.get_event_loop()
    races = await loop.run_in_executor(None, get_completed_races, year)
    return {"races": races}


@router.post("/replay/load")
async def load_replay_race(year: int, round: int, current_user=Depends(require_admin)):
    if _session_manager is None:
        raise HTTPException(status_code=503, detail="Session manager not ready")
    state = get_state()
    if state.get("mode") == "LIVE":
        raise HTTPException(status_code=400, detail="Cannot switch race during a live session")
    current_year = datetime.now(timezone.utc).year
    if year < current_year - 5 or year > current_year:
        raise HTTPException(status_code=400, detail=f"Year must be between {current_year - 5} and {current_year}")
    if round < 1 or round > 30:
        raise HTTPException(status_code=400, detail="Round must be between 1 and 30")
    ok = _session_manager.load_race(year, round)
    if not ok:
        raise HTTPException(status_code=400, detail="Cannot switch race during a live session")
    return {"ok": True, "message": f"Loading {year} Round {round}…"}


@router.post("/replay/seek")
async def replay_seek(
    lap: int,
    current_user=Depends(require_admin),
):
    if _session_manager is None:
        raise HTTPException(status_code=503, detail="Session manager not ready")

    state = get_state()
    if state.get("mode") != "REPLAY":
        raise HTTPException(status_code=400, detail="Only available in REPLAY mode")

    total_laps = state.get("session", {}).get("total_laps") or 0
    if lap < 1 or (total_laps and lap > total_laps):
        raise HTTPException(status_code=400, detail=f"Lap must be between 1 and {total_laps}")

    success = _session_manager.seek(lap)
    if not success:
        raise HTTPException(status_code=404, detail=f"Lap {lap} not found in session data")

    return {"ok": True, "sought_to_lap": lap}
