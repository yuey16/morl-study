"""Load processed Electric Nation charging-session reconstructions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from residential_grid_morl.utils import PROJECT_ROOT


def load_ev_sessions(path: str | Path | None = None) -> pd.DataFrame:
    source = Path(path) if path is not None else PROJECT_ROOT / "data/processed/electric_nation/sessions.parquet"
    sessions = pd.read_parquet(source)
    for column in ("arrival_time", "departure_time", "arrival_time_utc", "departure_time_utc"):
        if column in sessions:
            sessions[column] = pd.to_datetime(sessions[column])
    required = {"session_id", "charger_id", "arrival_time", "departure_time", "energy_kwh", "dwell_hours"}
    missing = required.difference(sessions.columns)
    if missing:
        raise ValueError(f"Processed EV sessions missing columns: {sorted(missing)}")
    return sessions

