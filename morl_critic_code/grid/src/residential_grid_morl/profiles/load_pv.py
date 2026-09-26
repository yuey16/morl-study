"""Load processed paired Ausgrid household load/PV shapes."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from residential_grid_morl.utils import PROJECT_ROOT


def load_profile_tables(directory: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(directory) if directory is not None else PROJECT_ROOT / "data/processed/ausgrid"
    load = pd.read_parquet(root / "load_pu.parquet")
    pv = pd.read_parquet(root / "pv_pu.parquet")
    load.index = pd.DatetimeIndex(load.index)
    pv.index = pd.DatetimeIndex(pv.index)
    if not load.index.equals(pv.index) or list(load.columns) != list(pv.columns):
        raise ValueError("Processed Ausgrid load/PV tables are not paired")
    return load.astype("float32"), pv.astype("float32")

