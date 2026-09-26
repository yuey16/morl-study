"""Load processed AEMO price profiles."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from residential_grid_morl.utils import PROJECT_ROOT


def load_price_table(path: str | Path | None = None) -> pd.DataFrame:
    source = Path(path) if path is not None else PROJECT_ROOT / "data/processed/price/price.parquet"
    prices = pd.read_parquet(source)
    prices.index = pd.DatetimeIndex(prices.index)
    required = {"spot_aud_per_mwh", "spot_aud_per_kwh", "retail_aud_per_kwh"}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(f"Processed price table missing columns: {sorted(missing)}")
    return prices

