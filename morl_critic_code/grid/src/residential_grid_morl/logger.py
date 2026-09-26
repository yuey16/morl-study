"""Trajectory logging contract used by validation and paper plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class EpisodeLogger:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.records.clear()

    def append(self, record: dict[str, Any]) -> None:
        self.records.append(record)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(self.records)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.frame().to_parquet(target, index=False, compression="zstd")
        return target
