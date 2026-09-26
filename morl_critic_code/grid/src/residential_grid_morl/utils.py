"""Shared configuration, path, serialization, and reproducibility helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(path: str | Path, base_path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config and optionally layer it over a base config."""
    config = load_yaml(path)
    if base_path is None:
        inherited = config.pop("extends", None)
        if inherited:
            inherited_path = Path(path).parent / inherited
            return deep_merge(load_config(inherited_path), config)
        return config
    return deep_merge(load_yaml(base_path), config)


def dump_json(payload: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def sha256_file(path: str | Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def chronological_split(items: list[str], fractions: tuple[float, float, float] = (0.70, 0.15, 0.15)) -> dict[str, list[str]]:
    """Split sorted day identifiers without leaking a day across partitions."""
    if not items:
        raise ValueError("Cannot split an empty sequence")
    ordered = sorted(dict.fromkeys(items))
    n = len(ordered)
    train_end = max(1, int(n * fractions[0]))
    validation_end = max(train_end + 1, int(n * (fractions[0] + fractions[1])))
    validation_end = min(validation_end, n - 1) if n >= 3 else min(validation_end, n)
    return {
        "train": ordered[:train_end],
        "validation": ordered[train_end:validation_end],
        "test": ordered[validation_end:],
    }

