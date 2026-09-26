"""Canonical IEEE feeder loading and immutable base-load helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandapower as pp
import pandapower.networks as pn

from .utils import PROJECT_ROOT


def load_network(name: str, network_dir: Path | None = None):
    """Return a fresh canonical feeder with frozen base-load columns."""
    normalized = name.lower().replace("-", "").replace("_", "")
    if normalized in {"ieee33", "33", "case33bw"}:
        net = pn.case33bw()
        # case33bw indexes are zero based; labels in the project configuration
        # use the conventional one-based IEEE bus numbers.
        net.bus["name"] = [str(index + 1) for index in net.bus.index]
    elif normalized in {"ieee69", "69", "case69"}:
        directory = network_dir or PROJECT_ROOT / "data/networks"
        path = directory / "ieee69.json"
        if not path.exists():
            raise FileNotFoundError(f"Build {path} first with scripts/build_ieee69.py")
        net = pp.from_json(path)
    elif normalized in {"ieee123", "123", "case123"}:
        directory = network_dir or PROJECT_ROOT / "data/networks"
        path = directory / "ieee123.json"
        if not path.exists():
            raise FileNotFoundError(f"Build {path} first with scripts/build_ieee123.py")
        net = pp.from_json(path)
    else:
        raise ValueError(f"Unknown network {name!r}; expected ieee33, ieee69, or ieee123")

    net.load["base_p_mw"] = net.load["p_mw"].astype(float)
    net.load["base_q_mvar"] = net.load["q_mvar"].astype(float)
    return net


def bus_number_to_index(net, bus_number: int) -> int:
    """Resolve a conventional one-based bus number without index ambiguity."""
    matches = net.bus.index[net.bus["name"].astype(str).eq(str(int(bus_number)))]
    if len(matches) != 1:
        raise KeyError(f"Bus label {bus_number} resolved to {len(matches)} buses")
    return int(matches[0])


def restore_base_load(net, scale: float = 1.0) -> None:
    """Restore every load from the immutable canonical values."""
    net.load["p_mw"] = net.load["base_p_mw"].to_numpy(dtype=float) * scale
    net.load["q_mvar"] = net.load["base_q_mvar"].to_numpy(dtype=float) * scale


def run_power_flow(net, init: str = "auto", algorithm: str = "nr") -> bool:
    """Run the common deterministic AC power-flow settings."""
    if algorithm not in {"nr", "bfsw"}:
        raise ValueError(f"Unsupported AC power-flow algorithm: {algorithm}")
    try:
        pp.runpp(
            net,
            algorithm=algorithm,
            init=init,
            calculate_voltage_angles=False,
            max_iteration=30,
            tolerance_mva=1e-8,
            numba=False,
        )
    except pp.LoadflowNotConverged:
        return False
    return bool(net.converged and np.isfinite(net.res_bus["vm_pu"]).all())
