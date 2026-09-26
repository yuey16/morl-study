"""The benchmark's three cost components and physical diagnostics."""

from __future__ import annotations

import numpy as np


def economic_cost_aud(grid_p_mw: float, buy_price_aud_per_kwh: float, dt_hours: float, feed_in_ratio: float) -> float:
    grid_kw = float(grid_p_mw) * 1000.0
    imported = max(grid_kw, 0.0)
    exported = max(-grid_kw, 0.0)
    sell_price = buy_price_aud_per_kwh * feed_in_ratio
    return (buy_price_aud_per_kwh * imported - sell_price * exported) * dt_hours


def voltage_cost(vm_pu: np.ndarray, v_min: float, v_max: float, alpha: float,
                 worst_bus_weight: float = 0.0) -> tuple[float, dict[str, float]]:
    vm = np.asarray(vm_pu, dtype=float)
    if not 0.0 <= worst_bus_weight <= 1.0:
        raise ValueError("worst_bus_weight must be in [0, 1]")
    deviation = np.abs(vm - 1.0)
    violation = np.maximum(0.0, v_min - vm) + np.maximum(0.0, vm - v_max)
    mean_cost = alpha * float(deviation.mean()) + (1.0 - alpha) * float(violation.mean())
    cost = (1.0 - worst_bus_weight) * mean_cost + worst_bus_weight * float(violation.max())
    return cost, {
        "min_vm_pu": float(vm.min()),
        "max_vm_pu": float(vm.max()),
        "mean_abs_voltage_deviation": float(deviation.mean()),
        "voltage_violation_rate": float(((vm < v_min) | (vm > v_max)).mean()),
        "any_voltage_violation": float(np.any((vm < v_min) | (vm > v_max))),
        "voltage_violation_degree": float(violation.mean()),
        "worst_bus_violation_degree": float(violation.max()),
    }
