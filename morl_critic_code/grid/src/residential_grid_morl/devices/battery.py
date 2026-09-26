"""Stationary battery state and feasible active-power projection."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Battery:
    """A signed-power battery model.

    Positive power discharges/injects into the grid; negative power charges.
    Energy is stored in kWh and power in kW.
    """

    capacity_kwh: float
    p_charge_max_kw: float
    p_discharge_max_kw: float
    soc_min: float = 0.10
    soc_max: float = 0.90
    eta_charge: float = 0.95
    eta_discharge: float = 0.95
    energy_kwh: float = field(init=False)

    def __post_init__(self) -> None:
        if self.capacity_kwh <= 0 or self.p_charge_max_kw < 0 or self.p_discharge_max_kw < 0:
            raise ValueError("Battery capacity must be positive and power limits nonnegative")
        if not 0 <= self.soc_min < self.soc_max <= 1:
            raise ValueError("Expected 0 <= soc_min < soc_max <= 1")
        if not 0 < self.eta_charge <= 1 or not 0 < self.eta_discharge <= 1:
            raise ValueError("Efficiencies must be in (0, 1]")
        self.reset((self.soc_min + self.soc_max) / 2)

    @property
    def min_energy_kwh(self) -> float:
        return self.capacity_kwh * self.soc_min

    @property
    def max_energy_kwh(self) -> float:
        return self.capacity_kwh * self.soc_max

    @property
    def soc(self) -> float:
        return self.energy_kwh / self.capacity_kwh

    def reset(self, soc: float) -> None:
        if not self.soc_min <= soc <= self.soc_max:
            raise ValueError(f"Initial SOC {soc} is outside [{self.soc_min}, {self.soc_max}]")
        self.energy_kwh = float(soc * self.capacity_kwh)

    def action_to_requested_power(self, action: float) -> float:
        action = float(np.clip(action, -1.0, 1.0))
        return action * (self.p_discharge_max_kw if action >= 0 else self.p_charge_max_kw)

    def project_power(self, requested_kw: float, dt_hours: float, reserve_kwh: float | None = None) -> float:
        if dt_hours <= 0:
            raise ValueError("dt_hours must be positive")
        lower_energy = max(self.min_energy_kwh, float(reserve_kwh or 0.0))
        lower_energy = min(lower_energy, self.max_energy_kwh)
        charge_by_space = max(0.0, (self.max_energy_kwh - self.energy_kwh) / (self.eta_charge * dt_hours))
        discharge_by_energy = max(0.0, (self.energy_kwh - lower_energy) * self.eta_discharge / dt_hours)
        charge_limit = min(self.p_charge_max_kw, charge_by_space)
        discharge_limit = min(self.p_discharge_max_kw, discharge_by_energy)
        return float(np.clip(requested_kw, -charge_limit, discharge_limit))

    def advance(self, applied_kw: float, dt_hours: float) -> float:
        applied_kw = self.project_power(applied_kw, dt_hours)
        if applied_kw >= 0:
            self.energy_kwh -= applied_kw / self.eta_discharge * dt_hours
        else:
            self.energy_kwh += (-applied_kw) * self.eta_charge * dt_hours
        self.energy_kwh = float(np.clip(self.energy_kwh, self.min_energy_kwh, self.max_energy_kwh))
        return applied_kw

    def apply_action(self, action: float, dt_hours: float, reserve_kwh: float | None = None) -> tuple[float, float]:
        requested = self.action_to_requested_power(action)
        applied = self.project_power(requested, dt_hours, reserve_kwh=reserve_kwh)
        # Apply the already-projected command directly so an optional reserve is
        # not lost through a second projection using only the physical minimum.
        if applied >= 0:
            self.energy_kwh -= applied / self.eta_discharge * dt_hours
        else:
            self.energy_kwh += (-applied) * self.eta_charge * dt_hours
        self.energy_kwh = float(np.clip(self.energy_kwh, self.min_energy_kwh, self.max_energy_kwh))
        return requested, applied
