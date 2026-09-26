"""Aggregated EV device built on the common battery equations."""

from __future__ import annotations

from dataclasses import dataclass

from .battery import Battery


@dataclass
class EVGroup(Battery):
    v2g_enabled: bool = True
    connected: bool = False
    required_energy_kwh: float = 0.0
    departure_step: int = -1
    session_id: str = ""

    def connect(self, *, initial_energy_kwh: float, required_energy_kwh: float, departure_step: int, session_id: str) -> None:
        self.connected = True
        self.energy_kwh = min(self.max_energy_kwh, max(self.min_energy_kwh, float(initial_energy_kwh)))
        self.required_energy_kwh = min(self.max_energy_kwh, max(self.min_energy_kwh, float(required_energy_kwh)))
        self.departure_step = int(departure_step)
        self.session_id = str(session_id)

    def disconnect(self) -> None:
        self.connected = False

    def project_power(self, requested_kw: float, dt_hours: float, reserve_kwh: float | None = None) -> float:
        if not self.connected:
            return 0.0
        if not self.v2g_enabled and requested_kw > 0:
            requested_kw = 0.0
        return super().project_power(requested_kw, dt_hours, reserve_kwh=reserve_kwh)

    def departure_deficit_kwh(self) -> float:
        return max(0.0, self.required_energy_kwh - self.energy_kwh)

    def urgency(self, remaining_steps: int, dt_hours: float, epsilon: float = 1e-9) -> float:
        if not self.connected:
            return 0.0
        gap = max(0.0, self.required_energy_kwh - self.energy_kwh)
        deliverable = self.p_charge_max_kw * max(0, remaining_steps) * dt_hours * self.eta_charge
        return max(0.0, gap / (deliverable + epsilon) - 1.0)
