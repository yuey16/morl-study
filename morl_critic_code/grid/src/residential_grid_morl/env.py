"""Gymnasium-compatible IEEE 33/69 residential BESS-EV environment."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import pandapower as pp
from gymnasium import spaces

from .devices import Battery, EVGroup
from .logger import EpisodeLogger
from .network import bus_number_to_index, load_network, run_power_flow
from .objectives import economic_cost_aud, voltage_cost
from .profiles import load_ev_sessions, load_price_table, load_profile_tables
from .utils import PROJECT_ROOT, load_config


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load an environment-owned view of every processed input.

    Avoiding a module-level mutable cache prevents one environment instance from
    changing another instance's data through a shared pandas object.
    """
    load, pv = load_profile_tables()
    ev = load_ev_sessions()
    price = load_price_table()
    with (PROJECT_ROOT / "data/processed/splits.json").open(encoding="utf-8") as handle:
        splits = json.load(handle)
    return load, pv, ev, price, splits


class ResidentialBESSEVEnv(gym.Env):
    """One-day vector-reward environment using steady-state AC power flow."""

    metadata = {"render_modes": ["ansi"], "render_fps": 4}
    reward_dim = 3
    objective_names = ("economic", "ev_service", "voltage")

    def __init__(self, config: str | Path | dict[str, Any] = "configs/ieee33.yaml", render_mode: str | None = None):
        super().__init__()
        if isinstance(config, (str, Path)):
            path = Path(config)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            self.config = load_config(path)
        else:
            self.config = config
        self.render_mode = render_mode
        self.dt_minutes = int(self.config["simulation"]["dt_minutes"])
        if self.dt_minutes < 15 or self.dt_minutes % 15 or 1440 % self.dt_minutes:
            raise ValueError("dt_minutes must be a 15-minute multiple that divides one day")
        if int(self.config["simulation"]["episode_days"]) != 1:
            raise ValueError("Environment version 1 supports one-day episodes")
        self.dt_hours = self.dt_minutes / 60.0
        self.steps_per_episode = int(round(24 * int(self.config["simulation"]["episode_days"]) / self.dt_hours))
        self.forecast_horizon = int(self.config["simulation"]["forecast_horizon_steps"])
        self.forecast_mode = str(self.config["simulation"].get("forecast_mode", "perfect_short_horizon"))
        if self.forecast_mode not in {"perfect_short_horizon", "noisy_short_horizon"}:
            raise ValueError(f"Unsupported forecast_mode: {self.forecast_mode}")
        self.forecast_noise_std = float(self.config["simulation"].get("forecast_noise_std_fraction", 0.05))
        if self.forecast_noise_std < 0:
            raise ValueError("forecast_noise_std_fraction cannot be negative")
        self.net = load_network(self.config["network"]["name"])
        self.power_flow_algorithm = str(self.config["simulation"].get("power_flow_algorithm", "nr"))
        self.load_profiles, self.pv_profiles, self.sessions, self.prices, self.splits = _load_inputs()
        self.load_indices = self.net.load.index.to_numpy()
        self.base_load_p_mw = self.net.load.loc[self.load_indices, "base_p_mw"].to_numpy(dtype=float)
        self.base_load_q_mvar = self.net.load.loc[self.load_indices, "base_q_mvar"].to_numpy(dtype=float)
        self.power_factor_ratio = np.divide(
            self.base_load_q_mvar, self.base_load_p_mw,
            out=np.zeros_like(self.base_load_q_mvar), where=self.base_load_p_mw != 0,
        )

        bess_cfg = self.config["bess"]
        ev_cfg = self.config["ev"]
        self.bess_bus_numbers = [int(bus) for bus in bess_cfg["buses"]]
        configured_ev_buses = [int(bus) for bus in ev_cfg["buses"]]
        charger_counts = ev_cfg.get("chargers_per_bus")
        if charger_counts is not None:
            count_by_bus = {int(bus): int(count) for bus, count in charger_counts.items()}
            if set(count_by_bus) != set(configured_ev_buses) or any(count <= 0 for count in count_by_bus.values()):
                raise ValueError("chargers_per_bus must give a positive count for every EV bus")
            self.ev_bus_numbers = [
                bus for bus in configured_ev_buses for _ in range(count_by_bus[bus])
            ]
            n_equiv = 1.0
            self.chargers_per_bus = count_by_bus
        else:
            self.ev_bus_numbers = configured_ev_buses
            n_equiv = float(ev_cfg["equivalent_vehicles_per_group"]) * float(self.config["network"]["ev_scale"])
            self.chargers_per_bus = {bus: 1 for bus in configured_ev_buses}
        self.bess_bus_indices = [bus_number_to_index(self.net, bus) for bus in self.bess_bus_numbers]
        self.ev_bus_indices = [bus_number_to_index(self.net, bus) for bus in self.ev_bus_numbers]
        self.batteries = [
            Battery(
                float(bess_cfg["capacity_kwh"]), float(bess_cfg["p_max_kw"]), float(bess_cfg["p_max_kw"]),
                float(bess_cfg["soc_min"]), float(bess_cfg["soc_max"]),
                float(bess_cfg["eta_charge"]), float(bess_cfg["eta_discharge"]),
            ) for _ in self.bess_bus_indices
        ]
        self.equivalent_vehicles = n_equiv
        self.ev_groups = [
            EVGroup(
                float(ev_cfg["capacity_kwh_per_vehicle"]) * n_equiv,
                float(ev_cfg["charger_kw_per_vehicle"]) * n_equiv,
                float(ev_cfg["charger_kw_per_vehicle"]) * n_equiv,
                float(ev_cfg["soc_min"]), float(ev_cfg["soc_max"]),
                float(ev_cfg["eta_charge"]), float(ev_cfg["eta_discharge"]),
                v2g_enabled=bool(ev_cfg["v2g_enabled"]),
            ) for _ in self.ev_bus_indices
        ]
        self.n_bess, self.n_ev = len(self.batteries), len(self.ev_groups)
        five_objective_cfg = self.config.get("five_objective", {})
        self.five_objective_mode = five_objective_cfg.get("mode")
        self.pv_curtailment_enabled = bool(
            five_objective_cfg.get("pv_curtailment_control", False)
        )
        self.separate_bess_degradation_objective = bool(
            five_objective_cfg.get("separate_bess_degradation", False)
        )
        if self.pv_curtailment_enabled != self.separate_bess_degradation_objective:
            raise ValueError(
                "The five-objective environment requires both PV curtailment control "
                "and a separate BESS degradation objective"
            )
        if self.pv_curtailment_enabled and self.five_objective_mode is None:
            # Backward-compatible meaning of the already released v3 configs.
            self.five_objective_mode = "cbess_pv"
        if self.five_objective_mode not in {None, "cbess_pv", "battery_renewable"}:
            raise ValueError(f"Unsupported five_objective.mode: {self.five_objective_mode}")
        self.bess_efc_weight = float(five_objective_cfg.get("bess_efc_weight", 0.5))
        self.ev_efc_weight = float(five_objective_cfg.get("ev_efc_weight", 0.5))
        if self.five_objective_mode == "battery_renewable" and (
            self.bess_efc_weight < 0
            or self.ev_efc_weight < 0
            or not np.isclose(self.bess_efc_weight + self.ev_efc_weight, 1.0)
        ):
            raise ValueError("BESS and EV EFC weights must be nonnegative and sum to one")
        self.n_pv_controls = len(self.load_indices) if self.pv_curtailment_enabled else 0
        if self.pv_curtailment_enabled:
            self.reward_dim = 5
            if self.five_objective_mode == "battery_renewable":
                self.objective_names = (
                    "economic", "ev_service", "voltage",
                    "battery_degradation", "renewable_non_utilisation",
                )
            else:
                self.objective_names = (
                    "economic", "ev_service", "voltage",
                    "cbess_degradation", "pv_curtailment",
                )
        action_size = self.n_bess + self.n_ev + self.n_pv_controls
        action_low = np.full(action_size, -1.0, dtype=np.float32)
        action_high = np.ones(action_size, dtype=np.float32)
        if not bool(ev_cfg["v2g_enabled"]):
            action_high[self.n_bess:self.n_bess + self.n_ev] = 0.0
        if self.pv_curtailment_enabled:
            action_low[self.n_bess + self.n_ev:] = 0.0
        self.action_space = spaces.Box(action_low, action_high, dtype=np.float32)

        self.pv_sgen_indices = [
            pp.create_sgen(self.net, int(bus), p_mw=0.0, q_mvar=0.0, name=f"PV-load-{int(load_idx)}")
            for load_idx, bus in zip(self.load_indices, self.net.load.loc[self.load_indices, "bus"])
        ]
        self.bess_sgen_indices = [
            pp.create_sgen(self.net, bus, p_mw=0.0, q_mvar=0.0, name=f"BESS-{number}")
            for bus, number in zip(self.bess_bus_indices, self.bess_bus_numbers)
        ]
        self.unique_ev_bus_indices = list(dict.fromkeys(self.ev_bus_indices))
        charger_slot_by_bus: dict[int, int] = {}
        self.ev_charger_slots = []
        for bus in self.ev_bus_numbers:
            slot = charger_slot_by_bus.get(bus, 0)
            self.ev_charger_slots.append(slot)
            charger_slot_by_bus[bus] = slot + 1
        self.ev_sgen_indices = [
            pp.create_sgen(
                self.net, bus, p_mw=0.0, q_mvar=0.0,
                name=f"EV-aggregate-{self.net.bus.loc[bus, 'name']}",
            )
            for bus in self.unique_ev_bus_indices
        ]
        self.ev_sgen_index_by_bus = dict(zip(self.unique_ev_bus_indices, self.ev_sgen_indices))
        self.control_bus_indices = list(dict.fromkeys(self.bess_bus_indices + self.ev_bus_indices))
        self.previous_actions = np.zeros(self.action_space.shape, dtype=np.float32)
        self.previous_control_voltages = np.ones(len(self.control_bus_indices), dtype=np.float32)
        self.previous_min_voltage = 1.0
        self.previous_max_voltage = 1.0
        self.previous_grid_import_mw = 0.0
        self._step = 0
        self._pf_has_results = False
        self._departures = 0
        self._successful_departures = 0
        self._departure_deficits: list[float] = []
        self.logger = EpisodeLogger()
        self._last_render = "Environment has not been reset"
        self._build_observation_space()

    def _build_observation_space(self) -> None:
        spatial_size = len(self.control_bus_indices) if bool(
            self.config["simulation"].get("include_spatial_net_power", False)
        ) else 0
        arrival_forecast_size = self.n_ev if bool(
            self.config["simulation"].get("include_ev_arrival_forecast", False)
        ) else 0
        size = (
            2 + (1 + self.forecast_horizon) + 2 * (1 + self.forecast_horizon)
            + self.n_bess + 4 * self.n_ev + len(self.control_bus_indices) + 3
            + self.action_space.shape[0] + spatial_size + arrival_forecast_size
        )
        # Finite engineering guard rails make checker failures informative while
        # remaining far outside every physically reachable observation.
        self.observation_space = spaces.Box(-1.0e7, 1.0e7, shape=(size,), dtype=np.float32)

    @property
    def current_step(self) -> int:
        return self._step

    def _valid_days(self, source: str, split: str) -> list[str]:
        days = self.splits[source][split]
        pool = set(days)
        return [day for day in days if (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%Y-%m-%d") in pool]

    def _choose_scenario(self, options: dict[str, Any]) -> dict[str, str]:
        if "scenario" in options:
            return dict(options["scenario"])
        if "scenario_id" in options:
            matches = [row for row in self.splits["test_scenarios"] if row["scenario_id"] == options["scenario_id"]]
            if not matches:
                raise KeyError(f"Unknown test scenario {options['scenario_id']}")
            return dict(matches[0])
        split = str(options.get("split", self.config["simulation"].get("split", "train")))
        scenario = {"scenario_id": f"sampled-{split}-{int(self.np_random.integers(2**31))}"}
        for key, source in (("ausgrid_day", "ausgrid"), ("electric_nation_day", "electric_nation"), ("aemo_price_day", "aemo_price")):
            days = self._valid_days(source, split)
            scenario[key] = days[int(self.np_random.integers(len(days)))]
        return scenario

    def _time_window(self, frame: pd.DataFrame, day: str) -> pd.DataFrame:
        start = pd.Timestamp(day) + pd.Timedelta(hours=int(self.config["simulation"]["episode_start_hour"]))
        index = pd.date_range(start, periods=self.steps_per_episode, freq=f"{self.dt_minutes}min")
        result = frame.reindex(index)
        if result.isna().any().any():
            raise ValueError(f"Incomplete profile window beginning {start}")
        return result

    def _prepare_profiles(self, scenario: dict[str, str]) -> None:
        load_window = self._time_window(self.load_profiles, scenario["ausgrid_day"])
        pv_window = self._time_window(self.pv_profiles, scenario["ausgrid_day"])
        price_window = self._time_window(self.prices, scenario["aemo_price_day"])
        count = int(self.config["profiles"]["households_per_load"])
        customer_count = len(load_window.columns)
        if bool(self.config["randomization"].get("random_household_assignment", True)):
            assignments = self.np_random.integers(0, customer_count, size=(len(self.load_indices), count))
        else:
            assignments = np.arange(len(self.load_indices) * count).reshape(len(self.load_indices), count)
            assignments %= customer_count
        load_values, pv_values = load_window.to_numpy(dtype=float), pv_window.to_numpy(dtype=float)
        self.load_shapes = np.stack([load_values[:, row].mean(axis=1) for row in assignments], axis=1)
        self.pv_shapes = np.stack([pv_values[:, row].mean(axis=1) for row in assignments], axis=1)
        self.price_profile = price_window["retail_aud_per_kwh"].to_numpy(dtype=float) * float(self.config["price"]["scale"])
        self.episode_timestamps = load_window.index
        load_noise = float(self.config["randomization"].get("load_scale_noise", 0.0))
        pv_noise = float(self.config["randomization"].get("pv_scale_noise", 0.0))
        self.episode_load_scale = float(self.config["network"]["load_scale"]) * float(self.np_random.uniform(1-load_noise, 1+load_noise))
        self.episode_pv_scale = float(self.config["network"]["pv_scale"]) * float(self.np_random.uniform(1-pv_noise, 1+pv_noise))
        spatial_load_noise = float(self.config["randomization"].get("spatial_load_scale_noise", 0.0))
        spatial_pv_noise = float(self.config["randomization"].get("spatial_pv_scale_noise", 0.0))
        self.load_bus_multipliers = (
            self.np_random.uniform(1.0 - spatial_load_noise, 1.0 + spatial_load_noise, len(self.load_indices))
            if spatial_load_noise else np.ones(len(self.load_indices))
        )
        self.pv_bus_multipliers = (
            self.np_random.uniform(1.0 - spatial_pv_noise, 1.0 + spatial_pv_noise, len(self.load_indices))
            if spatial_pv_noise else np.ones(len(self.load_indices))
        )
        self._forecast_noise: dict[str, np.ndarray] = {}
        for name in ("price", "load", "pv"):
            if self.forecast_mode == "noisy_short_horizon":
                noise = self.np_random.normal(
                    1.0,
                    self.forecast_noise_std,
                    size=(self.steps_per_episode, self.forecast_horizon + 1),
                )
                noise[:, 0] = 1.0  # the current measurement is observed, not forecast
                self._forecast_noise[name] = np.maximum(noise, 0.0)
            else:
                self._forecast_noise[name] = np.ones(
                    (self.steps_per_episode, self.forecast_horizon + 1), dtype=float
                )

    def _prepare_ev_sessions(self, day: str) -> None:
        start = pd.Timestamp(day) + pd.Timedelta(hours=int(self.config["simulation"]["episode_start_hour"]))
        end = start + pd.Timedelta(days=1)
        eligible = self.sessions.loc[
            self.sessions["arrival_time"].ge(start) & self.sessions["arrival_time"].lt(end)
            & self.sessions["departure_time"].gt(self.sessions["arrival_time"])
            & self.sessions["departure_time"].le(end)
        ]
        if eligible.empty:
            raise ValueError(f"No complete Electric Nation sessions in episode window {start}--{end}")
        self.ev_session_specs: list[dict[str, Any]] = []
        ev_cfg = self.config["ev"]
        schedule_mode = str(ev_cfg.get("schedule_mode", "empirical_clustered"))
        mixed_mode = schedule_mode == "mixed_node_archetypes"
        selected = self.np_random.choice(len(eligible), size=self.n_ev, replace=len(eligible) < self.n_ev)
        group_to_row = {index: int(row_index) for index, row_index in enumerate(selected)}
        mixed_schedule: dict[int, dict[str, Any]] = {}
        stress_center: int | None = None

        if mixed_mode:
            self.ev_stress_mode = schedule_mode
            archetypes = ev_cfg.get("schedule_archetypes", {})
            node_archetypes = {int(bus): str(name) for bus, name in ev_cfg.get("node_archetypes", {}).items()}
            if set(node_archetypes) != set(self.ev_bus_numbers):
                raise ValueError("node_archetypes must assign every configured EV bus exactly once")
            if len(set(node_archetypes.values())) < 2:
                raise ValueError("mixed_node_archetypes requires at least two node archetypes")
            local_hours = (
                int(self.config["simulation"]["episode_start_hour"])
                + np.arange(self.steps_per_episode) * self.dt_hours
            ) % 24.0
            load_matrix = (
                self.load_shapes * self.base_load_p_mw[None, :] * self.episode_load_scale
                * self.load_bus_multipliers[None, :]
            )
            load_signal = load_matrix.sum(axis=1)
            load_bus_array = self.net.load.loc[self.load_indices, "bus"].to_numpy(dtype=int)
            local_load_signals = {
                bus: load_matrix[:, load_bus_array == bus].sum(axis=1)
                if np.any(load_bus_array == bus) else load_signal
                for bus in self.ev_bus_indices
            }
            biases = list(ev_cfg.get("schedule_timing_biases", ["uniform", "high_price", "high_load"]))
            timing_regimes = ev_cfg.get("schedule_timing_regimes")
            if timing_regimes:
                regime_names = list(timing_regimes)
                regime_probabilities = np.asarray(
                    ev_cfg.get(
                        "schedule_timing_regime_probabilities",
                        np.ones(len(regime_names), dtype=float),
                    ),
                    dtype=float,
                )
                if len(regime_probabilities) != len(regime_names) or np.any(regime_probabilities < 0):
                    raise ValueError("Invalid schedule timing regime probabilities")
                regime_probabilities = regime_probabilities / regime_probabilities.sum()
                self.ev_timing_regime = str(
                    self.np_random.choice(regime_names, p=regime_probabilities)
                )
                probabilities = np.asarray(
                    timing_regimes[self.ev_timing_regime], dtype=float
                )
            else:
                self.ev_timing_regime = "per_charger_mixture"
                probabilities = np.asarray(
                    ev_cfg.get("schedule_timing_bias_probabilities", [0.50, 0.25, 0.25]),
                    dtype=float,
                )
            if len(biases) != len(probabilities) or any(
                bias not in {"uniform", "high_price", "high_load"} for bias in biases
            ) or np.any(probabilities < 0) or probabilities.sum() <= 0:
                raise ValueError("Invalid schedule timing biases")
            probabilities = probabilities / probabilities.sum()
            # Spread nodes of the same archetype across their available peak
            # windows before repeating any window. This prevents an accidental
            # return to the old all-nodes-at-one-time stress pattern.
            groups_by_archetype: dict[tuple[int, str], list[int]] = {}
            for group_index, bus in enumerate(self.ev_bus_numbers):
                key = (bus, node_archetypes[bus])
                groups_by_archetype.setdefault(key, []).append(group_index)
            window_by_group: dict[int, int] = {}
            for (_, archetype_name), group_indices in groups_by_archetype.items():
                window_count = len(archetypes[archetype_name]["arrival_windows_local_hour"])
                pending = list(group_indices)
                while pending:
                    order = self.np_random.permutation(window_count).tolist()
                    for window_index in order:
                        if not pending:
                            break
                        window_by_group[pending.pop(0)] = int(window_index)

            # A fraction of chargers belongs to a site-level arrival/departure
            # cohort (shift change, residential commute, event dismissal). The
            # rest keep independent timing. Cohorts create meaningful feeder
            # ramps without making every EV an identical synchronized copy.
            cohort_fraction = float(ev_cfg.get("schedule_cohort_fraction", 0.0))
            if not 0.0 <= cohort_fraction <= 1.0:
                raise ValueError("schedule_cohort_fraction must lie in [0, 1]")
            cohort_jitter = int(ev_cfg.get("schedule_cohort_jitter_steps", 1))
            if cohort_jitter < 0:
                raise ValueError("schedule_cohort_jitter_steps cannot be negative")
            cohort_groups: dict[tuple[int, str, int], list[int]] = {}
            for group_index, bus in enumerate(self.ev_bus_numbers):
                key = (bus, node_archetypes[bus], window_by_group[group_index])
                cohort_groups.setdefault(key, []).append(group_index)
            cohort_members: set[int] = set()
            cohort_targets: dict[tuple[int, str, int], tuple[int, int, str]] = {}
            dominant_bias = (
                "high_price" if self.ev_timing_regime == "high_price_day" else "high_load"
            )
            for key, group_indices in cohort_groups.items():
                bus, archetype_name, window_index = key
                member_count = int(round(cohort_fraction * len(group_indices)))
                if member_count:
                    chosen = self.np_random.permutation(group_indices)[:member_count]
                    cohort_members.update(int(index) for index in chosen)
                archetype = archetypes[archetype_name]
                window_start, window_end = map(
                    float, archetype["arrival_windows_local_hour"][window_index]
                )
                if window_start < window_end:
                    candidates = np.flatnonzero(
                        (local_hours >= window_start) & (local_hours < window_end)
                    )
                else:
                    candidates = np.flatnonzero(
                        (local_hours >= window_start) | (local_hours < window_end)
                    )
                signal = (
                    self.price_profile if dominant_bias == "high_price"
                    else local_load_signals[bus_number_to_index(self.net, bus)]
                )
                top_fraction = float(ev_cfg.get("schedule_peak_top_fraction", 0.35))
                top_count = max(
                    1, int(np.ceil(np.clip(top_fraction, 0.05, 1.0) * len(candidates)))
                )
                strongest = candidates[np.argpartition(signal[candidates], -top_count)[-top_count:]]
                arrival_target = int(strongest[int(self.np_random.integers(len(strongest)))])
                dwell_low, dwell_high = map(float, archetype["dwell_hours"])
                dwell_low_steps = max(1, int(np.ceil(dwell_low / self.dt_hours)))
                dwell_high_steps = max(
                    dwell_low_steps, int(np.floor(dwell_high / self.dt_hours))
                )
                dwell_target = int(
                    self.np_random.integers(dwell_low_steps, dwell_high_steps + 1)
                )
                cohort_targets[key] = (
                    arrival_target,
                    min(self.steps_per_episode, arrival_target + dwell_target),
                    dominant_bias,
                )
            for group_index, (group, bus) in enumerate(zip(self.ev_groups, self.ev_bus_numbers)):
                archetype_name = node_archetypes[bus]
                if archetype_name not in archetypes:
                    raise ValueError(f"Missing schedule archetype {archetype_name!r}")
                archetype = archetypes[archetype_name]
                windows = list(archetype["arrival_windows_local_hour"])
                window = windows[window_by_group[group_index]]
                window_start, window_end = map(float, window)
                if window_start < window_end:
                    candidates = np.flatnonzero((local_hours >= window_start) & (local_hours < window_end))
                else:
                    candidates = np.flatnonzero((local_hours >= window_start) | (local_hours < window_end))
                if not len(candidates):
                    raise ValueError(f"Arrival window {window} has no simulation steps")
                cohort_key = (bus, archetype_name, window_by_group[group_index])
                cohort_member = group_index in cohort_members
                if cohort_member:
                    arrival_target, departure_target, timing_bias = cohort_targets[cohort_key]
                    nearby = candidates[np.abs(candidates - arrival_target) <= cohort_jitter]
                    arrival_step = int(nearby[int(self.np_random.integers(len(nearby)))])
                else:
                    timing_bias = str(self.np_random.choice(biases, p=probabilities))
                    if timing_bias == "uniform":
                        arrival_step = int(candidates[int(self.np_random.integers(len(candidates)))])
                    else:
                        signal = (
                            self.price_profile if timing_bias == "high_price"
                            else local_load_signals[self.ev_bus_indices[group_index]]
                        )
                        top_fraction = float(ev_cfg.get("schedule_peak_top_fraction", 0.35))
                        top_count = max(1, int(np.ceil(np.clip(top_fraction, 0.05, 1.0) * len(candidates))))
                        strongest = candidates[np.argpartition(signal[candidates], -top_count)[-top_count:]]
                        arrival_step = int(strongest[int(self.np_random.integers(len(strongest)))])
                dwell_low, dwell_high = map(float, archetype["dwell_hours"])
                dwell_low_steps = max(1, int(np.ceil(dwell_low / self.dt_hours)))
                dwell_high_steps = max(dwell_low_steps, int(np.floor(dwell_high / self.dt_hours)))
                if cohort_member:
                    departure_step = int(np.clip(
                        departure_target + self.np_random.integers(-cohort_jitter, cohort_jitter + 1),
                        arrival_step + dwell_low_steps,
                        min(self.steps_per_episode, arrival_step + dwell_high_steps),
                    ))
                    dwell_steps = departure_step - arrival_step
                else:
                    # Discrete-uniform on the simulator grid: every admissible
                    # 15-minute dwell duration has the same sampling probability.
                    dwell_steps = int(self.np_random.integers(dwell_low_steps, dwell_high_steps + 1))
                    departure_step = min(self.steps_per_episode, arrival_step + dwell_steps)

                # Keep the empirical energy request, but condition short daytime
                # stays on requests that are physically deliverable in the
                # synthetic dwell window at the source charger rating.
                feasible_per_vehicle = (
                    float(ev_cfg["charger_kw_per_vehicle"]) * float(ev_cfg["eta_charge"])
                    * dwell_steps * self.dt_hours / float(ev_cfg.get("energy_demand_scale", 1.0))
                )
                feasible_rows = np.flatnonzero(eligible["energy_kwh"].to_numpy(dtype=float) <= feasible_per_vehicle)
                row_index = int(feasible_rows[int(self.np_random.integers(len(feasible_rows)))])
                group_to_row[group_index] = row_index
                mixed_schedule[group_index] = {
                    "arrival_step": arrival_step,
                    "departure_step": departure_step,
                    "scheduled_dwell_hours": dwell_steps * self.dt_hours,
                    "archetype": archetype_name,
                    "timing_bias": timing_bias,
                    "cohort_member": cohort_member,
                    "arrival_window_index": window_by_group[group_index],
                }
        else:
            self.ev_timing_regime = "empirical"
            stress_probability = float(ev_cfg.get("temporal_stress_probability", 0.0))
            self.ev_stress_mode = "natural"
            if stress_probability > 0.0 and self.np_random.random() < stress_probability:
                modes = list(ev_cfg.get("temporal_stress_modes", ["high_price", "high_load"]))
                if not modes or any(mode not in {"high_price", "high_load"} for mode in modes):
                    raise ValueError("temporal_stress_modes must contain high_price and/or high_load")
                self.ev_stress_mode = str(modes[int(self.np_random.integers(len(modes)))])
                stress_signal = self.price_profile if self.ev_stress_mode == "high_price" else (
                    self.load_shapes * self.base_load_p_mw[None, :] * self.episode_load_scale
                    * self.load_bus_multipliers[None, :]
                ).sum(axis=1)
                top_fraction = float(ev_cfg.get("stress_top_window_fraction", 0.10))
                top_count = max(1, int(np.ceil(np.clip(top_fraction, 0.01, 1.0) * self.steps_per_episode)))
                candidates = np.argpartition(stress_signal, -top_count)[-top_count:]
                stress_center = int(candidates[int(self.np_random.integers(len(candidates)))])
            if stress_center is not None:
                priority = [int(bus) for bus in ev_cfg.get("stress_bus_priority", self.ev_bus_numbers[::-1])]
                priority_rank = {bus: rank for rank, bus in enumerate(priority)}
                weak_first = sorted(
                    range(self.n_ev),
                    key=lambda index: priority_rank.get(self.ev_bus_numbers[index], len(priority) + index),
                )
                largest_first = sorted(
                    (int(row_index) for row_index in selected),
                    key=lambda row_index: float(eligible.iloc[row_index]["energy_kwh"]),
                    reverse=True,
                )
                group_to_row = dict(zip(weak_first, largest_first))

        cluster_half_width = int(ev_cfg.get("stress_cluster_half_width_steps", 2))
        for group_index, group in enumerate(self.ev_groups):
            row_index = group_to_row[group_index]
            row = eligible.iloc[int(row_index)]
            source_arrival_step = int(math.floor((row["arrival_time"] - start).total_seconds() / (self.dt_hours * 3600)))
            source_departure_step = int(math.ceil((row["departure_time"] - start).total_seconds() / (self.dt_hours * 3600)))
            schedule = mixed_schedule.get(group_index)
            if schedule is not None:
                arrival_step = int(schedule["arrival_step"])
                departure_step = int(schedule["departure_step"])
            elif stress_center is None:
                arrival_step = source_arrival_step
                departure_step = source_departure_step
            else:
                jitter = int(self.np_random.integers(-cluster_half_width, cluster_half_width + 1))
                arrival_step = int(np.clip(stress_center + jitter, 0, self.steps_per_episode - 1))
                empirical_dwell_steps = max(1, source_departure_step - source_arrival_step)
                departure_step = min(self.steps_per_episode, arrival_step + empirical_dwell_steps)
            departure_step = min(self.steps_per_episode, max(arrival_step + 1, departure_step))
            target = float(ev_cfg["target_soc"]) * group.capacity_kwh
            reconstructed_need = (
                float(row["energy_kwh"]) * self.equivalent_vehicles
                * float(ev_cfg.get("energy_demand_scale", 1.0))
            )
            early_probability = float(ev_cfg.get("early_departure_probability", 0.0))
            max_early_steps = int(ev_cfg.get("max_early_departure_steps", 0))
            if not mixed_mode and max_early_steps > 0 and self.np_random.random() < early_probability:
                departure_step = max(
                    arrival_step + 1,
                    departure_step - int(self.np_random.integers(1, max_early_steps + 1)),
                )
            initial = max(group.min_energy_kwh, target - reconstructed_need)
            group.connected = False
            group.energy_kwh = initial
            group.required_energy_kwh = target
            group.departure_step = departure_step
            group.session_id = str(row["session_id"])
            self.ev_session_specs.append({
                "arrival_step": arrival_step, "departure_step": departure_step,
                "initial_energy_kwh": initial, "required_energy_kwh": target,
                "session_id": str(row["session_id"]),
                "source_energy_kwh_per_vehicle": float(row["energy_kwh"]),
                "source_arrival_step": source_arrival_step,
                "source_departure_step": source_departure_step,
                "stress_mode": self.ev_stress_mode,
                "schedule_archetype": schedule["archetype"] if schedule else "empirical",
                "schedule_timing_bias": schedule["timing_bias"] if schedule else self.ev_stress_mode,
                "schedule_cohort_member": bool(schedule.get("cohort_member", False)) if schedule else False,
                "schedule_arrival_window_index": int(schedule.get("arrival_window_index", -1)) if schedule else -1,
                "scheduled_dwell_hours": (
                    schedule["scheduled_dwell_hours"] if schedule
                    else (departure_step - arrival_step) * self.dt_hours
                ),
            })

    def _process_boundary_events(self, boundary_step: int) -> tuple[float, list[float]]:
        normalized_deficit, deficits = 0.0, []
        success_fraction = float(self.config["ev"]["departure_success_fraction"])
        for group, spec in zip(self.ev_groups, self.ev_session_specs):
            if group.connected and spec["departure_step"] == boundary_step:
                deficit = group.departure_deficit_kwh()
                deficits.append(deficit)
                normalized_deficit += deficit / max(group.required_energy_kwh, 1e-9)
                self._departures += 1
                self._successful_departures += int(group.energy_kwh >= success_fraction * group.required_energy_kwh)
                self._departure_deficits.append(deficit)
                group.disconnect()
            if (not group.connected) and spec["arrival_step"] == boundary_step:
                group.connect(
                    initial_energy_kwh=spec["initial_energy_kwh"], required_energy_kwh=spec["required_energy_kwh"],
                    departure_step=spec["departure_step"], session_id=spec["session_id"],
                )
        return normalized_deficit, deficits

    def _set_exogenous(
        self, step: int, pv_curtailment_fraction: np.ndarray | None = None
    ) -> tuple[float, float, float, float]:
        t = min(step, self.steps_per_episode - 1)
        p_load = self.base_load_p_mw * self.load_shapes[t] * self.episode_load_scale * self.load_bus_multipliers
        q_load = p_load * self.power_factor_ratio
        pv_capacity = self.base_load_p_mw * float(self.config["network"]["pv_capacity_ratio_to_base_load"])
        p_pv_available = (
            pv_capacity * self.pv_shapes[t] * self.episode_pv_scale * self.pv_bus_multipliers
        )
        if pv_curtailment_fraction is None:
            pv_curtailment_fraction = np.zeros(len(self.pv_sgen_indices), dtype=float)
        pv_curtailment_fraction = np.asarray(pv_curtailment_fraction, dtype=float)
        if pv_curtailment_fraction.shape != (len(self.pv_sgen_indices),):
            raise ValueError(
                f"Expected {len(self.pv_sgen_indices)} PV curtailment controls, "
                f"got {pv_curtailment_fraction.shape}"
            )
        p_pv = p_pv_available * (1.0 - np.clip(pv_curtailment_fraction, 0.0, 1.0))
        self.net.load.loc[self.load_indices, "p_mw"] = p_load
        self.net.load.loc[self.load_indices, "q_mvar"] = q_load
        self.net.sgen.loc[self.pv_sgen_indices, "p_mw"] = p_pv
        available_kw = float(p_pv_available.sum() * 1000)
        injected_kw = float(p_pv.sum() * 1000)
        return (
            float(p_load.sum() * 1000), available_kw, injected_kw,
            available_kw - injected_kw,
        )

    def _run_pf(self) -> bool:
        converged = run_power_flow(
            self.net,
            init="results" if self._pf_has_results else "auto",
            algorithm=self.power_flow_algorithm,
        )
        self._pf_has_results = self._pf_has_results or converged
        if converged:
            self.previous_control_voltages = self.net.res_bus.loc[self.control_bus_indices, "vm_pu"].to_numpy(dtype=np.float32)
            self.previous_min_voltage = float(self.net.res_bus["vm_pu"].min())
            self.previous_max_voltage = float(self.net.res_bus["vm_pu"].max())
            self.previous_grid_import_mw = float(self.net.res_ext_grid["p_mw"].sum())
        return converged

    def _forecast(self, values: np.ndarray, t: int, name: str) -> list[float]:
        return [
            float(values[min(t + offset, self.steps_per_episode - 1)] * self._forecast_noise[name][t, offset])
            for offset in range(self.forecast_horizon + 1)
        ]

    def _control_bus_net_power_kw(self, t: int) -> list[float]:
        """Return current exogenous load minus PV at each controlled bus."""
        p_load = (
            self.base_load_p_mw * self.load_shapes[t] * self.episode_load_scale
            * self.load_bus_multipliers
        )
        pv_capacity = self.base_load_p_mw * float(self.config["network"]["pv_capacity_ratio_to_base_load"])
        p_pv = (
            pv_capacity * self.pv_shapes[t] * self.episode_pv_scale
            * self.pv_bus_multipliers
        )
        by_bus = {bus: 0.0 for bus in self.control_bus_indices}
        for bus, net_mw in zip(self.net.load.loc[self.load_indices, "bus"], p_load - p_pv):
            if int(bus) in by_bus:
                by_bus[int(bus)] += float(net_mw) * 1000.0
        return [by_bus[bus] for bus in self.control_bus_indices]

    def _observation(self) -> np.ndarray:
        t = min(self._step, self.steps_per_episode - 1)
        phase = 2 * math.pi * t / self.steps_per_episode
        load_total = (
            self.load_shapes * self.base_load_p_mw[None, :] * self.episode_load_scale
            * self.load_bus_multipliers[None, :]
        ).sum(axis=1) * 1000
        pv_capacity = self.base_load_p_mw * float(self.config["network"]["pv_capacity_ratio_to_base_load"])
        pv_total = (
            self.pv_shapes * pv_capacity[None, :] * self.episode_pv_scale
            * self.pv_bus_multipliers[None, :]
        ).sum(axis=1) * 1000
        ev_energy = [group.energy_kwh / group.capacity_kwh for group in self.ev_groups]
        ev_connected = [float(group.connected) for group in self.ev_groups]
        ev_ttd = [max(0, spec["departure_step"] - self._step) / self.steps_per_episode if group.connected else 0.0 for group, spec in zip(self.ev_groups, self.ev_session_specs)]
        ev_required = [group.required_energy_kwh / group.capacity_kwh for group in self.ev_groups]
        ev_tta = [
            max(0, spec["arrival_step"] - self._step) / self.steps_per_episode
            if self._step < spec["arrival_step"] else 0.0
            for spec in self.ev_session_specs
        ] if bool(self.config["simulation"].get("include_ev_arrival_forecast", False)) else []
        spatial_net_power = self._control_bus_net_power_kw(t) if bool(
            self.config["simulation"].get("include_spatial_net_power", False)
        ) else []
        values = (
            [math.sin(phase), math.cos(phase)] + self._forecast(self.price_profile, t, "price")
            + self._forecast(load_total, t, "load") + self._forecast(pv_total, t, "pv")
            + [group.soc for group in self.batteries]
            + ev_energy + ev_connected + ev_ttd + ev_required + ev_tta
            + spatial_net_power
            + self.previous_control_voltages.tolist()
            + [self.previous_min_voltage, self.previous_max_voltage, self.previous_grid_import_mw]
            + self.previous_actions.tolist()
        )
        observation = np.asarray(values, dtype=np.float32)
        if observation.shape != self.observation_space.shape:
            raise AssertionError(f"Observation shape {observation.shape} != {self.observation_space.shape}")
        return observation

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}
        self.scenario = self._choose_scenario(options)
        self.scenario_id = self.scenario["scenario_id"]
        self._prepare_profiles(self.scenario)
        pv_capacity = self.base_load_p_mw * float(
            self.config["network"]["pv_capacity_ratio_to_base_load"]
        )
        available_pv_kw = (
            self.pv_shapes * pv_capacity[None, :] * self.episode_pv_scale
            * self.pv_bus_multipliers[None, :]
        ).sum(axis=1) * 1000.0
        self.episode_available_pv_energy_kwh = float(
            available_pv_kw.sum() * self.dt_hours
        )
        self._prepare_ev_sessions(self.scenario["electric_nation_day"])
        bess_cfg = self.config["bess"]
        self.initial_bess_soc = []
        for battery in self.batteries:
            soc = float(self.np_random.uniform(bess_cfg["initial_soc_low"], bess_cfg["initial_soc_high"]))
            battery.reset(soc)
            self.initial_bess_soc.append(soc)
        self._step = 0
        self._pf_has_results = False
        self._departures = self._successful_departures = 0
        self._departure_deficits = []
        self.previous_actions.fill(0)
        self.net.sgen.loc[self.bess_sgen_indices + self.ev_sgen_indices, "p_mw"] = 0.0
        self.logger.reset()
        self._process_boundary_events(0)
        self._set_exogenous(0)
        if not self._run_pf():
            raise RuntimeError("Initial episode power flow did not converge")
        info = {
            "episode_scenario_id": self.scenario_id, "scenario": dict(self.scenario),
            "aggregation": {
                "control_level": "individual_charger" if self.equivalent_vehicles == 1.0 else "equivalent_group",
                "equivalent_vehicles_per_action": self.equivalent_vehicles,
                "chargers_per_bus": dict(self.chargers_per_bus),
                "source": "Electric Nation",
            },
            "environment_variant": (
                "v4_5obj_battery_renewable"
                if self.five_objective_mode == "battery_renewable"
                else "v3_5obj_cbess_pv" if self.reward_dim == 5 else "v2_3obj"
            ),
            "objective_names": self.objective_names,
            "action_layout": {
                "bess": [0, self.n_bess],
                "ev": [self.n_bess, self.n_bess + self.n_ev],
                "pv_curtailment": [
                    self.n_bess + self.n_ev,
                    self.n_bess + self.n_ev + self.n_pv_controls,
                ],
            },
            "ev_stress_mode": self.ev_stress_mode,
            "ev_timing_regime": self.ev_timing_regime,
            "ev_schedule": [
                {
                    "action_index": self.n_bess + ev_index,
                    "bus": bus,
                    "charger_slot": charger_slot,
                    "archetype": spec["schedule_archetype"],
                    "timing_bias": spec["schedule_timing_bias"],
                    "cohort_member": spec["schedule_cohort_member"],
                    "arrival_window_index": spec["schedule_arrival_window_index"],
                    "arrival_step": spec["arrival_step"],
                    "departure_step": spec["departure_step"],
                    "dwell_hours": spec["scheduled_dwell_hours"],
                }
                for ev_index, (bus, charger_slot, spec) in enumerate(zip(
                    self.ev_bus_numbers, self.ev_charger_slots, self.ev_session_specs
                ))
            ],
            "power_flow_converged": True,
        }
        return self._observation(), info

    def step(self, action: np.ndarray):
        if self._step >= self.steps_per_episode:
            raise RuntimeError("Call reset() after the episode ends")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError(f"Expected action shape {self.action_space.shape}, got {action.shape}")
        clipped = np.clip(action, self.action_space.low, self.action_space.high)
        requested, applied = [], []
        for device, normalized in zip(self.batteries, clipped[:self.n_bess]):
            req, app = device.apply_action(float(normalized), self.dt_hours)
            requested.append(req); applied.append(app)
        hard_reserve = bool(self.config["ev"].get("hard_departure_reserve", False))
        device_action_end = self.n_bess + self.n_ev
        for device, normalized in zip(
            self.ev_groups, clipped[self.n_bess:device_action_end]
        ):
            reserve = device.required_energy_kwh if hard_reserve else None
            req, app = device.apply_action(float(normalized), self.dt_hours, reserve_kwh=reserve)
            requested.append(req); applied.append(app)

        self.net.sgen.loc[self.bess_sgen_indices, "p_mw"] = np.asarray(applied[:self.n_bess]) / 1000.0
        ev_power_by_bus = {bus: 0.0 for bus in self.unique_ev_bus_indices}
        for bus, power_kw in zip(self.ev_bus_indices, applied[self.n_bess:]):
            ev_power_by_bus[bus] += float(power_kw)
        self.net.sgen.loc[self.ev_sgen_indices, "p_mw"] = np.asarray(
            [ev_power_by_bus[bus] for bus in self.unique_ev_bus_indices], dtype=float
        ) / 1000.0
        pv_curtailment_fraction = (
            clipped[device_action_end:] if self.pv_curtailment_enabled
            else np.zeros(len(self.pv_sgen_indices), dtype=np.float32)
        )
        base_load_kw, pv_available_kw, pv_kw, pv_curtailed_kw = self._set_exogenous(
            self._step, pv_curtailment_fraction
        )
        converged = self._run_pf()
        price = float(self.price_profile[self._step])
        bess_throughput_kwh = (
            sum(abs(value) for value in applied[:self.n_bess]) * self.dt_hours
        )
        ev_throughput_kwh = (
            sum(abs(value) for value in applied[self.n_bess:]) * self.dt_hours
        )
        total_bess_capacity_kwh = sum(device.capacity_kwh for device in self.batteries)
        total_ev_capacity_kwh = sum(device.capacity_kwh for device in self.ev_groups)
        cbess_efc = bess_throughput_kwh / max(
            2.0 * total_bess_capacity_kwh, 1e-9
        )
        ev_efc = ev_throughput_kwh / max(2.0 * total_ev_capacity_kwh, 1e-9)
        if self.five_objective_mode == "battery_renewable":
            battery_degradation = (
                self.bess_efc_weight * cbess_efc
                + self.ev_efc_weight * ev_efc
            )
        else:
            battery_degradation = cbess_efc
        pv_curtailment = (
            pv_curtailed_kw * self.dt_hours
            / max(self.episode_available_pv_energy_kwh, 1e-9)
        )
        degradation_rate = float(
            self.config["bess"].get("degradation_aud_per_kwh_throughput", 0.0)
        )
        bess_degradation_cost = degradation_rate * bess_throughput_kwh
        if converged:
            grid_p_mw = float(self.net.res_ext_grid["p_mw"].sum())
            grid_energy_cost = economic_cost_aud(
                grid_p_mw, price, self.dt_hours, float(self.config["price"]["feed_in_ratio"])
            )
            terminal_inventory_adjustment = 0.0
            if (
                self._step + 1 == self.steps_per_episode
                and bool(self.config["bess"].get("terminal_energy_valuation", False))
            ):
                initial_energy = sum(
                    soc * device.capacity_kwh
                    for soc, device in zip(self.initial_bess_soc, self.batteries)
                )
                final_energy = sum(device.energy_kwh for device in self.batteries)
                terminal_inventory_adjustment = (
                    (initial_energy - final_energy) * float(np.mean(self.price_profile))
                )
            economic = grid_energy_cost + terminal_inventory_adjustment
            if not self.separate_bess_degradation_objective:
                economic += bess_degradation_cost
            voltage, voltage_metrics = voltage_cost(
                self.net.res_bus["vm_pu"].to_numpy(),
                float(self.config["reward"].get("voltage_soft_min_pu", self.config["network"]["v_min"])),
                float(self.config["reward"].get("voltage_soft_max_pu", self.config["network"]["v_max"])),
                float(self.config["reward"]["voltage_alpha"]),
                float(self.config["reward"].get("voltage_worst_bus_weight", 0.0)),
            )
            vm = self.net.res_bus["vm_pu"].to_numpy(dtype=float)
        else:
            grid_p_mw, economic = float("nan"), 0.0
            grid_energy_cost = terminal_inventory_adjustment = 0.0
            voltage = float(self.config["reward"]["pf_failure_voltage_cost"])
            voltage_metrics = {
                "min_vm_pu": float("nan"), "max_vm_pu": float("nan"),
                "mean_abs_voltage_deviation": float("nan"), "voltage_violation_rate": 1.0,
                "any_voltage_violation": 1.0, "voltage_violation_degree": voltage,
                "worst_bus_violation_degree": voltage,
            }
            vm = np.full(len(self.net.bus), np.nan)
        local_absorption_kw = (
            base_load_kw
            + sum(max(-value, 0.0) for value in applied[:self.n_bess])
            + sum(max(-value, 0.0) for value in applied[self.n_bess:])
        )
        pv_export_kw = max(0.0, pv_kw - local_absorption_kw)
        renewable_non_utilisation = (
            (pv_curtailed_kw + pv_export_kw) * self.dt_hours
            / max(self.episode_available_pv_energy_kwh, 1e-9)
        )
        for device, app in zip(self.batteries + self.ev_groups, applied):
            if abs(app) > max(device.p_charge_max_kw, device.p_discharge_max_kw) + 1e-8:
                raise AssertionError("Applied device power exceeded its rating")

        next_boundary = self._step + 1
        departure_cost, step_deficits = self._process_boundary_events(next_boundary)
        departure_aggregation = str(
            self.config["reward"].get("ev_departure_aggregation", "sum")
        )
        if departure_aggregation == "fleet_mean":
            departure_cost /= max(1, self.n_ev)
        elif departure_aggregation != "sum":
            raise ValueError(
                "reward.ev_departure_aggregation must be 'sum' or 'fleet_mean'"
            )
        urgency = sum(
            group.urgency(max(0, spec["departure_step"] - next_boundary), self.dt_hours)
            for group, spec in zip(self.ev_groups, self.ev_session_specs)
        ) / max(1, self.n_ev)
        connected_groups = [group for group in self.ev_groups if group.connected]
        readiness_gap = sum(
            max(0.0, group.required_energy_kwh - group.energy_kwh) / max(group.required_energy_kwh, 1e-9)
            for group in connected_groups
        ) / max(1, len(connected_groups))
        ev_cost = (
            departure_cost
            + float(self.config["reward"]["ev_urgency_beta"]) * urgency
            + float(self.config["reward"].get("ev_readiness_beta", 0.0)) * readiness_gap
        )
        objective_values = [economic, ev_cost, voltage]
        if self.reward_dim == 5:
            if self.five_objective_mode == "battery_renewable":
                objective_values.extend(
                    [battery_degradation, renewable_non_utilisation]
                )
            else:
                objective_values.extend([cbess_efc, pv_curtailment])
        objective_costs = np.asarray(objective_values, dtype=np.float32)
        reward = -objective_costs
        self.previous_actions = clipped.astype(np.float32)

        record: dict[str, Any] = {
            "timestamp": self.episode_timestamps[self._step], "scenario_id": self.scenario_id,
            "network": self.config["network"]["name"], "ev_stress_mode": self.ev_stress_mode,
            "step": self._step, "price": price,
            "base_load_total_kw": base_load_kw,
            "pv_available_total_kw": pv_available_kw,
            "pv_total_kw": pv_kw,
            "pv_curtailed_total_kw": pv_curtailed_kw,
            "pv_export_attributed_kw": pv_export_kw,
            "grid_import_kw": grid_p_mw * 1000 if converged else float("nan"),
            "min_vm_pu": voltage_metrics["min_vm_pu"], "max_vm_pu": voltage_metrics["max_vm_pu"],
            "mean_abs_voltage_deviation": voltage_metrics["mean_abs_voltage_deviation"],
            "voltage_violation_rate": voltage_metrics["voltage_violation_rate"],
            "any_voltage_violation": voltage_metrics["any_voltage_violation"],
            "voltage_violation_degree": voltage_metrics["voltage_violation_degree"],
            "worst_bus_violation_degree": voltage_metrics["worst_bus_violation_degree"],
            "objective_cost_step": economic, "objective_ev_step": ev_cost, "objective_voltage_step": voltage,
            "objective_cbess_degradation_step": cbess_efc,
            "objective_pv_curtailment_step": pv_curtailment,
            "objective_battery_degradation_step": battery_degradation,
            "objective_renewable_non_utilisation_step": renewable_non_utilisation,
            "bess_throughput_kwh_step": bess_throughput_kwh,
            "ev_throughput_kwh_step": ev_throughput_kwh,
            "bess_efc_step": cbess_efc,
            "ev_efc_step": ev_efc,
            "grid_energy_cost_step": grid_energy_cost,
            "bess_degradation_cost_step": bess_degradation_cost,
            "bess_terminal_inventory_adjustment_step": terminal_inventory_adjustment,
            "departure_count_step": len(step_deficits), "departure_deficit_kwh_step": sum(step_deficits),
            "ev_readiness_gap_step": readiness_gap,
            "power_flow_converged": converged,
        }
        for i, (req, app) in enumerate(zip(requested, applied)):
            record[f"requested_action_kw_{i}"] = req; record[f"applied_action_kw_{i}"] = app
        for i, device in enumerate(self.batteries):
            record[f"bess_power_kw_{i}"] = applied[i]; record[f"bess_soc_{i}"] = device.soc
        for i, (device, spec) in enumerate(zip(self.ev_groups, self.ev_session_specs)):
            j = self.n_bess + i
            record[f"ev_power_kw_{i}"] = applied[j]; record[f"ev_energy_kwh_{i}"] = device.energy_kwh
            record[f"ev_connected_{i}"] = device.connected; record[f"ev_required_energy_kwh_{i}"] = device.required_energy_kwh
            record[f"ev_time_to_departure_steps_{i}"] = max(0, spec["departure_step"] - next_boundary) if device.connected else 0
        for bus_index, value in enumerate(vm, start=1):
            record[f"vm_pu_bus_{bus_index}"] = value
        self.logger.append(record)

        self._step = next_boundary
        terminated = not converged
        truncated = self._step >= self.steps_per_episode and not terminated
        # Avoid NaN in intermediate info: Gymnasium's determinism checker treats
        # otherwise identical NaN values as unequal.
        terminal_rate = self._successful_departures / max(1, self._departures)
        info = {
            "episode_scenario_id": self.scenario_id, "objective_costs": objective_costs.copy(),
            "ev_stress_mode": self.ev_stress_mode,
            "requested_action_kw": np.asarray(requested, dtype=np.float32),
            "applied_action_kw": np.asarray(applied, dtype=np.float32),
            "pv_curtailment_fraction": np.asarray(pv_curtailment_fraction, dtype=np.float32),
            "power_flow_converged": converged, "departure_deficits_kwh": step_deficits,
            "departures_so_far": self._departures, "departure_success_rate": terminal_rate,
            **voltage_metrics,
        }
        self._last_render = f"{self.scenario_id} step={self._step} grid={grid_p_mw:.3f} MW V=[{voltage_metrics['min_vm_pu']:.3f}, {voltage_metrics['max_vm_pu']:.3f}]"
        if self.config.get("logging", {}).get("enabled") and (terminated or truncated):
            directory = PROJECT_ROOT / self.config["logging"]["output_dir"]
            self.logger.save(directory / f"{self.config['network']['name']}_{self.scenario_id}.parquet")
        return self._observation(), reward, terminated, truncated, info

    def trajectory_frame(self) -> pd.DataFrame:
        return self.logger.frame()

    def render(self):
        return self._last_render if self.render_mode == "ansi" else None

    def close(self) -> None:
        return None


class ResidentialBESSEVPV5ObjEnv(ResidentialBESSEVEnv):
    """Named five-objective variant with controllable PV curtailment."""

    reward_dim = 5
    objective_names = (
        "economic", "ev_service", "voltage",
        "cbess_degradation", "pv_curtailment",
    )

    def __init__(
        self,
        config: str | Path | dict[str, Any] = "configs/paper_ieee33_v3_5obj.yaml",
        render_mode: str | None = None,
    ):
        super().__init__(config=config, render_mode=render_mode)
        if not self.pv_curtailment_enabled or self.reward_dim != 5:
            raise ValueError(
                "ResidentialBESSEVPV5ObjEnv requires a v3_5obj configuration"
            )


class ResidentialBESSEVRenewables5ObjEnv(ResidentialBESSEVEnv):
    """Five-objective variant with fleet wear and renewable utilisation."""

    reward_dim = 5
    objective_names = (
        "economic", "ev_service", "voltage",
        "battery_degradation", "renewable_non_utilisation",
    )

    def __init__(
        self,
        config: str | Path | dict[str, Any] = "configs/v4/ieee33_balanced.yaml",
        render_mode: str | None = None,
    ):
        super().__init__(config=config, render_mode=render_mode)
        if self.five_objective_mode != "battery_renewable" or self.reward_dim != 5:
            raise ValueError(
                "ResidentialBESSEVRenewables5ObjEnv requires a v4 battery_renewable config"
            )


class ScalarizeReward(gym.Wrapper):
    """Gym checker/traditional-RL adapter that preserves vector rewards."""

    def __init__(self, env: gym.Env, weights: np.ndarray | list[float] | None = None):
        super().__init__(env)
        reward_dim = int(getattr(env.unwrapped, "reward_dim", 3))
        self.weights = np.asarray(
            np.ones(reward_dim, dtype=np.float32) if weights is None else weights,
            dtype=np.float32,
        )
        if self.weights.shape != (reward_dim,):
            raise ValueError(f"Scalarization weights must have shape ({reward_dim},)")

    def step(self, action):
        observation, vector_reward, terminated, truncated, info = self.env.step(action)
        info["vector_reward"] = np.asarray(vector_reward, dtype=np.float32)
        return observation, float(np.dot(self.weights, vector_reward)), terminated, truncated, info
