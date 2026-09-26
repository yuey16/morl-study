"""Residential BESS-EV multi-objective distribution-grid environment."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("residential-grid-morl")
except PackageNotFoundError:  # source tree without an editable install
    __version__ = "1.0.0"

from .env import (
    ResidentialBESSEVEnv,
    ResidentialBESSEVPV5ObjEnv,
    ResidentialBESSEVRenewables5ObjEnv,
    ScalarizeReward,
)

__all__ = [
    "ResidentialBESSEVEnv", "ResidentialBESSEVPV5ObjEnv",
    "ResidentialBESSEVRenewables5ObjEnv",
    "ScalarizeReward", "__version__",
]
