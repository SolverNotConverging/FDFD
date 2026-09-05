"""Central physical constants used by FEM Waveguide Scattering.

Public geometry and frequency inputs use SI units.  Constants are sourced
from :mod:`scipy.constants` so every solver layer shares one CODATA dataset.
"""

from __future__ import annotations

from typing import Final

from scipy.constants import c, epsilon_0, mu_0

C0: Final[float] = float(c)
"""Vacuum speed of light in metres per second (exact in SI)."""

EPSILON_0: Final[float] = float(epsilon_0)
"""Vacuum electric permittivity in farads per metre."""

MU_0: Final[float] = float(mu_0)
"""Vacuum magnetic permeability in henries per metre."""

ETA_0: Final[float] = (MU_0 / EPSILON_0) ** 0.5
"""Vacuum wave impedance in ohms."""

# Descriptive aliases are useful at public boundaries and in generated
# metadata, while the compact names keep the weak forms readable.
SPEED_OF_LIGHT_M_PER_S: Final[float] = C0
VACUUM_PERMITTIVITY_F_PER_M: Final[float] = EPSILON_0
VACUUM_PERMEABILITY_H_PER_M: Final[float] = MU_0
VACUUM_IMPEDANCE_OHM: Final[float] = ETA_0

__all__ = [
    "C0",
    "EPSILON_0",
    "ETA_0",
    "MU_0",
    "SPEED_OF_LIGHT_M_PER_S",
    "VACUUM_IMPEDANCE_OHM",
    "VACUUM_PERMEABILITY_H_PER_M",
    "VACUUM_PERMITTIVITY_F_PER_M",
]
