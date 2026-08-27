"""Full-vector 2.5D Maxwell FEM waveguide scattering.

WaveFEM uses ``exp(-i*omega*t)`` in time and ``exp(+i*beta*z+i*ky*y)`` for
guided propagation.  Public lengths and frequencies are expressed in SI
units.
"""

from __future__ import annotations

from .constants import C0, EPSILON_0, ETA_0, MU_0
from .exceptions import (
    ConfigurationError,
    MaterialError,
    MeshError,
    ModeProjectionError,
    ModeSolverError,
    SolverError,
    WaveFEMError,
)
from .frequency import Frequency, resolve_frequency
from .hdf5 import (
    H5FileData,
    H5ModeData,
    H5ResultData,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    load_h5,
    save_result_h5,
    save_sweep_h5,
)
from .incident import IncidentMode
from .materials import Material
from .modes import CrossSection, Mode, ModeSet, ModeSolver
from .pml import PML, PMLLayout
from .results import Diagnostic, DiagnosticReport, ScatteringResult
from .scattering import Scattering2D, SolverOptions
from .sweep import FrequencySweepResult

__version__ = "0.0.1"

__all__ = [
    "C0",
    "ConfigurationError",
    "CrossSection",
    "Diagnostic",
    "DiagnosticReport",
    "EPSILON_0",
    "ETA_0",
    "Frequency",
    "FrequencySweepResult",
    "H5FileData",
    "H5ModeData",
    "H5ResultData",
    "IncidentMode",
    "MU_0",
    "Material",
    "MaterialError",
    "MeshError",
    "Mode",
    "ModeProjectionError",
    "ModeSet",
    "ModeSolver",
    "ModeSolverError",
    "PML",
    "PMLLayout",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "Scattering2D",
    "ScatteringResult",
    "SolverOptions",
    "SolverError",
    "WaveFEMError",
    "load_h5",
    "resolve_frequency",
    "save_result_h5",
    "save_sweep_h5",
]
