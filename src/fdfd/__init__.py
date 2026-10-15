"""Finite-difference frequency-domain electromagnetic solvers.

Every solver follows the same workflow::

    import fdfd

    solver = fdfd.ModeSolver2D(frequency=100e9, x_range=(0, 10e-3), y_range=(0, 10e-3))
    solver.add_circle(center=(5e-3, 5e-3), radius=3e-3, material=fdfd.Material(name="core", epsilon=6))
    solver.mesh(resolution=(50, 50))
    result = solver.solve(num_modes=4)
    result.plot(mode=1)
    result.save("modes.h5")
    same = fdfd.load("modes.h5")

Units are SI and phasors use ``exp(+i*omega*t)``; see the conventions page of
the documentation.
"""

from importlib.metadata import PackageNotFoundError, version

from .bands import BandStructureSolver2D, bloch_path
from .core import materials, shapes
from .core.constants import TIME_CONVENTION
from .core.errors import (
    BackendCapabilityError,
    ConfigurationError,
    FDFDError,
    GeometryError,
    NoResultError,
    PersistenceError,
    SolverError,
)
from .core.materials import GoodConductor, Material, SurfaceImpedance
from .core.plotting import plot_dispersion
from .core.pml import suggest_strength as suggest_pml_strength
from .core.resolution import suggest_resolution
from .core.results import BandResult, FieldResult, ModeResult, Result, ScatteringResult, SolveInfo, load
from .periodic import PeriodicModeSolver2D, PeriodicModeSolver3D
from .scattering import ScatteringSolver2D
from .tracking import ModeTracker1D, ModeTracker2D, PortSpec, TrackedSweep, TrackingConfig
from .waveguide import ModeSolver1D, ModeSolver2D

try:
    __version__ = version("fdfd")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0+unknown"

__all__ = [
    "TIME_CONVENTION",
    "BackendCapabilityError",
    "BandResult",
    "BandStructureSolver2D",
    "ConfigurationError",
    "FDFDError",
    "FieldResult",
    "GeometryError",
    "GoodConductor",
    "Material",
    "ModeResult",
    "ModeSolver1D",
    "ModeSolver2D",
    "ModeTracker1D",
    "ModeTracker2D",
    "NoResultError",
    "PeriodicModeSolver2D",
    "PeriodicModeSolver3D",
    "PersistenceError",
    "PortSpec",
    "Result",
    "ScatteringResult",
    "ScatteringSolver2D",
    "SolveInfo",
    "SolverError",
    "SurfaceImpedance",
    "TrackedSweep",
    "TrackingConfig",
    "bloch_path",
    "load",
    "materials",
    "plot_dispersion",
    "shapes",
    "suggest_pml_strength",
    "suggest_resolution",
]
