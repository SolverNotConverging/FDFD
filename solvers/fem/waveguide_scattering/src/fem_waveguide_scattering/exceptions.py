"""FEM Waveguide Scattering exception hierarchy."""

from __future__ import annotations

from cem_common import errors as _common




class FEMWaveguideScatteringError(_common.CEMError):
    """Base class for actionable FEM Waveguide Scattering errors."""


class ConfigurationError(FEMWaveguideScatteringError, _common.ConfigurationError):
    """The requested simulation configuration is incomplete or inconsistent."""


class MaterialError(ConfigurationError):
    """A material value or constitutive representation is invalid."""


class MeshError(FEMWaveguideScatteringError, _common.MeshError):
    """Mesh generation, import, or physical-region tagging failed."""


class ModeSolverError(FEMWaveguideScatteringError):
    """The guided-mode eigenproblem could not produce valid requested modes."""


class ModeProjectionError(FEMWaveguideScatteringError):
    """Fields could not be projected reliably onto the requested lead modes."""


class SolverError(FEMWaveguideScatteringError, _common.SolverError):
    """A finite-element linear or eigenvalue solve failed."""


class ViewerError(FEMWaveguideScatteringError, _common.ViewerError):
    """The standalone native viewer could not be found or launched."""


__all__ = [
    "ConfigurationError",
    "MaterialError",
    "MeshError",
    "ModeProjectionError",
    "ModeSolverError",
    "SolverError",
    "ViewerError",
    "FEMWaveguideScatteringError",
]
