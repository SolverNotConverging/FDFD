"""Immutable, solver-independent records returned by the HDF5 reader."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
IndexArray = NDArray[np.int64]
PortKey = tuple[str, int, int]
SceneLineKind = Literal["pec", "pmc", "wave_port", "pml"]


@dataclass(frozen=True, slots=True)
class SceneLine:
    """One line overlay in physical ``(x, z)`` coordinates.

    ``endpoints`` has shape ``(2, 2)``.  Its rows are the two endpoints and
    its columns are ``x`` and ``z`` respectively.
    """

    kind: SceneLineKind
    endpoints: FloatArray
    label: str


@dataclass(frozen=True, slots=True)
class SceneData:
    """Material mesh and boundary/port overlays for a vector-field plot.

    ``points`` has shape ``(2, npoints)`` in ``(x, z)`` order,
    ``triangles`` has shape ``(3, nelements)``, and ``eps_r`` contains one
    relative-permittivity value per triangle.
    """

    points: FloatArray
    triangles: IndexArray
    eps_r: ComplexArray
    x_span: FloatArray
    z_span: FloatArray
    lines: tuple[SceneLine, ...]


@dataclass(frozen=True, slots=True)
class ModeData:
    """One sampled modal electric and magnetic field."""

    x: FloatArray
    E: ComplexArray
    H: ComplexArray
    metadata: Mapping[str, Any]
    raw_components: Mapping[str, NDArray[Any]]


@dataclass(frozen=True, slots=True)
class ResultData:
    """Fields and observables for one saved simulation frequency."""

    frequency_hz: float | None
    ky: float | None
    coordinates: FloatArray
    E_incident: ComplexArray
    E_scattered: ComplexArray
    E_total: ComplexArray
    H_incident: ComplexArray
    H_scattered: ComplexArray
    H_total: ComplexArray
    s_parameters: Mapping[PortKey, complex]
    powers: Mapping[str, float]
    modes: tuple[ModeData, ...]
    metadata: Mapping[str, Any]
    scene: SceneData | None


@dataclass(frozen=True, slots=True)
class FileData:
    """Top-level contents of one WaveFEM schema-v1 HDF5 file."""

    path: Path
    kind: Literal["single", "sweep"]
    frequencies_hz: FloatArray
    results: tuple[ResultData, ...]


__all__ = [
    "ComplexArray",
    "FileData",
    "FloatArray",
    "IndexArray",
    "ModeData",
    "PortKey",
    "ResultData",
    "SceneData",
    "SceneLine",
    "SceneLineKind",
]
