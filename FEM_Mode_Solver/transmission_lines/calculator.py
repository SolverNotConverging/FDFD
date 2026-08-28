"""High-level lifecycle for quasi-TEM FEM transmission-line calculations."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..exceptions import ConfigurationError, NotDiscretizedError
from ..meshing import FEMMesh2D
from .electrostatics import solve_quasi_tem
from .results import TransmissionLineResult
from .specs import (
    Coaxial,
    CoplanarWaveguide,
    Microstrip,
    Stripline,
    TransmissionLineSpec,
    spec_from_type,
)
from .templates import BuiltTransmissionLine, build_transmission_line


def _positive_frequency(value: Any) -> float:
    if isinstance(value, (bool, np.bool_, str, bytes)):
        raise ConfigurationError("frequency must be finite and positive.")
    try:
        frequency = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError("frequency must be finite and positive.") from exc
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ConfigurationError("frequency must be finite and positive.")
    return frequency


class TransmissionLineCalculator:
    """FEM calculator for one TEM or quasi-TEM transmission-line cross-section.

    Geometry is constructed immediately from an immutable line specification,
    while meshing remains explicit.  The lifecycle deliberately mirrors the
    full-vector mode solvers::

        calculator -> discretize() -> solve() -> inspect/visualize
                              |-> refine() -> solve()

    The field solve is a two-potential P1 FEM formulation.  A dielectric solve
    supplies the electric field and capacitance; a vacuum dual solve supplies
    the unit-current magnetic field and external inductance.  This is the
    standard quasi-TEM formulation for nonmagnetic transmission lines.
    """

    def __init__(self, spec: TransmissionLineSpec, *, frequency: float) -> None:
        if not isinstance(spec, (Coaxial, Microstrip, Stripline, CoplanarWaveguide)):
            raise TypeError(
                "spec must be Coaxial, Microstrip, Stripline, or "
                "CoplanarWaveguide."
            )
        self.spec = spec
        self.frequency = _positive_frequency(frequency)
        self._built: BuiltTransmissionLine = build_transmission_line(
            spec,
            self.frequency,
        )
        self._solution: TransmissionLineResult | None = None
        self._solution_mesh_data: FEMMesh2D | None = None
        self._quadrature_order = 4

    @classmethod
    def from_type(
        cls,
        line_type: str,
        *,
        frequency: float,
        **parameters: Any,
    ) -> "TransmissionLineCalculator":
        """Construct a calculator from a user-facing line-type name."""

        return cls(spec_from_type(line_type, **parameters), frequency=frequency)

    @classmethod
    def coaxial(cls, *, frequency: float, **parameters: Any) -> "TransmissionLineCalculator":
        return cls(Coaxial(**parameters), frequency=frequency)

    @classmethod
    def microstrip(cls, *, frequency: float, **parameters: Any) -> "TransmissionLineCalculator":
        return cls(Microstrip(**parameters), frequency=frequency)

    @classmethod
    def stripline(cls, *, frequency: float, **parameters: Any) -> "TransmissionLineCalculator":
        return cls(Stripline(**parameters), frequency=frequency)

    @classmethod
    def coplanar_waveguide(
        cls,
        *,
        frequency: float,
        **parameters: Any,
    ) -> "TransmissionLineCalculator":
        """Construct the signal-to-tied-grounds CPW mode requested by the API."""

        return cls(CoplanarWaveguide(**parameters), frequency=frequency)

    @property
    def solver(self) -> Any:
        """Underlying geometry/mesh owner used by the common FEM backend."""

        return self._built.solver

    @property
    def discretized(self) -> bool:
        return bool(self.solver.discretized)

    @property
    def mesh(self) -> FEMMesh2D:
        return self.solver.mesh

    @property
    def mesh_data(self) -> FEMMesh2D:
        return self.solver.mesh_data

    @property
    def solution(self) -> TransmissionLineResult | None:
        # An advanced caller may edit ``calculator.solver.geometry`` directly.
        # Do not expose a result tied to a mesh invalidated by such an edit or
        # replaced through the publicly exposed underlying solver.
        current_mesh_data = (
            self.solver.mesh_data if self.solver.discretized else None
        )
        if (
            self._solution is not None
            and current_mesh_data is not self._solution_mesh_data
        ):
            self._solution = None
            self._solution_mesh_data = None
        return self._solution

    @property
    def result(self) -> TransmissionLineResult:
        result = self.solution
        if result is None:
            raise RuntimeError("solve() must be called before reading result.")
        return result

    def discretize(
        self,
        *,
        max_element_size: float | None = None,
        resolution: tuple[int, int] | None = None,
        wavelength_elements: int = 10,
        material_aware: bool = True,
        interface_refinement: float | None = 0.6,
        interface_refinement_width: float | None = None,
        boundary_refinement: float | None = 0.4,
        boundary_refinement_width: float | None = None,
        element_order: int = 1,
        quadrature_order: int = 4,
    ) -> FEMMesh2D:
        """Generate the conforming mesh after all template objects are placed."""

        mesh = self.solver.discretize(
            max_element_size=max_element_size,
            resolution=resolution,
            wavelength_elements=wavelength_elements,
            material_aware=material_aware,
            interface_refinement=interface_refinement,
            interface_refinement_width=interface_refinement_width,
            boundary_refinement=boundary_refinement,
            boundary_refinement_width=boundary_refinement_width,
            element_order=element_order,
            quadrature_order=quadrature_order,
        )
        self._quadrature_order = int(quadrature_order)
        self._solution = None
        self._solution_mesh_data = None
        return mesh

    def refine(self, factor: float = 2.0) -> FEMMesh2D:
        """Remesh all dielectric interfaces and conductor walls more densely."""

        mesh = self.solver.refine(factor=factor)
        self._solution = None
        self._solution_mesh_data = None
        return mesh

    def solve(self) -> TransmissionLineResult:
        """Solve the quasi-TEM FEM potentials and extract modal line metrics."""

        if not self.discretized:
            raise NotDiscretizedError(
                "The transmission-line scene has not been discretized; "
                "call discretize() before solve()."
            )
        fem_solution = solve_quasi_tem(
            self._built,
            frequency=self.frequency,
            quadrature_order=self._quadrature_order,
        )
        result = TransmissionLineResult.from_solution(
            self.spec,
            self._built,
            fem_solution,
            frequency=self.frequency,
        )
        self._solution = result
        self._solution_mesh_data = self.solver.mesh_data
        return result

    def visualize(self, **kwargs: Any) -> Any:
        """Plot both transverse vector fields for the last solution."""

        return self.result.visualize(**kwargs)

    def visualize_with_gui(self, **kwargs: Any) -> Any:
        """Open the phase/mesh-controlled transverse vector-field viewer."""

        return self.result.visualize_with_gui(**kwargs)


__all__ = ["TransmissionLineCalculator"]
