"""Finite-element TEM and quasi-TEM transmission-line calculator."""

from .calculator import TransmissionLineCalculator
from .electrostatics import QuasiTEMSolution, solve_quasi_tem
from .gui import TransmissionLineCalculatorGUI, launch_transmission_line_calculator
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
from .visualization import (
    TransmissionLineFieldViewer,
    visualize_transmission_line,
    visualize_transmission_line_with_gui,
)

__all__ = [
    "BuiltTransmissionLine",
    "Coaxial",
    "CoplanarWaveguide",
    "Microstrip",
    "QuasiTEMSolution",
    "Stripline",
    "TransmissionLineCalculator",
    "TransmissionLineCalculatorGUI",
    "TransmissionLineFieldViewer",
    "TransmissionLineResult",
    "TransmissionLineSpec",
    "build_transmission_line",
    "launch_transmission_line_calculator",
    "solve_quasi_tem",
    "spec_from_type",
    "visualize_transmission_line",
    "visualize_transmission_line_with_gui",
]
