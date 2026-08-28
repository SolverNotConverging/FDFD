"""Standalone finite-element waveguide mode solvers.

Geometry is placed in physical coordinates first and converted to a conforming
finite-element mesh only by an explicit :meth:`discretize` call.
"""

from .boundaries import (
    METAL_RESISTIVITIES_OHM_M,
    canonical_metal_name,
    good_conductor_surface_impedance,
    validate_surface_impedance,
)
from .exceptions import (
    BackendCapabilityError,
    ConfigurationError,
    FEMModeSolverError,
    GeometryError,
    MeshError,
    NotDiscretizedError,
    SolverError,
    StaleDiscretizationError,
)
from .geometry import (
    BoundaryRegion,
    Circle,
    Interval,
    MeshRefinement,
    PMLSpec,
    Polygon,
    Rectangle,
    Region,
)
from .materials import Material
from .meshing import FEMMesh1D, FEMMesh2D, MeshInfo
from .results import Mode, ModeSet, SampledFields
from .solver_1d import ModeSolver1D
from .solver_2d import ModeSolver2D
from .transmission_lines import (
    Coaxial,
    CoplanarWaveguide,
    Microstrip,
    Stripline,
    TransmissionLineCalculator,
    TransmissionLineCalculatorGUI,
    TransmissionLineFieldViewer,
    TransmissionLineResult,
    launch_transmission_line_calculator,
    visualize_transmission_line,
    visualize_transmission_line_with_gui,
)
from .visualization import ModeViewer, visualize, visualize_with_gui

__version__ = "0.1.0"

__all__ = [
    "BackendCapabilityError",
    "BoundaryRegion",
    "Circle",
    "Coaxial",
    "ConfigurationError",
    "CoplanarWaveguide",
    "FEMMesh1D",
    "FEMMesh2D",
    "FEMModeSolverError",
    "GeometryError",
    "Interval",
    "METAL_RESISTIVITIES_OHM_M",
    "Material",
    "MeshRefinement",
    "MeshError",
    "MeshInfo",
    "Mode",
    "ModeSet",
    "ModeSolver1D",
    "ModeSolver2D",
    "ModeViewer",
    "Microstrip",
    "NotDiscretizedError",
    "PMLSpec",
    "Polygon",
    "Rectangle",
    "Region",
    "SampledFields",
    "SolverError",
    "StaleDiscretizationError",
    "Stripline",
    "TransmissionLineCalculator",
    "TransmissionLineCalculatorGUI",
    "TransmissionLineFieldViewer",
    "TransmissionLineResult",
    "canonical_metal_name",
    "good_conductor_surface_impedance",
    "launch_transmission_line_calculator",
    "validate_surface_impedance",
    "visualize",
    "visualize_transmission_line",
    "visualize_transmission_line_with_gui",
    "visualize_with_gui",
]
