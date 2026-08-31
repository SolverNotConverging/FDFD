"""Self-contained finite-element solvers for electromagnetic periodic cells."""

from .assembly_2d import (
    PeriodicFEMSystem2D,
    assemble_periodic_system_2d,
    linearized_pencil,
    solve_qep_candidates,
)
from .assembly_3d import (
    PeriodicFEMSystem3D,
    assemble_periodic_system_3d,
    linearized_pencil_3d,
)
from .exceptions import (
    BackendCapabilityError,
    ConfigurationError,
    FEMPeriodicSolverError,
    GeometryError,
    MeshError,
    NotDiscretizedError,
    PersistenceError,
    SolverError,
    StaleDiscretizationError,
)
from .geometry import (
    BoundaryRegion,
    Box,
    Circle,
    Cylinder,
    GeometryModel2D,
    GeometryModel3D,
    MeshRefinement,
    PMLSpec,
    Polygon,
    Rectangle,
    Region,
    Sphere,
)
from .materials import Material
from .meshing_2d import FEMPeriodicMesh2D, MeshInfo
from .meshing_3d import FEMPeriodicMesh3D, MeshInfo3D, PeriodicMesh3D, discretize_3d
from .persistence import (
    H5ValidationReport,
    PeriodicH5Archive,
    launch_viewer,
    load_periodic_h5,
    open_periodic_h5,
    save_periodic_h5,
    save_periodic_sweep_h5,
    validate_periodic_h5,
)
from .periodic import (
    PeriodicProlongation,
    build_node_prolongation,
    build_signed_edge_prolongation,
    node_representatives,
)
from .results import (
    Mode,
    ModeSet,
    PeriodicMode,
    PeriodicModeSet,
    PeriodicSampledFields,
    SampledFields,
)
from .solver_2d import PeriodicModeSolver2D
from .solver_3d import PeriodicModeSolver3D
from .visualization import visualize

__version__ = "0.1.0"

__all__ = [
    "BackendCapabilityError",
    "BoundaryRegion",
    "Box",
    "Circle",
    "ConfigurationError",
    "Cylinder",
    "FEMPeriodicMesh2D",
    "FEMPeriodicMesh3D",
    "FEMPeriodicSolverError",
    "GeometryError",
    "GeometryModel2D",
    "GeometryModel3D",
    "H5ValidationReport",
    "Material",
    "MeshError",
    "MeshInfo",
    "MeshInfo3D",
    "MeshRefinement",
    "Mode",
    "ModeSet",
    "NotDiscretizedError",
    "PersistenceError",
    "PMLSpec",
    "PeriodicFEMSystem2D",
    "PeriodicFEMSystem3D",
    "PeriodicH5Archive",
    "PeriodicMesh3D",
    "PeriodicMode",
    "PeriodicModeSet",
    "PeriodicModeSolver2D",
    "PeriodicModeSolver3D",
    "PeriodicProlongation",
    "PeriodicSampledFields",
    "Polygon",
    "Rectangle",
    "Region",
    "SampledFields",
    "SolverError",
    "Sphere",
    "StaleDiscretizationError",
    "assemble_periodic_system_2d",
    "assemble_periodic_system_3d",
    "build_node_prolongation",
    "build_signed_edge_prolongation",
    "linearized_pencil",
    "linearized_pencil_3d",
    "launch_viewer",
    "load_periodic_h5",
    "node_representatives",
    "open_periodic_h5",
    "save_periodic_h5",
    "save_periodic_sweep_h5",
    "solve_qep_candidates",
    "visualize",
    "validate_periodic_h5",
    "discretize_3d",
]
