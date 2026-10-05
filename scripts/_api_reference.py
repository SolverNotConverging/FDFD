"""Render the curated public API inventory from checkout package signatures."""

# Run from the source checkout, including when launched by file path.
import sys as _sys
from pathlib import Path as _Path
_ROOT = next(parent for parent in _Path(__file__).resolve().parents
             if (parent / "fdfd_common" / "__init__.py").is_file())
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from importlib import import_module
import inspect
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESCRIPTIONS = {
    "frequency": "Operating frequency in hertz; finite and positive.",
    "x_range": "Physical x extent or increasing bounds, in metres.",
    "y_range": "Physical y extent or increasing bounds, in metres.",
    "z_range": "Physical z extent or increasing bounds, in metres.",
    "epsilon": "Relative permittivity; supported scalar/tensor forms are described below.",
    "mu": "Relative permeability; supported scalar/diagonal forms are described below.",
    "background_material": "Predefined bulk Material assigned to unfilled space.",
    "boundary": "Predefined PEC or PMC exterior-boundary material.",
    "material": "A predefined bulk, ideal-boundary, or supported SIBC material.",
    "geometry": "A handle returned by add_geometry or a geometry convenience method.",
    "max_element_size": "Maximum initial element edge length in metres.",
    "resolution": "Initial node counts; use instead of a maximum element size.",
    "wavelength_elements": "Minimum number of initial elements per local wavelength.",
    "element_order": "Finite-element polynomial order supported by this backend.",
    "quadrature_order": "Element integration order.",
    "num_modes": "Number of modes requested; positive integer.",
    "neff_guess": "Dimensionless complex effective-index search target.",
    "eigensolver_tolerance": "Algebraic eigensolver convergence tolerance.",
    "linear_solver_tolerance": "Algebraic linear-system residual tolerance.",
    "residual_tolerance": "Maximum accepted eigenproblem residual, separate from adaptation.",
    "divergence_tolerance": "Maximum accepted discrete Gauss-law residual.",
    "max_refinements": "Maximum mesh refinements after the initial solve; zero means one solve.",
    "adaptive_tolerance": "Relative discretization-residual stopping threshold.",
    "thickness": "PML thickness in metres, at each selected exterior end.",
    "order": "Polynomial order of the PML profile.",
    "direction": "Propagation direction for solve; selected coordinate direction for PML.",
    "sigma_max": "Backend PML-strength magnitude; the outgoing stretch has negative imaginary sign.",
    "target_reflection": "Desired PML amplitude reflection ratio in (0, 1).",
    "component": "Field component to display, such as Ey; electrostatics also accepts potential or mesh.",
    "quantity": "Displayed field quantity: real, imag, magnitude/abs, or phase; static fields support real or magnitude.",
    "mode": "Zero-based mode index.",
    "case": "Zero-based sweep case index.",
    "block": "Wait for the interactive viewer to close when true.",
    "path": "Destination/source HDF5 path. Saving is atomic; loading does not run a solver.",
    "frequencies": "Strictly increasing positive frequencies in hertz.",
    "angle": "Physical incidence angle in degrees, strictly between -90 and 90; mutually exclusive with ky.",
    "ky": "Real invariant-direction wavenumber in radians per metre; mutually exclusive with angle.",
    "amplitude": "Complex incident-mode amplitude.",
    "side": "Port label, left or right.",
    "reference_plane": "Incident phase reference position in metres.",
    "left": "Left monitor/reference-plane position in metres.",
    "right": "Right monitor/reference-plane position in metres.",
    "potential": "Prescribed electric potential in volts.",
    "outer_potential": "Exterior potential in volts; None permits natural boundaries.",
    "density": "Volume charge density in coulombs per cubic metre.",
    "region": "Geometry primitive or supported boundary name.",
    "shape": "A predefined fdfd_common.shapes object in metres.",
    "clip": "Intersect the shape with the solver domain; otherwise out-of-bounds objects raise GeometryError.",
    "name": "Optional name used for later identification and diagnostics.",
    "center": "Physical centre coordinates in metres.",
    "radius": "Positive radius in metres.",
    "points": "Ordered polygon vertex coordinates in metres.",
    "material_aware": "Use material-dependent initial mesh sizing.",
    "background": "Include this region in both the unperturbed lead and actual device.",
    "dim": "Electrostatic mesh dimension: 1 or 2.",
    "max_elements": "Adaptive mesh element budget.",
    "marking_fraction": "Fraction of squared error indicators marked for refinement.",
    "Zs": "Surface impedance in ohms; alternatively select a metal preset.",
    "preset": "Metal name for the good-conductor impedance model.",
    "results": "Nonempty sequence of completed periodic mode sets, in sweep order.",
    "number": "Zero-based mode index.",
    "num_points": "Number of field sampling points; positive integer.",
}
TYPES = {"block": "bool", "case": "int", "mode": "int", "number": "int",
    "center": "tuple[float, ...] / m", "points": "sequence[tuple[float, ...]] / m",
    "component": "str | None", "quantity": "str", "path": "str | PathLike",
    "epsilon": "float | complex | array-like / relative", "mu": "float | complex | array-like / relative",
    "x_range": "float | tuple[float, float] / m", "y_range": "float | tuple[float, float] / m",
    "results": "Sequence[PeriodicModeSet]"}


def section(name, character="-"):
    return name+"\n"+character*len(name)+"\n\n"


def entry(name, obj, returned):
    signature = inspect.signature(obj)
    parameters = [p for p in signature.parameters.values() if p.name not in ("self", "cls")]
    signature = signature.replace(parameters=parameters)
    result = section(f"``{name}``", "~")
    result += ".. code-block:: python\n\n    "+name+str(signature)+"\n\n"
    doc = inspect.getdoc(obj)
    if doc:
        first = doc.split("\n\n", 1)[0].replace("\n", " ")
        first = re.sub(r'(?<!`)\|([^|`\n]+)\|(?!`)', r'``|\1|``', first)
        if not any(s in first for s in (":meth:", "legacy", "compatibility")):
            result += first+"\n\n"
    if parameters:
        result += ".. list-table:: Arguments\n   :header-rows: 1\n   :widths: 16 20 12 16 36\n\n   * - Argument\n     - Type / units\n     - Required / optional\n     - Default\n     - Meaning\n"
        for p in parameters:
            annotation = str(p.annotation).replace("typing.", "") if p.annotation is not inspect.Parameter.empty else TYPES.get(p.name, type(p.default).__name__ if p.default is not inspect.Parameter.empty and p.default is not None else "array-like or scalar")
            if annotation.startswith("<class '"): annotation = annotation[8:-2]
            default = "—" if p.default is inspect.Parameter.empty else repr(p.default)
            meaning = DESCRIPTIONS.get(p.name, f"{p.name.replace('_', ' ').capitalize()} control for this operation.")
            result += f"   * - ``{p.name}``\n     - ``{annotation}``\n     - {'Required' if p.default is inspect.Parameter.empty else 'Optional'}\n     - ``{default}``\n     - {meaning}\n"
        result += "\n"
    return result+"Returns: "+returned+".\n\n"
