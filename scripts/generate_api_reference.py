"""Render the curated public API inventory from installed package signatures."""
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
    "shape": "A predefined cem_common.shapes object in metres.",
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


def main():
    inventory = json.loads((ROOT/'doc/public_api.json').read_text())
    for package, spec in inventory.items():
        module = import_module(package)
        out = section(package+" user API", "=")
        out += "Version 1.0.0. This reference covers the deliberately supported user API.\nImplementation helpers are documented in their source modules.\n\n"
        out += section("Workflow and units")
        out += ("All lengths are in metres, frequencies in hertz, and constitutive values are relative.\n"
                "Construction and configuration use keyword arguments. Call ``mesh()`` to build\n"
                "the initial mesh, ``solve()`` to obtain and store a typed result, and ``show()``\n"
                "to inspect it interactively. ``mesh_data`` and ``result`` are initially None.\n"
                "Geometry edits invalidate both; remeshing invalidates the result. Automatic\n"
                "meshing reuses the last explicit settings. Calling ``show()`` without a result\n"
                "raises ``NoResultError``.\n\n"
                "Adaptive defaults are ``max_refinements=2`` and ``adaptive_tolerance=0.05``.\n"
                "Zero refinements performs one solve. ``solve_info`` distinguishes algebraic\n"
                "residuals from adaptive discretization residuals and records stopping reasons.\n"
                "Python mode and case indices are zero-based.\n\n"
                "``solve()`` and frequency sweeps neither save nor open windows. Call\n"
                "``result.save(path)`` explicitly. ``load_result(path)`` returns inspection-ready\n"
                "results without solving. Archives use ``cem-fem-results`` schema ``1.0``;\n"
                "old or convention-incompatible files raise ``PersistenceError``. Loaded\n"
                "results can plot, show, and save; they do not restart a solver or restore callbacks.\n"
                "``plot()`` returns a Matplotlib Figure without opening a window.\n"
                "``show(block=True)`` waits for the viewer; ``block=False`` returns immediately.\n\n")
        if package=='fem_electrostatics':
            out += "Electrostatic fields are static. ``epsilon`` accepts a positive scalar, a\npositive diagonal, or a real symmetric positive-definite tensor. Potential is\nnodal; ``element_electric_field`` and ``element_displacement_field`` are cell\nfields. ``electric_field`` and ``displacement_field`` are nodal averages.\n\n"
        else:
            out += "The time convention is ``exp(+i*omega*t)`` with guided propagation\n``exp(-i*beta*z)``. Passive materials have nonpositive imaginary constitutive\nvalues; passive forward attenuation is ``-Im(beta)``.\n\n"
            out += ("Scattering uses a two-dimensional x/z mesh for **2.5D full-vector** fields,\nwith invariant factor ``exp(-i*ky*y)``. Scalar epsilon and mu are supported.\nIntegrated power accounting requires passive materials and a lossless uniform\nlead. Port modes remain a separate implementation from standalone mode solvers.\nPML supports x, z, or all; it applies at both ends of the selected axis.\n\n" if package=='fem_waveguide_scattering' else
                    "Materials support scalar or diagonal epsilon and mu. Unsupported off-diagonal\nforms raise an explicit configuration error. Periodic archives store periodic\nfield envelopes; the Bloch phase is recorded separately from these fields.\n\n")
        out += section("Supported exports")+", ".join(f"``{name}``" for name in spec['exports'])+".\n\n"
        out += section("Solver construction and operations")
        for solver, methods in spec['solvers'].items():
            cls=getattr(module,solver)
            for method in methods:
                name=solver if method=='__init__' else solver+'.'+method
                obj=cls if method=='__init__' else getattr(cls,method)
                returned={'__init__':f"a configured ``{solver}``",'mesh':'the initial mesh stored in ``mesh_data``','solve':'the physics-specific result stored in ``result``','show':'the viewer controller or native process','sweep':'a frequency sweep result'}.get(method,'the configured geometry/excitation handle, or None for in-place configuration')
                out += entry(name,obj,returned)
        out += section("Returned results")
        out += "Result objects are returned by solving or loading. Their constructors are\nimplementation details. Inspect ``mesh_data``, ``metadata``, ``solve_info``, and\n``frequency`` where applicable. Modal sets support iteration, indexing, ``neff``\nand ``beta``; each mode provides sampled fields and residual diagnostics.\n\n"
        for kind,methods in spec['results'].items():
            cls=getattr(module,kind)
            for method in methods:
                out+=entry(kind+'.'+method,getattr(cls,method),{'save':'the written Path','plot':'a Matplotlib Figure','show':'the viewer controller or native process','mode':'the selected mode','deembed':'a result with updated reference planes','conductor_charge':'conductor charge in coulombs (per metre for a 2D cross-section)'}.get(method,'the selected data or diagnostic report'))
        out += section("Result data and diagnostics")
        out += "``mesh_data.coordinates`` stores physical nodes in metres; ``elements`` stores\nzero-based connectivity. ``axes`` identifies physical coordinate order.\n``mesh_data.metadata['context']`` records material and boundary configuration.\nThe result is an inspection snapshot; editing it cannot restart a solver.\n\n"
        if package == 'fem_electrostatics':
            out += "``potential`` is nodal potential in V. ``electric_field`` and\n``displacement_field`` are recovered nodal arrays (V/m and C/m²);\n``element_electric_field`` and ``element_displacement_field`` retain cell fields.\n``conductor_charges`` maps configured conductor names to charge, and\n``energy`` records electrostatic energy (per unit transverse area in 1D,\nper unit length in 2D). ``residual_norm`` is the algebraic residual;\n``adaptive_history`` records mesh sizes, error indicators, and stopping status.\n\n"
        elif package == 'fem_waveguide_scattering':
            out += "``E_total``, ``E_incident``, ``E_scattered`` and the matching ``H_*``\narrays have shape (3, samples), in V/m and A/m. ``coordinates`` uses (x, z),\nin metres. ``S(side, out_mode, in_mode)`` returns a complex modal amplitude;\n``S11`` and ``S21`` select the fundamental ports. ``reflection``,\n``transmission``, ``absorption`` and ``power_balance_error`` are power ratios.\n``reference_planes`` stores positions in m; ``port_betas`` stores complex\nwavenumbers in rad/m. ``solve_info`` retains projection, algebraic, and\nadaptive diagnostics. Sweep ``results[index]`` loads one case;\n``frequencies_hz`` is the ordered frequency array. Sweep plotting accepts\nS11/S21 and quantity real, imag, phase, magnitude, abs, or db.\n\n"
        else:
            out += "Each selected mode exposes ``neff`` (dimensionless), ``beta`` (rad/m),\n``fields`` (sampled E/H in V/m and A/m), ``coefficients`` (FEM expansion),\nand ``residual`` (algebraic eigenproblem residual). Field component names\ninclude Ex, Ey, Ez, Hx, Hy, and Hz. Coordinate and cell ownership arrays\npreserve field locations; they must not be interpreted as interchangeable\nnodal or cell values. ``solve_info`` records adaptive history separately.\n\n"
            if package == 'fem_periodic_modes':
                out += "Periodic fields are Bloch envelopes. ``period`` is in m; each mode also\nprovides ``gamma``, ``bloch_multiplier``, folded propagation quantities,\nand Gauss-law/PML filtering diagnostics. Combine solved cases with\n``PeriodicSweepResult.from_results(results)`` and call ``save(path)``.\nLoaded multi-case archives index cases lazily.\n\n"
        out += section("Geometry and material values")
        out += ("Define reusable materials and shapes with ``cem_common`` before assigning them.\n"
                "Use ``Material(name=..., epsilon=..., mu=...)`` for bulk media,\n"
                "``materials.PEC`` or ``materials.PMC`` for ideal boundaries, and the\n"
                "documented ``materials.copper``-style presets where SIBC is supported.\n"
                "Continuous primitives and Boolean/transformed shapes live in\n"
                "``cem_common.shapes``. Solver packages do not re-export these shared values.\n\n")
        out += entry('load_result',module.load_result,'a typed result; multi-case archives provide lazy case access')
        out += section("Errors")+"Invalid inputs raise ``ConfigurationError`` or ``GeometryError`` where available.\nMesh and numerical failures raise the corresponding ``MeshError`` or\n``SolverError``. ``NoResultError`` requires a successful solve first.\n``PersistenceError`` identifies an incompatible or unreadable archive.\nViewer errors include the executable path or installation setting needed to\ncorrect a launch failure. Saving and loading do not require an active GUI.\n\n"
        (ROOT/spec['documentation']/'API_REFERENCE.rst').write_text(out.rstrip() + '\n',encoding='utf-8')


if __name__=='__main__':main()
