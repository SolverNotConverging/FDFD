"""One vocabulary for every solver: construction, meshing, PML, solve, results.

Each entry builds a tiny, fast problem; the same assertions then run against
every solver family so the public API cannot drift apart again.
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass, fields

import numpy as np
import pytest
from matplotlib.figure import Figure

import fdfd
from fdfd import Material, materials, shapes
from fdfd.core.constants import C0

DIELECTRIC = Material(name="dielectric", epsilon=2.25)


@dataclass
class Case:
    name: str
    build: Callable
    solve: Callable
    pml: bool
    polarization: str | None
    modal: bool = True


def waveguide_1d():
    s = fdfd.ModeSolver1D(frequency=C0, x_range=(0.0, 1.0), background_material=DIELECTRIC)
    s.add_layer(x_range=(0.0, 0.05), material=materials.PEC, name="left")
    s.add_layer(x_range=(0.95, 1.0), material=materials.PEC, name="right")
    return s


def waveguide_2d():
    s = fdfd.ModeSolver2D(frequency=C0, x_range=(0.0, 1.0), y_range=(0.0, 0.5))
    s.add_rectangle(x_range=(0.3, 0.7), y_range=(0.1, 0.4), material=DIELECTRIC)
    return s


def periodic_2d():
    return fdfd.PeriodicModeSolver2D(
        frequency=C0, x_range=(0.0, 1.0), z_range=(0.0, 0.25), background_material=DIELECTRIC
    )


def periodic_3d():
    return fdfd.PeriodicModeSolver3D(
        frequency=C0,
        x_range=(0.0, 1.0),
        y_range=(0.0, 0.5),
        z_range=(0.0, 0.25),
        background_material=DIELECTRIC,
    )


def bands():
    s = fdfd.BandStructureSolver2D(x_range=(0.0, 1.0), y_range=(0.0, 1.0))
    s.add_circle(center=(0.5, 0.5), radius=0.2, material=Material(name="rod", epsilon=4.0))
    return s


def scattering():
    s = fdfd.ScatteringSolver2D(frequency=C0, x_range=(-1.0, 1.0), y_range=(-1.0, 1.0))
    s.add_circle(center=(0.0, 0.0), radius=0.2, material=Material(name="cylinder", epsilon=2.0))
    s.set_source(angle=0.0, inset=0.3)
    return s


def tracker_1d():
    s = fdfd.ModeTracker1D(
        frequencies=(1.0 * C0, 1.1 * C0),
        x_range=(0.0, 1.0),
        background_material=DIELECTRIC,
        port=fdfd.PortSpec(boundary="enclosed"),
    )
    s.add_layer(x_range=(0.0, 0.05), material=materials.PEC, name="left")
    s.add_layer(x_range=(0.95, 1.0), material=materials.PEC, name="right")
    return s


def tracker_2d():
    s = fdfd.ModeTracker2D(
        frequencies=(0.9 * C0, 1.0 * C0),
        x_range=(0.0, 1.0),
        y_range=(0.0, 0.5),
        port=fdfd.PortSpec(boundary="enclosed"),
    )
    frame = shapes.Difference(
        shape=shapes.Rectangle(bounds=((0.0, 1.0), (0.0, 0.5))),
        tool=shapes.Rectangle(bounds=((0.1, 0.9), (0.1, 0.4))),
    )
    s.add_geometry(shape=frame, material=materials.PEC)
    return s


PATH = np.array([[0.0, np.pi / 2, np.pi], [0.0, 0.0, 0.0]])

CASES = [
    Case("ModeSolver1D", waveguide_1d, lambda s: s.solve(num_modes=2), True, "both"),
    Case("ModeSolver2D", waveguide_2d, lambda s: s.solve(num_modes=2), True, None),
    Case("PeriodicModeSolver2D", periodic_2d, lambda s: s.solve(num_modes=2, guess=1.4), True, "both"),
    Case("PeriodicModeSolver3D", periodic_3d, lambda s: s.solve(num_modes=1, guess=1.4), True, None),
    Case(
        "BandStructureSolver2D",
        bands,
        lambda s: s.solve(beta_path=PATH, num_modes=2, progress=False),
        False,
        "both",
    ),
    Case("ScatteringSolver2D", scattering, lambda s: s.solve(), True, "TE", modal=False),
    Case("ModeTracker1D", tracker_1d, lambda s: s.solve(num_modes=2, progress=False), False, "both"),
    Case("ModeTracker2D", tracker_2d, lambda s: s.solve(num_modes=2, progress=False), False, None),
]
RESOLUTIONS = {1: 20, 2: (10, 6), 3: (5, 4, 4)}


@pytest.fixture(params=CASES, ids=[c.name for c in CASES])
def case(request):
    return request.param


def test_every_public_argument_is_keyword_only(case):
    solver = case.build()
    for callable_ in (type(solver), solver.mesh, solver.solve, solver.add_geometry):
        for parameter in inspect.signature(callable_).parameters.values():
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, (callable_, parameter)


def test_mesh_has_one_signature_and_default(case):
    signature = inspect.signature(case.build().mesh)
    assert {name: p.default for name, p in signature.parameters.items()} == {
        "resolution": None,
        "max_element_size": None,
        "subpixels": 8,
    }


def test_pml_has_one_signature_or_reports_no_support(case):
    solver = case.build()
    signature = inspect.signature(solver.add_pml)
    assert {name: p.default for name, p in signature.parameters.items()} == {
        "thickness": inspect.Parameter.empty,
        "sides": "all",
        "strength": 5.0,
        "order": 3,
    }
    if not case.pml:
        with pytest.raises(fdfd.BackendCapabilityError):
            solver.add_pml(thickness=0.1)


def test_solve_vocabulary(case):
    parameters = inspect.signature(case.build().solve).parameters
    assert parameters["device"].default == "cpu"
    if case.modal:
        assert parameters["num_modes"].default == 4
        assert parameters["guess"].default is None
        assert parameters["tol"].default == 0.0
    if case.polarization is None:
        assert "polarization" not in parameters
    else:
        assert parameters["polarization"].default == case.polarization
    retired = {
        "backend",
        "neff_guess",
        "eigenvalue_guess",
        "eigensolver_tolerance",
        "eigensolver",
        "polarizations",
        "reuse_factorization",
        "tracking_config",
    }
    assert not retired & set(parameters)


def test_results_share_info_plot_save_and_load(case, tmp_path):
    solver = case.build()
    grid = solver.mesh(resolution=RESOLUTIONS[len(solver._physical_axes)])
    assert solver.grid is grid and grid.subpixels == 8
    result = case.solve(solver)
    assert solver.result is result
    assert isinstance(result, fdfd.Result)
    assert result.family == type(solver).family
    assert isinstance(result.info, fdfd.SolveInfo)
    assert [f.name for f in fields(result.info)] == [
        "device",
        "method",
        "tol",
        "guess",
        "residuals",
        "field_residuals",
    ]
    assert result.info.device == "cpu"
    assert np.all(np.isfinite(result.info.residuals))
    if case.modal:
        assert result.info.converged
    figure = result.plot(mode=None)
    assert isinstance(figure, Figure)
    restored = fdfd.load(result.save(tmp_path / "result.h5"))
    assert type(restored) is type(result)
    assert restored.family == result.family
    np.testing.assert_array_equal(restored.info.residuals, result.info.residuals)


def test_geometry_edits_invalidate_mesh_and_result(case):
    solver = case.build()
    solver.mesh(resolution=RESOLUTIONS[len(solver._physical_axes)])
    handle = solver.objects[-1] if solver.objects else None
    if handle is None:
        pytest.skip("solver has no geometry")
    solver.set_material(geometry=handle, material=handle.material)
    assert solver.result is None


def test_no_solver_exposes_internal_backends(case):
    solver = case.build()
    assert not hasattr(solver, "_backend")
    for legacy in ("add_source", "set_source_region", "make_bloch_path", "visualize", "visualize_with_gui"):
        assert not hasattr(solver, legacy)


def solve_on(case, solver, device):
    if isinstance(solver, fdfd.BandStructureSolver2D):
        return solver.solve(beta_path=PATH, num_modes=1, device=device, progress=False)
    if isinstance(solver, (fdfd.ModeTracker1D, fdfd.ModeTracker2D)):
        return solver.solve(num_modes=1, device=device, progress=False)
    if isinstance(solver, (fdfd.PeriodicModeSolver2D, fdfd.PeriodicModeSolver3D)):
        return solver.solve(num_modes=1, guess=1.4, device=device)
    return case.solve(solver) if device == "cpu" else solver.solve(device=device)


def test_unknown_device_is_rejected(case):
    solver = case.build()
    solver.mesh(resolution=RESOLUTIONS[len(solver._physical_axes)])
    with pytest.raises(fdfd.ConfigurationError, match="device"):
        solve_on(case, solver, "gpu")


def test_auto_device_falls_back_to_cpu_without_cuda(case, monkeypatch):
    from fdfd.core import cuda

    monkeypatch.setattr(cuda, "available", lambda: False)
    solver = case.build()
    solver.mesh(resolution=RESOLUTIONS[len(solver._physical_axes)])
    with pytest.warns(RuntimeWarning, match="falling back to CPU"):
        assert solve_on(case, solver, "auto").info.device == "cpu"


def test_ranges_must_be_pairs():
    with pytest.raises(fdfd.ConfigurationError, match="pair"):
        fdfd.ModeSolver1D(frequency=1e9, x_range=1e-3)


def test_one_loader_and_one_version():
    assert fdfd.__version__ != "0+unknown"
    assert not hasattr(fdfd.waveguide, "load_result")
    assert fdfd.TIME_CONVENTION == "exp(+i*omega*t)"
