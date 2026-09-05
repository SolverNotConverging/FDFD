"""Cross-family checks for the 1.0 material-first public contract."""

from __future__ import annotations

import inspect

import inspect

import numpy as np
import pytest

from cem_common import Material, SurfaceImpedance, materials, shapes
from cem_common.errors import BackendCapabilityError, ConfigurationError, GeometryError


def test_materials_are_named_reusable_values_with_exp_plus_iwt_loss_sign() -> None:
    dielectric = Material(name="lossy dielectric", epsilon=2.5 - 0.03j, mu=1.0)

    assert dielectric.is_passive
    assert not Material(name="active", epsilon=2.5 + 0.03j).is_passive
    assert materials.vacuum.name == "vacuum"
    assert materials.air.is_passive
    assert materials.copper.at_frequency(frequency=10e9).real > 0.0
    assert materials.copper.at_frequency(frequency=10e9).imag > 0.0
    assert SurfaceImpedance(impedance=1.0 + 1.0j).at_frequency(frequency=10e9) == 1.0 + 1.0j

    with pytest.raises(ConfigurationError):
        Material(name="", epsilon=1.0)
    with pytest.raises(BackendCapabilityError, match="off-diagonal"):
        materials.bulk_values(
            Material(name="tensor", epsilon=((2.0, 0.1), (0.1, 2.0))),
            dimension=2,
        )


def test_shared_shapes_support_boolean_and_transformed_geometry() -> None:
    disk = shapes.Circle(center=(0.0, 0.0), radius=1.0)
    aperture = shapes.Rectangle(bounds=((-0.25, 0.25), (-2.0, 2.0)))
    split_disk = shapes.Difference(shape=disk, tool=aperture)
    moved = split_disk.translated(offset=(2.0, 0.0)).rotated(
        angle=90.0,
        center=(2.0, 0.0),
    )

    assert split_disk.contains(0.75, 0.0)
    assert not split_disk.contains(0.0, 0.0)
    assert moved.contains(2.0, 0.75)
    assert shapes.Annulus(center=(0.0, 0.0), inner_radius=0.5, outer_radius=1.0).contains(0.75, 0.0)
    assert shapes.Ellipsoid(center=(0.0, 0.0, 0.0), radii=(1.0, 2.0, 3.0)).dimension == 3

    with pytest.raises(GeometryError):
        shapes.Polygon(points=((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)))


@pytest.mark.gmsh
@pytest.mark.parametrize(
    "shape",
    (
        shapes.Ellipse(center=(0.5, 0.5), radii=(0.25, 0.15)),
        shapes.RoundedRectangle(bounds=((0.2, 0.8), (0.2, 0.8)), radius=0.1),
        shapes.Difference(
            shape=shapes.Circle(center=(0.5, 0.5), radius=0.3),
            tool=shapes.Circle(center=(0.5, 0.5), radius=0.12),
        ),
    ),
)
def test_shared_extended_shapes_mesh_in_fem(shape) -> None:
    from fem_waveguide_modes import ModeSolver2D

    solver = ModeSolver2D(frequency=299_792_458.0, x_range=1.0, y_range=1.0)
    solver.add_geometry(
        shape=shape,
        material=Material(name="inclusion", epsilon=2.25),
    )
    mesh = solver.mesh(max_element_size=0.25, material_aware=False)
    assert mesh.elements.size > 0


@pytest.mark.parametrize(
    ("package", "solver_name"),
    (
        ("fdfd_waveguide_modes", "ModeSolver1D"),
        ("fdfd_waveguide_modes", "ModeSolver2D"),
        ("fdfd_periodic_modes", "PeriodicModeSolver2D"),
        ("fdfd_periodic_modes", "PeriodicModeSolver3D"),
        ("fdfd_scattering", "ScatteringSolver2D"),
        ("fdfd_band_structure", "BandStructureSolver2D"),
        ("fem_waveguide_modes", "ModeSolver1D"),
        ("fem_waveguide_modes", "ModeSolver2D"),
        ("fem_periodic_modes", "PeriodicModeSolver2D"),
        ("fem_periodic_modes", "PeriodicModeSolver3D"),
        ("fem_waveguide_scattering", "WaveguideScatteringSolver2D"),
        ("fem_electrostatics", "ElectrostaticSolver"),
    ),
)
def test_public_solver_configuration_is_keyword_only(package: str, solver_name: str) -> None:
    module = __import__(package, fromlist=(solver_name,))
    signature = inspect.signature(getattr(module, solver_name))
    assert all(
        parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD
        for parameter in signature.parameters.values()
    )


def test_clean_break_removes_obsolete_solver_workflows() -> None:
    solver_types = []
    for package, names in {
        "fdfd_waveguide_modes": ("ModeSolver1D", "ModeSolver2D"),
        "fdfd_periodic_modes": ("PeriodicModeSolver2D", "PeriodicModeSolver3D"),
        "fdfd_scattering": ("ScatteringSolver2D",),
        "fdfd_band_structure": ("BandStructureSolver2D",),
        "fem_waveguide_modes": ("ModeSolver1D", "ModeSolver2D"),
        "fem_periodic_modes": ("PeriodicModeSolver2D", "PeriodicModeSolver3D"),
        "fem_waveguide_scattering": ("WaveguideScatteringSolver2D",),
        "fem_electrostatics": ("ElectrostaticSolver",),
    }.items():
        module = __import__(package, fromlist=names)
        solver_types.extend(getattr(module, name) for name in names)

    obsolete = ("add_pec", "add_pmc", "add_object", "add_region", "run", "visualize_with_gui")
    for solver_type in solver_types:
        assert not any(hasattr(solver_type, name) for name in obsolete)


def test_fdfd_waveguide_material_geometry_lifecycle_and_round_trip(tmp_path) -> None:
    from fdfd_waveguide_modes import ModeSolver1D, load_result

    dielectric = Material(name="dielectric fill", epsilon=2.25)
    solver = ModeSolver1D(
        frequency=299_792_458.0,
        x_range=(0.0, 1.0),
        background_material=dielectric,
    )
    left = solver.add_geometry(
        shape=shapes.Interval(bounds=(0.0, 0.05)),
        material=materials.PEC,
        name="left wall",
    )
    solver.add_geometry(
        shape=shapes.Interval(bounds=(0.95, 1.0)),
        material=materials.PEC,
        name="right wall",
    )
    mesh = solver.mesh(resolution=20)
    result = solver.solve(num_modes=1, neff_guess=1.4, polarization="TE")

    assert solver.mesh_data is mesh
    assert solver.result is result
    assert result.neff.shape == (1,)
    assert result.mesh_data.axes == ("x",)
    restored = load_result(result.save(tmp_path / "modes.h5"))
    np.testing.assert_allclose(restored.neff, result.neff)

    solver.set_shape(
        geometry=left,
        shape=shapes.Interval(bounds=(0.0, 0.1)),
    )
    assert solver.mesh_data is None
    assert solver.result is None
    solver.solve(num_modes=1, neff_guess=1.4, polarization="TE")
    assert solver.mesh_data.resolution == (20,)


def test_fdfd_periodic_band_and_scattering_public_workflows() -> None:
    from fdfd_band_structure import BandStructureSolver2D
    from fdfd_periodic_modes import PeriodicModeSolver2D
    from fdfd_scattering import ScatteringSolver2D

    dielectric = Material(name="uniform dielectric", epsilon=2.25)
    periodic = PeriodicModeSolver2D(
        frequency=299_792_458.0,
        x_range=1.0,
        z_range=0.25,
        polarization="TM",
        background_material=dielectric,
    )
    periodic.mesh(resolution=(8, 8))
    modes = periodic.solve(num_modes=1, neff_guess=1.5)
    assert 0.0 < modes.neff[0].real <= 1.5
    assert abs(modes.neff[0].imag) < 1e-10

    bands = BandStructureSolver2D(x_range=1.0, y_range=1.0)
    bands.add_geometry(
        shape=shapes.Circle(center=(0.5, 0.5), radius=0.2),
        material=Material(name="rod", epsilon=4.0),
    )
    bands.mesh(resolution=(8, 8))
    path = bands.make_bloch_path(points=((0.0, 0.0), (np.pi, 0.0)), num_points=3)
    band_result = bands.solve(beta_path=path, num_modes=1, polarizations=("TE",))
    assert band_result.frequencies["TE"].shape == (1, 3)

    scattering = ScatteringSolver2D(
        frequency=299_792_458.0,
        x_range=(-1.0, 1.0),
        y_range=(-1.0, 1.0),
        polarization="TE",
    )
    scattering.add_circle(
        center=(0.0, 0.0),
        radius=0.2,
        material=Material(name="cylinder", epsilon=2.0),
    )
    scattering.add_pml(thickness=0.2)
    scattering.mesh(resolution=(20, 20))
    scattering.add_source(angle=0.0)
    scattering.set_source_region(inset=0.3)
    field = scattering.solve()
    assert field.fields["Ez"].shape == (20, 20, 1)
