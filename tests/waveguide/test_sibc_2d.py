"""2D surface-impedance boundaries: integral Ampere rows, corners, and wall loss."""

import numpy as np
import pytest

import fdfd
from fdfd import materials, shapes
from fdfd.core.constants import C0, EPSILON_0, ETA_0
from fdfd.waveguide._impedance_1d import SurfaceImpedanceDefinition
from fdfd.waveguide._impedance_2d import compile_impedance_boundary
from fdfd.waveguide._kernel_2d import WaveguideKernel2D, _flat_index

FREQUENCY = 10e9


def definition(impedance, label="test"):
    impedance = complex(impedance)
    return SurfaceImpedanceDefinition(
        key=("constant", impedance.real, impedance.imag), impedance=impedance, label=label
    )


def compile_owner(owner, *, impedance=40.0 + 5.0j, dx=0.7e-3, dy=0.9e-3, definitions=None):
    ones = np.ones(owner.shape, dtype=complex)
    return compile_impedance_boundary(
        owner=owner,
        definitions=definitions or (definition(impedance),),
        cell_eps_r_xx=ones,
        cell_eps_r_yy=ones,
        cell_eps_r_zz=ones,
        dx=dx,
        dy=dy,
        frequency=FREQUENCY,
        epsilon0=EPSILON_0,
    )


def kernel_with_wall(region, *, nx=4, ny=4, dx=1e-3, dy=1e-3, impedance=40.0 + 5.0j):
    kernel = WaveguideKernel2D(FREQUENCY, nx * dx, ny * dy, nx, ny)
    mask = np.zeros((nx, ny), dtype=bool)
    mask[region] = True
    kernel.add_impedance_surface(mask, impedance)
    return kernel


def load(row, area):
    return 1j * 2 * np.pi * FREQUENCY * EPSILON_0 * area * (row.relative_permittivity - 1.0)


def test_flat_wall_has_exact_half_cell_ampere_row():
    dy, impedance = 0.9e-3, 40.0 + 5.0j
    owner = np.full((4, 4), -1, dtype=np.int32)
    owner[:, 0] = 0
    boundary = compile_owner(owner, impedance=impedance, dy=dy)
    row = next(r for r in boundary.rows if r.electric_component == 0 and r.electric_index == (2, 1))
    assert row.retained_dual_area == pytest.approx(dy / 2)
    assert row.relative_permittivity == pytest.approx(
        1 + 2 / (1j * 2 * np.pi * FREQUENCY * EPSILON_0 * dy * impedance)
    )
    assert [(t.component, t.index, t.length) for t in row.magnetic_terms] == [(2, (2, 1), 1.0)]
    assert not boundary.electric_retained[0][2, 0] and boundary.electric_retained[0][2, 1]
    assert not boundary.magnetic_retained[2][2, 0] and boundary.magnetic_retained[2][2, 1]


@pytest.mark.parametrize("pattern", range(16))
def test_every_longitudinal_quadrant_pattern(pattern):
    dx, dy, impedance = 0.8e-3, 1.2e-3, 35.0 + 9.0j
    quadrants = ((0, 0), (1, 0), (1, 1), (0, 1))
    owner = np.full((3, 3), -1, dtype=np.int32)
    for bit, cell in enumerate(quadrants):
        if pattern & (1 << bit):
            owner[cell] = 0
    if pattern in (0b0101, 0b1010):
        with pytest.raises(ValueError, match="non-manifold"):
            compile_owner(owner, impedance=impedance, dx=dx, dy=dy)
        return
    boundary = compile_owner(owner, impedance=impedance, dx=dx, dy=dy)
    rows = [r for r in boundary.rows if r.electric_component == 2 and r.electric_index == (1, 1)]
    metal = pattern.bit_count()
    if metal in (0, 4):
        assert rows == [] and boundary.electric_retained[2][1, 1] == (metal == 0)
        return
    (row,) = rows
    area = (4 - metal) * dx * dy / 4
    assert row.retained_dual_area == pytest.approx(area)
    states = [bool(pattern & (1 << bit)) for bit in range(4)]
    exposed = sum(
        length
        for a, b, length in ((0, 1, dy / 2), (3, 2, dy / 2), (0, 3, dx / 2), (1, 2, dx / 2))
        if states[a] != states[b]
    )
    assert load(row, area) == pytest.approx(exposed / impedance)


def test_convex_corner_has_exact_ez_circulation():
    dx, dy = 0.8e-3, 1.2e-3
    owner = np.full((4, 4), -1, dtype=np.int32)
    owner[1, 1] = 0
    row = next(
        r
        for r in compile_owner(owner, dx=dx, dy=dy).rows
        if r.electric_component == 2 and r.electric_index == (2, 2)
    )
    assert {(t.component, t.index): t.length for t in row.magnetic_terms} == {
        (0, (2, 1)): dx / 2,
        (0, (2, 2)): -dx,
        (1, (1, 2)): -dy / 2,
        (1, (2, 2)): dy,
    }

    kernel = kernel_with_wall((slice(1, 2), slice(1, 2)), dx=dx, dy=dy)
    kernel.update_component_materials()
    d = kernel.derivatives()
    s = kernel.field_shapes
    scale = row.retained_dual_area * kernel.k0
    ez = _flat_index(2, 2, s["Ez"][0])
    hx = dict(
        zip(*[x.tolist() for x in (d["hx_ez"].getrow(ez).indices, d["hx_ez"].getrow(ez).data)], strict=True)
    )
    hy = dict(
        zip(*[x.tolist() for x in (d["hy_ez"].getrow(ez).indices, d["hy_ez"].getrow(ez).data)], strict=True)
    )
    assert hx == pytest.approx(
        {_flat_index(2, 1, s["Hx"][0]): -dx / 2 / scale, _flat_index(2, 2, s["Hx"][0]): dx / scale}
    )
    assert hy == pytest.approx(
        {_flat_index(1, 2, s["Hy"][0]): -dy / 2 / scale, _flat_index(2, 2, s["Hy"][0]): dy / scale}
    )


def test_corner_sums_ports_of_different_impedances():
    dx, dy, first, second = 0.8e-3, 1.2e-3, 30.0 + 4.0j, 70.0 + 11.0j
    owner = np.full((3, 3), -1, dtype=np.int32)
    owner[0, 0], owner[0, 1] = 0, 1
    boundary = compile_owner(
        owner, dx=dx, dy=dy, definitions=(definition(first), definition(second, "second"))
    )
    row = next(r for r in boundary.rows if r.electric_component == 2 and r.electric_index == (1, 1))
    assert load(row, row.retained_dual_area) == pytest.approx(dy / (2 * first) + dy / (2 * second))


def test_interface_normal_permeability_comes_only_from_retained_cell():
    horizontal = WaveguideKernel2D(FREQUENCY, 4e-3, 4e-3, 4, 4)
    horizontal.cell_mu_r_yy[:, 0], horizontal.cell_mu_r_yy[:, 1] = 99.0, 4.0
    mask = np.zeros((4, 4), dtype=bool)
    mask[:, 0] = True
    horizontal.add_impedance_surface(mask, 40.0 + 5.0j)
    assert horizontal._effective_materials_and_masks()[0]["mu_yy"][2, 1] == 4.0
    vertical = WaveguideKernel2D(FREQUENCY, 4e-3, 4e-3, 4, 4)
    vertical.cell_mu_r_xx[0, :], vertical.cell_mu_r_xx[1, :] = 88.0, 3.0
    vertical.add_impedance_surface(mask.T, 40.0 + 5.0j)
    assert vertical._effective_materials_and_masks()[0]["mu_xx"][1, 2] == 3.0


def test_transverse_row_replaces_the_sparse_ampere_curl():
    kernel = kernel_with_wall((slice(0, 4), slice(0, 1)))
    kernel.update_component_materials()
    d = kernel.derivatives()
    row = d["hz_hy"].getrow(_flat_index(2, 1, kernel.field_shapes["Ex"][0]))
    column = _flat_index(2, 1, kernel.field_shapes["Hz"][0])
    assert row.indices.tolist() == [column]
    assert row.data[0] == pytest.approx(2.0 / (kernel.k0 * kernel.dy))


def rectangular_guide(wall, *, aperture=(50, 25), wall_cells=1):
    width, height = 22.86e-3, 10.16e-3
    dx, dy = width / aperture[0], height / aperture[1]
    w, h = wall_cells * dx, wall_cells * dy
    solver = fdfd.ModeSolver2D(frequency=FREQUENCY, x_range=(-w, width + w), y_range=(-h, height + h))
    frame = shapes.Difference(
        shape=shapes.Rectangle(bounds=((-w, width + w), (-h, height + h))),
        tool=shapes.Rectangle(bounds=((0.0, width), (0.0, height))),
    )
    solver.add_geometry(shape=frame, material=wall)
    solver.mesh(resolution=(aperture[0] + 2 * wall_cells, aperture[1] + 2 * wall_cells), subpixels=1)
    expected = np.sqrt(1 - (C0 / (2 * width * FREQUENCY)) ** 2)
    return solver, solver.solve(num_modes=1, guess=expected), expected


def test_copper_te10_wall_loss_matches_first_order_theory():
    width, height = 22.86e-3, 10.16e-3
    _, copper, expected_neff = rectangular_guide(materials.copper)
    _, pec, _ = rectangular_guide(materials.PEC)
    k0 = 2 * np.pi * FREQUENCY / C0
    beta = k0 * expected_neff
    impedance = materials.copper.at_frequency(frequency=FREQUENCY)
    coefficient = (k0 / (beta * height) + 2 * (np.pi / width) ** 2 / (k0 * beta * width)) / ETA_0
    assert copper.neff[0].real == pytest.approx(expected_neff, abs=5e-4)
    assert copper.attenuation_constant[0] == pytest.approx(coefficient * impedance.real, rel=0.01)
    assert abs(pec.attenuation_constant[0]) < 1e-9
    shift = copper.neff[0].real - pec.neff[0].real
    assert shift == pytest.approx(coefficient * impedance.imag / k0, rel=0.01)


def test_opaque_wall_thickness_does_not_change_the_boundary_problem():
    results = []
    for wall_cells in (1, 2):
        solver, result, _ = rectangular_guide(materials.copper, aperture=(24, 12), wall_cells=wall_cells)
        boundary = solver._kernel._compiled_impedance_boundary
        results.append(
            (
                result.neff,
                int(np.count_nonzero(solver._kernel.impedance_surface_mask)),
                len(boundary.rows),
                tuple(np.count_nonzero(m) for m in boundary.electric_retained),
                tuple(np.count_nonzero(m) for m in boundary.magnetic_retained),
            )
        )
    thin, thick = results
    assert thick[1] > thin[1]
    assert thick[2:] == thin[2:]
    np.testing.assert_allclose(thick[0], thin[0], rtol=1e-9, atol=1e-11)


def test_purely_reactive_closed_wall_has_no_modal_attenuation():
    _, result, _ = rectangular_guide(fdfd.SurfaceImpedance(impedance=0.05j), aperture=(24, 12))
    assert abs(result.neff[0].imag) < 1e-10


@pytest.mark.parametrize("kind", ["pec", "pmc"])
def test_conductor_touching_an_impedance_surface_is_rejected(kind):
    kernel = kernel_with_wall((slice(0, 4), slice(0, 1)))
    touching = np.zeros((4, 4), dtype=bool)
    touching[:, 1] = True
    getattr(kernel, f"add_{kind}")(touching)
    with pytest.raises(ValueError, match="constraint"):
        kernel.update_component_materials()


def test_impedance_surface_touching_the_pml_is_rejected():
    solver = fdfd.ModeSolver2D(frequency=FREQUENCY, x_range=(0.0, 4e-3), y_range=(0.0, 4e-3))
    solver.add_rectangle(x_range=(1e-3, 2e-3), y_range=(0.0, 4e-3), material=materials.copper)
    solver.add_pml(thickness=1e-3, sides="x-")
    with pytest.raises(ValueError, match="PML"):
        solver.mesh(resolution=(4, 4), subpixels=1)
