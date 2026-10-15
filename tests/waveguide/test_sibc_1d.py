"""1D surface-impedance boundaries: exact Yee rows and perturbation theory."""

import numpy as np
import pytest

import fdfd
from fdfd import SurfaceImpedance, materials
from fdfd.core.constants import C0, EPSILON_0, ETA_0, MU_0
from fdfd.waveguide._kernel_1d import WaveguideKernel1D

FREQUENCY = 30e9
APERTURE = 10e-3
OMEGA = 2 * np.pi * FREQUENCY
K0 = OMEGA / C0
COPPER_RS = np.sqrt(np.pi * FREQUENCY * MU_0 / materials.copper.conductivity)
KC = np.pi / APERTURE
BETA_TE1 = np.sqrt(K0**2 - KC**2)
TEM_COEFFICIENT = 1 / (ETA_0 * APERTURE)
TE1_COEFFICIENT = 2 * KC**2 / (OMEGA * MU_0 * BETA_TE1 * APERTURE)


def cells(n, *indices):
    mask = np.zeros(n, dtype=bool)
    mask[list(indices)] = True
    return mask


def parallel_plate(wall, *, aperture_cells=100, wall_cells=1):
    """Two impedance walls bounding a vacuum gap, through the public API."""
    dx = APERTURE / aperture_cells
    solver = fdfd.ModeSolver1D(frequency=FREQUENCY, x_range=(0.0, (aperture_cells + 2 * wall_cells) * dx))
    solver.add_layer(x_range=(0.0, wall_cells * dx), material=wall, name="left")
    solver.add_layer(
        x_range=((aperture_cells + wall_cells) * dx, (aperture_cells + 2 * wall_cells) * dx),
        material=wall,
        name="right",
    )
    solver.mesh(resolution=aperture_cells + 2 * wall_cells, subpixels=1)
    return {pol: solver.solve(num_modes=1, polarization=pol, guess=1.0).neff[0] for pol in ("TE", "TM")}


def test_installed_half_cell_rows_and_material_loads():
    nx, dx, impedance = 6, 0.5e-3, 25.0 + 7.0j
    kernel = WaveguideKernel1D(FREQUENCY, nx * dx, nx)
    kernel.cell_eps_r_yy[:] = [91, 2.5, 3, 4, 4.5, 92]
    kernel.cell_eps_r_zz[:] = [81, 3.5, 4, 5, 5.5, 82]
    kernel.cell_mu_r_xx[:] = [71, 6, 1, 1, 7, 72]
    kernel.add_impedance_surface(cells(nx, 0, nx - 1), impedance)
    materials_, _ = kernel._effective_materials_and_masks()
    _, h_to_e = kernel.derivatives()

    load = 2 / (1j * OMEGA * EPSILON_0 * dx * impedance)
    assert materials_["eps_yy"][1] == pytest.approx(2.5 + load)
    assert materials_["eps_zz"][1] == pytest.approx(3.5 + load)
    assert materials_["eps_yy"][nx - 1] == pytest.approx(4.5 + load)
    assert materials_["eps_zz"][nx - 1] == pytest.approx(5.5 + load)
    # Normal permeability comes only from the retained cell, never the opaque one.
    assert materials_["mu_xx"][1] == 6.0
    assert materials_["mu_xx"][nx - 1] == 7.0
    left, right = h_to_e.getrow(1), h_to_e.getrow(nx - 1)
    assert left.indices.tolist() == [1] and right.indices.tolist() == [nx - 2]
    assert left.data[0] == pytest.approx(2 / (K0 * dx))
    assert right.data[0] == pytest.approx(-2 / (K0 * dx))


def test_copper_plates_match_first_order_attenuation_and_phase():
    copper = parallel_plate(materials.copper)
    resistive = parallel_plate(SurfaceImpedance(impedance=COPPER_RS))
    assert copper["TM"].imag < 0 and copper["TE"].imag < 0
    for pol, coefficient in (("TM", TEM_COEFFICIENT), ("TE", TE1_COEFFICIENT)):
        assert -K0 * copper[pol].imag == pytest.approx(coefficient * COPPER_RS, rel=0.01)
        assert K0 * (copper[pol].real - resistive[pol].real) == pytest.approx(
            coefficient * COPPER_RS, rel=0.01
        )


def test_reactive_walls_are_lossless_and_shift_phase():
    inductive = parallel_plate(SurfaceImpedance(impedance=1j * COPPER_RS))
    capacitive = parallel_plate(SurfaceImpedance(impedance=-1j * COPPER_RS))
    for result in (inductive, capacitive):
        assert abs(result["TM"].imag) < 1e-10 and abs(result["TE"].imag) < 1e-10
    for pol, coefficient in (("TM", TEM_COEFFICIENT), ("TE", TE1_COEFFICIENT)):
        shift = K0 * (inductive[pol].real - capacitive[pol].real) / 2
        assert shift == pytest.approx(coefficient * COPPER_RS, rel=0.01)


def test_opaque_wall_thickness_does_not_change_the_modes():
    thin = parallel_plate(materials.copper, aperture_cells=80, wall_cells=1)
    thick = parallel_plate(materials.copper, aperture_cells=80, wall_cells=4)
    for pol in ("TE", "TM"):
        assert thick[pol] == pytest.approx(thin[pol], rel=1e-9, abs=1e-11)


def test_regions_mark_opaque_cells_without_changing_bulk_permittivity():
    kernel = WaveguideKernel1D(10e9, 0.08, 8)
    before = [values.copy() for values in kernel.cell_tensors()[0]]
    kernel.add_impedance_surface(cells(8, 2, 3, 4), 40.0 + 5.0j)
    np.testing.assert_array_equal(kernel.impedance_surface_mask, cells(8, 2, 3, 4))
    np.testing.assert_array_equal(kernel._surface_impedance_owner[~cells(8, 2, 3, 4)], -1)
    for after, original in zip(kernel.cell_tensors()[0], before, strict=True):
        np.testing.assert_array_equal(after, original)


def test_equal_impedances_share_a_definition_and_different_ones_may_not_overlap():
    kernel = WaveguideKernel1D(10e9, 0.08, 8)
    kernel.add_impedance_surface(cells(8, 0, 1, 2), 40.0 + 5.0j)
    kernel.add_impedance_surface(cells(8, 2, 3, 4), 40.0 + 5.0j)
    assert len(kernel._surface_impedance_definitions) == 1
    with pytest.raises(ValueError, match="overlap"):
        kernel.add_impedance_surface(cells(8, 4, 5), 41.0 + 5.0j)
    kernel.add_impedance_surface(cells(8, 5, 6), 60.0 + 8.0j)
    assert len(kernel._surface_impedance_definitions) == 2


def test_all_opaque_geometry_is_rejected():
    kernel = WaveguideKernel1D(10e9, 0.08, 8)
    with pytest.raises(ValueError, match="no retained"):
        kernel.add_impedance_surface(np.ones(8, dtype=bool), 40.0 + 5.0j)


@pytest.mark.parametrize("kind", ["pec", "pmc"])
def test_conductors_may_not_overlap_or_touch_an_impedance_surface(kind):
    overlap = WaveguideKernel1D(10e9, 0.08, 8)
    getattr(overlap, f"add_{kind}")(cells(8, 0))
    with pytest.raises(ValueError, match=kind.upper()):
        overlap.add_impedance_surface(cells(8, 0), 40.0 + 5.0j)
    adjacent = WaveguideKernel1D(10e9, 0.08, 8)
    adjacent.add_impedance_surface(cells(8, 0), 40.0 + 5.0j)
    getattr(adjacent, f"add_{kind}")(cells(8, 1))
    with pytest.raises(ValueError, match="constraint"):
        adjacent.update_component_materials()


def test_impedance_surface_may_not_touch_the_pml():
    solver = fdfd.ModeSolver1D(frequency=10e9, x_range=(0.0, 0.08))
    solver.add_layer(x_range=(0.02, 0.03), material=materials.copper)
    solver.add_pml(thickness=0.02, sides="x-")
    with pytest.raises(ValueError, match="PML"):
        solver.mesh(resolution=8, subpixels=1)


def test_terminal_run_has_one_interface_and_internal_run_two():
    terminal = WaveguideKernel1D(10e9, 0.08, 8)
    terminal.add_impedance_surface(cells(8, 0, 1), 40.0 + 5.0j)
    terminal.update_component_materials()
    assert [row.electric_index for row in terminal._compiled_impedance_boundary.rows] == [2]
    internal = WaveguideKernel1D(10e9, 0.08, 8)
    internal.add_impedance_surface(cells(8, 3, 4), 40.0 + 5.0j)
    internal.update_component_materials()
    assert [row.electric_index for row in internal._compiled_impedance_boundary.rows] == [3, 5]
