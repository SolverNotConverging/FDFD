"""PEC/PMC volumes constrain exactly the Yee samples inside them.

PEC fixes every sample in the closed volume (tangential E and normal H on its
faces); PMC fixes only samples strictly inside it, leaving its faces to the
dual conditions. Along the periodic z axis a volume spanning the whole
period has no face.
"""

import numpy as np
import pytest

import fdfd
from fdfd import Material, materials
from fdfd.core import eigs
from fdfd.periodic._kernel_2d import PeriodicKernel2D
from fdfd.periodic._kernel_3d import PeriodicKernel3D

# Sample offsets in cell units along (x, y, z): E_c on cell centres along c.
OFFSETS_3D = {
    "Ex": (0.5, 0, 0),
    "Ey": (0, 0.5, 0),
    "Ez": (0, 0, 0.5),
    "Hx": (0, 0.5, 0.5),
    "Hy": (0.5, 0, 0.5),
    "Hz": (0.5, 0.5, 0),
}
# The 2D cell is (x, z) and invariant along y.
OFFSETS_2D = {
    name: (offset[0], offset[2]) if name not in ("Ey", "Hy") else None for name, offset in OFFSETS_3D.items()
}
OFFSETS_2D.update({"Ey": (0, 0), "Hy": (0.5, 0.5)})


def kernel(dimension):
    if dimension == 2:
        return PeriodicKernel2D(10e9, 1.0, 1.0, 4, 3, boundary="pmc")
    return PeriodicKernel3D(10e9, 1.0, 1.0, 1.0, 3, 3, 2)


def box(k, bounds):
    mask = np.zeros(k.cells, dtype=bool)
    mask[tuple(slice(lo, hi) for lo, hi in bounds)] = True
    return mask


def expected(k, name, bounds, *, strict):
    offsets = (OFFSETS_2D if len(k.cells) == 2 else OFFSETS_3D)[name]
    shape = k.field_shapes[name]
    result = np.ones(shape, dtype=bool)
    for axis, ((lo, hi), shift) in enumerate(zip(bounds, offsets, strict=True)):
        positions = np.arange(shape[axis]) + shift
        candidates = [positions]
        if axis == len(k.cells) - 1:
            if lo == 0 and hi == k.cells[-1]:
                candidates = [positions - np.inf]  # whole period: no face at the seam
            else:
                candidates.append(positions + k.cells[-1])  # node N is node 0
        inside = np.zeros(len(positions), dtype=bool)
        for p in candidates:
            if np.isinf(p).all():
                inside[:] = True
            else:
                inside |= (p > lo) & (p < hi) if strict else (p >= lo) & (p <= hi)
        view = [1] * len(shape)
        view[axis] = shape[axis]
        result &= inside.reshape(view)
    return result


VOLUMES_2D = [((1, 2), (0, 3)), ((0, 4), (0, 1)), ((1, 2), (0, 1)), ((1, 3), (1, 3))]
VOLUMES_3D = [
    ((1, 2), (0, 3), (0, 2)),
    ((0, 3), (1, 2), (0, 2)),
    ((0, 3), (0, 3), (0, 1)),
    ((1, 2), (1, 2), (0, 2)),
    ((1, 2), (1, 2), (0, 1)),
]
CASES = [(2, volume) for volume in VOLUMES_2D] + [(3, volume) for volume in VOLUMES_3D]


@pytest.mark.parametrize("kind", ["pec", "pmc"])
@pytest.mark.parametrize(("dimension", "bounds"), CASES)
def test_volume_constrains_the_samples_it_contains(kind, dimension, bounds):
    k = kernel(dimension)
    getattr(k, f"add_{kind}")(box(k, bounds))
    _, masks = k.materials_and_masks()
    for name, mask in masks.items():
        np.testing.assert_array_equal(mask, expected(k, name, bounds, strict=kind == "pmc"), err_msg=name)


@pytest.mark.parametrize("dimension", [2, 3])
def test_exact_boundaries_do_not_modify_material_tensors(dimension):
    k = kernel(dimension)
    full_z = k.cells[-1]
    k.add_pec(box(k, ((1, 2), *((0, n) for n in k.cells[1:-1]), (0, full_z))))
    k.add_pmc(box(k, ((2, 3), *((0, n) for n in k.cells[1:-1]), (0, full_z))))
    for tensor in k.cell_tensors():
        for values in tensor:
            np.testing.assert_array_equal(values, 1.0)


def test_pec_outer_boundary_fixes_tangential_e_and_normal_h_at_x_ends():
    k = PeriodicKernel2D(10e9, 1.0, 1.0, 4, 3, boundary="pec")
    _, masks = k.materials_and_masks()
    for name in ("Ey", "Ez", "Hx"):
        np.testing.assert_array_equal(masks[name][[0, -1]], True)
        np.testing.assert_array_equal(masks[name][1:-1], False)
    for name in ("Ex", "Hy", "Hz"):
        assert not masks[name].any()


@pytest.mark.parametrize("dimension", [2, 3])
def test_solve_restores_exact_zeros_on_constrained_samples(dimension):
    if dimension == 2:
        solver = fdfd.PeriodicModeSolver2D(
            frequency=10e9, x_range=(0.0, 1.0), z_range=(0.0, 1.0), boundary=materials.PMC
        )
        solver.add_rectangle(x_range=(0.25, 0.5), z_range=(0.0, 1.0), material=materials.PEC)
        solver.mesh(resolution=(8, 6), subpixels=1)
    else:
        solver = fdfd.PeriodicModeSolver3D(
            frequency=10e9, x_range=(0.0, 1.0), y_range=(0.0, 1.0), z_range=(0.0, 1.0)
        )
        solver.add_box(x_range=(0.25, 0.5), y_range=(0.25, 0.5), z_range=(0.0, 1.0), material=materials.PMC)
        solver.mesh(resolution=(5, 5, 4), subpixels=1)
    result = solver.solve(num_modes=2, guess=0.5)
    _, masks = solver._kernel.materials_and_masks()
    for name, mask in masks.items():
        np.testing.assert_array_equal(result.fields[name][mask], 0.0)


def test_spatial_eigenvalue_maps_to_exp_plus_iwt_effective_index(monkeypatch):
    solver = fdfd.PeriodicModeSolver2D(frequency=10e9, x_range=(0.0, 1.0), z_range=(0.0, 1.0))
    solver.mesh(resolution=(4, 3))
    gamma = solver._kernel.k0 * (0.02 + 1.5j)

    def fake_solve(A, B=None, **kwargs):
        vector = np.ones((A.shape[0], 1), dtype=complex)
        return eigs.Eigenpairs(np.array([gamma]), vector, np.zeros(1))

    monkeypatch.setattr(eigs, "solve", fake_solve)
    result = solver.solve(num_modes=1, polarization="TM", guess=1.5)
    assert result.neff[0] == pytest.approx(1.5 - 0.02j)
    assert result.beta[0] == pytest.approx(result.k0 * (1.5 - 0.02j))
    assert result.attenuation_constant[0] == pytest.approx(0.02 * result.k0)


@pytest.mark.parametrize("dimension", [2, 3])
def test_uniform_passive_medium_has_negative_imaginary_neff(dimension):
    epsilon = 2.0 - 0.1j
    medium = Material(name="lossy", epsilon=epsilon)
    if dimension == 2:
        solver = fdfd.PeriodicModeSolver2D(
            frequency=10e9,
            x_range=(0.0, 1.0),
            z_range=(0.0, 1.0),
            background_material=medium,
            boundary=materials.PMC,
        )
        solver.mesh(resolution=(4, 3))
        result = solver.solve(num_modes=1, polarization="TE", guess=1.03 * np.sqrt(epsilon))
    else:
        solver = fdfd.PeriodicModeSolver3D(
            frequency=10e9,
            x_range=(0.0, 1.0),
            y_range=(0.0, 1.0),
            z_range=(0.0, 1.0),
            background_material=medium,
        )
        solver.mesh(resolution=(3, 3, 2))
        result = solver.solve(num_modes=1, guess=1.03 * np.sqrt(epsilon))
    # A guess just above the top mode (and off its degenerate pole) selects it.
    # In 2D it is the uniform TE wave; the 3D cell's PMC-like transverse
    # closure excludes the uniform TEM wave, so its top mode lies just below.
    if dimension == 2:
        assert result.neff[0] == pytest.approx(np.sqrt(epsilon), abs=1e-10)
    else:
        assert result.neff[0] == pytest.approx(np.sqrt(epsilon), abs=1e-4)
    assert result.neff[0].real > 0 and result.neff[0].imag < 0


def test_fully_constrained_domain_is_rejected_before_the_eigensolve():
    solver = fdfd.PeriodicModeSolver2D(frequency=10e9, x_range=(0.0, 1.0), z_range=(0.0, 1.0))
    solver.add_rectangle(x_range=(0.0, 1.0), z_range=(0.0, 1.0), material=materials.PEC)
    solver.mesh(resolution=(4, 3))
    with pytest.raises(ValueError, match="unconstrained"):
        solver.solve(num_modes=1, guess=1.0)
