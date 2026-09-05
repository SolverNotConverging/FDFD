from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import spmatrix
from scipy.sparse.linalg import norm as sparse_norm

from fem_waveguide_scattering.materials import Material
from fem_waveguide_scattering.modes import CrossSection, ModeSolver
from fem_waveguide_scattering.pml import PML


class _IdentityStretchPML(PML):
    """Test-only zero-strength PML retaining mesh-conforming interfaces."""

    def stretch(
        self, depth: ArrayLike, k_reference: float
    ) -> NDArray[np.complex128]:
        del k_reference
        return np.ones(np.asarray(depth).shape, dtype=np.complex128)


def _relative_sparse_difference(left: spmatrix, right: spmatrix) -> float:
    denominator = float(sparse_norm(right))
    difference = float(sparse_norm(left - right))
    return difference if denominator == 0.0 else difference / denominator


def test_zero_strength_pml_is_exactly_the_scalar_pencil() -> None:
    """Diagonal assembly must reduce to the original isotropic QEP."""

    kwargs = {
        "x_span": (-0.5, 0.5),
        "background": Material(eps_r=2.7, mu_r=1.15),
        "boundary": "pec",
    }
    scalar = CrossSection(**kwargs)
    identity = CrossSection(**kwargs, pml=_IdentityStretchPML(0.25))
    scalar.add_layer(
        x=(-0.15, 0.2), material=Material(eps_r=4.1, mu_r=0.93), name="core"
    )
    identity.add_layer(
        x=(-0.15, 0.2), material=Material(eps_r=4.1, mu_r=0.93), name="core"
    )

    solver_kwargs = {
        "wavelength": 1.0,
        "ky": 0.27 * 2.0 * np.pi,
        "num_elements": 40,
    }
    reference = ModeSolver(scalar, **solver_kwargs).assemble()
    reduced = ModeSolver(identity, **solver_kwargs).assemble()

    np.testing.assert_allclose(reduced.x_nodes, reference.x_nodes, atol=2e-15)
    for actual, expected in zip(
        (reduced.A0, reduced.A1, reduced.A2),
        (reference.A0, reference.A1, reference.A2),
        strict=True,
    ):
        assert _relative_sparse_difference(actual, expected) < 2e-15
    assert _relative_sparse_difference(
        reduced.divergence_x, reference.divergence_x
    ) < 2e-15
    assert _relative_sparse_difference(
        reduced.epsilon_mass, reference.epsilon_mass
    ) < 2e-15
    assert _relative_sparse_difference(
        reduced.epsilon_mass_z, reference.epsilon_mass_z
    ) < 2e-15


@pytest.mark.slow
def test_bound_slab_mode_is_stable_under_pml_mesh_refinement() -> None:
    wavelength = 1.55e-6
    k0 = 2.0 * np.pi / wavelength
    neff: list[complex] = []

    for elements in (36, 72, 144):
        cross_section = CrossSection(
            x_span=(-3.0e-6, 3.0e-6),
            background=Material(eps_r=1.44**2),
            boundary="pec",
            pml=PML(0.75e-6, order=3, target_reflection=1e-7),
        )
        cross_section.add_layer(
            x=(-0.25e-6, 0.25e-6),
            material=Material(eps_r=3.45**2),
            name="core",
        )
        mode = ModeSolver(
            cross_section,
            wavelength=wavelength,
            ky=0.15 * k0,
            num_elements=elements,
            dense_linearization_limit=500,
        ).solve(max_refinements=0, num_modes=1, neff_guess=3.2)[0]
        neff.append(mode.neff)
        assert mode.direction == "forward"
        assert mode.power == pytest.approx(1.0, rel=2e-10)
        assert abs(mode.neff.imag) < 2e-10
        assert mode.divergence_residual < 1e-9

    coarse_change = abs(neff[1] - neff[0])
    fine_change = abs(neff[2] - neff[1])
    assert fine_change < coarse_change / 3.5


def test_outgoing_transverse_pml_has_passive_plus_iwt_stretch_sign() -> None:
    """Negative Im(sx) must produce negative Im(beta) for exp(+i*omega*t)."""

    wavelength = 1.0
    width = 1.0
    eps_r = 2.25
    pml = PML(0.25, order=2, target_reflection=0.1)
    cross_section = CrossSection(
        x_span=(-width / 2.0, width / 2.0),
        background=Material(eps_r=eps_r),
        boundary="pec",
        pml=pml,
    )
    modes = ModeSolver(
        cross_section,
        wavelength=wavelength,
        num_elements=80,
        dense_linearization_limit=420,
    ).solve(max_refinements=0, num_modes=2, neff_guess=1.44)

    k0 = 2.0 * np.pi / wavelength
    integrated_imaginary_stretch = (
        pml.maximum_imaginary_stretch(k0) * pml.thickness / (pml.order + 1)
    )
    stretched_width = width - 2j * integrated_imaginary_stretch
    expected = np.sqrt(eps_r - (np.pi / (k0 * stretched_width)) ** 2)

    assert expected.imag < 0.0
    for mode in modes:
        assert mode.neff.imag < 0.0
        assert mode.neff == pytest.approx(expected, rel=6e-5, abs=8e-5)
        assert mode.direction == "forward"
        assert mode.divergence_residual < 1e-9
