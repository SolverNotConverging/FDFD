import numpy as np
import pytest

from cem_common.errors import ConfigurationError
from fem_waveguide_scattering.pml import PML, PMLLayout


def test_stretch_is_identity_at_interface_and_absorbing_outward() -> None:
    pml = PML(thickness=1.0, order=3, target_reflection=1e-8)
    s = pml.stretch(np.array([0.0, 0.5, 1.0]), k_reference=2.0)
    assert s[0] == 1.0 + 0.0j
    assert np.all(np.diff(s.imag) < 0.0)


def test_polynomial_profile_integrates_to_target_at_reference_wavenumber() -> None:
    pml = PML(thickness=1.3, order=4, target_reflection=2e-7)
    kref = 3.1
    integral = pml.maximum_imaginary_stretch(kref) * pml.thickness / (pml.order + 1)
    assert np.exp(-2.0 * kref * integral) == pytest.approx(pml.target_reflection)


def test_transformation_optics_diagonal_tensor() -> None:
    layout = PMLLayout()
    eps, mu = layout.transform_isotropic(4.0, 2.0, 1.0 + 2.0j, 1.0 + 3.0j)
    factors = np.array(
        [(1.0 + 3.0j) / (1.0 + 2.0j), (1.0 + 2.0j) * (1.0 + 3.0j), (1.0 + 2.0j) / (1.0 + 3.0j)]
    )
    np.testing.assert_allclose(eps, 4.0 * factors)
    np.testing.assert_allclose(mu, 2.0 * factors)


def test_layout_rejects_pmls_that_consume_domain() -> None:
    with pytest.raises(ConfigurationError, match="no non-PML interior"):
        PMLLayout(x=PML(0.6)).validate_domain((-0.5, 0.5), (-1.0, 1.0))
