import numpy as np
import pytest

from wavefem.exceptions import ModeProjectionError
from wavefem.projection import (
    ElectromagneticProjector,
    ModalTrace,
    modal_power_from_gram,
)


def _trace(seed: int, common: np.ndarray | None = None) -> ModalTrace:
    rng = np.random.default_rng(seed)
    E = rng.normal(size=(3, 12)) + 1j * rng.normal(size=(3, 12))
    H = 1e-3 * (rng.normal(size=(3, 12)) + 1j * rng.normal(size=(3, 12)))
    if common is not None:
        E = 0.7 * common + 0.3 * E
    return ModalTrace(E, H, label=f"mode-{seed}")


def test_synthetic_nonorthogonal_superposition_is_recovered() -> None:
    first = _trace(1)
    second = _trace(2, common=first.E)
    third = _trace(3, common=first.E)
    traces = (first, second, third)
    expected = np.array([0.4 - 0.2j, -1.1 + 0.3j, 0.2 + 0.7j])
    E = sum(a * trace.E for a, trace in zip(expected, traces, strict=True))
    H = sum(a * trace.H for a, trace in zip(expected, traces, strict=True))
    result = ElectromagneticProjector(traces, np.linspace(0.5, 1.5, 12)).project(E, H)
    np.testing.assert_allclose(result.amplitudes, expected, rtol=2e-13, atol=2e-13)
    assert result.relative_residual < 1e-13
    assert result.labels == ("mode-1", "mode-2", "mode-3")


def test_duplicate_modal_trace_is_rejected_as_singular() -> None:
    trace = _trace(4)
    projector = ElectromagneticProjector(
        (trace, ModalTrace(trace.E.copy(), trace.H.copy(), "duplicate")),
        np.ones(12),
    )
    with pytest.raises(ModeProjectionError, match="near singular"):
        projector.project(trace.E, trace.H)


def test_projector_validates_monitor_quadrature() -> None:
    trace = _trace(5)
    with pytest.raises(ModeProjectionError, match="weights"):
        ElectromagneticProjector((trace,), np.ones(11))


def test_power_normalized_forward_backward_plane_waves_are_separated() -> None:
    npoints = 10
    weights = np.full(npoints, 0.2)
    width = float(np.sum(weights))
    impedance = 3.0
    electric_amplitude = np.sqrt(2.0 * impedance / width)
    magnetic_amplitude = electric_amplitude / impedance
    E = np.zeros((3, npoints), dtype=complex)
    E[0] = electric_amplitude
    H_forward = np.zeros_like(E)
    H_forward[1] = magnetic_amplitude
    H_backward = np.zeros_like(E)
    H_backward[1] = -magnetic_amplitude
    forward = ModalTrace(E, H_forward, "forward")
    backward = ModalTrace(E, H_backward, "backward")
    expected = np.array([0.7 + 0.2j, -0.3 + 0.4j])
    target_E = expected[0] * E + expected[1] * E
    target_H = expected[0] * H_forward + expected[1] * H_backward

    result = ElectromagneticProjector(
        (forward, backward), weights, impedance=impedance
    ).project(target_E, target_H)

    np.testing.assert_allclose(np.diag(result.gram_matrix), [1.0, -1.0], atol=1e-14)
    np.testing.assert_allclose(result.gram_matrix[0, 1], 0.0, atol=1e-14)
    np.testing.assert_allclose(result.amplitudes, expected, atol=1e-14)


def test_modal_power_uses_nonorthogonal_gram_cross_terms() -> None:
    amplitudes = np.asarray((1.0 + 0.2j, -0.4 + 0.7j))
    gram = np.asarray(
        ((1.0, 0.08 + 0.03j), (0.08 - 0.03j, 0.97)),
        dtype=np.complex128,
    )
    expected = float(np.real(amplitudes @ gram @ np.conj(amplitudes)))

    actual = modal_power_from_gram(amplitudes, gram)

    assert actual == pytest.approx(expected)
    assert actual != pytest.approx(float(np.sum(np.abs(amplitudes) ** 2)))
    assert modal_power_from_gram(amplitudes, gram, indices=np.asarray([0])) == pytest.approx(
        abs(amplitudes[0]) ** 2
    )
    unnormalized = np.diag((0.91, 1.07)).astype(np.complex128)
    assert modal_power_from_gram(
        amplitudes,
        unnormalized,
        normalize_diagonal=True,
    ) == pytest.approx(float(np.sum(np.abs(amplitudes) ** 2)))
