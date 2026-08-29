from dataclasses import replace

import numpy as np
import pytest

from wavefem.results import ScatteringResult


def make_result() -> ScatteringResult:
    coordinates = np.vstack((np.linspace(-1.0, 1.0, 5), np.zeros(5)))
    incident = np.ones((3, 5), dtype=complex)
    scattered = (0.1 + 0.2j) * np.ones((3, 5), dtype=complex)
    return ScatteringResult(
        coordinates=coordinates,
        E_incident=incident,
        E_scattered=scattered,
        H_incident=incident / 377.0,
        H_scattered=scattered / 377.0,
        s_parameters={("left", 0, 0): 0.2j, ("right", 0, 0): 0.9},
        reflected_power=0.04,
        transmitted_power=0.81,
        radiated_power=0.1,
        absorbed_power=0.05,
        incident_power=1.0,
        ndofs=42,
        reference_planes={"left": -1.0, "right": 1.0},
        port_betas={("left", 0): 2.0, ("right", 0): 2.0},
    )


def test_total_fields_and_power_properties() -> None:
    result = make_result()
    np.testing.assert_allclose(result.E_total, result.E_incident + result.E_scattered)
    assert result.S11 == 0.2j
    assert result.S21 == 0.9
    assert result.reflection == 0.04
    assert result.transmission == 0.81
    assert result.power_balance_error < 1e-15
    assert result.check().ok


def test_field_component_and_quantity() -> None:
    result = make_result()
    np.testing.assert_allclose(result.field("Ey", quantity="real"), 1.1)
    expected_norm = np.sqrt(3.0) * abs(1.1 + 0.2j)
    np.testing.assert_allclose(result.field("E", quantity="abs"), expected_norm)
    np.testing.assert_allclose(
        result.field("Hz", quantity="abs"), abs(1.1 + 0.2j) / 377.0
    )


@pytest.mark.parametrize(
    "key",
    [
        "left",
        ("left", 0),
        ("top", 0, 0),
        ("left", -1, 0),
        ("left", 0.0, 0),
        ("left", 0, True),
    ],
)
def test_rejects_malformed_s_parameter_keys(key: object) -> None:
    with pytest.raises(ValueError, match="S-parameter|s_parameters"):
        replace(make_result(), s_parameters={key: 0.1})


@pytest.mark.parametrize("value", [np.nan, np.inf + 0j, 1j * np.inf, [0.1]])
def test_rejects_nonfinite_or_nonscalar_s_parameters(value: object) -> None:
    with pytest.raises(ValueError, match="finite complex scalar"):
        replace(make_result(), s_parameters={("left", 0, 0): value})


def test_normalizes_valid_sides_and_rejects_ambiguous_duplicates() -> None:
    normalized = replace(
        make_result(),
        s_parameters={("LEFT", 0, 0): 0.2j, ("RIGHT", 0, 0): 0.9},
    )
    assert normalized.S("LEFT") == 0.2j
    with pytest.raises(ValueError, match="Duplicate normalized"):
        replace(
            make_result(),
            s_parameters={("left", 0, 0): 0.2j, ("LEFT", 0, 0): 0.2j},
        )


@pytest.mark.parametrize(
    "port_betas",
    [
        {("top", 0): 2.0},
        {("left", -1): 2.0},
        {("left", 0.0): 2.0},
        {("left", 0): np.inf},
        {("left", 0): -2.0},
        {("left", 0): -2.0j},
        {("left",): 2.0},
    ],
)
def test_rejects_invalid_port_betas(port_betas: object) -> None:
    with pytest.raises(ValueError, match="port.beta|port_betas"):
        replace(make_result(), port_betas=port_betas)


def test_accepts_positive_imaginary_right_decaying_beta_with_roundoff() -> None:
    beta = -1e-12 + 2.0j
    result = replace(
        make_result(),
        port_betas={('left', 0): beta, ('right', 0): beta},
    )
    assert result.port_betas[('right', 0)] == beta


@pytest.mark.parametrize(
    "reference_planes",
    [
        {"top": 0.0},
        {"left": np.nan},
        {"left": 1.0 + 0.0j},
        {"left": 1.0j},
        {0: 1.0},
    ],
)
def test_rejects_invalid_reference_planes(reference_planes: object) -> None:
    with pytest.raises(ValueError, match="reference-plane|reference plane"):
        replace(make_result(), reference_planes=reference_planes)


@pytest.mark.parametrize(
    "conditions",
    [
        {"left": np.nan},
        {"left": np.inf},
        {"left": 2.0 + 0.0j},
        {"left": 0.5},
        {"": 2.0},
        {0: 2.0},
    ],
)
def test_rejects_invalid_projection_condition_numbers(conditions: object) -> None:
    with pytest.raises(ValueError, match="Projection-condition|projection condition"):
        replace(make_result(), projection_condition_numbers=conditions)


@pytest.mark.parametrize(
    ("side", "out_mode", "in_mode"),
    [("top", 0, 0), ("left", -1, 0), ("left", 0.0, 0), ("left", 0, True)],
)
def test_s_lookup_rejects_invalid_indices_and_sides(
    side: object, out_mode: object, in_mode: object
) -> None:
    with pytest.raises(ValueError):
        make_result().S(side, out_mode=out_mode, in_mode=in_mode)


def test_check_reports_projection_quality_problems() -> None:
    result = replace(
        make_result(),
        projection_condition_numbers={"left": 2e11},
        solve_info={"left_projection_residual": 2e-2},
    )
    report = result.check(
        projection_condition_warning=1e10, projection_residual_warning=1e-3
    )
    assert {item.code for item in report.warnings} >= {
        "ill_conditioned_projection",
        "poor_projection_residual",
    }


def test_check_reports_invalid_projection_residual_metadata_as_error() -> None:
    result = replace(make_result(), solve_info={"left_projection_residual": np.nan})
    report = result.check()
    assert not report.ok
    assert {item.code for item in report.diagnostics} == {
        "invalid_projection_residual"
    }


def test_check_reports_incoming_projection_mismatch() -> None:
    result = replace(
        make_result(),
        solve_info={"incoming_projection_relative_error": 2e-2},
    )
    report = result.check(incoming_projection_warning=1e-3)
    assert "incoming_projection_mismatch" in {
        item.code for item in report.warnings
    }


def test_check_reports_port_gram_normalization_error() -> None:
    result = replace(
        make_result(),
        solve_info={"forward_port_gram_diagonal_error": 3e-2},
    )
    report = result.check(port_gram_diagonal_warning=1e-2)
    assert "port_gram_normalization_error" in {
        item.code for item in report.warnings
    }


def test_check_reports_independent_balance_and_negative_raw_power() -> None:
    result = replace(
        make_result(),
        solve_info={
            "independent_energy_residual": 2e-2,
            "raw_radiated_power": -3e-2,
        },
    )
    codes = {item.code for item in result.check().warnings}
    assert "poor_independent_energy_balance" in codes
    assert "negative_raw_power" in codes


def test_check_compares_unit_power_s_parameters_to_reported_port_power() -> None:
    result = replace(
        make_result(),
        transmitted_power=0.64,
        radiated_power=0.27,
    )
    report = result.check()
    mismatches = [
        item for item in report.warnings if item.code == "s_parameter_power_mismatch"
    ]
    assert len(mismatches) == 1
    assert "transmitted" in mismatches[0].message


def test_s_power_check_excludes_evanescent_output_amplitudes() -> None:
    result = replace(
        make_result(),
        s_parameters={
            ("left", 0, 0): 0.2j,
            ("left", 1, 0): 100.0,
            ("right", 0, 0): 0.9,
        },
        port_betas={
            ("left", 0): 2.0,
            ("left", 1): 1.0j,
            ("right", 0): 2.0,
        },
    )
    assert "s_parameter_power_mismatch" not in {
        item.code for item in result.check().diagnostics
    }


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("power_balance_tolerance", np.nan),
        ("projection_condition_warning", 0.5),
        ("projection_residual_warning", -1.0),
        ("incoming_projection_warning", -1.0),
        ("port_gram_diagonal_warning", -1.0),
        ("s_parameter_power_tolerance", np.inf),
    ],
)
def test_check_rejects_invalid_thresholds(name: str, value: float) -> None:
    with pytest.raises(ValueError):
        make_result().check(**{name: value})


def test_plot_field_returns_axes_for_sampled_magnetic_field() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    result = replace(
        make_result(),
        coordinates=np.asarray(
            ((-1.0, -0.5, 0.0, 0.5, 1.0), (2.0, 2.25, 2.5, 2.75, 3.0))
        ),
    )
    axes = result.plot_field("Hz", quantity="abs", colorbar=False)
    assert axes.get_xlabel() == "z (m)"
    assert axes.get_ylabel() == "x (m)"
    assert axes.collections
    np.testing.assert_array_equal(
        axes.collections[0].get_offsets(), result.coordinates[[1, 0]].T
    )
    plt.close(axes.figure)


def test_deembedding_uses_positive_beta_propagation_convention() -> None:
    result = replace(make_result(), h5_path="original.h5")
    shifted = result.deembed(left=-1.2, right=1.3)
    dl = 0.2
    dr = -0.3
    assert shifted.S11 == pytest.approx(result.S11 * np.exp(1j * 4.0 * dl))
    assert shifted.S21 == pytest.approx(
        result.S21 * np.exp(1j * 2.0 * dl - 1j * 2.0 * dr)
    )
    assert shifted.h5_path is None


@pytest.mark.parametrize(("left", "right"), [(np.nan, 1.0), (-1.0, np.inf)])
def test_deembedding_rejects_nonfinite_reference_planes(
    left: float, right: float
) -> None:
    with pytest.raises(ValueError, match="reference plane"):
        make_result().deembed(left=left, right=right)
