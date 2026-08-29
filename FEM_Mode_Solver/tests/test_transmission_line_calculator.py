"""Integration tests for the FEM quasi-TEM transmission-line calculator."""

from __future__ import annotations

from dataclasses import replace
from math import log, pi, sqrt
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg", force=True)

import FEM_Mode_Solver as fem
from FEM_Mode_Solver.constants import C_0, ETA_0, EPSILON_0, MU_0
from FEM_Mode_Solver.transmission_lines.electrostatics import solve_quasi_tem


def test_specs_factory_and_cpw_terminal_definition() -> None:
    calculator = fem.TransmissionLineCalculator.from_type(
        "cpw",
        frequency=10.0e9,
        signal_width=0.7e-3,
        gap=0.2e-3,
        ground_width=1.5e-3,
        substrate_height=0.8e-3,
        metal_thickness=35e-6,
        substrate_epsilon=3.48,
        tan_delta=0.001,
        domain_padding=1.5,
        metal_sigma=58.0e6,
    )

    assert isinstance(calculator.spec, fem.CoplanarWaveguide)
    assert calculator.spec.domain_padding_factor == pytest.approx(1.5)
    assert calculator.spec.metal_conductivity == pytest.approx(58.0e6)
    assert calculator.solution is None
    assert calculator._built.signal_boundaries == ("signal",)
    assert calculator._built.reference_boundaries == (
        "left_ground",
        "right_ground",
    )
    assert calculator._built.label == "CPW"
    assert calculator._built.metal_conductivity == pytest.approx(58.0e6)

    with pytest.raises(fem.NotDiscretizedError):
        calculator.solve()
    with pytest.raises(fem.ConfigurationError):
        fem.TransmissionLineCalculator.coaxial(
            frequency=10.0e9,
            inner_radius=2.0e-3,
            outer_radius=1.0e-3,
        )


@pytest.mark.parametrize(
    "spec_type",
    (fem.Coaxial, fem.Microstrip, fem.Stripline, fem.CoplanarWaveguide),
)
def test_all_specs_accept_optional_metal_conductivity(spec_type) -> None:
    assert spec_type().metal_conductivity is None
    assert spec_type(metal_conductivity=59.6e6).metal_conductivity == pytest.approx(
        59.6e6
    )


@pytest.mark.parametrize(
    "bad_value",
    (0.0, -1.0, np.nan, np.inf, True, "58e6", np.asarray((58e6,))),
)
@pytest.mark.parametrize(
    "spec_type",
    (fem.Coaxial, fem.Microstrip, fem.Stripline, fem.CoplanarWaveguide),
)
def test_metal_conductivity_must_be_finite_and_positive(
    spec_type, bad_value: object
) -> None:
    with pytest.raises(fem.ConfigurationError, match="metal_conductivity"):
        spec_type(metal_conductivity=bad_value)


@pytest.mark.parametrize(
    "alias",
    (
        "conductivity",
        "conductor_conductivity",
        "bulk_conductivity",
        "sigma",
        "metal_sigma",
        "conductor_sigma",
        "metal_conductance",
    ),
)
def test_factory_normalizes_metal_conductivity_aliases(alias: str) -> None:
    calculator = fem.TransmissionLineCalculator.from_type(
        "microstrip",
        frequency=2.5e9,
        **{alias: 58.0e6},
    )
    assert calculator.spec.metal_conductivity == pytest.approx(58.0e6)
    assert calculator._built.metal_conductivity == pytest.approx(58.0e6)

    with pytest.raises(fem.ConfigurationError, match="supplied more than once"):
        fem.TransmissionLineCalculator.from_type(
            "microstrip",
            frequency=2.5e9,
            metal_conductivity=58.0e6,
            **{alias: 59.0e6},
        )


def test_open_line_padding_controls_the_remote_pec_domain() -> None:
    microstrip_default = fem.TransmissionLineCalculator.microstrip(frequency=10.0e9)
    microstrip_expanded = fem.TransmissionLineCalculator.microstrip(
        frequency=10.0e9,
        domain_padding_factor=2.0,
    )
    assert microstrip_expanded.solver.x_span[0] < microstrip_default.solver.x_span[0]
    assert microstrip_expanded.solver.x_span[1] > microstrip_default.solver.x_span[1]
    assert microstrip_expanded.solver.y_span[1] > microstrip_default.solver.y_span[1]

    cpw_default = fem.TransmissionLineCalculator.coplanar_waveguide(frequency=10.0e9)
    cpw_expanded = fem.TransmissionLineCalculator.coplanar_waveguide(
        frequency=10.0e9,
        domain_padding_factor=2.0,
    )
    assert cpw_expanded.solver.x_span[0] < cpw_default.solver.x_span[0]
    assert cpw_expanded.solver.x_span[1] > cpw_default.solver.x_span[1]
    assert cpw_expanded.solver.y_span[0] < cpw_default.solver.y_span[0]
    assert cpw_expanded.solver.y_span[1] > cpw_default.solver.y_span[1]

    with pytest.raises(fem.ConfigurationError):
        fem.TransmissionLineCalculator.microstrip(
            frequency=10.0e9,
            domain_padding_factor=0.0,
        )


@pytest.fixture(scope="module")
def solved_lines() -> dict[str, tuple[fem.TransmissionLineCalculator, fem.TransmissionLineResult]]:
    cases = {
        "coaxial": (
            {
                "inner_radius": 0.50e-3,
                "outer_radius": 1.67e-3,
                "outer_conductor_thickness": 0.15e-3,
                "epsilon_r": 2.25,
                "loss_tangent": 0.0,
            },
            0.30e-3,
        ),
        "microstrip": (
            {
                "trace_width": 3.0e-3,
                "substrate_height": 1.524e-3,
                "conductor_thickness": 35e-6,
                "epsilon_r": 3.55,
                "loss_tangent": 0.0,
            },
            0.55e-3,
        ),
        "stripline": (
            {
                "trace_width": 0.80e-3,
                "ground_spacing": 1.524e-3,
                "conductor_thickness": 35e-6,
                "epsilon_r": 3.55,
                "loss_tangent": 0.0,
            },
            0.40e-3,
        ),
        "coplanar_waveguide": (
            {
                "center_width": 0.60e-3,
                "gap": 0.25e-3,
                "ground_width": 1.50e-3,
                "substrate_height": 0.80e-3,
                "conductor_thickness": 35e-6,
                "epsilon_r": 3.55,
                "loss_tangent": 0.0,
            },
            0.40e-3,
        ),
    }
    solved: dict[
        str,
        tuple[fem.TransmissionLineCalculator, fem.TransmissionLineResult],
    ] = {}
    for kind, (parameters, maximum) in cases.items():
        calculator = fem.TransmissionLineCalculator.from_type(
            kind,
            frequency=10.0e9,
            **parameters,
        )
        calculator.discretize(max_element_size=maximum)
        result = calculator.solve()
        solved[kind] = calculator, result
    return solved


@pytest.mark.gmsh
def test_all_line_templates_solve_to_mode_compatible_vector_fields(
    solved_lines: dict[
        str,
        tuple[fem.TransmissionLineCalculator, fem.TransmissionLineResult],
    ],
) -> None:
    for kind, (calculator, result) in solved_lines.items():
        assert calculator.solution is result
        assert len(result.modes) == 1
        assert result.modes[0] is result.mode
        assert result.mode.polarization == "quasi-TEM"
        assert result.mode.normalization == "unit-voltage"
        assert result.mode.fields.components == (
            "Ex",
            "Ey",
            "Ez",
            "Hx",
            "Hy",
            "Hz",
        )
        assert result.neff.real > 1.0
        assert result.neff.imag <= 1e-12
        assert result.characteristic_impedance.real > 0.0
        assert result.wave_impedance.real > 0.0
        assert result.capacitance_per_length.real > 0.0
        assert result.inductance_per_length > 0.0
        assert np.isfinite(result.power)
        assert result.mode.fields.mesh_cells is not None

        if kind == "coplanar_waveguide":
            assert result.label == "CPW"
            assert result.mode.metadata["line_label"] == "CPW"
            assert result.modes.metadata["line_label"] == "CPW"

        terminal_names = (
            *calculator._built.signal_boundaries,
            *calculator._built.reference_boundaries,
        )
        assert all(
            calculator.mesh_data.boundary_facets[name].size > 0
            for name in terminal_names
        ), kind


@pytest.mark.gmsh
def test_coaxial_fem_matches_exact_tem_solution(
    solved_lines: dict[
        str,
        tuple[fem.TransmissionLineCalculator, fem.TransmissionLineResult],
    ],
) -> None:
    calculator, result = solved_lines["coaxial"]
    spec = calculator.spec
    assert isinstance(spec, fem.Coaxial)
    logarithm = log(spec.outer_radius / spec.inner_radius)
    expected_capacitance = 2.0 * pi * EPSILON_0 * spec.epsilon_r / logarithm
    expected_inductance = MU_0 * logarithm / (2.0 * pi)
    expected_characteristic = sqrt(expected_inductance / expected_capacitance)
    expected_wave = ETA_0 / sqrt(spec.epsilon_r)

    assert result.capacitance_per_length.real == pytest.approx(
        expected_capacitance,
        rel=0.025,
    )
    assert result.inductance_per_length == pytest.approx(
        expected_inductance,
        rel=0.025,
    )
    assert result.neff.real == pytest.approx(sqrt(spec.epsilon_r), rel=0.015)
    assert result.characteristic_impedance.real == pytest.approx(
        expected_characteristic,
        rel=0.025,
    )
    assert result.wave_impedance.real == pytest.approx(expected_wave, rel=0.015)
    assert result.power == pytest.approx(0.5 * np.conj(result.current), rel=0.015)

    finite_local = result.local_wave_impedance[
        np.isfinite(result.local_wave_impedance)
    ]
    assert finite_local.size > 0
    assert np.median(finite_local.real) == pytest.approx(expected_wave, rel=0.02)
    with pytest.raises(ValueError):
        result.electric_potential.setflags(write=True)


@pytest.mark.gmsh
def test_lossy_coax_uses_the_passive_forward_branch_and_power_conjugation() -> None:
    calculator = fem.TransmissionLineCalculator.coaxial(
        frequency=10.0e9,
        inner_radius=0.50e-3,
        outer_radius=1.67e-3,
        outer_conductor_thickness=0.15e-3,
        epsilon_r=2.25,
        loss_tangent=0.02,
    )
    calculator.discretize(max_element_size=0.30e-3)
    result = calculator.solve()
    spec = calculator.spec
    assert isinstance(spec, fem.Coaxial)

    epsilon_complex = spec.epsilon_r * (1.0 - 1j * spec.loss_tangent)
    expected_neff = np.sqrt(epsilon_complex)
    expected_wave = ETA_0 / expected_neff
    expected_characteristic = (
        expected_wave
        * np.log(spec.outer_radius / spec.inner_radius)
        / (2.0 * np.pi)
    )
    expected_power = 0.5 / np.conj(expected_characteristic)

    assert result.neff == pytest.approx(expected_neff, rel=1e-9, abs=1e-12)
    assert result.wave_impedance == pytest.approx(expected_wave, rel=1e-7)
    assert result.characteristic_impedance == pytest.approx(
        expected_characteristic,
        rel=0.025,
    )
    assert result.power == pytest.approx(expected_power, rel=0.025)
    assert result.power == pytest.approx(
        0.5 * np.conj(result.current),
        rel=1e-9,
        abs=1e-12,
    )
    assert result.neff.imag < 0.0
    assert result.characteristic_impedance.imag > 0.0
    assert result.wave_impedance.imag > 0.0
    assert result.current.imag < 0.0
    assert result.power.imag > 0.0


@pytest.mark.gmsh
def test_result_wrapper_uses_its_explicit_frequency_for_rlgc_metadata() -> None:
    calculator = fem.TransmissionLineCalculator.coaxial(
        frequency=2.0e9,
        inner_radius=0.50e-3,
        outer_radius=1.67e-3,
        outer_conductor_thickness=0.15e-3,
        epsilon_r=2.25,
        loss_tangent=0.01,
    )
    calculator.discretize(max_element_size=0.50e-3)
    solution = solve_quasi_tem(
        calculator._built,
        frequency=calculator.frequency,
    )

    result_frequency = 3.0e9
    omega = 2.0 * np.pi * result_frequency
    expected_series = complex(
        solution.resistance_per_length,
        omega * solution.inductance_per_length,
    )
    expected_shunt = complex(
        solution.conductance_per_length,
        omega * solution.capacitance_per_length.real,
    )
    expected_beta = 2.0 * np.pi * result_frequency / C_0 * solution.neff

    for metadata in ({}, {"frequency": 17.0e9}):
        result = fem.TransmissionLineResult.from_solution(
            calculator.spec,
            calculator._built,
            replace(solution, metadata=metadata),
            frequency=result_frequency,
        )

        assert result.frequency == result_frequency
        assert result.mode.beta == pytest.approx(expected_beta)
        assert result.mode.metadata["series_impedance_per_length"] == pytest.approx(
            expected_series
        )
        assert result.mode.metadata["shunt_admittance_per_length"] == pytest.approx(
            expected_shunt
        )
        assert result.series_impedance_per_length == pytest.approx(expected_series)
        assert result.shunt_admittance_per_length == pytest.approx(expected_shunt)


@pytest.mark.gmsh
def test_calculator_refinement_invalidates_and_rebuilds_solution(
    solved_lines: dict[
        str,
        tuple[fem.TransmissionLineCalculator, fem.TransmissionLineResult],
    ],
) -> None:
    calculator, _ = solved_lines["stripline"]
    initial_elements = calculator.mesh.info.elements
    calculator.refine(1.35)
    assert calculator.solution is None
    assert calculator.mesh.info.elements > initial_elements
    refined = calculator.solve()
    assert refined is calculator.solution
    assert refined.neff.real == pytest.approx(sqrt(3.55), rel=0.02)

    solved_mesh = calculator.mesh
    calculator.solver.refine(factor=1.10)
    assert calculator.mesh is not solved_mesh
    assert calculator.solution is None
    with pytest.raises(RuntimeError, match=r"solve\(\)"):
        _ = calculator.result


@pytest.mark.gmsh
def test_vector_viewer_keeps_fixed_axes_during_updates(
    solved_lines: dict[
        str,
        tuple[fem.TransmissionLineCalculator, fem.TransmissionLineResult],
    ],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.quiver import Quiver

    def assert_direction_only_arrows(axis: object, element_count: int) -> np.ndarray:
        collections = getattr(axis, "collections")
        quivers = [item for item in collections if isinstance(item, Quiver)]
        assert len(quivers) == 1
        arrow_norms = np.hypot(quivers[0].U, quivers[0].V)
        assert arrow_norms.size > 0
        np.testing.assert_allclose(arrow_norms, 1.0, rtol=1e-12, atol=1e-12)

        element_colours = [
            item
            for item in collections
            if isinstance(item, PolyCollection)
            and not isinstance(item, Quiver)
            and item.get_array() is not None
            and np.asarray(item.get_array()).size == element_count
        ]
        assert element_colours
        return np.asarray(quivers[0].get_offsets(), dtype=np.float64)

    _, result = solved_lines["coaxial"]
    element_count = result.mode.fields.mesh_cells.shape[0]
    figure, axes = result.visualize(mesh=True, show=False)
    figure.canvas.draw()
    assert axes.shape == (2,)
    assert all(axis.collections for axis in axes)
    for axis in axes:
        assert_direction_only_arrows(axis, element_count)

    viewer = result.visualize_with_gui(show=False)
    try:
        viewer.figure.canvas.draw()
        initial_arrow_offsets = [
            assert_direction_only_arrows(axis, element_count) for axis in viewer.axes
        ]
        field_positions = np.asarray(
            [axis.get_position(original=True).bounds for axis in viewer.axes]
        )
        colorbar_positions = np.asarray(
            [axis.get_position(original=True).bounds for axis in viewer.colorbar_axes]
        )
        viewer.phase_control.set_val(np.pi / 3.0)
        viewer.mesh_control.set_active(0)
        viewer.figure.canvas.draw()
        np.testing.assert_allclose(
            [axis.get_position(original=True).bounds for axis in viewer.axes],
            field_positions,
        )
        np.testing.assert_allclose(
            [axis.get_position(original=True).bounds for axis in viewer.colorbar_axes],
            colorbar_positions,
        )
        for axis, initial_offsets in zip(
            viewer.axes,
            initial_arrow_offsets,
            strict=True,
        ):
            updated_offsets = assert_direction_only_arrows(axis, element_count)
            np.testing.assert_allclose(updated_offsets, initial_offsets)
    finally:
        viewer.close()
        plt.close(figure)


def test_calculator_gui_exposes_only_the_requested_cpw_mode(monkeypatch) -> None:
    gui = fem.TransmissionLineCalculatorGUI(show=False)
    try:
        cpw_labels = [label for label in gui.line_labels if "CPW" in label]
        assert cpw_labels == ["CPW"]
        gui.line_control.set_active(gui.line_labels.index(cpw_labels[0]))
        assert gui.line_kind == "coplanar_waveguide"
        assert set(gui.parameter_boxes) == {
            "frequency",
            "center_width",
            "gap",
            "ground_width",
            "substrate_height",
            "conductor_thickness",
            "domain_padding_factor",
            "epsilon_r",
            "loss_tangent",
            "metal_conductivity",
            "max_element_size",
        }

        rendered = []

        def capture_render(result, _axes, _colorbar_axes, **options) -> None:
            rendered.append((result, options["mesh"]))

        monkeypatch.setattr(
            "FEM_Mode_Solver.transmission_lines.gui._draw_transverse_fields",
            capture_render,
        )
        result = SimpleNamespace(
            resistance_per_length=0.25,
            capacitance_per_length=1.0,
            inductance_per_length=1.0,
            conductance_per_length=2.0e-4,
            power=1.0,
            neff=1.0,
            characteristic_impedance=50.0,
            wave_impedance=50.0,
            mode=SimpleNamespace(alpha=0.03),
        )
        gui.result = result
        assert gui.mesh is False
        gui.mesh_control.set_active(0)
        assert gui.mesh is True
        assert rendered == [(result, True)]
        result_text = gui.results_text.get_text()
        assert "R' = 0.25 ohm/m" in result_text
        assert "L' = 1 H/m" in result_text
        assert "G' = 0.0002 S/m" in result_text
        assert "C' = 1 F/m" in result_text
        assert "alpha = 0.03 1/m" in result_text
        assert "P = 1 W" in result_text
    finally:
        gui.close()


def test_calculator_gui_conductivity_scaling_mesh_defaults_and_layout() -> None:
    gui = fem.TransmissionLineCalculatorGUI(show=False)
    try:
        for index, label in enumerate(gui.line_labels):
            gui.line_control.set_active(index)
            assert gui.line_label == label
            assert gui.parameter_boxes["metal_conductivity"].text == ""
            assert gui.parameter_boxes["max_element_size"].text == "1.00"

            _, mesh_size, parameters = gui._read_inputs()
            assert mesh_size == pytest.approx(1.0e-3)
            assert parameters["metal_conductivity"] is None

            gui.parameter_boxes["metal_conductivity"].set_val("58")
            _, _, parameters = gui._read_inputs()
            assert parameters["metal_conductivity"] == pytest.approx(58.0e6)

            gui.parameter_boxes["metal_conductivity"].set_val("0")
            with pytest.raises(ValueError, match="greater than zero"):
                gui._read_inputs()

        lowest_entry = min(
            axes.get_position(original=True).y0 for axes, _box in gui._entry_rows
        )
        mesh_top = sum(gui.mesh_control.ax.get_position(original=True).bounds[1::2])
        assert lowest_entry > mesh_top
    finally:
        gui.close()


@pytest.mark.gmsh
def test_calculator_gui_invalidates_solved_state_after_edits_and_selection() -> None:
    gui = fem.TransmissionLineCalculatorGUI(show=False)
    try:
        gui.parameter_boxes["max_element_size"].set_val("0.40")
        first = gui.calculate()
        assert first is not None
        assert gui.calculator is not None
        assert gui.result is first
        assert all(axis.collections for axis in gui.axes)
        assert all(axis.get_visible() for axis in gui.colorbar_axes)

        gui.parameter_boxes["epsilon_r"].set_val("2.2")
        assert gui.calculator is None
        assert gui.result is None
        assert all(not axis.collections for axis in gui.axes)
        assert all(not axis.get_visible() for axis in gui.colorbar_axes)
        assert "calculate again" in gui.status_text.get_text().casefold()

        second = gui.calculate()
        assert second is not None
        assert gui.calculator is not None
        assert gui.calculator.spec.epsilon_r == pytest.approx(2.2)

        cpw_index = gui.line_labels.index("CPW")
        gui.line_control.set_active(cpw_index)
        assert gui.line_kind == "coplanar_waveguide"
        assert gui.calculator is None
        assert gui.result is None
        assert gui.refine() is None
        assert "calculate a line" in gui.status_text.get_text().casefold()
    finally:
        gui.close()
