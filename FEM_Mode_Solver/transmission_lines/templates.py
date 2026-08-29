"""Continuous FEM geometry templates for common transmission lines."""

from __future__ import annotations

from dataclasses import dataclass

from ..geometry import Circle, Rectangle
from ..solver_2d import ModeSolver2D
from .specs import (
    Coaxial,
    CoplanarWaveguide,
    Microstrip,
    Stripline,
    TransmissionLineSpec,
)


@dataclass(frozen=True, slots=True)
class BuiltTransmissionLine:
    """A continuous line model and its electrostatic conductor terminals."""

    solver: ModeSolver2D
    signal_boundaries: tuple[str, ...]
    reference_boundaries: tuple[str, ...]
    label: str
    metal_conductivity: float | None = None


def _effective_index(epsilon: complex, *, homogeneous: bool) -> complex:
    effective_epsilon = epsilon if homogeneous else 0.5 * (1.0 + epsilon)
    return complex(effective_epsilon**0.5)


def _new_solver(
    frequency: float,
    x_span: tuple[float, float],
    y_span: tuple[float, float],
    *,
    background_epsilon: complex,
    neff_guess: complex,
) -> ModeSolver2D:
    return ModeSolver2D(
        frequency=frequency,
        x_range=x_span,
        y_range=y_span,
        num_modes=1,
        neff_guess=neff_guess,
        background_epsilon=background_epsilon,
        boundary="pec",
    )


def _add_pec_circle(
    solver: ModeSolver2D, shape: Circle, *, name: str
) -> None:
    # ModeSolver2D's public convenience method covers rectangular walls; the
    # continuous geometry API provides the corresponding exact circle/annulus.
    solver.geometry.add_boundary(shape, "pec", name=name)


def _build_coaxial(spec: Coaxial, frequency: float) -> BuiltTransmissionLine:
    gap = spec.outer_radius - spec.inner_radius
    shield_outer_radius = spec.outer_radius + spec.outer_conductor_thickness
    exterior_clearance = max(0.35 * spec.outer_radius, 3.0 * spec.outer_conductor_thickness)
    half_extent = shield_outer_radius + exterior_clearance
    solver = _new_solver(
        frequency,
        (-half_extent, half_extent),
        (-half_extent, half_extent),
        background_epsilon=spec.complex_epsilon,
        neff_guess=_effective_index(spec.complex_epsilon, homogeneous=True),
    )
    _add_pec_circle(
        solver,
        Circle((0.0, 0.0), spec.inner_radius),
        name="signal",
    )
    _add_pec_circle(
        solver,
        Circle(
            (0.0, 0.0),
            shield_outer_radius,
            inner_radius=spec.outer_radius,
        ),
        name="outer_conductor",
    )

    # One circular refinement region resolves both conductor curves and the
    # high-field dielectric gap while allowing the disconnected exterior to
    # remain coarser.
    solver.add_mesh_refinement(
        Circle((0.0, 0.0), shield_outer_radius + 0.25 * gap),
        max_element_size=min(spec.inner_radius, gap) / 8.0,
        transition_width=0.35 * gap,
        name="coax_conductor_refinement",
    )
    return BuiltTransmissionLine(
        solver=solver,
        signal_boundaries=("signal",),
        reference_boundaries=("outer_conductor",),
        label="Coaxial",
        metal_conductivity=spec.metal_conductivity,
    )


def _build_microstrip(spec: Microstrip, frequency: float) -> BuiltTransmissionLine:
    side_clearance = spec.domain_padding_factor * max(
        1.5 * spec.trace_width, 3.0 * spec.substrate_height
    )
    half_width = 0.5 * spec.trace_width + side_clearance
    air_height = spec.domain_padding_factor * max(
        3.0 * spec.substrate_height, 1.5 * spec.trace_width
    )
    top = spec.substrate_height + spec.conductor_thickness + air_height
    solver = _new_solver(
        frequency,
        (-half_width, half_width),
        (-spec.conductor_thickness, top),
        background_epsilon=1.0 + 0.0j,
        neff_guess=_effective_index(spec.complex_epsilon, homogeneous=False),
    )
    solver.add_rectangle(
        spec.complex_epsilon,
        1.0,
        (-half_width, half_width),
        (0.0, spec.substrate_height),
        name="substrate",
    )
    solver.add_pec(
        (-half_width, half_width),
        (-spec.conductor_thickness, 0.0),
        name="ground",
    )
    solver.add_pec(
        (-0.5 * spec.trace_width, 0.5 * spec.trace_width),
        (
            spec.substrate_height,
            spec.substrate_height + spec.conductor_thickness,
        ),
        name="signal",
    )

    fringe_x = max(0.75 * spec.trace_width, 0.75 * spec.substrate_height)
    fringe_y = min(0.5 * spec.substrate_height, spec.trace_width)
    solver.add_mesh_refinement(
        Rectangle(
            (
                -0.5 * spec.trace_width - fringe_x,
                0.5 * spec.trace_width + fringe_x,
            ),
            (
                spec.substrate_height - fringe_y,
                spec.substrate_height + spec.conductor_thickness + fringe_y,
            ),
        ),
        max_element_size=min(spec.trace_width, spec.substrate_height) / 10.0,
        transition_width=max(0.5 * spec.substrate_height, 0.25 * spec.trace_width),
        name="signal_edge_refinement",
    )
    return BuiltTransmissionLine(
        solver=solver,
        signal_boundaries=("signal",),
        reference_boundaries=("ground",),
        label="Microstrip",
        metal_conductivity=spec.metal_conductivity,
    )


def _build_stripline(spec: Stripline, frequency: float) -> BuiltTransmissionLine:
    half_spacing = 0.5 * spec.ground_spacing
    signal_half_height = 0.5 * spec.conductor_thickness
    dielectric_gap = half_spacing - signal_half_height
    side_clearance = spec.domain_padding_factor * max(
        3.0 * spec.ground_spacing, 2.0 * spec.trace_width
    )
    half_width = 0.5 * spec.trace_width + side_clearance
    solver = _new_solver(
        frequency,
        (-half_width, half_width),
        (
            -half_spacing - spec.conductor_thickness,
            half_spacing + spec.conductor_thickness,
        ),
        background_epsilon=spec.complex_epsilon,
        neff_guess=_effective_index(spec.complex_epsilon, homogeneous=True),
    )
    solver.add_pec(
        (-half_width, half_width),
        (-half_spacing - spec.conductor_thickness, -half_spacing),
        name="lower_ground",
    )
    solver.add_pec(
        (-half_width, half_width),
        (half_spacing, half_spacing + spec.conductor_thickness),
        name="upper_ground",
    )
    solver.add_pec(
        (-0.5 * spec.trace_width, 0.5 * spec.trace_width),
        (-signal_half_height, signal_half_height),
        name="signal",
    )
    solver.add_mesh_refinement(
        Rectangle(
            (-1.25 * spec.trace_width, 1.25 * spec.trace_width),
            (-0.45 * dielectric_gap, 0.45 * dielectric_gap),
        ),
        max_element_size=min(spec.trace_width, dielectric_gap) / 10.0,
        transition_width=0.5 * dielectric_gap,
        name="signal_edge_refinement",
    )
    return BuiltTransmissionLine(
        solver=solver,
        signal_boundaries=("signal",),
        reference_boundaries=("lower_ground", "upper_ground"),
        label="Stripline",
        metal_conductivity=spec.metal_conductivity,
    )


def _build_cpw(
    spec: CoplanarWaveguide, frequency: float
) -> BuiltTransmissionLine:
    signal_edge = 0.5 * spec.center_width
    ground_inner_edge = signal_edge + spec.gap
    metal_half_width = ground_inner_edge + spec.ground_width
    side_clearance = spec.domain_padding_factor * max(
        2.0 * spec.substrate_height, 0.75 * metal_half_width
    )
    half_width = metal_half_width + side_clearance
    vertical_clearance = spec.domain_padding_factor * max(
        2.0 * spec.substrate_height, metal_half_width
    )
    bottom = -spec.substrate_height - vertical_clearance
    top = spec.conductor_thickness + vertical_clearance
    solver = _new_solver(
        frequency,
        (-half_width, half_width),
        (bottom, top),
        background_epsilon=1.0 + 0.0j,
        neff_guess=_effective_index(spec.complex_epsilon, homogeneous=False),
    )
    solver.add_rectangle(
        spec.complex_epsilon,
        1.0,
        (-half_width, half_width),
        (-spec.substrate_height, 0.0),
        name="substrate",
    )
    solver.add_pec(
        (-signal_edge, signal_edge),
        (0.0, spec.conductor_thickness),
        name="signal",
    )
    solver.add_pec(
        (-metal_half_width, -ground_inner_edge),
        (0.0, spec.conductor_thickness),
        name="left_ground",
    )
    solver.add_pec(
        (ground_inner_edge, metal_half_width),
        (0.0, spec.conductor_thickness),
        name="right_ground",
    )

    refinement_depth = min(0.5 * spec.substrate_height, 2.0 * spec.gap)
    solver.add_mesh_refinement(
        Rectangle(
            (-metal_half_width - spec.gap, metal_half_width + spec.gap),
            (
                -refinement_depth,
                spec.conductor_thickness + refinement_depth,
            ),
        ),
        max_element_size=min(
            spec.center_width,
            spec.gap,
            spec.ground_width,
            spec.substrate_height,
        )
        / 8.0,
        transition_width=max(2.0 * spec.gap, 0.25 * spec.substrate_height),
        name="cpw_gap_refinement",
    )
    return BuiltTransmissionLine(
        solver=solver,
        signal_boundaries=("signal",),
        reference_boundaries=("left_ground", "right_ground"),
        label="CPW",
        metal_conductivity=spec.metal_conductivity,
    )


def build_transmission_line(
    spec: TransmissionLineSpec, frequency: float
) -> BuiltTransmissionLine:
    """Build a continuous FEM cross-section without discretizing it."""

    if isinstance(spec, Coaxial):
        return _build_coaxial(spec, frequency)
    if isinstance(spec, Microstrip):
        return _build_microstrip(spec, frequency)
    if isinstance(spec, Stripline):
        return _build_stripline(spec, frequency)
    if isinstance(spec, CoplanarWaveguide):
        return _build_cpw(spec, frequency)
    raise TypeError(
        "spec must be Coaxial, Microstrip, Stripline, or CoplanarWaveguide."
    )


__all__ = ["BuiltTransmissionLine", "build_transmission_line"]
