"""Data-only records for reproducible, no-PML port mode tracking."""

from dataclasses import dataclass

import numpy as np

from ..core.errors import ConfigurationError
from ..core.results import Result


@dataclass(frozen=True, kw_only=True)
class PortSpec:
    """Port normal is +/-z; enclosed means the exterior conductors are physical."""

    boundary: str = "open"
    direction: int = 1
    reference_plane: float = 0.0
    name: str = "port"

    def __post_init__(self):
        if self.boundary not in ("open", "enclosed"):
            raise ConfigurationError("boundary must be open or enclosed.")
        if isinstance(self.direction, bool) or self.direction not in (-1, 1):
            raise ConfigurationError("direction must be +1 or -1 along z.")
        if not np.isfinite(self.reference_plane):
            raise ConfigurationError("reference_plane must be finite, in metres.")


@dataclass(frozen=True, kw_only=True)
class VerificationSpec:
    """Factory request: refine cells or add exterior padding at fixed spacing."""

    mesh_factor: int = 1
    padding_fraction: float = 0.0


@dataclass(frozen=True, kw_only=True)
class TrackingConfig:
    """Thresholds and budgets of :func:`track_modes`.

    ``guess``, ``polarization`` and ``tol`` are passed to each mode solve;
    the ``*_tolerance``, overlap and margin values control branch
    assignment and the independent mesh/padding verification solves.
    """

    num_candidates: int = 6
    max_candidates: int = 12
    guess: complex | None = None
    polarization: str = "both"
    tol: float = 0.0
    residual_tolerance: float = 1e-8
    cutoff_neff: float = 1e-6
    overlap_min: float = 0.9
    assignment_margin: float = 0.02
    unmatched_cost: float = 0.65
    cluster_gap: float = 1e-5
    degeneracy_tolerance: float = 1e-8
    subspace_margin: float = 0.02
    verification_beta_tolerance: float = 1e-3
    verification_overlap: float = 0.999
    edge_fraction_max: float = 1e-3
    max_depth: int = 8
    max_solves: int = 200
    min_relative_step: float = 1e-5

    def __post_init__(self):
        for name in ("num_candidates", "max_candidates", "max_solves"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ConfigurationError(f"{name} must be a positive integer.")
        if self.max_candidates < self.num_candidates:
            raise ConfigurationError("max_candidates must be >= num_candidates.")
        if isinstance(self.max_depth, bool) or int(self.max_depth) != self.max_depth or self.max_depth < 0:
            raise ConfigurationError("max_depth must be a nonnegative integer.")
        for name in (
            "residual_tolerance",
            "cutoff_neff",
            "unmatched_cost",
            "cluster_gap",
            "degeneracy_tolerance",
            "verification_beta_tolerance",
            "min_relative_step",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ConfigurationError(f"{name} must be finite and positive.")
        for name in (
            "overlap_min",
            "assignment_margin",
            "subspace_margin",
            "verification_overlap",
            "edge_fraction_max",
            "tol",
        ):
            if not np.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) <= 1:
                raise ConfigurationError(f"{name} must be in [0, 1].")
        if self.polarization not in ("both", "TE", "TM"):
            raise ConfigurationError("polarization must be both, TE or TM.")
        if self.guess is not None and not np.isfinite(self.guess):
            raise ConfigurationError("guess must be finite.")


@dataclass
class CandidateSet:
    result: object
    fields: dict
    vectors: np.ndarray
    scales: np.ndarray
    complex_power: np.ndarray
    numerical_valid: np.ndarray
    propagation: tuple
    confinement: tuple
    evidence: tuple

    @property
    def frequency(self):
        return self.result.frequency

    @property
    def eigenvalues(self):
        return -(self.result.neff**2)

    @property
    def eligible(self):
        return self.numerical_valid & np.array([x == "bound" for x in self.confinement])


@dataclass
class TrackingSample:
    candidates: CandidateSet
    candidate_indices: np.ndarray
    phases: np.ndarray
    overlaps: np.ndarray
    clusters: tuple = ()

    @property
    def frequency(self):
        return self.candidates.frequency


@dataclass(frozen=True, kw_only=True)
class TrackedSweep(Result):
    """Mode identities tracked across a frequency sweep.

    ``samples`` holds one :class:`TrackingSample` per solved frequency
    (requested and adaptively inserted), in increasing order. Track ``t`` at a
    sample is candidate ``sample.candidate_indices[t]`` (``-1`` if absent).
    ``events`` and ``unresolved_intervals`` record cutoff brackets,
    ambiguous assignments and verification failures.
    """

    port: PortSpec
    config: TrackingConfig
    requested_frequencies: tuple
    samples: tuple
    events: tuple = ()
    unresolved_intervals: tuple = ()
    _kind = "tracked sweep"

    def __post_init__(self):
        for sample in self.samples:
            metadata = sample.candidates.result.metadata
            if "pml" not in metadata or metadata["pml"]:
                raise ConfigurationError("Tracked sweeps require candidate results without a PML.")

    def export(self, track=0, *, frequencies=None, amplitude=1.0):
        """Certified port modes of one track; see :func:`fdfd.tracking.export.export_track`."""
        from .export import export_track

        return export_track(self, track, frequencies=frequencies, amplitude=amplitude)

    def plot(self, *, mode=None, component=None, quantity="real"):
        """Dispersion of every track with all candidate fields at the first frequency.

        ``component`` is ``"E"``/``"H"`` (cell-centred magnitude, the default)
        or one of ``"Ex"`` ... ``"Hz"``. ``mode`` must be ``None``: every
        candidate is shown.
        """
        if mode is not None:
            raise ConfigurationError("A tracked sweep shows every candidate; mode must be None.")
        from .viewer import plot_sweep

        return plot_sweep(self, component="E" if component is None else component, quantity=quantity)


@dataclass
class PortMode:
    frequency: float
    beta: complex
    fields: dict
    field_coordinates: dict
    complex_power: complex
    metadata: dict
