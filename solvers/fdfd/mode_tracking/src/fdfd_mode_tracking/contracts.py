"""Data-only records for reproducible, no-PML port mode tracking."""
from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True, kw_only=True)
class PortSpec:
    """Port normal is +/-z; enclosed means the exterior conductors are physical."""
    boundary: str = 'open'
    direction: int = 1
    reference_plane: float = 0.0
    name: str = 'port'

    def __post_init__(self):
        if self.boundary not in ('open', 'enclosed'):
            raise ValueError('boundary must be open or enclosed.')
        if isinstance(self.direction, bool) or self.direction not in (-1, 1):
            raise ValueError('direction must be +1 or -1 along z.')
        if not np.isfinite(self.reference_plane):
            raise ValueError('reference_plane must be finite, in metres.')


@dataclass(frozen=True, kw_only=True)
class VerificationSpec:
    """Factory request: refine cells or add exterior padding at fixed spacing."""
    mesh_factor: int = 1
    padding_fraction: float = 0.0


@dataclass(frozen=True, kw_only=True)
class TrackingConfig:
    num_candidates: int = 6
    max_candidates: int = 12
    neff_guess: complex | None = None
    polarization: str = 'both'
    eigensolver_tolerance: float = 1e-10
    residual_tolerance: float = 1e-8
    cutoff_neff: float = 1e-6
    overlap_min: float = 0.8
    assignment_margin: float = 0.02
    unmatched_cost: float = 0.65
    cluster_gap: float = 1e-5
    verification_beta_tolerance: float = 1e-3
    verification_overlap: float = 0.999
    edge_fraction_max: float = 1e-3
    max_depth: int = 8
    max_solves: int = 200
    min_relative_step: float = 1e-5
    neural_weight: float = 0.05

    def __post_init__(self):
        for name in ('num_candidates', 'max_candidates', 'max_solves'):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
        if self.max_candidates < self.num_candidates:
            raise ValueError('max_candidates must be >= num_candidates.')
        if isinstance(self.max_depth, bool) or int(self.max_depth) != self.max_depth or self.max_depth < 0:
            raise ValueError('max_depth must be a nonnegative integer.')
        for name in ('residual_tolerance', 'cutoff_neff', 'unmatched_cost', 'cluster_gap',
                     'verification_beta_tolerance', 'min_relative_step'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive.')
        for name in ('overlap_min', 'assignment_margin', 'verification_overlap',
                     'edge_fraction_max', 'neural_weight', 'eigensolver_tolerance'):
            if not np.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) <= 1:
                raise ValueError(f'{name} must be in [0, 1].')
        if self.polarization not in ('both', 'TE', 'TM'):
            raise ValueError('polarization must be both, TE or TM.')
        if self.neff_guess is not None and not np.isfinite(self.neff_guess):
            raise ValueError('neff_guess must be finite.')


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
        return -self.result.neff**2

    @property
    def eligible(self):
        return self.numerical_valid & np.array([x == 'bound' for x in self.confinement])


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


@dataclass(kw_only=True)
class TrackedSweep:
    port: PortSpec
    config: TrackingConfig
    requested_frequencies: tuple
    samples: tuple
    events: tuple = ()
    unresolved_intervals: tuple = ()
    metadata: dict = field(default_factory=dict)

    def save(self, path):
        from .persistence import save_sweep
        return save_sweep(self, path)

    def export(self, track=0, *, frequencies=None, amplitude=1.0):
        from .export import export_track
        return export_track(self, track, frequencies=frequencies, amplitude=amplitude)

    def plot(self, *, component='E', quantity='magnitude'):
        from .visualization import plot_sweep
        return plot_sweep(self, component=component, quantity=quantity)

    def show(self, *, component='E', quantity='magnitude', block=True):
        from .visualization import show_sweep
        return show_sweep(self, component=component, quantity=quantity, block=block)


@dataclass
class PortMode:
    frequency: float
    beta: complex
    fields: dict
    field_coordinates: dict
    complex_power: complex
    metadata: dict
