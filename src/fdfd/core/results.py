"""Result types shared by every solver family, and the single file loader.

``Result`` carries what every solve has: the solver family, the grid, a
:class:`SolveInfo` and free-form metadata. ``FieldResult`` adds sampled
fields at one frequency; ``ModeResult`` adds effective indices. Scattering,
band-structure and tracked-sweep results build on these. Every result has
``plot()``, ``show()`` and ``save()``; :func:`load` reads any saved result.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .constants import C0
from .errors import ConfigurationError, PersistenceError

QUANTITIES = {"real": np.real, "imag": np.imag, "magnitude": np.abs, "phase": np.angle}


@dataclass(frozen=True, kw_only=True)
class SolveInfo:
    """How a result was computed, with the same fields for every solver.

    Attributes
    ----------
    device : str
        ``"cpu"`` or ``"cuda"``.
    method : str
        ``"arpack"`` or ``"refined"`` for eigenproblems, ``"direct"`` for
        driven (scattering) solves.
    tol : float
        Requested eigensolver tolerance (0 means machine precision).
    guess : complex or None
        The search target actually used: an effective index for mode
        solvers, a frequency in hertz for band structures, ``None`` for
        scattering.
    residuals : ndarray
        Normwise backward error of each returned solution,
        ``||A x - lambda B x|| / ((||A|| + |lambda| ||B||) ||x||)``, or the
        relative residual of the linear system for driven solves.
    field_residuals : ndarray or None
        Residuals of the reconstructed field equations, where computed.
    """

    device: str
    method: str
    tol: float
    guess: complex | None
    residuals: np.ndarray
    field_residuals: np.ndarray | None = None

    @property
    def converged(self):
        """True when every residual is at most ``max(1e-8, 100 * tol)``."""
        residuals = np.asarray(self.residuals)
        return bool(residuals.size == 0 or np.all(residuals <= max(1e-8, 100 * self.tol)))


@dataclass(frozen=True, kw_only=True)
class Result:
    """Base class of every result."""

    family: str
    grid: object
    info: SolveInfo
    metadata: dict = field(default_factory=dict)
    _kind = "result"

    def save(self, path):
        """Write the result to an HDF5 file and return its path. Read it back with :func:`fdfd.load`."""
        from .persistence import atomic_h5, write_header, write_value

        with atomic_h5(path) as handle:
            write_header(handle, family=self.family, kind=self._kind)
            write_value(handle, "result", self)
        return Path(path)

    def show(self, *, block=True):
        """Open the family's interactive Matplotlib viewer and return its figure."""
        from importlib import import_module

        if not isinstance(block, bool):
            raise ConfigurationError("block must be a boolean.")
        return import_module(f"fdfd.{self.family}.viewer").show_result(self, block=block)

    def plot(self, *, mode=None, component=None, quantity="real"):
        """Return a static Matplotlib figure; see the subclass for the arguments."""
        from matplotlib.figure import Figure

        figure = Figure(figsize=(7, 5))
        self._draw(figure.subplots(), mode=mode, component=component, quantity=quantity)
        figure.tight_layout()
        return figure


@dataclass(frozen=True, kw_only=True)
class FieldResult(Result):
    """Fields sampled on their staggered Yee locations at one frequency.

    ``fields[name]`` has shape ``(*samples, n)`` where the last axis indexes
    the returned solutions; ``field_coordinates[name]`` gives the sample
    coordinates along each physical axis in metres.
    """

    frequency: float
    fields: dict
    field_coordinates: dict
    _kind = "fields"

    def __len__(self):
        return next(iter(self.fields.values())).shape[-1]

    def _default_component(self):
        return next((name for name, values in self.fields.items() if np.any(values)), next(iter(self.fields)))

    def _title(self, index):
        return ""

    def _draw(self, ax, *, mode=None, component=None, quantity="real", plane=None, position=None):
        mode = 1 if mode is None else mode
        if isinstance(mode, bool) or int(mode) != mode or not 1 <= mode <= len(self):
            raise ConfigurationError(f"mode must be a one-based integer from 1 to {len(self)}.")
        index = int(mode) - 1
        name = self._default_component() if component is None else component
        if name not in self.fields:
            raise ConfigurationError(f"Available components: {tuple(self.fields)}.")
        operation = QUANTITIES.get(quantity)
        if operation is None:
            raise ConfigurationError(f"quantity must be one of {tuple(QUANTITIES)}.")
        raw = np.asarray(self.fields[name])[..., index]
        coordinates = self.field_coordinates[name]
        axes = self.grid.axes
        if raw.ndim == 3:
            plane = "xy" if plane is None else plane
            if plane not in ("xy", "xz", "yz"):
                raise ConfigurationError("plane must be xy, xz, or yz.")
            cut = next(i for i, a in enumerate(axes) if a not in plane)
            position = np.mean(coordinates[cut]) if position is None else float(position)
            raw = np.take(raw, int(np.argmin(abs(coordinates[cut] - position))), axis=cut)
            coordinates = tuple(c for i, c in enumerate(coordinates) if i != cut)
            axes = tuple(a for i, a in enumerate(axes) if i != cut)
        values = operation(raw)
        if values.ndim == 1:
            ax.plot(coordinates[0], values)
            ax.set(xlabel=axes[0] + " (m)", ylabel=f"{name} ({quantity})")
        else:
            if axes == ("x", "z"):
                # Periodic cells read naturally with z (the period) horizontal.
                coordinates, axes, values = coordinates[::-1], axes[::-1], values.T
            ax.pcolormesh(*coordinates, values.T, shading="auto", cmap="viridis")
            ax.set(xlabel=axes[0] + " (m)", ylabel=axes[1] + " (m)", aspect="equal")
        ax.set_title(f"{self.family}: {name} ({quantity}){self._title(index)}")

    def plot(self, *, mode=None, component=None, quantity="real", plane=None, position=None):
        """Plot one field component.

        Parameters
        ----------
        mode : int, optional
            One-based solution index (default 1).
        component : str, optional
            ``"Ex"`` ... ``"Hz"``; defaults to the first nonzero component.
        quantity : {"real", "imag", "magnitude", "phase"}
        plane, position : optional
            For 3D results, the slice plane (``"xy"``, ``"xz"``, ``"yz"``)
            and its position in metres (default: the domain centre).
        """
        from matplotlib.figure import Figure

        figure = Figure(figsize=(7, 5))
        self._draw(
            figure.subplots(),
            mode=mode,
            component=component,
            quantity=quantity,
            plane=plane,
            position=position,
        )
        figure.tight_layout()
        return figure


@dataclass(frozen=True, kw_only=True)
class ModeResult(FieldResult):
    """Eigenmodes with complex effective indices ``neff`` (one per mode).

    Modes vary as ``exp(+i*omega*t - i*beta*z)`` with ``beta = neff * k0``;
    passive modes have ``Im(neff) <= 0``.
    """

    neff: np.ndarray

    @property
    def k0(self):
        """Free-space wavenumber in rad/m."""
        return 2 * np.pi * self.frequency / C0

    @property
    def beta(self):
        """Complex propagation constants in rad/m."""
        return np.asarray(self.neff) * self.k0

    @property
    def attenuation_constant(self):
        """Field attenuation ``-Im(beta)`` in Np/m."""
        return -self.beta.imag

    def __len__(self):
        return len(self.neff)

    def _title(self, index):
        value = complex(self.neff[index])
        return f"\nmode {index + 1}, neff = {value.real:.6g}{value.imag:+.6g}j"


@dataclass(frozen=True, kw_only=True)
class ScatteringResult(FieldResult):
    """Total/scattered fields of one driven solve (``len(result) == 1``)."""

    polarization: str


@dataclass(frozen=True, kw_only=True)
class BandResult(Result):
    """Complex eigenfrequencies along a Bloch path.

    ``frequencies[pol]`` has shape ``(num_modes, len(path))`` in hertz for each
    solved polarization. Passive loss gives ``Im(f) >= 0`` under
    ``exp(+i*omega*t)``.
    """

    beta_path: np.ndarray
    frequencies: dict
    _kind = "bands"

    @property
    def path_distance(self):
        """Cumulative distance along the Bloch path in rad/m."""
        return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.beta_path, axis=1), axis=0))]

    @property
    def normalized_frequencies(self):
        """Frequencies normalized as ``f * a / c`` with ``a`` the x period."""
        period = self.grid.bounds[0][1] - self.grid.bounds[0][0]
        return {key: value * period / C0 for key, value in self.frequencies.items()}

    def _draw(self, ax, *, mode=None, component=None, quantity="real"):
        operation = QUANTITIES.get(quantity)
        if operation is None or quantity == "phase":
            raise ConfigurationError("quantity must be real, imag, or magnitude.")
        selected = tuple(self.frequencies) if component is None else (component,)
        for polarization in selected:
            if polarization not in self.frequencies:
                raise ConfigurationError(f"Available polarizations: {tuple(self.frequencies)}.")
            values = self.frequencies[polarization]
            bands = range(1, len(values) + 1) if mode is None else (mode,)
            for band in bands:
                if isinstance(band, bool) or int(band) != band or not 1 <= band <= len(values):
                    raise ConfigurationError("mode must be a one-based band number.")
                ax.plot(
                    self.path_distance, operation(values[int(band) - 1]) / 1e9, label=f"{polarization} {band}"
                )
        ax.set(xlabel="Distance along Bloch path (rad/m)", ylabel=f"Frequency ({quantity}, GHz)")
        ax.legend()
        ax.grid(alpha=0.25)

    def plot(self, *, mode=None, component=None, quantity="real"):
        """Plot bands against path distance.

        Parameters
        ----------
        mode : int, optional
            One-based band number; all bands by default.
        component : {"TE", "TM"}, optional
            Polarization; all solved polarizations by default.
        quantity : {"real", "imag", "magnitude"}
        """
        return super().plot(mode=mode, component=component, quantity=quantity)


def _registry():
    from ..tracking import contracts as tracking
    from .grid import GridData

    types = [
        GridData,
        SolveInfo,
        ModeResult,
        ScatteringResult,
        BandResult,
        tracking.PortSpec,
        tracking.TrackingConfig,
        tracking.CandidateSet,
        tracking.TrackingSample,
        tracking.TrackedSweep,
        tracking.PortMode,
    ]
    return {cls.__name__: cls for cls in types}


def load(path):
    """Read any result written by ``Result.save``.

    Returns
    -------
    Result
        A :class:`ModeResult`, :class:`ScatteringResult`, :class:`BandResult`
        or :class:`fdfd.tracking.TrackedSweep`, as recorded in the file.

    Raises
    ------
    PersistenceError
        If the file is missing, corrupt, or not an fdfd 2.x result.
    """
    import h5py

    from .persistence import read_header, read_value

    try:
        with h5py.File(path, "r") as handle:
            family, _ = read_header(handle)
            result = read_value(handle["result"], _registry())
    except (OSError, KeyError, TypeError) as exc:
        raise PersistenceError(f"Cannot load {path}: {exc}") from exc
    if not isinstance(result, Result) or result.family != family:
        raise PersistenceError(f"{path} does not contain a {family!r} result.")
    return result
