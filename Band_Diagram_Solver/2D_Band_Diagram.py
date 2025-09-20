"""Two-dimensional FDFD band diagram solver.

This module now exposes a :class:`BandDiagramSolver2D` that wraps the
workflow for defining a periodic unit cell, sweeping Bloch wave vectors
and extracting the photonic band structure.  The API mirrors the style of
the other solvers in the repository – users can programmatically add
objects to the Yee grid, request a path through the irreducible Brillouin
zone and plot the resulting bands.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Any, Callable, Iterable, Sequence


import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.sparse import diags
from scipy.sparse.linalg import eigs

from yee_derivative import yeeder2d

MaskLike = np.ndarray | Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class BandStructureResult:
    """Container returned by :meth:`BandDiagramSolver2D.compute_band_structure`.

    Attributes
    ----------
    beta_path : np.ndarray
        2×N array whose columns contain the Bloch wave-vector samples.
    tick_positions : list[int]
        Index locations that delimit the symmetry-point segments.
    tick_labels : list[str]
        Labels (Γ, X, …) associated with ``tick_positions``.
    frequencies : dict[str, np.ndarray]
        Dictionary keyed by polarisation ('TE' and/or 'TM').  Each entry is
        an array of shape (num_bands, N) containing the normalised
        frequencies ``a/λ``.
    eigenvalues : dict[str, np.ndarray]
        Un-normalised eigen-values (ω/c)² returned by the eigensolver for
        each polarisation.
    """

    beta_path: np.ndarray
    tick_positions: list[int]
    tick_labels: list[str]
    frequencies: dict[str, np.ndarray]
    eigenvalues: dict[str, np.ndarray]


class BandDiagramSolver2D:
    """Finite-difference frequency-domain band diagram solver.

    Parameters
    ----------
    a : float
        Lattice constant of the (square) unit cell.
    Nx, Ny : int
        Number of Yee cells along x and y.  ``Ny`` defaults to ``Nx``.
    background_er, background_ur : float
        Permittivity and permeability that fill the unit cell before any
        user objects are added.
    boundary_conditions : tuple[int, int]
        Boundary conditions handed to :func:`yeeder2d`.  ``1`` denotes
        periodic boundaries, ``0`` would use Dirichlet walls.
    """

    def __init__(
        self,
        a: float,
        Nx: int,
        Ny: int | None = None,
        *,
        background_er: float = 1.0,
        background_ur: float = 1.0,
        boundary_conditions: tuple[int, int] = (1, 1),
    ) -> None:
        self.a = float(a)
        self.Nx = int(Nx)
        self.Ny = int(Ny) if Ny is not None else int(Nx)
        self.boundary_conditions = tuple(boundary_conditions)

        # Spatial resolution and double-resolution (2×) Yee helper grid
        self.dx = self.a / self.Nx
        self.dy = self.a / self.Ny
        self.Nx2 = 2 * self.Nx
        self.Ny2 = 2 * self.Ny
        self.dx2 = self.dx / 2
        self.dy2 = self.dy / 2

        xa2 = np.arange(1, self.Nx2 + 1) * self.dx2
        ya2 = np.arange(1, self.Ny2 + 1) * self.dy2
        self.xa2 = xa2 - np.mean(xa2)
        self.ya2 = ya2 - np.mean(ya2)
        self.X2, self.Y2 = np.meshgrid(self.xa2, self.ya2, indexing="xy")

        # Material maps on the double-resolution grid
        self.ER2 = np.full((self.Nx2, self.Ny2), background_er, dtype=complex)
        self.UR2 = np.full((self.Nx2, self.Ny2), background_ur, dtype=complex)

        self._beta_path: np.ndarray | None = None
        self._results: BandStructureResult | None = None

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def add_object(
        self,
        mask: MaskLike,
        *,
        er: complex | float | np.ndarray | None = None,
        ur: complex | float | np.ndarray | None = None,
    ) -> None:
        """Insert an object into the unit cell.

        Parameters
        ----------
        mask : array-like or callable
            Either a boolean array defined on the 2× Yee helper grid or a
            callable ``f(X, Y)`` returning such an array.  The helper grid
            coordinates ``X`` and ``Y`` are accessible through the
            :attr:`X2` and :attr:`Y2` attributes.
        er, ur : scalar or array, optional
            Relative permittivity/permeability assigned to cells selected
            by ``mask``.  If ``None`` the respective property is left
            untouched.  Scalars broadcast across the mask; arrays must have
            the same shape as the helper grid.
        """

        selection = self._resolve_mask(mask)
        if not selection.shape == self.ER2.shape:
            raise ValueError("Mask must match the helper grid shape (2× resolution).")

        if er is not None:
            er_array = np.asarray(er, dtype=complex)
            if er_array.shape not in ((), selection.shape):
                raise ValueError("er must be a scalar or have the same shape as the mask.")
            self.ER2 = np.where(selection, er_array, self.ER2)

        if ur is not None:
            ur_array = np.asarray(ur, dtype=complex)
            if ur_array.shape not in ((), selection.shape):
                raise ValueError("ur must be a scalar or have the same shape as the mask.")
            self.UR2 = np.where(selection, ur_array, self.UR2)

    def add_circular_inclusion(
        self,
        radius: float,
        *,
        center: tuple[float, float] = (0.0, 0.0),
        er: complex | float | None = None,
        ur: complex | float | None = None,
    ) -> None:
        """Convenience wrapper that inserts a circular inclusion."""

        cx, cy = center
        mask = (self.X2 - cx) ** 2 + (self.Y2 - cy) ** 2 <= radius ** 2
        self.add_object(mask, er=er, ur=ur)

    def _resolve_mask(self, mask: MaskLike) -> np.ndarray:
        if callable(mask):
            selection = mask(self.X2, self.Y2)
        else:
            selection = mask
        selection = np.asarray(selection, dtype=bool)
        return selection

    # ------------------------------------------------------------------
    # Bloch-path utilities
    # ------------------------------------------------------------------
    def default_high_symmetry_path(self) -> tuple[list[np.ndarray], list[str]]:
        """Return the Γ–X–M–Γ path for a square lattice."""

        T1 = (2 * np.pi / self.a) * np.array([1.0, 0.0])
        T2 = (2 * np.pi / self.a) * np.array([0.0, 1.0])

        gamma = np.array([0.0, 0.0])
        x_point = 0.5 * T1
        m_point = 0.5 * (T1 + T2)

        points = [gamma, x_point, m_point, gamma]
        labels = ["Γ", "X", "M", "Γ"]
        return points, labels

    def generate_bloch_path(
        self,
        symmetry_points: Sequence[Sequence[float]],
        total_points: int,
    ) -> tuple[np.ndarray, list[int]]:
        """Sample a polyline connecting the supplied symmetry points."""

        if total_points < len(symmetry_points):
            raise ValueError("total_points must be at least the number of symmetry points.")

        pts = np.asarray(symmetry_points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("symmetry_points must be an iterable of 2D coordinates.")

        segment_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        total_length = segment_lengths.sum()
        if total_length == 0:
            # Degenerate case – distribute points uniformly
            segment_lengths = np.ones_like(segment_lengths)
            total_length = segment_lengths.sum()

        # Determine how many interpolation points to allocate to each segment
        points_remaining = total_points - 1  # first point already counted
        counts = []
        for length in segment_lengths:
            weight = length / total_length
            counts.append(max(1, int(round(points_remaining * weight))))

        # Adjust counts so that the total matches points_remaining
        diff = points_remaining - sum(counts)
        idx = 0
        while diff != 0 and counts:
            if diff > 0:
                counts[idx % len(counts)] += 1
                diff -= 1
            elif counts[idx % len(counts)] > 1:
                counts[idx % len(counts)] -= 1
                diff += 1
            idx += 1

        betas = [pts[0]]
        tick_positions = [0]
        accumulated = 0

        for seg_idx, (start, stop, n_seg) in enumerate(zip(pts[:-1], pts[1:], counts)):
            for step in range(1, n_seg + 1):
                t = step / n_seg
                betas.append(start + t * (stop - start))
            accumulated += n_seg
            tick_positions.append(accumulated)

        beta_path = np.column_stack(betas)
        self._beta_path = beta_path
        self._tick_positions = list(tick_positions)
        return beta_path, tick_positions

    # ------------------------------------------------------------------
    # Solver core
    # ------------------------------------------------------------------
    def compute_band_structure(
        self,
        beta_path: np.ndarray,
        *,
        num_bands: int,
        polarisations: Iterable[str] = ("TE", "TM"),
        eig_sigma: float = 0.0,
    ) -> BandStructureResult:
        """Solve for the requested polarisations along ``beta_path``."""

        polarisations = tuple(pol.upper() for pol in polarisations)
        allowed = {"TE", "TM"}
        if any(pol not in allowed for pol in polarisations):
            raise ValueError(f"polarisations must be drawn from {allowed}.")

        if beta_path.shape[0] != 2:
            raise ValueError("beta_path must be a 2×N array of Bloch vectors.")

        num_samples = beta_path.shape[1]
        frequencies: dict[str, np.ndarray] = {}
        eigenvalues: dict[str, np.ndarray] = {}

        tensors = self._yee_tensors()
        ERxx = tensors["ERxx"]
        ERyy = tensors["ERyy"]
        ERzz = tensors["ERzz"]
        URxx = tensors["URxx"]
        URyy = tensors["URyy"]
        URzz = tensors["URzz"]

        ERxx_diag = diags(ERxx.flatten(order="F"))
        ERyy_diag = diags(ERyy.flatten(order="F"))
        ERzz_diag = diags(ERzz.flatten(order="F"))
        URxx_diag = diags(URxx.flatten(order="F"))
        URyy_diag = diags(URyy.flatten(order="F"))
        URzz_diag = diags(URzz.flatten(order="F"))

        URxx_inv = URxx_diag.power(-1)
        URyy_inv = URyy_diag.power(-1)
        ERxx_inv = ERxx_diag.power(-1)
        ERyy_inv = ERyy_diag.power(-1)

        for pol in polarisations:
            frequencies[pol] = np.zeros((num_bands, num_samples), dtype=float)
            eigenvalues[pol] = np.zeros((num_bands, num_samples), dtype=complex)

        for idx in range(num_samples):
            beta = beta_path[:, idx]
            DEX, DEY, DHX, DHY = yeeder2d(
                [self.Nx, self.Ny],
                [self.dx, self.dy],
                list(self.boundary_conditions),
                beta,
            )

            if "TM" in polarisations:
                A_tm = -DHX @ URyy_inv @ DEX - DHY @ URxx_inv @ DEY
                vals_tm = eigs(A_tm, M=ERzz_diag, k=num_bands, sigma=eig_sigma)[0]
                eig_tm = self._sort_eigenvalues(vals_tm, num_bands)
                eigenvalues["TM"][:, idx] = eig_tm
                frequencies["TM"][:, idx] = self._normalise_eigenvalues(eig_tm)

            if "TE" in polarisations:
                A_te = -DEX @ ERyy_inv @ DHX - DEY @ ERxx_inv @ DHY
                vals_te = eigs(A_te, M=URzz_diag, k=num_bands, sigma=eig_sigma)[0]
                eig_te = self._sort_eigenvalues(vals_te, num_bands)
                eigenvalues["TE"][:, idx] = eig_te
                frequencies["TE"][:, idx] = self._normalise_eigenvalues(eig_te)

        tick_positions = self._tick_positions_from_path(beta_path)
        tick_labels = getattr(self, "_tick_labels", []) or [""] * len(tick_positions)

        result = BandStructureResult(
            beta_path=beta_path,
            tick_positions=tick_positions,
            tick_labels=tick_labels,
            frequencies=frequencies,
            eigenvalues=eigenvalues,
        )
        self._results = result
        return result

    def _tick_positions_from_path(self, beta_path: np.ndarray) -> list[int]:
        if self._beta_path is None or not np.array_equal(beta_path, self._beta_path):
            return list(range(beta_path.shape[1]))
        # _beta_path is built alongside tick positions in generate_bloch_path
        if hasattr(self, "_tick_positions"):
            return list(self._tick_positions)
        return list(range(beta_path.shape[1]))

    def set_tick_labels(self, labels: Sequence[str], positions: Sequence[int]) -> None:
        """Attach labels to the symmetry points in the Brillouin zone path."""

        if len(labels) != len(positions):
            raise ValueError("labels and positions must have the same length.")
        self._tick_labels = list(labels)
        self._tick_positions = list(positions)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot_band_diagram(
        self,
        result: BandStructureResult,
        *,
        wnmax: float | None = None,

        path_artist_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Figure, tuple[Axes, Axes, Axes]]:
        """Create the unit-cell, Bloch-path and band-diagram figure.

        Parameters
        ----------
        result : BandStructureResult
            Output from :meth:`compute_band_structure`.
        wnmax : float, optional
            Upper limit for the normalised frequency axis.
        path_artist_kwargs : dict, optional
            Extra keyword arguments forwarded to
            :meth:`matplotlib.axes.Axes.plot` when drawing the Bloch path.
        """


        beta_count = result.beta_path.shape[1]
        x_axis = np.arange(beta_count)

        fig = plt.figure(constrained_layout=True)
        gs = gridspec.GridSpec(6, 8, figure=fig)
        ax_structure = fig.add_subplot(gs[0:3, 0:3])

        ax_path = fig.add_subplot(gs[3:6, 0:3])
        ax_bands = fig.add_subplot(gs[:, 3:])


        structure_map = np.real_if_close(self.ER2)
        if np.iscomplexobj(structure_map):
            structure_map = np.abs(structure_map)

        im = ax_structure.imshow(
            np.asarray(structure_map, dtype=float).T,

            extent=(self.xa2.min(), self.xa2.max(), self.ya2.min(), self.ya2.max()),
            origin="lower",
            cmap="viridis",
        )
        ax_structure.set_title("Unit Cell")
        cbar = fig.colorbar(im, ax=ax_structure)
        cbar.set_label(r"$\epsilon_r$")


        self._draw_bloch_path_panel(ax_path, result, path_artist_kwargs)


        for pol, style in (("TM", "b"), ("TE", "r")):
            if pol in result.frequencies:
                ax_bands.plot(
                    x_axis,
                    result.frequencies[pol].T,
                    f".{style}",
                    label=pol,
                )

        handles, labels = ax_bands.get_legend_handles_labels()
        if handles:
            unique = dict(zip(labels, handles))
            ax_bands.legend(unique.values(), unique.keys())

        ticks = result.tick_positions
        labels = result.tick_labels if any(result.tick_labels) else [""] * len(ticks)
        ax_bands.set_xticks(ticks)
        ax_bands.set_xticklabels(labels)
        ax_bands.set_xlim([0, beta_count - 1])
        if wnmax is not None:
            ax_bands.set_ylim([0, wnmax])
        ax_bands.set_xlabel(r"Bloch wave vector $\vec{\beta}$")
        ax_bands.set_ylabel(r"Normalised frequency $a / \lambda_0$")
        ax_bands.set_title("Photonic Band Diagram")


        return fig, (ax_structure, ax_path, ax_bands)

    def _draw_bloch_path_panel(
        self,
        ax: Axes,
        result: BandStructureResult,
        path_artist_kwargs: dict[str, Any] | None,
    ) -> None:
        """Render the sampled Bloch path in reciprocal space."""

        beta_path = result.beta_path
        if beta_path.size == 0:
            ax.axis("off")
            return

        default_kwargs: dict[str, Any] = {
            "color": "tab:blue",
            "linewidth": 1.5,
        }
        if path_artist_kwargs:
            default_kwargs.update(path_artist_kwargs)

        ax.plot(beta_path[0], beta_path[1], **default_kwargs)
        ax.scatter(
            beta_path[0],
            beta_path[1],
            s=10,
            color=default_kwargs.get("color", "tab:blue"),
            alpha=0.3,
        )

        ticks = result.tick_positions
        labels = result.tick_labels if any(result.tick_labels) else [""] * len(ticks)

        for label, idx in zip(labels, ticks):
            if idx >= beta_path.shape[1]:
                continue
            bx, by = beta_path[:, idx]
            ax.scatter(bx, by, s=30, color="black", zorder=3)
            if label:
                ax.annotate(
                    label,
                    xy=(bx, by),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=9,
                    weight="bold",
                )

        g = np.pi / self.a
        square = np.array([
            [-g, -g],
            [g, -g],
            [g, g],
            [-g, g],
            [-g, -g],
        ])
        ax.plot(
            square[:, 0],
            square[:, 1],
            linestyle="--",
            linewidth=1.0,
            color="0.5",
            label="1st BZ",
        )

        if "1st BZ" not in [text.get_text() for text in ax.texts]:
            ax.text(
                g,
                g,
                "1st BZ",
                ha="right",
                va="bottom",
                fontsize=8,
                color="0.4",
            )

        x_vals = beta_path[0]
        y_vals = beta_path[1]
        x_min, x_max = float(np.min(x_vals)), float(np.max(x_vals))
        y_min, y_max = float(np.min(y_vals)), float(np.max(y_vals))
        x_span = x_max - x_min
        y_span = y_max - y_min
        x_pad = 0.05 * x_span if x_span else max(abs(x_max), 1.0) * 0.05
        y_pad = 0.05 * y_span if y_span else max(abs(y_max), 1.0) * 0.05

        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle=":", linewidth=0.5)
        ax.set_xlabel(r"$\beta_x$ (rad/m)")
        ax.set_ylabel(r"$\beta_y$ (rad/m)")
        ax.set_title("Bloch Path")


    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _yee_tensors(self) -> dict[str, np.ndarray]:
        ERxx = self.ER2[1::2, ::2]
        ERyy = self.ER2[::2, 1::2]
        ERzz = self.ER2[::2, 1::2]
        URxx = self.UR2[1::2, ::2]
        URyy = self.UR2[::2, 1::2]
        URzz = self.UR2[::2, 1::2]
        return {
            "ERxx": ERxx,
            "ERyy": ERyy,
            "ERzz": ERzz,
            "URxx": URxx,
            "URyy": URyy,
            "URzz": URzz,
        }

    def _sort_eigenvalues(self, values: np.ndarray, num_bands: int) -> np.ndarray:
        real_parts = np.real(values)
        order = np.argsort(real_parts)
        return values[order][:num_bands]

    def _normalise_eigenvalues(self, values: np.ndarray) -> np.ndarray:
        vals = np.real_if_close(values)
        vals = np.clip(vals.real, 0.0, None)
        return self.a / (2 * np.pi) * np.sqrt(vals)


def _example() -> None:
    """Executable example mirroring the original script."""

    solver = BandDiagramSolver2D(a=1.0, Nx=40, background_er=10.2)
    solver.add_circular_inclusion(radius=0.4, er=1.0)

    points, labels = solver.default_high_symmetry_path()
    beta_path, tick_positions = solver.generate_bloch_path(points, total_points=200)
    solver.set_tick_labels(labels, tick_positions)

    result = solver.compute_band_structure(beta_path, num_bands=5)

    fig, axes = solver.plot_band_diagram(result, wnmax=0.6)
    axes[1].set_title("Bloch Path: Γ–X–M–Γ")

    plt.show()


if __name__ == "__main__":
    _example()
