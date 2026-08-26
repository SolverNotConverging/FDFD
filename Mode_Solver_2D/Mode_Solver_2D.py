import numpy as np
from scipy.sparse import bmat, coo_matrix, diags
from scipy.sparse.linalg import eigs

if __package__:
    from .metal_surface_impedance import (
        canonical_metal_name,
        good_conductor_surface_impedance,
    )
    from .surface_impedance_boundary import (
        SurfaceImpedanceDefinition,
        compile_impedance_boundary,
        validate_impedance_pml_separation,
    )
else:
    # Direct-folder examples put only this directory on sys.path. Make the
    # repository-level shared preset module importable without duplicating it,
    # and give the local boundary compiler a dimension-specific private name.
    import importlib.util
    import sys
    from pathlib import Path

    module_directory = Path(__file__).resolve().parent
    repository_root = str(module_directory.parent)
    if repository_root in sys.path:
        sys.path.remove(repository_root)
    sys.path.insert(0, repository_root)
    from metal_surface_impedance import (
        canonical_metal_name,
        good_conductor_surface_impedance,
    )

    boundary_module_name = "_fdfd_mode_solver_2d_surface_impedance_boundary"
    boundary_module = sys.modules.get(boundary_module_name)
    if boundary_module is None:
        boundary_path = module_directory / "surface_impedance_boundary.py"
        boundary_spec = importlib.util.spec_from_file_location(
            boundary_module_name,
            boundary_path,
        )
        if boundary_spec is None or boundary_spec.loader is None:
            raise ImportError(f"Cannot load the 2D boundary compiler from {boundary_path}.")
        boundary_module = importlib.util.module_from_spec(boundary_spec)
        sys.modules[boundary_module_name] = boundary_module
        try:
            boundary_spec.loader.exec_module(boundary_module)
        except Exception:
            sys.modules.pop(boundary_module_name, None)
            raise

    SurfaceImpedanceDefinition = boundary_module.SurfaceImpedanceDefinition
    compile_impedance_boundary = boundary_module.compile_impedance_boundary
    validate_impedance_pml_separation = (
        boundary_module.validate_impedance_pml_separation
    )


class ModeSolver2D:
    """2D Yee-grid mode solver using ``exp(+j*omega*t - j*beta*z)``."""

    def __init__(self, frequency, x_range, y_range, Nx, Ny, num_modes, guess=None):
        self.frequency = frequency
        self.x_range = x_range
        self.y_range = y_range
        self.Nx = int(Nx)
        self.Ny = int(Ny)
        if self.Nx <= 0 or self.Ny <= 0:
            raise ValueError("Nx and Ny must be positive.")

        self.dx = x_range / self.Nx
        self.dy = y_range / self.Ny
        self.epsilon0 = 8.854187817e-12
        self.mu0 = 4e-7 * np.pi
        self.c = 1 / np.sqrt(self.epsilon0 * self.mu0)
        self.k_0 = 2 * np.pi * frequency / self.c
        self.dx_normalized = self.k_0 * self.dx
        self.dy_normalized = self.k_0 * self.dy

        self.shape_cell = (self.Nx, self.Ny)
        self.shape_ex = (self.Nx, self.Ny + 1)
        self.shape_ey = (self.Nx + 1, self.Ny)
        self.shape_ez = (self.Nx + 1, self.Ny + 1)
        self.shape_hx = self.shape_ey
        self.shape_hy = self.shape_ex
        self.shape_hz = self.shape_cell

        self.n_ex = int(np.prod(self.shape_ex))
        self.n_ey = int(np.prod(self.shape_ey))
        self.n_ez = int(np.prod(self.shape_ez))
        self.n_hx = self.n_ey
        self.n_hy = self.n_ex
        self.n_hz = self.Nx * self.Ny
        self.n_e = self.n_ex + self.n_ey
        self.n_h = self.n_hx + self.n_hy

        shape = self.shape_cell
        self.cell_eps_r_xx = np.ones(shape, dtype=complex)
        self.cell_eps_r_yy = np.ones(shape, dtype=complex)
        self.cell_eps_r_zz = np.ones(shape, dtype=complex)
        self.cell_mu_r_xx = np.ones(shape, dtype=complex)
        self.cell_mu_r_yy = np.ones(shape, dtype=complex)
        self.cell_mu_r_zz = np.ones(shape, dtype=complex)
        self.material_no_average_mask = np.zeros(shape, dtype=bool)

        self.eps_r_xx = np.ones(self.shape_ex, dtype=complex)
        self.eps_r_yy = np.ones(self.shape_ey, dtype=complex)
        self.eps_r_zz = np.ones(self.shape_ez, dtype=complex)
        self.mu_r_xx = np.ones(self.shape_hx, dtype=complex)
        self.mu_r_yy = np.ones(self.shape_hy, dtype=complex)
        self.mu_r_zz = np.ones(self.shape_hz, dtype=complex)

        self.pec_xx_mask = np.zeros(self.shape_ex, dtype=bool)
        self.pec_yy_mask = np.zeros(self.shape_ey, dtype=bool)
        self.pec_zz_mask = np.zeros(self.shape_ez, dtype=bool)
        self.pmc_xx_mask = np.zeros(self.shape_hx, dtype=bool)
        self.pmc_yy_mask = np.zeros(self.shape_hy, dtype=bool)
        self.pmc_zz_mask = np.zeros(self.shape_hz, dtype=bool)
        self._pec_cell_mask = np.zeros(self.shape_cell, dtype=bool)
        self._pmc_cell_mask = np.zeros(self.shape_cell, dtype=bool)
        self._pml_cell_mask = np.zeros(self.shape_cell, dtype=bool)
        self._pec_regions = []
        self._pmc_regions = []
        self._surface_impedance_owner = np.full(self.shape_cell, -1, dtype=np.int32)
        self._surface_impedance_definitions = []
        self._surface_impedance_regions = []
        self._compiled_impedance_boundary = None

        self.num_modes = int(num_modes)
        if self.num_modes <= 0:
            raise ValueError("num_modes must be positive.")
        self.guess = guess
        self._auto_guess = guess is None
        if self._auto_guess:
            self.guess = self._default_guess()
        self._invalidate_solution()

    @staticmethod
    def _max_magnitude(arr):
        values = np.abs(np.asarray(arr))
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            return 0.0
        return np.max(finite_values)

    def _default_guess(self):
        return -max(
            self._max_magnitude(arr)
            for arr in [self.cell_eps_r_xx, self.cell_eps_r_yy, self.cell_eps_r_zz,
                        self.cell_mu_r_xx, self.cell_mu_r_yy, self.cell_mu_r_zz]
        )

    def _resolve_eigs_guess(self, sigma):
        if sigma is not None:
            return sigma
        if self._auto_guess:
            self.guess = self._default_guess()
        return self.guess

    def _invalidate_solution(self):
        self.eigenvalues = None
        self.eigenvectors = None
        self.neff = None
        self.propagation_constant = None
        self.attenuation_constant = None
        for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            if hasattr(self, name):
                delattr(self, name)

    def _normalise_three(self, name, value):
        if np.isscalar(value):
            return np.full(3, value, dtype=complex)
        array = np.asarray(value, dtype=complex)
        if array.ndim == 1 and array.size == 3:
            return array
        raise ValueError(f"{name} must be a scalar or a length-3 1D array (xx, yy, zz).")

    def _validate_components(self, components):
        if isinstance(components, str):
            components = (components,)
        components = tuple(components)
        invalid = set(components) - {"xx", "yy", "zz"}
        if invalid:
            raise ValueError(f"components contains invalid tensor component(s): {sorted(invalid)}.")
        return components

    def _bound_to_index(self, value, axis):
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            step = self.dx if axis == "x" else self.dy
            return int(round(float(value) / step))
        raise ValueError("Region bounds must be int grid indices or float physical positions in metres.")

    def _coordinate_to_length(self, value, axis):
        if isinstance(value, (int, np.integer)):
            step = self.dx if axis == "x" else self.dy
            return int(value) * step
        if isinstance(value, (float, np.floating)):
            return float(value)
        raise ValueError("Coordinates must be int grid indices or float physical positions in metres.")

    def _point_to_lengths(self, name, point):
        try:
            x, y = point
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be an (x, y) pair.")
        return self._coordinate_to_length(x, "x"), self._coordinate_to_length(y, "y")

    def _range_to_lengths(self, name, values, axis):
        try:
            start, stop = values
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a (min, max) pair.")
        start = self._coordinate_to_length(start, axis)
        stop = self._coordinate_to_length(stop, axis)
        if stop <= start:
            raise ValueError(f"{name} must satisfy max > min.")
        limit = self.x_range if axis == "x" else self.y_range
        if start < 0 or stop > limit:
            raise ValueError(f"{name} is out of bounds of the simulation grid.")
        return start, stop

    def _radius_to_length(self, name, radius):
        if isinstance(radius, (int, np.integer)):
            return int(radius) * min(self.dx, self.dy)
        if isinstance(radius, (float, np.floating)):
            return float(radius)
        raise ValueError(f"{name} must be an int number of cells or a float physical radius in metres.")

    @staticmethod
    def _validate_subpixels(subpixels):
        subpixels = int(subpixels)
        if subpixels <= 0:
            raise ValueError("subpixels must be positive.")
        return subpixels

    def _region_slices(self, x_range, y_range):
        try:
            x0 = self._bound_to_index(x_range[0], "x")
            x1 = self._bound_to_index(x_range[1], "x")
            y0 = self._bound_to_index(y_range[0], "y")
            y1 = self._bound_to_index(y_range[1], "y")
        except (TypeError, IndexError):
            raise ValueError("x_range and y_range must be (min, max) pairs.")

        if not (x1 > x0 and y1 > y0):
            raise ValueError("x_range and y_range must satisfy max > min.")
        if not (0 <= x0 < x1 <= self.Nx and 0 <= y0 < y1 <= self.Ny):
            raise ValueError("Region is out of bounds of the simulation grid.")
        return slice(x0, x1), slice(y0, y1)

    def _cell_material_array(self, prefix, component):
        if prefix == "eps":
            return {"xx": self.cell_eps_r_xx, "yy": self.cell_eps_r_yy, "zz": self.cell_eps_r_zz}[component]
        if prefix == "mu":
            return {"xx": self.cell_mu_r_xx, "yy": self.cell_mu_r_yy, "zz": self.cell_mu_r_zz}[component]
        raise ValueError(f"Unknown {prefix} component {component!r}.")

    def _component_array(self, prefix, component):
        if prefix == "pec":
            return {"xx": self.pec_xx_mask, "yy": self.pec_yy_mask, "zz": self.pec_zz_mask}[component]
        if prefix == "pmc":
            return {"xx": self.pmc_xx_mask, "yy": self.pmc_yy_mask, "zz": self.pmc_zz_mask}[component]
        raise ValueError(f"Unknown {prefix} component {component!r}.")

    def _subpixel_axis(self, start, stop, step, subpixels):
        indices = np.arange(start, stop, dtype=float)
        offsets = (np.arange(subpixels, dtype=float) + 0.5) / subpixels
        return (indices[:, None] + offsets[None, :]) * step

    def _clipped_cell_bbox(self, xmin, xmax, ymin, ymax):
        x0 = max(0, int(np.floor(xmin / self.dx)))
        x1 = min(self.Nx, int(np.ceil(xmax / self.dx)))
        y0 = max(0, int(np.floor(ymin / self.dy)))
        y1 = min(self.Ny, int(np.ceil(ymax / self.dy)))
        return x0, x1, y0, y1

    def _apply_fractional_material(self, epsilon, mu, fraction, sl_x, sl_y):
        epsilon = self._normalise_three("epsilon", epsilon)
        mu = self._normalise_three("mu", mu)
        fraction = np.asarray(fraction, dtype=float)
        if fraction.shape != self.cell_eps_r_xx[sl_x, sl_y].shape:
            raise ValueError("fraction shape does not match target cell region.")

        covered = fraction > 0.0
        if not np.any(covered):
            return

        for component, value in zip(("xx", "yy", "zz"), epsilon):
            target = self._cell_material_array("eps", component)[sl_x, sl_y]
            target[covered] = target[covered] * (1.0 - fraction[covered]) + value * fraction[covered]
        for component, value in zip(("xx", "yy", "zz"), mu):
            target = self._cell_material_array("mu", component)[sl_x, sl_y]
            target[covered] = target[covered] * (1.0 - fraction[covered]) + value * fraction[covered]

        local_no_average = self.material_no_average_mask[sl_x, sl_y]
        local_no_average[covered] = False
        self.update_component_materials()
        self._invalidate_solution()

    def add_rectangle(self, epsilon, mu, x_range, y_range, *, subpixels=8):
        """
        Add a rectangular isotropic or diagonal-anisotropic material region on the cell grid.

        Boundaries are converted to fractional per-cell fill ratios before the
        material is averaged to Yee-grid locations.
        """
        x_min, x_max = self._range_to_lengths("x_range", x_range, "x")
        y_min, y_max = self._range_to_lengths("y_range", y_range, "y")
        subpixels = self._validate_subpixels(subpixels)
        x0, x1, y0, y1 = self._clipped_cell_bbox(x_min, x_max, y_min, y_max)
        if x0 >= x1 or y0 >= y1:
            return

        xs = self._subpixel_axis(x0, x1, self.dx, subpixels)
        ys = self._subpixel_axis(y0, y1, self.dy, subpixels)
        x_inside = (xs >= x_min) & (xs <= x_max)
        y_inside = (ys >= y_min) & (ys <= y_max)
        fraction = (x_inside[:, :, None, None] & y_inside[None, None, :, :]).mean(axis=(1, 3))
        self._apply_fractional_material(epsilon, mu, fraction, slice(x0, x1), slice(y0, y1))

    def add_circle(self, epsilon, mu, center, r1, r2=None, *, subpixels=8):
        """
        Add a subpixel-smoothed circular or annular material region.

        ``center`` may be integer grid coordinates or physical coordinates in metres.
        Float radii are interpreted as metres; integer radii are interpreted as cells.
        ``r1`` is the outer radius and optional ``r2`` is the inner radius.
        """
        cx, cy = self._point_to_lengths("center", center)
        r1 = self._radius_to_length("r1", r1)
        r2 = 0.0 if r2 is None else self._radius_to_length("r2", r2)
        subpixels = self._validate_subpixels(subpixels)

        if r1 <= 0:
            raise ValueError("r1 must be positive.")
        if r2 < 0 or r2 >= r1:
            raise ValueError("r2 must be non-negative and smaller than r1.")

        x0, x1, y0, y1 = self._clipped_cell_bbox(cx - r1, cx + r1, cy - r1, cy + r1)
        if x0 >= x1 or y0 >= y1:
            return

        xs = self._subpixel_axis(x0, x1, self.dx, subpixels)
        ys = self._subpixel_axis(y0, y1, self.dy, subpixels)
        radius_squared = (xs[:, :, None, None] - cx) ** 2 + (ys[None, None, :, :] - cy) ** 2
        mask = radius_squared <= r1 ** 2
        if r2 > 0:
            mask &= radius_squared >= r2 ** 2
        fraction = mask.mean(axis=(1, 3))
        self._apply_fractional_material(epsilon, mu, fraction, slice(x0, x1), slice(y0, y1))

    @staticmethod
    def _points_in_triangle(x, y, p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        d1 = (x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)
        d2 = (x - x3) * (y2 - y3) - (x2 - x3) * (y - y3)
        d3 = (x - x1) * (y3 - y1) - (x3 - x1) * (y - y1)
        has_negative = (d1 < 0) | (d2 < 0) | (d3 < 0)
        has_positive = (d1 > 0) | (d2 > 0) | (d3 > 0)
        return ~(has_negative & has_positive)

    def add_triangle(self, epsilon, mu, p1, p2, p3, *, subpixels=8):
        """
        Add a subpixel-smoothed triangular material region.

        Points may be integer grid coordinates or physical coordinates in metres.
        Coverage is first averaged on a per-cell subpixel grid, then interpolated
        to the staggered Yee material arrays.
        """
        p1 = self._point_to_lengths("p1", p1)
        p2 = self._point_to_lengths("p2", p2)
        p3 = self._point_to_lengths("p3", p3)
        subpixels = self._validate_subpixels(subpixels)

        area2 = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
        if abs(area2) <= 1e-30:
            raise ValueError("Triangle points must not be collinear.")

        xs_points = (p1[0], p2[0], p3[0])
        ys_points = (p1[1], p2[1], p3[1])
        x0, x1, y0, y1 = self._clipped_cell_bbox(min(xs_points), max(xs_points), min(ys_points), max(ys_points))
        if x0 >= x1 or y0 >= y1:
            return

        xs = self._subpixel_axis(x0, x1, self.dx, subpixels)
        ys = self._subpixel_axis(y0, y1, self.dy, subpixels)
        mask = self._points_in_triangle(xs[:, :, None, None], ys[None, None, :, :], p1, p2, p3)
        fraction = mask.mean(axis=(1, 3))
        self._apply_fractional_material(epsilon, mu, fraction, slice(x0, x1), slice(y0, y1))

    def add_pec(self, x_range, y_range, components=None):
        """Add a PEC cell region and expand it onto surrounding Yee electric components."""
        sl_x, sl_y = self._region_slices(x_range, y_range)
        selected = ("xx", "yy", "zz") if components is None else self._validate_components(components)
        cell_mask = np.zeros(self.shape_cell, dtype=bool)
        cell_mask[sl_x, sl_y] = True
        if np.any(cell_mask & self.impedance_surface_mask):
            raise ValueError("PEC region overlaps a surface-impedance region.")
        previous_masks = tuple(
            values.copy()
            for values in (self.pec_xx_mask, self.pec_yy_mask, self.pec_zz_mask)
        )
        previous_cell_mask = self._pec_cell_mask.copy()
        previous_region_count = len(self._pec_regions)
        previous_compiled_boundary = self._compiled_impedance_boundary
        try:
            self._pec_regions.append((sl_x, sl_y))
            self._pec_cell_mask |= cell_mask
            xx_mask, yy_mask, zz_mask = self.component_masks_from_cell_mask(cell_mask, field="electric")
            for comp, mask in (("xx", xx_mask), ("yy", yy_mask), ("zz", zz_mask)):
                if comp in selected:
                    self._component_array("pec", comp)[:] |= mask
            self._effective_materials_and_masks()
        except Exception:
            for target, previous in zip(
                    (self.pec_xx_mask, self.pec_yy_mask, self.pec_zz_mask),
                    previous_masks,
            ):
                target[:] = previous
            self._pec_cell_mask[:] = previous_cell_mask
            del self._pec_regions[previous_region_count:]
            self._compiled_impedance_boundary = previous_compiled_boundary
            raise
        self._invalidate_solution()

    def add_pmc(self, x_range, y_range, components=None):
        """Add a PMC cell region and expand it onto surrounding Yee magnetic components."""
        sl_x, sl_y = self._region_slices(x_range, y_range)
        selected = ("xx", "yy", "zz") if components is None else self._validate_components(components)
        cell_mask = np.zeros(self.shape_cell, dtype=bool)
        cell_mask[sl_x, sl_y] = True
        if np.any(cell_mask & self.impedance_surface_mask):
            raise ValueError("PMC region overlaps a surface-impedance region.")
        previous_masks = tuple(
            values.copy()
            for values in (self.pmc_xx_mask, self.pmc_yy_mask, self.pmc_zz_mask)
        )
        previous_cell_mask = self._pmc_cell_mask.copy()
        previous_region_count = len(self._pmc_regions)
        previous_compiled_boundary = self._compiled_impedance_boundary
        try:
            self._pmc_regions.append((sl_x, sl_y))
            self._pmc_cell_mask |= cell_mask
            xx_mask, yy_mask, zz_mask = self.component_masks_from_cell_mask(cell_mask, field="magnetic")
            for comp, mask in (("xx", xx_mask), ("yy", yy_mask), ("zz", zz_mask)):
                if comp in selected:
                    self._component_array("pmc", comp)[:] |= mask
            self._effective_materials_and_masks()
        except Exception:
            for target, previous in zip(
                    (self.pmc_xx_mask, self.pmc_yy_mask, self.pmc_zz_mask),
                    previous_masks,
            ):
                target[:] = previous
            self._pmc_cell_mask[:] = previous_cell_mask
            del self._pmc_regions[previous_region_count:]
            self._compiled_impedance_boundary = previous_compiled_boundary
            raise
        self._invalidate_solution()

    def component_masks_from_cell_mask(self, cell_mask, field="electric"):
        """Expand a cell mask to the component locations surrounding each marked cell."""
        mask = np.asarray(cell_mask, dtype=bool)
        if mask.shape != self.shape_cell:
            raise ValueError(f"cell_mask must have shape {self.shape_cell}.")

        if field == "electric":
            xx_mask = np.zeros(self.shape_ex, dtype=bool)
            yy_mask = np.zeros(self.shape_ey, dtype=bool)
            zz_mask = np.zeros(self.shape_ez, dtype=bool)
            ii, jj = np.nonzero(mask)
            xx_mask[ii, jj] = True
            xx_mask[ii, jj + 1] = True
            yy_mask[ii, jj] = True
            yy_mask[ii + 1, jj] = True
            zz_mask[ii, jj] = True
            zz_mask[ii + 1, jj] = True
            zz_mask[ii, jj + 1] = True
            zz_mask[ii + 1, jj + 1] = True
            return xx_mask, yy_mask, zz_mask

        if field == "magnetic":
            xx_mask = np.zeros(self.shape_hx, dtype=bool)
            yy_mask = np.zeros(self.shape_hy, dtype=bool)
            zz_mask = np.zeros(self.shape_hz, dtype=bool)
            ii, jj = np.nonzero(mask)
            xx_mask[ii, jj] = True
            xx_mask[ii + 1, jj] = True
            yy_mask[ii, jj] = True
            yy_mask[ii, jj + 1] = True
            zz_mask[ii, jj] = True
            return xx_mask, yy_mask, zz_mask

        raise ValueError("field must be 'electric' or 'magnetic'.")

    def add_pml(self, pml_width=50, n=3, sigma_max=5, direction="all"):
        """Add a uniaxial PML for the ``exp(+j*omega*t)`` convention."""
        pml_width = int(pml_width)
        if pml_width <= 0:
            raise ValueError("pml_width must be positive.")
        sigma_max = float(sigma_max)
        if not np.isfinite(sigma_max) or sigma_max < 0:
            raise ValueError("sigma_max must be finite and nonnegative.")
        if direction not in ("x-", "x+", "x", "y-", "y+", "y", "all"):
            raise ValueError("direction must be one of 'x-', 'x+', 'x', 'y-', 'y+', 'y', or 'all'.")

        sigma_x = np.zeros(self.shape_cell, dtype=float)
        sigma_y = np.zeros(self.shape_cell, dtype=float)

        if direction in ("x-", "x", "all"):
            for i in range(min(pml_width, self.Nx)):
                sigma_x[i, :] = sigma_max * ((pml_width - i) / pml_width) ** n
        if direction in ("x+", "x", "all"):
            for i in range(min(pml_width, self.Nx)):
                sigma_x[-i - 1, :] = sigma_max * ((pml_width - i) / pml_width) ** n
        if direction in ("y-", "y", "all"):
            for j in range(min(pml_width, self.Ny)):
                sigma_y[:, j] = sigma_max * ((pml_width - j) / pml_width) ** n
        if direction in ("y+", "y", "all"):
            for j in range(min(pml_width, self.Ny)):
                sigma_y[:, -j - 1] = sigma_max * ((pml_width - j) / pml_width) ** n

        pml_cells = (sigma_x != 0.0) | (sigma_y != 0.0)
        prospective_pml_cells = self._pml_cell_mask | pml_cells
        validate_impedance_pml_separation(
            self.impedance_surface_mask,
            prospective_pml_cells,
        )

        omega = 2 * np.pi * self.frequency
        # With exp(+j*omega*t), outgoing exp(-j*k*x) and exp(-j*k*y)
        # waves decay for stretches with negative imaginary parts.
        Sx = 1.0 - 1j * sigma_x / (self.epsilon0 * omega)
        Sy = 1.0 - 1j * sigma_y / (self.epsilon0 * omega)

        self.cell_eps_r_xx *= Sy / Sx
        self.cell_eps_r_yy *= Sx / Sy
        self.cell_eps_r_zz *= Sx * Sy
        self.cell_mu_r_xx *= Sy / Sx
        self.cell_mu_r_yy *= Sx / Sy
        self.cell_mu_r_zz *= Sx * Sy
        self._pml_cell_mask |= pml_cells
        self.update_component_materials()
        self._invalidate_solution()

    def add_UPML(self, pml_width=50, n=3, sigma_max=5, direction="all"):
        """Backward-compatible alias for add_pml()."""
        self.add_pml(pml_width=pml_width, n=n, sigma_max=sigma_max, direction=direction)

    @property
    def impedance_surface_mask(self):
        """Return a copy of the opaque cell mask used by impedance surfaces."""
        return self._surface_impedance_owner >= 0

    @staticmethod
    def _validate_surface_impedance_value(value):
        if isinstance(value, (bool, np.bool_)) or isinstance(value, (str, bytes)):
            raise TypeError("Zs must be a scalar complex impedance in ohms.")
        if not np.isscalar(value):
            raise TypeError("Zs must be a scalar complex impedance in ohms.")
        try:
            impedance = complex(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("Zs must be a scalar complex impedance in ohms.") from exc
        if not np.isfinite(impedance):
            raise ValueError("Zs must be finite.")
        if impedance == 0:
            raise ValueError("Zs must be nonzero; use add_pec(...) for the PEC limit.")
        if impedance.real < 0:
            raise ValueError("Zs must be passive with Re(Zs) >= 0.")
        return impedance

    def _surface_impedance_definition(self, Zs, preset):
        if (Zs is None) == (preset is None):
            raise ValueError("Provide exactly one of Zs or preset.")
        if not np.isfinite(self.frequency) or self.frequency <= 0:
            raise ValueError(
                "frequency must be finite and positive for surface impedance."
            )

        if preset is not None:
            canonical = canonical_metal_name(preset)
            impedance = good_conductor_surface_impedance(canonical, self.frequency)
            return SurfaceImpedanceDefinition(
                key=("preset", canonical),
                impedance=impedance,
                label=canonical,
                preset=canonical,
            )

        impedance = self._validate_surface_impedance_value(Zs)
        return SurfaceImpedanceDefinition(
            key=("constant", impedance.real, impedance.imag),
            impedance=impedance,
            label=f"Zs={impedance!r}",
        )

    def add_impedance_surface(
            self,
            Zs: complex | None = None,
            *,
            preset: str | None = None,
            x_range,
            y_range,
    ):
        """Mark an opaque cell region whose exposed faces obey a scalar SIBC.

        Exactly one of ``Zs`` or ``preset`` is required.  ``Zs`` is the scalar
        surface impedance in ohms at ``frequency``.  A metal preset evaluates
        the good-conductor impedance directly at that frequency.
        """
        definition = self._surface_impedance_definition(Zs, preset)
        sl_x, sl_y = self._region_slices(x_range, y_range)
        region_mask = np.zeros(self.shape_cell, dtype=bool)
        region_mask[sl_x, sl_y] = True

        for label, conflict_mask in (
                ("PEC", self._pec_cell_mask),
                ("PMC", self._pmc_cell_mask),
                ("PML", self._pml_cell_mask),
        ):
            if np.any(region_mask & conflict_mask):
                raise ValueError(
                    f"Surface-impedance region overlaps an existing {label} region."
                )

        definition_index = next(
            (
                index
                for index, existing in enumerate(self._surface_impedance_definitions)
                if existing.key == definition.key
            ),
            None,
        )
        existing_owners = np.unique(self._surface_impedance_owner[region_mask])
        occupied_owners = existing_owners[existing_owners >= 0]
        if occupied_owners.size:
            if definition_index is None or np.any(occupied_owners != definition_index):
                raise ValueError(
                    "Surface-impedance region has an impedance overlap with a different definition."
                )

        prospective_mask = self.impedance_surface_mask | region_mask
        if np.all(prospective_mask):
            raise ValueError("Surface-impedance geometry leaves no retained field cells.")
        validate_impedance_pml_separation(
            prospective_mask,
            self._pml_cell_mask,
        )

        if definition_index is None:
            definition_index = len(self._surface_impedance_definitions)
            self._surface_impedance_definitions.append(definition)
        unowned = region_mask & (self._surface_impedance_owner < 0)
        self._surface_impedance_owner[unowned] = definition_index
        self._surface_impedance_regions.append((sl_x, sl_y, definition.label))
        self._invalidate_solution()

    @staticmethod
    def _flat_index(i, j, nx):
        return i + j * nx

    def _difference_matrix_x(self, in_shape, out_shape, scale, forward=True):
        rows = []
        cols = []
        data = []
        in_nx, _ = in_shape
        out_nx, out_ny = out_shape
        for j in range(out_ny):
            for i in range(out_nx):
                row = self._flat_index(i, j, out_nx)
                entries = ((i + 1, j, 1.0), (i, j, -1.0)) if forward else ((i, j, 1.0), (i - 1, j, -1.0))
                for ci, cj, value in entries:
                    if 0 <= ci < in_shape[0] and 0 <= cj < in_shape[1]:
                        rows.append(row)
                        cols.append(self._flat_index(ci, cj, in_nx))
                        data.append(value / scale)
        return coo_matrix((data, (rows, cols)), shape=(out_nx * out_ny, in_shape[0] * in_shape[1])).tocsr()

    def _difference_matrix_y(self, in_shape, out_shape, scale, forward=True):
        rows = []
        cols = []
        data = []
        in_nx, _ = in_shape
        out_nx, out_ny = out_shape
        for j in range(out_ny):
            for i in range(out_nx):
                row = self._flat_index(i, j, out_nx)
                entries = ((i, j + 1, 1.0), (i, j, -1.0)) if forward else ((i, j, 1.0), (i, j - 1, -1.0))
                for ci, cj, value in entries:
                    if 0 <= ci < in_shape[0] and 0 <= cj < in_shape[1]:
                        rows.append(row)
                        cols.append(self._flat_index(ci, cj, in_nx))
                        data.append(value / scale)
        return coo_matrix((data, (rows, cols)), shape=(out_nx * out_ny, in_shape[0] * in_shape[1])).tocsr()

    def _yeeder2d(self):
        """Generate rectangular derivative matrices between true Yee component grids."""
        dx = self.dx_normalized
        dy = self.dy_normalized

        self.DEX_EZ_TO_EX = self._difference_matrix_x(self.shape_ez, self.shape_ex, dx, forward=True)
        self.DEY_EZ_TO_EY = self._difference_matrix_y(self.shape_ez, self.shape_ey, dy, forward=True)
        self.DEX_EY_TO_HZ = self._difference_matrix_x(self.shape_ey, self.shape_hz, dx, forward=True)
        self.DEY_EX_TO_HZ = self._difference_matrix_y(self.shape_ex, self.shape_hz, dy, forward=True)

        self.DHX_HY_TO_EZ = -self.DEX_EZ_TO_EX.conj().T
        self.DHY_HX_TO_EZ = -self.DEY_EZ_TO_EY.conj().T
        self.DHX_HZ_TO_HX = -self.DEX_EY_TO_HZ.conj().T
        self.DHY_HZ_TO_HY = -self.DEY_EX_TO_HZ.conj().T
        self._apply_impedance_ampere_rows()

        self.DEX = self.DEX_EY_TO_HZ
        self.DEY = self.DEY_EX_TO_HZ
        self.DHX = self.DHX_HY_TO_EZ
        self.DHY = self.DHY_HX_TO_EZ
        return (
            self.DEX_EZ_TO_EX,
            self.DEY_EZ_TO_EY,
            self.DEX_EY_TO_HZ,
            self.DEY_EX_TO_HZ,
            self.DHX_HY_TO_EZ,
            self.DHY_HX_TO_EZ,
            self.DHX_HZ_TO_HX,
            self.DHY_HZ_TO_HY,
        )

    @staticmethod
    def _replace_sparse_rows(matrix, replacements):
        editable = matrix.tolil(copy=True)
        for row, values in replacements.items():
            nonzero = sorted(
                (column, value)
                for column, value in values.items()
                if value != 0.0
            )
            editable.rows[row] = [column for column, _ in nonzero]
            editable.data[row] = [value for _, value in nonzero]
        return editable.tocsr()

    def _apply_impedance_ampere_rows(self):
        """Replace rectangular Ampere rows with clipped integral SIBC rows."""
        boundary = self._compiled_impedance_boundary
        if boundary is None or not boundary.rows:
            return

        replacements = {
            "DHX_HY_TO_EZ": {},
            "DHY_HX_TO_EZ": {},
            "DHX_HZ_TO_HX": {},
            "DHY_HZ_TO_HY": {},
        }
        electric_shapes = (self.shape_ex, self.shape_ey, self.shape_ez)
        magnetic_shapes = (self.shape_hx, self.shape_hy, self.shape_hz)

        for boundary_row in boundary.rows:
            component = boundary_row.electric_component
            row_index = self._flat_index(
                *boundary_row.electric_index,
                electric_shapes[component][0],
            )
            denominator = boundary_row.retained_dual_area * self.k_0
            if component == 2:
                replacements["DHX_HY_TO_EZ"][row_index] = {}
                replacements["DHY_HX_TO_EZ"][row_index] = {}

            for term in boundary_row.magnetic_terms:
                coefficient = term.length / denominator
                if component == 0:
                    if term.component != 2:
                        raise ValueError("Ex impedance rows may reference only Hz.")
                    matrix_name = "DHY_HZ_TO_HY"
                elif component == 1:
                    if term.component != 2:
                        raise ValueError("Ey impedance rows may reference only Hz.")
                    matrix_name = "DHX_HZ_TO_HX"
                    coefficient *= -1.0
                elif term.component == 0:
                    matrix_name = "DHY_HX_TO_EZ"
                    coefficient *= -1.0
                elif term.component == 1:
                    matrix_name = "DHX_HY_TO_EZ"
                else:
                    raise ValueError("Ez impedance rows may reference only Hx and Hy.")

                column = self._flat_index(
                    *term.index,
                    magnetic_shapes[term.component][0],
                )
                values = replacements[matrix_name].setdefault(row_index, {})
                values[column] = values.get(column, 0.0) + coefficient

        for matrix_name, rows in replacements.items():
            if rows:
                setattr(
                    self,
                    matrix_name,
                    self._replace_sparse_rows(getattr(self, matrix_name), rows),
                )

    def _average_to_ex(self, values, no_average_mask=None):
        out = np.zeros(self.shape_ex, dtype=complex)
        counts = np.zeros(self.shape_ex, dtype=float)
        out[:, :self.Ny] += values
        counts[:, :self.Ny] += 1
        out[:, 1:] += values
        counts[:, 1:] += 1
        out = out / counts
        if no_average_mask is not None:
            ii, jj = np.nonzero(no_average_mask)
            out[ii, jj] = values[ii, jj]
            out[ii, jj + 1] = values[ii, jj]
        return out

    def _average_to_ey(self, values, no_average_mask=None):
        out = np.zeros(self.shape_ey, dtype=complex)
        counts = np.zeros(self.shape_ey, dtype=float)
        out[:self.Nx, :] += values
        counts[:self.Nx, :] += 1
        out[1:, :] += values
        counts[1:, :] += 1
        out = out / counts
        if no_average_mask is not None:
            ii, jj = np.nonzero(no_average_mask)
            out[ii, jj] = values[ii, jj]
            out[ii + 1, jj] = values[ii, jj]
        return out

    def _average_to_ez(self, values, no_average_mask=None):
        out = np.zeros(self.shape_ez, dtype=complex)
        counts = np.zeros(self.shape_ez, dtype=float)
        out[:self.Nx, :self.Ny] += values
        counts[:self.Nx, :self.Ny] += 1
        out[1:, :self.Ny] += values
        counts[1:, :self.Ny] += 1
        out[:self.Nx, 1:] += values
        counts[:self.Nx, 1:] += 1
        out[1:, 1:] += values
        counts[1:, 1:] += 1
        out = out / counts
        if no_average_mask is not None:
            ii, jj = np.nonzero(no_average_mask)
            out[ii, jj] = values[ii, jj]
            out[ii + 1, jj] = values[ii, jj]
            out[ii, jj + 1] = values[ii, jj]
            out[ii + 1, jj + 1] = values[ii, jj]
        return out

    def _material_on_fields(self, eps_r_xx, eps_r_yy, eps_r_zz, mu_r_xx, mu_r_yy, mu_r_zz, no_average_mask):
        return {
            "eps_xx": self._average_to_ex(eps_r_xx, no_average_mask),
            "eps_yy": self._average_to_ey(eps_r_yy, no_average_mask),
            "eps_zz": self._average_to_ez(eps_r_zz, no_average_mask),
            "mu_xx": self._average_to_ey(mu_r_xx, no_average_mask),
            "mu_yy": self._average_to_ex(mu_r_yy, no_average_mask),
            "mu_zz": mu_r_zz.copy(),
        }

    def _set_component_materials(self, materials):
        self.eps_r_xx = materials["eps_xx"].copy()
        self.eps_r_yy = materials["eps_yy"].copy()
        self.eps_r_zz = materials["eps_zz"].copy()
        self.mu_r_xx = materials["mu_xx"].copy()
        self.mu_r_yy = materials["mu_yy"].copy()
        self.mu_r_zz = materials["mu_zz"].copy()

    def update_component_materials(self):
        """Refresh component-location material tensors from the cell-centered source grid."""
        materials = self._material_on_fields(
            self.cell_eps_r_xx,
            self.cell_eps_r_yy,
            self.cell_eps_r_zz,
            self.cell_mu_r_xx,
            self.cell_mu_r_yy,
            self.cell_mu_r_zz,
            self.material_no_average_mask,
        )
        self._set_component_materials(materials)
        return materials

    @staticmethod
    def _apply_transverse_cross_constraints(pec_xx_mask, pec_yy_mask, pmc_xx_mask, pmc_yy_mask):
        """Close the collocated transverse PEC/PMC Yee-mask pairs in place.

        A tangential PEC constraint on Ex (Ey) also constrains the collocated
        normal Hy (Hx) as PMC.  The dual PMC-to-PEC implication is required for
        the same reason, so each collocated pair uses the union of both masks.
        """
        ex_hy_mask = pec_xx_mask | pmc_yy_mask
        ey_hx_mask = pec_yy_mask | pmc_xx_mask
        pec_xx_mask[:] = ex_hy_mask
        pmc_yy_mask[:] = ex_hy_mask
        pec_yy_mask[:] = ey_hx_mask
        pmc_xx_mask[:] = ey_hx_mask

    def _restore_impedance_normal_permeability(
            self,
            materials,
            cell_mu_r_xx,
            cell_mu_r_yy,
            boundary,
    ):
        """Sample interface-normal H permeability only from retained cells."""
        owner = self._surface_impedance_owner
        for row in boundary.rows:
            i, j = row.electric_index
            if row.electric_component == 0:
                candidates = []
                if j > 0 and owner[i, j - 1] < 0:
                    candidates.append((i, j - 1))
                if j < self.Ny and owner[i, j] < 0:
                    candidates.append((i, j))
                if len(candidates) != 1:
                    raise ValueError(f"Invalid normal-Hy sampling topology at Ex({i}, {j}).")
                materials["mu_yy"][i, j] = cell_mu_r_yy[candidates[0]]
            elif row.electric_component == 1:
                candidates = []
                if i > 0 and owner[i - 1, j] < 0:
                    candidates.append((i - 1, j))
                if i < self.Nx and owner[i, j] < 0:
                    candidates.append((i, j))
                if len(candidates) != 1:
                    raise ValueError(f"Invalid normal-Hx sampling topology at Ey({i}, {j}).")
                materials["mu_xx"][i, j] = cell_mu_r_xx[candidates[0]]

    @staticmethod
    def _validate_impedance_row_conflicts(
            boundary,
            electric_constraints,
            magnetic_constraints,
    ):
        component_names = ("Ex", "Ey", "Ez")
        magnetic_names = ("Hx", "Hy", "Hz")
        occupied = set()
        for row in boundary.rows:
            key = (row.electric_component, row.electric_index)
            if key in occupied:
                raise ValueError(f"Duplicate surface-impedance Ampere row {key}.")
            occupied.add(key)
            if electric_constraints[row.electric_component][row.electric_index]:
                name = component_names[row.electric_component]
                raise ValueError(
                    f"Surface-impedance boundary conflicts with a PEC/PMC constraint at "
                    f"{name}{row.electric_index}."
                )
            for term in row.magnetic_terms:
                if magnetic_constraints[term.component][term.index]:
                    name = magnetic_names[term.component]
                    raise ValueError(
                        "Surface-impedance boundary references a PEC/PMC-constrained "
                        f"{name}{term.index}."
                    )

    def _effective_materials_and_masks(self):
        eps_r_xx = self.cell_eps_r_xx.copy()
        eps_r_yy = self.cell_eps_r_yy.copy()
        eps_r_zz = self.cell_eps_r_zz.copy()
        mu_r_xx = self.cell_mu_r_xx.copy()
        mu_r_yy = self.cell_mu_r_yy.copy()
        mu_r_zz = self.cell_mu_r_zz.copy()
        no_average_mask = self.material_no_average_mask.copy()

        pec_xx_mask = self.pec_xx_mask.copy()
        pec_yy_mask = self.pec_yy_mask.copy()
        pec_zz_mask = self.pec_zz_mask.copy()
        pmc_xx_mask = self.pmc_xx_mask.copy()
        pmc_yy_mask = self.pmc_yy_mask.copy()
        pmc_zz_mask = self.pmc_zz_mask.copy()

        opaque_cells = self.impedance_surface_mask
        if np.any(opaque_cells):
            electric_bad = (
                ~np.isfinite(eps_r_xx)
                | ~np.isfinite(eps_r_yy)
                | ~np.isfinite(eps_r_zz)
            )
            magnetic_bad = (
                ~np.isfinite(mu_r_xx)
                | ~np.isfinite(mu_r_yy)
                | ~np.isfinite(mu_r_zz)
            )
            if np.any(opaque_cells & electric_bad):
                raise ValueError("Surface-impedance cells overlap a PEC material region.")
            if np.any(opaque_cells & magnetic_bad):
                raise ValueError("Surface-impedance cells overlap a PMC material region.")

        electric_targets = {"xx": pec_xx_mask, "yy": pec_yy_mask, "zz": pec_zz_mask}
        magnetic_targets = {"xx": pmc_xx_mask, "yy": pmc_yy_mask, "zz": pmc_zz_mask}
        mask_index = {"xx": 0, "yy": 1, "zz": 2}
        for component, values in (("xx", eps_r_xx), ("yy", eps_r_yy), ("zz", eps_r_zz)):
            bad_cells = ~np.isfinite(values)
            if np.any(bad_cells):
                masks = self.component_masks_from_cell_mask(bad_cells, field="electric")
                electric_targets[component][:] |= masks[mask_index[component]]
                values[bad_cells] = 1.0 + 0j

        for component, values in (("xx", mu_r_xx), ("yy", mu_r_yy), ("zz", mu_r_zz)):
            bad_cells = ~np.isfinite(values)
            if np.any(bad_cells):
                masks = self.component_masks_from_cell_mask(bad_cells, field="magnetic")
                magnetic_targets[component][:] |= masks[mask_index[component]]
                values[bad_cells] = 1.0 + 0j

        self._apply_transverse_cross_constraints(
            pec_xx_mask,
            pec_yy_mask,
            pmc_xx_mask,
            pmc_yy_mask,
        )

        materials = self._material_on_fields(
            eps_r_xx,
            eps_r_yy,
            eps_r_zz,
            mu_r_xx,
            mu_r_yy,
            mu_r_zz,
            no_average_mask,
        )

        if np.any(opaque_cells):
            boundary = compile_impedance_boundary(
                owner=self._surface_impedance_owner,
                definitions=tuple(self._surface_impedance_definitions),
                cell_eps_r_xx=eps_r_xx,
                cell_eps_r_yy=eps_r_yy,
                cell_eps_r_zz=eps_r_zz,
                dx=self.dx,
                dy=self.dy,
                frequency=self.frequency,
                epsilon0=self.epsilon0,
                pml_cells=self._pml_cell_mask,
            )
            self._compiled_impedance_boundary = boundary
            electric_constraints = (pec_xx_mask, pec_yy_mask, pec_zz_mask)
            magnetic_constraints = (pmc_xx_mask, pmc_yy_mask, pmc_zz_mask)
            self._validate_impedance_row_conflicts(
                boundary,
                electric_constraints,
                magnetic_constraints,
            )
            electric_materials = (
                materials["eps_xx"],
                materials["eps_yy"],
                materials["eps_zz"],
            )
            for row in boundary.rows:
                electric_materials[row.electric_component][row.electric_index] = (
                    row.relative_permittivity
                )
            self._restore_impedance_normal_permeability(
                materials,
                mu_r_xx,
                mu_r_yy,
                boundary,
            )
            for constraint, retained in zip(electric_constraints, boundary.electric_retained):
                constraint[:] |= ~retained
            for constraint, retained in zip(magnetic_constraints, boundary.magnetic_retained):
                constraint[:] |= ~retained
        else:
            self._compiled_impedance_boundary = None

        materials["eps_xx"][pec_xx_mask] = 1.0 + 0j
        materials["eps_yy"][pec_yy_mask] = 1.0 + 0j
        materials["eps_zz"][pec_zz_mask] = 1.0 + 0j
        materials["mu_xx"][pmc_xx_mask] = 1.0 + 0j
        materials["mu_yy"][pmc_yy_mask] = 1.0 + 0j
        materials["mu_zz"][pmc_zz_mask] = 1.0 + 0j
        self._set_component_materials(materials)

        return materials, pec_xx_mask, pec_yy_mask, pec_zz_mask, pmc_xx_mask, pmc_yy_mask, pmc_zz_mask

    def _inverse_diag_on_free(self, values, constrained_mask):
        flat = values.ravel(order="F")
        constrained = constrained_mask.ravel(order="F")
        inverse = np.zeros_like(flat, dtype=complex)
        inverse[~constrained] = 1.0 / flat[~constrained]
        return diags(inverse, format="csr")

    @staticmethod
    def _diag(values):
        return diags(values.ravel(order="F"), format="csr")

    def _unflatten_modes(self, flat_modes, shape):
        # Return independent field storage.  Electric slices can otherwise
        # alias self.eigenvectors and receive the final common phase twice.
        return np.asarray(flat_modes, dtype=complex).reshape(
            (*shape, flat_modes.shape[1]),
            order="F",
        ).copy()

    def _zero_constrained_fields(self, pec_xx_mask, pec_yy_mask, pec_zz_mask, pmc_xx_mask, pmc_yy_mask, pmc_zz_mask):
        self.Ex[pec_xx_mask, :] = 0.0
        self.Ey[pec_yy_mask, :] = 0.0
        self.Ez[pec_zz_mask, :] = 0.0
        self.Hx[pmc_xx_mask, :] = 0.0
        self.Hy[pmc_yy_mask, :] = 0.0
        self.Hz[pmc_zz_mask, :] = 0.0

    @staticmethod
    def _most_real_phase(*fields):
        values = np.concatenate([np.asarray(field).ravel() for field in fields])
        finite = np.isfinite(values)
        values = values[finite]
        if values.size == 0 or np.max(np.abs(values)) == 0:
            return 1.0 + 0j
        return np.exp(-0.5j * np.angle(np.sum(values ** 2)))

    def _rotate_modes_to_most_real(self):
        for mode in range(self.num_modes):
            phase = self._most_real_phase(
                self.Ex[:, :, mode],
                self.Ey[:, :, mode],
                self.Ez[:, :, mode],
                self.Hx[:, :, mode],
                self.Hy[:, :, mode],
                self.Hz[:, :, mode],
            )
            self.Ex[:, :, mode] *= phase
            self.Ey[:, :, mode] *= phase
            self.Ez[:, :, mode] *= phase
            self.Hx[:, :, mode] *= phase
            self.Hy[:, :, mode] *= phase
            self.Hz[:, :, mode] *= phase
            self.eigenvectors[:, mode] *= phase

    def solve(self, sigma=None):
        """Solve for transverse modes and recover all six staggered field components."""
        sigma = self._resolve_eigs_guess(sigma)
        materials, pec_xx_mask, pec_yy_mask, pec_zz_mask, pmc_xx_mask, pmc_yy_mask, pmc_zz_mask = (
            self._effective_materials_and_masks()
        )

        eps_xx_diag = self._diag(materials["eps_xx"])
        eps_yy_diag = self._diag(materials["eps_yy"])
        mu_xx_diag = self._diag(materials["mu_xx"])
        mu_yy_diag = self._diag(materials["mu_yy"])
        eps_zz_inv = self._inverse_diag_on_free(materials["eps_zz"], pec_zz_mask)
        mu_zz_inv = self._inverse_diag_on_free(materials["mu_zz"], pmc_zz_mask)

        (
            d_ez_ex,
            d_ez_ey,
            d_ey_hz,
            d_ex_hz,
            d_hy_ez,
            d_hx_ez,
            d_hz_hx,
            d_hz_hy,
        ) = self._yeeder2d()

        P11 = d_ez_ex @ eps_zz_inv @ d_hx_ez
        P12 = -(d_ez_ex @ eps_zz_inv @ d_hy_ez + mu_yy_diag)
        P21 = d_ez_ey @ eps_zz_inv @ d_hx_ez + mu_xx_diag
        P22 = -d_ez_ey @ eps_zz_inv @ d_hy_ez
        P = bmat([[P11, P12], [P21, P22]], format="csr")

        Q11 = d_hz_hx @ mu_zz_inv @ d_ex_hz
        Q12 = -(d_hz_hx @ mu_zz_inv @ d_ey_hz + eps_yy_diag)
        Q21 = d_hz_hy @ mu_zz_inv @ d_ex_hz + eps_xx_diag
        Q22 = -d_hz_hy @ mu_zz_inv @ d_ey_hz
        Q = bmat([[Q11, Q12], [Q21, Q22]], format="csr")

        free_ex = ~pec_xx_mask.ravel(order="F")
        free_ey = ~pec_yy_mask.ravel(order="F")
        free_hx = ~pmc_xx_mask.ravel(order="F")
        free_hy = ~pmc_yy_mask.ravel(order="F")
        free_exy = np.concatenate((free_ex, free_ey))
        free_hxy = np.concatenate((free_hx, free_hy))

        P_reduced = P[:, free_hxy]
        Q_reduced = Q[free_hxy, :]
        Omega = P_reduced @ Q_reduced
        Omega = Omega[free_exy, :][:, free_exy]
        if Omega.shape[0] <= self.num_modes:
            raise ValueError(
                f"Not enough unconstrained electric DOFs ({Omega.shape[0]}) to solve {self.num_modes} modes."
            )

        eigenvalues, eigenvectors_reduced = eigs(Omega, k=self.num_modes, sigma=sigma)
        eigenvectors = np.zeros((self.n_e, self.num_modes), dtype=complex)
        eigenvectors[free_exy, :] = eigenvectors_reduced

        order = np.argsort(np.real(eigenvalues))
        self.eigenvalues = eigenvalues[order]
        Exy_flat = eigenvectors[:, order]
        self.eigenvectors = Exy_flat
        self.neff = self._passive_positive_neff(-self.eigenvalues)
        self.propagation_constant = np.real(self.neff)
        self.attenuation_constant = -np.imag(self.neff)

        # eigenvalue = -neff**2.  For exp(+j*omega*t - j*beta*z), the
        # propagation branch used in the E/H reconstruction is +j*neff.
        sqrt_eigenvalues = 1j * self.neff
        if np.any(np.abs(sqrt_eigenvalues) < 1e-300):
            raise ValueError("Encountered a near-zero eigenvalue while reconstructing magnetic fields.")
        eigenvalues_inv = diags(1.0 / sqrt_eigenvalues, format="csr")

        ex_flat = np.asarray(Exy_flat[:self.n_ex, :], dtype=complex)
        ey_flat = np.asarray(Exy_flat[self.n_ex:, :], dtype=complex)
        Hxy_reduced = Q_reduced @ Exy_flat @ eigenvalues_inv
        Hxy_flat = np.zeros((self.n_h, self.num_modes), dtype=complex)
        Hxy_flat[free_hxy, :] = Hxy_reduced
        hx_flat = np.asarray(Hxy_flat[:self.n_hx, :], dtype=complex)
        hy_flat = np.asarray(Hxy_flat[self.n_hx:, :], dtype=complex)
        ez_flat = np.asarray(eps_zz_inv @ (d_hy_ez @ hy_flat - d_hx_ez @ hx_flat), dtype=complex)
        hz_flat = np.asarray(mu_zz_inv @ (d_ey_hz @ ey_flat - d_ex_hz @ ex_flat), dtype=complex)

        self.Ex = self._unflatten_modes(ex_flat, self.shape_ex)
        self.Ey = self._unflatten_modes(ey_flat, self.shape_ey)
        self.Ez = self._unflatten_modes(ez_flat, self.shape_ez)
        self.Hx = self._unflatten_modes(hx_flat, self.shape_hx)
        self.Hy = self._unflatten_modes(hy_flat, self.shape_hy)
        self.Hz = self._unflatten_modes(hz_flat, self.shape_hz)
        self._zero_constrained_fields(pec_xx_mask, pec_yy_mask, pec_zz_mask, pmc_xx_mask, pmc_yy_mask, pmc_zz_mask)
        self._rotate_modes_to_most_real()

    def _has_lossy_material(self):
        for values in (
                self.cell_eps_r_xx,
                self.cell_eps_r_yy,
                self.cell_eps_r_zz,
                self.cell_mu_r_xx,
                self.cell_mu_r_yy,
                self.cell_mu_r_zz,
        ):
            finite = np.isfinite(values)
            if np.any(np.imag(values[finite]) < -1e-14):
                return True
        return any(
            definition.impedance.real > 0
            for definition in self._surface_impedance_definitions
        )

    def _passive_positive_neff(self, neff_squared):
        """Select positive-phase or purely evanescent-decaying modal roots."""
        root = np.sqrt(neff_squared)
        tolerance = 1e-12 * np.maximum(1.0, np.abs(root))
        flip = (np.real(root) < -tolerance) | (
            (np.abs(np.real(root)) <= tolerance) & (np.imag(root) > tolerance)
        )
        neff = np.where(flip, -root, root)
        real = np.real(neff)
        imag = np.imag(neff)
        real = np.where(np.abs(real) <= tolerance, 0.0, real)
        imag = np.where(np.abs(imag) <= tolerance, 0.0, imag)
        return real + 1j * imag

    def _field_array(self, field, mode):
        return field[:, :, mode]

    def _center_to_cells(self, name, data):
        if name in ("ex", "hy"):
            return 0.5 * (data[:, :self.Ny] + data[:, 1:])
        if name in ("ey", "hx"):
            return 0.5 * (data[:self.Nx, :] + data[1:, :])
        if name == "ez":
            return 0.25 * (data[:self.Nx, :self.Ny] + data[1:, :self.Ny] + data[:self.Nx, 1:] + data[1:, 1:])
        if name == "hz":
            return data
        raise ValueError(f"Unknown field name {name!r}.")

    def _material_background_for_field(self, name):
        if name == "ex":
            return self.eps_r_xx
        if name == "ey":
            return self.eps_r_yy
        if name == "ez":
            return self.eps_r_zz
        if name == "hx":
            return self.mu_r_xx
        if name == "hy":
            return self.mu_r_yy
        if name == "hz":
            return self.mu_r_zz
        if name == "eabs":
            return (self.cell_eps_r_xx + self.cell_eps_r_yy + self.cell_eps_r_zz) / 3.0
        if name == "habs":
            return (self.cell_mu_r_xx + self.cell_mu_r_yy + self.cell_mu_r_zz) / 3.0
        raise ValueError(f"Unknown field name {name!r}.")

    def _plot_material_background(self, ax, field_name):
        material = np.abs(self._material_background_for_field(field_name))
        ax.imshow(
            material.T,
            cmap="inferno",
            origin="lower",
            extent=[0, self.x_range * 1e3, 0, self.y_range * 1e3],
            alpha=0.5,
            zorder=0,
        )

    def _add_boundary_background(self, ax):
        for sl_x, sl_y in self._pec_regions:
            self._add_boundary_rectangle(ax, sl_x, sl_y, label="PEC")
        for sl_x, sl_y in self._pmc_regions:
            self._add_boundary_rectangle(ax, sl_x, sl_y, label="PMC")
        for sl_x, sl_y, surface_label in self._surface_impedance_regions:
            self._add_boundary_rectangle(
                ax,
                sl_x,
                sl_y,
                label=f"SIBC: {surface_label}",
            )

    def _add_boundary_rectangle(self, ax, sl_x, sl_y, label):
        from matplotlib.patches import Rectangle

        x0 = sl_x.start * self.dx * 1e3
        x1 = sl_x.stop * self.dx * 1e3
        y0 = sl_y.start * self.dy * 1e3
        y1 = sl_y.stop * self.dy * 1e3
        if label == "PEC":
            facecolor, edgecolor = "yellow", "goldenrod"
        elif label == "PMC":
            facecolor, edgecolor = "blue", "navy"
        else:
            facecolor, edgecolor = "magenta", "purple"
        ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=1.0,
                alpha=0.5,
                label=label,
                zorder=1,
            )
        )

    def _plot_field_subplot(self, ax, field_name, field_data, title, norm=None):
        self._plot_material_background(ax, field_name)
        self._add_boundary_background(ax)
        if norm is None:
            norm = np.max(np.abs(field_data))
        field = field_data / norm if norm > 0 else field_data
        image = ax.imshow(
            np.abs(field).T,
            cmap="viridis",
            origin="lower",
            extent=[0, self.x_range * 1e3, 0, self.y_range * 1e3],
            vmin=0 if norm is not None else None,
            vmax=1 if norm is not None else None,
            alpha=0.9,
            zorder=2,
        )
        ax.set_title(title)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        return image

    def _component_fields_for_mode(self, mode):
        return {
            "ex": (self._field_array(self.Ex, mode), "Ex"),
            "ey": (self._field_array(self.Ey, mode), "Ey"),
            "ez": (self._field_array(self.Ez, mode), "Ez"),
            "hx": (self._field_array(self.Hx, mode), "Hx"),
            "hy": (self._field_array(self.Hy, mode), "Hy"),
            "hz": (self._field_array(self.Hz, mode), "Hz"),
        }

    def visualize(self, mode=1, **kwargs):
        """Visualize selected field components for a given one-based mode index."""
        from matplotlib import pyplot as plt

        if self.eigenvalues is None:
            raise RuntimeError("solve() must be called before visualize().")
        mode -= 1
        if not (0 <= mode < self.num_modes):
            raise ValueError("mode is out of range.")

        fields = self._component_fields_for_mode(mode)
        e_abs = np.sqrt(sum(np.abs(self._center_to_cells(key, fields[key][0])) ** 2 for key in ("ex", "ey", "ez")))
        h_abs = np.sqrt(sum(np.abs(self._center_to_cells(key, fields[key][0])) ** 2 for key in ("hx", "hy", "hz")))
        fields["eabs"] = (e_abs, "|E| cell-centered")
        fields["habs"] = (h_abs, "|H| cell-centered")

        selected = [key for key in fields if kwargs.get(key)]
        if not selected:
            selected = ["ex", "ey", "ez", "hx", "hy", "hz"]

        ncols = 3
        nrows = int(np.ceil(len(selected) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), layout="compressed")
        axes = np.array(axes).reshape(-1)

        last_image = None
        for i, field_name in enumerate(selected):
            field_data, title = fields[field_name]
            ax = axes[i]
            last_image = self._plot_field_subplot(ax, field_name, field_data, title)

        for j in range(len(selected), len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(rf"Mode {mode + 1}: $n_{{eff}}$ = {self.neff[mode]:.4g}", fontsize=16)
        if last_image is not None:
            fig.colorbar(last_image, ax=axes[:len(selected)], location="right", shrink=1, pad=0.02)
        plt.show()

    def visualize_with_gui(self):
        """Visualize field components with a dropdown menu for mode selection."""
        import tkinter as tk
        from tkinter import ttk

        from matplotlib import pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        if self.eigenvalues is None:
            raise RuntimeError("solve() must be called before visualize_with_gui().")

        root = tk.Tk()
        root.title("Field Visualization")

        plot_frame = tk.Frame(root)
        plot_frame.grid(row=0, column=0, sticky="nsew")
        controls_frame = tk.Frame(root)
        controls_frame.grid(row=1, column=0, sticky="ew", pady=10)

        fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        colorbar = [None]

        def plot_mode(selected_mode):
            mode = int(selected_mode) - 1
            fields = self._component_fields_for_mode(mode)
            order = ("ex", "ey", "ez", "hx", "hy", "hz")
            e_norm = max(np.max(np.abs(fields[name][0])) for name in order[:3])
            h_norm = max(np.max(np.abs(fields[name][0])) for name in order[3:])

            for ax in axes.flat:
                ax.clear()
            if colorbar[0] is not None:
                colorbar[0].remove()
                colorbar[0] = None

            for i, ax in enumerate(axes.flat):
                field_name = order[i]
                field_data, title = fields[field_name]
                norm = e_norm if i < 3 else h_norm
                self._plot_field_subplot(ax, field_name, field_data, title, norm=norm)

            fig.subplots_adjust(right=0.86)
            cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
            sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=1))
            colorbar[0] = fig.colorbar(sm, cax=cbar_ax)
            colorbar[0].set_label("Normalized Magnitude")
            fig.suptitle(rf"Mode {mode + 1}: $n_{{eff}}$ = {self.neff[mode]:.4g}", fontsize=16)
            canvas.draw()

        mode_var = tk.StringVar(value="1")
        mode_menu = ttk.Combobox(controls_frame, textvariable=mode_var, values=list(range(1, self.num_modes + 1)))
        mode_menu.grid(row=0, column=0, padx=10, sticky="w")
        mode_menu.bind("<<ComboboxSelected>>", lambda event: plot_mode(mode_var.get()))
        tk.Button(controls_frame, text="Quit", command=root.destroy).grid(row=0, column=1, padx=10, sticky="e")

        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        controls_frame.columnconfigure(1, weight=1)
        plot_mode(mode_var.get())
        root.mainloop()
