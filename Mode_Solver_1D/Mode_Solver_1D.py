import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigs

if __package__:
    from metal_surface_impedance import (
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

    boundary_module_name = "_fdfd_mode_solver_1d_surface_impedance_boundary"
    boundary_module = sys.modules.get(boundary_module_name)
    if boundary_module is None:
        boundary_path = module_directory / "surface_impedance_boundary.py"
        boundary_spec = importlib.util.spec_from_file_location(
            boundary_module_name,
            boundary_path,
        )
        if boundary_spec is None or boundary_spec.loader is None:
            raise ImportError(f"Cannot load the 1D boundary compiler from {boundary_path}.")
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


class ModeSolver1D:
    """1D FDFD mode solver on a true staggered Yee grid."""

    def __init__(self, frequency, x_range, Nx, num_modes, guess=None):
        self.frequency = frequency
        self.x_range = x_range
        self.Nx = int(Nx)
        if self.Nx <= 0:
            raise ValueError("Nx must be positive.")

        self.dx = x_range / self.Nx
        self.epsilon0 = 8.854187817e-12
        self.mu0 = 4e-7 * np.pi
        self.c = 1 / np.sqrt(self.epsilon0 * self.mu0)
        self.k_0 = 2 * np.pi * frequency / self.c
        self.dx_normalized = self.k_0 * self.dx

        self.shape_cell = (self.Nx,)
        self.shape_node = (self.Nx + 1,)

        self.cell_eps_r_xx = np.ones(self.shape_cell, dtype=complex)
        self.cell_eps_r_yy = np.ones(self.shape_cell, dtype=complex)
        self.cell_eps_r_zz = np.ones(self.shape_cell, dtype=complex)
        self.cell_mu_r_xx = np.ones(self.shape_cell, dtype=complex)
        self.cell_mu_r_yy = np.ones(self.shape_cell, dtype=complex)
        self.cell_mu_r_zz = np.ones(self.shape_cell, dtype=complex)
        self.material_no_average_mask = np.zeros(self.shape_cell, dtype=bool)

        self.eps_r_xx = np.ones(self.shape_cell, dtype=complex)
        self.eps_r_yy = np.ones(self.shape_node, dtype=complex)
        self.eps_r_zz = np.ones(self.shape_node, dtype=complex)
        self.mu_r_xx = np.ones(self.shape_node, dtype=complex)
        self.mu_r_yy = np.ones(self.shape_cell, dtype=complex)
        self.mu_r_zz = np.ones(self.shape_cell, dtype=complex)

        self.pec_xx_mask = np.zeros(self.shape_cell, dtype=bool)
        self.pec_yy_mask = np.zeros(self.shape_node, dtype=bool)
        self.pec_zz_mask = np.zeros(self.shape_node, dtype=bool)
        self.pmc_xx_mask = np.zeros(self.shape_node, dtype=bool)
        self.pmc_yy_mask = np.zeros(self.shape_cell, dtype=bool)
        self.pmc_zz_mask = np.zeros(self.shape_cell, dtype=bool)
        self._pec_cell_mask = np.zeros(self.shape_cell, dtype=bool)
        self._pmc_cell_mask = np.zeros(self.shape_cell, dtype=bool)
        self._pml_cell_mask = np.zeros(self.shape_cell, dtype=bool)
        self._pec_regions = []
        self._pmc_regions = []
        self._surface_impedance_owner = np.full(
            self.shape_cell,
            -1,
            dtype=np.int32,
        )
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
            for arr in (
                self.cell_eps_r_xx,
                self.cell_eps_r_yy,
                self.cell_eps_r_zz,
                self.cell_mu_r_xx,
                self.cell_mu_r_yy,
                self.cell_mu_r_zz,
            )
        )

    def _resolve_eigs_guess(self, sigma):
        if sigma is not None:
            return sigma
        if self._auto_guess:
            self.guess = self._default_guess()
        return self.guess

    def _invalidate_solution(self):
        self.eigenvalues_TE = None
        self.eigenvalues_TM = None
        self.eigenvectors_TE = None
        self.eigenvectors_TM = None
        self.neff_TE = None
        self.neff_TM = None
        self.propagation_constant_TE = None
        self.propagation_constant_TM = None
        self.attenuation_constant_TE = None
        self.attenuation_constant_TM = None
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

    def _bound_to_index(self, value):
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            return int(round(float(value) / self.dx))
        raise ValueError("Region bounds must be int grid indices or float physical positions in metres.")

    def _coordinate_to_length(self, value):
        if isinstance(value, (int, np.integer)):
            return int(value) * self.dx
        if isinstance(value, (float, np.floating)):
            return float(value)
        raise ValueError("Coordinates must be int grid indices or float physical positions in metres.")

    def _range_to_lengths(self, x_range):
        try:
            x0, x1 = x_range
        except (TypeError, ValueError):
            raise ValueError("x_range must be a (min, max) pair.")
        x0 = self._coordinate_to_length(x0)
        x1 = self._coordinate_to_length(x1)
        if x1 <= x0:
            raise ValueError("x_range must satisfy max > min.")
        if x0 < 0 or x1 > self.x_range:
            raise ValueError("Region is out of bounds of the simulation grid.")
        return x0, x1

    @staticmethod
    def _validate_subpixels(subpixels):
        subpixels = int(subpixels)
        if subpixels <= 0:
            raise ValueError("subpixels must be positive.")
        return subpixels

    def _region_slice(self, x_range):
        try:
            x0 = self._bound_to_index(x_range[0])
            x1 = self._bound_to_index(x_range[1])
        except (TypeError, IndexError):
            raise ValueError("x_range must be a (min, max) pair.")

        if not x1 > x0:
            raise ValueError("x_range must satisfy max > min.")
        if not 0 <= x0 < x1 <= self.Nx:
            raise ValueError("Region is out of bounds of the simulation grid.")
        return slice(x0, x1)

    def _cell_material_array(self, prefix, component):
        if prefix == "eps":
            return {"xx": self.cell_eps_r_xx, "yy": self.cell_eps_r_yy, "zz": self.cell_eps_r_zz}[component]
        if prefix == "mu":
            return {"xx": self.cell_mu_r_xx, "yy": self.cell_mu_r_yy, "zz": self.cell_mu_r_zz}[component]
        raise ValueError(f"Unknown {prefix} component {component!r}.")

    def _component_mask(self, prefix, component):
        if prefix == "pec":
            return {"xx": self.pec_xx_mask, "yy": self.pec_yy_mask, "zz": self.pec_zz_mask}[component]
        if prefix == "pmc":
            return {"xx": self.pmc_xx_mask, "yy": self.pmc_yy_mask, "zz": self.pmc_zz_mask}[component]
        raise ValueError(f"Unknown {prefix} component {component!r}.")

    def _apply_fractional_material(self, epsilon, mu, fraction, sl_x):
        epsilon = self._normalise_three("epsilon", epsilon)
        mu = self._normalise_three("mu", mu)
        fraction = np.asarray(fraction, dtype=float)
        if fraction.shape != self.cell_eps_r_xx[sl_x].shape:
            raise ValueError("fraction shape does not match target cell region.")

        covered = fraction > 0.0
        if not np.any(covered):
            return

        for component, value in zip(("xx", "yy", "zz"), epsilon):
            target = self._cell_material_array("eps", component)[sl_x]
            target[covered] = target[covered] * (1.0 - fraction[covered]) + value * fraction[covered]
        for component, value in zip(("xx", "yy", "zz"), mu):
            target = self._cell_material_array("mu", component)[sl_x]
            target[covered] = target[covered] * (1.0 - fraction[covered]) + value * fraction[covered]

        local_no_average = self.material_no_average_mask[sl_x]
        local_no_average[covered] = False
        self.update_component_materials()
        self._invalidate_solution()

    def add_layer(self, epsilon, mu, x_range, *, subpixels=100):
        """Add a subpixel-smoothed isotropic or diagonal-anisotropic material layer."""
        x_min, x_max = self._range_to_lengths(x_range)
        subpixels = self._validate_subpixels(subpixels)
        x0 = max(0, int(np.floor(x_min / self.dx)))
        x1 = min(self.Nx, int(np.ceil(x_max / self.dx)))
        if x0 >= x1:
            return

        indices = np.arange(x0, x1, dtype=float)
        offsets = (np.arange(subpixels, dtype=float) + 0.5) / subpixels
        samples = (indices[:, None] + offsets[None, :]) * self.dx
        fraction = ((samples >= x_min) & (samples <= x_max)).mean(axis=1)
        self._apply_fractional_material(epsilon, mu, fraction, slice(x0, x1))

    def add_pec(self, x_range, components=None):
        """Add a PEC cell region and expand it onto surrounding electric components."""
        sl_x = self._region_slice(x_range)
        selected = (
            ("xx", "yy", "zz")
            if components is None
            else self._validate_components(components)
        )
        cell_mask = np.zeros(self.shape_cell, dtype=bool)
        cell_mask[sl_x] = True
        if np.any(cell_mask & self.impedance_surface_mask):
            raise ValueError("Surface-impedance overlap with PEC region.")

        previous_masks = tuple(
            values.copy()
            for values in (
                self.pec_xx_mask,
                self.pec_yy_mask,
                self.pec_zz_mask,
            )
        )
        previous_cell_mask = self._pec_cell_mask.copy()
        previous_region_count = len(self._pec_regions)
        previous_compiled_boundary = self._compiled_impedance_boundary
        self._pec_regions.append(sl_x)
        self._pec_cell_mask |= cell_mask
        xx_mask, yy_mask, zz_mask = self.component_masks_from_cell_mask(cell_mask, field="electric")
        try:
            for comp, mask in (("xx", xx_mask), ("yy", yy_mask), ("zz", zz_mask)):
                if comp in selected:
                    self._component_mask("pec", comp)[:] |= mask
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

    def add_pmc(self, x_range, components=None):
        """Add a PMC cell region and expand it onto surrounding magnetic components."""
        sl_x = self._region_slice(x_range)
        selected = (
            ("xx", "yy", "zz")
            if components is None
            else self._validate_components(components)
        )
        cell_mask = np.zeros(self.shape_cell, dtype=bool)
        cell_mask[sl_x] = True
        if np.any(cell_mask & self.impedance_surface_mask):
            raise ValueError("Surface-impedance overlap with PMC region.")

        previous_masks = tuple(
            values.copy()
            for values in (
                self.pmc_xx_mask,
                self.pmc_yy_mask,
                self.pmc_zz_mask,
            )
        )
        previous_cell_mask = self._pmc_cell_mask.copy()
        previous_region_count = len(self._pmc_regions)
        previous_compiled_boundary = self._compiled_impedance_boundary
        self._pmc_regions.append(sl_x)
        self._pmc_cell_mask |= cell_mask
        xx_mask, yy_mask, zz_mask = self.component_masks_from_cell_mask(cell_mask, field="magnetic")
        try:
            for comp, mask in (("xx", xx_mask), ("yy", yy_mask), ("zz", zz_mask)):
                if comp in selected:
                    self._component_mask("pmc", comp)[:] |= mask
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
        mask = np.asarray(cell_mask, dtype=bool)
        if mask.shape != self.shape_cell:
            raise ValueError(f"cell_mask must have shape {self.shape_cell}.")
        ii = np.nonzero(mask)[0]

        if field == "electric":
            xx_mask = mask.copy()
            yy_mask = np.zeros(self.shape_node, dtype=bool)
            zz_mask = np.zeros(self.shape_node, dtype=bool)
            yy_mask[ii] = True
            yy_mask[ii + 1] = True
            zz_mask[ii] = True
            zz_mask[ii + 1] = True
            return xx_mask, yy_mask, zz_mask

        if field == "magnetic":
            xx_mask = np.zeros(self.shape_node, dtype=bool)
            yy_mask = mask.copy()
            zz_mask = mask.copy()
            xx_mask[ii] = True
            xx_mask[ii + 1] = True
            return xx_mask, yy_mask, zz_mask

        raise ValueError("field must be 'electric' or 'magnetic'.")

    def add_pml(self, pml_width=50, n=3, sigma_max=25, direction="all"):
        """Add a simple uniaxial PML by stretching cell-centered epsilon and mu tensors."""
        pml_width = int(pml_width)
        if pml_width <= 0:
            raise ValueError("pml_width must be positive.")
        if direction not in ("x-", "x+", "x", "all"):
            raise ValueError("direction must be one of 'x-', 'x+', 'x', or 'all'.")

        sigma_x = np.zeros(self.Nx, dtype=float)
        if direction in ("x-", "x", "all"):
            for i in range(min(pml_width, self.Nx)):
                sigma_x[i] = sigma_max * ((pml_width - i) / pml_width) ** n
        if direction in ("x+", "x", "all"):
            for i in range(min(pml_width, self.Nx)):
                sigma_x[-i - 1] = sigma_max * ((pml_width - i) / pml_width) ** n

        pml_cells = sigma_x != 0.0
        prospective_pml_cells = self._pml_cell_mask | pml_cells
        validate_impedance_pml_separation(
            self.impedance_surface_mask,
            prospective_pml_cells,
        )

        omega = 2 * np.pi * self.frequency
        Sx = 1.0 + 1j * sigma_x / (self.epsilon0 * omega)

        self.cell_eps_r_xx *= 1 / Sx
        self.cell_eps_r_yy *= Sx
        self.cell_eps_r_zz *= Sx
        self.cell_mu_r_xx *= 1 / Sx
        self.cell_mu_r_yy *= Sx
        self.cell_mu_r_zz *= Sx
        self._pml_cell_mask |= pml_cells
        self.update_component_materials()
        self._invalidate_solution()

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
            impedance = good_conductor_surface_impedance(
                canonical,
                self.frequency,
            )
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
    ):
        """Mark opaque cells whose exposed interfaces obey a scalar SIBC."""
        definition = self._surface_impedance_definition(Zs, preset)
        sl_x = self._region_slice(x_range)
        region_mask = np.zeros(self.shape_cell, dtype=bool)
        region_mask[sl_x] = True

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
                for index, existing in enumerate(
                    self._surface_impedance_definitions
                )
                if existing.key == definition.key
            ),
            None,
        )
        occupied_owners = np.unique(
            self._surface_impedance_owner[region_mask]
        )
        occupied_owners = occupied_owners[occupied_owners >= 0]
        if occupied_owners.size:
            if definition_index is None or np.any(
                    occupied_owners != definition_index
            ):
                raise ValueError(
                    "Surface-impedance region has an impedance overlap with "
                    "a different definition."
                )

        prospective_mask = self.impedance_surface_mask | region_mask
        if np.all(prospective_mask):
            raise ValueError(
                "Surface-impedance geometry leaves no retained field cells."
            )
        validate_impedance_pml_separation(
            prospective_mask,
            self._pml_cell_mask,
        )

        previous_owner = self._surface_impedance_owner.copy()
        previous_definition_count = len(self._surface_impedance_definitions)
        previous_region_count = len(self._surface_impedance_regions)
        previous_compiled_boundary = self._compiled_impedance_boundary
        try:
            if definition_index is None:
                definition_index = len(self._surface_impedance_definitions)
                self._surface_impedance_definitions.append(definition)
            unowned = region_mask & (self._surface_impedance_owner < 0)
            self._surface_impedance_owner[unowned] = definition_index
            self._surface_impedance_regions.append((sl_x, definition.label))
            self._effective_materials_and_masks()
        except Exception:
            self._surface_impedance_owner[:] = previous_owner
            del self._surface_impedance_definitions[previous_definition_count:]
            del self._surface_impedance_regions[previous_region_count:]
            self._compiled_impedance_boundary = previous_compiled_boundary
            raise
        self._invalidate_solution()

    @staticmethod
    def _average_to_node(values, no_average_mask=None):
        out = np.zeros(values.size + 1, dtype=complex)
        counts = np.zeros(values.size + 1, dtype=float)
        out[:-1] += values
        counts[:-1] += 1
        out[1:] += values
        counts[1:] += 1
        out = out / counts
        if no_average_mask is not None:
            ii = np.nonzero(no_average_mask)[0]
            out[ii] = values[ii]
            out[ii + 1] = values[ii]
        return out

    def _material_on_fields(self, eps_r_xx, eps_r_yy, eps_r_zz, mu_r_xx, mu_r_yy, mu_r_zz, no_average_mask):
        return {
            "eps_xx": eps_r_xx.copy(),
            "eps_yy": self._average_to_node(eps_r_yy, no_average_mask),
            "eps_zz": self._average_to_node(eps_r_zz, no_average_mask),
            "mu_xx": self._average_to_node(mu_r_xx, no_average_mask),
            "mu_yy": mu_r_yy.copy(),
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

    def _yeeder1d(self):
        """Generate rectangular derivative matrices between node and cell locations."""
        rows = []
        cols = []
        data = []
        for i in range(self.Nx):
            rows.extend((i, i))
            cols.extend((i + 1, i))
            data.extend((1.0 / self.dx_normalized, -1.0 / self.dx_normalized))
        D_e_to_h = coo_matrix((data, (rows, cols)), shape=(self.Nx, self.Nx + 1)).tocsr()
        D_h_to_e = -D_e_to_h.conj().T
        D_h_to_e = self._apply_impedance_ampere_rows(D_h_to_e)
        self.DEX = D_e_to_h
        self.DHX = D_h_to_e
        return D_e_to_h, D_h_to_e

    def _apply_impedance_ampere_rows(self, derivative):
        """Replace full-cell Ampere differences with clipped half-cell rows."""
        boundary = self._compiled_impedance_boundary
        if boundary is None or not boundary.rows:
            return derivative

        editable = derivative.tolil(copy=True)
        occupied_rows = set()
        for row in boundary.rows:
            if row.electric_index in occupied_rows:
                raise ValueError(
                    "Duplicate surface-impedance Ampere row at node "
                    f"{row.electric_index}."
                )
            occupied_rows.add(row.electric_index)
            editable.rows[row.electric_index] = [row.retained_cell_index]
            editable.data[row.electric_index] = [
                row.magnetic_coefficient / self.k_0
            ]
        return editable.tocsr()

    @staticmethod
    def _apply_transverse_cross_constraints(
            pec_xx_mask,
            pec_yy_mask,
            pmc_xx_mask,
            pmc_yy_mask,
    ):
        """Close the collocated Ex/Hy and Ey/Hx constraint pairs."""
        ex_hy_mask = pec_xx_mask | pmc_yy_mask
        ey_hx_mask = pec_yy_mask | pmc_xx_mask
        pec_xx_mask[:] = ex_hy_mask
        pmc_yy_mask[:] = ex_hy_mask
        pec_yy_mask[:] = ey_hx_mask
        pmc_xx_mask[:] = ey_hx_mask

    @staticmethod
    def _validate_impedance_row_conflicts(
            boundary,
            pec_yy_mask,
            pec_zz_mask,
            pmc_yy_mask,
            pmc_zz_mask,
    ):
        for row in boundary.rows:
            node = row.electric_index
            cell = row.retained_cell_index
            if pec_yy_mask[node] or pec_zz_mask[node]:
                raise ValueError(
                    "Surface-impedance boundary conflicts with a PEC/PMC "
                    f"constraint at electric node {node}."
                )
            if pmc_yy_mask[cell] or pmc_zz_mask[cell]:
                raise ValueError(
                    "Surface-impedance boundary has a PEC/PMC constraint on "
                    f"magnetic cell {cell}."
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
                raise ValueError(
                    "Surface-impedance cells overlap a PEC material region."
                )
            if np.any(opaque_cells & magnetic_bad):
                raise ValueError(
                    "Surface-impedance cells overlap a PMC material region."
                )

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
                cell_eps_r_yy=eps_r_yy,
                cell_eps_r_zz=eps_r_zz,
                dx=self.dx,
                frequency=self.frequency,
                epsilon0=self.epsilon0,
                pml_cells=self._pml_cell_mask,
            )
            self._compiled_impedance_boundary = boundary
            self._validate_impedance_row_conflicts(
                boundary,
                pec_yy_mask,
                pec_zz_mask,
                pmc_yy_mask,
                pmc_zz_mask,
            )
            for row in boundary.rows:
                node = row.electric_index
                materials["eps_yy"][node] = row.relative_permittivity_yy
                materials["eps_zz"][node] = row.relative_permittivity_zz
                materials["mu_xx"][node] = mu_r_xx[
                    row.retained_cell_index
                ]

            electric_constraints = (
                pec_xx_mask,
                pec_yy_mask,
                pec_zz_mask,
            )
            magnetic_constraints = (
                pmc_xx_mask,
                pmc_yy_mask,
                pmc_zz_mask,
            )
            for constraint, retained in zip(
                    electric_constraints,
                    boundary.electric_retained,
            ):
                constraint[:] |= ~retained
            for constraint, retained in zip(
                    magnetic_constraints,
                    boundary.magnetic_retained,
            ):
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

    @staticmethod
    def _diag(values):
        return diags(np.asarray(values).ravel(), format="csr")

    def _inverse_diag_on_free(self, values, constrained_mask):
        inverse = np.zeros_like(values, dtype=complex)
        inverse[~constrained_mask] = 1.0 / values[~constrained_mask]
        return diags(inverse, format="csr")

    def _solve_reduced(self, Omega, free_mask, full_size, sigma):
        Omega = Omega[free_mask, :][:, free_mask]
        if Omega.shape[0] <= self.num_modes:
            raise ValueError(f"Not enough unconstrained DOFs ({Omega.shape[0]}) to solve {self.num_modes} modes.")
        eigenvalues, eigenvectors_reduced = eigs(Omega, k=self.num_modes, sigma=sigma)
        order = np.argsort(np.real(eigenvalues))
        eigenvalues = eigenvalues[order]
        eigenvectors_reduced = eigenvectors_reduced[:, order]
        eigenvectors = np.zeros((full_size, self.num_modes), dtype=complex)
        eigenvectors[free_mask, :] = eigenvectors_reduced
        return eigenvalues, eigenvectors

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
            te_phase = self._most_real_phase(self.Ey[:, mode], self.Hx[:, mode], self.Hz[:, mode])
            tm_phase = self._most_real_phase(self.Hy[:, mode], self.Ex[:, mode], self.Ez[:, mode])
            self.Ey[:, mode] *= te_phase
            self.Hx[:, mode] *= te_phase
            self.Hz[:, mode] *= te_phase
            self.eigenvectors_TE[:, mode] *= te_phase
            self.Hy[:, mode] *= tm_phase
            self.Ex[:, mode] *= tm_phase
            self.Ez[:, mode] *= tm_phase
            self.eigenvectors_TM[:, mode] *= tm_phase

    def solve(self, sigma=None):
        """Solve TE and TM slab modes and recover staggered field components."""
        sigma = self._resolve_eigs_guess(sigma)
        materials, pec_xx_mask, pec_yy_mask, pec_zz_mask, pmc_xx_mask, pmc_yy_mask, pmc_zz_mask = (
            self._effective_materials_and_masks()
        )

        D_e_to_h, D_h_to_e = self._yeeder1d()
        eps_xx_diag = self._diag(materials["eps_xx"])
        eps_yy_diag = self._diag(materials["eps_yy"])
        mu_xx_diag = self._diag(materials["mu_xx"])
        mu_yy_diag = self._diag(materials["mu_yy"])
        eps_zz_inv = self._inverse_diag_on_free(materials["eps_zz"], pec_zz_mask)
        mu_zz_inv = self._inverse_diag_on_free(materials["mu_zz"], pmc_zz_mask)

        Omega_TE = -mu_xx_diag @ (D_h_to_e @ mu_zz_inv @ D_e_to_h + eps_yy_diag)
        Omega_TM = -eps_xx_diag @ (D_e_to_h @ eps_zz_inv @ D_h_to_e + mu_yy_diag)

        self.eigenvalues_TE, self.eigenvectors_TE = self._solve_reduced(
            Omega_TE, ~pec_yy_mask, self.Nx + 1, sigma
        )
        self.eigenvalues_TM, self.eigenvectors_TM = self._solve_reduced(
            Omega_TM, ~pmc_yy_mask, self.Nx, sigma
        )

        self.neff_TE = self._passive_positive_neff(-self.eigenvalues_TE)
        self.neff_TM = self._passive_positive_neff(-self.eigenvalues_TM)
        self.propagation_constant_TE = np.real(self.neff_TE)
        self.propagation_constant_TM = np.real(self.neff_TM)
        self.attenuation_constant_TE = -np.imag(self.neff_TE)
        self.attenuation_constant_TM = -np.imag(self.neff_TM)

        self.Ey = np.asarray(self.eigenvectors_TE, dtype=complex)
        self.Hy = np.asarray(self.eigenvectors_TM, dtype=complex)
        self.Hx = np.zeros_like(self.Ey)
        self.Hz = np.asarray(mu_zz_inv @ (D_e_to_h @ self.Ey), dtype=complex)
        self.Ex = np.zeros_like(self.Hy)
        self.Ez = np.asarray(eps_zz_inv @ (D_h_to_e @ self.Hy), dtype=complex)

        for mode in range(self.num_modes):
            self.Hx[:, mode] = self.neff_TE[mode] * (1.0 / materials["mu_xx"]) * self.Ey[:, mode]
            self.Ex[:, mode] = self.neff_TM[mode] * (1.0 / materials["eps_xx"]) * self.Hy[:, mode]

        self._zero_constrained_fields(
            pec_xx_mask,
            pec_yy_mask,
            pec_zz_mask,
            pmc_xx_mask,
            pmc_yy_mask,
            pmc_zz_mask,
        )
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
            if np.any(np.abs(np.imag(values[finite])) > 1e-14):
                return True
        return any(
            definition.impedance.real > 0
            for definition in self._surface_impedance_definitions
        )

    def _passive_positive_neff(self, neff_squared):
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

    def _field_x(self, field_name):
        if field_name in ("Ey", "Ez", "Hx"):
            return np.linspace(0, self.x_range * 1e3, self.Nx + 1)
        return (np.arange(self.Nx) + 0.5) * self.dx * 1e3

    def _component_fields_for_mode(self, mode):
        return {
            "ey": (self.Ey[:, mode], "Ey", "TE"),
            "hx": (self.Hx[:, mode], "Hx", "TE"),
            "hz": (self.Hz[:, mode], "Hz", "TE"),
            "hy": (self.Hy[:, mode], "Hy", "TM"),
            "ex": (self.Ex[:, mode], "Ex", "TM"),
            "ez": (self.Ez[:, mode], "Ez", "TM"),
        }

    def _field_to_cells(self, name, data):
        if name in ("ey", "ez", "hx"):
            return 0.5 * (data[:-1] + data[1:])
        return data

    def _material_background_for_field(self, name):
        if name == "ex":
            return self.cell_eps_r_xx
        if name == "ey":
            return self.cell_eps_r_yy
        if name == "ez":
            return self.cell_eps_r_zz
        if name == "hx":
            return self.cell_mu_r_xx
        if name == "hy":
            return self.cell_mu_r_yy
        if name == "hz":
            return self.cell_mu_r_zz
        if name == "eabs":
            return (self.cell_eps_r_xx + self.cell_eps_r_yy + self.cell_eps_r_zz) / 3.0
        if name == "habs":
            return (self.cell_mu_r_xx + self.cell_mu_r_yy + self.cell_mu_r_zz) / 3.0
        raise ValueError(f"Unknown field name {name!r}.")

    def _add_layer_background(self, ax, field_name):
        material = np.abs(self._material_background_for_field(field_name))[None, :]
        ax.imshow(
            material,
            cmap="inferno",
            origin="lower",
            aspect="auto",
            extent=[0, self.x_range * 1e3, -1.0, 1.0],
            alpha=0.5,
            zorder=0,
        )
        for sl_x in self._pec_regions:
            ax.axvspan(sl_x.start * self.dx * 1e3, sl_x.stop * self.dx * 1e3, color="yellow", alpha=0.5, zorder=1)
        for sl_x in self._pmc_regions:
            ax.axvspan(sl_x.start * self.dx * 1e3, sl_x.stop * self.dx * 1e3, color="blue", alpha=0.5, zorder=1)
        for sl_x, _ in self._surface_impedance_regions:
            ax.axvspan(
                sl_x.start * self.dx * 1e3,
                sl_x.stop * self.dx * 1e3,
                color="magenta",
                alpha=0.45,
                zorder=1,
            )
        ax.set_xlim(0, self.x_range * 1e3)
        ax.set_ylim(-1.0, 1.0)

    @staticmethod
    def _show_layer_background(field_name):
        return field_name not in ("hx", "hz", "hy")

    def _plot_field_profile(self, ax, field_name, field_data, title, pol, norm=None):
        if norm is None:
            norm = np.max(np.abs(field_data))
        field = field_data / norm if norm > 0 else field_data
        x = (np.arange(self.Nx) + 0.5) * self.dx * 1e3 if field_name in ("eabs", "habs") else self._field_x(title)
        ax.set_xlim(0, self.x_range * 1e3)
        ax.set_ylim(-1.0, 1.0)
        if self._show_layer_background(field_name):
            self._add_layer_background(ax, field_name)
        ax.plot(x, np.real(field), label=f"Re({title})", zorder=3)
        ax.plot(x, np.abs(field), "--", label=f"|{title}|", zorder=3)
        ax.set_ylim(-1.0, 1.0)
        ax.set_title(f"{pol}: {title}")
        ax.set_xlabel("x (mm)")
        ax.grid(True, zorder=4)
        ax.legend(loc="best", fontsize=8)

    def _field_group_norms(self, fields):
        e_abs = np.sqrt(sum(np.abs(self._field_to_cells(key, fields[key][0])) ** 2 for key in ("ey", "ex", "ez")))
        h_abs = np.sqrt(sum(np.abs(self._field_to_cells(key, fields[key][0])) ** 2 for key in ("hx", "hy", "hz")))
        return np.max(e_abs), np.max(h_abs), e_abs, h_abs

    @staticmethod
    def _norm_for_field(field_name, e_norm, h_norm):
        return e_norm if field_name in ("ex", "ey", "ez", "eabs") else h_norm

    def visualize(self, mode=1, **kwargs):
        """Visualize selected field components for a given one-based mode index."""
        if self.neff_TE is None:
            raise RuntimeError("solve() must be called before visualize().")
        mode -= 1
        if not (0 <= mode < self.num_modes):
            raise ValueError("mode is out of range.")

        import matplotlib.pyplot as plt

        fields = self._component_fields_for_mode(mode)
        e_norm, h_norm, e_abs, h_abs = self._field_group_norms(fields)
        fields["eabs"] = (e_abs, "|E| cell-centered", "E")
        fields["habs"] = (h_abs, "|H| cell-centered", "H")

        selected = [key for key in fields if kwargs.get(key)]
        if not selected:
            selected = ["ey", "hx", "hz", "hy", "ex", "ez"]

        ncols = min(3, len(selected))
        nrows = int(np.ceil(len(selected) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), layout="compressed")
        axes = np.array(axes).reshape(-1)

        for i, field_name in enumerate(selected):
            field_data, title, pol = fields[field_name]
            ax = axes[i]
            norm = self._norm_for_field(field_name, e_norm, h_norm)
            self._plot_field_profile(ax, field_name, field_data, title, pol, norm=norm)

        for j in range(len(selected), len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(
            rf"Mode {mode + 1}: TE $n_{{eff}}$ = {self.neff_TE[mode]:.4g}, "
            rf"TM $n_{{eff}}$ = {self.neff_TM[mode]:.4g}",
            fontsize=14,
        )
        plt.show()

    def visualize_with_gui(self):
        """Launch an interactive Tk GUI to inspect mode profiles."""
        if self.neff_TE is None:
            raise RuntimeError("solve() must be called before visualize_with_gui().")

        import tkinter as tk
        from tkinter import ttk

        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        root = tk.Tk()
        root.title("FDFD 1D Mode Visualizer")

        fig, axes = plt.subplots(2, 3, figsize=(12, 10), dpi=100)
        plot_frame = tk.Frame(root)
        plot_frame.grid(row=0, column=0, sticky="nsew")
        controls_frame = tk.Frame(root)
        controls_frame.grid(row=1, column=0, sticky="ew", pady=10)

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        mode_var = tk.IntVar(value=1)
        ttk.Label(controls_frame, text="Select mode:").grid(row=0, column=0, padx=10, sticky="w")
        mode_menu = ttk.Combobox(
            controls_frame,
            textvariable=mode_var,
            values=list(range(1, self.num_modes + 1)),
            state="readonly",
            width=5,
        )
        mode_menu.grid(row=0, column=1, padx=10, sticky="w")
        tk.Button(controls_frame, text="Quit", command=root.destroy).grid(row=0, column=2, padx=10, sticky="e")

        def update_plots(event=None):
            mode = int(mode_var.get()) - 1
            fields = self._component_fields_for_mode(mode)
            e_norm, h_norm, _, _ = self._field_group_norms(fields)
            for ax in axes.flat:
                ax.clear()

            for ax, key in zip(axes.flat, ("ey", "hx", "hz", "hy", "ex", "ez")):
                data, title, pol = fields[key]
                norm = self._norm_for_field(key, e_norm, h_norm)
                self._plot_field_profile(ax, key, data, title, pol, norm=norm)

            fig.suptitle(
                rf"Mode {mode + 1}: TE $n_{{eff}}$ = {self.neff_TE[mode]:.4g}, "
                rf"TM $n_{{eff}}$ = {self.neff_TM[mode]:.4g}",
                fontsize=14,
            )
            canvas.draw_idle()

        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        controls_frame.columnconfigure(2, weight=1)
        mode_menu.bind("<<ComboboxSelected>>", update_plots)
        update_plots()
        root.mainloop()
