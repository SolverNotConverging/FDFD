"""2D TE/TM total-field/scattered-field kernel on an (x, y) Yee grid.

TE has Ez (on nodes) as the primary unknown with Hx, Hy derived; TM has Hz
(at cell centres) with Ex, Ey derived. The outer boundary is PEC, normally
hidden behind a PML. Kernel coordinates are relative to the domain centre.
Stored fields are SI (V/m and A/m) under ``exp(+i*omega*t)``.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg as spla
from scipy.special import hankel2

from ..core import yee
from ..core.constants import C0, ETA_0
from ..core.errors import SolverError
from ..core.grid import CellMaterials

AXES = ("x", "y")
PRIMARY = {"TE": "Ez", "TM": "Hz"}
DERIVED = {"TE": ("Hx", "Hy"), "TM": ("Ex", "Ey")}


class ScatteringKernel(CellMaterials):
    """``Nx`` by ``Ny`` cells over a domain of the given lengths."""

    def __init__(self, frequency, x_length, y_length, nx, ny):
        self.frequency = float(frequency)
        self.k0 = 2 * np.pi * self.frequency / C0
        self.Nx, self.Ny = int(nx), int(ny)
        self.x_length, self.y_length = float(x_length), float(y_length)
        self.dx, self.dy = self.x_length / self.Nx, self.y_length / self.Ny
        self._init_materials((self.Nx, self.Ny))
        nodes = (
            np.arange(self.Nx + 1) * self.dx - self.x_length / 2,
            np.arange(self.Ny + 1) * self.dy - self.y_length / 2,
        )
        centres = (nodes[0][:-1] + self.dx / 2, nodes[1][:-1] + self.dy / 2)
        self.coordinates = {}
        for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            shape = yee.staggered_shape((self.Nx, self.Ny), name[1].lower(), AXES, name[0])
            self.coordinates[name] = tuple(
                nodes[a] if n == len(nodes[a]) else centres[a] for a, n in enumerate(shape)
            )
        self.pec_cells = np.zeros((self.Nx, self.Ny), dtype=bool)
        self._systems = {}

    def shape(self, name):
        return tuple(len(axis) for axis in self.coordinates[name])

    def update_component_materials(self):
        self.close()

    def add_pec(self, cell_mask):
        """Constrain a metal volume and its boundary at the actual Yee samples."""
        self.pec_cells |= np.asarray(cell_mask, dtype=bool)
        self.close()

    def pec_field_masks(self):
        return {
            name: yee.footprint(self.pec_cells, name[1].lower(), AXES, name[0]) for name in self.coordinates
        }

    def _field_materials(self):
        eps, mu = self.cell_tensors()
        result = {}
        for index, component in enumerate("xyz"):
            result["E" + component] = yee.staggered_average(eps[index], component, AXES, "E")
            result["H" + component] = yee.staggered_average(mu[index], component, AXES, "H")
        return result

    def _derivatives(self, polarization):
        kdx, kdy = self.k0 * self.dx, self.k0 * self.dy
        if polarization == "TE":
            dex = yee.difference(self.shape("Ez"), self.shape("Hy"), 0, kdx)
            dey = yee.difference(self.shape("Ez"), self.shape("Hx"), 1, kdy)
        else:
            dex = yee.difference(self.shape("Ey"), self.shape("Hz"), 0, kdx)
            dey = yee.difference(self.shape("Ex"), self.shape("Hz"), 1, kdy)
        return dex, dey, -dex.T, -dey.T

    def _inverse_transverse(self, polarization):
        tensors = self._field_materials()
        if polarization == "TE":
            return 1 / tensors["Hx"], 1 / tensors["Hy"]
        inverse_ex, inverse_ey = 1 / tensors["Ex"], 1 / tensors["Ey"]
        # Tangential E is zero on the outer PEC walls; both end traces exist.
        inverse_ex[:, [0, -1]] = 0
        inverse_ey[[0, -1], :] = 0
        masks = self.pec_field_masks()
        inverse_ex[masks["Ex"]] = 0
        inverse_ey[masks["Ey"]] = 0
        return inverse_ex, inverse_ey

    def _system(self, polarization):
        dex, dey, dhx, dhy = self._derivatives(polarization)
        inverse_x, inverse_y = self._inverse_transverse(polarization)
        tensors = self._field_materials()
        primary = PRIMARY[polarization]
        if polarization == "TE":
            matrix = (
                dhx @ yee.diag(inverse_y) @ dex + dhy @ yee.diag(inverse_x) @ dey + yee.diag(tensors["Ez"])
            )
            free = np.ones(self.shape(primary), dtype=bool)
            free[[0, -1], :] = False
            free[:, [0, -1]] = False
        else:
            matrix = (
                dex @ yee.diag(inverse_y) @ dhx + dey @ yee.diag(inverse_x) @ dhy + yee.diag(tensors["Hz"])
            )
            free = np.ones(self.shape(primary), dtype=bool)
        free &= ~self.pec_field_masks()[primary]
        return matrix.tocsr(), free.ravel(order="F")

    def incident(self, polarization, *, kind, angle, location, amplitude):
        """Incident primary field: a plane wave at ``angle`` degrees from +x, or a line source."""
        x, y = np.meshgrid(*self.coordinates[PRIMARY[polarization]], indexing="ij")
        if kind == "plane_wave":
            theta = np.deg2rad(angle)
            field = np.exp(-1j * self.k0 * (np.cos(theta) * x + np.sin(theta) * y))
        else:
            radius = np.hypot(x - location[0], y - location[1])
            radius[radius == 0] = self.dx / 50
            # Outgoing cylindrical wave for exp(+i*omega*t).
            field = hankel2(0, self.k0 * radius)
            if polarization == "TM":
                field = field * (-1j / 4)
        return (amplitude * field).ravel(order="F")

    def total_field_masks(self, inset):
        """Diagonal masks equal to 1 in the scattered-field (outer) region of each lattice."""
        masks = {}
        for name, (x, y) in self.coordinates.items():
            xx, yy = np.meshgrid(x, y, indexing="ij")
            inside = (
                (xx >= -self.x_length / 2 + inset)
                & (xx <= self.x_length / 2 - inset)
                & (yy >= -self.y_length / 2 + inset)
                & (yy <= self.y_length / 2 - inset)
            )
            masks[name] = sp.diags((~inside).ravel(order="F").astype(float), format="csr")
        return masks

    def solve(self, polarization, source, masks, *, device="cpu"):
        """Solve the TF/SF system and return ``(fields, relative_residual)``."""
        key = (polarization, device)
        if key not in self._systems:
            matrix, free = self._system(polarization)
            reduced = matrix[free][:, free].tocsc()
            if device in ("cuda", "hybrid"):
                from ..core.cuda import CudaFactor

                factor = CudaFactor(reduced)
            else:
                factor = spla.factorized(reduced)
            self._systems[key] = matrix, free, reduced, factor
        matrix, free, reduced, factor = self._systems[key]
        primary = PRIMARY[polarization]
        q = masks[primary]
        rhs = (q @ matrix - matrix @ q) @ source
        values = np.zeros(source.shape, dtype=complex)
        values[free] = factor(rhs[free])
        residual = reduced @ values[free] - rhs[free]
        scale = np.linalg.norm(abs(reduced) @ abs(values[free])) + np.linalg.norm(rhs[free])
        relative = float(np.linalg.norm(residual) / max(scale, np.finfo(float).tiny))
        if not np.isfinite(values).all() or relative > 1e-8:
            raise SolverError("Scattering solve failed the original-system residual check.")
        return self._fields(polarization, values, source, masks), relative

    def _fields(self, polarization, values, source, masks):
        dex, dey, dhx, dhy = self._derivatives(polarization)
        inverse_x, inverse_y = self._inverse_transverse(polarization)
        q = masks[PRIMARY[polarization]]

        def derivative(operator, name):
            # The incident field enters only where the TF/SF masks differ.
            return operator @ values + (operator @ q - masks[name] @ operator) @ source

        if polarization == "TE":
            flat = {
                "Ez": values,
                "Hx": 1j * derivative(dey, "Hx") * inverse_x.ravel(order="F") / ETA_0,
                "Hy": -1j * derivative(dex, "Hy") * inverse_y.ravel(order="F") / ETA_0,
            }
        else:
            flat = {
                "Hz": values,
                "Ex": -1j * ETA_0 * derivative(dhy, "Ex") * inverse_x.ravel(order="F"),
                "Ey": 1j * ETA_0 * derivative(dhx, "Ey") * inverse_y.ravel(order="F"),
            }
        fields = {name: field.reshape(self.shape(name), order="F") for name, field in flat.items()}
        pec = self.pec_field_masks()
        for name, field in fields.items():
            field[pec[name]] = 0
        return fields

    def close(self):
        """Release cached factorizations (CUDA factors hold GPU memory)."""
        for *_, factor in self._systems.values():
            if hasattr(factor, "close"):
                factor.close()
        self._systems.clear()
