import matplotlib.pyplot as plt
import numpy as np


class ElectrostaticSolver:
    def __init__(self, mesh_size, dim=2):
        """
        Initialize the electrostatic solver.
        :param mesh_size: Tuple for mesh size (nx,) for 1D or (nx, ny) for 2D.
        :param dim: Dimension of the problem (1 or 2).
        """
        self.dim = dim
        self.mesh_size = mesh_size
        self.potential = np.zeros(mesh_size)
        self.fixed_mask = np.zeros(mesh_size, dtype=bool)  # True for fixed potential points
        self.permittivity = {
            "erxx": np.ones(mesh_size),
            "eryy": np.ones(mesh_size) if dim == 2 else None,
        }
        # Force zero-potential boundary condition
        if dim == 1:
            self.fixed_mask[0] = True
            self.fixed_mask[-1] = True

        if dim == 2:
            self.fixed_mask[0, :] = True
            self.fixed_mask[-1, :] = True
            self.fixed_mask[:, 0] = True
            self.fixed_mask[:, -1] = True

    def set_potential(self, region, potential_value):
        """
        Set a region with a fixed potential.
        :param region: Slice object (1D) or tuple of slice objects (2D).
        :param potential_value: Electric potential value for the region.
        """
        self.potential[region] = potential_value
        self.fixed_mask[region] = True

    def add_object(self, region, erxx=1.0, eryy=None):
        """
        Add an object with anisotropic permittivity.
        :param region: Slice object (1D) or tuple of slice objects (2D).
        :param erxx: Relative permittivity in the x-direction.
        :param eryy: Relative permittivity in the y-direction (only for 2D).
        """
        self.permittivity["erxx"][region] = erxx
        if self.dim == 2 and eryy is not None:
            self.permittivity["eryy"][region] = eryy
        elif self.dim == 1 and eryy is not None:
            raise ValueError("eryy is not applicable for 1D problems.")

    def solve(self, tol=1e-8, max_iter=100000):
        """
        Solve the electrostatic problem using iterative relaxation.
        :param tol: Convergence tolerance.
        :param max_iter: Maximum number of iterations.
        """
        for _ in range(max_iter):
            potential_new = self.potential.copy()
            if self.dim == 1:
                exx = self.permittivity["erxx"]
                potential_new[1:-1] = (exx[:-2] * self.potential[:-2] + exx[2:] * self.potential[2:]) / (
                        exx[:-2] + exx[2:])
            elif self.dim == 2:
                exx = self.permittivity["erxx"]
                eyy = self.permittivity["eryy"]
                potential_new[1:-1, 1:-1] = (
                        (exx[1:-1, :-2] * self.potential[1:-1, :-2] +
                         exx[1:-1, 2:] * self.potential[1:-1, 2:] +
                         eyy[:-2, 1:-1] * self.potential[:-2, 1:-1] +
                         eyy[2:, 1:-1] * self.potential[2:, 1:-1]) /
                        (exx[1:-1, :-2] + exx[1:-1, 2:] + eyy[:-2, 1:-1] + eyy[2:, 1:-1])
                )
            # Apply fixed potential constraints
            potential_new[self.fixed_mask] = self.potential[self.fixed_mask]

            # Check for convergence
            if np.max(np.abs(potential_new - self.potential)) < tol:
                self.potential = potential_new
                break
            self.potential = potential_new

    def compute_electric_field(self):
        """
        Compute the electric field from the potential.
        """
        if self.dim == 1:
            ex = np.gradient(-self.potential)
            return ex  # Electric field between points in 1D
        elif self.dim == 2:
            ex, ey = np.gradient(-self.potential)
            return ex, ey

    def visualize(self):
        """
        Visualize the results.
        """
        if self.dim == 1:
            # 1D plot
            x = np.arange(self.mesh_size[0])
            e_field = self.compute_electric_field()

            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

            # Plot potential
            ax1.plot(x, self.potential, label='Potential', color='blue')
            ax1.set_ylabel('Potential (V)')
            ax1.legend(loc='upper right')
            ax1.grid()

            # Plot electric field
            ax2.plot(x, e_field, label='Electric Field', color='red')
            ax2.set_xlabel('x')
            ax2.set_ylabel('Electric Field (V/m)')
            ax2.legend(loc='upper right')
            ax2.grid()
            plt.show()


        elif self.dim == 2:
            # 2D visualization
            ex, ey = self.compute_electric_field()
            x = np.arange(self.mesh_size[0])
            y = np.arange(self.mesh_size[1])
            X, Y = np.meshgrid(x, y, indexing='ij')

            # Compute the magnitude of the electric field
            magnitude = np.sqrt(ex ** 2 + ey ** 2)
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Plot potential
            c1 = axes[0].contourf(X, Y, self.potential, cmap='viridis', levels=50)
            fig.colorbar(c1, ax=axes[0], label='Potential (V)')
            axes[0].set_title('Electric Potential')
            axes[0].set_xlabel('x')
            axes[0].set_ylabel('y')

            cmap = plt.cm.plasma
            norm = plt.Normalize(vmin=np.min(magnitude), vmax=np.max(magnitude))
            c2 = axes[1].quiver(X, Y, ex, ey, magnitude, cmap=cmap, norm=norm, scale=1000)
            fig.colorbar(c2, ax=axes[1], label='|E| (V/m)')
            axes[1].set_title('Electric Field (colored by magnitude)')
            axes[1].set_xlabel('x')
            axes[1].set_ylabel('y')
            plt.tight_layout()
            plt.show()
