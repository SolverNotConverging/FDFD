from pathlib import Path

from Periodic_Solver_3D import PeriodicModeSolver3D

Nx = 24
Ny = 20
Nz = 16

x_range = 6e-3
y_range = 6e-3
z_range = 8e-3

solver = PeriodicModeSolver3D(Nx=Nx, Ny=Ny, Nz=Nz,
                                 x_range=x_range, y_range=y_range, z_range=z_range,
                                 freq=22e9, num_modes=2, tol=0.1)

# Leaky-wave antenna unit cell (slightly simplified)
solver.add_object(1e8, 1, slice(6, 18), slice(Ny - 8, Ny - 7), slice(0, 3))
solver.add_object(1e8, 1, slice(6, 18), slice(Ny - 8, Ny - 7), slice(5, 8))
solver.add_object(6, 1, slice(6, 18), slice(Ny - 7, Ny - 1), slice(0, Nz))
solver.add_object(1e8, 1, slice(0, Nx), slice(Ny - 1, Ny), slice(0, Nz))

solver.add_UPML(['+y'], width=6, max_loss=5, n=3)
solver.solve()
solver.visualize_with_gui()

output_dir = Path(__file__).resolve().parent / "example_outputs"
output_dir.mkdir(parents=True, exist_ok=True)
solver.save_results(output_dir / "modes_full.npz", include_eigenvectors=True)
