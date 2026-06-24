from pathlib import Path

import numpy as np

from Periodic_Solver_3D import PeriodicModeSolver3D

Nx = 30
Ny = 30
Nz = 32

x_range = 6e-3
y_range = 6e-3
z_range = 8e-3
freq = 22e9
k0 = 2 * np.pi * freq / 3e8
sigma_guess = 0

solver = PeriodicModeSolver3D(Nx=Nx, Ny=Ny, Nz=Nz,
                              x_range=x_range, y_range=y_range, z_range=z_range,
                              freq=freq, num_modes=2, sigma_guess=sigma_guess,
                              tol=0.1, ncv=30)

# Leaky-wave antenna unit cell (slightly simplified)
solver.add_pec((1.5e-3, 4.5e-3), (1.5e-3, 1.55e-3), (0, 1.5e-3))
solver.add_block(6, 1, (1.5e-3, 4.5e-3), (0, 1.5e-3), (0, Nz), subpixels=8)

solver.add_UPML(['+y'], width=10, max_loss=5, n=3)
solver.solve(method="refined", max_restarts=8)
print(f"gammas={solver.gammas}")
print(f"refined residuals={solver.refined_residuals}, restarts={solver.refined_restarts}")
solver.visualize_with_gui()

output_dir = Path(__file__).resolve().parent / "example_outputs"
output_dir.mkdir(parents=True, exist_ok=True)
solver.save_results(output_dir / "modes_full.npz", include_eigenvectors=True)
