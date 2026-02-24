from pathlib import Path

import numpy as np
from tqdm import tqdm

from Periodic_Mode_Solver_3D import Periodic_3D_Mode_Solver

Nx = 20
Ny = 18
Nz = 16

x_range = 6e-3
y_range = 6e-3
z_range = 8e-3

f_start = 21e9
f_stop = 25e9
f_step = 1e9
frequencies = np.arange(f_start, f_stop + f_step, f_step)
num_modes = 2
tol = 1e-2


def guess_func(f):
    k0 = 2 * np.pi * f / 3e8
    f_ghz = f / 1e9
    return 1j * k0 * (0.12 * f_ghz - 2.7)


output_dir = Path(__file__).resolve().parent / "example_outputs"
output_dir.mkdir(parents=True, exist_ok=True)

for f in tqdm(frequencies, desc="Frequency sweep"):
    sigma_guess = guess_func(f) if guess_func else 0

    solver = Periodic_3D_Mode_Solver(Nx=Nx, Ny=Ny, Nz=Nz,
                                     x_range=x_range, y_range=y_range, z_range=z_range,
                                     freq=f, num_modes=num_modes, sigma_guess=sigma_guess,
                                     tol=tol, ncv=None)

    solver.add_object(1e8, 1, slice(0, 3), slice(Ny - 8, Ny - 7), slice(0, Nz))
    solver.add_object(1e8, 1, slice(5, 8), slice(Ny - 8, Ny - 7), slice(0, Nz))
    solver.add_object(6, 1, slice(3, 17), slice(Ny - 7, Ny - 1), slice(0, Nz))
    solver.add_object(1e8, 1, slice(0, Nx), slice(Ny - 1, Ny), slice(0, Nz))
    solver.add_UPML(['+y'], width=8, max_loss=10, n=3)

    try:
        solver.solve()
        solver.save_results(output_dir / f"{f / 1e9:.0f}_GHz.npz")
        print(f"{f / 1e9:.1f} GHz: beta={solver.gammas[0].real}, alpha={solver.gammas[0].imag}")
    except Exception as exc:
        print(f"[WARN] eigs failed at {f / 1e9:.2f} GHz: {exc}")
