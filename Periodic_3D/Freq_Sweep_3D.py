import numpy as np
from tqdm import tqdm

from Periodic_Mode_Solver_3D import Periodic_3D_Mode_Solver

Nx = 20
Ny = 20
Nz = 20

x_range = 6e-3
y_range = 6e-3
z_range = 8e-3

f_start = 21e9
f_stop = 27e9
f_step = 1e9
frequencies = np.arange(f_start, f_stop, f_step)
num_modes = 4
tol = 1e-2


def guess_func(f):
    k0 = 2 * np.pi * f / 3e8
    f_ghz = f / 1e9
    return 1j * k0 * (0.14 * f_ghz - 3.25)


for f in tqdm(frequencies, desc="Frequency sweep"):
    sigma_guess = guess_func(f) if guess_func else 0

    solver = Periodic_3D_Mode_Solver(Nx=Nx, Ny=Ny, Nz=Nz, x_range=x_range, y_range=y_range, z_range=z_range, freq=f,
                                     num_modes=num_modes, sigma_guess=guess_func(f), tol=tol, ncv=None)

    solver.add_object(1e8, 1, slice(0, 3), slice(Ny - 9, Ny - 8), slice(0, Nz))
    solver.add_object(1e8, 1, slice(5, 8), slice(Ny - 9, Ny - 8), slice(0, Nz))
    solver.add_object(6, 1, slice(3, 17), slice(Ny - 8, Ny - 1), slice(0, Nz))
    solver.add_object(1e8, 1, slice(0, Nx), slice(Ny - 1, Ny), slice(0, Nz))

    # Absorbing layers along ±y
    solver.add_UPML(['+y'], width=10, max_loss=10, n=3)

    try:
        solver.solve()
        print(f"Frequency Sweep at {f / 1e9:.3f} GHz")
        for n in range(num_modes):
            print(f"Mode {n}: Beta: {solver.gammas[n].real}; Alpha: {solver.gammas[n].imag}")
        solver.save_results(f"Freq_Sweep_3D/{f / 1e9:.0f}_GHz.npz")
    except Exception as e:
        # fill with NaNs if fails
        print(f"[WARN] eigs failed at {f / 1e9:.2f} GHz: {e}")
