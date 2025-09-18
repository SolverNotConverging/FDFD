import numpy as np
from tqdm import tqdm

from Periodic_Mode_Solver_3D import Periodic_3D_Mode_Solver

Nx = 39
Ny = 60
Nz = 28

x_range = 7.8e-3
y_range = 6e-3
z_range = 5.7e-3

f_start = 21e9
f_stop = 27e9
f_step = 1e9
frequencies = np.arange(f_start, f_stop, f_step)
num_modes = 4
tol = 1e-2


def guess_func(f):
    return 0


for f in tqdm(frequencies, desc="Frequency sweep"):
    sigma_guess = guess_func(f) if guess_func else 0

    solver = Periodic_3D_Mode_Solver(Nx=Nx, Ny=Ny, Nz=Nz, x_range=x_range, y_range=y_range, z_range=z_range, freq=f,
                                     num_modes=num_modes, sigma_guess_func=sigma_guess, tol=tol)
    solver.add_object(1e8, 1, slice(0, Nx), slice(Ny - 31, Ny - 30), slice(0, Nz))
    solver.add_object(1, 1, slice(5, Nx - 5), slice(Ny - 31, Ny - 30), slice(0, 7))
    solver.add_object(3, 1, slice(0, Nx), slice(Ny - 29, Ny - 14), slice(0, Nz))
    solver.add_object(10.2, 1, slice(0, Nx), slice(Ny - 14, Ny - 1), slice(0, Nz))
    solver.add_object(1e8, 1, slice(0, Nx), slice(Ny - 1, Ny), slice(0, Nz))
    solver.add_object(1, 1e8, slice(0, 1), slice(0, Ny), slice(0, Nz))
    solver.add_object(1, 1e8, slice(Nx - 1, Nx), slice(0, Ny), slice(0, Nz))

    # Absorbing layers along ±y
    solver.add_UPML(['+y'], width=15, max_loss=10, n=3)

    try:
        solver.solve()
        print(f"Frequency Sweep at {f / 1e9:.3f} GHz")
        for n in range(num_modes):
            print(f"Mode {n}: Beta: {solver.gammas[n].real}; Alpha: {solver.gammas[n].imag}")
        solver.save_results(f"Freq_Sweep_3D/modes_{f}.npz")
    except Exception as e:
        # fill with NaNs if fails
        print(f"[WARN] eigs failed at {f / 1e9:.2f} GHz: {e}")
