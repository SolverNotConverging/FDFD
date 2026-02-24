from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from Periodic_Mode_Solver_2D import PeriodicTMModeSolver

x_range = 10e-3
z_range = 6e-3
Nx = 200
Nz = 120
f_start = 15e9
f_stop = 30e9
f_step = 1e9
frequencies = np.arange(f_start, f_stop + f_step, f_step)
num_modes = 4


def guess_func(f):
    f_ghz = f / 1e9
    k_0 = 2 * np.pi * f / 3e8
    return 1j * (0.18 * f_ghz - 3.0) * k_0


data = {"Frequency (Hz)": frequencies}
for mode in range(1, num_modes + 1):
    data[f"Alpha_Mode_{mode}"] = []
    data[f"Beta_Mode_{mode}"] = []

for f in tqdm(frequencies, desc="Frequency sweep"):
    sigma_guess = guess_func(f) if guess_func else 0

    solver = PeriodicTMModeSolver(freq=f, x_range=x_range, z_range=z_range, Nx=Nx, Nz=Nz,
                            num_modes=num_modes, guess=sigma_guess, ncv=None)

    solver.add_object(-1e8, 1, x_indices=[25], z_indices=range(0, 10))
    solver.add_object(-1e8, 1, x_indices=[9], z_indices=range(0, Nz))

    # Dielectric loading (two regions)
    solver.add_object(10.2, 1, x_indices=range(10, 25), z_indices=range(Nz))

    solver.add_UPML(pml_width=80, n=3, sigma_max=5, direction="top")

    try:
        solver.solve()
        for mode in range(num_modes):
            gamma = solver.gammas[mode]
            data[f"Alpha_Mode_{mode + 1}"].append(gamma.real)
            data[f"Beta_Mode_{mode + 1}"].append(gamma.imag)
    except Exception as exc:
        print(f"[WARN] eigs failed at {f / 1e9:.2f} GHz: {exc}")
        for mode in range(num_modes):
            data[f"Alpha_Mode_{mode + 1}"].append(np.nan)
            data[f"Beta_Mode_{mode + 1}"].append(np.nan)

output_dir = Path(__file__).resolve().parent / "example_outputs"
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.DataFrame(data)
df.to_csv(output_dir / "mode_data.csv", index=False)
