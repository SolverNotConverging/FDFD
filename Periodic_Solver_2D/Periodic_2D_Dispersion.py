from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from Periodic_Solver_2D import PeriodicModeSolver2D

x_range = 10e-3
z_range = 8e-3
Nx = 200
Nz = 80
f_start = 16e9
f_stop = 25e9
f_step = 0.1e9
frequencies = np.arange(f_start, f_stop + f_step, f_step)
num_modes = 4


def guess_func(f):
    return 0


data = {"Frequency (Hz)": frequencies}
for mode in range(1, num_modes + 1):
    data[f"Alpha_Mode_{mode}"] = []
    data[f"Beta_Mode_{mode}"] = []

for f in tqdm(frequencies, desc="Frequency sweep"):
    sigma_guess = guess_func(f) if guess_func else 0

    solver = PeriodicModeSolver2D("TM", freq=f, x_range=x_range, z_range=z_range, Nx=Nx, Nz=Nz,
                                  num_modes=num_modes, guess=sigma_guess, ncv=None)

    # Ground/metal edges
    solver.add_pec((2.27e-3, 2.28e-3), (1.0e-3, 2.0e-3))
    solver.add_pec((0.9e-3, 1.0e-3), (0, Nz))

    # Dielectric loading
    solver.add_rectangle(10.2, 1, (1.0e-3, 2.27e-3), (0, Nz), subpixels=16)

    solver.add_pml(pml_width=50, n=3, sigma_max=5, direction="x+")

    try:
        solver.solve()
        for mode in range(num_modes):
            neff = solver.neff[mode]
            data[f"Alpha_Mode_{mode + 1}"].append(neff.real)
            data[f"Beta_Mode_{mode + 1}"].append(neff.imag)
    except Exception as exc:
        print(f"[WARN] eigs failed at {f / 1e9:.2f} GHz: {exc}")
        for mode in range(num_modes):
            data[f"Alpha_Mode_{mode + 1}"].append(np.nan)
            data[f"Beta_Mode_{mode + 1}"].append(np.nan)

output_dir = Path(__file__).resolve().parent / "example_outputs"
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.DataFrame(data)
df.to_csv(output_dir / "mode_data.csv", index=False)
