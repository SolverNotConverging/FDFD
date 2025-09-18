import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from Periodic_Mode_Solver import TM_Mode_Solver

x_range = 20e-3  # 20 mm in x-direction
z_range = 5e-3  # 5 mm in z-direction
Nx = 200  # Number of grid points in x-direction
Nz = 50  # Number of grid points in z-direction
f_start = 22e9
f_stop = 26e9
f_step = 0.02e9
frequencies = np.arange(f_start, f_stop, f_step)
num_modes = 4


def guess_func(f):
    f_GHz = f / 1e9
    k_0 = 2 * np.pi * f / 3e8
    return 1j * (0.2 * f_GHz - 5) * k_0


data = {
    "Frequency (Hz)": frequencies
}
for mode in range(1, num_modes + 1):
    data[f"Alpha_Mode_{mode}"] = []
    data[f"Beta_Mode_{mode}"] = []

for f in tqdm(frequencies, desc="Frequency sweep"):
    sigma_guess = guess_func(f) if guess_func else 0

    solver = TM_Mode_Solver(freq=f, x_range=x_range, z_range=z_range, Nx=Nx, Nz=Nz, num_modes=num_modes,
                            guess=sigma_guess, ncv=None)
    # Define structure
    solver.add_object(-1e8, 1, x_indices=[161], z_indices=range(0, 14))
    solver.add_object(3, 1, x_indices=range(162, 177), z_indices=range(solver.Nz))
    solver.add_object(10.2, 1, x_indices=range(177, 190), z_indices=range(solver.Nz))
    solver.add_object(-1e8, 1, x_indices=[190], z_indices=range(solver.Nz))
    solver.add_UPML(pml_width=50, sigma_max=5)

    try:
        solver.solve()
        for mode in range(num_modes):
            gamma = solver.gammas[mode]
            data[f"Alpha_Mode_{mode + 1}"].append(gamma.real)
            data[f"Beta_Mode_{mode + 1}"].append(gamma.imag)
    except Exception as e:
        # fill with NaNs if fails
        print(f"[WARN] eigs failed at {f / 1e9:.2f} GHz: {e}")
        for mode in range(num_modes):
            data[f"Alpha_Mode_{mode + 1}"].append(np.nan)
            data[f"Beta_Mode_{mode + 1}"].append(np.nan)

# ----------------------
# Save data
# ----------------------
df = pd.DataFrame(data)
df.to_excel(r"Freq_Swep\mode_data.xlsx", index=False)

# ----------------------
# Plot
# ----------------------
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

for mode in range(1, num_modes + 1):
    axs[0].scatter(df["Frequency (Hz)"] / 1e9, df[f"Alpha_Mode_{mode}"], label=f'Mode {mode}', s=15)
    axs[1].scatter(df["Frequency (Hz)"] / 1e9, df[f"Beta_Mode_{mode}"], label=f'Mode {mode}', s=15)

axs[0].set_ylabel(r'$\alpha / k_0$')
axs[0].legend()
axs[0].grid(True)

axs[1].set_xlabel('Frequency (GHz)')
axs[1].set_ylabel(r'$\beta / k_0$')
axs[1].grid(True)

plt.tight_layout()
plt.savefig(r"Freq_Swep\mode_data.png", dpi=300)
plt.show()
