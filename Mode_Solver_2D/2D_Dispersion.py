import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from FDFD_Mode_Solver import FDFDModeSolver

# Parameters
x_range = 40e-3
y_range = 10e-3
Nx = 400
Ny = 100
frequencies = np.arange(18e9, 20e9, 1e9)
num_modes = 10

# Storage for DataFrame
data_rows = []

with tqdm(total=len(frequencies), desc="Calculating frequencies") as pbar:
    for freq in frequencies:
        solver = FDFDModeSolver(frequency=freq, x_range=x_range, y_range=y_range, Nx=Nx, Ny=Ny, num_modes=num_modes)
        solver.add_object({'xx': -1e8, 'yy': -1e8, 'zz': -1e8}, {'xx': 1, 'yy': 1, 'zz': 1}, (130, 150), (44, 45))
        solver.add_object({'xx': -1e8, 'yy': -1e8, 'zz': -1e8}, {'xx': 1, 'yy': 1, 'zz': 1}, (250, 270), (44, 45))
        solver.add_object({'xx': 11.8, 'yy': 11.8, 'zz': 11.8}, {'xx': 1, 'yy': 1, 'zz': 1}, (0, 400), (45, 58))
        solver.add_object({'xx': -1e8, 'yy': -1e8, 'zz': -1e8}, {'xx': 1, 'yy': 1, 'zz': 1}, (150, 250), (58, 59))
        solver.add_object({'xx': -1e8, 'yy': -1e8, 'zz': -1e8}, {'xx': 1, 'yy': 1, 'zz': 1}, (0, 130), (58, 59))
        solver.add_object({'xx': -1e8, 'yy': -1e8, 'zz': -1e8}, {'xx': 1, 'yy': 1, 'zz': 1}, (270, 400), (58, 59))
        solver.add_absorbing_boundaries(80, 3, 20, direction='x')

        solver.solve()

        # Store each mode's data
        for mode_idx in range(num_modes):
            data_rows.append({
                "frequency_GHz": freq / 1e9,
                "mode_index": mode_idx + 1,
                "propagation_constant": solver.propagation_constant[mode_idx],
                "attenuation_constant": (solver.attenuation_constant[mode_idx])
            })

        pbar.update(1)

# Create DataFrame and save
df = pd.DataFrame(data_rows)
df.to_excel(r"Dispersion_2D\2D_modes_dispersion.xlsx", index=False)
print(r"Data saved to Dispersion_2D\2D_modes_dispersion.xlsx")

# Extract unique modes and frequencies
modes = df["mode_index"].unique()
frequencies = sorted(df["frequency_GHz"].unique())

# Plotting
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

# Plot propagation constant for each mode
for mode in modes:
    sub_df = df[df["mode_index"] == mode]
    ax[0].plot(sub_df["frequency_GHz"], sub_df["propagation_constant"], label=f"Mode {mode}")
ax[0].set_title("Propagation Constant vs Frequency")
ax[0].set_xlabel("Frequency (GHz)")
ax[0].set_ylabel(r"$\hat{\beta}$")
ax[0].grid(True)
ax[0].legend()

# Plot attenuation constant for each mode
for mode in modes:
    sub_df = df[df["mode_index"] == mode]
    ax[1].plot(sub_df["frequency_GHz"], sub_df["attenuation_constant"], label=f"Mode {mode}")
ax[1].set_title("Attenuation Constant vs Frequency")
ax[1].set_xlabel("Frequency (GHz)")
ax[1].set_ylabel(r"$\alpha$")
ax[1].grid(True)
ax[1].legend()

plt.tight_layout()
plt.savefig(r"Dispersion_2D\2D_modes_plot.png", dpi=300)
plt.show()
