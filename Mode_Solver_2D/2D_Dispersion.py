import os
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
frequencies = np.arange(15e9, 30e9, 0.2e9)
num_modes = 40

# Output folder
out_dir = r"Dispersion_2D"
os.makedirs(out_dir, exist_ok=True)
xlsx_path = os.path.join(out_dir, "2D_modes_dispersion.xlsx")
png_path  = os.path.join(out_dir, "2D_modes_plot.png")

# We'll build one row per frequency. Each row will have:
# frequency_GHz, Mode 1 beta, Mode 1 alpha, Mode 2 beta, Mode 2 alpha, ...
rows = []

with tqdm(total=len(frequencies), desc="Calculating frequencies") as pbar:
    for freq in frequencies:
        solver = FDFDModeSolver(
            frequency=freq, x_range=x_range, y_range=y_range, Nx=Nx, Ny=Ny, num_modes=num_modes
        )

        # Geometry
        solver.add_object(1e8, 1, (130, 150), (56, 57))
        solver.add_object(1e8, 1, (250, 270), (56, 57))
        solver.add_object(10.2, 1, (0, Nx), (57, 70))
        solver.add_object(1e8, 1, (150, 250), (70, 71))
        solver.add_object(1e8, 1, (0, 130), (70, 71))
        solver.add_object(1e8, 1, (270, Nx), (70, 71))

        solver.add_UPML(50, 3, 5, direction='x')
        solver.solve()

        # Build the frequency row
        row = {"frequency_GHz": freq / 1e9}
        # Ensure we don't overrun if solver returns fewer modes than requested
        n_available = min(num_modes, len(solver.propagation_constant), len(solver.attenuation_constant))

        for m in range(n_available):
            mode_num = m + 1
            row[f"Mode {mode_num} beta"]  = solver.propagation_constant[m]
            row[f"Mode {mode_num} alpha"] = solver.attenuation_constant[m]

        # If fewer modes are available, fill the rest with NaN so columns are consistent
        for m in range(n_available, num_modes):
            mode_num = m + 1
            row[f"Mode {mode_num} beta"]  = np.nan
            row[f"Mode {mode_num} alpha"] = np.nan

        rows.append(row)
        pbar.update(1)

# Create wide DataFrame (no mode_index column anywhere)
df_wide = pd.DataFrame(rows)

# Optional: order columns nicely: frequency_GHz, then (Mode 1 beta, Mode 1 alpha, Mode 2 beta, Mode 2 alpha, ...)
ordered_cols = ["frequency_GHz"]
for m in range(1, num_modes + 1):
    ordered_cols += [f"Mode {m} beta", f"Mode {m} alpha"]
df_wide = df_wide.reindex(columns=ordered_cols)

# Save to Excel
df_wide.to_excel(xlsx_path, index=False)
print(f"Data saved to {xlsx_path}")

# -------------------------
# Plotting from wide format
# -------------------------
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

# Propagation constants
for m in range(1, num_modes + 1):
    beta_col = f"Mode {m} beta"
    if beta_col in df_wide.columns:
        ax[0].plot(df_wide["frequency_GHz"], df_wide[beta_col], label=f"Mode {m}")

ax[0].set_title("Propagation Constant vs Frequency")
ax[0].set_xlabel("Frequency (GHz)")
ax[0].set_ylabel(r"$\hat{\beta}$")
ax[0].grid(True)
ax[0].legend(ncol=2, fontsize=8)

# Attenuation constants
for m in range(1, num_modes + 1):
    alpha_col = f"Mode {m} alpha"
    if alpha_col in df_wide.columns:
        ax[1].plot(df_wide["frequency_GHz"], df_wide[alpha_col], label=f"Mode {m}")

ax[1].set_title("Attenuation Constant vs Frequency")
ax[1].set_xlabel("Frequency (GHz)")
ax[1].set_ylabel(r"$\alpha$")
ax[1].grid(True)
ax[1].legend(ncol=2, fontsize=8)

plt.tight_layout()
plt.savefig(png_path, dpi=300)
plt.show()
print(f"Plot saved to {png_path}")
