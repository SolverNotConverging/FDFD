import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from FDFD_1D_Mode_Solver import FDFDModeSolver  # Your custom solver

# Parameters
freqs = np.linspace(10e9, 100e9, 100)  # 10 GHz to 100 GHz
x_range = 20e-3
Nx = 2000
num_modes = 5

# Initialize a dictionary to store mode data
data = {"Frequency (GHz)": freqs * 1e-9}
for m in range(num_modes):
    data[f"Beta_TE_{m + 1}"] = []
    data[f"Alpha_TE_{m + 1}"] = []
    data[f"Beta_TM_{m + 1}"] = []
    data[f"Alpha_TM_{m + 1}"] = []

# Frequency scan
for f in freqs:
    solver = FDFDModeSolver(frequency=f, x_range=x_range, Nx=Nx, num_modes=num_modes)
    solver.add_object(10.2, 1, (1322, 1449))
    solver.add_object(2.2, 1, (1449, 1500))
    solver.add_object(-1e8, 1, (1500, 1501))
    solver.solve()

    # Append all modes
    for m in range(num_modes):
        data[f"Beta_TE_{m + 1}"].append(solver.beta_TE[m] if m < len(solver.beta_TE) else np.nan)
        data[f"Alpha_TE_{m + 1}"].append(solver.alpha_TE[m] if m < len(solver.alpha_TE) else np.nan)
        data[f"Beta_TM_{m + 1}"].append(solver.beta_TM[m] if m < len(solver.beta_TM) else np.nan)
        data[f"Alpha_TM_{m + 1}"].append(solver.alpha_TM[m] if m < len(solver.alpha_TM) else np.nan)

# Save to Excel
df = pd.DataFrame(data)
df.to_excel(r"Dispersion_1D\1D_modes_dispersion.xlsx", index=False)

# Plotting
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# Beta plot
for m in range(num_modes):
    axes[0].plot(data["Frequency (GHz)"], data[f"Beta_TE_{m + 1}"], '--', label=f"TE Mode {m + 1}")
    axes[0].plot(data["Frequency (GHz)"], data[f"Beta_TM_{m + 1}"], '-', label=f"TM Mode {m + 1}")
axes[0].set_ylabel(r"Normalized Beta $\beta/k_0$")
axes[0].set_title("Propagation Constants (β) for TE and TM Modes")
axes[0].legend(loc='upper right')
axes[0].grid(True)
axes[0].set_ylim(0)

# Alpha plot
for m in range(num_modes):
    axes[1].plot(data["Frequency (GHz)"], data[f"Alpha_TE_{m + 1}"], '--', label=f"TE Mode {m + 1}")
    axes[1].plot(data["Frequency (GHz)"], data[f"Alpha_TM_{m + 1}"], '-', label=f"TM Mode {m + 1}")
axes[1].set_ylabel(r"Normalized Alpha $\alpha/k_0$")
axes[1].set_xlabel("Frequency (GHz)")
axes[1].set_title("Attenuation Constants (α) for TE and TM Modes")
axes[1].legend(loc='upper right')
axes[1].grid(True)

plt.tight_layout()
plt.savefig(r"Dispersion_1D\1D_modes_plot.png", dpi=300)
plt.show()
