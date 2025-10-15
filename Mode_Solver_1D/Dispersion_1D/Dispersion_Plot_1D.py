import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file
file_path = r"1D_modes_dispersion.xlsx"
df = pd.read_excel(file_path)

# Determine number of modes from column names
num_modes = sum(col.startswith("Beta_TE_") for col in df.columns)

# Plotting
fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

# Beta plot
for m in range(num_modes):
    axes[0].plot(df["Frequency (GHz)"], df[f"Beta_TE_{m + 1}"], '--', label=f"TE Mode {m + 1}")
    axes[0].plot(df["Frequency (GHz)"], df[f"Beta_TM_{m + 1}"], '-', label=f"TM Mode {m + 1}")
axes[0].set_ylabel(r"Normalized Beta $\beta/k_0$")
axes[0].set_title("Propagation Constants (β) for TE and TM Modes")
axes[0].legend(loc='upper right')
axes[0].grid(True)
axes[0].set_ylim(0)

# Alpha plot
for m in range(num_modes):
    axes[1].plot(df["Frequency (GHz)"], df[f"Alpha_TE_{m + 1}"], '--', label=f"TE Mode {m + 1}")
    axes[1].plot(df["Frequency (GHz)"], df[f"Alpha_TM_{m + 1}"], '-', label=f"TM Mode {m + 1}")
axes[1].set_ylabel(r"Normalized Alpha $\alpha/k_0$")
axes[1].set_xlabel("Frequency (GHz)")
axes[1].set_title("Attenuation Constants (α) for TE and TM Modes")
axes[1].legend(loc='upper right')
axes[1].grid(True)

plt.tight_layout()
plt.show()
