import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_excel("2D_modes_dispersion.xlsx")

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
plt.show()
