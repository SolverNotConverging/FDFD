import matplotlib.pyplot as plt
import pandas as pd

# ----------------------
# Load data
# ----------------------
df = pd.read_excel("mode_data.xlsx")
frequencies_GHz = df["Frequency (Hz)"] / 1e9

# Determine number of modes automatically
alpha_columns = [col for col in df.columns if col.startswith("Alpha")]
beta_columns = [col for col in df.columns if col.startswith("Beta")]
num_modes = len(alpha_columns)

# ----------------------
# Scatter plots
# ----------------------
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Alpha / k0
for i, col in enumerate(alpha_columns):
    axs[0].scatter(frequencies_GHz, -df[col], label=f'Mode {i + 1}', s=20)
axs[0].set_ylabel(r'$\alpha / k_0$')
axs[0].legend()
axs[0].grid(True)
axs[0].set_ylim([0, 0.1])

# Beta / k0
for i, col in enumerate(beta_columns):
    axs[1].scatter(frequencies_GHz, df[col], label=f'Mode {i + 1}', s=20)
axs[1].set_xlabel("Frequency (GHz)")
axs[1].set_ylabel(r'$\beta / k_0$')
axs[1].grid(True)
axs[1].set_ylim([-1, 1])

plt.tight_layout()
plt.show()
