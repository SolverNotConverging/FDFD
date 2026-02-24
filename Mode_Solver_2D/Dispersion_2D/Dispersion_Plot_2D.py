import pandas as pd
import matplotlib.pyplot as plt

# Load the wide-format Excel file
df = pd.read_excel("2D_modes_dispersion.xlsx")

# Identify all mode columns dynamically
beta_cols  = [c for c in df.columns if "beta" in c.lower()]
alpha_cols = [c for c in df.columns if "alpha" in c.lower()]

# Extract mode numbers
modes = sorted({int(c.split()[1]) for c in beta_cols})

# --- Plotting ---
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

# Plot propagation constants (β)
for m in modes:
    beta_col = f"Mode {m} beta"
    if beta_col in df.columns:
        ax[0].plot(df["frequency_GHz"], df[beta_col],'x-', label=f"Mode {m}")
ax[0].set_title("Propagation Constant vs Frequency")
ax[0].set_xlabel("Frequency (GHz)")
ax[0].set_ylabel(r"$\hat{\beta}$")
ax[0].grid(True)
ax[0].legend(ncol=3, fontsize=8, loc="best")

# Plot attenuation constants (α)
for m in modes:
    alpha_col = f"Mode {m} alpha"
    if alpha_col in df.columns:
        ax[1].plot(df["frequency_GHz"], df[alpha_col],'x-', label=f"Mode {m}")
ax[1].set_title("Attenuation Constant vs Frequency")
ax[1].set_xlabel("Frequency (GHz)")
ax[1].set_ylabel(r"$\alpha$")
ax[1].grid(True)
ax[1].legend(ncol=3, fontsize=8, loc="best")

plt.tight_layout()
plt.show()
