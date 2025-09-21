import numpy as np
import matplotlib.pyplot as plt
import glob, os, re

import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Periodic_Mode_Solver_3D import Periodic_3D_Mode_Solver  # use your module name

data_dir = ""


# Helper to extract numeric freq from "modes_XX.XX.npz"
def freq_from_name(path):
    m = re.search(r"modes_([0-9]+(?:\.[0-9]+)?)\.npz$", os.path.basename(path))
    return float(m.group(1)) if m else None


files = [p for p in glob.glob(os.path.join(data_dir, "modes_*.npz")) if freq_from_name(p) is not None]
files.sort(key=freq_from_name)  # numeric sort

freqs = []
beta_rows = []  # rows will be variable-length; we'll pad later
alpha_rows = []
max_modes = 0

for fpath in files:
    f = freq_from_name(fpath)
    try:
        solver = Periodic_3D_Mode_Solver.load_results(fpath)
    except Exception as e:
        print(f"Skipping {fpath}: {e}")
        continue

    gammas_norm = np.asarray(solver.gammas)

    freqs.append(f)
    beta_rows.append(np.imag(gammas_norm))
    alpha_rows.append(np.real(gammas_norm))
    max_modes = max(max_modes, gammas_norm.size)


# Pad rows to same length with NaN
def pad_rows(rows, width):
    out = np.full((len(rows), width), np.nan, dtype=float)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


freqs = np.array(freqs)
beta = pad_rows(beta_rows, max_modes)
alpha = pad_rows(alpha_rows, max_modes)

# ---- Plot ----
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

for m in range(max_modes):
    mask_b = ~np.isnan(beta[:, m])
    mask_a = ~np.isnan(alpha[:, m])
    if mask_b.any():
        ax1.scatter(freqs[mask_b], beta[mask_b, m], s=20, label=f"Mode {m + 1}")
    if mask_a.any():
        ax2.scatter(freqs[mask_a], np.abs(alpha[mask_a, m]), s=20)

ax1.set_ylabel(r"Normalized Beta  ($\beta/k0$)")
ax1.grid(True)
ax1.legend(loc="best")

ax2.set_xlabel("Frequency (GHz)")
ax2.set_ylabel(r"Normalized Alpha ($\alpha/k0$)")
ax2.grid(True)

plt.tight_layout()
plt.show()
plt.savefig('Sweep_3D.png')
