# example_sphere_scatter.py
import numpy as np
from matplotlib import pyplot as plt

from Scattering_Solver_2D import FDFD2DScatteringSolver  # ← adjust if you used a different file name

# --------------------------------------------------------------------------
# 1. numerical grid & frequency
f0 = 10e9  # 10 GHz
λ0 = 299792458 / f0
Lx, Ly = 6 * λ0, 6 * λ0  # 6×6 wavelengths of simulation window
Nx, Ny = 300, 300  # 20 pixels / λ  → 300×300 unknowns

sim = FDFD2DScatteringSolver(frequency=f0,
                             x_range=Lx, y_range=Ly,
                             Nx=Nx, Ny=Ny)

# --------------------------------------------------------------------------
# 2. add a dielectric sphere (== infinite z‑directed cylinder in 2‑D)
radius = 0.5 * λ0  # 0.5 λ cylinder
eps_r = -1e6  # relative permittivity

mask_sphere = (sim.X ** 2 + sim.Y ** 2) <= radius ** 2
sim.add_object(er_tensor=eps_r, mr_tensor=1.0, region_mask=mask_sphere)

# --------------------------------------------------------------------------
# 3. absorbing boundary & total/scattered‑field mask
sim.add_UPML(pml_width=40, sigma_max=10)  # perfectly matched layer, 20 cells thick
sim.add_mask(value=80)  # total field region = inner 50‑cell frame

# --------------------------------------------------------------------------
# 4. plane‑wave excitation (45° from +x axis)
sim.add_source(src_type="plane_wave", angle_deg=10.0, polarization="TE")
sim.solve_total_field_TM()  # returns Ny×Nx array
sim.TM_Visualization()
