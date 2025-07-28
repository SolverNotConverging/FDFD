from Periodic_Mode_Solver_3D import Periodic_3D_Mode_Solver  # adjust to your module name

# --- Small grid
solver = Periodic_3D_Mode_Solver(Nx=30, Ny=30, Nz=10,
                                 x_range=15e-3, y_range=15e-3, z_range=10e-3,
                                 freq=18e9, num_modes=5)

# Build waveguide unit cell
solver.add_object(slice(10, 20), slice(15, 20), slice(0, 10), erxx=7, eryy=7, erzz=7)

# Periodic inclusions within the unit cell
solver.add_object(slice(10, 20), slice(12, 15), slice(0, 3), erxx=7, eryy=7, erzz=7)

# Absorbing layers along ±y
solver.add_absorbing_boundary(['-x', '+x', '-y', '+y'], width=5, max_loss=30)

solver.solve()

# View modes in this periodic unit cell
solver.visualize_with_gui()
