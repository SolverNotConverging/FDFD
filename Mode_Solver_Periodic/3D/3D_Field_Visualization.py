from Periodic_Mode_Solver_3D import Periodic_3D_Mode_Solver  # adjust to your module name

# --- Small grid
solver = Periodic_3D_Mode_Solver(Nx=40, Ny=40, Nz=10,
                                 x_range=4e-3, y_range=4e-3, z_range=1e-3,
                                 freq=50e9, num_modes=2, tol=1e-2)

# Build waveguide unit cell
solver.add_object(7, 1, slice(15, 25), slice(18, 28), slice(0, 10))

# Periodic inclusions within the unit cell
solver.add_object(7, 1, slice(15, 25), slice(15, 18), slice(0, 3))

# Absorbing layers along ±y
solver.add_absorbing_boundary(['-x', '+x', '-y', '+y'], width=5, max_loss=5, n=3)

solver.solve()

# View modes in this periodic unit cell
solver.visualize_with_gui()
