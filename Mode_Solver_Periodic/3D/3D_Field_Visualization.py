from Periodic_Mode_Solver_3D import Periodic_3D_Mode_Solver  # adjust to your module name

# --- Small grid

Nx = 39
Ny = 60
Nz = 28

solver = Periodic_3D_Mode_Solver(Nx=Nx, Ny=Ny, Nz=Nz,
                                 x_range=7.8e-3, y_range=6e-3, z_range=5.7e-3,
                                 freq=22e9, num_modes=2, tol=1e-2)

# Build waveguide unit cell
solver.add_object(1e8, 1, slice(0, Nx), slice(Ny - 31, Ny - 30), slice(0, Nz))
solver.add_object(1, 1, slice(5, Nx - 5), slice(Ny - 31, Ny - 30), slice(0, 7))
solver.add_object(3, 1, slice(0, Nx), slice(Ny - 29, Ny - 14), slice(0, Nz))
solver.add_object(10.2, 1, slice(0, Nx), slice(Ny - 14, Ny - 1), slice(0, Nz))
solver.add_object(1e8, 1, slice(0, Nx), slice(Ny - 1, Ny), slice(0, Nz))
solver.add_object(1, 1e8, slice(0, 1), slice(0, Ny), slice(0, Nz))
solver.add_object(1, 1e8, slice(Nx - 1, Nx), slice(0, Ny), slice(0, Nz))

# Absorbing layers along ±y
solver.add_absorbing_boundary(['+y'], width=15, max_loss=10, n=3)

solver.solve()

# View modes in this periodic unit cell
solver.visualize_with_gui()

# Save including raw eigenvectors (big file)
solver.save_results("modes_full.npz", include_eigenvectors=True)
