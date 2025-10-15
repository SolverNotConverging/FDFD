from Periodic_Mode_Solver_3D import Periodic_3D_Mode_Solver

# --- Small grid

Nx = 30
Ny = 20
Nz = 20

x_range = 6e-3
y_range = 6e-3
z_range = 8e-3

solver = Periodic_3D_Mode_Solver(Nx=Nx, Ny=Ny, Nz=Nz,
                                 x_range=x_range, y_range=y_range, z_range=z_range,
                                 freq=22e9, num_modes=2, tol=0.1)

# Build waveguide unit cell
solver.add_object(1e8, 1, slice(8, 22), slice(Ny - 9, Ny - 8), slice(0, 3))
solver.add_object(1e8, 1, slice(8, 22), slice(Ny - 9, Ny - 8), slice(5, 8))
solver.add_object(6, 1, slice(8, 22), slice(Ny - 8, Ny - 1), slice(0, Nz))
solver.add_object(1e8, 1, slice(0, Nx), slice(Ny - 1, Ny), slice(0, Nz))

# Absorbing layers along ±y
solver.add_UPML(['+y'], width=8, max_loss=5, n=3)

solver.solve()

# View modes in this periodic unit cell
solver.visualize_with_gui()

# Save including raw eigenvectors (big file)
solver.save_results("modes_full.npz", include_eigenvectors=True)
