from Periodic_Solver_2D import PeriodicModeSolver2D

# Periodic leaky-wave antenna unit cell (TM)
x_range = 10e-3
z_range = 8e-3
Nx = 200
Nz = 80
frequency = 25e9
num_modes = 8
guess = 5

solver = PeriodicModeSolver2D("TM", freq=frequency, x_range=x_range, z_range=z_range, Nx=Nx, Nz=Nz,
                              num_modes=num_modes, guess=guess, mode_filter=False)

# Ground/metal edges
solver.add_pec((25, 26), (0, 10))
solver.add_pec((9, 10), (0, Nz))

# Dielectric loading
solver.add_rectangle(10.2, 1, (10, 23), (0, Nz))
solver.add_rectangle(7, 1, (23, 25), (0, Nz))

solver.add_pml(pml_width=50, n=3, sigma_max=5, direction="x+")
solver.solve()
solver.visualize_with_gui()
