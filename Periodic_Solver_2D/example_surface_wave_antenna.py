from Periodic_Solver_2D import PeriodicModeSolver2D

# Periodic leaky-wave antenna unit cell (TM)
x_range = 10e-3
z_range = 8e-3
Nx = 200
Nz = 80
frequency = 20e9
num_modes = 8
guess = 5

solver = PeriodicModeSolver2D("TM", freq=frequency, x_range=x_range, z_range=z_range, Nx=Nx, Nz=Nz,
                              num_modes=num_modes, guess=guess, mode_filter=False)

# Ground/metal edges
solver.add_pec((2.27e-3, 2.37e-3), (1.0e-3, 2.0e-3))
solver.add_pec((0.9e-3, 1.0e-3), (0, Nz))

# Dielectric loading
solver.add_rectangle(10.2, 1, (1.0e-3, 2.27e-3), (0, Nz), subpixels=8)

solver.add_pml(pml_width=30, n=3, sigma_max=5, direction="x+")
solver.solve()
solver.visualize_with_gui()
