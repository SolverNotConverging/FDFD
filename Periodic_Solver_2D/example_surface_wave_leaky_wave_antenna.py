from Periodic_Mode_Solver_2D import PeriodicTMModeSolver

# Periodic leaky-wave antenna unit cell (TM)
x_range = 10e-3
z_range = 6e-3
Nx = 200
Nz = 120
frequency = 26e9
num_modes = 6
guess = 0

solver = PeriodicTMModeSolver(freq=frequency, x_range=x_range, z_range=z_range, Nx=Nx, Nz=Nz,
                        num_modes=num_modes, guess=guess)

# Ground/metal edges
solver.add_object(-1e8, 1, x_indices=[25], z_indices=range(0, 10))
solver.add_object(-1e8, 1, x_indices=[9], z_indices=range(0, Nz))

# Dielectric loading (two regions)
solver.add_object(10.2, 1, x_indices=range(10, 25), z_indices=range(Nz))

solver.add_UPML(pml_width=80, n=3, sigma_max=5, direction="top")
solver.solve()
solver.visualize_with_gui()
