from Periodic_Solver_2D import PeriodicModeSolver2D

# Periodic leaky-wave antenna unit cell (TM)
x_range = 10e-3
z_range = 8e-3
Nx = 200
Nz = 80
frequency = 21e9
num_modes = 8
guess = 0

solver = PeriodicModeSolver2D(
    "TM",
    freq=frequency,
    x_range=x_range,
    z_range=z_range,
    Nx=Nx,
    Nz=Nz,
    num_modes=num_modes,
    guess=guess,
)

# Ground/metal edges
solver.add_rectangle(1e8, 1, (2.3e-3, 2.4e-3), (0, 1e-3))
solver.add_rectangle(1e8, 1, (0.9e-3, 1e-3), (0, Nz))
# Dielectric loading
solver.add_rectangle(10.2, 1, (1e-3, 2.3e-3), (0, Nz))

solver.add_pml(pml_width=50, n=3, sigma_max=5, direction="x+")
solver.solve()
solver.visualize_with_gui()
