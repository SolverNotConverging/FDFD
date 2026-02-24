from Periodic_Mode_Solver import TM_Mode_Solver

# Periodic leaky-wave antenna unit cell (TM)
x_range = 8e-3
z_range = 4e-3
Nx = 320
Nz = 60
frequency = 24e9
num_modes = 6
guess = 0

solver = TM_Mode_Solver(freq=frequency, x_range=x_range, z_range=z_range, Nx=Nx, Nz=Nz,
                        num_modes=num_modes, guess=guess)

# Ground/metal edges
solver.add_object(-1e8, 1, x_indices=[int(0.70 * Nx)], z_indices=range(0, Nz))
solver.add_object(-1e8, 1, x_indices=[Nx - 1], z_indices=range(0, Nz))

# Dielectric loading (two regions)
solver.add_object(3.2, 1, x_indices=range(int(0.70 * Nx) + 1, int(0.85 * Nx)), z_indices=range(Nz))
solver.add_object(10.2, 1, x_indices=range(int(0.85 * Nx), Nx - 1), z_indices=range(Nz))

solver.add_UPML(pml_width=80, n=3, sigma_max=5, direction="top")
solver.solve()
solver.visualize_with_gui()
