from Periodic_Mode_Solver import TM_Mode_Solver

# Parameters for the simulation
x_range = 20e-3  # 20 mm in x-direction
z_range = 5e-3  # 5 mm in y-direction
Nx = 200  # Number of grid points in x-direction
Nz = 50  # Number of grid points in y-direction
frequency = 23.5e9  # Frequency
num_modes = 10  # Number of modes to solve for
guess = 0

solver = TM_Mode_Solver(freq=frequency, x_range=x_range, z_range=z_range, Nx=Nx, Nz=Nz, num_modes=num_modes,
                        guess=guess, ncv=40)

# Define structure
solver.add_object(-1e8, 1, x_indices=[161], z_indices=range(0, 14))
solver.add_object(3, 1, x_indices=range(162, 177), z_indices=range(solver.Nz))
solver.add_object(10.2, 1, x_indices=range(177, 190), z_indices=range(solver.Nz))
solver.add_object(-1e8, 1, x_indices=[190], z_indices=range(solver.Nz))
solver.add_UPML(pml_width=50, n=3, sigma_max=5, direction='top')

solver.solve()
solver.visualize_with_gui()
