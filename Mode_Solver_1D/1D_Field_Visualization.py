from FDFD_1D_Mode_Solver import FDFDModeSolver

# Parameters for the simulation
x_range = 50e-3  # 50 mm in x-direction
Nx = 5000  # Grid points in x
frequency = 100e9  # 100 GHz
num_modes = 15  # Number of modes to compute

# Initialize solver and define structure
solver = FDFDModeSolver(frequency, x_range, Nx, num_modes)
solver.add_object([3, 4, 5], 1, (2000, 3000))

# Solve for the eigenmodes
solver.solve()
# solver.visualize_with_gui()
solver.visualize_with_gui()
