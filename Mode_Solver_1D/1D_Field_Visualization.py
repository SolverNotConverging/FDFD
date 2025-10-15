from FDFD_1D_Mode_Solver import FDFDModeSolver

# Parameters for the simulation
x_range = 20e-3  # 50 mm in x-direction
Nx = 2000  # Grid points in x
frequency = 50e9  # 100 GHz
num_modes = 15  # Number of modes to compute

# Initialize solver and define structure
solver = FDFDModeSolver(frequency, x_range, Nx, num_modes)
solver.add_object(10.2, 1, (1373, 1470))
solver.add_object(3, 1, (1470, 1500))
solver.add_object(-1e8, 1, (1500, 1501))

# Solve for the eigenmodes
solver.solve()
# solver.visualize_with_gui()
solver.visualize_with_gui()
