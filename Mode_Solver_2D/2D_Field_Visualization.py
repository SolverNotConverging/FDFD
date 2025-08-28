from FDFD_Mode_Solver import FDFDModeSolver

# Parameters for the simulation
x_range = 40e-3  # 40 mm in x-direction
y_range = 10e-3  # 10 mm in y-direction
Nx = 400  # Grid points in x
Ny = 100  # Grid points in y
frequency = 20e9  # 20 GHz
num_modes = 15  # Number of modes to compute

# Initialize solver and define structure
solver = FDFDModeSolver(frequency, x_range, y_range, Nx, Ny, num_modes)
solver.add_object(-1e8, 1, (0, 150), (44, 45))
solver.add_object(-1e8, 1, (250, 400), (44, 45))
solver.add_object(10.2, 1, (0, 400), (45, 58))
solver.add_object(-1e8, 1, (150, 250), (58, 59))
solver.add_UPML(50, 3, 5, direction='x')

# Solve for the eigenmodes
solver.solve()
# solver.visualize_with_gui()
solver.visualize_with_gui()
