from FDFD_Mode_Solver import FDFDModeSolver

# Parameters for the simulation
x_range = 40e-3  # 40 mm in x-direction
y_range = 10e-3  # 10 mm in y-direction
Nx = 400  # Grid points in x
Ny = 100  # Grid points in y
frequency = 20e9  # 25 GHz
num_modes = 15  # Number of modes to compute

# Initialize solver and define structure
solver = FDFDModeSolver(frequency, x_range, y_range, Nx, Ny, num_modes)
solver.add_object({'xx': 3, 'yy': 4, 'zz': 3}, {'xx': 1, 'yy': 1, 'zz': 1}, (0, 100), (45, 58))
solver.add_object({'xx': 10.2, 'yy': 10.2, 'zz': 10.2}, {'xx': 1, 'yy': 1, 'zz': 1}, (100, 300), (45, 58))
solver.add_object({'xx': 3, 'yy': 4, 'zz': 3}, {'xx': 1, 'yy': 1, 'zz': 1}, (300, 400), (45, 58))
solver.add_UPML(50, 3, 5, direction='x')

# Solve for the eigenmodes
solver.solve()
# solver.visualize_with_gui()
solver.visualize_with_gui()
