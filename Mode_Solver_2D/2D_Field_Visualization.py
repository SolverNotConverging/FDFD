from FDFD_Mode_Solver import FDFDModeSolver

# Parameters for the simulation
x_range = 40e-3  # 15 mm in x-direction
y_range = 10e-3  # 10 mm in y-direction
Nx = 400  # Grid points in x
Ny = 100  # Grid points in y
frequency = 25e9  # 25 GHz
num_modes = 20  # Number of modes to compute

# Initialize solver and define structure
solver = FDFDModeSolver(frequency, x_range, y_range, Nx, Ny, num_modes)

solver.add_object(1e8, 1, (130, 150), (41, 42))
solver.add_object(1e8, 1, (250, 270), (41, 42))
solver.add_object(10.2, 1, (0, Nx), (42, 55))
solver.add_object(2.2, 1, (0, Nx), (55, 60))
solver.add_object(1e8, 1, (150, 250), (60, 61))
solver.add_object(1e8, 1, (0, 130), (60, 61))
solver.add_object(1e8, 1, (270, Nx), (60, 61))

solver.add_UPML(50, 3, 5, direction='x')

# Solve for the eigenmodes
solver.solve()
# solver.visualize_with_gui()
solver.visualize_with_gui()
