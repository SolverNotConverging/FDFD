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

left_1 = 180
right_1 = 220
left_2 = 160
right_2 = 240

top = 56
bottom = 70

solver.add_object(1e8, 1, (left_2, left_1), (top, top + 1))
solver.add_object(1e8, 1, (right_1, right_2), (top, top + 1))
solver.add_object(10.2, 1, (0, Nx), (top + 1, bottom))
solver.add_object(1e8, 1, (left_1, right_1), (bottom, bottom + 1))
solver.add_object(1e8, 1, (0, left_2), (bottom, bottom + 1))
solver.add_object(1e8, 1, (right_2, Nx), (bottom, bottom + 1))

solver.add_UPML(50, 3, 5, direction='x')

# Solve for the eigenmodes
solver.solve()
# solver.visualize_with_gui()
solver.visualize_with_gui()
