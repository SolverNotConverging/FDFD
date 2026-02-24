from FDFD_Mode_Solver import FDFDModeSolver

# Rectangular waveguide cross section
x_range = 50e-3
y_range = 50e-3
Nx = 200
Ny = 200
frequency = 50e9
num_modes = 6

solver = FDFDModeSolver(frequency, x_range, y_range, Nx, Ny, num_modes)

solver.add_object(10, 1, (60, 140), (70, 130))

solver.solve()
solver.visualize_with_gui()
