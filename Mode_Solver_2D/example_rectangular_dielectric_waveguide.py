from Mode_Solver_2D import ModeSolver2D

# Rectangular waveguide cross section
x_range = 30e-3
y_range = 20e-3
Nx = 200
Ny = 200
frequency = 30e9
num_modes = 6

solver = ModeSolver2D(frequency, x_range, y_range, Nx, Ny, num_modes)

solver.add_object([10, 11, 12], 1, (60, 140), (70, 130))

solver.solve()
solver.visualize_with_gui()
