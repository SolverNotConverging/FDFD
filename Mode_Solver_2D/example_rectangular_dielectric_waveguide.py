from Mode_Solver_2D import ModeSolver2D

# Rectangular waveguide cross section
x_range = 12e-3
y_range = 10e-3
Nx = 120
Ny = 100
frequency = 100e9
num_modes = 6

solver = ModeSolver2D(frequency, x_range, y_range, Nx, Ny, num_modes)

solver.add_object(4, 1, (30, 60), (40, 60))
solver.add_pec((10, 80), (38, 40))

solver.solve()
solver.visualize_with_gui()
