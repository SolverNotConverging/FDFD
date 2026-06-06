from Mode_Solver_2D import ModeSolver2D

# Microstrip cross section
x_range = 12e-3
y_range = 10e-3
Nx = 120
Ny = 100
frequency = 100e9
num_modes = 6

solver = ModeSolver2D(frequency, x_range, y_range, Nx, Ny, num_modes)

solver.add_rectangle(3 + 0.1j, 1, (2e-3, 10e-3), (2e-3, 8e-3))
solver.add_pec((2e-3, 10e-3), (1.9e-3, 2e-3))
solver.add_pec((2e-3, 10e-3), (8e-3, 8.1e-3))
solver.add_pec((1.9e-3, 2e-3), (2e-3, 8e-3))
solver.add_pec((10e-3, 10.1e-3), (2e-3, 8e-3))

solver.solve()
solver.visualize_with_gui()
