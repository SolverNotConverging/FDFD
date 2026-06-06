from Mode_Solver_2D import ModeSolver2D

# Microstrip cross section
x_range = 10e-3
y_range = 10e-3
Nx = 100
Ny = 100
frequency = 100e9
num_modes = 6

solver = ModeSolver2D(frequency, x_range, y_range, Nx, Ny, num_modes)

# Signal strip, dielectric substrate, and ground plane.

solver.add_circle(6, 1, (5e-3, 5e-3), 3e-3)

solver.solve()
solver.visualize_with_gui()
