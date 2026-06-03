from Mode_Solver_2D import ModeSolver2D

# Microstrip cross section
x_range = 12e-3
y_range = 10e-3
Nx = 360
Ny = 300
frequency = 100e9
num_modes = 6

solver = ModeSolver2D(frequency, x_range, y_range, Nx, Ny, num_modes)

# Signal strip, dielectric substrate, and ground plane.
solver.add_pec((5e-3, 7e-3), (5e-3, 5.1e-3))
solver.add_rectangle(4, 1, (2e-3, 10e-3), (4e-3, 5e-3))
solver.add_pec((0.5e-3, 11.5e-3), (3.9e-3, 4e-3))

solver.solve()
solver.visualize_with_gui()
