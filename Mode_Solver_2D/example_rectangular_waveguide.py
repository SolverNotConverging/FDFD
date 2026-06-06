from Mode_Solver_2D import ModeSolver2D

# Microstrip cross section
x_range = 12e-3
y_range = 10e-3
Nx = 120
Ny = 100
frequency = 100e9
num_modes = 30

solver = ModeSolver2D(frequency, x_range, y_range, Nx, Ny, num_modes)

# Signal strip, dielectric substrate, and ground plane.

solver.add_rectangle(6 + 0.1j, 1, (4e-3, 8e-3), (4e-3, 6e-3))
solver.add_pec((4e-3, 8e-3), (3.9e-3, 4e-3))
solver.add_pec((4e-3, 8e-3), (6e-3, 6.1e-3))
solver.add_pec((3.9e-3, 4e-3), (4e-3, 6e-3))
solver.add_pec((8e-3, 8.1e-3), (4e-3, 6e-3))

solver.solve()
solver.visualize_with_gui()
