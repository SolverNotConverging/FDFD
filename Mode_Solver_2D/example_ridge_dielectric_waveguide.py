from Mode_Solver_2D import ModeSolver2D

# Ridge waveguide cross-section
x_range = 24e-3
y_range = 16e-3
Nx = 240
Ny = 160
frequency = 30e9
num_modes = 6

solver = ModeSolver2D(frequency, x_range, y_range, Nx, Ny, num_modes)

# Slab
solver.add_object(6.0, 1, (0, Nx), (60, 80))

# Ridge
solver.add_object(12.0, 1, (100, 140), (80, 100))

solver.add_UPML(pml_width=20, n=3, sigma_max=4, direction="x")
solver.solve()
solver.visualize_with_gui()
