from Mode_Solver_1D import ModeSolver1D

# Anisotropic slab waveguide (diagonal epsilon tensor)
x_range = 8e-3
Nx = 800
frequency = 100e9
num_modes = 6

solver = ModeSolver1D(frequency, x_range, Nx, num_modes)

solver.add_pmc((4.5e-3, 4.6e-3))
solver.add_layer((4, 5, 6), 1, (3e-3, 4.5e-3))
solver.add_pmc((2.9e-3, 3e-3))

solver.add_pml(pml_width=80, n=3, sigma_max=10, direction="all")
solver.solve()
solver.visualize_with_gui()
