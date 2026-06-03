from Mode_Solver_1D import ModeSolver1D

# Isotropic slab waveguide
x_range = 10e-3
Nx = 1000
frequency = 30e9
num_modes = 6

solver = ModeSolver1D(frequency, x_range, Nx, num_modes)

solver.add_layer(10.2, 1, (3e-3, 4.27e-3))
solver.add_pec((2.9e-3, 3e-3))

solver.solve()
solver.visualize_with_gui()
