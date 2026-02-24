from Mode_Solver_1D import ModeSolver1D

# Isotropic slab waveguide
x_range = 10e-3
Nx = 1000
frequency = 100e9
num_modes = 6

solver = ModeSolver1D(frequency, x_range, Nx, num_modes)

core_start = Nx // 2 - 100
core_stop = Nx // 2 + 100
solver.add_object(11.5, 1, (core_start, core_stop))

solver.add_UPML(pml_width=100, n=3, sigma_max=8, direction="both")
solver.solve()
solver.visualize_with_gui()
