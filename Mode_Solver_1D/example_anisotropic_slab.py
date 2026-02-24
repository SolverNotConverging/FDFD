from FDFD_1D_Mode_Solver import FDFDModeSolver

# Anisotropic slab waveguide (diagonal epsilon tensor)
x_range = 8e-3
Nx = 800
frequency = 100e9
num_modes = 6

solver = FDFDModeSolver(frequency, x_range, Nx, num_modes)

core_start = Nx // 2 - 100
core_stop = Nx // 2 + 100
solver.add_object([12.0, 9.5, 8.5], 1, (core_start, core_stop))

solver.add_UPML(pml_width=80, n=3, sigma_max=10, direction="both")
solver.solve()
solver.visualize_with_gui()
