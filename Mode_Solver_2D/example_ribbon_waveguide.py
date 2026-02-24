from FDFD_Mode_Solver import FDFDModeSolver

# Ribbon waveguide (strip) in air
x_range = 5e-6
y_range = 5e-6
Nx = 200
Ny = 200
frequency = 193.5e12
num_modes = 6

solver = FDFDModeSolver(frequency, x_range, y_range, Nx, Ny, num_modes)

core_eps = 11.5

strip_width = 60
strip_height = 40
x0 = Nx // 2 - strip_width // 2
x1 = x0 + strip_width
y0 = Ny // 2 - strip_height // 2
y1 = y0 + strip_height
solver.add_object(core_eps, 1, (x0, x1), (y0, y1))

solver.add_UPML(pml_width=20, n=3, sigma_max=4, direction="both")
solver.solve()
solver.visualize_with_gui()
