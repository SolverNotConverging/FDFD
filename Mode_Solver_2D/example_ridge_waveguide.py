from FDFD_Mode_Solver import FDFDModeSolver

# Ridge waveguide cross-section
x_range = 6e-6
y_range = 4e-6
Nx = 240
Ny = 160
frequency = 200e12
num_modes = 6

solver = FDFDModeSolver(frequency, x_range, y_range, Nx, Ny, num_modes)

core_eps = 12.0
sub_eps = 2.1

# Substrate
solver.add_object(sub_eps, 1, (0, Nx), (0, Ny // 2))

# Slab layer
slab_thickness = 30
solver.add_object(core_eps, 1, (0, Nx), (Ny // 2, Ny // 2 + slab_thickness))

# Ridge on top of the slab
ridge_width = 60
ridge_height = 20
x0 = Nx // 2 - ridge_width // 2
x1 = x0 + ridge_width
y0 = Ny // 2 + slab_thickness
y1 = y0 + ridge_height
solver.add_object(core_eps, 1, (x0, x1), (y0, y1))

solver.add_UPML(pml_width=20, n=3, sigma_max=4, direction="both")
solver.solve()
solver.visualize_with_gui()
