from FDFD_Mode_Solver import FDFDModeSolver

# Ridge waveguide cross-section
x_range = 60e-3
y_range = 50e-3
Nx = 240
Ny = 200
frequency = 50e9
num_modes = 6

solver = FDFDModeSolver(frequency, x_range, y_range, Nx, Ny, num_modes)

# Slab
solver.add_object(8.0, 1, (0, Nx), (60, 100))

# Ridge
slab_thickness = 30
solver.add_object(12.0, 1, (80, 160), (100, 140))

solver.add_UPML(pml_width=20, n=3, sigma_max=4, direction="both")
solver.solve()
solver.visualize_with_gui()
