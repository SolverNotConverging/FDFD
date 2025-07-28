from Electrostatic_Solver import ElectrostaticSolver

# Create a 1D solver with 100 grid points
solver_1d = ElectrostaticSolver(mesh_size=(100,), dim=1)

# Set a fixed potential of 10V at point 59
solver_1d.set_potential(slice(59, 60), potential_value=10)

# Set a fixed potential of -10V from points 10 to 19
solver_1d.set_potential(slice(10, 20), potential_value=-10)

# Define a region (x = 30 to 39) with permittivity changed to 2
solver_1d.add_object(slice(30, 40), erxx=2)

# Solve the system and visualize the results
solver_1d.solve()
solver_1d.visualize()