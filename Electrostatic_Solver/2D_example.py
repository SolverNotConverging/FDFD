from Electrostatic_Solver import ElectrostaticSolver

# 2D example
solver_2d = ElectrostaticSolver(mesh_size=(50, 50), dim=2)

# set potential from 0 to 50 in x and 0 to 1 in y as -30
solver_2d.set_potential((slice(0, 50), slice(0, 1)), potential_value=-30)
solver_2d.set_potential((slice(0, 1), slice(0, 50)), potential_value=0)
solver_2d.set_potential((slice(49, 50), slice(0, 50)), potential_value=0)
solver_2d.set_potential((slice(0, 25), slice(30, 31)), potential_value=100)
solver_2d.set_potential((slice(24, 25), slice(30, 50)), potential_value=100)
solver_2d.set_potential((slice(25, 50), slice(49, 50)), potential_value=100)

# change the relative permittivity to 7 in x direction and 3 in y direction in a rectangular area from 31 to 40 in x and 41 to 50 in y
solver_2d.add_object((slice(30, 40), slice(40, 50)), erxx=7, eryy=3)

solver_2d.solve()
solver_2d.visualize()
