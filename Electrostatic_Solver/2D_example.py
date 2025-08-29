from Electrostatic_Solver import ElectrostaticSolver

# 2D example
solver_2d = ElectrostaticSolver(mesh_size=(50, 50), dim=2)

solver_2d.set_potential((slice(0, 50), slice(0, 1)), potential_value=-30)
solver_2d.set_potential((slice(0, 1), slice(0, 50)), potential_value=0)
solver_2d.set_potential((slice(49, 50), slice(0, 50)), potential_value=0)
solver_2d.set_potential((slice(0, 25), slice(30, 31)), potential_value=100)
solver_2d.set_potential((slice(24, 25), slice(30, 50)), potential_value=100)
solver_2d.set_potential((slice(25, 50), slice(49, 50)), potential_value=100)

solver_2d.add_object((slice(30, 40), slice(40, 50)), erxx=7, eryy=3)

solver_2d.solve()
solver_2d.visualize()
