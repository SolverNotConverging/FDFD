"""Anisotropic dielectric and internal electrode on a refined 2D FEM mesh."""

from fem_electrostatics import Circle, ElectrostaticSolver, Rectangle


solver = ElectrostaticSolver(dim=2, outer_potential=0.0, x_range=((0.0, 0.02), (0.0, 0.01))[0], y_range=((0.0, 0.02), (0.0, 0.01))[1])
solver.add_object(region=Rectangle((0.006, 0.014), (0.002, 0.008)), name='high_dk_substrate', epsilon=((9.0, 0.4), (0.4, 6.0)))
solver.set_potential(region=Circle((0.01, 0.005), 0.001), potential=100.0, name='electrode')

mesh = solver.mesh(max_element_size=0.001, interface_refinement=0.65, boundary_refinement=0.4)
result = solver.solve(max_refinements=0)
print(f"nodes={mesh.info.nodes}, triangles={mesh.info.elements}")
print(f"electrode charge={result.conductor_charge('electrode'):.6e} C/m")
solver.show()
