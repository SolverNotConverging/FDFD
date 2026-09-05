"""Layered 1D FEM capacitor with geometry created before the mesh."""

from fem_electrostatics import ElectrostaticSolver, Interval


solver = ElectrostaticSolver(dim=1, outer_potential=None, x_range=(0.0, 0.01))
solver.add_object(region=Interval((0.004, 0.007)), name='dielectric', epsilon=6.0)
solver.set_potential(region='left', potential=0.0, name='ground')
solver.set_potential(region='right', potential=10.0, name='drive')

solver.mesh(max_element_size=0.0004)
result = solver.solve(max_refinements=0)
print(f"nodes={result.mesh.info.nodes}, energy={result.energy:.6e} J/m2")
solver.show()
