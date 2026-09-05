"""Layered 1D FEM capacitor with geometry created before the mesh."""

from Electrostatic_Solver import ElectrostaticSolver, Interval


solver = ElectrostaticSolver(dim=1, domain=(0.0, 10e-3), outer_potential=None)
solver.add_object(Interval((4e-3, 7e-3)), erxx=6.0, name="dielectric")
solver.set_potential("left", 0.0, name="ground")
solver.set_potential("right", 10.0, name="drive")

solver.discretize(max_element_size=0.4e-3)
result = solver.solve(max_refinements=0)
print(f"nodes={result.mesh.info.nodes}, energy={result.energy:.6e} J/m2")
solver.visualize()
