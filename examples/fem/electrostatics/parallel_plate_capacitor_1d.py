"""Layered 1D FEM capacitor with geometry created before the mesh."""
from cem_common import Material, shapes

from fem_electrostatics import ElectrostaticSolver


def main():
    # Define geometry and relative permittivity before creating the mesh.
    dielectric = Material(name="dielectric slab", epsilon=6.0)
    solver = ElectrostaticSolver(dim=1, outer_potential=None, x_range=(0.0, 0.01))
    solver.add_geometry(name='dielectric', material=dielectric, shape=shapes.Interval(bounds=(0.004, 0.007)))
    solver.set_potential(potential=0.0, name='ground', geometry='left')
    solver.set_potential(potential=10.0, name='drive', geometry='right')

    # Potentials are in volts; max_element_size is in metres.
    solver.mesh(max_element_size=0.0004)
    result = solver.solve(max_refinements=0)
    print(f"nodes={len(result.mesh_data.coordinates)}, energy={result.energy:.6e} J/m2")
    solver.show()
    return result


if __name__ == "__main__":
    main()
