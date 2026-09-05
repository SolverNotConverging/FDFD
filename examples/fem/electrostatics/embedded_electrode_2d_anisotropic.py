"""Anisotropic dielectric and internal electrode on a refined 2D FEM mesh."""
from cem_common import Material, materials, shapes

from fem_electrostatics import ElectrostaticSolver


def main():
    # All lengths are in metres. The outer boundary is grounded.
    substrate = Material(
        name="anisotropic substrate",
        epsilon=((9.0, 0.4), (0.4, 6.0)),
    )
    solver = ElectrostaticSolver(
        dim=2,
        outer_potential=0.0,
        x_range=(0.0, 0.02),
        y_range=(0.0, 0.01),
    )
    # The off-diagonal entries couple the x and y dielectric responses.
    solver.add_geometry(name='high_dk_substrate', material=substrate, shape=shapes.Rectangle(bounds=((0.006, 0.014), (0.002, 0.008))))
    electrode = solver.add_geometry(
        name="electrode",
        material=materials.PEC,
        shape=shapes.Circle(center=(0.01, 0.005), radius=0.001),
    )
    solver.set_potential(geometry=electrode, potential=100.0)

    # Resolve material interfaces and electrode boundaries on the initial mesh.
    mesh = solver.mesh(max_element_size=0.001, interface_refinement=0.65, boundary_refinement=0.4)
    # Zero refinements performs one solve; use the defaults for adaptive refinement.
    result = solver.solve(max_refinements=0)
    print(f"nodes={mesh.info.nodes}, triangles={mesh.info.elements}")
    print(f"electrode charge={result.conductor_charge('electrode'):.6e} C/m")
    solver.show()
    return result


if __name__ == "__main__":
    main()
