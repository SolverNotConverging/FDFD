"""Parallel plates with uniform space charge and an analytic peak potential."""
from cem_common import shapes

from fem_electrostatics import ElectrostaticSolver
from scipy.constants import epsilon_0 as EPSILON_0


def main():
    charged = ElectrostaticSolver(dim=2, outer_potential=None, x_range=1., y_range=.5)
    charged.set_potential(potential=0.0, geometry='left')
    charged.set_potential(potential=0.0, geometry='right')
    charged.add_charge_density(density=EPSILON_0, geometry=shapes.Rectangle(bounds=((0.0, 1.0), (0.0, 0.5))))
    charged.mesh(max_element_size=.12)
    poisson = charged.solve(max_refinements=0)
    print("Charged guide peak potential (exact 0.125 V):", poisson.potential.max())
    charged.show()
    return poisson


if __name__ == "__main__":
    main()
