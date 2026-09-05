"""Fixed-mesh dielectric capacitor and charged 2D parallel-plate examples."""

from Electrostatic_Solver import ElectrostaticSolver, Interval, Rectangle, EPSILON_0
from matplotlib import pyplot as plt


def main():
    capacitor = ElectrostaticSolver(dim=1, domain=(0., 1.), outer_potential=None)
    capacitor.add_object(Interval((.5, 1.)), permittivity=4.)
    capacitor.set_potential("left", 0., name="ground")
    capacitor.set_potential("right", 1., name="drive")
    capacitor.discretize(max_element_size=.1)
    dielectric = capacitor.solve(max_refinements=0)
    print("Layered capacitor energy:", dielectric.energy)
    print("Expected energy:", .5 * EPSILON_0 / (.5 + .5 / 4.))

    charged = ElectrostaticSolver(dim=2, domain=((0., 1.), (0., .5)), outer_potential=None)
    charged.set_potential("left", 0.)
    charged.set_potential("right", 0.)
    charged.add_charge_density(Rectangle((0., 1.), (0., .5)), EPSILON_0)
    charged.discretize(max_element_size=.12)
    poisson = charged.solve(max_refinements=0)
    print("Charged guide peak potential (exact 0.125 V):", poisson.potential.max())
    capacitor.visualize(show=False)
    charged.visualize(show=False)
    plt.show()
    return dielectric, poisson


if __name__ == "__main__":
    main()
