"""Fixed-mesh dielectric capacitor and charged 2D parallel-plate examples."""

from fem_electrostatics import ElectrostaticSolver, Rectangle
from scipy.constants import epsilon_0 as EPSILON_0
from matplotlib import pyplot as plt


def main():
    capacitor = ElectrostaticSolver(dim=1, outer_potential=None, x_range=(0.0, 1.0))
    capacitor.add_layer(x_range=(.5, 1.), epsilon=4.)
    capacitor.set_potential(region="left", potential=0., name="ground")
    capacitor.set_potential(region="right", potential=1., name="drive")
    capacitor.mesh(max_element_size=.1)
    dielectric = capacitor.solve(max_refinements=0)
    print("Layered capacitor energy:", dielectric.energy)
    print("Expected energy:", .5 * EPSILON_0 / (.5 + .5 / 4.))

    charged = ElectrostaticSolver(dim=2, outer_potential=None, x_range=1., y_range=.5)
    charged.set_potential(region="left", potential=0.)
    charged.set_potential(region="right", potential=0.)
    charged.add_charge_density(region=Rectangle((0., 1.), (0., .5)), density=EPSILON_0)
    charged.mesh(max_element_size=.12)
    poisson = charged.solve(max_refinements=0)
    print("Charged guide peak potential (exact 0.125 V):", poisson.potential.max())
    capacitor.show(block=False)
    charged.show(block=False)
    plt.show()
    return dielectric, poisson


if __name__ == "__main__":
    main()
