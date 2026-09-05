"""Dielectric rods in a periodic cell: material, geometry, grid, Bloch bands."""
from pathlib import Path
import numpy as np
from cem_common import Material
from fdfd_band_structure import BandStructureSolver2D

OUTPUT = Path(__file__).resolve().parents[3] / "outputs/examples/fdfd/band_structure/square_lattice_2d"


def main():
    a, b = .01, .01
    rod = Material(name="dielectric rod", epsilon=8.9)
    solver = BandStructureSolver2D(x_range=(-a/2, a/2), y_range=(-b/2, b/2))
    solver.add_circle(center=(0., 0.), radius=.2*min(a,b), material=rod)
    solver.mesh(resolution=(20, 20))
    path = solver.make_bloch_path(
        points=((0.,0.), (np.pi/a,0.), (np.pi/a,np.pi/b), (0.,0.)),
        num_points=16,
    )
    result = solver.solve(beta_path=path, num_modes=4)
    print("First TE band (Hz):", result.frequencies["TE"][0])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result.save(OUTPUT / "bands.h5")
    result.plot().savefig(OUTPUT / "bands.png", dpi=160)
    result.show()
    return result


if __name__ == "__main__":
    main()
