"""TM plane-wave scattering by a dielectric cylinder, with explicit saving."""

# Run directly from the checkout without installing solver packages.
import sys as _sys
from pathlib import Path as _Path
_ROOT = next(parent for parent in _Path(__file__).resolve().parents
             if (parent / "fdfd_common" / "__init__.py").is_file())
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from pathlib import Path
from fdfd_common import Material
from fdfd_scattering import ScatteringSolver2D

OUTPUT = _ROOT / "fdfd_scattering/outputs/examples/dielectric_cylinder_2d"


def main():
    dielectric = Material(name="cylinder", epsilon=4.)
    solver = ScatteringSolver2D(frequency=3e9, x_range=(-.15,.15), y_range=(-.15,.15), polarization="TM")
    solver.add_circle(center=(0.,0.), radius=.025, material=dielectric)
    solver.add_pml(thickness=.05)
    solver.add_source(kind="plane_wave", angle=0., amplitude=1.)
    solver.set_source_region(inset=.075)
    solver.mesh(resolution=(120,120))
    result = solver.solve()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    result.save(OUTPUT / "scattering.h5")
    result.plot(component="Hz",quantity="magnitude").savefig(OUTPUT / "field.png",dpi=160)
    result.show()
    return result


if __name__ == "__main__":
    main()
