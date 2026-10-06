"""Grounded-slab leaky-wave cell, matching the validated FEM example."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from fdfd_periodic_modes import Material, materials, PeriodicModeSolver2D

OUTPUT = _ROOT / "fdfd_periodic_modes/outputs/examples/2d_surface_wave_antenna"

substrate = Material(name="antenna substrate", epsilon=10.2)
solver = PeriodicModeSolver2D(frequency=20e9, x_range=(0., 10e-3),
    z_range=(0., 8e-3), polarization="TM", boundary=materials.PEC)
solver.add_rectangle(x_range=(0., 1.27e-3), z_range=(0., 8e-3),
    material=substrate, name="grounded_dielectric_slab")
solver.add_rectangle(x_range=(1.27e-3, 1.37e-3), z_range=(1e-3, 2e-3),
    material=materials.PEC, name="top_pec_perturbation")
solver.add_pml(thickness=2.5e-3, direction="x+")
# Resolve the 50 um PEC patch and slab interfaces; the benchmark shows refinement.
solver.mesh(resolution=(100, 80))
result = solver.solve(num_modes=4, neff_guess=0., eigensolver_tolerance=1e-9, ncv=36)
print("Bloch effective indices:", result.neff)
print("Maxwell residuals:", result.solve_info["residuals"])
OUTPUT.mkdir(parents=True, exist_ok=True)
result.save(OUTPUT / "modes.h5")
result.show()
