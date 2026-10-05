"""Track TE1 through cutoff between physical PEC plates, without PML."""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import csv
import numpy as np
from fdfd_waveguide_modes import ModeSolver1D
from fdfd_mode_tracking import materials, PortSpec, TrackingConfig, track_modes

OUTPUT = _ROOT / 'fdfd_mode_tracking/outputs/examples/parallel_plate_cutoff'
C = 1/np.sqrt(8.854187817e-12*4e-7*np.pi)

width, cells = 22.86e-3, 128
dx = width / cells

# track_modes calls this for each frequency and verification mesh.
def solver_for_frequency(frequency, verification):
    solver = ModeSolver1D(frequency=frequency, x_range=(-dx, width+dx))
    solver.add_layer(x_range=(-dx, 0.), material=materials.PEC, name='left plate')
    solver.add_layer(x_range=(width, width+dx), material=materials.PEC, name='right plate')
    solver.mesh(resolution=(cells+2)*verification.mesh_factor, subpixels=1)
    return solver

cutoff = C/(2*width)
sweep = track_modes(solver_for_frequency, cutoff*np.array([.75, .85, 1.15, 1.25]),
                    port=PortSpec(boundary='enclosed'),
                    config=TrackingConfig(num_candidates=2, max_candidates=4,
                        polarization='TE', neff_guess=-.8j, max_depth=6))
OUTPUT.mkdir(parents=True, exist_ok=True)
sweep.save(OUTPUT/'tracked_modes.h5')
profiles = sweep.export()
with (OUTPUT/'port_modes.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=('frequency_hz', 'beta_real', 'beta_imag',
                            'real_power_w_per_m', 'reactive_power_var_per_m', 'propagation'))
    writer.writeheader()
    for p in profiles:
        writer.writerow(dict(frequency_hz=p.frequency, beta_real=p.beta.real, beta_imag=p.beta.imag,
            real_power_w_per_m=p.complex_power.real, reactive_power_var_per_m=p.complex_power.imag,
            propagation=p.metadata['propagation']))
print(f'Tracked {len(sweep.samples)} samples using {sweep.metadata["eigensolves"]} solves.')
print(f'Cutoff/unresolved intervals (Hz): {sweep.unresolved_intervals}')
print(f'Port profiles: {OUTPUT}')
