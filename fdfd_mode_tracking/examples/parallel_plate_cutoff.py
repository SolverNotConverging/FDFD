"""Track TE1 through cutoff between physical PEC plates, without PML."""

# Run directly from the checkout without installing solver packages.
import sys as _sys
from pathlib import Path as _Path
_ROOT = next(parent for parent in _Path(__file__).resolve().parents
             if (parent / "fdfd_common" / "__init__.py").is_file())
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import csv
from pathlib import Path
import numpy as np
from fdfd_common import materials
from fdfd_waveguide_modes import ModeSolver1D
from fdfd_mode_tracking import PortSpec, TrackingConfig, track_modes

OUTPUT = _ROOT / 'fdfd_mode_tracking/outputs/examples/parallel_plate_cutoff'
C = 1/np.sqrt(8.854187817e-12*4e-7*np.pi)


def plate_factory(width=.02286, cells=128):
    """Keep physical plates and outer bounds fixed during mesh verification."""
    dx = width/cells
    def factory(frequency, verification):
        if verification.padding_fraction:
            raise ValueError('The plate boundaries are physical and cannot be padded.')
        solver = ModeSolver1D(frequency=frequency, x_range=(-dx, width+dx))
        solver.add_layer(x_range=(-dx, 0.), material=materials.PEC, name='left plate')
        solver.add_layer(x_range=(width, width+dx), material=materials.PEC, name='right plate')
        solver.mesh(resolution=(cells+2)*verification.mesh_factor, subpixels=1)
        return solver
    return factory


def main():
    width = .02286
    cutoff = C/(2*width)
    sweep = track_modes(plate_factory(width), cutoff*np.array([.75, .85, 1.15, 1.25]),
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
    return sweep


if __name__ == '__main__':
    main()
