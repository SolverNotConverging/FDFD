"""Install the complete wheel outside the checkout and exercise solvers and apps."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import tempfile
import venv
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SMOKE = r'''
from importlib import import_module
from pathlib import Path
import sys
import numpy as np
import scipy.sparse as sp
from cem_common import Material, materials, shapes

packages = ('cem_common', 'fem_adaptivity', 'periodic_eigensolver',
    'fdfd_waveguide_modes', 'fdfd_periodic_modes', 'fdfd_band_structure', 'fdfd_scattering',
    'fem_waveguide_modes', 'fem_periodic_modes', 'fem_waveguide_scattering', 'fem_electrostatics')
for name in packages:
    module = import_module(name)
    assert module.__version__ == '1.0.0', name
    assert Path(module.__file__).is_relative_to(Path(sys.prefix)), module.__file__

from periodic_eigensolver import native_backend_available, solve_generalized
assert native_backend_available(), 'The release wheel must activate the native eigensolver.'
result = solve_generalized(sp.diags(np.arange(1., 21.), format='csc'), sp.eye(20, format='csc'),
    sigma=3.1, num_modes=2, backend='cython')
assert np.max(result.residuals) < 1e-8

import importlib.util
assert importlib.util.find_spec('fdfd_periodic_modes.refined_shift_invert_arnoldi') is None
assert importlib.util.find_spec('fem_common') is None

from fem_waveguide_modes import ModeSolver1D, ModeSolver2D, load_result
for solver in (ModeSolver1D(frequency=10e9, x_range=.02),
               ModeSolver2D(frequency=10e9, x_range=.02, y_range=.01)):
    solver.mesh(max_element_size=.004)
    result = solver.solve(num_modes=1, neff_guess=.66, max_refinements=0)
    result.save('modes.h5')
    loaded = load_result('modes.h5')
    np.testing.assert_allclose(loaded.neff, result.neff)
    loaded.plot(component='Ey').savefig('modes.png')

from fem_electrostatics import ElectrostaticSolver, load_result
for dimension in (1, 2):
    solver = ElectrostaticSolver(dim=dimension, x_range=1., outer_potential=None)
    solver.set_potential(geometry='left', potential=0.)
    solver.set_potential(geometry='right', potential=1.)
    result = solver.solve(max_refinements=0)
    np.testing.assert_allclose(result.potential, result.coordinates[:, 0], atol=1e-12)
    result.save('static.h5')
    load_result('static.h5').plot().savefig('static.png')

from fem_periodic_modes import PeriodicModeSolver2D, PeriodicModeSolver3D, load_result
for solver in (PeriodicModeSolver2D(frequency=10e9, x_range=.02, z_range=.005),
               PeriodicModeSolver3D(frequency=10e9, x_range=.02, y_range=.01, z_range=.005)):
    solver.mesh(max_element_size=.006)
    result = solver.solve(num_modes=1, neff_guess=.66, max_refinements=0, eigensolver='dense')
    result.save('periodic.h5')
    np.testing.assert_array_equal(load_result('periodic.h5').neff, result.neff)

from fem_waveguide_scattering import WaveguideScatteringSolver2D, load_result
solver = WaveguideScatteringSolver2D(frequency=299792458., x_range=.5, z_range=(-2., 2.), boundary=materials.PEC)
solver.add_pml(thickness=.5, direction='z')
solver.mesh(max_element_size=.1)
solver.solve_modes(num_modes=1, neff_guess=1., max_refinements=0)
solver.set_incident_mode(0)
result = solver.solve(max_refinements=0)
assert abs(result.S21-1) < 1e-8 and abs(result.S11) < 1e-8
result.save('scattering.h5')
np.testing.assert_array_equal(load_result('scattering.h5').E_total, result.E_total)

from fdfd_waveguide_modes import ModeSolver1D as GridModeSolver1D, load_result as load_grid_modes
grid = GridModeSolver1D(frequency=299792458., x_range=1.,
                        background_material=Material(name='fill', epsilon=2.25))
grid.add_geometry(shape=shapes.Interval(bounds=(0., .05)), material=materials.PEC)
grid.add_geometry(shape=shapes.Interval(bounds=(.95, 1.)), material=materials.PEC)
grid.mesh(resolution=20)
grid_result = grid.solve(num_modes=1, neff_guess=1.4, polarization='TE')
grid_result.save('grid-modes.h5')
np.testing.assert_allclose(load_grid_modes('grid-modes.h5').neff, grid_result.neff)

from fdfd_periodic_modes import PeriodicModeSolver2D as GridPeriodicModeSolver2D
periodic_grid = GridPeriodicModeSolver2D(frequency=299792458., x_range=1., z_range=.25,
    polarization='TM', background_material=Material(name='fill', epsilon=2.25))
periodic_grid.mesh(resolution=(8, 8))
assert np.isfinite(periodic_grid.solve(num_modes=1, neff_guess=1.4).neff).all()

from fdfd_band_structure import BandStructureSolver2D
band = BandStructureSolver2D(x_range=1., y_range=1.)
band.add_circle(center=(.5, .5), radius=.2, material=Material(name='rod', epsilon=4.))
band.mesh(resolution=(8, 8))
path = band.make_bloch_path(points=((0., 0.), (np.pi, 0.)), num_points=3)
assert band.solve(beta_path=path, num_modes=1, polarizations=('TE',)).frequencies['TE'].shape == (1, 3)

from fdfd_scattering import ScatteringSolver2D as GridScatteringSolver2D
scatter = GridScatteringSolver2D(frequency=299792458., x_range=(-1., 1.), y_range=(-1., 1.))
scatter.add_circle(center=(0., 0.), radius=.2, material=Material(name='rod', epsilon=2.))
scatter.add_pml(thickness=.2)
scatter.mesh(resolution=(20, 20))
scatter.add_source(angle=0.)
scatter.set_source_region(inset=.3)
assert np.isfinite(scatter.solve().fields['Ez']).all()

print('Installed distributions, native eigensolver, all solver families, physics, and archives: PASS')

import fdfd
from importlib.metadata import distribution
assert Path(fdfd.__file__).is_relative_to(Path(sys.prefix))
installed = distribution('fdfd')
assert not any(requirement.startswith(('cem-common', 'fem-', 'fdfd-', 'periodic-eigensolver')) for requirement in installed.requires)
from cem_common._native import bundled_executable
from fem_waveguide_scattering.viewer import find_viewer_executable
from fem_periodic_modes.persistence import _viewer_candidates
assert find_viewer_executable() == bundled_executable('fem-waveguide-scattering-viewer')
assert _viewer_candidates('fem-periodic-mode-viewer.exe')[0] == bundled_executable('fem-periodic-mode-viewer')
'''

NATIVE_SMOKE = r'''
import os
import subprocess
from pathlib import Path
import sys
import fdfd
from cem_common._native import bundled_executable, bundled_environment

native = Path(fdfd.__file__).parent / 'native'
for name, arguments in (
    ('transmission-line-calculator', ['--calculate-smoke-test']),
    ('transmission-line-calculator-cli', ['--smoke-test']),
    ('fem-periodic-mode-viewer', ['--smoke-test', str(native / 'samples/periodic-3d.h5')]),
    ('fem-waveguide-scattering-viewer', ['--smoke-test', str(native / 'samples/scattering-sweep.h5')]),
    ('fem-periodic-mode-inspect', [str(native / 'samples/periodic-sweep.h5'), '1', '0']),
    ('fem-waveguide-scattering-viewer-inspect', [str(native / 'samples/scattering-sweep.h5'), '1']),
):
    exe = bundled_executable(name)
    env = bundled_environment(exe)
    env['PATH'] = str(exe.parent) + os.pathsep + str(Path(os.environ['SystemRoot']) / 'System32')
    env['QT_QPA_PLATFORM'] = 'offscreen'
    subprocess.run([str(exe), *arguments], env=env, check=True, timeout=90,
                   creationflags=subprocess.CREATE_NO_WINDOW)
subprocess.run([sys.executable, '-I', '-m', 'fdfd', 'info'], check=True)
subprocess.run([str(Path(sys.prefix) / 'Scripts/transmission-line-calculator-cli.exe'), '--smoke-test'], check=True)
print('Bundled native applications and installed command entry points: PASS')
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dist', type=Path, default=ROOT/'outputs/dist')
    parser.add_argument('--fresh', action='store_true', help='Download dependencies into a clean environment.')
    args = parser.parse_args()
    wheels = sorted(args.dist.resolve().glob('*.whl'))
    if len(wheels) != 1 or wheels[0].name != 'fdfd-1.0.0-cp312-cp312-win_amd64.whl':
        raise SystemExit(f'Expected the one complete FDFD Windows 3.12 wheel, found {wheels}.')
    native = wheels[0]
    with zipfile.ZipFile(native) as archive:
        if not any(name.endswith(('.pyd', '.so')) for name in archive.namelist()):
            raise SystemExit('The periodic eigensolver wheel lacks its compiled extension.')
    with tempfile.TemporaryDirectory(prefix='fdfd-wheel-qualification-') as temporary:
        work = Path(temporary)
        venv.EnvBuilder(with_pip=True, system_site_packages=not args.fresh).create(work/'env')
        python = work/'env'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
        for wheel in wheels:
            subprocess.run([str(python), '-I', '-m', 'pip', 'install',
                            *([] if args.fresh else ['--no-index', '--no-deps']), str(wheel)], cwd=work, check=True)
        subprocess.run([str(python), '-I', '-m', 'pip', 'check'], cwd=work, check=True)
        subprocess.run([str(python), '-I', '-c', SMOKE], cwd=work, check=True)
        subprocess.run([str(python), '-I', '-c', NATIVE_SMOKE], cwd=work, check=True)
    print('Wheel qualification passed outside the checkout.')


if __name__=='__main__':main()
