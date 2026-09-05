"""Render the deliberately selected material-first FDFD user API."""
from importlib import import_module
import json
from pathlib import Path
from generate_api_reference import entry, section, DESCRIPTIONS, TYPES

ROOT=Path(__file__).resolve().parents[1]
INVENTORY={
 'fdfd_waveguide_modes': {
  'ModeSolver1D': ['__init__','add_geometry','add_layer','set_material','set_shape','add_pml','remove','mesh','solve','show'],
  'ModeSolver2D': ['__init__','add_geometry','add_rectangle','add_circle','add_polygon','set_material','set_shape','add_pml','remove','mesh','solve','show']},
 'fdfd_periodic_modes': {
  'PeriodicModeSolver2D': ['__init__','add_geometry','add_rectangle','add_circle','add_polygon','set_material','set_shape','add_pml','remove','mesh','solve','show'],
  'PeriodicModeSolver3D': ['__init__','add_geometry','add_box','add_sphere','add_cylinder','set_material','set_shape','add_pml','remove','mesh','solve','show']},
 'fdfd_band_structure': {
  'BandStructureSolver2D': ['__init__','add_geometry','add_rectangle','add_circle','add_polygon','set_material','set_shape','remove','mesh','make_bloch_path','solve','show']},
 'fdfd_scattering': {
  'ScatteringSolver2D': ['__init__','add_geometry','add_rectangle','add_circle','add_polygon','set_material','set_shape','add_pml','remove','mesh','add_source','set_source_region','solve','show']},
}
RESULT_TYPES = {
    'fdfd_waveguide_modes': 'ModeSet',
    'fdfd_periodic_modes': 'PeriodicModeSet',
    'fdfd_scattering': 'ScatteringResult',
    'fdfd_band_structure': 'BandStructureResult',
}


def main():
    DESCRIPTIONS.update({
        'background_material':'Predefined bulk Material assigned to unfilled grid cells.',
        'material':'Predefined Material, PEC/PMC, or supported SIBC assignment.',
        'shape':'Continuous cem_common shape expressed in metres.',
        'resolution':'Positive cell count for each physical axis.',
        'neff_guess':'Dimensionless complex effective-index search target.',
        'eigensolver_tolerance':'Algebraic eigensolver convergence tolerance.',
        'thickness':'PML thickness in metres.', 'order':'Polynomial PML order.',
        'sigma_max':'Maximum PML-strength magnitude.',
        'subpixels':'Number of subcell samples used to average region material values.',
        'mode':'Zero-based mode or band index.',
        'kernel_backend':'Refined-kernel backend: auto, numpy, or cython.',
        'ncv':'Arnoldi subspace size; None selects the backend default.',
        'max_restarts':'Maximum refined Arnoldi restarts.', 'random_seed':'Deterministic initial-vector seed.',
        'beta_path':'Bloch vectors with shape (2, samples), in radians per metre.',
        'polarizations':'Requested TE/TM polarization names.',
        'eigenvalue_guess':'Frequency-eigenvalue spectral shift.',
        'inset':'Physical inset of the FDFD total-field source region, in metres.',
        'kind':'Source kind: plane_wave or point.',
        'location':'Physical point-source position in metres.',
    })
    TYPES.update({key:'int' for key in ('Nx','Ny','Nz','n','pml_width','width','num_modes','num_bands','subpixels')})
    for package,solvers in INVENTORY.items():
        module=import_module(package)
        family=package.removeprefix('fdfd_')
        out=section(package+' user API','=')
        out+='Version 1.0.0. This reference covers the deliberately supported user API.\nAll Python solvers use the same material-first ``mesh()``, ``solve()``, and\n``show()`` lifecycle. Phasors use exp(+i omega t); passive relative materials\nhave nonpositive imaginary values.\n\n'
        out+=section('Configuration and units')
        out+='Constructor extents and shape coordinates use metres; frequencies use hertz.\n``mesh(resolution=...)`` gives Yee-cell counts, while ``max_element_size`` is a\nphysical grid-spacing limit. Define reusable ``cem_common.Material`` and shape\nobjects before assigning them. Grid-index geometry is private backend detail.\nAll plotting and selection indices are zero-based.\n\n'
        for clsname,methods in solvers.items():
            cls=getattr(module,clsname)
            for method in methods:
                for axis in ('x','y','z'):
                    DESCRIPTIONS[axis+'_range'] = f'Physical {axis} extent or increasing bounds in metres.'
                    TYPES[axis+'_range'] = 'float | tuple[float, float] / m'
                label=clsname if method=='__init__' else clsname+'.'+method
                target=cls if method=='__init__' else getattr(cls,method)
                returned='a configured solver' if method=='__init__' else 'the documented data or None when storing state on the solver'
                if method=='solve':returned='a typed result stored on solver.result'
                if method=='mesh':returned='the initial GridData stored on solver.mesh_data'
                if method=='show':returned='the interactive Matplotlib figure'
                out+=entry(label,target,returned)
        result_name = RESULT_TYPES[package]
        result_type = getattr(module, result_name)
        out += section('Returned result')
        out += ('Result objects come from ``solve()`` or ``load_result()``; users do not\n'
                'construct them directly. Field results expose ``mesh_data``, ``metadata``,\n'
                '``solve_info``, and explicit physical field coordinates.\n\n')
        for method, returned in (
            ('plot', 'a Matplotlib Figure without opening a window'),
            ('show', 'the interactive Matplotlib Figure'),
            ('save', 'the atomically written HDF5 path'),
        ):
            out += entry(f'{result_name}.{method}', getattr(result_type, method), returned)
        out += entry('load_result', module.load_result, f'a typed ``{result_name}`` without solving')
        out+=section('Results and examples')
        if family=='band_structure':out+='``solve`` returns ``BandStructureResult`` with frequency arrays in hertz and\neigenvalues indexed by TE/TM polarization.\n\n'
        elif family=='scattering':out+='``solve`` returns ``ScatteringResult`` with scalar total fields at their\nphysical Yee-grid locations.\n\n'
        else:out+='``solve`` returns a modal set with dimensionless ``neff``, ``beta`` in rad/m,\nexplicit staggered field coordinates, and zero-based mode selection.\n\n'
        out+='Results provide ``plot()``, ``show()``, and atomic ``save()``; each package\nexports ``load_result()``. Invalid dimensions, materials, and controls raise\nactionable ``cem_common`` exceptions. See the `user guide <guide.rst>`_ and\nroot examples. Assembly routines, matrix builders, grid-index records, and\nArnoldi kernels are excluded from this user reference.\n'
        (ROOT/'doc/solvers/fdfd'/family/'API_REFERENCE.rst').write_text(out,encoding='utf-8')
    (ROOT/'doc/fdfd_public_api.json').write_text(json.dumps(INVENTORY,indent=2)+'\n')


if __name__=='__main__':main()
