"""Render the deliberately selected FDFD API, retaining its numerical workflow."""
from importlib import import_module
import json
from pathlib import Path
from generate_api_reference import entry, section, DESCRIPTIONS, TYPES

ROOT=Path(__file__).resolve().parents[1]
INVENTORY={
 'fdfd_waveguide_modes': {
  'ModeSolver1D': ['__init__','add_layer','add_pec','add_pmc','add_pml','add_impedance_surface','solve','visualize','visualize_with_gui'],
  'ModeSolver2D': ['__init__','add_rectangle','add_circle','add_triangle','add_pec','add_pmc','add_pml','add_impedance_surface','solve','visualize','visualize_with_gui']},
 'fdfd_periodic_modes': {
  'PeriodicModeSolver2D': ['__init__','add_rectangle','add_pec','add_pmc','add_pml','solve','visualize_with_gui'],
  'PeriodicModeSolver3D': ['__init__','add_block','add_pec','add_pmc','add_UPML','solve','plot','plot_field_plane','visualize_with_gui','save_results','load_results']},
 'fdfd_band_structure': {
  'BandStructureSolver2D': ['__init__','add_object','add_circular_inclusion','default_rectangular_lattice_path','generate_bloch_path','compute_band_structure','set_tick_labels','plot_band_diagram']},
 'fdfd_scattering': {
  'ScatteringSolver2D': ['__init__','add_object','add_source','add_UPML','add_mask','solve_total_field_TE','solve_total_field_TM','TE_Visualization','TM_Visualization']},
}


def main():
    DESCRIPTIONS.update({
        'Nx':'Number of Yee cells along x; positive integer.', 'Ny':'Number of Yee cells along y; positive integer.',
        'Nz':'Number of Yee cells along z; positive integer.',
        'guess':'Spectral shift for the existing FDFD eigenproblem; see the solver example.',
        'sigma':'Optional spectral shift overriding the stored eigenproblem target.',
        'tol':'Algebraic eigenproblem tolerance; zero retains the existing backend convention.',
        'pml_width':'PML thickness in Yee cells.', 'width':'PML thickness in Yee cells.',
        'n':'Polynomial PML order.', 'max_loss':'Maximum PML stretch loss magnitude.',
        'subpixels':'Number of subcell samples used to average region material values.',
        'mode':'Visualization mode index; waveguide visualize uses 1-based selection, 3D periodic plot uses 0-based selection.',
        'mode_index':'Zero-based stored periodic mode index.',
        'kernel_backend':'Refined-kernel backend: auto, numpy, or cython.',
        'ncv':'Arnoldi subspace size; None selects the backend default.',
        'max_restarts':'Maximum refined Arnoldi restarts.', 'random_seed':'Deterministic initial-vector seed.',
        'region_mask':'Boolean array identifying cell centres in the object.',
        'er_tensor':'Relative diagonal electric constitutive tensor.', 'mr_tensor':'Relative diagonal magnetic constitutive tensor.',
        'a':'Unit-cell x period in metres.', 'b':'Unit-cell y period in metres; None uses a.',
        'num_bands':'Positive number of frequency bands to compute.',
        'beta_path':'Bloch vectors with shape (2, samples), in radians per metre.',
        'polarisations':'Requested TE/TM polarization names.',
        'eig_sigma':'Frequency eigenproblem shift.',
        'include_eigenvectors':'Include large raw eigenvector arrays in the NPZ archive.',
        'compressed':'Compress NPZ datasets.',
    })
    TYPES.update({key:'int' for key in ('Nx','Ny','Nz','n','pml_width','width','num_modes','num_bands','subpixels')})
    for package,solvers in INVENTORY.items():
        module=import_module(package)
        family=package.removeprefix('fdfd_')
        out=section(package+' user API','=')
        out+='Version 1.0.0. These FDFD implementations retain their existing numerical\nworkflow. The uniform mesh/solve/show contract applies to the FEM families.\nPhasors use exp(+i omega t); passive relative materials have nonpositive\nimaginary values.\n\n'
        out+=section('Configuration and units')
        out+='Constructor extents and frequencies use metres and hertz. Nx/Ny/Nz are\nYee cell counts. Geometry range helpers distinguish integer grid-index bounds\nfrom floating-point physical positions in metres; slices select grid indices.\nBand-structure shapes use physical coordinates. Materials are relative\ndiagonal values. Mode normalization, field locations, and existing selectors\nare preserved by this release.\n\n'
        for clsname,methods in solvers.items():
            cls=getattr(module,clsname)
            for method in methods:
                for axis in ('x','y','z'):
                    DESCRIPTIONS[axis+'_range'] = f'Physical {axis} extent in metres.' if method=='__init__' else f'Range along {axis}: floating-point positions in metres, integer grid indices, or an index slice.'
                    TYPES[axis+'_range'] = 'float / m' if method=='__init__' else 'tuple[float, float] | tuple[int, int] | slice'
                label=clsname if method=='__init__' else clsname+'.'+method
                target=cls if method=='__init__' else getattr(cls,method)
                returned='a configured solver' if method=='__init__' else 'the documented data or None when storing state on the solver'
                if method=='compute_band_structure':returned='a BandStructureResult with frequencies and eigenvalues by polarization'
                if method=='load_results':returned='a stored periodic solver result for inspection'
                out+=entry(label,target,returned)
        if family=='waveguide_modes':
            out+=entry('good_conductor_surface_impedance',module.good_conductor_surface_impedance,'surface impedance in ohms')
            out+='``METAL_RESISTIVITIES_OHM_M`` contains the supported metal presets.\n\n'
        out+=section('Results and examples')
        if family=='band_structure':out+='``compute_band_structure`` returns frequency arrays in Hz and eigenvalues,\nindexed by TE/TM polarization. Use ``plot_band_diagram`` to display them.\n\n'
        elif family=='scattering':out+='TE/TM solves retain sampled fields on the solver. Geometry and sources\nare configured before solving; field arrays follow the Yee-grid locations.\n\n'
        else:out+='Solves store effective indices and field arrays on the solver. ``neff`` is\ndimensionless; attenuation follows -Im(neff), multiplied by the free-space\nwavenumber for inverse metres. The 1D waveguide implementation separates\nTE and TM arrays. Consult the bundled example for the corresponding viewer.\n\n'
        out+='Invalid dimensions, materials, and solver controls raise ValueError or\nNotImplementedError. Numerical backend failures remain visible.\n\nRun the examples with the installed package; no repository path changes\nare required. See `README.rst <README.rst>`_ and the ``examples/`` directory.\nAssembly routines, matrix builders, and Arnoldi kernels are implementation\ndetails and are excluded from this reference.\n'
        (ROOT/'solvers/fdfd'/family/'API_REFERENCE.rst').write_text(out,encoding='utf-8')
    (ROOT/'docs/fdfd_public_api.json').write_text(json.dumps(INVENTORY,indent=2)+'\n')


if __name__=='__main__':main()
