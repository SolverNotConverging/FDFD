"""Adapt shared scenes to the existing FDFD cell/material interfaces."""
from itertools import product
import numpy as np
from . import materials
from .grid import fractions
from .errors import ConfigurationError, GeometryError


def populate(solver, backend, resolution, subpixels):
    all_slices = tuple(slice(0, n) for n in resolution)
    backend._apply_fractional_material(*materials.bulk_values(solver.background_material), np.ones(resolution), *all_slices)
    records = [r for r, _ in solver._objects.values()]
    for record in records:
        if isinstance(record.material, materials.Material):
            occupancy, slices = fractions(record.shape, solver._ranges, resolution, subpixels)
            if occupancy.size:
                backend._apply_fractional_material(*materials.bulk_values(record.material), occupancy, *slices)
    owner = np.zeros(resolution, dtype=bool)
    for record in records:
        material = record.material
        if isinstance(material, materials.Material):
            continue
        # Opaque objects select cell centres; a separate occupancy mask expands
        # onto the existing Yee traces. No fictitious large permittivity is used.
        occupancy, slices = fractions(record.shape, solver._ranges, resolution, 1)
        mask = np.zeros(resolution, dtype=bool)
        mask[slices] = occupancy.astype(bool)
        if not np.any(mask):
            raise GeometryError(f'Conductor {record.name!r} contains no grid cells; refine the mesh.')
        if np.any(mask & owner):
            raise GeometryError('Conductor objects overlap on the grid; combine them with Union or refine the mesh.')
        owner |= mask
        method = backend.add_pec if material == materials.PEC else backend.add_pmc if material == materials.PMC else backend.add_impedance_surface
        # Group contiguous x cells in each transverse row. Existing constraint
        # compilation retains its numerical behavior on rectangular objects.
        for index in product(*(range(n) for n in resolution[1:])):
            column = mask[(slice(None), *index)]
            transitions = np.diff(np.r_[False, column, False].astype(int))
            for start, stop in zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)):
                spans = ((int(start), int(stop)), *((int(i), int(i+1)) for i in index))
                kwargs = {axis+'_range': span for axis, span in zip(solver._physical_axes, spans)}
                if isinstance(material, materials.IdealBoundary):
                    method(**kwargs)
                else:
                    method(Zs=material.at_frequency(frequency=solver.frequency), **kwargs)


def apply_pml(solver, backend, resolution, spec):
    axes = tuple(a for a in solver._physical_axes if not solver._periodic or a != 'z')
    direction = spec['direction']
    selected = axes if direction == 'all' else (direction[0],)
    for axis in selected:
        i = solver._physical_axes.index(axis)
        lo, hi = solver._ranges[i]
        width = int(np.ceil(spec['thickness']*resolution[i]/(hi-lo)-1e-12))
        sides = (axis+'-', axis+'+') if direction in ('all', axis) else (direction,)
        if len(resolution) == 3:
            backend.add_UPML(sides=tuple(side[-1]+axis for side in sides), width=width, max_loss=spec['sigma_max'], n=spec['order'])
        else:
            backend.add_pml(pml_width=width, n=spec['order'], sigma_max=spec['sigma_max'], direction=axis if len(sides)==2 else sides[0])


def field_coordinates(solver, fields):
    result = {}
    for name, values in fields.items():
        axes = []
        for n, cells, (lo, hi) in zip(values.shape[:-1], solver.mesh_data.resolution, solver._ranges):
            if n == cells+1:
                axes.append(np.linspace(lo, hi, n))
            elif n == cells:
                axes.append(lo+(np.arange(n)+.5)*(hi-lo)/cells)
            else:
                raise ConfigurationError(f'Unexpected staggered shape for {name}: {values.shape}.')
        result[name] = tuple(axes)
    return result


def validate_solve(num_modes, neff_guess, eigensolver_tolerance):
    if isinstance(num_modes, bool) or int(num_modes) != num_modes or num_modes < 1:
        raise ConfigurationError('num_modes must be a positive integer.')
    if neff_guess is not None and not np.isfinite(complex(neff_guess)):
        raise ConfigurationError('neff_guess must be finite.')
    if not np.isfinite(eigensolver_tolerance) or eigensolver_tolerance < 0:
        raise ConfigurationError('eigensolver_tolerance must be finite and nonnegative.')
