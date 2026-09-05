from __future__ import annotations
from cem_common import Material, SurfaceImpedance, materials, shapes

import numpy as np
import pytest

from fem_electrostatics import ElectrostaticSolver
from cem_common.shapes import Rectangle
from cem_common.errors import GeometryError, BackendCapabilityError
from fem_electrostatics.exceptions import NotDiscretizedError


def test_constructor_and_objects_do_not_discretize() -> None:
    solver = ElectrostaticSolver(dim=2, x_range=((0.0, 2.0), (-1.0, 1.0))[0], y_range=((0.0, 2.0), (-1.0, 1.0))[1])
    solver.add_geometry(name='high_dk', material=materials.Material(epsilon=((9.0, 0.0), (0.0, 1.0)), mu=1.0), shape=shapes.Rectangle(bounds=((0.5, 1.5), (-0.4, 0.4))))
    solver.set_potential(potential=2.0, name='left_electrode', geometry='left')

    assert solver.mesh_data is None
    assert solver.potential is None
    assert solver.geometry.revision == 2
    with pytest.raises(NotDiscretizedError):
        _ = solver.coordinates


def test_full_symmetric_permittivity_is_supported() -> None:
    solver = ElectrostaticSolver(dim=2)
    region = solver.add_geometry(material=materials.Material(epsilon=((4.0, 0.5), (0.5, 2.0)), mu=1.0), shape=shapes.Rectangle(bounds=((0.2, 0.8), (0.2, 0.8))))

    np.testing.assert_allclose(region.material.tensor('epsilon', dimension=2), [[4.0, 0.5], [0.5, 2.0]])


def test_non_positive_definite_permittivity_is_rejected() -> None:
    solver = ElectrostaticSolver(dim=2)
    with pytest.raises(BackendCapabilityError, match="positive-definite"):
        solver.add_geometry(material=materials.Material(epsilon=((1.0, 2.0), (2.0, 1.0)), mu=1.0), shape=shapes.Rectangle(bounds=((0.2, 0.8), (0.2, 0.8))))


def test_convenience_shapes_use_physical_bounds() -> None:
    solver = ElectrostaticSolver(x_range=20., y_range=10.)
    material = solver.add_rectangle(x_range=(2.0, 8.0), y_range=(3.0, 7.0), material=materials.Material(epsilon=((4.0, 0.0), (0.0, 1.0)), mu=1.0))
    electrode = solver.set_potential(potential=3.0, geometry=shapes.Rectangle(bounds=((9.0, 11.0), (1.0, 9.0))))

    assert material.shape.bounds == (2.0, 8.0, 3.0, 7.0)
    assert electrode.shape.bounds == (9.0, 11.0, 1.0, 9.0)
    assert solver.mesh_data is None

    with pytest.raises(GeometryError, match='shapes object'):
        solver.add_geometry(shape=(slice(2, 8), slice(3, 7)), material=materials.vacuum)
