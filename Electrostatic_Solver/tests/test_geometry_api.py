from __future__ import annotations

import numpy as np
import pytest

from Electrostatic_Solver import (
    ElectrostaticSolver,
    GeometryError,
    NotDiscretizedError,
    Rectangle,
)


def test_constructor_and_objects_do_not_discretize() -> None:
    solver = ElectrostaticSolver(dim=2, domain=((0.0, 2.0), (-1.0, 1.0)))
    solver.add_object(Rectangle((0.5, 1.5), (-0.4, 0.4)), erxx=9.0, name="high_dk")
    solver.set_potential("left", 2.0, name="left_electrode")

    assert solver.mesh is None
    assert solver.potential is None
    assert solver.geometry.revision == 2
    with pytest.raises(NotDiscretizedError):
        _ = solver.coordinates


def test_full_symmetric_permittivity_is_supported() -> None:
    solver = ElectrostaticSolver(dim=2)
    region = solver.add_object(
        Rectangle((0.2, 0.8), (0.2, 0.8)),
        permittivity=((4.0, 0.5), (0.5, 2.0)),
    )

    np.testing.assert_allclose(region.permittivity.array, [[4.0, 0.5], [0.5, 2.0]])


def test_non_positive_definite_permittivity_is_rejected() -> None:
    solver = ElectrostaticSolver(dim=2)
    with pytest.raises(GeometryError, match="positive definite"):
        solver.add_object(
            Rectangle((0.2, 0.8), (0.2, 0.8)),
            permittivity=((1.0, 2.0), (2.0, 1.0)),
        )


def test_legacy_slices_are_continuous_geometry() -> None:
    solver = ElectrostaticSolver(mesh_size=(20, 10), dim=2)
    material = solver.add_object((slice(2, 8), slice(3, 7)), erxx=4.0)
    electrode = solver.set_potential((slice(9, 11), slice(1, 9)), 3.0)

    assert material.shape.bounds == (2.0, 8.0, 3.0, 7.0)
    assert electrode.shape.bounds == (9.0, 11.0, 1.0, 9.0)
    assert solver.mesh is None
