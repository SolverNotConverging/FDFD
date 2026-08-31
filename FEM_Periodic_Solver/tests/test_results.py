from __future__ import annotations

import numpy as np
import pytest

from FEM_Periodic_Solver.results import (
    PeriodicMode,
    PeriodicModeSet,
    PeriodicSampledFields,
)


def _fields() -> PeriodicSampledFields:
    return PeriodicSampledFields(
        [[0.25, 0.25]],
        {"Ey": [1.0 + 2.0j], "Hx": [-0.5]},
        dimension=2,
        mesh_points=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        mesh_cells=[[0, 1, 2]],
        sample_element_indices=[0],
    )


def test_periodic_mode_derived_quantities_and_immutability() -> None:
    mode = PeriodicMode(
        neff=1.25 - 0.02j,
        k0=10.0,
        period=0.3,
        fields=_fields(),
        coefficients=[1.0, 2.0j],
    )
    assert mode.gamma == pytest.approx(0.2 + 12.5j)
    assert mode.bloch_multiplier == pytest.approx(np.exp(-mode.gamma * mode.period))
    assert -np.pi / mode.period <= mode.folded_beta.real < np.pi / mode.period
    with pytest.raises(ValueError):
        mode.coefficients.setflags(write=True)

    modes = PeriodicModeSet([mode], frequency=1e9, period=0.3, dimension=2)
    assert modes.mode(1) is mode
    assert modes.neff[0] == mode.neff
    assert modes.folded_neff[0] == mode.folded_neff
    assert modes.folded_beta[0] == mode.folded_beta
    assert modes.bloch_multiplier[0] == mode.bloch_multiplier
    assert modes.directions == ("indeterminate",)
