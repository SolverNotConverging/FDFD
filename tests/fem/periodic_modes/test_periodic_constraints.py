from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import diags

from fem_periodic_modes.periodic import (
    build_node_prolongation,
    build_signed_edge_prolongation,
)


def test_node_prolongation_reduces_all_matrix_coefficients() -> None:
    prolongation = build_node_prolongation(
        6,
        slave_nodes=[4, 5],
        master_nodes=[0, 1],
        constrained_nodes=[2],
    )
    assert prolongation.matrix.shape == (6, 3)
    reduced = prolongation.reduce_matrix(diags(np.arange(1.0, 7.0)))
    assert reduced.shape == (3, 3)
    expanded = prolongation.expand(np.asarray([2.0, 3.0, 4.0], complex))
    assert expanded[4] == expanded[0]
    assert expanded[5] == expanded[1]
    assert expanded[2] == 0.0
    assert prolongation.equality_error(expanded) < 1e-15


def test_signed_edge_mapping_uses_actual_master_orientation() -> None:
    # Edge 1 is the reversed periodic copy of edge 0.  Edge 2 touches one
    # slave node but leaves the trace and must remain independent from edge 3.
    edges = np.asarray(
        [
            [0, 1],  # actual master orientation
            [3, 2],  # maps to (1, 0), hence sign -1
            [2, 4],  # one slave endpoint plus an interior endpoint
            [0, 4],
        ],
        dtype=np.int64,
    )
    prolongation = build_signed_edge_prolongation(
        edges,
        slave_nodes=[2, 3],
        master_nodes=[0, 1],
        node_count=5,
    )
    assert prolongation.matrix.shape == (4, 3)
    coefficients = prolongation.expand(np.asarray([2.0, 5.0, 7.0], complex))
    assert coefficients[1] == -coefficients[0]
    assert coefficients[2] != coefficients[3]
    assert prolongation.representatives[2] == 2
    assert prolongation.representatives[3] == 3


def test_trace_edge_requires_a_corresponding_master_edge() -> None:
    with pytest.raises(ValueError, match="no corresponding master edge"):
        build_signed_edge_prolongation(
            np.asarray([[2, 3], [0, 4], [1, 4]]),
            slave_nodes=[2, 3],
            master_nodes=[0, 1],
            node_count=5,
        )
