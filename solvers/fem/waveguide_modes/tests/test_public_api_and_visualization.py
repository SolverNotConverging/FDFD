"""Public-package and backend-neutral visualization integration tests."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg", force=True)

import fem_waveguide_modes as fem
from fem_waveguide_modes.visualization import visualize
from fem_common import NoResultError


def test_top_level_api_and_results_are_read_only() -> None:
    expected_exports = {
        "Material",
        "Mode",
        "ModeSet",
        "ModeSolver1D",
        "ModeSolver2D",
        "SampledFields",
        "load_result",
    }
    assert expected_exports <= set(fem.__all__)

    solver = fem.ModeSolver1D(frequency=20000000000.0, x_range=0.02)
    assert solver.result is None
    mesh = solver.mesh(resolution=24)
    assert solver.mesh_data is mesh
    assert solver.mesh_data is mesh
    assert solver.native_mesh is mesh.mesh
    modes = solver.solve(max_refinements=0, num_modes=1)

    assert solver.result is modes
    assert modes.mode(0) is modes[0]
    assert modes[0].index == 0
    assert not modes.neff.flags.writeable
    assert not modes[0].fields.x.flags.writeable
    assert not modes[0].component("Ex").flags.writeable
    with pytest.raises(ValueError):
        modes[0].component("Ex").setflags(write=True)
    with pytest.raises(ValueError):
        modes[0].fields.mesh_cells.setflags(write=True)  # type: ignore[union-attr]
    with pytest.raises(ValueError):
        modes.neff.setflags(write=True)
    with pytest.raises(ValueError):
        modes.neff[0] = 0.0
    with pytest.raises(TypeError):
        modes[0].fields.values["Ex"] = np.zeros_like(modes[0].component("Ex"))


def test_1d_static_and_gui_visualizers_render_with_shared_api(monkeypatch) -> None:
    solver = fem.ModeSolver1D(frequency=20000000000.0, x_range=0.02)
    solver.mesh(resolution=20)
    solver.solve(max_refinements=0, num_modes=1)

    figure, axes = visualize(solver.result,
        mode=0,
        components=("Ex", "Hy"),
        quantity="real",
        mesh=True,
        show=False,
    )
    assert axes.shape == (2,)
    figure.canvas.draw()

    captured: list[object] = []
    marker = object()
    monkeypatch.setattr(
        "fem_waveguide_modes.visualization.visualize_with_gui",
        lambda source, **kwargs: captured.append(source) or marker,
    )
    assert solver.show() is marker
    assert captured == [solver.result]

    import matplotlib.pyplot as plt

    plt.close(figure)


def test_1d_material_only_view_masks_unit_index_and_has_own_scale() -> None:
    from matplotlib.collections import QuadMesh

    fields = fem.SampledFields(
        np.asarray((0.5, 1.5, 2.5)),
        {"Ey": np.asarray((1.0, 0.5, -0.25), dtype=np.complex128)},
        dimension=1,
        material=np.asarray((1.0, 4.0, 1.0), dtype=np.complex128),
        metadata={
            "x_nodes": np.asarray((0.0, 1.0, 2.0, 3.0)),
            "material_index": np.asarray((1.0, 2.0, 1.0)),
        },
    )
    mode = fem.Mode(neff=1.5, beta=2.0, fields=fields, index=1)
    figure, axes = visualize(mode, component="Ey", field=False, show=False)
    figure.canvas.draw()

    assert not axes[0].lines
    material_artists = [item for item in axes[0].collections if isinstance(item, QuadMesh)]
    assert len(material_artists) == 1
    assert material_artists[0].cmap.name == "coolwarm"
    assert np.ma.count_masked(material_artists[0].get_array()) == 2
    assert any(axis.get_ylabel() == r"material $|n_{eff}|$" for axis in figure.axes)

    import matplotlib.pyplot as plt

    plt.close(figure)


@pytest.mark.gmsh
def test_2d_static_and_gui_visualizers_accept_fem_quadrature_samples(
    monkeypatch,
) -> None:
    solver = fem.ModeSolver2D(frequency=299792458.0, x_range=1.0, y_range=0.5)
    with pytest.raises(NoResultError, match=r"solve\(\)"):
        solver.show()
    mesh = solver.mesh(resolution=(6, 3))
    assert solver.mesh_data is mesh
    assert solver.mesh_data is mesh
    assert solver.native_mesh is mesh.mesh
    solver.solve(max_refinements=0, num_modes=1, neff_guess=np.sqrt(0.75))

    figure, axes = visualize(solver.result,
        mode=0,
        component="Ey",
        quantity="magnitude",
        mesh_overlay=True,
        show=False,
    )
    assert axes.shape == (1,)
    figure.canvas.draw()

    captured: list[object] = []
    marker = object()
    monkeypatch.setattr(
        "fem_waveguide_modes.visualization.visualize_with_gui",
        lambda source, **kwargs: captured.append(source) or marker,
    )
    assert solver.show() is marker
    assert captured == [solver.result]

    import matplotlib.pyplot as plt

    plt.close(figure)
