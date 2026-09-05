FEM Waveguide Mode Solver
=========================

Part of **Computational Electromagnetics**, version 1.0.0.

Installation
------------

From the repository root, create the project environment and install the maintained packages::

    conda env create -f environment.yml
    conda activate cem
    python scripts/install_python.py

This family also builds as an independent wheel. Install its declared dependencies
along with its wheel; source packages use the standard ``src/`` layout.

Workflow
--------

.. code-block:: python

    from fem_waveguide_modes import ModeSolver2D, load_result

    solver = ModeSolver2D(frequency=10e9, x_range=22.86e-3,
                         y_range=10.16e-3, boundary="pec")
    solver.mesh(max_element_size=1e-3)
    result = solver.solve(num_modes=4, neff_guess=.8, max_refinements=0)
    result.save("outputs/modes.h5")
    loaded = load_result("outputs/modes.h5")
    figure = loaded.plot(component="Ey", mode=0)
    solver.show()

``solve()`` automatically meshes when needed and never saves or opens a window.
The adaptive defaults are two refinements and a relative tolerance of 0.05.
The example uses a fixed mesh for reproducibility. Geometry edits invalidate
``mesh_data`` and ``result``; an automatic rebuild reuses explicit mesh settings.

``show()`` uses the interactive Matplotlib mode viewer. ``plot()`` returns
a Figure without opening a GUI. Both dimensions support scalar and diagonal
relative materials; the 2D solver also supports surface-impedance conductors.

Electromagnetic fields use ``exp(+i*omega*t)`` and guided propagation
``exp(-i*beta*z)``. Passive constitutive values have nonpositive imaginary
parts; forward attenuation is ``-Im(beta)``.

Persistence and API
-------------------

Results use the ``cem-fem-results`` HDF5 envelope, schema ``1.0``. Old archives
are rejected. Loading supports inspection, plotting, and saving, not solver
restart or Python callback restoration. Mode indices are zero-based.

See `API_REFERENCE.rst <API_REFERENCE.rst>`_ for supported configuration,
results, units, defaults, and exceptions. Run bundled examples from the repository
root with ``python solvers/fem/waveguide_modes/examples/<example>.py``.
