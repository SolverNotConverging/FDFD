FEM Waveguide Scattering Solver
===============================

Part of **Computational Electromagnetics**, version 1.0.0.

A **2.5D full-vector** scattered-field solver on a two-dimensional x/z mesh.
The invariant-direction factor is ``exp(-i*ky*y)``.

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

    from fem_waveguide_scattering import WaveguideScatteringSolver2D, load_result

    solver = WaveguideScatteringSolver2D(frequency=10e9, x_range=.04,
        z_range=(-.1, .1), transverse_boundary="pec")
    solver.add_pml(thickness=.03, direction="z")
    solver.mesh(max_element_size=.005)
    solver.solve_modes(num_modes=1, neff_guess=.9, max_refinements=0)
    solver.set_incident_mode(0)
    result = solver.solve(max_refinements=0)
    result.save("outputs/scattering.h5")
    loaded = load_result("outputs/scattering.h5")
    print(loaded.S11, loaded.S21)
    loaded.show()

``solve()`` automatically meshes when needed and never saves or opens a window.
The adaptive defaults are two refinements and a relative tolerance of 0.05.
The example uses a fixed mesh for reproducibility. Geometry edits invalidate
``mesh_data`` and ``result``; an automatic rebuild reuses explicit mesh settings.

``show()`` uses the separately built native viewer. Launch failures report
the executable discovery setting; saving and loading work without the viewer.

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
root with ``python solvers/fem/waveguide_scattering/examples/<example>.py``.
