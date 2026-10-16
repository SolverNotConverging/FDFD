fdfd
====

Finite-difference frequency-domain (FDFD) electromagnetic solvers on Yee
grids, with one workflow for every problem type:

.. code-block:: python

   import fdfd

   solver = fdfd.ModeSolver2D(frequency=100e9, x_range=(0, 10e-3), y_range=(0, 10e-3))
   solver.add_circle(center=(5e-3, 5e-3), radius=3e-3, material=fdfd.Material(name="core", epsilon=6))
   solver.mesh(resolution=(50, 50))
   result = solver.solve(num_modes=4, device="auto")
   result.plot(mode=1)
   result.save("modes.h5")

.. list-table:: Solvers
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Problem
   * - :class:`~fdfd.ModeSolver1D`, :class:`~fdfd.ModeSolver2D`
     - Guided modes of a z-invariant waveguide (slab or cross-section)
   * - :class:`~fdfd.PeriodicModeSolver2D`, :class:`~fdfd.PeriodicModeSolver3D`
     - Complex Bloch modes of a structure periodic along z (leaky-wave antennas)
   * - :class:`~fdfd.BandStructureSolver2D`
     - Photonic band structure of a cell periodic in x and y
   * - :class:`~fdfd.ScatteringSolver2D`
     - Plane-wave or line-source scattering (TF/SF)
   * - :class:`~fdfd.ModeTracker1D`, :class:`~fdfd.ModeTracker2D`
     - Waveguide modes followed across a frequency sweep, with verification

.. toctree::
   :maxdepth: 2
   :caption: Guide

   getting_started
   conventions
   capabilities
   acceleration
   benchmarks/cuda_vs_scipy

.. toctree::
   :maxdepth: 2
   :caption: Solvers

   solvers/waveguide
   solvers/periodic
   solvers/bands
   solvers/scattering
   solvers/tracking

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
   changelog
