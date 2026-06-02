Band Diagram Solver
===================

``BandDiagramSolver2D`` computes two-dimensional photonic-crystal band diagrams for rectangular unit cells. It samples a Bloch path through reciprocal space and solves TE and/or TM sparse eigenproblems at each Bloch vector.

What It Solves
--------------

Use this solver for 2D periodic lattices where the desired output is normalized frequency versus Bloch wave vector. The class includes helper-grid material assignment and plotting utilities for the unit cell, reciprocal path, and band diagram.

Main Classes
------------

.. code-block:: python

   BandDiagramSolver2D(a, Nx, Ny=None, b=None, background_er=1.0, background_ur=1.0, boundary_conditions=(1, 1))
   BandStructureResult

Parameters:

* ``a``: period along ``x``.
* ``b``: period along ``y``. Defaults to ``a``.
* ``Nx``, ``Ny``: Yee grid cells. ``Ny`` defaults to ``Nx``.
* ``background_er`` and ``background_ur``: base relative material values.
* ``boundary_conditions``: boundary-condition flags passed to the derivative builder.

Geometry API
------------

.. code-block:: python

   add_object(mask, er=None, ur=None)
   add_circular_inclusion(radius, center=(0.0, 0.0), er=None, ur=None)

Notes:

* ``mask`` can be a boolean array on the 2x helper grid or a callable ``mask(X, Y)``.
* ``add_circular_inclusion`` is a convenience helper for common rod/hole lattices.

Bloch Path And Solve API
------------------------

.. code-block:: python

   default_rectangular_lattice_path()
   generate_bloch_path(points, total_points)
   set_tick_labels(labels, positions)
   compute_band_structure(beta_path, *, num_bands, polarisations=("TE", "TM"), eig_sigma=0.0)

The returned ``BandStructureResult`` contains:

* ``beta_path``: sampled Bloch vectors.
* ``tick_positions`` and ``tick_labels``: symmetry-point labels for plotting.
* ``frequencies``: normalized frequencies keyed by polarization.
* ``eigenvalues``: raw eigenvalues keyed by polarization.

Plotting API
------------

.. code-block:: python

   plot_band_diagram(result, wnmax=None, path_artist_kwargs=None)

The plotting routine renders the unit cell, Bloch path, and TE/TM bands.

Minimal Example
---------------

.. code-block:: python

   from Band_Diagram_Solver import BandDiagramSolver2D

   solver = BandDiagramSolver2D(a=1.0, Nx=40, background_er=10.2)
   solver.add_circular_inclusion(radius=0.4, er=1.0)

   points, labels = solver.default_rectangular_lattice_path()
   beta_path, tick_positions = solver.generate_bloch_path(points, total_points=200)
   solver.set_tick_labels(labels, tick_positions)

   result = solver.compute_band_structure(beta_path, num_bands=5)
   solver.plot_band_diagram(result, wnmax=0.6)

Examples
--------

* ``example_square_lattice.py``
* ``example_rectangular_unitcell.py``
