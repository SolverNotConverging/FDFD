Scattering Solver 2D
====================

``FDFD2DScatteringSolver`` solves two-dimensional frequency-domain scattering problems on a Yee grid. It supports scalar TEz and TMz total-field/scattered-field formulations with plane-wave or point-source excitation.

What It Solves
--------------

Use this solver for 2D scattering from cylinders or cross-sections that are invariant along the out-of-plane direction. The solver builds sparse FDFD operators, applies a total-field/scattered-field mask, and solves for the total scattered response.

Supported features:

* TEz solve for ``Ez``.
* TMz solve for ``Hz``.
* Isotropic or diagonal-anisotropic relative ``epsilon`` and ``mu`` tensors.
* Plane-wave and point-source excitation.
* Simple uniaxial PML in ``x``, ``y``, or both directions.
* Mask operator for total-field/scattered-field separation.

Main Class
----------

.. code-block:: python

   FDFD2DScatteringSolver(frequency, x_range, y_range, Nx, Ny)

Parameters:

* ``frequency``: source frequency in Hz.
* ``x_range``, ``y_range``: physical domain spans in metres.
* ``Nx``, ``Ny``: grid cells in ``x`` and ``y``.

Geometry API
------------

.. code-block:: python

   add_object(er_tensor, mr_tensor, region_mask)

Notes:

* ``region_mask`` is a boolean array with shape ``(Ny, Nx)``.
* ``er_tensor`` and ``mr_tensor`` can be scalars or length-3 values ordered as tensor components.
* Coordinate helper arrays ``X`` and ``Y`` are available for constructing masks.

Source And Boundary API
-----------------------

.. code-block:: python

   add_source(src_type="plane_wave", angle_deg=0.0, polarization="TE", location=None, amplitude=1.0)
   add_UPML(pml_width=20, n=3, sigma_max=5.0, direction="both")
   add_mask(value=30)

Notes:

* ``src_type`` accepts ``"plane_wave"`` or ``"point"``.
* ``angle_deg`` is measured from ``+x`` for plane waves.
* Point sources require ``location=(x0, y0)``.
* ``add_mask`` accepts a scalar frame width, a dense mask array, or a sparse mask matrix.

Solve API
---------

.. code-block:: python

   solve_total_field_TE(reuse_factorisation=True)
   solve_total_field_TM(reuse_factorisation=True)

Outputs:

* ``Ez`` after ``solve_total_field_TE``.
* ``Hz`` after ``solve_total_field_TM``.
* Cached sparse matrices and factorizations are reused by default for repeated solves.

Visualization
-------------

.. code-block:: python

   TE_Visualization()
   TM_Visualization()

These routines show quick diagnostic plots for the solved fields.

Minimal Example
---------------

.. code-block:: python

   import numpy as np
   from Scattering_Solver_2D import FDFD2DScatteringSolver

   f0 = 10e9
   wavelength = 299792458 / f0
   sim = FDFD2DScatteringSolver(f0, 6 * wavelength, 6 * wavelength, 300, 300)

   radius = 0.5 * wavelength
   cylinder = (sim.X ** 2 + sim.Y ** 2) <= radius ** 2
   sim.add_object(er_tensor=4.0, mr_tensor=1.0, region_mask=cylinder)

   sim.add_UPML(pml_width=40, sigma_max=10)
   sim.add_mask(value=80)
   sim.add_source(src_type="plane_wave", angle_deg=45.0, polarization="TE")
   sim.solve_total_field_TE()
   sim.TE_Visualization()

Examples
--------

* ``example_scattering_by_cylinder.py``
