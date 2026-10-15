Scattering
==========

:class:`~fdfd.ScatteringSolver2D` solves 2D scattering by objects invariant
along z, for a plane wave or a line source, using a total-field /
scattered-field (TF/SF) split. The result is a :class:`~fdfd.ScatteringResult`
with three field components.

Example
-------

.. code-block:: python

   import fdfd
   from fdfd import materials

   solver = fdfd.ScatteringSolver2D(frequency=3e9, x_range=(-0.15, 0.15), y_range=(-0.15, 0.15))
   solver.add_circle(center=(0, 0), radius=25e-3, material=materials.PEC)
   solver.add_pml(thickness=50e-3)
   solver.set_source(kind="plane_wave", angle=30.0, inset=75e-3)
   solver.mesh(resolution=(240, 240))
   te = solver.solve(polarization="TE")
   tm = solver.solve(polarization="TM")

``set_source`` defines both the incident field and the total-field rectangle,
inset from the domain boundary by ``inset`` metres. Scatterers must lie inside
it and the PML outside it. A line source uses ``kind="point"`` and
``location=(x, y)``. Changing only the source keeps the factorization, so
solving many incidence angles is cheap.

Polarizations and fields
------------------------

.. list-table::
   :header-rows: 1

   * - Polarization
     - Field
     - x / y location
     - Shape
   * - TE
     - Ez (primary)
     - node / node
     - (Nx+1, Ny+1)
   * - TE
     - Hx
     - node / cell
     - (Nx+1, Ny)
   * - TE
     - Hy
     - cell / node
     - (Nx, Ny+1)
   * - TM
     - Hz (primary)
     - cell / cell
     - (Nx, Ny)
   * - TM
     - Ex
     - cell / node
     - (Nx, Ny+1)
   * - TM
     - Ey
     - node / cell
     - (Nx+1, Ny)

Fields are SI; inside the total-field rectangle they are the total field and
outside it the scattered field. The last array axis has length 1.

Formulation
-----------

With normalized differences, the TE operator is

.. math::

   A = D_{hx}\,\mu_{yy}^{-1}D_{ex} + D_{hy}\,\mu_{xx}^{-1}D_{ey} + \varepsilon_{zz},

acting on :math:`E_z` (the TM operator is the dual on :math:`H_z`). With
:math:`Q` the diagonal mask of the scattered-field region, the TF/SF source is

.. math::

   A\,u = (QA - AQ)\,u_{\rm inc},

which injects the incident wave exactly on the rectangle boundary; the
incident field is a discrete plane wave or the outgoing cylindrical wave
:math:`H_0^{(2)}(k_0 r)`. Transverse fields are reconstructed with the same
masks evaluated on their own lattices, so they carry no spurious jump at the
TF/SF boundary. The outer boundary is PEC (normally behind the PML) and PEC
objects constrain their Yee samples. ``info.residuals`` is the relative
residual of the linear system; solves above :math:`10^{-8}` raise
:class:`~fdfd.SolverError`.

The tests compare dielectric and PEC cylinders against the exact
cylindrical-wave series; with the default PML and ``subpixels`` the field error
is about 0.4-2.6% at 240 x 240 cells and converges with refinement.
