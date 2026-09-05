fdfd_band_structure user API
============================

Version 1.0.0. This reference covers the deliberately supported user API.
All Python solvers use the same material-first ``mesh()``, ``solve()``, and
``show()`` lifecycle. Phasors use exp(+i omega t); passive relative materials
have nonpositive imaginary values.

Configuration and units
-----------------------

Constructor extents and shape coordinates use metres; frequencies use hertz.
``mesh(resolution=...)`` gives Yee-cell counts, while ``max_element_size`` is a
physical grid-spacing limit. Define reusable ``cem_common.Material`` and shape
objects before assigning them. Grid-index geometry is private backend detail.
All plotting and selection indices are zero-based.

``BandStructureSolver2D``
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    BandStructureSolver2D(*, x_range, y_range, background_material=Material(name='vacuum', epsilon=(1+0j), mu=(1+0j)))

Store physical objects; family adapters implement native insertion/removal.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``x_range``
     - ``float | tuple[float, float] / m``
     - Required
     - ``—``
     - Physical x extent or increasing bounds in metres.
   * - ``y_range``
     - ``float | tuple[float, float] / m``
     - Required
     - ``—``
     - Physical y extent or increasing bounds in metres.
   * - ``background_material``
     - ``Material``
     - Optional
     - ``Material(name='vacuum', epsilon=(1+0j), mu=(1+0j))``
     - Predefined bulk Material assigned to unfilled grid cells.

Returns: a configured solver.

``BandStructureSolver2D.add_geometry``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    BandStructureSolver2D.add_geometry(*, shape, material, name=None, clip=False)

Assign a predefined material to a continuous shape in metres.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``shape``
     - ``array-like or scalar``
     - Required
     - ``—``
     - Continuous cem_common shape expressed in metres.
   * - ``material``
     - ``array-like or scalar``
     - Required
     - ``—``
     - Predefined Material, PEC/PMC, or supported SIBC assignment.
   * - ``name``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - Optional name used for later identification and diagnostics.
   * - ``clip``
     - ``bool``
     - Optional
     - ``False``
     - Intersect the shape with the solver domain; otherwise out-of-bounds objects raise GeometryError.

Returns: the documented data or None when storing state on the solver.

``BandStructureSolver2D.add_rectangle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    BandStructureSolver2D.add_rectangle(*, x_range, y_range, material, name=None, clip=False)

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``x_range``
     - ``float | tuple[float, float] / m``
     - Required
     - ``—``
     - Physical x extent or increasing bounds in metres.
   * - ``y_range``
     - ``float | tuple[float, float] / m``
     - Required
     - ``—``
     - Physical y extent or increasing bounds in metres.
   * - ``material``
     - ``array-like or scalar``
     - Required
     - ``—``
     - Predefined Material, PEC/PMC, or supported SIBC assignment.
   * - ``name``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - Optional name used for later identification and diagnostics.
   * - ``clip``
     - ``bool``
     - Optional
     - ``False``
     - Intersect the shape with the solver domain; otherwise out-of-bounds objects raise GeometryError.

Returns: the documented data or None when storing state on the solver.

``BandStructureSolver2D.add_circle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    BandStructureSolver2D.add_circle(*, center, radius, material, name=None, clip=False)

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``center``
     - ``tuple[float, ...] / m``
     - Required
     - ``—``
     - Physical centre coordinates in metres.
   * - ``radius``
     - ``array-like or scalar``
     - Required
     - ``—``
     - Positive radius in metres.
   * - ``material``
     - ``array-like or scalar``
     - Required
     - ``—``
     - Predefined Material, PEC/PMC, or supported SIBC assignment.
   * - ``name``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - Optional name used for later identification and diagnostics.
   * - ``clip``
     - ``bool``
     - Optional
     - ``False``
     - Intersect the shape with the solver domain; otherwise out-of-bounds objects raise GeometryError.

Returns: the documented data or None when storing state on the solver.

``BandStructureSolver2D.add_polygon``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    BandStructureSolver2D.add_polygon(*, points, material, name=None, clip=False)

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``points``
     - ``sequence[tuple[float, ...]] / m``
     - Required
     - ``—``
     - Ordered polygon vertex coordinates in metres.
   * - ``material``
     - ``array-like or scalar``
     - Required
     - ``—``
     - Predefined Material, PEC/PMC, or supported SIBC assignment.
   * - ``name``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - Optional name used for later identification and diagnostics.
   * - ``clip``
     - ``bool``
     - Optional
     - ``False``
     - Intersect the shape with the solver domain; otherwise out-of-bounds objects raise GeometryError.

Returns: the documented data or None when storing state on the solver.

``BandStructureSolver2D.set_material``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    BandStructureSolver2D.set_material(*, geometry, material)

Reassign a predefined material and invalidate mesh/result.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``geometry``
     - ``array-like or scalar``
     - Required
     - ``—``
     - A handle returned by add_geometry or a geometry convenience method.
   * - ``material``
     - ``array-like or scalar``
     - Required
     - ``—``
     - Predefined Material, PEC/PMC, or supported SIBC assignment.

Returns: the documented data or None when storing state on the solver.

``BandStructureSolver2D.set_shape``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    BandStructureSolver2D.set_shape(*, geometry, shape)

Replace a shape in metres and invalidate mesh/result.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``geometry``
     - ``array-like or scalar``
     - Required
     - ``—``
     - A handle returned by add_geometry or a geometry convenience method.
   * - ``shape``
     - ``array-like or scalar``
     - Required
     - ``—``
     - Continuous cem_common shape expressed in metres.

Returns: the documented data or None when storing state on the solver.

``BandStructureSolver2D.remove``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    BandStructureSolver2D.remove(*, geometry)

Remove an object and invalidate its mesh/result.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``geometry``
     - ``array-like or scalar``
     - Required
     - ``—``
     - A handle returned by add_geometry or a geometry convenience method.

Returns: the documented data or None when storing state on the solver.

``BandStructureSolver2D.mesh``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    BandStructureSolver2D.mesh(*, resolution=None, max_element_size=None)

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``resolution``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - Positive cell count for each physical axis.
   * - ``max_element_size``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - Maximum initial element edge length in metres.

Returns: the initial GridData stored on solver.mesh_data.

``BandStructureSolver2D.make_bloch_path``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    BandStructureSolver2D.make_bloch_path(*, points, num_points=40)

Sample a polyline of (kx,ky) points in rad/m; return a 2-by-N array.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``points``
     - ``sequence[tuple[float, ...]] / m``
     - Required
     - ``—``
     - Ordered polygon vertex coordinates in metres.
   * - ``num_points``
     - ``int``
     - Optional
     - ``40``
     - Number of field sampling points; positive integer.

Returns: the documented data or None when storing state on the solver.

``BandStructureSolver2D.solve``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    BandStructureSolver2D.solve(*, beta_path, num_modes=4, polarizations=('TE', 'TM'), eigenvalue_guess=0.0, eigensolver_tolerance=0.0)

Solve complex eigenfrequencies; dispersive/SIBC materials are unsupported.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``beta_path``
     - ``array-like or scalar``
     - Required
     - ``—``
     - Bloch vectors with shape (2, samples), in radians per metre.
   * - ``num_modes``
     - ``int``
     - Optional
     - ``4``
     - Number of modes requested; positive integer.
   * - ``polarizations``
     - ``tuple``
     - Optional
     - ``('TE', 'TM')``
     - Requested TE/TM polarization names.
   * - ``eigenvalue_guess``
     - ``float``
     - Optional
     - ``0.0``
     - Frequency-eigenvalue spectral shift.
   * - ``eigensolver_tolerance``
     - ``float``
     - Optional
     - ``0.0``
     - Algebraic eigensolver convergence tolerance.

Returns: a typed result stored on solver.result.

``BandStructureSolver2D.show``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    BandStructureSolver2D.show(*, block=True)

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``block``
     - ``bool``
     - Optional
     - ``True``
     - Wait for the interactive viewer to close when true.

Returns: the interactive Matplotlib figure.

Returned result
---------------

Result objects come from ``solve()`` or ``load_result()``; users do not
construct them directly. Field results expose ``mesh_data``, ``metadata``,
``solve_info``, and explicit physical field coordinates.

``BandStructureResult.plot``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    BandStructureResult.plot(*, component=None, quantity='real', mode=None)

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``component``
     - ``str | None``
     - Optional
     - ``None``
     - Field component to display, such as Ey; electrostatics also accepts potential or mesh.
   * - ``quantity``
     - ``str``
     - Optional
     - ``'real'``
     - Displayed field quantity: real, imag, magnitude/abs, or phase; static fields support real or magnitude.
   * - ``mode``
     - ``int``
     - Optional
     - ``None``
     - Zero-based mode or band index.

Returns: a Matplotlib Figure without opening a window.

``BandStructureResult.show``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    BandStructureResult.show(*, block=True)

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``block``
     - ``bool``
     - Optional
     - ``True``
     - Wait for the interactive viewer to close when true.

Returns: the interactive Matplotlib Figure.

``BandStructureResult.save``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    BandStructureResult.save(path)

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``path``
     - ``str | PathLike``
     - Required
     - ``—``
     - Destination/source HDF5 path. Saving is atomic; loading does not run a solver.

Returns: the atomically written HDF5 path.

``load_result``
~~~~~~~~~~~~~~~

.. code-block:: python

    load_result(path)

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``path``
     - ``str | PathLike``
     - Required
     - ``—``
     - Destination/source HDF5 path. Saving is atomic; loading does not run a solver.

Returns: a typed ``BandStructureResult`` without solving.

Results and examples
--------------------

``solve`` returns ``BandStructureResult`` with frequency arrays in hertz and
eigenvalues indexed by TE/TM polarization.

Results provide ``plot()``, ``show()``, and atomic ``save()``; each package
exports ``load_result()``. Invalid dimensions, materials, and controls raise
actionable ``cem_common`` exceptions. See the `user guide <guide.rst>`_ and
root examples. Assembly routines, matrix builders, grid-index records, and
Arnoldi kernels are excluded from this user reference.
