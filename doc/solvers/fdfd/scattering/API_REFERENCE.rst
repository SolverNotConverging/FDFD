fdfd_scattering user API
========================

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

``ScatteringSolver2D``
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSolver2D(*, frequency, x_range, y_range, polarization='TE', background_material=Material(name='vacuum', epsilon=(1+0j), mu=(1+0j)))

Store physical objects; family adapters implement native insertion/removal.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``frequency``
     - ``array-like or scalar``
     - Required
     - ``—``
     - Operating frequency in hertz; finite and positive.
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
   * - ``polarization``
     - ``str``
     - Optional
     - ``'TE'``
     - Polarization control for this operation.
   * - ``background_material``
     - ``Material``
     - Optional
     - ``Material(name='vacuum', epsilon=(1+0j), mu=(1+0j))``
     - Predefined bulk Material assigned to unfilled grid cells.

Returns: a configured solver.

``ScatteringSolver2D.add_geometry``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSolver2D.add_geometry(*, shape, material, name=None, clip=False)

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

``ScatteringSolver2D.add_rectangle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSolver2D.add_rectangle(*, x_range, y_range, material, name=None, clip=False)

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

``ScatteringSolver2D.add_circle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSolver2D.add_circle(*, center, radius, material, name=None, clip=False)

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

``ScatteringSolver2D.add_polygon``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSolver2D.add_polygon(*, points, material, name=None, clip=False)

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

``ScatteringSolver2D.set_material``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSolver2D.set_material(*, geometry, material)

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

``ScatteringSolver2D.set_shape``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSolver2D.set_shape(*, geometry, shape)

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

``ScatteringSolver2D.add_pml``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSolver2D.add_pml(*, thickness, direction='all', order=3, sigma_max=5.0)

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``thickness``
     - ``array-like or scalar``
     - Required
     - ``—``
     - PML thickness in metres.
   * - ``direction``
     - ``str``
     - Optional
     - ``'all'``
     - Propagation direction for solve; selected coordinate direction for PML.
   * - ``order``
     - ``int``
     - Optional
     - ``3``
     - Polynomial PML order.
   * - ``sigma_max``
     - ``float``
     - Optional
     - ``5.0``
     - Maximum PML-strength magnitude.

Returns: the documented data or None when storing state on the solver.

``ScatteringSolver2D.remove``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSolver2D.remove(*, geometry)

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

``ScatteringSolver2D.mesh``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSolver2D.mesh(*, resolution=None, max_element_size=None)

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

``ScatteringSolver2D.add_source``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSolver2D.add_source(*, kind='plane_wave', angle=0.0, location=None, amplitude=1.0)

Set the incident field; angles are degrees from physical +x.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``kind``
     - ``str``
     - Optional
     - ``'plane_wave'``
     - Source kind: plane_wave or point.
   * - ``angle``
     - ``float``
     - Optional
     - ``0.0``
     - Physical incidence angle in degrees, strictly between -90 and 90; mutually exclusive with ky.
   * - ``location``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - Physical point-source position in metres.
   * - ``amplitude``
     - ``float``
     - Optional
     - ``1.0``
     - Complex incident-mode amplitude.

Returns: the documented data or None when storing state on the solver.

``ScatteringSolver2D.set_source_region``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSolver2D.set_source_region(*, inset)

Set the rectangular total-field region's physical inset in metres.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``inset``
     - ``array-like or scalar``
     - Required
     - ``—``
     - Physical inset of the FDFD total-field source region, in metres.

Returns: the documented data or None when storing state on the solver.

``ScatteringSolver2D.solve``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSolver2D.solve(*, reuse_factorization=True)

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``reuse_factorization``
     - ``bool``
     - Optional
     - ``True``
     - Reuse factorization control for this operation.

Returns: a typed result stored on solver.result.

``ScatteringSolver2D.show``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSolver2D.show(*, block=True)

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

``ScatteringResult.plot``
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringResult.plot(*, component=None, quantity='real', mode=0, plane=None, position=None)

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
     - ``0``
     - Zero-based mode or band index.
   * - ``plane``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - Plane control for this operation.
   * - ``position``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - Position control for this operation.

Returns: a Matplotlib Figure without opening a window.

``ScatteringResult.show``
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringResult.show(*, block=True)

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

``ScatteringResult.save``
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringResult.save(path)

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

Returns: a typed ``ScatteringResult`` without solving.

Results and examples
--------------------

``solve`` returns ``ScatteringResult`` with scalar total fields at their
physical Yee-grid locations.

Results provide ``plot()``, ``show()``, and atomic ``save()``; each package
exports ``load_result()``. Invalid dimensions, materials, and controls raise
actionable ``cem_common`` exceptions. See the `user guide <guide.rst>`_ and
root examples. Assembly routines, matrix builders, grid-index records, and
Arnoldi kernels are excluded from this user reference.
