fdfd_periodic_modes user API
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

``PeriodicModeSolver2D``
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver2D(*, frequency, x_range, z_range, polarization='TE', background_material=Material(name='vacuum', epsilon=(1+0j), mu=(1+0j)))

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
   * - ``z_range``
     - ``float | tuple[float, float] / m``
     - Required
     - ``—``
     - Physical z extent or increasing bounds in metres.
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

``PeriodicModeSolver2D.add_geometry``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver2D.add_geometry(*, shape, material, name=None, clip=False)

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

``PeriodicModeSolver2D.add_rectangle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver2D.add_rectangle(*, x_range, z_range, material, name=None, clip=False)

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
   * - ``z_range``
     - ``float | tuple[float, float] / m``
     - Required
     - ``—``
     - Physical z extent or increasing bounds in metres.
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

``PeriodicModeSolver2D.add_circle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver2D.add_circle(*, center, radius, material, name=None, clip=False)

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

``PeriodicModeSolver2D.add_polygon``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver2D.add_polygon(*, points, material, name=None, clip=False)

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

``PeriodicModeSolver2D.set_material``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver2D.set_material(*, geometry, material)

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

``PeriodicModeSolver2D.set_shape``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver2D.set_shape(*, geometry, shape)

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

``PeriodicModeSolver2D.add_pml``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver2D.add_pml(*, thickness, direction='all', order=3, sigma_max=5.0)

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

``PeriodicModeSolver2D.remove``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver2D.remove(*, geometry)

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

``PeriodicModeSolver2D.mesh``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver2D.mesh(*, resolution=None, max_element_size=None, subpixels=8)

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
   * - ``subpixels``
     - ``int``
     - Optional
     - ``8``
     - Number of subcell samples used to average region material values.

Returns: the initial GridData stored on solver.mesh_data.

``PeriodicModeSolver2D.solve``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver2D.solve(*, num_modes=4, neff_guess=1.0, eigensolver_tolerance=0.0, eigensolver='eigs', ncv=None, max_restarts=12, random_seed=0, arnoldi_backend='auto')

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``num_modes``
     - ``int``
     - Optional
     - ``4``
     - Number of modes requested; positive integer.
   * - ``neff_guess``
     - ``float``
     - Optional
     - ``1.0``
     - Dimensionless complex effective-index search target.
   * - ``eigensolver_tolerance``
     - ``float``
     - Optional
     - ``0.0``
     - Algebraic eigensolver convergence tolerance.
   * - ``eigensolver``
     - ``str``
     - Optional
     - ``'eigs'``
     - Eigensolver control for this operation.
   * - ``ncv``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - Arnoldi subspace size; None selects the backend default.
   * - ``max_restarts``
     - ``int``
     - Optional
     - ``12``
     - Maximum refined Arnoldi restarts.
   * - ``random_seed``
     - ``int``
     - Optional
     - ``0``
     - Deterministic initial-vector seed.
   * - ``arnoldi_backend``
     - ``str``
     - Optional
     - ``'auto'``
     - Arnoldi backend control for this operation.

Returns: a typed result stored on solver.result.

``PeriodicModeSolver2D.show``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver2D.show(*, block=True)

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

``PeriodicModeSolver3D``
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver3D(*, frequency, x_range, y_range, z_range, background_material=Material(name='vacuum', epsilon=(1+0j), mu=(1+0j)))

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
   * - ``z_range``
     - ``float | tuple[float, float] / m``
     - Required
     - ``—``
     - Physical z extent or increasing bounds in metres.
   * - ``background_material``
     - ``Material``
     - Optional
     - ``Material(name='vacuum', epsilon=(1+0j), mu=(1+0j))``
     - Predefined bulk Material assigned to unfilled grid cells.

Returns: a configured solver.

``PeriodicModeSolver3D.add_geometry``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver3D.add_geometry(*, shape, material, name=None, clip=False)

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

``PeriodicModeSolver3D.add_box``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver3D.add_box(*, x_range, y_range, z_range, material, name=None, clip=False)

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
   * - ``z_range``
     - ``float | tuple[float, float] / m``
     - Required
     - ``—``
     - Physical z extent or increasing bounds in metres.
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

``PeriodicModeSolver3D.add_sphere``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver3D.add_sphere(*, center, radius, material, name=None, clip=False)

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

``PeriodicModeSolver3D.add_cylinder``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver3D.add_cylinder(*, center, radius, z_range, material, name=None, clip=False)

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
   * - ``z_range``
     - ``float | tuple[float, float] / m``
     - Required
     - ``—``
     - Physical z extent or increasing bounds in metres.
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

``PeriodicModeSolver3D.set_material``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver3D.set_material(*, geometry, material)

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

``PeriodicModeSolver3D.set_shape``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver3D.set_shape(*, geometry, shape)

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

``PeriodicModeSolver3D.add_pml``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver3D.add_pml(*, thickness, direction='all', order=3, sigma_max=5.0)

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

``PeriodicModeSolver3D.remove``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver3D.remove(*, geometry)

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

``PeriodicModeSolver3D.mesh``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver3D.mesh(*, resolution=None, max_element_size=None, subpixels=8)

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
   * - ``subpixels``
     - ``int``
     - Optional
     - ``8``
     - Number of subcell samples used to average region material values.

Returns: the initial GridData stored on solver.mesh_data.

``PeriodicModeSolver3D.solve``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver3D.solve(*, num_modes=4, neff_guess=1.0, eigensolver_tolerance=0.0, eigensolver='refined', ncv=None, max_restarts=12, random_seed=0, arnoldi_backend='auto')

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``num_modes``
     - ``int``
     - Optional
     - ``4``
     - Number of modes requested; positive integer.
   * - ``neff_guess``
     - ``float``
     - Optional
     - ``1.0``
     - Dimensionless complex effective-index search target.
   * - ``eigensolver_tolerance``
     - ``float``
     - Optional
     - ``0.0``
     - Algebraic eigensolver convergence tolerance.
   * - ``eigensolver``
     - ``str``
     - Optional
     - ``'refined'``
     - Eigensolver control for this operation.
   * - ``ncv``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - Arnoldi subspace size; None selects the backend default.
   * - ``max_restarts``
     - ``int``
     - Optional
     - ``12``
     - Maximum refined Arnoldi restarts.
   * - ``random_seed``
     - ``int``
     - Optional
     - ``0``
     - Deterministic initial-vector seed.
   * - ``arnoldi_backend``
     - ``str``
     - Optional
     - ``'auto'``
     - Arnoldi backend control for this operation.

Returns: a typed result stored on solver.result.

``PeriodicModeSolver3D.show``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver3D.show(*, block=True)

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

``PeriodicModeSet.plot``
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSet.plot(*, component=None, quantity='real', mode=0, plane=None, position=None)

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

``PeriodicModeSet.show``
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSet.show(*, block=True)

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

``PeriodicModeSet.save``
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSet.save(path)

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

Returns: a typed ``PeriodicModeSet`` without solving.

Results and examples
--------------------

``solve`` returns a modal set with dimensionless ``neff``, ``beta`` in rad/m,
explicit staggered field coordinates, and zero-based mode selection.

Results provide ``plot()``, ``show()``, and atomic ``save()``; each package
exports ``load_result()``. Invalid dimensions, materials, and controls raise
actionable ``cem_common`` exceptions. See the `user guide <guide.rst>`_ and
root examples. Assembly routines, matrix builders, grid-index records, and
Arnoldi kernels are excluded from this user reference.
