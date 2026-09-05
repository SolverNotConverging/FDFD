fem_waveguide_scattering user API
=================================

Version 1.0.0. This reference covers the deliberately supported user API.
Implementation helpers are documented in their source modules.

Workflow and units
------------------

All lengths are in metres, frequencies in hertz, and constitutive values are relative.
Construction and configuration use keyword arguments. Call ``mesh()`` to build
the initial mesh, ``solve()`` to obtain and store a typed result, and ``show()``
to inspect it interactively. ``mesh_data`` and ``result`` are initially None.
Geometry edits invalidate both; remeshing invalidates the result. Automatic
meshing reuses the last explicit settings. Calling ``show()`` without a result
raises ``NoResultError``.

Adaptive defaults are ``max_refinements=2`` and ``adaptive_tolerance=0.05``.
Zero refinements performs one solve. ``solve_info`` distinguishes algebraic
residuals from adaptive discretization residuals and records stopping reasons.
Python mode and case indices are zero-based.

``solve()`` and frequency sweeps neither save nor open windows. Call
``result.save(path)`` explicitly. ``load_result(path)`` returns inspection-ready
results without solving. Archives use ``cem-fem-results`` schema ``1.0``;
old or convention-incompatible files raise ``PersistenceError``. Loaded
results can plot, show, and save; they do not restart a solver or restore callbacks.
``plot()`` returns a Matplotlib Figure without opening a window.
``show(block=True)`` waits for the viewer; ``block=False`` returns immediately.

The time convention is ``exp(+i*omega*t)`` with guided propagation
``exp(-i*beta*z)``. Passive materials have nonpositive imaginary constitutive
values; passive forward attenuation is ``-Im(beta)``.

Scattering uses a two-dimensional x/z mesh for **2.5D full-vector** fields,
with invariant factor ``exp(-i*ky*y)``. Scalar epsilon and mu are supported.
Integrated power accounting requires passive materials and a lossless uniform
lead. Port modes remain a separate implementation from standalone mode solvers.
PML supports x, z, or all; it applies at both ends of the selected axis.

Supported exports
-----------------

``WaveguideScatteringSolver2D``, ``ScatteringResult``, ``FrequencySweepResult``, ``IncidentMode``, ``Mode``, ``ModeSet``, ``Diagnostic``, ``DiagnosticReport``, ``BackendCapabilityError``, ``ConfigurationError``, ``GeometryError``, ``MaterialError``, ``MeshError``, ``ModeProjectionError``, ``ModeSolverError``, ``SolverError``, ``ViewerError``, ``NoResultError``, ``PersistenceError``, ``load_result``.

Solver construction and operations
----------------------------------

``WaveguideScatteringSolver2D``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    WaveguideScatteringSolver2D(*, frequency: 'float', angle: 'float | None' = None, ky: 'float | None' = None, x_range: 'float | Sequence[float]', z_range: 'float | Sequence[float]', background_material: 'materials.Material' = Material(name='vacuum', epsilon=(1+0j), mu=(1+0j)), boundary: 'materials.IdealBoundary | None' = None) -> 'None'

Full-vector 2.5D FEM scattering simulation in SI units.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``frequency``
     - ``float``
     - Required
     - ``—``
     - Operating frequency in hertz; finite and positive.
   * - ``angle``
     - ``float | None``
     - Optional
     - ``None``
     - Physical incidence angle in degrees, strictly between -90 and 90; mutually exclusive with ky.
   * - ``ky``
     - ``float | None``
     - Optional
     - ``None``
     - Real invariant-direction wavenumber in radians per metre; mutually exclusive with angle.
   * - ``x_range``
     - ``float | Sequence[float]``
     - Required
     - ``—``
     - Physical x extent or increasing bounds, in metres.
   * - ``z_range``
     - ``float | Sequence[float]``
     - Required
     - ``—``
     - Physical z extent or increasing bounds, in metres.
   * - ``background_material``
     - ``materials.Material``
     - Optional
     - ``Material(name='vacuum', epsilon=(1+0j), mu=(1+0j))``
     - Predefined bulk Material assigned to unfilled space.
   * - ``boundary``
     - ``materials.IdealBoundary | None``
     - Optional
     - ``None``
     - Predefined PEC or PMC exterior-boundary material.

Returns: a configured ``WaveguideScatteringSolver2D``.

``WaveguideScatteringSolver2D.add_geometry``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    WaveguideScatteringSolver2D.add_geometry(*, shape, material, name=None, clip=False, background=False)

Assign a material; background objects also belong to the straight lead.

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
     - A predefined cem_common.shapes object in metres.
   * - ``material``
     - ``array-like or scalar``
     - Required
     - ``—``
     - A predefined bulk, ideal-boundary, or supported SIBC material.
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
   * - ``background``
     - ``bool``
     - Optional
     - ``False``
     - Include this region in both the unperturbed lead and actual device.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``WaveguideScatteringSolver2D.add_rectangle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    WaveguideScatteringSolver2D.add_rectangle(*, x_range, z_range, material, name=None, clip=False, background=False)

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
     - Physical x extent or increasing bounds, in metres.
   * - ``z_range``
     - ``array-like or scalar``
     - Required
     - ``—``
     - Physical z extent or increasing bounds, in metres.
   * - ``material``
     - ``array-like or scalar``
     - Required
     - ``—``
     - A predefined bulk, ideal-boundary, or supported SIBC material.
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
   * - ``background``
     - ``bool``
     - Optional
     - ``False``
     - Include this region in both the unperturbed lead and actual device.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``WaveguideScatteringSolver2D.add_circle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    WaveguideScatteringSolver2D.add_circle(*, center, radius, material, name=None, clip=False)

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
     - A predefined bulk, ideal-boundary, or supported SIBC material.
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

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``WaveguideScatteringSolver2D.add_polygon``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    WaveguideScatteringSolver2D.add_polygon(*, points, material, name=None, clip=False)

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
     - A predefined bulk, ideal-boundary, or supported SIBC material.
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

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``WaveguideScatteringSolver2D.add_slot``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    WaveguideScatteringSolver2D.add_slot(*, geometry, z_range, name=None)

Cut a finite opening in an existing background PEC sheet.

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
   * - ``z_range``
     - ``array-like or scalar``
     - Required
     - ``—``
     - Physical z extent or increasing bounds, in metres.
   * - ``name``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - Optional name used for later identification and diagnostics.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``WaveguideScatteringSolver2D.set_material``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    WaveguideScatteringSolver2D.set_material(*, geometry, material)

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
     - A predefined bulk, ideal-boundary, or supported SIBC material.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``WaveguideScatteringSolver2D.set_shape``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    WaveguideScatteringSolver2D.set_shape(*, geometry, shape)

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
     - A predefined cem_common.shapes object in metres.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``WaveguideScatteringSolver2D.set_material_field``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    WaveguideScatteringSolver2D.set_material_field(*, material, background_material)

Use named actual/background SpatialMaterial fields instead of objects.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``material``
     - ``array-like or scalar``
     - Required
     - ``—``
     - A predefined bulk, ideal-boundary, or supported SIBC material.
   * - ``background_material``
     - ``array-like or scalar``
     - Required
     - ``—``
     - Predefined bulk Material assigned to unfilled space.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``WaveguideScatteringSolver2D.set_boundary``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    WaveguideScatteringSolver2D.set_boundary(*, material)

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``material``
     - ``array-like or scalar``
     - Required
     - ``—``
     - A predefined bulk, ideal-boundary, or supported SIBC material.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``WaveguideScatteringSolver2D.add_pml``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    WaveguideScatteringSolver2D.add_pml(*, thickness: 'float', direction: 'str' = 'all', order: 'int' = 3, target_reflection: 'float' = 1e-08) -> 'None'

Add matched layers on both ends of the selected x/z axis.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``thickness``
     - ``float``
     - Required
     - ``—``
     - PML thickness in metres, at each selected exterior end.
   * - ``direction``
     - ``str``
     - Optional
     - ``'all'``
     - Propagation direction for solve; selected coordinate direction for PML.
   * - ``order``
     - ``int``
     - Optional
     - ``3``
     - Polynomial order of the PML profile.
   * - ``target_reflection``
     - ``float``
     - Optional
     - ``1e-08``
     - Desired PML amplitude reflection ratio in (0, 1).

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``WaveguideScatteringSolver2D.set_monitors``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    WaveguideScatteringSolver2D.set_monitors(*, left: 'float', right: 'float') -> 'None'

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``left``
     - ``float``
     - Required
     - ``—``
     - Left monitor/reference-plane position in metres.
   * - ``right``
     - ``float``
     - Required
     - ``—``
     - Right monitor/reference-plane position in metres.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``WaveguideScatteringSolver2D.remove``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    WaveguideScatteringSolver2D.remove(*, geometry)

Remove an owned geometry object or the slot returned by add_slot().

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

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``WaveguideScatteringSolver2D.mesh``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    WaveguideScatteringSolver2D.mesh(*, max_element_size: 'float | None' = None, element_order: 'int' = 1, quadrature_order: 'int' = 4, wavelength_elements: 'int' = 4, refine_interfaces: 'bool' = True, dielectric_refinement_factor: 'float' = 0.5, pec_refinement_factor: 'float' = 0.5, pec_refinement_distance: 'float | None' = None) -> 'Mesh2D'

Generate the Gmsh mesh and reveal the selected maximum edge size.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``max_element_size``
     - ``float | None``
     - Optional
     - ``None``
     - Maximum initial element edge length in metres.
   * - ``element_order``
     - ``int``
     - Optional
     - ``1``
     - Finite-element polynomial order supported by this backend.
   * - ``quadrature_order``
     - ``int``
     - Optional
     - ``4``
     - Element integration order.
   * - ``wavelength_elements``
     - ``int``
     - Optional
     - ``4``
     - Minimum number of initial elements per local wavelength.
   * - ``refine_interfaces``
     - ``bool``
     - Optional
     - ``True``
     - Refine interfaces control for this operation.
   * - ``dielectric_refinement_factor``
     - ``float``
     - Optional
     - ``0.5``
     - Dielectric refinement factor control for this operation.
   * - ``pec_refinement_factor``
     - ``float``
     - Optional
     - ``0.5``
     - Pec refinement factor control for this operation.
   * - ``pec_refinement_distance``
     - ``float | None``
     - Optional
     - ``None``
     - Pec refinement distance control for this operation.

Returns: the initial mesh stored in ``mesh_data``.

``WaveguideScatteringSolver2D.solve_modes``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    WaveguideScatteringSolver2D.solve_modes(*, side: "Literal['left', 'right']" = 'left', num_modes: 'int' = 4, neff_guess: 'complex | None' = None, num_elements: 'int | None' = None, max_refinements: 'int' = 2, adaptive_tolerance: 'float' = 0.05) -> 'ModeSet'

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``side``
     - ``Literal['left', 'right']``
     - Optional
     - ``'left'``
     - Port label, left or right.
   * - ``num_modes``
     - ``int``
     - Optional
     - ``4``
     - Number of modes requested; positive integer.
   * - ``neff_guess``
     - ``complex | None``
     - Optional
     - ``None``
     - Dimensionless complex effective-index search target.
   * - ``num_elements``
     - ``int | None``
     - Optional
     - ``None``
     - Num elements control for this operation.
   * - ``max_refinements``
     - ``int``
     - Optional
     - ``2``
     - Maximum mesh refinements after the initial solve; zero means one solve.
   * - ``adaptive_tolerance``
     - ``float``
     - Optional
     - ``0.05``
     - Relative discretization-residual stopping threshold.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``WaveguideScatteringSolver2D.set_incident_mode``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    WaveguideScatteringSolver2D.set_incident_mode(mode: 'int | Mode', *, side: "Literal['left', 'right']" = 'left', reference_plane: 'float | None' = None, amplitude: 'complex' = 1.0) -> 'IncidentMode'

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``mode``
     - ``int | Mode``
     - Required
     - ``—``
     - Zero-based mode index.
   * - ``side``
     - ``Literal['left', 'right']``
     - Optional
     - ``'left'``
     - Port label, left or right.
   * - ``reference_plane``
     - ``float | None``
     - Optional
     - ``None``
     - Incident phase reference position in metres.
   * - ``amplitude``
     - ``complex``
     - Optional
     - ``1.0``
     - Complex incident-mode amplitude.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``WaveguideScatteringSolver2D.solve``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    WaveguideScatteringSolver2D.solve(*, linear_solver: "Literal['direct']" = 'direct', linear_solver_tolerance: 'float' = 1e-10, projection_condition_limit: 'float' = 1000000000000.0, max_refinements: 'int' = 2, adaptive_tolerance: 'float' = 0.05) -> 'ScatteringResult'

Solve without saving or opening a viewer.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``linear_solver``
     - ``Literal['direct']``
     - Optional
     - ``'direct'``
     - Linear solver control for this operation.
   * - ``linear_solver_tolerance``
     - ``float``
     - Optional
     - ``1e-10``
     - Algebraic linear-system residual tolerance.
   * - ``projection_condition_limit``
     - ``float``
     - Optional
     - ``1000000000000.0``
     - Projection condition limit control for this operation.
   * - ``max_refinements``
     - ``int``
     - Optional
     - ``2``
     - Maximum mesh refinements after the initial solve; zero means one solve.
   * - ``adaptive_tolerance``
     - ``float``
     - Optional
     - ``0.05``
     - Relative discretization-residual stopping threshold.

Returns: the physics-specific result stored in ``result``.

``WaveguideScatteringSolver2D.sweep``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    WaveguideScatteringSolver2D.sweep(frequencies: 'Sequence[float]', *, linear_solver: "Literal['direct']" = 'direct', linear_solver_tolerance: 'float' = 1e-10, projection_condition_limit: 'float' = 1000000000000.0, max_refinements: 'int' = 2, adaptive_tolerance: 'float' = 0.05, mesh_options: 'Mapping[str, object] | None' = None, mode_options: 'Mapping[str, object] | None' = None, incident_mode: 'int' = 0, amplitude: 'complex' = 1.0, reference_plane: 'float | None' = None, mode_factory: 'Callable[[float], ModeSet] | None' = None) -> 'FrequencySweepResult'

Solve independent points on an increasing ordinary-frequency grid.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``frequencies``
     - ``Sequence[float]``
     - Required
     - ``—``
     - Strictly increasing positive frequencies in hertz.
   * - ``linear_solver``
     - ``Literal['direct']``
     - Optional
     - ``'direct'``
     - Linear solver control for this operation.
   * - ``linear_solver_tolerance``
     - ``float``
     - Optional
     - ``1e-10``
     - Algebraic linear-system residual tolerance.
   * - ``projection_condition_limit``
     - ``float``
     - Optional
     - ``1000000000000.0``
     - Projection condition limit control for this operation.
   * - ``max_refinements``
     - ``int``
     - Optional
     - ``2``
     - Maximum mesh refinements after the initial solve; zero means one solve.
   * - ``adaptive_tolerance``
     - ``float``
     - Optional
     - ``0.05``
     - Relative discretization-residual stopping threshold.
   * - ``mesh_options``
     - ``Mapping[str, object] | None``
     - Optional
     - ``None``
     - Mesh options control for this operation.
   * - ``mode_options``
     - ``Mapping[str, object] | None``
     - Optional
     - ``None``
     - Mode options control for this operation.
   * - ``incident_mode``
     - ``int``
     - Optional
     - ``0``
     - Incident mode control for this operation.
   * - ``amplitude``
     - ``complex``
     - Optional
     - ``1.0``
     - Complex incident-mode amplitude.
   * - ``reference_plane``
     - ``float | None``
     - Optional
     - ``None``
     - Incident phase reference position in metres.
   * - ``mode_factory``
     - ``Callable[[float], ModeSet] | None``
     - Optional
     - ``None``
     - Mode factory control for this operation.

Returns: a frequency sweep result.

``WaveguideScatteringSolver2D.show``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    WaveguideScatteringSolver2D.show(*, block: 'bool' = True)

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

Returns: the viewer controller or native process.

Returned results
----------------

Result objects are returned by solving or loading. Their constructors are
implementation details. Inspect ``mesh_data``, ``metadata``, ``solve_info``, and
``frequency`` where applicable. Modal sets support iteration, indexing, ``neff``
and ``beta``; each mode provides sampled fields and residual diagnostics.

``ScatteringResult.plot``
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringResult.plot(*, component: 'str | None' = None, quantity: 'str' = 'real')

Return a sampled-field figure without opening a window.

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

Returns: a Matplotlib Figure.

``ScatteringResult.show``
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringResult.show(*, block: 'bool' = True)

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

Returns: the viewer controller or native process.

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

Returns: the written Path.

``ScatteringResult.S``
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringResult.S(side: 'str', *, out_mode: 'int' = 0, in_mode: 'int' = 0) -> 'complex'

Return an indexed outgoing modal amplitude.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``side``
     - ``str``
     - Required
     - ``—``
     - Port label, left or right.
   * - ``out_mode``
     - ``int``
     - Optional
     - ``0``
     - Out mode control for this operation.
   * - ``in_mode``
     - ``int``
     - Optional
     - ``0``
     - In mode control for this operation.

Returns: the selected data or diagnostic report.

``ScatteringResult.check``
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringResult.check(*, power_balance_tolerance: 'float' = 0.001, projection_condition_warning: 'float' = 10000000000.0, projection_residual_warning: 'float' = 0.001, incoming_projection_warning: 'float' = 0.001, port_gram_diagonal_warning: 'float' = 0.01, s_parameter_power_tolerance: 'float' = 1e-06) -> 'DiagnosticReport'

Return structured solver-quality diagnostics without printing.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``power_balance_tolerance``
     - ``float``
     - Optional
     - ``0.001``
     - Power balance tolerance control for this operation.
   * - ``projection_condition_warning``
     - ``float``
     - Optional
     - ``10000000000.0``
     - Projection condition warning control for this operation.
   * - ``projection_residual_warning``
     - ``float``
     - Optional
     - ``0.001``
     - Projection residual warning control for this operation.
   * - ``incoming_projection_warning``
     - ``float``
     - Optional
     - ``0.001``
     - Incoming projection warning control for this operation.
   * - ``port_gram_diagonal_warning``
     - ``float``
     - Optional
     - ``0.01``
     - Port gram diagonal warning control for this operation.
   * - ``s_parameter_power_tolerance``
     - ``float``
     - Optional
     - ``1e-06``
     - S parameter power tolerance control for this operation.

Returns: the selected data or diagnostic report.

``ScatteringResult.deembed``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringResult.deembed(*, left: 'float', right: 'float') -> "'ScatteringResult'"

Move left/right reference planes for a left-incident result.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``left``
     - ``float``
     - Required
     - ``—``
     - Left monitor/reference-plane position in metres.
   * - ``right``
     - ``float``
     - Required
     - ``—``
     - Right monitor/reference-plane position in metres.

Returns: a result with updated reference planes.

``FrequencySweepResult.plot``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    FrequencySweepResult.plot(*, component: 'str | None' = None, quantity: 'str' = 'magnitude')

Return an S11/S21 spectrum figure; frequency is measured in hertz.

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
     - ``'magnitude'``
     - Displayed field quantity: real, imag, magnitude/abs, or phase; static fields support real or magnitude.

Returns: a Matplotlib Figure.

``FrequencySweepResult.show``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    FrequencySweepResult.show(*, block: 'bool' = True)

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

Returns: the viewer controller or native process.

``FrequencySweepResult.save``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    FrequencySweepResult.save(path)

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

Returns: the written Path.

``FrequencySweepResult.S``
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    FrequencySweepResult.S(side: 'str', *, out_mode: 'int' = 0, in_mode: 'int' = 0) -> 'ComplexArray'

Return one indexed modal amplitude across the sweep.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``side``
     - ``str``
     - Required
     - ``—``
     - Port label, left or right.
   * - ``out_mode``
     - ``int``
     - Optional
     - ``0``
     - Out mode control for this operation.
   * - ``in_mode``
     - ``int``
     - Optional
     - ``0``
     - In mode control for this operation.

Returns: the selected data or diagnostic report.

Result data and diagnostics
---------------------------

``mesh_data.coordinates`` stores physical nodes in metres; ``elements`` stores
zero-based connectivity. ``axes`` identifies physical coordinate order.
``mesh_data.metadata['context']`` records material and boundary configuration.
The result is an inspection snapshot; editing it cannot restart a solver.

``E_total``, ``E_incident``, ``E_scattered`` and the matching ``H_*``
arrays have shape (3, samples), in V/m and A/m. ``coordinates`` uses (x, z),
in metres. ``S(side, out_mode, in_mode)`` returns a complex modal amplitude;
``S11`` and ``S21`` select the fundamental ports. ``reflection``,
``transmission``, ``absorption`` and ``power_balance_error`` are power ratios.
``reference_planes`` stores positions in m; ``port_betas`` stores complex
wavenumbers in rad/m. ``solve_info`` retains projection, algebraic, and
adaptive diagnostics. Sweep ``results[index]`` loads one case;
``frequencies_hz`` is the ordered frequency array. Sweep plotting accepts
S11/S21 and quantity real, imag, phase, magnitude, abs, or db.

Geometry and material values
----------------------------

Define reusable materials and shapes with ``cem_common`` before assigning them.
Use ``Material(name=..., epsilon=..., mu=...)`` for bulk media,
``materials.PEC`` or ``materials.PMC`` for ideal boundaries, and the
documented ``materials.copper``-style presets where SIBC is supported.
Continuous primitives and Boolean/transformed shapes live in
``cem_common.shapes``. Solver packages do not re-export these shared values.

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

Returns: a typed result; multi-case archives provide lazy case access.

Errors
------

Invalid inputs raise ``ConfigurationError`` or ``GeometryError`` where available.
Mesh and numerical failures raise the corresponding ``MeshError`` or
``SolverError``. ``NoResultError`` requires a successful solve first.
``PersistenceError`` identifies an incompatible or unreadable archive.
Viewer errors include the executable path or installation setting needed to
correct a launch failure. Saving and loading do not require an active GUI.
