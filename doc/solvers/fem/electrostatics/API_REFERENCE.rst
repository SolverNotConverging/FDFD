fem_electrostatics user API
===========================

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

Electrostatic fields are static. ``epsilon`` accepts a positive scalar, a
positive diagonal, or a real symmetric positive-definite tensor. Potential is
nodal; ``element_electric_field`` and ``element_displacement_field`` are cell
fields. ``electric_field`` and ``displacement_field`` are nodal averages.

Supported exports
-----------------

``ElectrostaticSolver``, ``ElectrostaticResult``, ``ElectrostaticSolverError``, ``GeometryError``, ``MeshError``, ``SolverError``, ``NoResultError``, ``PersistenceError``, ``load_result``.

Solver construction and operations
----------------------------------

``ElectrostaticSolver``
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ElectrostaticSolver(*, dim: 'int' = 2, x_range: 'float | Sequence[float]' = 1.0, y_range: 'float | Sequence[float] | None' = None, background_material: 'materials.Material' = Material(name='vacuum', epsilon=(1+0j), mu=(1+0j)), outer_potential: 'float | None' = 0.0) -> 'None'

Solve ``-div(epsilon_0 epsilon_r grad(phi)) = rho`` with P1 FEM.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``dim``
     - ``int``
     - Optional
     - ``2``
     - Electrostatic mesh dimension: 1 or 2.
   * - ``x_range``
     - ``float | Sequence[float]``
     - Optional
     - ``1.0``
     - Physical x extent or increasing bounds, in metres.
   * - ``y_range``
     - ``float | Sequence[float] | None``
     - Optional
     - ``None``
     - Physical y extent or increasing bounds, in metres.
   * - ``background_material``
     - ``materials.Material``
     - Optional
     - ``Material(name='vacuum', epsilon=(1+0j), mu=(1+0j))``
     - Predefined bulk Material assigned to unfilled space.
   * - ``outer_potential``
     - ``float | None``
     - Optional
     - ``0.0``
     - Exterior potential in volts; None permits natural boundaries.

Returns: a configured ``ElectrostaticSolver``.

``ElectrostaticSolver.add_geometry``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ElectrostaticSolver.add_geometry(*, shape, material, name=None, clip=False)

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

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``ElectrostaticSolver.add_layer``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ElectrostaticSolver.add_layer(*, x_range, material, name=None, clip=False)

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

``ElectrostaticSolver.add_rectangle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ElectrostaticSolver.add_rectangle(*, x_range, y_range, material, name=None, clip=False)

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
   * - ``y_range``
     - ``float | tuple[float, float] / m``
     - Required
     - ``—``
     - Physical y extent or increasing bounds, in metres.
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

``ElectrostaticSolver.add_circle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ElectrostaticSolver.add_circle(*, center, radius, material, name=None, clip=False)

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

``ElectrostaticSolver.add_polygon``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ElectrostaticSolver.add_polygon(*, points, material, name=None, clip=False)

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

``ElectrostaticSolver.set_material``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ElectrostaticSolver.set_material(*, geometry, material)

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

``ElectrostaticSolver.set_shape``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ElectrostaticSolver.set_shape(*, geometry, shape)

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

``ElectrostaticSolver.set_potential``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ElectrostaticSolver.set_potential(*, geometry, potential, name=None)

Prescribe volts on a conductor handle, shape, or named domain boundary.

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
   * - ``potential``
     - ``array-like or scalar``
     - Required
     - ``—``
     - Prescribed electric potential in volts.
   * - ``name``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - Optional name used for later identification and diagnostics.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``ElectrostaticSolver.add_charge_density``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ElectrostaticSolver.add_charge_density(*, geometry, density, name=None)

Assign volume charge density in C/m^3 to a dielectric handle or shape.

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
   * - ``density``
     - ``array-like or scalar``
     - Required
     - ``—``
     - Volume charge density in coulombs per cubic metre.
   * - ``name``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - Optional name used for later identification and diagnostics.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``ElectrostaticSolver.remove``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ElectrostaticSolver.remove(*, geometry)

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

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``ElectrostaticSolver.mesh``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ElectrostaticSolver.mesh(*, max_element_size: 'float | None' = None, material_aware: 'bool' = True, interface_refinement: 'float | None' = 0.7, boundary_refinement: 'float | None' = 0.5, interface_refinement_width: 'float | None' = None, boundary_refinement_width: 'float | None' = None) -> 'FEMMesh'

Discretize after geometry; high-Dk regions and boundaries refine locally.

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
   * - ``material_aware``
     - ``bool``
     - Optional
     - ``True``
     - Use material-dependent initial mesh sizing.
   * - ``interface_refinement``
     - ``float | None``
     - Optional
     - ``0.7``
     - Interface refinement control for this operation.
   * - ``boundary_refinement``
     - ``float | None``
     - Optional
     - ``0.5``
     - Boundary refinement control for this operation.
   * - ``interface_refinement_width``
     - ``float | None``
     - Optional
     - ``None``
     - Interface refinement width control for this operation.
   * - ``boundary_refinement_width``
     - ``float | None``
     - Optional
     - ``None``
     - Boundary refinement width control for this operation.

Returns: the initial mesh stored in ``mesh_data``.

``ElectrostaticSolver.solve``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ElectrostaticSolver.solve(*, linear_solver_tolerance: 'float' = 1e-10, max_refinements: 'int' = 2, adaptive_tolerance: 'float' = 0.05, marking_fraction: 'float' = 0.5, max_elements: 'int' = 200000) -> 'ElectrostaticResult'

Solve with bounded, solution-driven local refinement by default.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``linear_solver_tolerance``
     - ``float``
     - Optional
     - ``1e-10``
     - Algebraic linear-system residual tolerance.
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
   * - ``marking_fraction``
     - ``float``
     - Optional
     - ``0.5``
     - Fraction of squared error indicators marked for refinement.
   * - ``max_elements``
     - ``int``
     - Optional
     - ``200000``
     - Adaptive mesh element budget.

Returns: the physics-specific result stored in ``result``.

``ElectrostaticSolver.show``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ElectrostaticSolver.show(*, block: 'bool' = True)

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

``ElectrostaticResult.plot``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ElectrostaticResult.plot(*, component: 'str | None' = None, quantity: 'str' = 'real')

Return a potential, field, or mesh figure without opening a window.

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

``ElectrostaticResult.show``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ElectrostaticResult.show(*, block: 'bool' = True)

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

``ElectrostaticResult.save``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ElectrostaticResult.save(path)

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

``ElectrostaticResult.conductor_charge``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ElectrostaticResult.conductor_charge(name: 'str') -> 'float'

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``name``
     - ``str``
     - Required
     - ``—``
     - Optional name used for later identification and diagnostics.

Returns: conductor charge in coulombs (per metre for a 2D cross-section).

Result data and diagnostics
---------------------------

``mesh_data.coordinates`` stores physical nodes in metres; ``elements`` stores
zero-based connectivity. ``axes`` identifies physical coordinate order.
``mesh_data.metadata['context']`` records material and boundary configuration.
The result is an inspection snapshot; editing it cannot restart a solver.

``potential`` is nodal potential in V. ``electric_field`` and
``displacement_field`` are recovered nodal arrays (V/m and C/m²);
``element_electric_field`` and ``element_displacement_field`` retain cell fields.
``conductor_charges`` maps configured conductor names to charge, and
``energy`` records electrostatic energy (per unit transverse area in 1D,
per unit length in 2D). ``residual_norm`` is the algebraic residual;
``adaptive_history`` records mesh sizes, error indicators, and stopping status.

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
