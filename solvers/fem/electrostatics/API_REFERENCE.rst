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

``ElectrostaticSolver``, ``ElectrostaticResult``, ``Interval``, ``Rectangle``, ``Circle``, ``Polygon``, ``ElectrostaticSolverError``, ``GeometryError``, ``MeshError``, ``SolverError``, ``load_result``, ``NoResultError``, ``PersistenceError``.

Solver construction and operations
----------------------------------

``ElectrostaticSolver``
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ElectrostaticSolver(*, dim: 'int' = 2, x_range: 'float | Sequence[float]' = 1.0, y_range: 'float | Sequence[float] | None' = None, background_epsilon: 'float | Sequence[float] | Sequence[Sequence[float]]' = 1.0, outer_potential: 'float | None' = 0.0) -> 'None'

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
   * - ``background_epsilon``
     - ``float | Sequence[float] | Sequence[Sequence[float]]``
     - Optional
     - ``1.0``
     - Relative permittivity of the unfilled domain.
   * - ``outer_potential``
     - ``float | None``
     - Optional
     - ``0.0``
     - Exterior potential in volts; None permits natural boundaries.

Returns: a configured ``ElectrostaticSolver``.

``ElectrostaticSolver.set_potential``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ElectrostaticSolver.set_potential(*, region: 'RegionInput', potential: 'float', name: 'str | None' = None) -> 'PotentialRegion'

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``region``
     - ``RegionInput``
     - Required
     - ``—``
     - Geometry primitive or supported boundary name.
   * - ``potential``
     - ``float``
     - Required
     - ``—``
     - Prescribed electric potential in volts.
   * - ``name``
     - ``str | None``
     - Optional
     - ``None``
     - Optional name used for later identification and diagnostics.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``ElectrostaticSolver.add_object``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ElectrostaticSolver.add_object(*, region: 'RegionInput', epsilon=1.0, name: 'str | None' = None) -> 'MaterialRegion'

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``region``
     - ``RegionInput``
     - Required
     - ``—``
     - Geometry primitive or supported boundary name.
   * - ``epsilon``
     - ``float | complex | array-like / relative``
     - Optional
     - ``1.0``
     - Relative permittivity; supported scalar/tensor forms are described below.
   * - ``name``
     - ``str | None``
     - Optional
     - ``None``
     - Optional name used for later identification and diagnostics.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``ElectrostaticSolver.add_layer``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ElectrostaticSolver.add_layer(*, x_range, epsilon=1.0, name: 'str | None' = None)

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
   * - ``epsilon``
     - ``float | complex | array-like / relative``
     - Optional
     - ``1.0``
     - Relative permittivity; supported scalar/tensor forms are described below.
   * - ``name``
     - ``str | None``
     - Optional
     - ``None``
     - Optional name used for later identification and diagnostics.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``ElectrostaticSolver.add_rectangle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ElectrostaticSolver.add_rectangle(*, x_range, y_range, epsilon=1.0, name: 'str | None' = None)

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
   * - ``epsilon``
     - ``float | complex | array-like / relative``
     - Optional
     - ``1.0``
     - Relative permittivity; supported scalar/tensor forms are described below.
   * - ``name``
     - ``str | None``
     - Optional
     - ``None``
     - Optional name used for later identification and diagnostics.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``ElectrostaticSolver.add_circle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ElectrostaticSolver.add_circle(*, center, radius: 'float', epsilon=1.0, name: 'str | None' = None)

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
     - ``float``
     - Required
     - ``—``
     - Positive radius in metres.
   * - ``epsilon``
     - ``float | complex | array-like / relative``
     - Optional
     - ``1.0``
     - Relative permittivity; supported scalar/tensor forms are described below.
   * - ``name``
     - ``str | None``
     - Optional
     - ``None``
     - Optional name used for later identification and diagnostics.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``ElectrostaticSolver.add_polygon``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ElectrostaticSolver.add_polygon(*, points, epsilon=1.0, name: 'str | None' = None)

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
   * - ``epsilon``
     - ``float | complex | array-like / relative``
     - Optional
     - ``1.0``
     - Relative permittivity; supported scalar/tensor forms are described below.
   * - ``name``
     - ``str | None``
     - Optional
     - ``None``
     - Optional name used for later identification and diagnostics.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``ElectrostaticSolver.add_charge_density``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ElectrostaticSolver.add_charge_density(*, region: 'RegionInput', density: 'float', name: 'str | None' = None) -> 'ChargeRegion'

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``region``
     - ``RegionInput``
     - Required
     - ``—``
     - Geometry primitive or supported boundary name.
   * - ``density``
     - ``float``
     - Required
     - ``—``
     - Volume charge density in coulombs per cubic metre.
   * - ``name``
     - ``str | None``
     - Optional
     - ``None``
     - Optional name used for later identification and diagnostics.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``ElectrostaticSolver.remove``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ElectrostaticSolver.remove(item: 'MaterialRegion | PotentialRegion | ChargeRegion') -> 'None'

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``item``
     - ``MaterialRegion | PotentialRegion | ChargeRegion``
     - Required
     - ``—``
     - Item control for this operation.

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

``ElectrostaticSolver.compute_electric_field``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ElectrostaticSolver.compute_electric_field() -> 'NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]'

Returns: the configured geometry/excitation handle, or None for in-place configuration.

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

``Interval``
~~~~~~~~~~~~

.. code-block:: python

    Interval(x: 'tuple[float, float]') -> None

Interval(x: 'tuple[float, float]')

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``x``
     - ``tuple[float, float]``
     - Required
     - ``—``
     - X control for this operation.

Returns: an immutable geometry/material value for solver configuration.

``Rectangle``
~~~~~~~~~~~~~

.. code-block:: python

    Rectangle(x: 'tuple[float, float]', y: 'tuple[float, float]') -> None

Rectangle(x: 'tuple[float, float]', y: 'tuple[float, float]')

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``x``
     - ``tuple[float, float]``
     - Required
     - ``—``
     - X control for this operation.
   * - ``y``
     - ``tuple[float, float]``
     - Required
     - ``—``
     - Y control for this operation.

Returns: an immutable geometry/material value for solver configuration.

``Circle``
~~~~~~~~~~

.. code-block:: python

    Circle(center: 'tuple[float, float]', radius: 'float') -> None

Circle(center: 'tuple[float, float]', radius: 'float')

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``center``
     - ``tuple[float, float]``
     - Required
     - ``—``
     - Physical centre coordinates in metres.
   * - ``radius``
     - ``float``
     - Required
     - ``—``
     - Positive radius in metres.

Returns: an immutable geometry/material value for solver configuration.

``Polygon``
~~~~~~~~~~~

.. code-block:: python

    Polygon(points: 'tuple[tuple[float, float], ...]') -> None

Polygon(points: 'tuple[tuple[float, float], ...]')

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``points``
     - ``tuple[tuple[float, float], ...]``
     - Required
     - ``—``
     - Ordered polygon vertex coordinates in metres.

Returns: an immutable geometry/material value for solver configuration.

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
