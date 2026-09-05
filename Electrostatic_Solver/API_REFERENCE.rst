Electrostatic_Solver API reference
==================================

See `README.rst <README.rst>`_ for installation, an introduction, and runnable tutorials.

This reference covers package exports and the public research APIs exported by its modules, including constructors, methods, properties, result records, and exception categories. Private implementation helpers are excluded. Every callable/property is followed by an input table. Required means the caller must supply the argument; optional means a default exists. An argument accepting None is not necessarily optional.

The signatures show actual library defaults. All runnable examples explicitly set ``max_refinements=0``; the library default remains two adaptive refinements and a 0.05 discretization threshold. A zero budget still performs one solve and estimates its residual. Threshold convergence and a spent refinement budget are reported separately. Algebraic residual tolerances do not control adaptive mesh refinement.

Coordinates are metres, potential is volts, relative permittivity is dimensionless, and free charge density is C/m^3. In 1D charge/energy are per unit transverse area; in 2D they are per unit invariant depth.

Array and selector conventions
------------------------------

``ArrayLike`` accepts NumPy arrays and compatible Python sequences; ``FloatArray``/``IntArray``/``ComplexArray`` mean arrays of real/integer/complex values. ``NDArray[...]`` gives the dtype. ``MaterialInput`` means a scalar or three Cartesian diagonal entries; electrostatic Permittivity also supports a full symmetric positive-definite tensor. ``Literal[...]`` enumerates accepted choices. ``Sequence`` preserves order and ``Mapping`` associates keys with values.

Native scikit-fem coordinates/connectivity use (dimension, nodes)/(vertices per cell, cells). Standalone exported sample/mesh arrays generally use one row per point/cell. WaveFEM field arrays use (3, samples). Use the result's own mesh when indexing fields after adaptation. Python sequence indexing is zero based; explicit ``mode(number)`` and standalone modal visualization use their documented one-based selectors.

.. contents:: API index
   :local:
   :depth: 1

Electrostatic_Solver.ChargeRegion
---------------------------------

``Electrostatic_Solver.ChargeRegion``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Construct a ``ChargeRegion`` record. Its public data fields use the same names and types as the constructor inputs below. Solvers normally construct mesh/system/result records for you.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.ChargeRegion(id: 'int', name: 'str', shape: 'Shape', density: 'float') -> None

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``id``
     - Required
     - ``int``
     - Stable integer identifier attached to a geometry or physical-region record.
   * - ``name``
     - Required
     - ``str``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported.
   * - ``shape``
     - Required
     - ``Shape``
     - Geometry primitive or region selector identifying the physical support. Use the class/union in the signature; electrostatics also accepts named exterior boundaries.
   * - ``density``
     - Required
     - ``float``
     - Free volume-charge density in C/m^3. Positive values are positive free charge in the Poisson equation.

Returns: ``ChargeRegion``.

Electrostatic_Solver.Circle
---------------------------

``Electrostatic_Solver.Circle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define a circular region, optionally annular where an inner radius is supported. Use this immutable primitive in geometry/material placement APIs.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.Circle(center: 'tuple[float, float]', radius: 'float') -> None

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``center``
     - Required
     - ``tuple[float, float]``
     - Centre coordinates in metres, ordered as the package computational axes: x-y, x-z, or x-y-z.
   * - ``radius``
     - Required
     - ``float``
     - Positive radius in metres; inner_radius describes the hollow inner boundary where supported.

Returns: ``Circle``.

``Electrostatic_Solver.Circle.bounds``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's bounds value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.Circle.bounds

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - None
     - Not applicable
     - Not applicable
     - This operation takes no input arguments.

Returns: ``tuple[float, float, float, float]``.

``Electrostatic_Solver.Circle.contains``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``contains`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.Circle.contains(x: 'ArrayLike', y: 'ArrayLike') -> 'NDArray[np.bool_]'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``x``
     - Required
     - ``ArrayLike``
     - Physical x coordinate samples in metres; arrays must broadcast with the other coordinate arguments.
   * - ``y``
     - Required
     - ``ArrayLike``
     - Physical y coordinate samples in metres; arrays must broadcast with the other coordinate arguments.

Returns: ``NDArray[np.bool_]``.

Electrostatic_Solver.ElectrostaticResult
----------------------------------------

``Electrostatic_Solver.ElectrostaticResult``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Construct a ``ElectrostaticResult`` record. Its public data fields use the same names and types as the constructor inputs below. Solvers normally construct mesh/system/result records for you.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.ElectrostaticResult(mesh: 'FEMMesh', potential: 'NDArray[np.float64]', electric_field: 'NDArray[np.float64]', displacement_field: 'NDArray[np.float64]', reaction: 'NDArray[np.float64]', conductor_charges: 'Mapping[str, float]', energy: 'float', residual_norm: 'float', element_electric_field: 'NDArray[np.float64] | None' = None, element_displacement_field: 'NDArray[np.float64] | None' = None, adaptive_history: 'tuple[Mapping[str, object], ...]' = ()) -> None

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``mesh``
     - Required
     - ``FEMMesh``
     - Conforming FEM mesh object. Physical meshes carry SI coordinates; computational meshes may be scaled by the reference length for assembly.
   * - ``potential``
     - Required
     - ``NDArray[np.float64]``
     - Electrostatic potential in volts; a scalar for prescribed boundaries, a nodal array for solution data.
   * - ``electric_field``
     - Required
     - ``NDArray[np.float64]``
     - Electrostatic E in V/m or D in C/m^2. Nodal arrays have one row per node; element arrays have one row per simplex and preserve interface jumps.
   * - ``displacement_field``
     - Required
     - ``NDArray[np.float64]``
     - Electrostatic E in V/m or D in C/m^2. Nodal arrays have one row per node; element arrays have one row per simplex and preserve interface jumps.
   * - ``reaction``
     - Required
     - ``NDArray[np.float64]``
     - Named conductor charges or nodal weak-form reactions. Electrostatic 2D charges are per unit out-of-plane length.
   * - ``conductor_charges``
     - Required
     - ``Mapping[str, float]``
     - Named conductor charges or nodal weak-form reactions. Electrostatic 2D charges are per unit out-of-plane length.
   * - ``energy``
     - Required
     - ``float``
     - Stored electrostatic energy: joules in the 1D per-unit-area convention or joules per metre of invariant depth in 2D.
   * - ``residual_norm``
     - Required
     - ``float``
     - Stored numerical-validation diagnostic. Algebraic and Gauss residuals measure discrete equation defects; pml_fraction measures energy in absorbing layers.
   * - ``element_electric_field``
     - Optional
     - ``NDArray[np.float64] | None``
     - Electrostatic E in V/m or D in C/m^2. Nodal arrays have one row per node; element arrays have one row per simplex and preserve interface jumps. Default: ``None``.
   * - ``element_displacement_field``
     - Optional
     - ``NDArray[np.float64] | None``
     - Electrostatic E in V/m or D in C/m^2. Nodal arrays have one row per node; element arrays have one row per simplex and preserve interface jumps. Default: ``None``.
   * - ``adaptive_history``
     - Optional
     - ``tuple[Mapping[str, object], ...]``
     - Per-pass records of element count, residual estimator, and stopping status. Normally produced by solve rather than supplied manually. Default: ``()``.

Returns: ``ElectrostaticResult``.

``Electrostatic_Solver.ElectrostaticResult.adaptive_converged``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's adaptive converged value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.ElectrostaticResult.adaptive_converged

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - None
     - Not applicable
     - Not applicable
     - This operation takes no input arguments.

Returns: ``bool``.

``Electrostatic_Solver.ElectrostaticResult.adaptive_residual``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's adaptive residual value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.ElectrostaticResult.adaptive_residual

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - None
     - Not applicable
     - Not applicable
     - This operation takes no input arguments.

Returns: ``float | None``.

``Electrostatic_Solver.ElectrostaticResult.conductor_charge``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``conductor_charge`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.ElectrostaticResult.conductor_charge(name: 'str') -> 'float'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``name``
     - Required
     - ``str``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported.

Returns: ``float``.

``Electrostatic_Solver.ElectrostaticResult.coordinates``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's coordinates value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.ElectrostaticResult.coordinates

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - None
     - Not applicable
     - Not applicable
     - This operation takes no input arguments.

Returns: ``NDArray[np.float64]``.

``Electrostatic_Solver.ElectrostaticResult.elements``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's elements value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.ElectrostaticResult.elements

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - None
     - Not applicable
     - Not applicable
     - This operation takes no input arguments.

Returns: ``NDArray[np.int64]``.

Electrostatic_Solver.ElectrostaticSolver
----------------------------------------

``Electrostatic_Solver.ElectrostaticSolver``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Solve ``-div(epsilon_0 epsilon_r grad(phi)) = rho`` with P1 FEM.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.ElectrostaticSolver(mesh_size: 'Sequence[int] | int | None' = None, dim: 'int' = 2, *, domain: 'object | None' = None, background_permittivity: 'float | Sequence[float] | Sequence[Sequence[float]]' = 1.0, outer_potential: 'float | None' = 0.0) -> 'None'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``mesh_size``
     - Optional
     - ``Sequence[int] | int | None``
     - Initial mesh-resolution request. 1D uses an interval count; 2D accepts axis counts. Physical maximum size and wavelength constraints may increase the generated count. Default: ``None``.
   * - ``dim``
     - Optional
     - ``int``
     - Spatial dimension of the mesh/geometry. Electrostatics supports 1 or 2; periodic vector fields support 3. Default: ``2``.
   * - ``domain``
     - Optional
     - ``object | None``
     - Physical domain bounds. Electrostatics: one interval in 1D or a pair of intervals in 2D. WaveFEM callbacks: (x_span, z_span). Keyword-only. Default: ``None``.
   * - ``background_permittivity``
     - Optional
     - ``float | Sequence[float] | Sequence[Sequence[float]]``
     - Electrostatic relative-permittivity tensor: positive scalar, diagonal entries, or real symmetric positive-definite matrix of the geometry dimension. Keyword-only. Default: ``1.0``.
   * - ``outer_potential``
     - Optional
     - ``float | None``
     - Dirichlet potential on the complete exterior in volts. None leaves unselected exterior facets at the natural zero-flux boundary condition. Keyword-only. Default: ``0.0``.

Returns: ``ElectrostaticSolver``.

The constructor records only a physical domain and optional target mesh counts. Materials, charge regions, and fixed potentials remain continuous geometry until ``discretize`` (or an auto-discretizing ``solve``) is called. ``mesh_size`` and slice regions are retained for migration.

``Electrostatic_Solver.ElectrostaticSolver.add_charge_density``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add charge density; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.ElectrostaticSolver.add_charge_density(region: 'RegionInput', density: 'float', *, name: 'str | None' = None) -> 'ChargeRegion'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``region``
     - Required
     - ``RegionInput``
     - Geometry primitive or region selector identifying the physical support. Use the class/union in the signature; electrostatics also accepts named exterior boundaries.
   * - ``density``
     - Required
     - ``float``
     - Free volume-charge density in C/m^3. Positive values are positive free charge in the Poisson equation.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Returns: ``ChargeRegion``.

``Electrostatic_Solver.ElectrostaticSolver.add_object``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add object; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.ElectrostaticSolver.add_object(region: 'RegionInput', erxx: 'float' = 1.0, eryy: 'float | None' = None, *, erxy: 'float' = 0.0, permittivity: 'float | Sequence[float] | Sequence[Sequence[float]] | None' = None, name: 'str | None' = None) -> 'MaterialRegion'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``region``
     - Required
     - ``RegionInput``
     - Geometry primitive or region selector identifying the physical support. Use the class/union in the signature; electrostatics also accepts named exterior boundaries.
   * - ``erxx``
     - Optional
     - ``float``
     - Legacy electrostatic diagonal permittivity entry. erxx alone does not make a 2D material isotropic; use permittivity=scalar for that. Default: ``1.0``.
   * - ``eryy``
     - Optional
     - ``float | None``
     - Legacy electrostatic diagonal permittivity entry. erxx alone does not make a 2D material isotropic; use permittivity=scalar for that. Default: ``None``.
   * - ``erxy``
     - Optional
     - ``float``
     - Symmetric off-diagonal electrostatic relative-permittivity entry. The full tensor must remain positive definite. Keyword-only. Default: ``0.0``.
   * - ``permittivity``
     - Optional
     - ``float | Sequence[float] | Sequence[Sequence[float]] | None``
     - Electrostatic relative-permittivity tensor: positive scalar, diagonal entries, or real symmetric positive-definite matrix of the geometry dimension. Keyword-only. Default: ``None``.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Returns: ``MaterialRegion``.

``Electrostatic_Solver.ElectrostaticSolver.compute_electric_field``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute electric field; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.ElectrostaticSolver.compute_electric_field() -> 'NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - None
     - Not applicable
     - Not applicable
     - This operation takes no input arguments.

Returns: ``NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]``.

``Electrostatic_Solver.ElectrostaticSolver.coordinates``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's coordinates value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.ElectrostaticSolver.coordinates

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - None
     - Not applicable
     - Not applicable
     - This operation takes no input arguments.

Returns: ``NDArray[np.float64]``.

``Electrostatic_Solver.ElectrostaticSolver.discretize``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Discretize after geometry; high-Dk regions and boundaries refine locally.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.ElectrostaticSolver.discretize(*, max_element_size: 'float | None' = None, material_aware: 'bool' = True, interface_refinement: 'float | None' = 0.7, boundary_refinement: 'float | None' = 0.5, interface_refinement_width: 'float | None' = None, boundary_refinement_width: 'float | None' = None) -> 'FEMMesh'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``max_element_size``
     - Optional
     - ``float | None``
     - Physical element-edge length in metres. A maximum target is an upper sizing request; material, boundary, and wavelength constraints may produce smaller cells. Keyword-only. Default: ``None``.
   * - ``material_aware``
     - Optional
     - ``bool``
     - Enable element-size reduction in high-index/high-permittivity material regions while retaining the global maximum-edge cap. Keyword-only. Default: ``True``.
   * - ``interface_refinement``
     - Optional
     - ``float | None``
     - Optional size multiplier in (0, 1] near conductor/exterior boundaries or material interfaces. None disables this sizing field. Keyword-only. Default: ``0.7``.
   * - ``boundary_refinement``
     - Optional
     - ``float | None``
     - Optional size multiplier in (0, 1] near conductor/exterior boundaries or material interfaces. None disables this sizing field. Keyword-only. Default: ``0.5``.
   * - ``interface_refinement_width``
     - Optional
     - ``float | None``
     - Physical distance in metres over which the local mesh-size target transitions back to the surrounding target; None selects the mesher default when permitted. Keyword-only. Default: ``None``.
   * - ``boundary_refinement_width``
     - Optional
     - ``float | None``
     - Physical distance in metres over which the local mesh-size target transitions back to the surrounding target; None selects the mesher default when permitted. Keyword-only. Default: ``None``.

Returns: ``FEMMesh``.

``Electrostatic_Solver.ElectrostaticSolver.elements``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's elements value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.ElectrostaticSolver.elements

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - None
     - Not applicable
     - Not applicable
     - This operation takes no input arguments.

Returns: ``NDArray[np.int64]``.

``Electrostatic_Solver.ElectrostaticSolver.remove``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Remove ; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.ElectrostaticSolver.remove(item: 'MaterialRegion | PotentialRegion | ChargeRegion') -> 'None'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``item``
     - Required
     - ``MaterialRegion | PotentialRegion | ChargeRegion``
     - Previously returned region/boundary/PML handle to remove, or an index/key for a container operation as indicated by the method.

Returns: ``None``.

``Electrostatic_Solver.ElectrostaticSolver.set_potential``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set potential; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.ElectrostaticSolver.set_potential(region: 'RegionInput', potential_value: 'float', *, name: 'str | None' = None) -> 'PotentialRegion'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``region``
     - Required
     - ``RegionInput``
     - Geometry primitive or region selector identifying the physical support. Use the class/union in the signature; electrostatics also accepts named exterior boundaries.
   * - ``potential_value``
     - Required
     - ``float``
     - Electrostatic potential in volts; a scalar for prescribed boundaries, a nodal array for solution data.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Returns: ``PotentialRegion``.

``Electrostatic_Solver.ElectrostaticSolver.solve``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Solve with bounded, solution-driven local refinement by default.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.ElectrostaticSolver.solve(tol: 'float' = 1e-10, max_iter: 'int | None' = None, *, adaptive: 'bool' = True, max_refinements: 'int' = 2, adaptive_tolerance: 'float' = 0.05, marking_fraction: 'float' = 0.5, max_elements: 'int' = 200000) -> 'ElectrostaticResult'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``tol``
     - Optional
     - ``float``
     - Positive algebraic residual tolerance for the direct electrostatic linear solve; does not control mesh adaptation.
   * - ``max_iter``
     - Optional
     - ``int | None``
     - Iteration or Arnoldi-restart budget. None selects the backend default; the direct electrostatic solve accepts max_iter for compatibility. Default: ``None``.
   * - ``adaptive``
     - Optional
     - ``bool``
     - Enable the electrostatic estimate/refine loop. False returns the initial-mesh solution even if the estimator exceeds the threshold. Keyword-only. Default: ``True``.
   * - ``max_refinements``
     - Optional
     - ``int``
     - Maximum mesh updates after the initial solve; nonnegative integer. Zero keeps the initial mesh. Solver default is 2; examples explicitly use 0. Keyword-only. Default: ``2``.
   * - ``adaptive_tolerance``
     - Optional
     - ``float``
     - Positive finite threshold for the normalized discretization estimator. Stop when residual <= threshold. Independent of algebraic tolerances; exhausting the budget is not convergence. Keyword-only. Default: ``0.05``.
   * - ``marking_fraction``
     - Optional
     - ``float``
     - Bulk-marking fraction in (0, 1]; select the smallest cell set carrying this fraction of the squared residual indicator. Keyword-only. Default: ``0.5``.
   * - ``max_elements``
     - Optional
     - ``int``
     - Positive upper bound on the final electrostatic element count; a refinement that would exceed it is not applied. Keyword-only. Default: ``200000``.

Returns: ``ElectrostaticResult``.

Normal displacement jumps and the Poisson volume residual select cells by bulk marking. ``adaptive_tolerance`` controls that relative indicator; ``tol`` independently controls the algebraic solve. Inspect ``result.adaptive_history`` for the stopping reason. Use ``adaptive=False`` to keep the supplied mesh exactly.

``Electrostatic_Solver.ElectrostaticSolver.visualize``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``visualize`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.ElectrostaticSolver.visualize(*, show: 'bool' = True) -> 'object'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``show``
     - Optional
     - ``bool``
     - Enable display of the figure, mesh overlay, or colorbar respectively. show=False returns plotting objects without opening a window. Keyword-only. Default: ``True``.

Returns: ``Python object described by this operation``.

Electrostatic_Solver.ElectrostaticSolverError
---------------------------------------------

``Electrostatic_Solver.ElectrostaticSolverError``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Base exception for this package.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.ElectrostaticSolverError(*args: 'object')

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``*args``
     - Optional
     - ``object``
     - Positional exception payload or extra positional values explicitly accepted by this callable. Numerical solve wrappers list their forwarded parameters individually instead.

Returns: ``ElectrostaticSolverError``.

Electrostatic_Solver.FEMMesh
----------------------------

``Electrostatic_Solver.FEMMesh``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Construct a ``FEMMesh`` record. Its public data fields use the same names and types as the constructor inputs below. Solvers normally construct mesh/system/result records for you.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.FEMMesh(mesh: 'object', nodes: 'FloatArray', elements: 'IntArray', element_tags: 'NDArray[np.int32]', physical_names: 'dict[int, str]', info: 'MeshInfo', geometry_revision: 'int') -> None

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``mesh``
     - Required
     - ``object``
     - Conforming FEM mesh object. Physical meshes carry SI coordinates; computational meshes may be scaled by the reference length for assembly.
   * - ``nodes``
     - Required
     - ``FloatArray``
     - Mesh-node coordinates. Native scikit-fem arrays are dimension by node; standalone result coordinates are commonly node by dimension.
   * - ``elements``
     - Required
     - ``IntArray``
     - Integer simplex connectivity. scikit-fem uses vertices-per-cell by cell; standalone exported geometry commonly uses cell by vertices-per-cell.
   * - ``element_tags``
     - Required
     - ``NDArray[np.int32]``
     - Stable Gmsh physical material tag(s) linking mesh cells to material regions.
   * - ``physical_names``
     - Required
     - ``dict[int, str]``
     - Mapping from physical tags to human-readable material or boundary names.
   * - ``info``
     - Required
     - ``MeshInfo``
     - Structured metadata/diagnostic container associated with the mesh or result. solve_info/metadata retain applied adaptive controls and stopping history.
   * - ``geometry_revision``
     - Required
     - ``int``
     - Geometry version captured when this mesh was built; stale versions invalidate cached systems and results.

Returns: ``FEMMesh``.

Electrostatic_Solver.GeometryError
----------------------------------

``Electrostatic_Solver.GeometryError``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The continuous geometry or material definition is invalid.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.GeometryError(*args: 'object')

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``*args``
     - Optional
     - ``object``
     - Positional exception payload or extra positional values explicitly accepted by this callable. Numerical solve wrappers list their forwarded parameters individually instead.

Returns: ``GeometryError``.

Electrostatic_Solver.Interval
-----------------------------

``Electrostatic_Solver.Interval``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define a one-dimensional physical interval. Use this immutable primitive in geometry/material placement APIs.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.Interval(x: 'tuple[float, float]') -> None

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``x``
     - Required
     - ``tuple[float, float]``
     - Physical x-axis bounds (minimum, maximum), in metres; both finite with maximum > minimum.

Returns: ``Interval``.

``Electrostatic_Solver.Interval.bounds``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's bounds value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.Interval.bounds

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - None
     - Not applicable
     - Not applicable
     - This operation takes no input arguments.

Returns: ``tuple[float, float]``.

``Electrostatic_Solver.Interval.contains``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``contains`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.Interval.contains(x: 'ArrayLike', *_: 'ArrayLike') -> 'NDArray[np.bool_]'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``x``
     - Required
     - ``ArrayLike``
     - Physical x coordinate samples in metres; arrays must broadcast with the other coordinate arguments.
   * - ``*_``
     - Optional
     - ``ArrayLike``
     - Unused extra coordinates accepted for a common shape-evaluation interface; they do not affect a 1D containment test.

Returns: ``NDArray[np.bool_]``.

Electrostatic_Solver.MaterialRegion
-----------------------------------

``Electrostatic_Solver.MaterialRegion``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Construct a ``MaterialRegion`` record. Its public data fields use the same names and types as the constructor inputs below. Solvers normally construct mesh/system/result records for you.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.MaterialRegion(id: 'int', name: 'str', shape: 'Shape', permittivity: 'Permittivity') -> None

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``id``
     - Required
     - ``int``
     - Stable integer identifier attached to a geometry or physical-region record.
   * - ``name``
     - Required
     - ``str``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported.
   * - ``shape``
     - Required
     - ``Shape``
     - Geometry primitive or region selector identifying the physical support. Use the class/union in the signature; electrostatics also accepts named exterior boundaries.
   * - ``permittivity``
     - Required
     - ``Permittivity``
     - Electrostatic relative-permittivity tensor: positive scalar, diagonal entries, or real symmetric positive-definite matrix of the geometry dimension.

Returns: ``MaterialRegion``.

Electrostatic_Solver.MeshError
------------------------------

``Electrostatic_Solver.MeshError``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Gmsh could not produce a valid conforming mesh.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.MeshError(*args: 'object')

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``*args``
     - Optional
     - ``object``
     - Positional exception payload or extra positional values explicitly accepted by this callable. Numerical solve wrappers list their forwarded parameters individually instead.

Returns: ``MeshError``.

Electrostatic_Solver.MeshInfo
-----------------------------

``Electrostatic_Solver.MeshInfo``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Construct a ``MeshInfo`` record. Its public data fields use the same names and types as the constructor inputs below. Solvers normally construct mesh/system/result records for you.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.MeshInfo(nodes: 'int', elements: 'int', minimum_edge: 'float', maximum_edge: 'float', requested_maximum_edge: 'float', material_aware: 'bool', interface_refinement: 'float | None', boundary_refinement: 'float | None', material_element_sizes: 'dict[str, float]') -> None

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``nodes``
     - Required
     - ``int``
     - Mesh-node coordinates. Native scikit-fem arrays are dimension by node; standalone result coordinates are commonly node by dimension.
   * - ``elements``
     - Required
     - ``int``
     - Integer simplex connectivity. scikit-fem uses vertices-per-cell by cell; standalone exported geometry commonly uses cell by vertices-per-cell.
   * - ``minimum_edge``
     - Required
     - ``float``
     - Physical element-edge length in metres. A maximum target is an upper sizing request; material, boundary, and wavelength constraints may produce smaller cells.
   * - ``maximum_edge``
     - Required
     - ``float``
     - Physical element-edge length in metres. A maximum target is an upper sizing request; material, boundary, and wavelength constraints may produce smaller cells.
   * - ``requested_maximum_edge``
     - Required
     - ``float``
     - Physical element-edge length in metres. A maximum target is an upper sizing request; material, boundary, and wavelength constraints may produce smaller cells.
   * - ``material_aware``
     - Required
     - ``bool``
     - Enable element-size reduction in high-index/high-permittivity material regions while retaining the global maximum-edge cap.
   * - ``interface_refinement``
     - Required
     - ``float | None``
     - Optional size multiplier in (0, 1] near conductor/exterior boundaries or material interfaces. None disables this sizing field.
   * - ``boundary_refinement``
     - Required
     - ``float | None``
     - Optional size multiplier in (0, 1] near conductor/exterior boundaries or material interfaces. None disables this sizing field.
   * - ``material_element_sizes``
     - Required
     - ``dict[str, float]``
     - Mapping from physical material tags to local maximum element sizes in metres.

Returns: ``MeshInfo``.

Electrostatic_Solver.NotDiscretizedError
----------------------------------------

``Electrostatic_Solver.NotDiscretizedError``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An operation requires a current finite-element mesh.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.NotDiscretizedError(*args: 'object')

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``*args``
     - Optional
     - ``object``
     - Positional exception payload or extra positional values explicitly accepted by this callable. Numerical solve wrappers list their forwarded parameters individually instead.

Returns: ``NotDiscretizedError``.

Electrostatic_Solver.Permittivity
---------------------------------

``Electrostatic_Solver.Permittivity``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Real symmetric relative-permittivity tensor.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.Permittivity(tensor: 'tuple[tuple[float, ...], ...]') -> None

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``tensor``
     - Required
     - ``tuple[tuple[float, ...], ...]``
     - Electrostatic relative-permittivity tensor: positive scalar, diagonal entries, or real symmetric positive-definite matrix of the geometry dimension.

Returns: ``Permittivity``.

``Electrostatic_Solver.Permittivity.array``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's array value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.Permittivity.array

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - None
     - Not applicable
     - Not applicable
     - This operation takes no input arguments.

Returns: ``FloatArray``.

``Electrostatic_Solver.Permittivity.dk_scale``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's dk scale value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.Permittivity.dk_scale

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - None
     - Not applicable
     - Not applicable
     - This operation takes no input arguments.

Returns: ``float``.

``Electrostatic_Solver.Permittivity.from_input``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``from_input`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.Permittivity.from_input(value: 'float | Sequence[float] | Sequence[Sequence[float]]', dim: 'int') -> "'Permittivity'"

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``value``
     - Required
     - ``float | Sequence[float] | Sequence[Sequence[float]]``
     - Scalar or array to validate, transform, interpolate, or store. It must satisfy the owning operation and the expected type in this table.
   * - ``dim``
     - Required
     - ``int``
     - Spatial dimension of the mesh/geometry. Electrostatics supports 1 or 2; periodic vector fields support 3.

Returns: ``Permittivity``.

Electrostatic_Solver.Polygon
----------------------------

``Electrostatic_Solver.Polygon``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define a closed polygon from ordered physical vertices. Use this immutable primitive in geometry/material placement APIs.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.Polygon(points: 'tuple[tuple[float, float], ...]') -> None

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``points``
     - Required
     - ``tuple[tuple[float, float], ...]``
     - Coordinates used by the object or evaluation operation. Mesh geometry uses physical metres; low-level FE operators use their basis coordinate scale. See the array-shape conventions.

Returns: ``Polygon``.

``Electrostatic_Solver.Polygon.bounds``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's bounds value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.Polygon.bounds

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - None
     - Not applicable
     - Not applicable
     - This operation takes no input arguments.

Returns: ``tuple[float, float, float, float]``.

``Electrostatic_Solver.Polygon.contains``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``contains`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.Polygon.contains(x: 'ArrayLike', y: 'ArrayLike') -> 'NDArray[np.bool_]'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``x``
     - Required
     - ``ArrayLike``
     - Physical x coordinate samples in metres; arrays must broadcast with the other coordinate arguments.
   * - ``y``
     - Required
     - ``ArrayLike``
     - Physical y coordinate samples in metres; arrays must broadcast with the other coordinate arguments.

Returns: ``NDArray[np.bool_]``.

Electrostatic_Solver.PotentialRegion
------------------------------------

``Electrostatic_Solver.PotentialRegion``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Construct a ``PotentialRegion`` record. Its public data fields use the same names and types as the constructor inputs below. Solvers normally construct mesh/system/result records for you.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.PotentialRegion(id: 'int', name: 'str', shape: 'Shape | str', value: 'float') -> None

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``id``
     - Required
     - ``int``
     - Stable integer identifier attached to a geometry or physical-region record.
   * - ``name``
     - Required
     - ``str``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported.
   * - ``shape``
     - Required
     - ``Shape | str``
     - Geometry primitive or region selector identifying the physical support. Use the class/union in the signature; electrostatics also accepts named exterior boundaries.
   * - ``value``
     - Required
     - ``float``
     - Scalar or array to validate, transform, interpolate, or store. It must satisfy the owning operation and the expected type in this table.

Returns: ``PotentialRegion``.

Electrostatic_Solver.Rectangle
------------------------------

``Electrostatic_Solver.Rectangle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define an axis-aligned physical rectangle. Use this immutable primitive in geometry/material placement APIs.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.Rectangle(x: 'tuple[float, float]', y: 'tuple[float, float]') -> None

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``x``
     - Required
     - ``tuple[float, float]``
     - Physical x-axis bounds (minimum, maximum), in metres; both finite with maximum > minimum.
   * - ``y``
     - Required
     - ``tuple[float, float]``
     - Physical y-axis bounds (minimum, maximum), in metres; both finite with maximum > minimum.

Returns: ``Rectangle``.

``Electrostatic_Solver.Rectangle.bounds``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's bounds value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.Rectangle.bounds

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - None
     - Not applicable
     - Not applicable
     - This operation takes no input arguments.

Returns: ``tuple[float, float, float, float]``.

``Electrostatic_Solver.Rectangle.contains``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``contains`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.Rectangle.contains(x: 'ArrayLike', y: 'ArrayLike') -> 'NDArray[np.bool_]'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``x``
     - Required
     - ``ArrayLike``
     - Physical x coordinate samples in metres; arrays must broadcast with the other coordinate arguments.
   * - ``y``
     - Required
     - ``ArrayLike``
     - Physical y coordinate samples in metres; arrays must broadcast with the other coordinate arguments.

Returns: ``NDArray[np.bool_]``.

Electrostatic_Solver.SolverError
--------------------------------

``Electrostatic_Solver.SolverError``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The assembled electrostatic boundary-value problem cannot be solved.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.SolverError(*args: 'object')

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``*args``
     - Optional
     - ``object``
     - Positional exception payload or extra positional values explicitly accepted by this callable. Numerical solve wrappers list their forwarded parameters individually instead.

Returns: ``SolverError``.

Electrostatic_Solver.geometry.GeometryModel
-------------------------------------------

``Electrostatic_Solver.geometry.GeometryModel``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mutable continuous scene shared by the public solver.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.geometry.GeometryModel(dim: 'int', x_span: 'tuple[float, float]', y_span: 'tuple[float, float] | None', background: 'Permittivity') -> 'None'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``dim``
     - Required
     - ``int``
     - Spatial dimension of the mesh/geometry. Electrostatics supports 1 or 2; periodic vector fields support 3.
   * - ``x_span``
     - Required
     - ``tuple[float, float]``
     - Physical axis bounds in metres, with upper > lower. Range-based solver constructors also accept a positive extent for (0, extent).
   * - ``y_span``
     - Required
     - ``tuple[float, float] | None``
     - Physical axis bounds in metres, with upper > lower. Range-based solver constructors also accept a positive extent for (0, extent).
   * - ``background``
     - Required
     - ``Permittivity``
     - In placement methods, True adds a z-invariant background region/sheet; False adds an actual-device perturbation. In material/geometry constructors, this is the exterior Material object.

Returns: ``GeometryModel``.

``Electrostatic_Solver.geometry.GeometryModel.add_change_listener``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add change listener; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.geometry.GeometryModel.add_change_listener(callback: 'Callable[[], None]') -> 'None'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``callback``
     - Required
     - ``Callable[[], None]``
     - Vectorized material/field callback evaluated at the coordinates supplied by the calling API; return the scalar, diagonal array, or field shape specified by that operation.

Returns: ``None``.

``Electrostatic_Solver.geometry.GeometryModel.add_charge``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add charge; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.geometry.GeometryModel.add_charge(shape: 'Shape', density: 'float', *, name: 'str | None' = None) -> 'ChargeRegion'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``shape``
     - Required
     - ``Shape``
     - Geometry primitive or region selector identifying the physical support. Use the class/union in the signature; electrostatics also accepts named exterior boundaries.
   * - ``density``
     - Required
     - ``float``
     - Free volume-charge density in C/m^3. Positive values are positive free charge in the Poisson equation.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Returns: ``ChargeRegion``.

``Electrostatic_Solver.geometry.GeometryModel.add_material``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add material; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.geometry.GeometryModel.add_material(shape: 'Shape', permittivity: 'Permittivity', *, name: 'str | None' = None) -> 'MaterialRegion'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``shape``
     - Required
     - ``Shape``
     - Geometry primitive or region selector identifying the physical support. Use the class/union in the signature; electrostatics also accepts named exterior boundaries.
   * - ``permittivity``
     - Required
     - ``Permittivity``
     - Electrostatic relative-permittivity tensor: positive scalar, diagonal entries, or real symmetric positive-definite matrix of the geometry dimension.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Returns: ``MaterialRegion``.

``Electrostatic_Solver.geometry.GeometryModel.add_potential``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add potential; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.geometry.GeometryModel.add_potential(shape: 'Shape | str', value: 'float', *, name: 'str | None' = None) -> 'PotentialRegion'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``shape``
     - Required
     - ``Shape | str``
     - Geometry primitive or region selector identifying the physical support. Use the class/union in the signature; electrostatics also accepts named exterior boundaries.
   * - ``value``
     - Required
     - ``float``
     - Scalar or array to validate, transform, interpolate, or store. It must satisfy the owning operation and the expected type in this table.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Returns: ``PotentialRegion``.

``Electrostatic_Solver.geometry.GeometryModel.all_area_shapes``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``all_area_shapes`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.geometry.GeometryModel.all_area_shapes() -> 'tuple[Shape, ...]'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - None
     - Not applicable
     - Not applicable
     - This operation takes no input arguments.

Returns: ``tuple[Shape, ...]``.

``Electrostatic_Solver.geometry.GeometryModel.charge_at``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``charge_at`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.geometry.GeometryModel.charge_at(coordinates: 'FloatArray') -> 'FloatArray'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``coordinates``
     - Required
     - ``FloatArray``
     - Coordinates used by the object or evaluation operation. Mesh geometry uses physical metres; low-level FE operators use their basis coordinate scale. See the array-shape conventions.

Returns: ``FloatArray``.

``Electrostatic_Solver.geometry.GeometryModel.material_indices_at``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``material_indices_at`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.geometry.GeometryModel.material_indices_at(coordinates: 'FloatArray') -> 'NDArray[np.int32]'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``coordinates``
     - Required
     - ``FloatArray``
     - Coordinates used by the object or evaluation operation. Mesh geometry uses physical metres; low-level FE operators use their basis coordinate scale. See the array-shape conventions.

Returns: ``NDArray[np.int32]``.

``Electrostatic_Solver.geometry.GeometryModel.material_table``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's material table value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.geometry.GeometryModel.material_table

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - None
     - Not applicable
     - Not applicable
     - This operation takes no input arguments.

Returns: ``dict[int, Permittivity]``.

``Electrostatic_Solver.geometry.GeometryModel.remove``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Remove ; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.geometry.GeometryModel.remove(item: 'MaterialRegion | PotentialRegion | ChargeRegion') -> 'None'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``item``
     - Required
     - ``MaterialRegion | PotentialRegion | ChargeRegion``
     - Previously returned region/boundary/PML handle to remove, or an index/key for a container operation as indicated by the method.

Returns: ``None``.

``Electrostatic_Solver.geometry.GeometryModel.validate_shape``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``validate_shape`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.geometry.GeometryModel.validate_shape(shape: 'Shape') -> 'None'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``shape``
     - Required
     - ``Shape``
     - Geometry primitive or region selector identifying the physical support. Use the class/union in the signature; electrostatics also accepts named exterior boundaries.

Returns: ``None``.

Electrostatic_Solver.meshing.discretize_1d
------------------------------------------

``Electrostatic_Solver.meshing.discretize_1d``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate a conforming Gmsh line mesh with high-Dk local sizing.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.meshing.discretize_1d(geometry: 'GeometryModel', *, max_element_size: 'float', material_aware: 'bool' = True, interface_refinement: 'float | None' = 0.7, boundary_refinement: 'float | None' = 0.5) -> 'FEMMesh'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``geometry``
     - Required
     - ``GeometryModel``
     - Geometry model containing physical bounds, material regions, conductor boundaries, PMLs, and sizing requests.
   * - ``max_element_size``
     - Required
     - ``float``
     - Physical element-edge length in metres. A maximum target is an upper sizing request; material, boundary, and wavelength constraints may produce smaller cells. Keyword-only.
   * - ``material_aware``
     - Optional
     - ``bool``
     - Enable element-size reduction in high-index/high-permittivity material regions while retaining the global maximum-edge cap. Keyword-only. Default: ``True``.
   * - ``interface_refinement``
     - Optional
     - ``float | None``
     - Optional size multiplier in (0, 1] near conductor/exterior boundaries or material interfaces. None disables this sizing field. Keyword-only. Default: ``0.7``.
   * - ``boundary_refinement``
     - Optional
     - ``float | None``
     - Optional size multiplier in (0, 1] near conductor/exterior boundaries or material interfaces. None disables this sizing field. Keyword-only. Default: ``0.5``.

Returns: ``FEMMesh``.

Electrostatic_Solver.meshing.discretize_2d
------------------------------------------

``Electrostatic_Solver.meshing.discretize_2d``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate a conforming triangular mesh with high-Dk and boundary refinement.

Signature (defaults are library defaults):

.. code-block:: text

   Electrostatic_Solver.meshing.discretize_2d(geometry: 'GeometryModel', *, max_element_size: 'float', material_aware: 'bool' = True, interface_refinement: 'float | None' = 0.7, boundary_refinement: 'float | None' = 0.5, interface_refinement_width: 'float | None' = None, boundary_refinement_width: 'float | None' = None) -> 'FEMMesh'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``geometry``
     - Required
     - ``GeometryModel``
     - Geometry model containing physical bounds, material regions, conductor boundaries, PMLs, and sizing requests.
   * - ``max_element_size``
     - Required
     - ``float``
     - Physical element-edge length in metres. A maximum target is an upper sizing request; material, boundary, and wavelength constraints may produce smaller cells. Keyword-only.
   * - ``material_aware``
     - Optional
     - ``bool``
     - Enable element-size reduction in high-index/high-permittivity material regions while retaining the global maximum-edge cap. Keyword-only. Default: ``True``.
   * - ``interface_refinement``
     - Optional
     - ``float | None``
     - Optional size multiplier in (0, 1] near conductor/exterior boundaries or material interfaces. None disables this sizing field. Keyword-only. Default: ``0.7``.
   * - ``boundary_refinement``
     - Optional
     - ``float | None``
     - Optional size multiplier in (0, 1] near conductor/exterior boundaries or material interfaces. None disables this sizing field. Keyword-only. Default: ``0.5``.
   * - ``interface_refinement_width``
     - Optional
     - ``float | None``
     - Physical distance in metres over which the local mesh-size target transitions back to the surrounding target; None selects the mesher default when permitted. Keyword-only. Default: ``None``.
   * - ``boundary_refinement_width``
     - Optional
     - ``float | None``
     - Physical distance in metres over which the local mesh-size target transitions back to the surrounding target; None selects the mesher default when permitted. Keyword-only. Default: ``None``.

Returns: ``FEMMesh``.

Export aliases and constants
----------------------------

Aliases below have exactly the same input tables and return contracts as their targets. Constants/type aliases are values, not calls, and take no input arguments.

.. list-table:: Exports
   :header-rows: 1

   * - Name
     - Value or target
   * - ``Electrostatic_Solver.exceptions.ElectrostaticSolverError``
     - ``Electrostatic_Solver.ElectrostaticSolverError``
   * - ``Electrostatic_Solver.exceptions.GeometryError``
     - ``Electrostatic_Solver.GeometryError``
   * - ``Electrostatic_Solver.exceptions.MeshError``
     - ``Electrostatic_Solver.MeshError``
   * - ``Electrostatic_Solver.exceptions.NotDiscretizedError``
     - ``Electrostatic_Solver.NotDiscretizedError``
   * - ``Electrostatic_Solver.exceptions.SolverError``
     - ``Electrostatic_Solver.SolverError``
   * - ``Electrostatic_Solver.geometry.ChargeRegion``
     - ``Electrostatic_Solver.ChargeRegion``
   * - ``Electrostatic_Solver.geometry.Circle``
     - ``Electrostatic_Solver.Circle``
   * - ``Electrostatic_Solver.geometry.Interval``
     - ``Electrostatic_Solver.Interval``
   * - ``Electrostatic_Solver.geometry.MaterialRegion``
     - ``Electrostatic_Solver.MaterialRegion``
   * - ``Electrostatic_Solver.geometry.Permittivity``
     - ``Electrostatic_Solver.Permittivity``
   * - ``Electrostatic_Solver.geometry.Polygon``
     - ``Electrostatic_Solver.Polygon``
   * - ``Electrostatic_Solver.geometry.PotentialRegion``
     - ``Electrostatic_Solver.PotentialRegion``
   * - ``Electrostatic_Solver.geometry.Rectangle``
     - ``Electrostatic_Solver.Rectangle``
   * - ``Electrostatic_Solver.meshing.FEMMesh``
     - ``Electrostatic_Solver.FEMMesh``
   * - ``Electrostatic_Solver.meshing.MeshInfo``
     - ``Electrostatic_Solver.MeshInfo``
   * - ``Electrostatic_Solver.results.ElectrostaticResult``
     - ``Electrostatic_Solver.ElectrostaticResult``
   * - ``Electrostatic_Solver.solver.ElectrostaticSolver``
     - ``Electrostatic_Solver.ElectrostaticSolver``
   * - ``Electrostatic_Solver.EPSILON_0``
     - ``8.8541878128e-12``
   * - ``Electrostatic_Solver.geometry.Shape``
     - ``'Interval | Rectangle | Circle | Polygon'``
   * - ``Electrostatic_Solver.solver.EPSILON_0``
     - ``8.8541878128e-12``
