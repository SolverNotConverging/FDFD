FEM_Periodic_Solver API reference
=================================

See `README.rst <README.rst>`_ for installation, an introduction, and runnable tutorials.

This reference covers package exports and the public research APIs exported by its modules, including constructors, methods, properties, result records, and exception categories. Private implementation helpers are excluded. Every callable/property is followed by an input table. Required means the caller must supply the argument; optional means a default exists. An argument accepting None is not necessarily optional.

The signatures show actual library defaults. All runnable examples explicitly set ``max_refinements=0``; the library default remains two adaptive refinements and a 0.05 discretization threshold. A zero budget still performs one solve and estimates its residual. Threshold convergence and a spent refinement budget are reported separately. Algebraic residual tolerances do not control adaptive mesh refinement.

Coordinates are metres and ordinary frequency is Hz. The phasor convention is exp(+j omega t - j beta z); passive forward attenuation has nonpositive Im(beta). Relative material tensors use Cartesian x,y,z order.

Array and selector conventions
------------------------------

``ArrayLike`` accepts NumPy arrays and compatible Python sequences; ``FloatArray``/``IntArray``/``ComplexArray`` mean arrays of real/integer/complex values. ``NDArray[...]`` gives the dtype. ``MaterialInput`` means a scalar or three Cartesian diagonal entries; electrostatic Permittivity also supports a full symmetric positive-definite tensor. ``Literal[...]`` enumerates accepted choices. ``Sequence`` preserves order and ``Mapping`` associates keys with values.

Native scikit-fem coordinates/connectivity use (dimension, nodes)/(vertices per cell, cells). Standalone exported sample/mesh arrays generally use one row per point/cell. WaveFEM field arrays use (3, samples). Use the result's own mesh when indexing fields after adaptation. Python sequence indexing is zero based; explicit ``mode(number)`` and standalone modal visualization use their documented one-based selectors.

.. contents:: API index
   :local:
   :depth: 1

FEM_Periodic_Solver.BackendCapabilityError
------------------------------------------

``FEM_Periodic_Solver.BackendCapabilityError``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A requested feature is intentionally unavailable in this backend.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.BackendCapabilityError(*args: 'object')

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

Returns: ``BackendCapabilityError``.

FEM_Periodic_Solver.BoundaryRegion
----------------------------------

``FEM_Periodic_Solver.BoundaryRegion``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Construct a ``BoundaryRegion`` record. Its public data fields use the same names and types as the constructor inputs below. Solvers normally construct mesh/system/result records for you.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.BoundaryRegion(id: 'int', name: 'str', shape: 'Shape2D | Shape3D', kind: 'str') -> None

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
     - ``Shape2D | Shape3D``
     - Geometry primitive or region selector identifying the physical support. Use the class/union in the signature; electrostatics also accepts named exterior boundaries.
   * - ``kind``
     - Required
     - ``str``
     - Object/boundary/scene kind. Use the permitted Literal values or the documented selector for the owning class.

Returns: ``BoundaryRegion``.

FEM_Periodic_Solver.Box
-----------------------

``FEM_Periodic_Solver.Box``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define an axis-aligned three-dimensional box. Use this immutable primitive in geometry/material placement APIs.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.Box(x: 'tuple[float, float]', y: 'tuple[float, float]', z: 'tuple[float, float]') -> None

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
   * - ``z``
     - Required
     - ``tuple[float, float]``
     - Physical z-axis bounds (minimum, maximum), in metres; both finite with maximum > minimum.

Returns: ``Box``.

``FEM_Periodic_Solver.Box.bounds``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's bounds value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.Box.bounds

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

Returns: ``tuple[float, float, float, float, float, float]``.

``FEM_Periodic_Solver.Box.contains``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``contains`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.Box.contains(x: 'ArrayLike', y: 'ArrayLike', z: 'ArrayLike') -> 'NDArray[np.bool_]'

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
   * - ``z``
     - Required
     - ``ArrayLike``
     - Physical z coordinate samples in metres; arrays must broadcast with the other coordinate arguments.

Returns: ``NDArray[np.bool_]``.

FEM_Periodic_Solver.Circle
--------------------------

``FEM_Periodic_Solver.Circle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define a circular region, optionally annular where an inner radius is supported. Use this immutable primitive in geometry/material placement APIs.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.Circle(center: 'tuple[float, float]', radius: 'float', inner_radius: 'float | None' = None) -> None

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
   * - ``inner_radius``
     - Optional
     - ``float | None``
     - Positive radius in metres; inner_radius describes the hollow inner boundary where supported. Default: ``None``.

Returns: ``Circle``.

``FEM_Periodic_Solver.Circle.bounds``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's bounds value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.Circle.bounds

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

``FEM_Periodic_Solver.Circle.contains``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``contains`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.Circle.contains(x: 'ArrayLike', z: 'ArrayLike') -> 'NDArray[np.bool_]'

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
   * - ``z``
     - Required
     - ``ArrayLike``
     - Physical z coordinate samples in metres; arrays must broadcast with the other coordinate arguments.

Returns: ``NDArray[np.bool_]``.

FEM_Periodic_Solver.ConfigurationError
--------------------------------------

``FEM_Periodic_Solver.ConfigurationError``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A solver option or material is invalid.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.ConfigurationError(*args: 'object')

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

Returns: ``ConfigurationError``.

FEM_Periodic_Solver.Cylinder
----------------------------

``FEM_Periodic_Solver.Cylinder``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define a cylinder with the physical axis and dimensions supplied below. Use this immutable primitive in geometry/material placement APIs.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.Cylinder(center: 'tuple[float, float]', radius: 'float', z: 'tuple[float, float]') -> None

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
   * - ``z``
     - Required
     - ``tuple[float, float]``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature.

Returns: ``Cylinder``.

``FEM_Periodic_Solver.Cylinder.bounds``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's bounds value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.Cylinder.bounds

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

Returns: ``tuple[float, float, float, float, float, float]``.

``FEM_Periodic_Solver.Cylinder.contains``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``contains`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.Cylinder.contains(x: 'ArrayLike', y: 'ArrayLike', z: 'ArrayLike') -> 'NDArray[np.bool_]'

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
   * - ``z``
     - Required
     - ``ArrayLike``
     - Physical z coordinate samples in metres; arrays must broadcast with the other coordinate arguments.

Returns: ``NDArray[np.bool_]``.

FEM_Periodic_Solver.FEMPeriodicMesh2D
-------------------------------------

``FEM_Periodic_Solver.FEMPeriodicMesh2D``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Physical mesh plus exact periodic and boundary associations.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.FEMPeriodicMesh2D(mesh: 'MeshTri', element_tags: 'NDArray[np.int32]', physical_names: 'dict[int, str]', boundary_facets: 'dict[str, NDArray[np.int64]]', slave_nodes: 'NDArray[np.int64]', master_nodes: 'NDArray[np.int64]', info: 'MeshInfo', geometry_revision: 'int') -> None

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``mesh``
     - Required
     - ``MeshTri``
     - Conforming FEM mesh object. Physical meshes carry SI coordinates; computational meshes may be scaled by the reference length for assembly.
   * - ``element_tags``
     - Required
     - ``NDArray[np.int32]``
     - Stable Gmsh physical material tag(s) linking mesh cells to material regions.
   * - ``physical_names``
     - Required
     - ``dict[int, str]``
     - Mapping from physical tags to human-readable material or boundary names.
   * - ``boundary_facets``
     - Required
     - ``dict[str, NDArray[np.int64]]``
     - Boundary-kind/name to facet-index mapping, using the owning scikit-fem mesh numbering.
   * - ``slave_nodes``
     - Required
     - ``NDArray[np.int64]``
     - Matched periodic seam indices. Pair arrays store slave/master correspondence; edge pairs additionally retain orientation through the signed constraint operator.
   * - ``master_nodes``
     - Required
     - ``NDArray[np.int64]``
     - Matched periodic seam indices. Pair arrays store slave/master correspondence; edge pairs additionally retain orientation through the signed constraint operator.
   * - ``info``
     - Required
     - ``MeshInfo``
     - Structured metadata/diagnostic container associated with the mesh or result. solve_info/metadata retain applied adaptive controls and stopping history.
   * - ``geometry_revision``
     - Required
     - ``int``
     - Geometry version captured when this mesh was built; stale versions invalidate cached systems and results.

Returns: ``FEMPeriodicMesh2D``.

``FEM_Periodic_Solver.FEMPeriodicMesh2D.elements``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's elements value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.FEMPeriodicMesh2D.elements

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

``FEM_Periodic_Solver.FEMPeriodicMesh2D.nodes``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's nodes value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.FEMPeriodicMesh2D.nodes

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

``FEM_Periodic_Solver.FEMPeriodicMesh2D.period``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's period value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.FEMPeriodicMesh2D.period

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

FEM_Periodic_Solver.FEMPeriodicMesh3D
-------------------------------------

``FEM_Periodic_Solver.FEMPeriodicMesh3D``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A physical tetrahedral mesh and its periodic-topology metadata.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.FEMPeriodicMesh3D(mesh: 'MeshTet', element_tags: 'NDArray[np.int32]', physical_names: 'Mapping[int, str]', boundary_facets: 'Mapping[str, NDArray[np.int64]]', periodic_node_pairs: 'NDArray[np.int64]', periodic_edge_pairs: 'NDArray[np.int64]', edge_nodes: 'NDArray[np.int64]', cell_edges: 'NDArray[np.int64]', cell_edge_signs: 'NDArray[np.int8]', info: 'MeshInfo3D', geometry_revision: 'int') -> None

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``mesh``
     - Required
     - ``MeshTet``
     - Conforming FEM mesh object. Physical meshes carry SI coordinates; computational meshes may be scaled by the reference length for assembly.
   * - ``element_tags``
     - Required
     - ``NDArray[np.int32]``
     - Stable Gmsh physical material tag(s) linking mesh cells to material regions.
   * - ``physical_names``
     - Required
     - ``Mapping[int, str]``
     - Mapping from physical tags to human-readable material or boundary names.
   * - ``boundary_facets``
     - Required
     - ``Mapping[str, NDArray[np.int64]]``
     - Boundary-kind/name to facet-index mapping, using the owning scikit-fem mesh numbering.
   * - ``periodic_node_pairs``
     - Required
     - ``NDArray[np.int64]``
     - Matched periodic seam indices. Pair arrays store slave/master correspondence; edge pairs additionally retain orientation through the signed constraint operator.
   * - ``periodic_edge_pairs``
     - Required
     - ``NDArray[np.int64]``
     - Matched periodic seam indices. Pair arrays store slave/master correspondence; edge pairs additionally retain orientation through the signed constraint operator.
   * - ``edge_nodes``
     - Required
     - ``NDArray[np.int64]``
     - Integer edge connectivity or edge-index arrays matching the canonical Nedelec ordering stored in the mesh/system.
   * - ``cell_edges``
     - Required
     - ``NDArray[np.int64]``
     - Integer edge connectivity or edge-index arrays matching the canonical Nedelec ordering stored in the mesh/system.
   * - ``cell_edge_signs``
     - Required
     - ``NDArray[np.int8]``
     - Orientation factors (+1 or -1) mapping local/periodic Nedelec edges to canonical global directions.
   * - ``info``
     - Required
     - ``MeshInfo3D``
     - Structured metadata/diagnostic container associated with the mesh or result. solve_info/metadata retain applied adaptive controls and stopping history.
   * - ``geometry_revision``
     - Required
     - ``int``
     - Geometry version captured when this mesh was built; stale versions invalidate cached systems and results.

Returns: ``PeriodicMesh3D``.

``FEM_Periodic_Solver.FEMPeriodicMesh3D.elements``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's elements value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.FEMPeriodicMesh3D.elements

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

``FEM_Periodic_Solver.FEMPeriodicMesh3D.nodes``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's nodes value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.FEMPeriodicMesh3D.nodes

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

FEM_Periodic_Solver.FEMPeriodicSolverError
------------------------------------------

``FEM_Periodic_Solver.FEMPeriodicSolverError``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Base class for errors raised by the periodic FEM package.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.FEMPeriodicSolverError(*args: 'object')

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

Returns: ``FEMPeriodicSolverError``.

FEM_Periodic_Solver.GeometryError
---------------------------------

``FEM_Periodic_Solver.GeometryError``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A geometry object cannot be represented by the periodic cell.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.GeometryError(*args: 'object')

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

FEM_Periodic_Solver.GeometryModel2D
-----------------------------------

``FEM_Periodic_Solver.GeometryModel2D``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Construct a ``GeometryModel2D`` record. Its public data fields use the same names and types as the constructor inputs below. Solvers normally construct mesh/system/result records for you.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.GeometryModel2D(x_span: 'float | Sequence[float]', z_span: 'float | Sequence[float]', background: 'Material') -> 'None'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``x_span``
     - Required
     - ``float | Sequence[float]``
     - Physical axis bounds in metres, with upper > lower. Range-based solver constructors also accept a positive extent for (0, extent).
   * - ``z_span``
     - Required
     - ``float | Sequence[float]``
     - Physical axis bounds in metres, with upper > lower. Range-based solver constructors also accept a positive extent for (0, extent).
   * - ``background``
     - Required
     - ``Material``
     - In placement methods, True adds a z-invariant background region/sheet; False adds an actual-device perturbation. In material/geometry constructors, this is the exterior Material object.

Returns: ``GeometryModel2D``.

``FEM_Periodic_Solver.GeometryModel2D.add_boundary``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add boundary; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.GeometryModel2D.add_boundary(shape: 'Shape2D', kind: 'str', *, name: 'str | None' = None) -> 'BoundaryRegion'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``shape``
     - Required
     - ``Shape2D``
     - Geometry primitive or region selector identifying the physical support. Use the class/union in the signature; electrostatics also accepts named exterior boundaries.
   * - ``kind``
     - Required
     - ``str``
     - Object/boundary/scene kind. Use the permitted Literal values or the documented selector for the owning class.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Returns: ``BoundaryRegion``.

``FEM_Periodic_Solver.GeometryModel2D.add_change_listener``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add change listener; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.GeometryModel2D.add_change_listener(listener: 'Callable[[], None]') -> 'None'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``listener``
     - Required
     - ``Callable[[], None]``
     - No-argument callback notified after geometry changes so owning solvers can invalidate cached data.

Returns: ``None``.

``FEM_Periodic_Solver.GeometryModel2D.add_mesh_refinement``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add mesh refinement; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.GeometryModel2D.add_mesh_refinement(shape: 'Shape2D', max_element_size: 'float', *, name: 'str | None' = None) -> 'MeshRefinement'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``shape``
     - Required
     - ``Shape2D``
     - Geometry primitive or region selector identifying the physical support. Use the class/union in the signature; electrostatics also accepts named exterior boundaries.
   * - ``max_element_size``
     - Required
     - ``float``
     - Physical element-edge length in metres. A maximum target is an upper sizing request; material, boundary, and wavelength constraints may produce smaller cells.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Returns: ``MeshRefinement``.

``FEM_Periodic_Solver.GeometryModel2D.add_pml``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add pml; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.GeometryModel2D.add_pml(spec: 'PMLSpec') -> 'PMLSpec'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``spec``
     - Required
     - ``PMLSpec``
     - PML specification/layout describing the absorbing strips and their grading profile.

Returns: ``PMLSpec``.

``FEM_Periodic_Solver.GeometryModel2D.add_region``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add region; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.GeometryModel2D.add_region(shape: 'Shape2D', material: 'Material', *, name: 'str | None' = None) -> 'Region'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``shape``
     - Required
     - ``Shape2D``
     - Geometry primitive or region selector identifying the physical support. Use the class/union in the signature; electrostatics also accepts named exterior boundaries.
   * - ``material``
     - Required
     - ``Material``
     - Material object defining relative electric and magnetic response. Exterior fills points outside placed material regions.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Returns: ``Region``.

``FEM_Periodic_Solver.GeometryModel2D.material_at``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``material_at`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.GeometryModel2D.material_at(x: 'ArrayLike', z: 'ArrayLike') -> 'tuple[NDArray[np.complex128], NDArray[np.complex128]]'

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
   * - ``z``
     - Required
     - ``ArrayLike``
     - Physical z coordinate samples in metres; arrays must broadcast with the other coordinate arguments.

Returns: ``tuple[NDArray[np.complex128], NDArray[np.complex128]]``.

``FEM_Periodic_Solver.GeometryModel2D.pml_mask``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``pml_mask`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.GeometryModel2D.pml_mask(x: 'ArrayLike') -> 'NDArray[np.bool_]'

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
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature.

Returns: ``NDArray[np.bool_]``.

``FEM_Periodic_Solver.GeometryModel2D.remove``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Remove ; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.GeometryModel2D.remove(handle: 'Region | BoundaryRegion | MeshRefinement | PMLSpec') -> 'None'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``handle``
     - Required
     - ``Region | BoundaryRegion | MeshRefinement | PMLSpec``
     - Previously returned region/boundary/PML handle to remove, or an index/key for a container operation as indicated by the method.

Returns: ``None``.

``FEM_Periodic_Solver.GeometryModel2D.set_outer_boundary``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set outer boundary; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.GeometryModel2D.set_outer_boundary(kind: 'str') -> 'None'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``kind``
     - Required
     - ``str``
     - Object/boundary/scene kind. Use the permitted Literal values or the documented selector for the owning class.

Returns: ``None``.

``FEM_Periodic_Solver.GeometryModel2D.transformed_material_at``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``transformed_material_at`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.GeometryModel2D.transformed_material_at(x: 'ArrayLike', z: 'ArrayLike') -> 'tuple[NDArray[np.complex128], NDArray[np.complex128]]'

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
   * - ``z``
     - Required
     - ``ArrayLike``
     - Physical z coordinate samples in metres; arrays must broadcast with the other coordinate arguments.

Returns: ``tuple[NDArray[np.complex128], NDArray[np.complex128]]``.

FEM_Periodic_Solver.GeometryModel3D
-----------------------------------

``FEM_Periodic_Solver.GeometryModel3D``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Continuous ``x-y-z`` geometry for one cell periodic in ``z``.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.GeometryModel3D(x_span: 'float | Sequence[float]', y_span: 'float | Sequence[float]', z_span: 'float | Sequence[float]', background: 'Material') -> 'None'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``x_span``
     - Required
     - ``float | Sequence[float]``
     - Physical axis bounds in metres, with upper > lower. Range-based solver constructors also accept a positive extent for (0, extent).
   * - ``y_span``
     - Required
     - ``float | Sequence[float]``
     - Physical axis bounds in metres, with upper > lower. Range-based solver constructors also accept a positive extent for (0, extent).
   * - ``z_span``
     - Required
     - ``float | Sequence[float]``
     - Physical axis bounds in metres, with upper > lower. Range-based solver constructors also accept a positive extent for (0, extent).
   * - ``background``
     - Required
     - ``Material``
     - In placement methods, True adds a z-invariant background region/sheet; False adds an actual-device perturbation. In material/geometry constructors, this is the exterior Material object.

Returns: ``GeometryModel3D``.

``FEM_Periodic_Solver.GeometryModel3D.add_boundary``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add boundary; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.GeometryModel3D.add_boundary(shape: 'Shape3D', kind: 'str', *, name: 'str | None' = None) -> 'BoundaryRegion'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``shape``
     - Required
     - ``Shape3D``
     - Geometry primitive or region selector identifying the physical support. Use the class/union in the signature; electrostatics also accepts named exterior boundaries.
   * - ``kind``
     - Required
     - ``str``
     - Object/boundary/scene kind. Use the permitted Literal values or the documented selector for the owning class.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Returns: ``BoundaryRegion``.

``FEM_Periodic_Solver.GeometryModel3D.add_change_listener``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add change listener; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.GeometryModel3D.add_change_listener(listener: 'Callable[[], None]') -> 'None'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``listener``
     - Required
     - ``Callable[[], None]``
     - No-argument callback notified after geometry changes so owning solvers can invalidate cached data.

Returns: ``None``.

``FEM_Periodic_Solver.GeometryModel3D.add_mesh_refinement``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add mesh refinement; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.GeometryModel3D.add_mesh_refinement(shape: 'Shape3D', max_element_size: 'float', *, name: 'str | None' = None) -> 'MeshRefinement'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``shape``
     - Required
     - ``Shape3D``
     - Geometry primitive or region selector identifying the physical support. Use the class/union in the signature; electrostatics also accepts named exterior boundaries.
   * - ``max_element_size``
     - Required
     - ``float``
     - Physical element-edge length in metres. A maximum target is an upper sizing request; material, boundary, and wavelength constraints may produce smaller cells.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Returns: ``MeshRefinement``.

``FEM_Periodic_Solver.GeometryModel3D.add_pml``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add pml; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.GeometryModel3D.add_pml(spec: 'PMLSpec') -> 'PMLSpec'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``spec``
     - Required
     - ``PMLSpec``
     - PML specification/layout describing the absorbing strips and their grading profile.

Returns: ``PMLSpec``.

``FEM_Periodic_Solver.GeometryModel3D.add_region``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add region; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.GeometryModel3D.add_region(shape: 'Shape3D', material: 'Material', *, name: 'str | None' = None) -> 'Region'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``shape``
     - Required
     - ``Shape3D``
     - Geometry primitive or region selector identifying the physical support. Use the class/union in the signature; electrostatics also accepts named exterior boundaries.
   * - ``material``
     - Required
     - ``Material``
     - Material object defining relative electric and magnetic response. Exterior fills points outside placed material regions.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Returns: ``Region``.

``FEM_Periodic_Solver.GeometryModel3D.material_at``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``material_at`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.GeometryModel3D.material_at(x: 'ArrayLike', y: 'ArrayLike', z: 'ArrayLike') -> 'tuple[NDArray[np.complex128], NDArray[np.complex128]]'

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
   * - ``z``
     - Required
     - ``ArrayLike``
     - Physical z coordinate samples in metres; arrays must broadcast with the other coordinate arguments.

Returns: ``tuple[NDArray[np.complex128], NDArray[np.complex128]]``.

``FEM_Periodic_Solver.GeometryModel3D.pml_mask``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``pml_mask`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.GeometryModel3D.pml_mask(x: 'ArrayLike', y: 'ArrayLike') -> 'NDArray[np.bool_]'

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
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature.
   * - ``y``
     - Required
     - ``ArrayLike``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature.

Returns: ``NDArray[np.bool_]``.

``FEM_Periodic_Solver.GeometryModel3D.remove``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Remove ; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.GeometryModel3D.remove(handle: 'Region | BoundaryRegion | MeshRefinement | PMLSpec') -> 'None'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``handle``
     - Required
     - ``Region | BoundaryRegion | MeshRefinement | PMLSpec``
     - Previously returned region/boundary/PML handle to remove, or an index/key for a container operation as indicated by the method.

Returns: ``None``.

``FEM_Periodic_Solver.GeometryModel3D.set_outer_boundary``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set outer boundary; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.GeometryModel3D.set_outer_boundary(kind: 'str') -> 'None'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``kind``
     - Required
     - ``str``
     - Object/boundary/scene kind. Use the permitted Literal values or the documented selector for the owning class.

Returns: ``None``.

``FEM_Periodic_Solver.GeometryModel3D.transformed_material_at``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``transformed_material_at`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.GeometryModel3D.transformed_material_at(x: 'ArrayLike', y: 'ArrayLike', z: 'ArrayLike') -> 'tuple[NDArray[np.complex128], NDArray[np.complex128]]'

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
   * - ``z``
     - Required
     - ``ArrayLike``
     - Physical z coordinate samples in metres; arrays must broadcast with the other coordinate arguments.

Returns: ``tuple[NDArray[np.complex128], NDArray[np.complex128]]``.

FEM_Periodic_Solver.H5ValidationReport
--------------------------------------

``FEM_Periodic_Solver.H5ValidationReport``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Summary returned after a successful schema validation.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.H5ValidationReport(path: 'Path', schema_major: 'int', schema_minor: 'int', case_count: 'int', mode_count: 'int', deep: 'bool') -> None

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``path``
     - Required
     - ``Path``
     - Filesystem destination/source for HDF5 persistence or viewer launch. A directory is accepted only by viewer/directory-inspection APIs; None follows the method-specific default.
   * - ``schema_major``
     - Required
     - ``int``
     - Stored HDF5 schema version components used to validate reader/writer compatibility.
   * - ``schema_minor``
     - Required
     - ``int``
     - Stored HDF5 schema version components used to validate reader/writer compatibility.
   * - ``case_count``
     - Required
     - ``int``
     - Number of independently stored cases in the HDF5 archive.
   * - ``mode_count``
     - Required
     - ``int``
     - Number of modes or candidate eigenpairs requested; a positive integer. Candidate pools may include roots later rejected by validation.
   * - ``deep``
     - Required
     - ``bool``
     - Validate stored array payloads and cross-dataset invariants in addition to lightweight HDF5 schema checks.

Returns: ``H5ValidationReport``.

FEM_Periodic_Solver.Material
----------------------------

``FEM_Periodic_Solver.Material``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Relative diagonal permittivity and permeability.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.Material(epsilon: 'MaterialInput' = 1.0, mu: 'MaterialInput' = 1.0) -> 'None'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``epsilon``
     - Optional
     - ``MaterialInput``
     - Relative permittivity. Scalar or Cartesian diagonal material inputs are supported where the signature permits; callbacks return scalar/broadcast-compatible values at physical coordinates. Default: ``1.0``.
   * - ``mu``
     - Optional
     - ``MaterialInput``
     - Relative permeability, scalar or Cartesian diagonal where supported. WaveFEM scattering requires equal actual and background permeability. Default: ``1.0``.

Returns: ``Material``.

Complex values follow the package's ``exp(+1j*omega*t)`` convention.

``FEM_Periodic_Solver.Material.eps_array``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``eps_array`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.Material.eps_array() -> 'np.ndarray'

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

Returns: ``np.ndarray``.

``FEM_Periodic_Solver.Material.isotropic``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's isotropic value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.Material.isotropic

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

``FEM_Periodic_Solver.Material.mu_array``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``mu_array`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.Material.mu_array() -> 'np.ndarray'

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

Returns: ``np.ndarray``.

FEM_Periodic_Solver.MeshError
-----------------------------

``FEM_Periodic_Solver.MeshError``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Gmsh could not create a conforming periodic mesh.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.MeshError(*args: 'object')

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

FEM_Periodic_Solver.MeshInfo
----------------------------

``FEM_Periodic_Solver.MeshInfo``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Construct a ``MeshInfo`` record. Its public data fields use the same names and types as the constructor inputs below. Solvers normally construct mesh/system/result records for you.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.MeshInfo(nodes: 'int', elements: 'int', minimum_edge: 'float', maximum_edge: 'float', requested_maximum_edge: 'float', element_order: 'int' = 1) -> None

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
   * - ``element_order``
     - Optional
     - ``int``
     - Finite-element polynomial-order selection. Standalone 2D modes and WaveFEM scattering accept 1 (N1/P1) or 2 (N2/P2); other backends retain their fixed compatible spaces. Default: ``1``.

Returns: ``MeshInfo``.

FEM_Periodic_Solver.MeshInfo3D
------------------------------

``FEM_Periodic_Solver.MeshInfo3D``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Construct a ``MeshInfo3D`` record. Its public data fields use the same names and types as the constructor inputs below. Solvers normally construct mesh/system/result records for you.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.MeshInfo3D(nodes: 'int', elements: 'int', minimum_edge: 'float', maximum_edge: 'float', requested_maximum_edge: 'float', element_order: 'int' = 1, material_aware: 'bool' = True, refinement_regions: 'int' = 0) -> None

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
   * - ``element_order``
     - Optional
     - ``int``
     - Finite-element polynomial-order selection. Standalone 2D modes and WaveFEM scattering accept 1 (N1/P1) or 2 (N2/P2); other backends retain their fixed compatible spaces. Default: ``1``.
   * - ``material_aware``
     - Optional
     - ``bool``
     - Enable element-size reduction in high-index/high-permittivity material regions while retaining the global maximum-edge cap. Default: ``True``.
   * - ``refinement_regions``
     - Optional
     - ``int``
     - Count or collection of explicit geometry-based local sizing regions, as indicated by the mesh metadata type. Default: ``0``.

Returns: ``MeshInfo3D``.

FEM_Periodic_Solver.MeshRefinement
----------------------------------

``FEM_Periodic_Solver.MeshRefinement``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Construct a ``MeshRefinement`` record. Its public data fields use the same names and types as the constructor inputs below. Solvers normally construct mesh/system/result records for you.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.MeshRefinement(id: 'int', name: 'str', shape: 'Shape2D | Shape3D', max_element_size: 'float') -> None

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
     - ``Shape2D | Shape3D``
     - Geometry primitive or region selector identifying the physical support. Use the class/union in the signature; electrostatics also accepts named exterior boundaries.
   * - ``max_element_size``
     - Required
     - ``float``
     - Physical element-edge length in metres. A maximum target is an upper sizing request; material, boundary, and wavelength constraints may produce smaller cells.

Returns: ``MeshRefinement``.

FEM_Periodic_Solver.Mode
------------------------

``FEM_Periodic_Solver.Mode``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Construct a ``PeriodicMode`` record. Its public data fields use the same names and types as the constructor inputs below. Solvers normally construct mesh/system/result records for you.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.Mode(neff: 'complex', k0: 'float', period: 'float', fields: 'PeriodicSampledFields', coefficients: 'ComplexArray', index: 'int' = 1, polarization: 'str | None' = None, power: 'complex | None' = None, direction: 'str' = 'indeterminate', normalization: 'str' = 'unnormalized', residual: 'float | None' = None, gauss_residual: 'float | None' = None, pml_fraction: 'float' = 0.0, metadata: 'Mapping[str, Any]' = <factory>) -> None

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``neff``
     - Required
     - ``complex``
     - Dimensionless effective index beta/k0. A guess selects roots near that complex value; None uses the solver estimate.
   * - ``k0``
     - Required
     - ``float``
     - Vacuum wavenumber in rad/m, used for coordinate scaling and material/PML evaluation.
   * - ``period``
     - Required
     - ``float``
     - Longitudinal periodic-cell length in metres; finite and positive.
   * - ``fields``
     - Required
     - ``PeriodicSampledFields``
     - Complex electromagnetic field samples or FE field objects. Cartesian components follow x,y,z order; left/right denote the selected trace side.
   * - ``coefficients``
     - Required
     - ``ComplexArray``
     - Complex finite-element coefficient vector or coefficient values, ordered exactly as the associated basis/system. Full vectors include constrained/periodic copies.
   * - ``index``
     - Optional
     - ``int``
     - Mode, case, array, or mapping selector. Python indexing is zero based; explicit mode(number) and standalone visualization use their documented one-based numbers. Default: ``1``.
   * - ``polarization``
     - Optional
     - ``str | None``
     - Mode-family label: TE/TM/hybrid or propagating/evanescent as appropriate to the owning result or solver. Default: ``None``.
   * - ``power``
     - Optional
     - ``complex | None``
     - Modal or electromagnetic power in W/m of invariant length for 2.5D fields. Complex power retains reactive flux; power ratios are reported separately. Default: ``None``.
   * - ``direction``
     - Optional
     - ``str``
     - Propagation filter forward/backward/all for mode solves; for PML placement, the selected transverse side(s), such as x-, x+, or x. Default: ``'indeterminate'``.
   * - ``normalization``
     - Optional
     - ``str``
     - Modal/field normalization convention or flag. Unit-longitudinal-power normalization applies only to modes with usable real power. Default: ``'unnormalized'``.
   * - ``residual``
     - Optional
     - ``float | None``
     - Stored numerical-validation diagnostic. Algebraic and Gauss residuals measure discrete equation defects; pml_fraction measures energy in absorbing layers. Default: ``None``.
   * - ``gauss_residual``
     - Optional
     - ``float | None``
     - Stored numerical-validation diagnostic. Algebraic and Gauss residuals measure discrete equation defects; pml_fraction measures energy in absorbing layers. Default: ``None``.
   * - ``pml_fraction``
     - Optional
     - ``float``
     - Stored numerical-validation diagnostic. Algebraic and Gauss residuals measure discrete equation defects; pml_fraction measures energy in absorbing layers. Default: ``0.0``.
   * - ``metadata``
     - Optional
     - ``Mapping[str, Any]``
     - Structured metadata/diagnostic container associated with the mesh or result. solve_info/metadata retain applied adaptive controls and stopping history. Default: ``fresh default container``.

Returns: ``PeriodicMode``.

``FEM_Periodic_Solver.Mode.attenuation_constant``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's attenuation constant value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.Mode.attenuation_constant

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

``FEM_Periodic_Solver.Mode.beta``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's beta value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.Mode.beta

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

Returns: ``complex``.

``FEM_Periodic_Solver.Mode.bloch_multiplier``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's bloch multiplier value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.Mode.bloch_multiplier

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

Returns: ``complex``.

``FEM_Periodic_Solver.Mode.component``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``component`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.Mode.component(name: 'str') -> 'ComplexArray'

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

Returns: ``ComplexArray``.

``FEM_Periodic_Solver.Mode.folded_beta``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's folded beta value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.Mode.folded_beta

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

Returns: ``complex``.

``FEM_Periodic_Solver.Mode.folded_neff``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's folded neff value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.Mode.folded_neff

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

Returns: ``complex``.

``FEM_Periodic_Solver.Mode.gamma``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's gamma value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.Mode.gamma

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

Returns: ``complex``.

FEM_Periodic_Solver.ModeSet
---------------------------

``FEM_Periodic_Solver.ModeSet``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Construct a ``PeriodicModeSet`` record. Its public data fields use the same names and types as the constructor inputs below. Solvers normally construct mesh/system/result records for you.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.ModeSet(modes: 'Sequence[PeriodicMode]', *, frequency: 'float', period: 'float', dimension: 'int', metadata: 'Mapping[str, Any] | None' = None) -> 'None'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``modes``
     - Required
     - ``Sequence[PeriodicMode]``
     - Mode or mode collection. Integer selectors use the owning API indexing convention; supplied mode sets must match frequency, ky, and transverse geometry.
   * - ``frequency``
     - Required
     - ``float``
     - Ordinary frequency in Hz; must be finite and positive. Keyword-only.
   * - ``period``
     - Required
     - ``float``
     - Longitudinal periodic-cell length in metres; finite and positive. Keyword-only.
   * - ``dimension``
     - Required
     - ``int``
     - Spatial dimension of the mesh/geometry. Electrostatics supports 1 or 2; periodic vector fields support 3. Keyword-only.
   * - ``metadata``
     - Optional
     - ``Mapping[str, Any] | None``
     - Structured metadata/diagnostic container associated with the mesh or result. solve_info/metadata retain applied adaptive controls and stopping history. Keyword-only. Default: ``None``.

Returns: ``PeriodicModeSet``.

``FEM_Periodic_Solver.ModeSet.__getitem__``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Select an item using Python square-bracket indexing; integer indices are zero based.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.ModeSet.__getitem__(index: 'int | slice') -> 'PeriodicMode | tuple[PeriodicMode, ...]'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``index``
     - Required
     - ``int | slice``
     - Mode, case, array, or mapping selector. Python indexing is zero based; explicit mode(number) and standalone visualization use their documented one-based numbers.

Returns: ``PeriodicMode | tuple[PeriodicMode, ...]``.

``FEM_Periodic_Solver.ModeSet.__iter__``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Iterate over stored items in their existing order.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.ModeSet.__iter__() -> 'Iterator[PeriodicMode]'

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

Returns: ``Iterator[PeriodicMode]``.

``FEM_Periodic_Solver.ModeSet.__len__``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Return the number of stored items through Python len().

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.ModeSet.__len__() -> 'int'

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

Returns: ``int``.

``FEM_Periodic_Solver.ModeSet.beta``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's beta value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.ModeSet.beta

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

Returns: ``ComplexArray``.

``FEM_Periodic_Solver.ModeSet.bloch_multiplier``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's bloch multiplier value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.ModeSet.bloch_multiplier

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

Returns: ``ComplexArray``.

``FEM_Periodic_Solver.ModeSet.by_polarization``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``by_polarization`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.ModeSet.by_polarization(polarization: 'str') -> 'tuple[PeriodicMode, ...]'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``polarization``
     - Required
     - ``str``
     - Mode-family label: TE/TM/hybrid or propagating/evanescent as appropriate to the owning result or solver.

Returns: ``tuple[PeriodicMode, ...]``.

``FEM_Periodic_Solver.ModeSet.count``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

S.count(value) -> integer -- return number of occurrences of value

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.ModeSet.count(value)

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``value``
     - Required
     - ``object``
     - Scalar or array to validate, transform, interpolate, or store. It must satisfy the owning operation and the expected type in this table.

Returns: ``Python object described by this operation``.

``FEM_Periodic_Solver.ModeSet.directions``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's directions value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.ModeSet.directions

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

Returns: ``tuple[str, ...]``.

``FEM_Periodic_Solver.ModeSet.folded_beta``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's folded beta value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.ModeSet.folded_beta

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

Returns: ``ComplexArray``.

``FEM_Periodic_Solver.ModeSet.folded_neff``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's folded neff value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.ModeSet.folded_neff

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

Returns: ``ComplexArray``.

``FEM_Periodic_Solver.ModeSet.gamma``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's gamma value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.ModeSet.gamma

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

Returns: ``ComplexArray``.

``FEM_Periodic_Solver.ModeSet.index``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

S.index(value, [start, [stop]]) -> integer -- return first index of value. Raises ValueError if the value is not present.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.ModeSet.index(value, start=0, stop=None)

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``value``
     - Required
     - ``object``
     - Scalar or array to validate, transform, interpolate, or store. It must satisfy the owning operation and the expected type in this table.
   * - ``start``
     - Optional
     - ``int``
     - HDF5 case selector or mode/block slice bounds, using zero-based indexing and a stop-exclusive interval. Default: ``0``.
   * - ``stop``
     - Optional
     - ``object``
     - HDF5 case selector or mode/block slice bounds, using zero-based indexing and a stop-exclusive interval. Default: ``None``.

Returns: ``Python object described by this operation``.

Supporting start and stop arguments is optional, but recommended.

``FEM_Periodic_Solver.ModeSet.mode``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``mode`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.ModeSet.mode(number: 'int') -> 'PeriodicMode'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``number``
     - Required
     - ``int``
     - Mode, case, array, or mapping selector. Python indexing is zero based; explicit mode(number) and standalone visualization use their documented one-based numbers.

Returns: ``PeriodicMode``.

``FEM_Periodic_Solver.ModeSet.neff``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's neff value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.ModeSet.neff

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

Returns: ``ComplexArray``.

``FEM_Periodic_Solver.ModeSet.save_h5``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Persist this result through the package's versioned HDF5 writer.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.ModeSet.save_h5(path: 'str | Path') -> 'Path'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``path``
     - Required
     - ``str | Path``
     - Filesystem destination/source for HDF5 persistence or viewer launch. A directory is accepted only by viewer/directory-inspection APIs; None follows the method-specific default.

Returns: ``Path``.

FEM_Periodic_Solver.NotDiscretizedError
---------------------------------------

``FEM_Periodic_Solver.NotDiscretizedError``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An assembled operation was requested before ``discretize``.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.NotDiscretizedError(*args: 'object')

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

FEM_Periodic_Solver.PersistenceError
------------------------------------

``FEM_Periodic_Solver.PersistenceError``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A periodic-mode HDF5 archive is invalid or cannot be written.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PersistenceError(*args: 'object')

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

Returns: ``PersistenceError``.

FEM_Periodic_Solver.PMLSpec
---------------------------

``FEM_Periodic_Solver.PMLSpec``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Construct a ``PMLSpec`` record. Its public data fields use the same names and types as the constructor inputs below. Solvers normally construct mesh/system/result records for you.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PMLSpec(thickness: 'float', order: 'int' = 3, sigma_max: 'float' = 5.0, direction: 'str' = 'x') -> None

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``thickness``
     - Required
     - ``float``
     - Positive physical thickness in metres. PML thickness must fit within the selected domain side(s).
   * - ``order``
     - Optional
     - ``int``
     - Polynomial grading exponent for PML stretching; must satisfy the positive-integer constraints of the PML API. Default: ``3``.
   * - ``sigma_max``
     - Optional
     - ``float``
     - Maximum PML conductivity/stretching strength used by the package PML formulation. Default: ``5.0``.
   * - ``direction``
     - Optional
     - ``str``
     - Propagation filter forward/backward/all for mode solves; for PML placement, the selected transverse side(s), such as x-, x+, or x. Default: ``'x'``.

Returns: ``PMLSpec``.

``FEM_Periodic_Solver.PMLSpec.stretch``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``stretch`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PMLSpec.stretch(depth: 'ArrayLike') -> 'NDArray[np.complex128]'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``depth``
     - Required
     - ``ArrayLike``
     - Positive physical thickness in metres. PML thickness must fit within the selected domain side(s).

Returns: ``NDArray[np.complex128]``.

FEM_Periodic_Solver.PeriodicFEMSystem2D
---------------------------------------

``FEM_Periodic_Solver.PeriodicFEMSystem2D``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One reduced scalar QEP and the data needed for field reconstruction.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicFEMSystem2D(polarization: 'Polarization', basis: 'Basis', mesh_data: 'FEMPeriodicMesh2D', prolongation: 'PeriodicProlongation', A0: 'csr_matrix', A1: 'csr_matrix', A2: 'csr_matrix', frequency: 'float', k0: 'float', material_at: 'MaterialEvaluator', quadrature_order: 'int') -> None

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``polarization``
     - Required
     - ``Polarization``
     - Mode-family label: TE/TM/hybrid or propagating/evanescent as appropriate to the owning result or solver.
   * - ``basis``
     - Required
     - ``Basis``
     - scikit-fem Basis with the mesh, compatible element space, quadrature, and degree-of-freedom ordering used by the coefficients.
   * - ``mesh_data``
     - Required
     - ``FEMPeriodicMesh2D``
     - Conforming FEM mesh object. Physical meshes carry SI coordinates; computational meshes may be scaled by the reference length for assembly.
   * - ``prolongation``
     - Required
     - ``PeriodicProlongation``
     - Sparse periodic constraint operator mapping independent coefficients to the full edge or scalar nodal space.
   * - ``A0``
     - Required
     - ``csr_matrix``
     - Sparse/dense operator matrix. QEP coefficients represent A0 + neff*A1 + neff**2*A2 in the system coefficient ordering.
   * - ``A1``
     - Required
     - ``csr_matrix``
     - Sparse/dense operator matrix. QEP coefficients represent A0 + neff*A1 + neff**2*A2 in the system coefficient ordering.
   * - ``A2``
     - Required
     - ``csr_matrix``
     - Sparse/dense operator matrix. QEP coefficients represent A0 + neff*A1 + neff**2*A2 in the system coefficient ordering.
   * - ``frequency``
     - Required
     - ``float``
     - Ordinary frequency in Hz; must be finite and positive.
   * - ``k0``
     - Required
     - ``float``
     - Vacuum wavenumber in rad/m, used for coordinate scaling and material/PML evaluation.
   * - ``material_at``
     - Required
     - ``MaterialEvaluator``
     - Vectorized material/field callback evaluated at the coordinates supplied by the calling API; return the scalar, diagonal array, or field shape specified by that operation.
   * - ``quadrature_order``
     - Required
     - ``int``
     - Finite-element integration order. Higher-order mixed elements require at least fourth-order quadrature; PML and material variation can require more.

Returns: ``PeriodicFEMSystem2D``.

``FEM_Periodic_Solver.PeriodicFEMSystem2D.expand``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``expand`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicFEMSystem2D.expand(vector: 'ArrayLike') -> 'ComplexArray'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``vector``
     - Required
     - ``ArrayLike``
     - Complex finite-element coefficient vector or coefficient values, ordered exactly as the associated basis/system. Full vectors include constrained/periodic copies.

Returns: ``ComplexArray``.

``FEM_Periodic_Solver.PeriodicFEMSystem2D.full_size``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's full size value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicFEMSystem2D.full_size

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

Returns: ``int``.

``FEM_Periodic_Solver.PeriodicFEMSystem2D.ndofs``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's ndofs value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicFEMSystem2D.ndofs

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

Returns: ``int``.

``FEM_Periodic_Solver.PeriodicFEMSystem2D.polynomial``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``polynomial`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicFEMSystem2D.polynomial(neff: 'complex') -> 'csr_matrix'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``neff``
     - Required
     - ``complex``
     - Dimensionless effective index beta/k0. A guess selects roots near that complex value; None uses the solver estimate.

Returns: ``csr_matrix``.

``FEM_Periodic_Solver.PeriodicFEMSystem2D.relative_hermiticity_errors``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``relative_hermiticity_errors`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicFEMSystem2D.relative_hermiticity_errors() -> 'tuple[float, float, float]'

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

Returns: ``tuple[float, float, float]``.

``FEM_Periodic_Solver.PeriodicFEMSystem2D.relative_residual``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``relative_residual`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicFEMSystem2D.relative_residual(vector: 'ArrayLike', neff: 'complex') -> 'float'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``vector``
     - Required
     - ``ArrayLike``
     - Complex finite-element coefficient vector or coefficient values, ordered exactly as the associated basis/system. Full vectors include constrained/periodic copies.
   * - ``neff``
     - Required
     - ``complex``
     - Dimensionless effective index beta/k0. A guess selects roots near that complex value; None uses the solver estimate.

Returns: ``float``.

FEM_Periodic_Solver.PeriodicFEMSystem3D
---------------------------------------

``FEM_Periodic_Solver.PeriodicFEMSystem3D``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Construct a ``PeriodicFEMSystem3D`` record. Its public data fields use the same names and types as the constructor inputs below. Solvers normally construct mesh/system/result records for you.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicFEMSystem3D(basis: 'Basis', scalar_basis: 'Basis', physical_mesh: 'MeshTet', computational_mesh: 'MeshTet', mesh_data: 'PeriodicMesh3D', A0: 'csr_matrix', A1: 'csr_matrix', A2: 'csr_matrix', prolongation: 'PeriodicProlongation', scalar_prolongation: 'PeriodicProlongation', gauss0: 'csr_matrix', gauss1: 'csr_matrix', frequency: 'float', k0: 'float', material_at: 'MaterialEvaluator3D', quadrature_order: 'int') -> None

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``basis``
     - Required
     - ``Basis``
     - scikit-fem Basis with the mesh, compatible element space, quadrature, and degree-of-freedom ordering used by the coefficients.
   * - ``scalar_basis``
     - Required
     - ``Basis``
     - scikit-fem Basis with the mesh, compatible element space, quadrature, and degree-of-freedom ordering used by the coefficients.
   * - ``physical_mesh``
     - Required
     - ``MeshTet``
     - Conforming FEM mesh object. Physical meshes carry SI coordinates; computational meshes may be scaled by the reference length for assembly.
   * - ``computational_mesh``
     - Required
     - ``MeshTet``
     - Conforming FEM mesh object. Physical meshes carry SI coordinates; computational meshes may be scaled by the reference length for assembly.
   * - ``mesh_data``
     - Required
     - ``PeriodicMesh3D``
     - Conforming FEM mesh object. Physical meshes carry SI coordinates; computational meshes may be scaled by the reference length for assembly.
   * - ``A0``
     - Required
     - ``csr_matrix``
     - Sparse/dense operator matrix. QEP coefficients represent A0 + neff*A1 + neff**2*A2 in the system coefficient ordering.
   * - ``A1``
     - Required
     - ``csr_matrix``
     - Sparse/dense operator matrix. QEP coefficients represent A0 + neff*A1 + neff**2*A2 in the system coefficient ordering.
   * - ``A2``
     - Required
     - ``csr_matrix``
     - Sparse/dense operator matrix. QEP coefficients represent A0 + neff*A1 + neff**2*A2 in the system coefficient ordering.
   * - ``prolongation``
     - Required
     - ``PeriodicProlongation``
     - Sparse periodic constraint operator mapping independent coefficients to the full edge or scalar nodal space.
   * - ``scalar_prolongation``
     - Required
     - ``PeriodicProlongation``
     - Sparse periodic constraint operator mapping independent coefficients to the full edge or scalar nodal space.
   * - ``gauss0``
     - Required
     - ``csr_matrix``
     - Discrete weak divergence/Gauss operator used to validate modal charge consistency in the associated scalar test space.
   * - ``gauss1``
     - Required
     - ``csr_matrix``
     - Discrete weak divergence/Gauss operator used to validate modal charge consistency in the associated scalar test space.
   * - ``frequency``
     - Required
     - ``float``
     - Ordinary frequency in Hz; must be finite and positive.
   * - ``k0``
     - Required
     - ``float``
     - Vacuum wavenumber in rad/m, used for coordinate scaling and material/PML evaluation.
   * - ``material_at``
     - Required
     - ``MaterialEvaluator3D``
     - Vectorized material/field callback evaluated at the coordinates supplied by the calling API; return the scalar, diagonal array, or field shape specified by that operation.
   * - ``quadrature_order``
     - Required
     - ``int``
     - Finite-element integration order. Higher-order mixed elements require at least fourth-order quadrature; PML and material variation can require more.

Returns: ``PeriodicFEMSystem3D``.

``FEM_Periodic_Solver.PeriodicFEMSystem3D.divergence_residual``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Return the dimensionless weak Gauss-defect energy.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicFEMSystem3D.divergence_residual(vector: 'ArrayLike', neff: 'complex') -> 'float'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``vector``
     - Required
     - ``ArrayLike``
     - Complex finite-element coefficient vector or coefficient values, ordered exactly as the associated basis/system. Full vectors include constrained/periodic copies.
   * - ``neff``
     - Required
     - ``complex``
     - Dimensionless effective index beta/k0. A guess selects roots near that complex value; None uses the solver estimate.

Returns: ``float``.

Both numerator and operator scale are squared 2-norms. This energy convention is insensitive to eigenvector scaling and is more useful for mesh-convergence filtering than reporting the amplitude ratio. Explicitly, the returned value is ``(||(G0 + neff G1)x|| / ((||G0|| + |neff| ||G1||)||x||))**2``.

``FEM_Periodic_Solver.PeriodicFEMSystem3D.expand``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``expand`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicFEMSystem3D.expand(vector: 'ArrayLike') -> 'ComplexArray'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``vector``
     - Required
     - ``ArrayLike``
     - Complex finite-element coefficient vector or coefficient values, ordered exactly as the associated basis/system. Full vectors include constrained/periodic copies.

Returns: ``ComplexArray``.

``FEM_Periodic_Solver.PeriodicFEMSystem3D.full_size``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's full size value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicFEMSystem3D.full_size

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

Returns: ``int``.

``FEM_Periodic_Solver.PeriodicFEMSystem3D.ndofs``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's ndofs value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicFEMSystem3D.ndofs

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

Returns: ``int``.

``FEM_Periodic_Solver.PeriodicFEMSystem3D.polynomial``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``polynomial`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicFEMSystem3D.polynomial(neff: 'complex') -> 'csr_matrix'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``neff``
     - Required
     - ``complex``
     - Dimensionless effective index beta/k0. A guess selects roots near that complex value; None uses the solver estimate.

Returns: ``csr_matrix``.

``FEM_Periodic_Solver.PeriodicFEMSystem3D.relative_hermiticity_errors``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``relative_hermiticity_errors`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicFEMSystem3D.relative_hermiticity_errors() -> 'tuple[float, float, float]'

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

Returns: ``tuple[float, float, float]``.

``FEM_Periodic_Solver.PeriodicFEMSystem3D.relative_residual``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``relative_residual`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicFEMSystem3D.relative_residual(vector: 'ArrayLike', neff: 'complex') -> 'float'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``vector``
     - Required
     - ``ArrayLike``
     - Complex finite-element coefficient vector or coefficient values, ordered exactly as the associated basis/system. Full vectors include constrained/periodic copies.
   * - ``neff``
     - Required
     - ``complex``
     - Dimensionless effective index beta/k0. A guess selects roots near that complex value; None uses the solver estimate.

Returns: ``float``.

FEM_Periodic_Solver.PeriodicH5Archive
-------------------------------------

``FEM_Periodic_Solver.PeriodicH5Archive``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Lazy reader whose constructor touches only root metadata and ``/index``.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicH5Archive(path: 'str | os.PathLike[str]') -> 'None'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``path``
     - Required
     - ``str | os.PathLike[str]``
     - Filesystem destination/source for HDF5 persistence or viewer launch. A directory is accepted only by viewer/directory-inspection APIs; None follows the method-specific default.

Returns: ``PeriodicH5Archive``.

``FEM_Periodic_Solver.PeriodicH5Archive.__enter__``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``__enter__`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicH5Archive.__enter__() -> 'PeriodicH5Archive'

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

Returns: ``PeriodicH5Archive``.

``FEM_Periodic_Solver.PeriodicH5Archive.__exit__``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``__exit__`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicH5Archive.__exit__(*args: 'Any') -> 'None'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``*args``
     - Optional
     - ``Any``
     - Positional exception payload or extra positional values explicitly accepted by this callable. Numerical solve wrappers list their forwarded parameters individually instead.

Returns: ``None``.

``FEM_Periodic_Solver.PeriodicH5Archive.close``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Release cached arrays; no HDF5 handle is kept open between operations.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicH5Archive.close() -> 'None'

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

Returns: ``None``.

``FEM_Periodic_Solver.PeriodicH5Archive.load_case``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Load selected zero-based modes from one case using mode hyperslabs.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicH5Archive.load_case(case: 'int' = 0, *, modes: 'int | slice | Iterable[int] | None' = None) -> 'PeriodicModeSet'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``case``
     - Optional
     - ``int``
     - HDF5 case selector or mode/block slice bounds, using zero-based indexing and a stop-exclusive interval. Default: ``0``.
   * - ``modes``
     - Optional
     - ``int | slice | Iterable[int] | None``
     - Mode or mode collection. Integer selectors use the owning API indexing convention; supplied mode sets must match frequency, ky, and transverse geometry. Keyword-only. Default: ``None``.

Returns: ``PeriodicModeSet``.

``FEM_Periodic_Solver.PeriodicH5Archive.mode_count``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's mode count value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicH5Archive.mode_count

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

Returns: ``int``.

FEM_Periodic_Solver.PeriodicModeSolver2D
----------------------------------------

``FEM_Periodic_Solver.PeriodicModeSolver2D``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Solve complex Floquet propagation constants in one ``x-z`` unit cell.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicModeSolver2D(frequency: 'float', x_range: 'float | Sequence[float]', z_range: 'float | Sequence[float]', num_modes: 'int' = 4, neff_guess: 'complex | None' = None, *, polarization: 'str' = 'both', background_epsilon: 'MaterialInput' = 1.0, background_mu: 'MaterialInput' = 1.0, boundary: 'str' = 'pec', eigensolver: 'str' = 'auto', arnoldi_backend: 'str' = 'auto') -> 'None'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``frequency``
     - Required
     - ``float``
     - Ordinary frequency in Hz; must be finite and positive.
   * - ``x_range``
     - Required
     - ``float | Sequence[float]``
     - Physical axis bounds in metres, with upper > lower. Range-based solver constructors also accept a positive extent for (0, extent).
   * - ``z_range``
     - Required
     - ``float | Sequence[float]``
     - Physical axis bounds in metres, with upper > lower. Range-based solver constructors also accept a positive extent for (0, extent).
   * - ``num_modes``
     - Optional
     - ``int``
     - Number of modes or candidate eigenpairs requested; a positive integer. Candidate pools may include roots later rejected by validation. Default: ``4``.
   * - ``neff_guess``
     - Optional
     - ``complex | None``
     - Dimensionless effective index beta/k0. A guess selects roots near that complex value; None uses the solver estimate. Default: ``None``.
   * - ``polarization``
     - Optional
     - ``str``
     - Mode-family label: TE/TM/hybrid or propagating/evanescent as appropriate to the owning result or solver. Keyword-only. Default: ``'both'``.
   * - ``background_epsilon``
     - Optional
     - ``MaterialInput``
     - Relative permittivity. Scalar or Cartesian diagonal material inputs are supported where the signature permits; callbacks return scalar/broadcast-compatible values at physical coordinates. Keyword-only. Default: ``1.0``.
   * - ``background_mu``
     - Optional
     - ``MaterialInput``
     - Relative permeability, scalar or Cartesian diagonal where supported. WaveFEM scattering requires equal actual and background permeability. Keyword-only. Default: ``1.0``.
   * - ``boundary``
     - Optional
     - ``str``
     - Outer transverse boundary condition. Modal solvers support PEC/PMC as documented; integrated WaveFEM supports PEC or transverse PML terminated by PEC. Keyword-only. Default: ``'pec'``.
   * - ``eigensolver``
     - Optional
     - ``str``
     - Eigensolver backend selector. Accepted choices are given by the signature Literal or the method behavior notes; auto selects by system size. Keyword-only. Default: ``'auto'``.
   * - ``arnoldi_backend``
     - Optional
     - ``str``
     - Eigensolver backend selector. Accepted choices are given by the signature Literal or the method behavior notes; auto selects by system size. Keyword-only. Default: ``'auto'``.

Returns: ``PeriodicModeSolver2D``.

``FEM_Periodic_Solver.PeriodicModeSolver2D.add_UPML``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add UPML; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicModeSolver2D.add_UPML(pml_width: 'float', n: 'int' = 3, sigma_max: 'float' = 5.0, direction: 'str' = 'x') -> 'PMLSpec'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``pml_width``
     - Required
     - ``float``
     - Positive physical thickness in metres. PML thickness must fit within the selected domain side(s).
   * - ``n``
     - Optional
     - ``int``
     - Polynomial grading exponent for PML stretching; must satisfy the positive-integer constraints of the PML API. Default: ``3``.
   * - ``sigma_max``
     - Optional
     - ``float``
     - Maximum PML conductivity/stretching strength used by the package PML formulation. Default: ``5.0``.
   * - ``direction``
     - Optional
     - ``str``
     - Propagation filter forward/backward/all for mode solves; for PML placement, the selected transverse side(s), such as x-, x+, or x. Default: ``'x'``.

Returns: ``PMLSpec``.

``FEM_Periodic_Solver.PeriodicModeSolver2D.add_circle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add circle; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicModeSolver2D.add_circle(epsilon: 'MaterialInput', mu: 'MaterialInput', center: 'Sequence[float]', radius: 'float', inner_radius: 'float | None' = None, *, name: 'str | None' = None) -> 'Region'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``epsilon``
     - Required
     - ``MaterialInput``
     - Relative permittivity. Scalar or Cartesian diagonal material inputs are supported where the signature permits; callbacks return scalar/broadcast-compatible values at physical coordinates.
   * - ``mu``
     - Required
     - ``MaterialInput``
     - Relative permeability, scalar or Cartesian diagonal where supported. WaveFEM scattering requires equal actual and background permeability.
   * - ``center``
     - Required
     - ``Sequence[float]``
     - Centre coordinates in metres, ordered as the package computational axes: x-y, x-z, or x-y-z.
   * - ``radius``
     - Required
     - ``float``
     - Positive radius in metres; inner_radius describes the hollow inner boundary where supported.
   * - ``inner_radius``
     - Optional
     - ``float | None``
     - Positive radius in metres; inner_radius describes the hollow inner boundary where supported. Default: ``None``.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Returns: ``Region``.

``FEM_Periodic_Solver.PeriodicModeSolver2D.add_mesh_refinement``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add mesh refinement; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicModeSolver2D.add_mesh_refinement(shape: 'Shape2D', max_element_size: 'float', *, name: 'str | None' = None) -> 'MeshRefinement'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``shape``
     - Required
     - ``Shape2D``
     - Geometry primitive or region selector identifying the physical support. Use the class/union in the signature; electrostatics also accepts named exterior boundaries.
   * - ``max_element_size``
     - Required
     - ``float``
     - Physical element-edge length in metres. A maximum target is an upper sizing request; material, boundary, and wavelength constraints may produce smaller cells.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Returns: ``MeshRefinement``.

``FEM_Periodic_Solver.PeriodicModeSolver2D.add_pec``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add pec; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicModeSolver2D.add_pec(x_range: 'Sequence[float] | None' = None, z_range: 'Sequence[float] | None' = None, components: 'Sequence[str] | str | None' = None, *, name: 'str | None' = None) -> 'BoundaryRegion | None'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``x_range``
     - Optional
     - ``Sequence[float] | None``
     - Physical axis bounds in metres, with upper > lower. Range-based solver constructors also accept a positive extent for (0, extent). Default: ``None``.
   * - ``z_range``
     - Optional
     - ``Sequence[float] | None``
     - Physical axis bounds in metres, with upper > lower. Range-based solver constructors also accept a positive extent for (0, extent). Default: ``None``.
   * - ``components``
     - Optional
     - ``Sequence[str] | str | None``
     - Field component selection, using Cartesian electric/magnetic names such as Ex or Hy. The visualization API also supports its documented aggregate quantities. Default: ``None``.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Returns: ``BoundaryRegion | None``.

``FEM_Periodic_Solver.PeriodicModeSolver2D.add_pmc``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add pmc; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicModeSolver2D.add_pmc(x_range: 'Sequence[float] | None' = None, z_range: 'Sequence[float] | None' = None, components: 'Sequence[str] | str | None' = None, *, name: 'str | None' = None) -> 'BoundaryRegion | None'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``x_range``
     - Optional
     - ``Sequence[float] | None``
     - Physical axis bounds in metres, with upper > lower. Range-based solver constructors also accept a positive extent for (0, extent). Default: ``None``.
   * - ``z_range``
     - Optional
     - ``Sequence[float] | None``
     - Physical axis bounds in metres, with upper > lower. Range-based solver constructors also accept a positive extent for (0, extent). Default: ``None``.
   * - ``components``
     - Optional
     - ``Sequence[str] | str | None``
     - Field component selection, using Cartesian electric/magnetic names such as Ex or Hy. The visualization API also supports its documented aggregate quantities. Default: ``None``.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Returns: ``BoundaryRegion | None``.

``FEM_Periodic_Solver.PeriodicModeSolver2D.add_pml``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add pml; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicModeSolver2D.add_pml(pml_width: 'float', order: 'int' = 3, sigma_max: 'float' = 5.0, direction: 'str' = 'x') -> 'PMLSpec'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``pml_width``
     - Required
     - ``float``
     - Positive physical thickness in metres. PML thickness must fit within the selected domain side(s).
   * - ``order``
     - Optional
     - ``int``
     - Polynomial grading exponent for PML stretching; must satisfy the positive-integer constraints of the PML API. Default: ``3``.
   * - ``sigma_max``
     - Optional
     - ``float``
     - Maximum PML conductivity/stretching strength used by the package PML formulation. Default: ``5.0``.
   * - ``direction``
     - Optional
     - ``str``
     - Propagation filter forward/backward/all for mode solves; for PML placement, the selected transverse side(s), such as x-, x+, or x. Default: ``'x'``.

Returns: ``PMLSpec``.

``FEM_Periodic_Solver.PeriodicModeSolver2D.add_polygon``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add polygon; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicModeSolver2D.add_polygon(epsilon: 'MaterialInput', mu: 'MaterialInput', points: 'Sequence[Sequence[float]]', *, name: 'str | None' = None) -> 'Region'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``epsilon``
     - Required
     - ``MaterialInput``
     - Relative permittivity. Scalar or Cartesian diagonal material inputs are supported where the signature permits; callbacks return scalar/broadcast-compatible values at physical coordinates.
   * - ``mu``
     - Required
     - ``MaterialInput``
     - Relative permeability, scalar or Cartesian diagonal where supported. WaveFEM scattering requires equal actual and background permeability.
   * - ``points``
     - Required
     - ``Sequence[Sequence[float]]``
     - Coordinates used by the object or evaluation operation. Mesh geometry uses physical metres; low-level FE operators use their basis coordinate scale. See the array-shape conventions.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Returns: ``Region``.

``FEM_Periodic_Solver.PeriodicModeSolver2D.add_rectangle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add rectangle; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicModeSolver2D.add_rectangle(epsilon: 'MaterialInput', mu: 'MaterialInput', x_range: 'Sequence[float]', z_range: 'Sequence[float]', *, name: 'str | None' = None) -> 'Region'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``epsilon``
     - Required
     - ``MaterialInput``
     - Relative permittivity. Scalar or Cartesian diagonal material inputs are supported where the signature permits; callbacks return scalar/broadcast-compatible values at physical coordinates.
   * - ``mu``
     - Required
     - ``MaterialInput``
     - Relative permeability, scalar or Cartesian diagonal where supported. WaveFEM scattering requires equal actual and background permeability.
   * - ``x_range``
     - Required
     - ``Sequence[float]``
     - Physical axis bounds in metres, with upper > lower. Range-based solver constructors also accept a positive extent for (0, extent).
   * - ``z_range``
     - Required
     - ``Sequence[float]``
     - Physical axis bounds in metres, with upper > lower. Range-based solver constructors also accept a positive extent for (0, extent).
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Returns: ``Region``.

``FEM_Periodic_Solver.PeriodicModeSolver2D.add_triangle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add triangle; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicModeSolver2D.add_triangle(epsilon: 'MaterialInput', mu: 'MaterialInput', p1: 'Sequence[float]', p2: 'Sequence[float]', p3: 'Sequence[float]', *, name: 'str | None' = None) -> 'Region'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``epsilon``
     - Required
     - ``MaterialInput``
     - Relative permittivity. Scalar or Cartesian diagonal material inputs are supported where the signature permits; callbacks return scalar/broadcast-compatible values at physical coordinates.
   * - ``mu``
     - Required
     - ``MaterialInput``
     - Relative permeability, scalar or Cartesian diagonal where supported. WaveFEM scattering requires equal actual and background permeability.
   * - ``p1``
     - Required
     - ``Sequence[float]``
     - Triangle vertex coordinates in metres, in the computational-plane axis order.
   * - ``p2``
     - Required
     - ``Sequence[float]``
     - Triangle vertex coordinates in metres, in the computational-plane axis order.
   * - ``p3``
     - Required
     - ``Sequence[float]``
     - Triangle vertex coordinates in metres, in the computational-plane axis order.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Returns: ``Region``.

``FEM_Periodic_Solver.PeriodicModeSolver2D.discretize``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``discretize`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicModeSolver2D.discretize(*, max_element_size: 'float | None' = None, resolution: 'tuple[int, int] | None' = None, wavelength_elements: 'int' = 4, element_order: 'int' = 1, quadrature_order: 'int' = 4) -> 'FEMPeriodicMesh2D'

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
   * - ``resolution``
     - Optional
     - ``tuple[int, int] | None``
     - Initial mesh-resolution request. 1D uses an interval count; 2D accepts axis counts. Physical maximum size and wavelength constraints may increase the generated count. Keyword-only. Default: ``None``.
   * - ``wavelength_elements``
     - Optional
     - ``int``
     - Minimum requested elements per shortest local material wavelength; integer at least four. Public adaptive workflows default to four. Keyword-only. Default: ``4``.
   * - ``element_order``
     - Optional
     - ``int``
     - Finite-element polynomial-order selection. Standalone 2D modes and WaveFEM scattering accept 1 (N1/P1) or 2 (N2/P2); other backends retain their fixed compatible spaces. Keyword-only. Default: ``1``.
   * - ``quadrature_order``
     - Optional
     - ``int``
     - Finite-element integration order. Higher-order mixed elements require at least fourth-order quadrature; PML and material variation can require more. Keyword-only. Default: ``4``.

Returns: ``FEMPeriodicMesh2D``.

``FEM_Periodic_Solver.PeriodicModeSolver2D.discretized``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's discretized value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicModeSolver2D.discretized

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

``FEM_Periodic_Solver.PeriodicModeSolver2D.mesh``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's mesh value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicModeSolver2D.mesh

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

Returns: ``FEMPeriodicMesh2D``.

``FEM_Periodic_Solver.PeriodicModeSolver2D.mesh_data``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's mesh data value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicModeSolver2D.mesh_data

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

Returns: ``FEMPeriodicMesh2D``.

``FEM_Periodic_Solver.PeriodicModeSolver2D.native_mesh``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's native mesh value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicModeSolver2D.native_mesh

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

Returns: ``Any``.

``FEM_Periodic_Solver.PeriodicModeSolver2D.refine``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``refine`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicModeSolver2D.refine(factor: 'float' = 2.0) -> 'FEMPeriodicMesh2D'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``factor``
     - Optional
     - ``float``
     - Mesh-density multiplier. Public refine requires a finite value greater than one; the internal refinement scale accumulates requested sizing changes. Default: ``2.0``.

Returns: ``FEMPeriodicMesh2D``.

``FEM_Periodic_Solver.PeriodicModeSolver2D.remove``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Remove ; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicModeSolver2D.remove(handle: 'Region | BoundaryRegion | MeshRefinement | PMLSpec') -> 'None'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``handle``
     - Required
     - ``Region | BoundaryRegion | MeshRefinement | PMLSpec``
     - Previously returned region/boundary/PML handle to remove, or an index/key for a container operation as indicated by the method.

Returns: ``None``.

``FEM_Periodic_Solver.PeriodicModeSolver2D.result``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's result value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicModeSolver2D.result

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

Returns: ``PeriodicModeSet``.

``FEM_Periodic_Solver.PeriodicModeSolver2D.set_outer_boundary``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set outer boundary; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicModeSolver2D.set_outer_boundary(kind: 'str') -> 'None'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``kind``
     - Required
     - ``str``
     - Object/boundary/scene kind. Use the permitted Literal values or the documented selector for the owning class.

Returns: ``None``.

``FEM_Periodic_Solver.PeriodicModeSolver2D.solution``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's solution value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicModeSolver2D.solution

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

Returns: ``PeriodicModeSet | None``.

``FEM_Periodic_Solver.PeriodicModeSolver2D.solve``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Solve and remesh until the interface residual meets the threshold or the refinement budget is exhausted. Gmsh regenerates periodic node and edge constraints on every mesh. Zero refinements means one solve.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicModeSolver2D.solve(neff_guess: 'complex | None' = None, num_modes: 'int | None' = None, *, direction: 'Direction' = 'forward', eigensolver: 'str | None' = None, arnoldi_backend: 'str | None' = None, eigensolver_tolerance: 'float' = 1e-10, residual_tolerance: 'float' = 1e-08, propagation_ratio_tolerance: 'float' = 0.001, max_pml_fraction: 'float | None' = 0.5, dense_linearization_limit: 'int' = 700, max_refinements: 'int' = 2, adaptive_tolerance: 'float' = 0.05) -> 'PeriodicModeSet'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``neff_guess``
     - Optional
     - ``complex | None``
     - Dimensionless effective index beta/k0. A guess selects roots near that complex value; None uses the solver estimate. Default: ``None``.
   * - ``num_modes``
     - Optional
     - ``int | None``
     - Number of modes or candidate eigenpairs requested; a positive integer. Candidate pools may include roots later rejected by validation. None uses the mode count stored on the solver. Default: ``None``.
   * - ``direction``
     - Optional
     - ``Direction``
     - Propagation filter forward/backward/all for mode solves; for PML placement, the selected transverse side(s), such as x-, x+, or x. Keyword-only. Default: ``'forward'``.
   * - ``eigensolver``
     - Optional
     - ``str | None``
     - Eigensolver backend selector. Accepted choices are given by the signature Literal or the method behavior notes; auto selects by system size. None retains the solver constructor's backend selection. Keyword-only. Default: ``None``.
   * - ``arnoldi_backend``
     - Optional
     - ``str | None``
     - Eigensolver backend selector. Accepted choices are given by the signature Literal or the method behavior notes; auto selects by system size. None retains the solver constructor's backend selection. Keyword-only. Default: ``None``.
   * - ``eigensolver_tolerance``
     - Optional
     - ``float``
     - Positive numerical tolerance. Linear/QEP residual tolerances validate the algebraic solve; they do not set the adaptive mesh threshold. Keyword-only. Default: ``1e-10``.
   * - ``residual_tolerance``
     - Optional
     - ``float``
     - Positive numerical tolerance. Linear/QEP residual tolerances validate the algebraic solve; they do not set the adaptive mesh threshold. Keyword-only. Default: ``1e-08``.
   * - ``propagation_ratio_tolerance``
     - Optional
     - ``float``
     - Positive relative real/imaginary propagation criterion used to classify propagating and evanescent roots. Keyword-only. Default: ``0.001``.
   * - ``max_pml_fraction``
     - Optional
     - ``float | None``
     - Maximum permitted modal energy fraction in PML; None disables this candidate filter. Keyword-only. Default: ``0.5``.
   * - ``dense_linearization_limit``
     - Optional
     - ``int``
     - Matrix-size cutoff for dense eigensolving; larger systems use a sparse backend. This is a dimension limit, not a mesh-error threshold. Keyword-only. Default: ``700``.
   * - ``max_refinements``
     - Optional
     - ``int``
     - Maximum mesh updates after the initial solve; nonnegative integer. Zero keeps the initial mesh. Solver default is 2; examples explicitly use 0. Keyword-only. Default: ``2``.
   * - ``adaptive_tolerance``
     - Optional
     - ``float``
     - Positive finite threshold for the normalized discretization estimator. Stop when residual <= threshold. Independent of algebraic tolerances; exhausting the budget is not convergence. Keyword-only. Default: ``0.05``.

Returns: ``PeriodicModeSet``.

``FEM_Periodic_Solver.PeriodicModeSolver2D.system``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's system value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicModeSolver2D.system

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

Returns: ``PeriodicFEMSystem2D``.

``FEM_Periodic_Solver.PeriodicModeSolver2D.systems``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's systems value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicModeSolver2D.systems

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

Returns: ``dict[str, PeriodicFEMSystem2D]``.

``FEM_Periodic_Solver.PeriodicModeSolver2D.visualize``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``visualize`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicModeSolver2D.visualize(mode: 'int | PeriodicMode' = 1, *, component: 'str' = 'Ey', quantity: 'str' = 'real', ax: 'Any' = None, cmap: 'str' = 'RdBu_r', show_mesh: 'bool' = False, colorbar: 'bool' = True, show: 'bool' = True, slice_axis: 'str | None' = None, slice_fraction: 'float' = 0.5, max_points: 'int' = 5000) -> 'Any'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``mode``
     - Optional
     - ``int | PeriodicMode``
     - Mode or mode collection. Integer selectors use the owning API indexing convention; supplied mode sets must match frequency, ky, and transverse geometry. Default: ``1``.
   * - ``component``
     - Optional
     - ``str``
     - Field component selection, using Cartesian electric/magnetic names such as Ex or Hy. The visualization API also supports its documented aggregate quantities. Keyword-only. Default: ``'Ey'``.
   * - ``quantity``
     - Optional
     - ``str``
     - Complex-data display selection, such as real, imag, abs, or phase; accepted values are given by the owning plotting API. Keyword-only. Default: ``'real'``.
   * - ``ax``
     - Optional
     - ``Any``
     - Existing Matplotlib axes for embedding a plot; None creates suitable axes when accepted. Keyword-only. Default: ``None``.
   * - ``cmap``
     - Optional
     - ``str``
     - Matplotlib colormap name or object used for scalar field rendering. Keyword-only. Default: ``'RdBu_r'``.
   * - ``show_mesh``
     - Optional
     - ``bool``
     - Enable display of the figure, mesh overlay, or colorbar respectively. show=False returns plotting objects without opening a window. Keyword-only. Default: ``False``.
   * - ``colorbar``
     - Optional
     - ``bool``
     - Enable display of the figure, mesh overlay, or colorbar respectively. show=False returns plotting objects without opening a window. Keyword-only. Default: ``True``.
   * - ``show``
     - Optional
     - ``bool``
     - Enable display of the figure, mesh overlay, or colorbar respectively. show=False returns plotting objects without opening a window. Keyword-only. Default: ``True``.
   * - ``slice_axis``
     - Optional
     - ``str | None``
     - 3D visualization cut: Cartesian axis and fractional location in [0, 1] along that domain extent. Keyword-only. Default: ``None``.
   * - ``slice_fraction``
     - Optional
     - ``float``
     - 3D visualization cut: Cartesian axis and fractional location in [0, 1] along that domain extent. Keyword-only. Default: ``0.5``.
   * - ``max_points``
     - Optional
     - ``int``
     - Positive sample/glyph budget for rendering; affects visualization density, not the FEM solution. Keyword-only. Default: ``5000``.

Returns: ``Any``.

``FEM_Periodic_Solver.PeriodicModeSolver2D.visualize_with_gui``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open every solved mode in the standalone native viewer.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicModeSolver2D.visualize_with_gui() -> 'Any'

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

Returns: ``Any``.

FEM_Periodic_Solver.PeriodicModeSolver3D
----------------------------------------

``FEM_Periodic_Solver.PeriodicModeSolver3D``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Solve complex fixed-frequency Bloch propagation in a tetrahedral cell.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicModeSolver3D(frequency: 'float', x_range: 'float | tuple[float, float]', y_range: 'float | tuple[float, float]', z_range: 'float | tuple[float, float]', num_modes: 'int' = 4, neff_guess: 'complex | None' = None, *, background_epsilon: 'MaterialInput' = 1.0, background_mu: 'MaterialInput' = 1.0, boundary: 'str' = 'pec', eigensolver: 'str' = 'auto', arnoldi_backend: 'str' = 'auto') -> 'None'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``frequency``
     - Required
     - ``float``
     - Ordinary frequency in Hz; must be finite and positive.
   * - ``x_range``
     - Required
     - ``float | tuple[float, float]``
     - Physical axis bounds in metres, with upper > lower. Range-based solver constructors also accept a positive extent for (0, extent).
   * - ``y_range``
     - Required
     - ``float | tuple[float, float]``
     - Physical axis bounds in metres, with upper > lower. Range-based solver constructors also accept a positive extent for (0, extent).
   * - ``z_range``
     - Required
     - ``float | tuple[float, float]``
     - Physical axis bounds in metres, with upper > lower. Range-based solver constructors also accept a positive extent for (0, extent).
   * - ``num_modes``
     - Optional
     - ``int``
     - Number of modes or candidate eigenpairs requested; a positive integer. Candidate pools may include roots later rejected by validation. Default: ``4``.
   * - ``neff_guess``
     - Optional
     - ``complex | None``
     - Dimensionless effective index beta/k0. A guess selects roots near that complex value; None uses the solver estimate. Default: ``None``.
   * - ``background_epsilon``
     - Optional
     - ``MaterialInput``
     - Relative permittivity. Scalar or Cartesian diagonal material inputs are supported where the signature permits; callbacks return scalar/broadcast-compatible values at physical coordinates. Keyword-only. Default: ``1.0``.
   * - ``background_mu``
     - Optional
     - ``MaterialInput``
     - Relative permeability, scalar or Cartesian diagonal where supported. WaveFEM scattering requires equal actual and background permeability. Keyword-only. Default: ``1.0``.
   * - ``boundary``
     - Optional
     - ``str``
     - Outer transverse boundary condition. Modal solvers support PEC/PMC as documented; integrated WaveFEM supports PEC or transverse PML terminated by PEC. Keyword-only. Default: ``'pec'``.
   * - ``eigensolver``
     - Optional
     - ``str``
     - Eigensolver backend selector. Accepted choices are given by the signature Literal or the method behavior notes; auto selects by system size. Keyword-only. Default: ``'auto'``.
   * - ``arnoldi_backend``
     - Optional
     - ``str``
     - Eigensolver backend selector. Accepted choices are given by the signature Literal or the method behavior notes; auto selects by system size. Keyword-only. Default: ``'auto'``.

Returns: ``PeriodicModeSolver3D``.

``FEM_Periodic_Solver.PeriodicModeSolver3D.add_box``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add box; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicModeSolver3D.add_box(epsilon: 'MaterialInput', mu: 'MaterialInput', x_range: 'tuple[float, float]', y_range: 'tuple[float, float]', z_range: 'tuple[float, float]', *, name: 'str | None' = None) -> 'object'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``epsilon``
     - Required
     - ``MaterialInput``
     - Relative permittivity. Scalar or Cartesian diagonal material inputs are supported where the signature permits; callbacks return scalar/broadcast-compatible values at physical coordinates.
   * - ``mu``
     - Required
     - ``MaterialInput``
     - Relative permeability, scalar or Cartesian diagonal where supported. WaveFEM scattering requires equal actual and background permeability.
   * - ``x_range``
     - Required
     - ``tuple[float, float]``
     - Physical axis bounds in metres, with upper > lower. Range-based solver constructors also accept a positive extent for (0, extent).
   * - ``y_range``
     - Required
     - ``tuple[float, float]``
     - Physical axis bounds in metres, with upper > lower. Range-based solver constructors also accept a positive extent for (0, extent).
   * - ``z_range``
     - Required
     - ``tuple[float, float]``
     - Physical axis bounds in metres, with upper > lower. Range-based solver constructors also accept a positive extent for (0, extent).
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Returns: ``Python object described by this operation``.

``FEM_Periodic_Solver.PeriodicModeSolver3D.add_cylinder``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add cylinder; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicModeSolver3D.add_cylinder(epsilon: 'MaterialInput', mu: 'MaterialInput', center: 'tuple[float, float]', radius: 'float', z_range: 'tuple[float, float]', *, name: 'str | None' = None) -> 'object'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``epsilon``
     - Required
     - ``MaterialInput``
     - Relative permittivity. Scalar or Cartesian diagonal material inputs are supported where the signature permits; callbacks return scalar/broadcast-compatible values at physical coordinates.
   * - ``mu``
     - Required
     - ``MaterialInput``
     - Relative permeability, scalar or Cartesian diagonal where supported. WaveFEM scattering requires equal actual and background permeability.
   * - ``center``
     - Required
     - ``tuple[float, float]``
     - Centre coordinates in metres, ordered as the package computational axes: x-y, x-z, or x-y-z.
   * - ``radius``
     - Required
     - ``float``
     - Positive radius in metres; inner_radius describes the hollow inner boundary where supported.
   * - ``z_range``
     - Required
     - ``tuple[float, float]``
     - Physical axis bounds in metres, with upper > lower. Range-based solver constructors also accept a positive extent for (0, extent).
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Returns: ``Python object described by this operation``.

``FEM_Periodic_Solver.PeriodicModeSolver3D.add_mesh_refinement``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add mesh refinement; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicModeSolver3D.add_mesh_refinement(shape: 'Shape3D', max_element_size: 'float', *, name: 'str | None' = None) -> 'object'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``shape``
     - Required
     - ``Shape3D``
     - Geometry primitive or region selector identifying the physical support. Use the class/union in the signature; electrostatics also accepts named exterior boundaries.
   * - ``max_element_size``
     - Required
     - ``float``
     - Physical element-edge length in metres. A maximum target is an upper sizing request; material, boundary, and wavelength constraints may produce smaller cells.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Returns: ``Python object described by this operation``.

``FEM_Periodic_Solver.PeriodicModeSolver3D.add_pec``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add pec; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicModeSolver3D.add_pec(shape: 'Shape3D', *, name: 'str | None' = None) -> 'object'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``shape``
     - Required
     - ``Shape3D``
     - Geometry primitive or region selector identifying the physical support. Use the class/union in the signature; electrostatics also accepts named exterior boundaries.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Returns: ``Python object described by this operation``.

``FEM_Periodic_Solver.PeriodicModeSolver3D.add_pmc``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add pmc; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicModeSolver3D.add_pmc(shape: 'Shape3D', *, name: 'str | None' = None) -> 'object'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``shape``
     - Required
     - ``Shape3D``
     - Geometry primitive or region selector identifying the physical support. Use the class/union in the signature; electrostatics also accepts named exterior boundaries.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Returns: ``Python object described by this operation``.

``FEM_Periodic_Solver.PeriodicModeSolver3D.add_pml``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add pml; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicModeSolver3D.add_pml(thickness: 'float', *, order: 'int' = 3, sigma_max: 'float' = 5.0, direction: 'str' = 'all') -> 'object'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``thickness``
     - Required
     - ``float``
     - Positive physical thickness in metres. PML thickness must fit within the selected domain side(s).
   * - ``order``
     - Optional
     - ``int``
     - Polynomial grading exponent for PML stretching; must satisfy the positive-integer constraints of the PML API. Keyword-only. Default: ``3``.
   * - ``sigma_max``
     - Optional
     - ``float``
     - Maximum PML conductivity/stretching strength used by the package PML formulation. Keyword-only. Default: ``5.0``.
   * - ``direction``
     - Optional
     - ``str``
     - Propagation filter forward/backward/all for mode solves; for PML placement, the selected transverse side(s), such as x-, x+, or x. Keyword-only. Default: ``'all'``.

Returns: ``Python object described by this operation``.

``FEM_Periodic_Solver.PeriodicModeSolver3D.add_sphere``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add sphere; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicModeSolver3D.add_sphere(epsilon: 'MaterialInput', mu: 'MaterialInput', center: 'tuple[float, float, float]', radius: 'float', *, name: 'str | None' = None) -> 'object'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``epsilon``
     - Required
     - ``MaterialInput``
     - Relative permittivity. Scalar or Cartesian diagonal material inputs are supported where the signature permits; callbacks return scalar/broadcast-compatible values at physical coordinates.
   * - ``mu``
     - Required
     - ``MaterialInput``
     - Relative permeability, scalar or Cartesian diagonal where supported. WaveFEM scattering requires equal actual and background permeability.
   * - ``center``
     - Required
     - ``tuple[float, float, float]``
     - Centre coordinates in metres, ordered as the package computational axes: x-y, x-z, or x-y-z.
   * - ``radius``
     - Required
     - ``float``
     - Positive radius in metres; inner_radius describes the hollow inner boundary where supported.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Returns: ``Python object described by this operation``.

``FEM_Periodic_Solver.PeriodicModeSolver3D.discretize``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``discretize`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicModeSolver3D.discretize(*, max_element_size: 'float | None' = None, wavelength_elements: 'int' = 4, material_aware: 'bool' = True, quadrature_order: 'int' = 3) -> 'PeriodicMesh3D'

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
   * - ``wavelength_elements``
     - Optional
     - ``int``
     - Minimum requested elements per shortest local material wavelength; integer at least four. Public adaptive workflows default to four. Keyword-only. Default: ``4``.
   * - ``material_aware``
     - Optional
     - ``bool``
     - Enable element-size reduction in high-index/high-permittivity material regions while retaining the global maximum-edge cap. Keyword-only. Default: ``True``.
   * - ``quadrature_order``
     - Optional
     - ``int``
     - Finite-element integration order. Higher-order mixed elements require at least fourth-order quadrature; PML and material variation can require more. Keyword-only. Default: ``3``.

Returns: ``PeriodicMesh3D``.

``FEM_Periodic_Solver.PeriodicModeSolver3D.refine``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``refine`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicModeSolver3D.refine(factor: 'float' = 2.0) -> 'PeriodicMesh3D'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``factor``
     - Optional
     - ``float``
     - Mesh-density multiplier. Public refine requires a finite value greater than one; the internal refinement scale accumulates requested sizing changes. Default: ``2.0``.

Returns: ``PeriodicMesh3D``.

``FEM_Periodic_Solver.PeriodicModeSolver3D.remove``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Remove ; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicModeSolver3D.remove(handle: 'object') -> 'None'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``handle``
     - Required
     - ``object``
     - Previously returned region/boundary/PML handle to remove, or an index/key for a container operation as indicated by the method.

Returns: ``None``.

``FEM_Periodic_Solver.PeriodicModeSolver3D.set_outer_boundary``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set outer boundary; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicModeSolver3D.set_outer_boundary(kind: 'str') -> 'None'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``kind``
     - Required
     - ``str``
     - Object/boundary/scene kind. Use the permitted Literal values or the documented selector for the owning class.

Returns: ``None``.

``FEM_Periodic_Solver.PeriodicModeSolver3D.solve``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Solve and remesh until the interface residual meets the threshold or the refinement budget is exhausted. Gmsh regenerates periodic node and edge constraints on every mesh. Zero refinements means one solve.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicModeSolver3D.solve(neff_guess: 'complex | None' = None, num_modes: 'int | None' = None, *, direction: 'Direction' = 'forward', eigensolver: 'str | None' = None, arnoldi_backend: 'str | None' = None, eigensolver_tolerance: 'float' = 1e-10, residual_tolerance: 'float' = 1e-08, divergence_tolerance: 'float' = 1e-06, propagation_ratio_tolerance: 'float' = 0.001, max_pml_fraction: 'float | None' = 0.5, dense_linearization_limit: 'int' = 700, ncv: 'int | None' = None, max_restarts: 'int' = 12, random_seed: 'int' = 0, max_refinements: 'int' = 2, adaptive_tolerance: 'float' = 0.05) -> 'PeriodicModeSet'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``neff_guess``
     - Optional
     - ``complex | None``
     - Dimensionless effective index beta/k0. A guess selects roots near that complex value; None uses the solver estimate. Default: ``None``.
   * - ``num_modes``
     - Optional
     - ``int | None``
     - Number of modes or candidate eigenpairs requested; a positive integer. Candidate pools may include roots later rejected by validation. None uses the mode count stored on the solver. Default: ``None``.
   * - ``direction``
     - Optional
     - ``Direction``
     - Propagation filter forward/backward/all for mode solves; for PML placement, the selected transverse side(s), such as x-, x+, or x. Keyword-only. Default: ``'forward'``.
   * - ``eigensolver``
     - Optional
     - ``str | None``
     - Eigensolver backend selector. Accepted choices are given by the signature Literal or the method behavior notes; auto selects by system size. None retains the solver constructor's backend selection. Keyword-only. Default: ``None``.
   * - ``arnoldi_backend``
     - Optional
     - ``str | None``
     - Eigensolver backend selector. Accepted choices are given by the signature Literal or the method behavior notes; auto selects by system size. None retains the solver constructor's backend selection. Keyword-only. Default: ``None``.
   * - ``eigensolver_tolerance``
     - Optional
     - ``float``
     - Positive numerical tolerance. Linear/QEP residual tolerances validate the algebraic solve; they do not set the adaptive mesh threshold. Keyword-only. Default: ``1e-10``.
   * - ``residual_tolerance``
     - Optional
     - ``float``
     - Positive numerical tolerance. Linear/QEP residual tolerances validate the algebraic solve; they do not set the adaptive mesh threshold. Keyword-only. Default: ``1e-08``.
   * - ``divergence_tolerance``
     - Optional
     - ``float``
     - Maximum accepted weak Gauss-law defect. Periodic 3D uses a squared normalized defect; other modal backends use their documented divergence residual. Keyword-only. Default: ``1e-06``.
   * - ``propagation_ratio_tolerance``
     - Optional
     - ``float``
     - Positive relative real/imaginary propagation criterion used to classify propagating and evanescent roots. Keyword-only. Default: ``0.001``.
   * - ``max_pml_fraction``
     - Optional
     - ``float | None``
     - Maximum permitted modal energy fraction in PML; None disables this candidate filter. Keyword-only. Default: ``0.5``.
   * - ``dense_linearization_limit``
     - Optional
     - ``int``
     - Matrix-size cutoff for dense eigensolving; larger systems use a sparse backend. This is a dimension limit, not a mesh-error threshold. Keyword-only. Default: ``700``.
   * - ``ncv``
     - Optional
     - ``int | None``
     - Arnoldi/Krylov subspace size; None lets the eigensolver choose a size consistent with candidate count and matrix dimension. Keyword-only. Default: ``None``.
   * - ``max_restarts``
     - Optional
     - ``int``
     - Iteration or Arnoldi-restart budget. None selects the backend default; the direct electrostatic solve accepts max_iter for compatibility. Keyword-only. Default: ``12``.
   * - ``random_seed``
     - Optional
     - ``int``
     - Integer seed for deterministic eigensolver starting vectors. Keyword-only. Default: ``0``.
   * - ``max_refinements``
     - Optional
     - ``int``
     - Maximum mesh updates after the initial solve; nonnegative integer. Zero keeps the initial mesh. Solver default is 2; examples explicitly use 0. Keyword-only. Default: ``2``.
   * - ``adaptive_tolerance``
     - Optional
     - ``float``
     - Positive finite threshold for the normalized discretization estimator. Stop when residual <= threshold. Independent of algebraic tolerances; exhausting the budget is not convergence. Keyword-only. Default: ``0.05``.

Returns: ``PeriodicModeSet``.

``FEM_Periodic_Solver.PeriodicModeSolver3D.visualize``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a Matplotlib figure for one solved three-dimensional mode.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicModeSolver3D.visualize(mode: 'int | PeriodicMode' = 1, *, component: 'str' = 'Ey', quantity: 'str' = 'real', ax: 'Any' = None, cmap: 'str' = 'RdBu_r', show_mesh: 'bool' = False, colorbar: 'bool' = True, show: 'bool' = True, slice_axis: 'str | None' = None, slice_fraction: 'float' = 0.5, max_points: 'int' = 5000) -> 'Any'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``mode``
     - Optional
     - ``int | PeriodicMode``
     - Mode or mode collection. Integer selectors use the owning API indexing convention; supplied mode sets must match frequency, ky, and transverse geometry. Default: ``1``.
   * - ``component``
     - Optional
     - ``str``
     - Field component selection, using Cartesian electric/magnetic names such as Ex or Hy. The visualization API also supports its documented aggregate quantities. Keyword-only. Default: ``'Ey'``.
   * - ``quantity``
     - Optional
     - ``str``
     - Complex-data display selection, such as real, imag, abs, or phase; accepted values are given by the owning plotting API. Keyword-only. Default: ``'real'``.
   * - ``ax``
     - Optional
     - ``Any``
     - Existing Matplotlib axes for embedding a plot; None creates suitable axes when accepted. Keyword-only. Default: ``None``.
   * - ``cmap``
     - Optional
     - ``str``
     - Matplotlib colormap name or object used for scalar field rendering. Keyword-only. Default: ``'RdBu_r'``.
   * - ``show_mesh``
     - Optional
     - ``bool``
     - Enable display of the figure, mesh overlay, or colorbar respectively. show=False returns plotting objects without opening a window. Keyword-only. Default: ``False``.
   * - ``colorbar``
     - Optional
     - ``bool``
     - Enable display of the figure, mesh overlay, or colorbar respectively. show=False returns plotting objects without opening a window. Keyword-only. Default: ``True``.
   * - ``show``
     - Optional
     - ``bool``
     - Enable display of the figure, mesh overlay, or colorbar respectively. show=False returns plotting objects without opening a window. Keyword-only. Default: ``True``.
   * - ``slice_axis``
     - Optional
     - ``str | None``
     - 3D visualization cut: Cartesian axis and fractional location in [0, 1] along that domain extent. Keyword-only. Default: ``None``.
   * - ``slice_fraction``
     - Optional
     - ``float``
     - 3D visualization cut: Cartesian axis and fractional location in [0, 1] along that domain extent. Keyword-only. Default: ``0.5``.
   * - ``max_points``
     - Optional
     - ``int``
     - Positive sample/glyph budget for rendering; affects visualization density, not the FEM solution. Keyword-only. Default: ``5000``.

Returns: ``Any``.

``FEM_Periodic_Solver.PeriodicModeSolver3D.visualize_with_gui``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open every solved mode in the standalone native viewer.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicModeSolver3D.visualize_with_gui() -> 'Any'

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

Returns: ``Any``.

FEM_Periodic_Solver.PeriodicProlongation
----------------------------------------

``FEM_Periodic_Solver.PeriodicProlongation``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A sparse map from independent coefficients to full mesh coefficients.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicProlongation(matrix: 'csr_matrix', representatives: 'IntArray', independent_representatives: 'IntArray', signs: 'NDArray[np.int8]') -> None

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``matrix``
     - Required
     - ``csr_matrix``
     - Sparse/dense operator matrix. QEP coefficients represent A0 + neff*A1 + neff**2*A2 in the system coefficient ordering.
   * - ``representatives``
     - Required
     - ``IntArray``
     - Integer canonical representative indices used to identify periodic equality classes and independent unknowns.
   * - ``independent_representatives``
     - Required
     - ``IntArray``
     - Integer canonical representative indices used to identify periodic equality classes and independent unknowns.
   * - ``signs``
     - Required
     - ``NDArray[np.int8]``
     - Orientation factors (+1 or -1) mapping local/periodic Nedelec edges to canonical global directions.

Returns: ``PeriodicProlongation``.

``FEM_Periodic_Solver.PeriodicProlongation.equality_error``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``equality_error`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicProlongation.equality_error(full_coefficients: 'ArrayLike') -> 'float'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``full_coefficients``
     - Required
     - ``ArrayLike``
     - Complex finite-element coefficient vector or coefficient values, ordered exactly as the associated basis/system. Full vectors include constrained/periodic copies.

Returns: ``float``.

``FEM_Periodic_Solver.PeriodicProlongation.expand``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``expand`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicProlongation.expand(coefficients: 'ArrayLike') -> 'NDArray[np.complex128]'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``coefficients``
     - Required
     - ``ArrayLike``
     - Complex finite-element coefficient vector or coefficient values, ordered exactly as the associated basis/system. Full vectors include constrained/periodic copies.

Returns: ``NDArray[np.complex128]``.

``FEM_Periodic_Solver.PeriodicProlongation.full_size``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's full size value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicProlongation.full_size

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

Returns: ``int``.

``FEM_Periodic_Solver.PeriodicProlongation.reduce_matrix``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``reduce_matrix`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicProlongation.reduce_matrix(matrix: 'spmatrix') -> 'csr_matrix'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``matrix``
     - Required
     - ``spmatrix``
     - Sparse/dense operator matrix. QEP coefficients represent A0 + neff*A1 + neff**2*A2 in the system coefficient ordering.

Returns: ``csr_matrix``.

``FEM_Periodic_Solver.PeriodicProlongation.reduced_size``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's reduced size value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicProlongation.reduced_size

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

Returns: ``int``.

FEM_Periodic_Solver.PeriodicSampledFields
-----------------------------------------

``FEM_Periodic_Solver.PeriodicSampledFields``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Complex Cartesian fields sampled at element-owned points.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicSampledFields(coordinates: 'ArrayLike', values: 'Mapping[str, ArrayLike] | ArrayLike', *, dimension: 'int', mesh_points: 'ArrayLike', mesh_cells: 'ArrayLike', sample_element_indices: 'ArrayLike', material: 'ArrayLike | None' = None, metadata: 'Mapping[str, Any] | None' = None) -> 'None'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``coordinates``
     - Required
     - ``ArrayLike``
     - Coordinates used by the object or evaluation operation. Mesh geometry uses physical metres; low-level FE operators use their basis coordinate scale. See the array-shape conventions.
   * - ``values``
     - Required
     - ``Mapping[str, ArrayLike] | ArrayLike``
     - Scalar or array to validate, transform, interpolate, or store. It must satisfy the owning operation and the expected type in this table.
   * - ``dimension``
     - Required
     - ``int``
     - Spatial dimension of the mesh/geometry. Electrostatics supports 1 or 2; periodic vector fields support 3. Keyword-only.
   * - ``mesh_points``
     - Required
     - ``ArrayLike``
     - Mesh-node coordinates. Native scikit-fem arrays are dimension by node; standalone result coordinates are commonly node by dimension. Keyword-only.
   * - ``mesh_cells``
     - Required
     - ``ArrayLike``
     - Integer simplex connectivity. scikit-fem uses vertices-per-cell by cell; standalone exported geometry commonly uses cell by vertices-per-cell. Keyword-only.
   * - ``sample_element_indices``
     - Required
     - ``ArrayLike``
     - Integer map from each field sample to its parent element in the stored connectivity. Keyword-only.
   * - ``material``
     - Optional
     - ``ArrayLike | None``
     - Material object defining relative electric and magnetic response. Exterior fills points outside placed material regions. Keyword-only. Default: ``None``.
   * - ``metadata``
     - Optional
     - ``Mapping[str, Any] | None``
     - Structured metadata/diagnostic container associated with the mesh or result. solve_info/metadata retain applied adaptive controls and stopping history. Keyword-only. Default: ``None``.

Returns: ``PeriodicSampledFields``.

``FEM_Periodic_Solver.PeriodicSampledFields.component``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``component`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicSampledFields.component(name: 'str') -> 'ComplexArray'

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

Returns: ``ComplexArray``.

``FEM_Periodic_Solver.PeriodicSampledFields.components``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's components value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicSampledFields.components

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

Returns: ``tuple[str, ...]``.

``FEM_Periodic_Solver.PeriodicSampledFields.quantity``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``quantity`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicSampledFields.quantity(component: 'str', quantity: 'str' = 'real') -> 'FloatArray'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``component``
     - Required
     - ``str``
     - Field component selection, using Cartesian electric/magnetic names such as Ex or Hy. The visualization API also supports its documented aggregate quantities.
   * - ``quantity``
     - Optional
     - ``str``
     - Complex-data display selection, such as real, imag, abs, or phase; accepted values are given by the owning plotting API. Default: ``'real'``.

Returns: ``FloatArray``.

``FEM_Periodic_Solver.PeriodicSampledFields.x``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's x value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicSampledFields.x

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

``FEM_Periodic_Solver.PeriodicSampledFields.y``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's y value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicSampledFields.y

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

Returns: ``FloatArray | None``.

``FEM_Periodic_Solver.PeriodicSampledFields.z``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's z value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.PeriodicSampledFields.z

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

FEM_Periodic_Solver.Polygon
---------------------------

``FEM_Periodic_Solver.Polygon``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define a closed polygon from ordered physical vertices. Use this immutable primitive in geometry/material placement APIs.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.Polygon(points: 'tuple[tuple[float, float], ...]') -> None

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

``FEM_Periodic_Solver.Polygon.bounds``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's bounds value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.Polygon.bounds

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

``FEM_Periodic_Solver.Polygon.contains``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``contains`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.Polygon.contains(x: 'ArrayLike', z: 'ArrayLike') -> 'NDArray[np.bool_]'

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
   * - ``z``
     - Required
     - ``ArrayLike``
     - Physical z coordinate samples in metres; arrays must broadcast with the other coordinate arguments.

Returns: ``NDArray[np.bool_]``.

FEM_Periodic_Solver.Rectangle
-----------------------------

``FEM_Periodic_Solver.Rectangle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define an axis-aligned physical rectangle. Use this immutable primitive in geometry/material placement APIs.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.Rectangle(x: 'tuple[float, float]', z: 'tuple[float, float]') -> None

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
   * - ``z``
     - Required
     - ``tuple[float, float]``
     - Physical z-axis bounds (minimum, maximum), in metres; both finite with maximum > minimum.

Returns: ``Rectangle``.

``FEM_Periodic_Solver.Rectangle.bounds``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's bounds value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.Rectangle.bounds

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

``FEM_Periodic_Solver.Rectangle.contains``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``contains`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.Rectangle.contains(x: 'ArrayLike', z: 'ArrayLike') -> 'NDArray[np.bool_]'

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
   * - ``z``
     - Required
     - ``ArrayLike``
     - Physical z coordinate samples in metres; arrays must broadcast with the other coordinate arguments.

Returns: ``NDArray[np.bool_]``.

FEM_Periodic_Solver.Region
--------------------------

``FEM_Periodic_Solver.Region``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Construct a ``Region`` record. Its public data fields use the same names and types as the constructor inputs below. Solvers normally construct mesh/system/result records for you.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.Region(id: 'int', name: 'str', shape: 'Shape2D | Shape3D', material: 'Material') -> None

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
     - ``Shape2D | Shape3D``
     - Geometry primitive or region selector identifying the physical support. Use the class/union in the signature; electrostatics also accepts named exterior boundaries.
   * - ``material``
     - Required
     - ``Material``
     - Material object defining relative electric and magnetic response. Exterior fills points outside placed material regions.

Returns: ``Region``.

FEM_Periodic_Solver.SolverError
-------------------------------

``FEM_Periodic_Solver.SolverError``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The polynomial eigenproblem did not produce the requested modes.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.SolverError(*args: 'object')

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

FEM_Periodic_Solver.Sphere
--------------------------

``FEM_Periodic_Solver.Sphere``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define a sphere with a physical centre and radius. Use this immutable primitive in geometry/material placement APIs.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.Sphere(center: 'tuple[float, float, float]', radius: 'float') -> None

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``center``
     - Required
     - ``tuple[float, float, float]``
     - Centre coordinates in metres, ordered as the package computational axes: x-y, x-z, or x-y-z.
   * - ``radius``
     - Required
     - ``float``
     - Positive radius in metres; inner_radius describes the hollow inner boundary where supported.

Returns: ``Sphere``.

``FEM_Periodic_Solver.Sphere.bounds``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's bounds value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.Sphere.bounds

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

Returns: ``tuple[float, float, float, float, float, float]``.

``FEM_Periodic_Solver.Sphere.contains``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``contains`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.Sphere.contains(x: 'ArrayLike', y: 'ArrayLike', z: 'ArrayLike') -> 'NDArray[np.bool_]'

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
   * - ``z``
     - Required
     - ``ArrayLike``
     - Physical z coordinate samples in metres; arrays must broadcast with the other coordinate arguments.

Returns: ``NDArray[np.bool_]``.

FEM_Periodic_Solver.StaleDiscretizationError
--------------------------------------------

``FEM_Periodic_Solver.StaleDiscretizationError``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The continuous geometry changed after discretization.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.StaleDiscretizationError(*args: 'object')

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

Returns: ``StaleDiscretizationError``.

FEM_Periodic_Solver.assemble_periodic_system_2d
-----------------------------------------------

``FEM_Periodic_Solver.assemble_periodic_system_2d``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Assemble and periodically reduce the analytic scalar QEP.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.assemble_periodic_system_2d(mesh_data: 'FEMPeriodicMesh2D', *, polarization: 'Polarization', frequency: 'float', k0: 'float', material_at: 'MaterialEvaluator', quadrature_order: 'int' = 4) -> 'PeriodicFEMSystem2D'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``mesh_data``
     - Required
     - ``FEMPeriodicMesh2D``
     - Conforming FEM mesh object. Physical meshes carry SI coordinates; computational meshes may be scaled by the reference length for assembly.
   * - ``polarization``
     - Required
     - ``Polarization``
     - Mode-family label: TE/TM/hybrid or propagating/evanescent as appropriate to the owning result or solver. Keyword-only.
   * - ``frequency``
     - Required
     - ``float``
     - Ordinary frequency in Hz; must be finite and positive. Keyword-only.
   * - ``k0``
     - Required
     - ``float``
     - Vacuum wavenumber in rad/m, used for coordinate scaling and material/PML evaluation. Keyword-only.
   * - ``material_at``
     - Required
     - ``MaterialEvaluator``
     - Vectorized material/field callback evaluated at the coordinates supplied by the calling API; return the scalar, diagonal array, or field shape specified by that operation. Keyword-only.
   * - ``quadrature_order``
     - Optional
     - ``int``
     - Finite-element integration order. Higher-order mixed elements require at least fourth-order quadrature; PML and material variation can require more. Keyword-only. Default: ``4``.

Returns: ``PeriodicFEMSystem2D``.

FEM_Periodic_Solver.assemble_periodic_system_3d
-----------------------------------------------

``FEM_Periodic_Solver.assemble_periodic_system_3d``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Assemble and periodically reduce the electric-field QEP.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.assemble_periodic_system_3d(mesh_data: 'PeriodicMesh3D', *, frequency: 'float', k0: 'float', material_at: 'MaterialEvaluator3D', quadrature_order: 'int' = 3) -> 'PeriodicFEMSystem3D'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``mesh_data``
     - Required
     - ``PeriodicMesh3D``
     - Conforming FEM mesh object. Physical meshes carry SI coordinates; computational meshes may be scaled by the reference length for assembly.
   * - ``frequency``
     - Required
     - ``float``
     - Ordinary frequency in Hz; must be finite and positive. Keyword-only.
   * - ``k0``
     - Required
     - ``float``
     - Vacuum wavenumber in rad/m, used for coordinate scaling and material/PML evaluation. Keyword-only.
   * - ``material_at``
     - Required
     - ``MaterialEvaluator3D``
     - Vectorized material/field callback evaluated at the coordinates supplied by the calling API; return the scalar, diagonal array, or field shape specified by that operation. Keyword-only.
   * - ``quadrature_order``
     - Optional
     - ``int``
     - Finite-element integration order. Higher-order mixed elements require at least fourth-order quadrature; PML and material variation can require more. Keyword-only. Default: ``3``.

Returns: ``PeriodicFEMSystem3D``.

FEM_Periodic_Solver.build_node_prolongation
-------------------------------------------

``FEM_Periodic_Solver.build_node_prolongation``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Build an unsigned scalar-P1 prolongation for periodic node pairs.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.build_node_prolongation(node_count: 'int', slave_nodes: 'ArrayLike', master_nodes: 'ArrayLike', *, constrained_nodes: 'ArrayLike' = ()) -> 'PeriodicProlongation'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``node_count``
     - Required
     - ``int``
     - Total number of nodes or degrees of freedom in the relevant full space; a nonnegative/positive integer as required by the constructor.
   * - ``slave_nodes``
     - Required
     - ``ArrayLike``
     - Matched periodic seam indices. Pair arrays store slave/master correspondence; edge pairs additionally retain orientation through the signed constraint operator.
   * - ``master_nodes``
     - Required
     - ``ArrayLike``
     - Matched periodic seam indices. Pair arrays store slave/master correspondence; edge pairs additionally retain orientation through the signed constraint operator.
   * - ``constrained_nodes``
     - Optional
     - ``ArrayLike``
     - Integer degree-of-freedom indices selecting constrained/free unknowns or admissible scalar test functions. Keyword-only. Default: ``()``.

Returns: ``PeriodicProlongation``.

FEM_Periodic_Solver.build_signed_edge_prolongation
--------------------------------------------------

``FEM_Periodic_Solver.build_signed_edge_prolongation``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Build a signed first-kind Nedelec edge prolongation.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.build_signed_edge_prolongation(edges: 'ArrayLike', slave_nodes: 'ArrayLike', master_nodes: 'ArrayLike', *, node_count: 'int | None' = None, constrained_edges: 'ArrayLike' = ()) -> 'PeriodicProlongation'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``edges``
     - Required
     - ``ArrayLike``
     - Integer edge connectivity or edge-index arrays matching the canonical Nedelec ordering stored in the mesh/system.
   * - ``slave_nodes``
     - Required
     - ``ArrayLike``
     - Matched periodic seam indices. Pair arrays store slave/master correspondence; edge pairs additionally retain orientation through the signed constraint operator.
   * - ``master_nodes``
     - Required
     - ``ArrayLike``
     - Matched periodic seam indices. Pair arrays store slave/master correspondence; edge pairs additionally retain orientation through the signed constraint operator.
   * - ``node_count``
     - Optional
     - ``int | None``
     - Total number of nodes or degrees of freedom in the relevant full space; a nonnegative/positive integer as required by the constructor. Keyword-only. Default: ``None``.
   * - ``constrained_edges``
     - Optional
     - ``ArrayLike``
     - Integer degree-of-freedom indices selecting constrained/free unknowns or admissible scalar test functions. Keyword-only. Default: ``()``.

Returns: ``PeriodicProlongation``.

``edges`` accepts either ``(2, E)`` (scikit-fem convention) or ``(E, 2)``. An edge is identified with the master trace only when both endpoints occur in ``slave_nodes``; incident volume edges stay independent. Reversing the mapped endpoints relative to the actual master-edge orientation contributes ``-1`` to the prolongation row.

FEM_Periodic_Solver.linearized_pencil
-------------------------------------

``FEM_Periodic_Solver.linearized_pencil``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``linearized_pencil`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.linearized_pencil(system: 'PeriodicFEMSystem2D') -> 'tuple[csc_matrix, csc_matrix]'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``system``
     - Required
     - ``PeriodicFEMSystem2D``
     - Assembled FEM system containing compatible bases, sparse operators, constraints, and material/reference-scale metadata.

Returns: ``tuple[csc_matrix, csc_matrix]``.

FEM_Periodic_Solver.linearized_pencil_3d
----------------------------------------

``FEM_Periodic_Solver.linearized_pencil_3d``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``linearized_pencil_3d`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.linearized_pencil_3d(system: 'PeriodicFEMSystem3D') -> 'tuple[csc_matrix, csc_matrix]'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``system``
     - Required
     - ``PeriodicFEMSystem3D``
     - Assembled FEM system containing compatible bases, sparse operators, constraints, and material/reference-scale metadata.

Returns: ``tuple[csc_matrix, csc_matrix]``.

FEM_Periodic_Solver.launch_viewer
---------------------------------

``FEM_Periodic_Solver.launch_viewer``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open a result or result directory in the standalone native viewer.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.launch_viewer(path: 'str | os.PathLike[str] | None' = None, *, _remove_on_exit: 'bool' = False) -> 'subprocess.Popen[bytes]'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``path``
     - Optional
     - ``str | os.PathLike[str] | None``
     - Filesystem destination/source for HDF5 persistence or viewer launch. A directory is accepted only by viewer/directory-inspection APIs; None follows the method-specific default. Default: ``None``.
   * - ``_remove_on_exit``
     - Optional
     - ``bool``
     - Internal viewer-cleanup flag for a temporary archive. Not intended for user-owned result files. Keyword-only. Default: ``False``.

Returns: ``subprocess.Popen[bytes]``.

A file is schema-validated before launch. Passing a directory lets the native GUI present its HDF5 selector; omitting ``path`` uses the current working directory.

FEM_Periodic_Solver.load_periodic_h5
------------------------------------

``FEM_Periodic_Solver.load_periodic_h5``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Load one archive; sweeps return one immutable mode set per case.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.load_periodic_h5(path: 'str | os.PathLike[str]', modes: 'int | slice | Iterable[int] | None' = None) -> 'PeriodicModeSet | tuple[PeriodicModeSet, ...]'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``path``
     - Required
     - ``str | os.PathLike[str]``
     - Filesystem destination/source for HDF5 persistence or viewer launch. A directory is accepted only by viewer/directory-inspection APIs; None follows the method-specific default.
   * - ``modes``
     - Optional
     - ``int | slice | Iterable[int] | None``
     - Mode or mode collection. Integer selectors use the owning API indexing convention; supplied mode sets must match frequency, ky, and transverse geometry. Default: ``None``.

Returns: ``PeriodicModeSet | tuple[PeriodicModeSet, ...]``.

FEM_Periodic_Solver.node_representatives
----------------------------------------

``FEM_Periodic_Solver.node_representatives``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Return the master-root representative of every mesh node.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.node_representatives(node_count: 'int', slave_nodes: 'ArrayLike', master_nodes: 'ArrayLike') -> 'IntArray'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``node_count``
     - Required
     - ``int``
     - Total number of nodes or degrees of freedom in the relevant full space; a nonnegative/positive integer as required by the constructor.
   * - ``slave_nodes``
     - Required
     - ``ArrayLike``
     - Matched periodic seam indices. Pair arrays store slave/master correspondence; edge pairs additionally retain orientation through the signed constraint operator.
   * - ``master_nodes``
     - Required
     - ``ArrayLike``
     - Matched periodic seam indices. Pair arrays store slave/master correspondence; edge pairs additionally retain orientation through the signed constraint operator.

Returns: ``IntArray``.

FEM_Periodic_Solver.open_periodic_h5
------------------------------------

``FEM_Periodic_Solver.open_periodic_h5``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open only the archive root and index, returning a lazy reader.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.open_periodic_h5(path: 'str | os.PathLike[str]') -> 'PeriodicH5Archive'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``path``
     - Required
     - ``str | os.PathLike[str]``
     - Filesystem destination/source for HDF5 persistence or viewer launch. A directory is accepted only by viewer/directory-inspection APIs; None follows the method-specific default.

Returns: ``PeriodicH5Archive``.

FEM_Periodic_Solver.save_periodic_h5
------------------------------------

``FEM_Periodic_Solver.save_periodic_h5``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Atomically save one periodic FEM result case.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.save_periodic_h5(mode_set: 'PeriodicModeSet', path: 'str | os.PathLike[str]') -> 'Path'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``mode_set``
     - Required
     - ``PeriodicModeSet``
     - Mode or mode collection. Integer selectors use the owning API indexing convention; supplied mode sets must match frequency, ky, and transverse geometry.
   * - ``path``
     - Required
     - ``str | os.PathLike[str]``
     - Filesystem destination/source for HDF5 persistence or viewer launch. A directory is accepted only by viewer/directory-inspection APIs; None follows the method-specific default.

Returns: ``Path``.

FEM_Periodic_Solver.save_periodic_sweep_h5
------------------------------------------

``FEM_Periodic_Solver.save_periodic_sweep_h5``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Atomically save a nonempty frequency/parameter sweep with deduplicated states.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.save_periodic_sweep_h5(mode_sets: 'Iterable[PeriodicModeSet]', path: 'str | os.PathLike[str]') -> 'Path'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``mode_sets``
     - Required
     - ``Iterable[PeriodicModeSet]``
     - Mode or mode collection. Integer selectors use the owning API indexing convention; supplied mode sets must match frequency, ky, and transverse geometry.
   * - ``path``
     - Required
     - ``str | os.PathLike[str]``
     - Filesystem destination/source for HDF5 persistence or viewer launch. A directory is accepted only by viewer/directory-inspection APIs; None follows the method-specific default.

Returns: ``Path``.

FEM_Periodic_Solver.solve_qep_candidates
----------------------------------------

``FEM_Periodic_Solver.solve_qep_candidates``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Return roots and scalar coefficient vectors nearest ``target``.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.solve_qep_candidates(system: 'PeriodicFEMSystem2D', *, target: 'complex', candidate_count: 'int', tolerance: 'float' = 1e-10, eigensolver: 'str' = 'auto', arnoldi_backend: 'str' = 'auto', dense_linearization_limit: 'int' = 700) -> 'tuple[ComplexArray, ComplexArray, NDArray[np.float64], str]'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``system``
     - Required
     - ``PeriodicFEMSystem2D``
     - Assembled FEM system containing compatible bases, sparse operators, constraints, and material/reference-scale metadata.
   * - ``target``
     - Required
     - ``complex``
     - Target material/field object or requested eigenspectral value for this conversion/evaluation operation. Keyword-only.
   * - ``candidate_count``
     - Required
     - ``int``
     - Number of modes or candidate eigenpairs requested; a positive integer. Candidate pools may include roots later rejected by validation. Keyword-only.
   * - ``tolerance``
     - Optional
     - ``float``
     - Positive numerical tolerance. Linear/QEP residual tolerances validate the algebraic solve; they do not set the adaptive mesh threshold. Keyword-only. Default: ``1e-10``.
   * - ``eigensolver``
     - Optional
     - ``str``
     - Eigensolver backend selector. Accepted choices are given by the signature Literal or the method behavior notes; auto selects by system size. Keyword-only. Default: ``'auto'``.
   * - ``arnoldi_backend``
     - Optional
     - ``str``
     - Eigensolver backend selector. Accepted choices are given by the signature Literal or the method behavior notes; auto selects by system size. Keyword-only. Default: ``'auto'``.
   * - ``dense_linearization_limit``
     - Optional
     - ``int``
     - Matrix-size cutoff for dense eigensolving; larger systems use a sparse backend. This is a dimension limit, not a mesh-error threshold. Keyword-only. Default: ``700``.

Returns: ``tuple[ComplexArray, ComplexArray, NDArray[np.float64], str]``.

FEM_Periodic_Solver.visualize
-----------------------------

``FEM_Periodic_Solver.visualize``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create and show a Matplotlib figure for one selected mode.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.visualize(mode: 'PeriodicMode', component: 'str' = 'Ey', quantity: 'str' = 'real', *, ax: 'Any' = None, cmap: 'str' = 'RdBu_r', show_mesh: 'bool' = False, colorbar: 'bool' = True, show: 'bool' = True, slice_axis: 'str | None' = None, slice_fraction: 'float' = 0.5, max_points: 'int' = 5000) -> 'Any'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``mode``
     - Required
     - ``PeriodicMode``
     - Mode or mode collection. Integer selectors use the owning API indexing convention; supplied mode sets must match frequency, ky, and transverse geometry.
   * - ``component``
     - Optional
     - ``str``
     - Field component selection, using Cartesian electric/magnetic names such as Ex or Hy. The visualization API also supports its documented aggregate quantities. Default: ``'Ey'``.
   * - ``quantity``
     - Optional
     - ``str``
     - Complex-data display selection, such as real, imag, abs, or phase; accepted values are given by the owning plotting API. Default: ``'real'``.
   * - ``ax``
     - Optional
     - ``Any``
     - Existing Matplotlib axes for embedding a plot; None creates suitable axes when accepted. Keyword-only. Default: ``None``.
   * - ``cmap``
     - Optional
     - ``str``
     - Matplotlib colormap name or object used for scalar field rendering. Keyword-only. Default: ``'RdBu_r'``.
   * - ``show_mesh``
     - Optional
     - ``bool``
     - Enable display of the figure, mesh overlay, or colorbar respectively. show=False returns plotting objects without opening a window. Keyword-only. Default: ``False``.
   * - ``colorbar``
     - Optional
     - ``bool``
     - Enable display of the figure, mesh overlay, or colorbar respectively. show=False returns plotting objects without opening a window. Keyword-only. Default: ``True``.
   * - ``show``
     - Optional
     - ``bool``
     - Enable display of the figure, mesh overlay, or colorbar respectively. show=False returns plotting objects without opening a window. Keyword-only. Default: ``True``.
   * - ``slice_axis``
     - Optional
     - ``str | None``
     - 3D visualization cut: Cartesian axis and fractional location in [0, 1] along that domain extent. Keyword-only. Default: ``None``.
   * - ``slice_fraction``
     - Optional
     - ``float``
     - 3D visualization cut: Cartesian axis and fractional location in [0, 1] along that domain extent. Keyword-only. Default: ``0.5``.
   * - ``max_points``
     - Optional
     - ``int``
     - Positive sample/glyph budget for rendering; affects visualization density, not the FEM solution. Keyword-only. Default: ``5000``.

Returns: ``Any``.

FEM_Periodic_Solver.validate_periodic_h5
----------------------------------------

``FEM_Periodic_Solver.validate_periodic_h5``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Validate schema/index data and optionally every referenced heavy dataset.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.validate_periodic_h5(path: 'str | os.PathLike[str]', *, deep: 'bool' = False) -> 'H5ValidationReport'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``path``
     - Required
     - ``str | os.PathLike[str]``
     - Filesystem destination/source for HDF5 persistence or viewer launch. A directory is accepted only by viewer/directory-inspection APIs; None follows the method-specific default.
   * - ``deep``
     - Optional
     - ``bool``
     - Validate stored array payloads and cross-dataset invariants in addition to lightweight HDF5 schema checks. Keyword-only. Default: ``False``.

Returns: ``H5ValidationReport``.

FEM_Periodic_Solver.discretize_3d
---------------------------------

``FEM_Periodic_Solver.discretize_3d``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate a first-order tetrahedral mesh with matching ``z`` faces.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.discretize_3d(geometry: 'GeometryModel3D', *, max_element_size: 'float | None' = None, wavelength_elements: 'int' = 8, material_aware: 'bool' = True, element_order: 'int' = 1, k0: 'float | None' = None, _refinement_scale: 'float' = 1.0) -> 'PeriodicMesh3D'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``geometry``
     - Required
     - ``GeometryModel3D``
     - Geometry model containing physical bounds, material regions, conductor boundaries, PMLs, and sizing requests.
   * - ``max_element_size``
     - Optional
     - ``float | None``
     - Physical element-edge length in metres. A maximum target is an upper sizing request; material, boundary, and wavelength constraints may produce smaller cells. Keyword-only. Default: ``None``.
   * - ``wavelength_elements``
     - Optional
     - ``int``
     - Minimum requested elements per shortest local material wavelength; integer at least four. Public adaptive workflows default to four. Keyword-only. Default: ``8``.
   * - ``material_aware``
     - Optional
     - ``bool``
     - Enable element-size reduction in high-index/high-permittivity material regions while retaining the global maximum-edge cap. Keyword-only. Default: ``True``.
   * - ``element_order``
     - Optional
     - ``int``
     - Finite-element polynomial-order selection. Standalone 2D modes and WaveFEM scattering accept 1 (N1/P1) or 2 (N2/P2); other backends retain their fixed compatible spaces. Keyword-only. Default: ``1``.
   * - ``k0``
     - Optional
     - ``float | None``
     - Vacuum wavenumber in rad/m, used for coordinate scaling and material/PML evaluation. Keyword-only. Default: ``None``.
   * - ``_refinement_scale``
     - Optional
     - ``float``
     - Mesh-density multiplier. Public refine requires a finite value greater than one; the internal refinement scale accumulates requested sizing changes. Keyword-only. Default: ``1.0``.

Returns: ``PeriodicMesh3D``.

FEM_Periodic_Solver.assembly_2d.evaluate_material
-------------------------------------------------

``FEM_Periodic_Solver.assembly_2d.evaluate_material``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluate material; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.assembly_2d.evaluate_material(material_at: 'MaterialEvaluator', x: 'NDArray[np.floating]', z: 'NDArray[np.floating]') -> 'tuple[ComplexArray, ComplexArray]'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``material_at``
     - Required
     - ``MaterialEvaluator``
     - Vectorized material/field callback evaluated at the coordinates supplied by the calling API; return the scalar, diagonal array, or field shape specified by that operation.
   * - ``x``
     - Required
     - ``NDArray[np.floating]``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature.
   * - ``z``
     - Required
     - ``NDArray[np.floating]``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature.

Returns: ``tuple[ComplexArray, ComplexArray]``.

FEM_Periodic_Solver.assembly_3d.evaluate_material_3d
----------------------------------------------------

``FEM_Periodic_Solver.assembly_3d.evaluate_material_3d``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluate material 3d; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.assembly_3d.evaluate_material_3d(material_at: 'MaterialEvaluator3D', x: 'NDArray[np.floating]', y: 'NDArray[np.floating]', z: 'NDArray[np.floating]') -> 'tuple[ComplexArray, ComplexArray]'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``material_at``
     - Required
     - ``MaterialEvaluator3D``
     - Vectorized material/field callback evaluated at the coordinates supplied by the calling API; return the scalar, diagonal array, or field shape specified by that operation.
   * - ``x``
     - Required
     - ``NDArray[np.floating]``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature.
   * - ``y``
     - Required
     - ``NDArray[np.floating]``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature.
   * - ``z``
     - Required
     - ``NDArray[np.floating]``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature.

Returns: ``tuple[ComplexArray, ComplexArray]``.

FEM_Periodic_Solver.geometry.Shape2D
------------------------------------

``FEM_Periodic_Solver.geometry.Shape2D``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Structural typing protocol for compatible field/visualization objects. Implement the declared attributes and methods; this protocol is not instantiated directly.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.geometry.Shape2D()

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

Returns: ``Shape2D``.

Protocol classes are defined as:

class Proto(Protocol): def meth(self) -> int: ...

Such classes are primarily used with static type checkers that recognize structural subtyping (static duck-typing).

For example:

class C: def meth(self) -> int: return 0

def func(x: Proto) -> int: return x.meth()

func(C()) # Passes static type check

See PEP 544 for details. Protocol classes decorated with @typing.runtime_checkable act as simple-minded runtime protocols that check only the presence of given attributes, ignoring their type signatures. Protocol classes can be generic, they are defined as:

class GenProto[T](Protocol): def meth(self) -> T: ...

``FEM_Periodic_Solver.geometry.Shape2D.bounds``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's bounds value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.geometry.Shape2D.bounds

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

``FEM_Periodic_Solver.geometry.Shape2D.contains``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``contains`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.geometry.Shape2D.contains(x: 'ArrayLike', z: 'ArrayLike') -> 'NDArray[np.bool_]'

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
   * - ``z``
     - Required
     - ``ArrayLike``
     - Physical z coordinate samples in metres; arrays must broadcast with the other coordinate arguments.

Returns: ``NDArray[np.bool_]``.

FEM_Periodic_Solver.geometry.Shape3D
------------------------------------

``FEM_Periodic_Solver.geometry.Shape3D``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Structural typing protocol for compatible field/visualization objects. Implement the declared attributes and methods; this protocol is not instantiated directly.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.geometry.Shape3D()

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

Returns: ``Shape3D``.

Protocol classes are defined as:

class Proto(Protocol): def meth(self) -> int: ...

Such classes are primarily used with static type checkers that recognize structural subtyping (static duck-typing).

For example:

class C: def meth(self) -> int: return 0

def func(x: Proto) -> int: return x.meth()

func(C()) # Passes static type check

See PEP 544 for details. Protocol classes decorated with @typing.runtime_checkable act as simple-minded runtime protocols that check only the presence of given attributes, ignoring their type signatures. Protocol classes can be generic, they are defined as:

class GenProto[T](Protocol): def meth(self) -> T: ...

``FEM_Periodic_Solver.geometry.Shape3D.bounds``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's bounds value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.geometry.Shape3D.bounds

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

Returns: ``tuple[float, float, float, float, float, float]``.

``FEM_Periodic_Solver.geometry.Shape3D.contains``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``contains`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.geometry.Shape3D.contains(x: 'ArrayLike', y: 'ArrayLike', z: 'ArrayLike') -> 'NDArray[np.bool_]'

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
   * - ``z``
     - Required
     - ``ArrayLike``
     - Physical z coordinate samples in metres; arrays must broadcast with the other coordinate arguments.

Returns: ``NDArray[np.bool_]``.

FEM_Periodic_Solver.geometry.physical_span
------------------------------------------

``FEM_Periodic_Solver.geometry.physical_span``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``physical_span`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.geometry.physical_span(value: 'float | Sequence[float]', name: 'str') -> 'tuple[float, float]'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``value``
     - Required
     - ``float | Sequence[float]``
     - Scalar or array to validate, transform, interpolate, or store. It must satisfy the owning operation and the expected type in this table.
   * - ``name``
     - Required
     - ``str``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported.

Returns: ``tuple[float, float]``.

FEM_Periodic_Solver.materials.diagonal_values
---------------------------------------------

``FEM_Periodic_Solver.materials.diagonal_values``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Return a scalar or three entries as the physical ``xx, yy, zz`` diagonal.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.materials.diagonal_values(value: 'MaterialInput', name: 'str') -> 'tuple[complex, complex, complex]'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``value``
     - Required
     - ``MaterialInput``
     - Scalar or array to validate, transform, interpolate, or store. It must satisfy the owning operation and the expected type in this table.
   * - ``name``
     - Required
     - ``str``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported.

Returns: ``tuple[complex, complex, complex]``.

FEM_Periodic_Solver.meshing_2d.discretize_periodic_2d
-----------------------------------------------------

``FEM_Periodic_Solver.meshing_2d.discretize_periodic_2d``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a first-order triangular mesh with a Gmsh periodic node map.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.meshing_2d.discretize_periodic_2d(geometry: 'GeometryModel2D', *, max_element_size: 'float', element_order: 'int' = 1) -> 'FEMPeriodicMesh2D'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``geometry``
     - Required
     - ``GeometryModel2D``
     - Geometry model containing physical bounds, material regions, conductor boundaries, PMLs, and sizing requests.
   * - ``max_element_size``
     - Required
     - ``float``
     - Physical element-edge length in metres. A maximum target is an upper sizing request; material, boundary, and wavelength constraints may produce smaller cells. Keyword-only.
   * - ``element_order``
     - Optional
     - ``int``
     - Finite-element polynomial-order selection. Standalone 2D modes and WaveFEM scattering accept 1 (N1/P1) or 2 (N2/P2); other backends retain their fixed compatible spaces. Keyword-only. Default: ``1``.

Returns: ``FEMPeriodicMesh2D``.

FEM_Periodic_Solver.persistence._viewer_candidates
--------------------------------------------------

``FEM_Periodic_Solver.persistence._viewer_candidates``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Return override, checkout-build, and installed viewer candidates.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.persistence._viewer_candidates(executable_name: 'str') -> 'tuple[Path, ...]'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``executable_name``
     - Required
     - ``str``
     - Native viewer executable basename; discovery uses the documented environment variable, build paths, and PATH.

Returns: ``tuple[Path, ...]``.

FEM_Periodic_Solver.visualization.visualize_with_gui
----------------------------------------------------

``FEM_Periodic_Solver.visualization.visualize_with_gui``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open every available mode in a temporary native-viewer archive.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Periodic_Solver.visualization.visualize_with_gui(mode_set: 'PeriodicModeSet') -> 'Any'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``mode_set``
     - Required
     - ``PeriodicModeSet``
     - Mode or mode collection. Integer selectors use the owning API indexing convention; supplied mode sets must match frequency, ky, and transverse geometry.

Returns: ``Any``.

Export aliases and constants
----------------------------

Aliases below have exactly the same input tables and return contracts as their targets. Constants/type aliases are values, not calls, and take no input arguments.

.. list-table:: Exports
   :header-rows: 1

   * - Name
     - Value or target
   * - ``FEM_Periodic_Solver.PeriodicMesh3D``
     - ``FEM_Periodic_Solver.FEMPeriodicMesh3D``
   * - ``FEM_Periodic_Solver.PeriodicMode``
     - ``FEM_Periodic_Solver.Mode``
   * - ``FEM_Periodic_Solver.PeriodicModeSet``
     - ``FEM_Periodic_Solver.ModeSet``
   * - ``FEM_Periodic_Solver.SampledFields``
     - ``FEM_Periodic_Solver.PeriodicSampledFields``
   * - ``FEM_Periodic_Solver.assembly_2d.PeriodicFEMSystem2D``
     - ``FEM_Periodic_Solver.PeriodicFEMSystem2D``
   * - ``FEM_Periodic_Solver.assembly_2d.assemble_periodic_system_2d``
     - ``FEM_Periodic_Solver.assemble_periodic_system_2d``
   * - ``FEM_Periodic_Solver.assembly_2d.linearized_pencil``
     - ``FEM_Periodic_Solver.linearized_pencil``
   * - ``FEM_Periodic_Solver.assembly_2d.solve_qep_candidates``
     - ``FEM_Periodic_Solver.solve_qep_candidates``
   * - ``FEM_Periodic_Solver.assembly_3d.PeriodicFEMSystem3D``
     - ``FEM_Periodic_Solver.PeriodicFEMSystem3D``
   * - ``FEM_Periodic_Solver.assembly_3d.assemble_periodic_system_3d``
     - ``FEM_Periodic_Solver.assemble_periodic_system_3d``
   * - ``FEM_Periodic_Solver.assembly_3d.linearized_pencil_3d``
     - ``FEM_Periodic_Solver.linearized_pencil_3d``
   * - ``FEM_Periodic_Solver.exceptions.BackendCapabilityError``
     - ``FEM_Periodic_Solver.BackendCapabilityError``
   * - ``FEM_Periodic_Solver.exceptions.ConfigurationError``
     - ``FEM_Periodic_Solver.ConfigurationError``
   * - ``FEM_Periodic_Solver.exceptions.FEMPeriodicSolverError``
     - ``FEM_Periodic_Solver.FEMPeriodicSolverError``
   * - ``FEM_Periodic_Solver.exceptions.GeometryError``
     - ``FEM_Periodic_Solver.GeometryError``
   * - ``FEM_Periodic_Solver.exceptions.MeshError``
     - ``FEM_Periodic_Solver.MeshError``
   * - ``FEM_Periodic_Solver.exceptions.NotDiscretizedError``
     - ``FEM_Periodic_Solver.NotDiscretizedError``
   * - ``FEM_Periodic_Solver.exceptions.PersistenceError``
     - ``FEM_Periodic_Solver.PersistenceError``
   * - ``FEM_Periodic_Solver.exceptions.SolverError``
     - ``FEM_Periodic_Solver.SolverError``
   * - ``FEM_Periodic_Solver.exceptions.StaleDiscretizationError``
     - ``FEM_Periodic_Solver.StaleDiscretizationError``
   * - ``FEM_Periodic_Solver.geometry.BoundaryRegion``
     - ``FEM_Periodic_Solver.BoundaryRegion``
   * - ``FEM_Periodic_Solver.geometry.Box``
     - ``FEM_Periodic_Solver.Box``
   * - ``FEM_Periodic_Solver.geometry.Circle``
     - ``FEM_Periodic_Solver.Circle``
   * - ``FEM_Periodic_Solver.geometry.Cylinder``
     - ``FEM_Periodic_Solver.Cylinder``
   * - ``FEM_Periodic_Solver.geometry.GeometryModel2D``
     - ``FEM_Periodic_Solver.GeometryModel2D``
   * - ``FEM_Periodic_Solver.geometry.GeometryModel3D``
     - ``FEM_Periodic_Solver.GeometryModel3D``
   * - ``FEM_Periodic_Solver.geometry.MeshRefinement``
     - ``FEM_Periodic_Solver.MeshRefinement``
   * - ``FEM_Periodic_Solver.geometry.PMLSpec``
     - ``FEM_Periodic_Solver.PMLSpec``
   * - ``FEM_Periodic_Solver.geometry.Polygon``
     - ``FEM_Periodic_Solver.Polygon``
   * - ``FEM_Periodic_Solver.geometry.Rectangle``
     - ``FEM_Periodic_Solver.Rectangle``
   * - ``FEM_Periodic_Solver.geometry.Region``
     - ``FEM_Periodic_Solver.Region``
   * - ``FEM_Periodic_Solver.geometry.Sphere``
     - ``FEM_Periodic_Solver.Sphere``
   * - ``FEM_Periodic_Solver.materials.Material``
     - ``FEM_Periodic_Solver.Material``
   * - ``FEM_Periodic_Solver.meshing_2d.FEMPeriodicMesh2D``
     - ``FEM_Periodic_Solver.FEMPeriodicMesh2D``
   * - ``FEM_Periodic_Solver.meshing_2d.MeshInfo``
     - ``FEM_Periodic_Solver.MeshInfo``
   * - ``FEM_Periodic_Solver.meshing_3d.FEMPeriodicMesh3D``
     - ``FEM_Periodic_Solver.FEMPeriodicMesh3D``
   * - ``FEM_Periodic_Solver.meshing_3d.MeshInfo3D``
     - ``FEM_Periodic_Solver.MeshInfo3D``
   * - ``FEM_Periodic_Solver.meshing_3d.PeriodicMesh3D``
     - ``FEM_Periodic_Solver.FEMPeriodicMesh3D``
   * - ``FEM_Periodic_Solver.meshing_3d.discretize_3d``
     - ``FEM_Periodic_Solver.discretize_3d``
   * - ``FEM_Periodic_Solver.periodic.PeriodicProlongation``
     - ``FEM_Periodic_Solver.PeriodicProlongation``
   * - ``FEM_Periodic_Solver.periodic.build_node_prolongation``
     - ``FEM_Periodic_Solver.build_node_prolongation``
   * - ``FEM_Periodic_Solver.periodic.build_signed_edge_prolongation``
     - ``FEM_Periodic_Solver.build_signed_edge_prolongation``
   * - ``FEM_Periodic_Solver.periodic.node_representatives``
     - ``FEM_Periodic_Solver.node_representatives``
   * - ``FEM_Periodic_Solver.persistence.H5ValidationReport``
     - ``FEM_Periodic_Solver.H5ValidationReport``
   * - ``FEM_Periodic_Solver.persistence.PeriodicH5Archive``
     - ``FEM_Periodic_Solver.PeriodicH5Archive``
   * - ``FEM_Periodic_Solver.persistence.launch_viewer``
     - ``FEM_Periodic_Solver.launch_viewer``
   * - ``FEM_Periodic_Solver.persistence.load_periodic_h5``
     - ``FEM_Periodic_Solver.load_periodic_h5``
   * - ``FEM_Periodic_Solver.persistence.open_periodic_h5``
     - ``FEM_Periodic_Solver.open_periodic_h5``
   * - ``FEM_Periodic_Solver.persistence.save_periodic_h5``
     - ``FEM_Periodic_Solver.save_periodic_h5``
   * - ``FEM_Periodic_Solver.persistence.save_periodic_sweep_h5``
     - ``FEM_Periodic_Solver.save_periodic_sweep_h5``
   * - ``FEM_Periodic_Solver.persistence.validate_periodic_h5``
     - ``FEM_Periodic_Solver.validate_periodic_h5``
   * - ``FEM_Periodic_Solver.results.Mode``
     - ``FEM_Periodic_Solver.Mode``
   * - ``FEM_Periodic_Solver.results.ModeSet``
     - ``FEM_Periodic_Solver.ModeSet``
   * - ``FEM_Periodic_Solver.results.PeriodicMode``
     - ``FEM_Periodic_Solver.Mode``
   * - ``FEM_Periodic_Solver.results.PeriodicModeSet``
     - ``FEM_Periodic_Solver.ModeSet``
   * - ``FEM_Periodic_Solver.results.PeriodicSampledFields``
     - ``FEM_Periodic_Solver.PeriodicSampledFields``
   * - ``FEM_Periodic_Solver.results.SampledFields``
     - ``FEM_Periodic_Solver.PeriodicSampledFields``
   * - ``FEM_Periodic_Solver.solver_2d.PeriodicModeSolver2D``
     - ``FEM_Periodic_Solver.PeriodicModeSolver2D``
   * - ``FEM_Periodic_Solver.solver_3d.PeriodicModeSolver3D``
     - ``FEM_Periodic_Solver.PeriodicModeSolver3D``
   * - ``FEM_Periodic_Solver.visualization.visualize``
     - ``FEM_Periodic_Solver.visualize``
   * - ``FEM_Periodic_Solver.assembly_2d.MaterialEvaluator``
     - ``Type alias; see the array/material conventions and owning module annotation.``
   * - ``FEM_Periodic_Solver.assembly_3d.MaterialEvaluator3D``
     - ``Type alias; see the array/material conventions and owning module annotation.``
   * - ``FEM_Periodic_Solver.constants.C_0``
     - ``299792458.0``
   * - ``FEM_Periodic_Solver.constants.EPSILON_0``
     - ``8.8541878188e-12``
   * - ``FEM_Periodic_Solver.constants.ETA_0``
     - ``376.7303134118051``
   * - ``FEM_Periodic_Solver.constants.MU_0``
     - ``1.25663706127e-06``
   * - ``FEM_Periodic_Solver.materials.MaterialInput``
     - ``float | complex | numpy.number | tuple[float | complex | numpy.number, float | complex | numpy.number, float | complex | numpy.number] | list[float | complex | numpy.number] | numpy.ndarray``
