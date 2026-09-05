FEM_Mode_Solver API reference
=============================

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

FEM_Mode_Solver.BackendCapabilityError
--------------------------------------

``FEM_Mode_Solver.BackendCapabilityError``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A requested physical feature is not supported by this FEM backend.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.BackendCapabilityError(*args: 'object')

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

FEM_Mode_Solver.BoundaryRegion
------------------------------

``FEM_Mode_Solver.BoundaryRegion``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Construct a ``BoundaryRegion`` record. Its public data fields use the same names and types as the constructor inputs below. Solvers normally construct mesh/system/result records for you.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.BoundaryRegion(id: 'int', name: 'str', shape: 'Interval | Shape2D', kind: 'str', impedance: 'complex | None' = None) -> None

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
     - ``Interval | Shape2D``
     - Geometry primitive or region selector identifying the physical support. Use the class/union in the signature; electrostatics also accepts named exterior boundaries.
   * - ``kind``
     - Required
     - ``str``
     - Object/boundary/scene kind. Use the permitted Literal values or the documented selector for the owning class.
   * - ``impedance``
     - Optional
     - ``complex | None``
     - Surface impedance in ohms. Supply an explicit passive complex value or a supported metal through the alternative material input. Default: ``None``.

Returns: ``BoundaryRegion``.

FEM_Mode_Solver.Circle
----------------------

``FEM_Mode_Solver.Circle``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Define a circular region, optionally annular where an inner radius is supported. Use this immutable primitive in geometry/material placement APIs.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.Circle(center: 'tuple[float, float]', radius: 'float', inner_radius: 'float | None' = None) -> None

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

``FEM_Mode_Solver.Circle.bounds``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's bounds value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.Circle.bounds

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

``FEM_Mode_Solver.Circle.contains``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``contains`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.Circle.contains(x: 'ArrayLike', y: 'ArrayLike') -> 'NDArray[np.bool_]'

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

FEM_Mode_Solver.ConfigurationError
----------------------------------

``FEM_Mode_Solver.ConfigurationError``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The continuous model or solver configuration is invalid.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ConfigurationError(*args: 'object')

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

FEM_Mode_Solver.FEMMesh1D
-------------------------

``FEM_Mode_Solver.FEMMesh1D``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Construct a ``FEMMesh1D`` record. Its public data fields use the same names and types as the constructor inputs below. Solvers normally construct mesh/system/result records for you.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.FEMMesh1D(mesh: 'MeshLine', nodes: 'NDArray[np.float64]', info: 'MeshInfo', geometry_revision: 'int') -> None

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``mesh``
     - Required
     - ``MeshLine``
     - Conforming FEM mesh object. Physical meshes carry SI coordinates; computational meshes may be scaled by the reference length for assembly.
   * - ``nodes``
     - Required
     - ``NDArray[np.float64]``
     - Mesh-node coordinates. Native scikit-fem arrays are dimension by node; standalone result coordinates are commonly node by dimension.
   * - ``info``
     - Required
     - ``MeshInfo``
     - Structured metadata/diagnostic container associated with the mesh or result. solve_info/metadata retain applied adaptive controls and stopping history.
   * - ``geometry_revision``
     - Required
     - ``int``
     - Geometry version captured when this mesh was built; stale versions invalidate cached systems and results.

Returns: ``FEMMesh1D``.

FEM_Mode_Solver.FEMMesh2D
-------------------------

``FEM_Mode_Solver.FEMMesh2D``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Construct a ``FEMMesh2D`` record. Its public data fields use the same names and types as the constructor inputs below. Solvers normally construct mesh/system/result records for you.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.FEMMesh2D(mesh: 'MeshTri', element_tags: 'NDArray[np.int32]', physical_names: 'dict[int, str]', boundary_facets: 'dict[str, NDArray[np.int64]]', info: 'MeshInfo', geometry_revision: 'int') -> None

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
   * - ``info``
     - Required
     - ``MeshInfo``
     - Structured metadata/diagnostic container associated with the mesh or result. solve_info/metadata retain applied adaptive controls and stopping history.
   * - ``geometry_revision``
     - Required
     - ``int``
     - Geometry version captured when this mesh was built; stale versions invalidate cached systems and results.

Returns: ``FEMMesh2D``.

``FEM_Mode_Solver.FEMMesh2D.elements``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's elements value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.FEMMesh2D.elements

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

``FEM_Mode_Solver.FEMMesh2D.nodes``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's nodes value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.FEMMesh2D.nodes

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

FEM_Mode_Solver.FEMModeSolverError
----------------------------------

``FEM_Mode_Solver.FEMModeSolverError``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Base class for all public FEM mode-solver errors.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.FEMModeSolverError(*args: 'object')

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

Returns: ``FEMModeSolverError``.

FEM_Mode_Solver.GeometryError
-----------------------------

``FEM_Mode_Solver.GeometryError``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A geometry primitive or region operation is invalid.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.GeometryError(*args: 'object')

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

FEM_Mode_Solver.Interval
------------------------

``FEM_Mode_Solver.Interval``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define a one-dimensional physical interval. Use this immutable primitive in geometry/material placement APIs.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.Interval(x: 'tuple[float, float]') -> None

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

``FEM_Mode_Solver.Interval.contains``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``contains`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.Interval.contains(x: 'ArrayLike') -> 'NDArray[np.bool_]'

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

Returns: ``NDArray[np.bool_]``.

FEM_Mode_Solver.Material
------------------------

``FEM_Mode_Solver.Material``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Relative diagonal permittivity and permeability.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.Material(epsilon: 'MaterialInput' = 1.0, mu: 'MaterialInput' = 1.0) -> 'None'

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

Complex values follow the package's ``exp(+i*omega*t)`` convention.

``FEM_Mode_Solver.Material.eps_array``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``eps_array`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.Material.eps_array() -> 'np.ndarray'

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

``FEM_Mode_Solver.Material.epsilon``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's epsilon value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.Material.epsilon

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

Returns: ``tuple[complex, complex, complex]``.

``FEM_Mode_Solver.Material.isotropic``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's isotropic value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.Material.isotropic

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

``FEM_Mode_Solver.Material.mu``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's mu value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.Material.mu

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

Returns: ``tuple[complex, complex, complex]``.

``FEM_Mode_Solver.Material.mu_array``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``mu_array`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.Material.mu_array() -> 'np.ndarray'

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

FEM_Mode_Solver.MeshRefinement
------------------------------

``FEM_Mode_Solver.MeshRefinement``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A geometry-only 2D mesh-size control.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.MeshRefinement(id: 'int', name: 'str', shape: 'Shape2D', max_element_size: 'float', transition_width: 'float' = 0.0) -> None

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
     - ``Shape2D``
     - Geometry primitive or region selector identifying the physical support. Use the class/union in the signature; electrostatics also accepts named exterior boundaries.
   * - ``max_element_size``
     - Required
     - ``float``
     - Physical element-edge length in metres. A maximum target is an upper sizing request; material, boundary, and wavelength constraints may produce smaller cells.
   * - ``transition_width``
     - Optional
     - ``float``
     - Physical distance in metres over which the local mesh-size target transitions back to the surrounding target; None selects the mesher default when permitted. Default: ``0.0``.

Returns: ``MeshRefinement``.

Refinement regions participate in OCC fragmentation so their boundary is represented exactly, but they never change material or boundary tags. ``transition_width`` controls the physical distance over which Gmsh grades from ``max_element_size`` back to the surrounding target size.

FEM_Mode_Solver.MeshError
-------------------------

``FEM_Mode_Solver.MeshError``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The continuous model could not be discretized.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.MeshError(*args: 'object')

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

FEM_Mode_Solver.MeshInfo
------------------------

``FEM_Mode_Solver.MeshInfo``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Construct a ``MeshInfo`` record. Its public data fields use the same names and types as the constructor inputs below. Solvers normally construct mesh/system/result records for you.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.MeshInfo(nodes: 'int', elements: 'int', minimum_edge: 'float', maximum_edge: 'float', requested_maximum_edge: 'float', element_order: 'int', material_aware: 'bool' = False, interface_refinement: 'float | None' = None, boundary_refinement: 'float | None' = None, refinement_regions: 'int' = 0) -> None

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
     - Required
     - ``int``
     - Finite-element polynomial-order selection. Standalone 2D modes and WaveFEM scattering accept 1 (N1/P1) or 2 (N2/P2); other backends retain their fixed compatible spaces.
   * - ``material_aware``
     - Optional
     - ``bool``
     - Enable element-size reduction in high-index/high-permittivity material regions while retaining the global maximum-edge cap. Default: ``False``.
   * - ``interface_refinement``
     - Optional
     - ``float | None``
     - Optional size multiplier in (0, 1] near conductor/exterior boundaries or material interfaces. None disables this sizing field. Default: ``None``.
   * - ``boundary_refinement``
     - Optional
     - ``float | None``
     - Optional size multiplier in (0, 1] near conductor/exterior boundaries or material interfaces. None disables this sizing field. Default: ``None``.
   * - ``refinement_regions``
     - Optional
     - ``int``
     - Count or collection of explicit geometry-based local sizing regions, as indicated by the mesh metadata type. Default: ``0``.

Returns: ``MeshInfo``.

FEM_Mode_Solver.Mode
--------------------

``FEM_Mode_Solver.Mode``
~~~~~~~~~~~~~~~~~~~~~~~~

One immutable mode and its sampled electric/magnetic fields.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.Mode(neff: 'complex', beta: 'complex', fields: 'SampledFields', index: 'int' = 1, polarization: 'str | None' = None, eigenvalue: 'complex | None' = None, power: 'complex | None' = None, normalization: 'str' = 'unnormalized', residual: 'float | None' = None, divergence_residual: 'float | None' = None, metadata: 'Mapping[str, Any]' = <factory>) -> None

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
   * - ``beta``
     - Required
     - ``complex``
     - Complex longitudinal propagation constant(s), in rad/m.
   * - ``fields``
     - Required
     - ``SampledFields``
     - Complex electromagnetic field samples or FE field objects. Cartesian components follow x,y,z order; left/right denote the selected trace side.
   * - ``index``
     - Optional
     - ``int``
     - Mode, case, array, or mapping selector. Python indexing is zero based; explicit mode(number) and standalone visualization use their documented one-based numbers. Default: ``1``.
   * - ``polarization``
     - Optional
     - ``str | None``
     - Mode-family label: TE/TM/hybrid or propagating/evanescent as appropriate to the owning result or solver. Default: ``None``.
   * - ``eigenvalue``
     - Optional
     - ``complex | None``
     - Dimensionless effective index beta/k0. A guess selects roots near that complex value; None uses the solver estimate. Default: ``None``.
   * - ``power``
     - Optional
     - ``complex | None``
     - Modal or electromagnetic power in W/m of invariant length for 2.5D fields. Complex power retains reactive flux; power ratios are reported separately. Default: ``None``.
   * - ``normalization``
     - Optional
     - ``str``
     - Modal/field normalization convention or flag. Unit-longitudinal-power normalization applies only to modes with usable real power. Default: ``'unnormalized'``.
   * - ``residual``
     - Optional
     - ``float | None``
     - Stored numerical-validation diagnostic. Algebraic and Gauss residuals measure discrete equation defects; pml_fraction measures energy in absorbing layers. Default: ``None``.
   * - ``divergence_residual``
     - Optional
     - ``float | None``
     - Stored numerical-validation diagnostic. Algebraic and Gauss residuals measure discrete equation defects; pml_fraction measures energy in absorbing layers. Default: ``None``.
   * - ``metadata``
     - Optional
     - ``Mapping[str, Any]``
     - Structured metadata/diagnostic container associated with the mesh or result. solve_info/metadata retain applied adaptive controls and stopping history. Default: ``fresh default container``.

Returns: ``Mode``.

``FEM_Mode_Solver.Mode.alpha``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dimensional forward attenuation ``-Im(beta)`` in inverse metres.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.Mode.alpha

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

``FEM_Mode_Solver.Mode.attenuation_constant``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Legacy normalized attenuation for ``exp(+j wt - j beta z)``.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.Mode.attenuation_constant

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

``FEM_Mode_Solver.Mode.component``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``component`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.Mode.component(name: 'str') -> 'NumericArray'

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

Returns: ``NumericArray``.

``FEM_Mode_Solver.Mode.propagation_constant``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Legacy normalized phase constant, ``Re(neff)``.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.Mode.propagation_constant

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

``FEM_Mode_Solver.Mode.quantity``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``quantity`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.Mode.quantity(component: 'str', quantity: 'str' = 'real') -> 'NumericArray'

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

Returns: ``NumericArray``.

FEM_Mode_Solver.ModeSet
-----------------------

``FEM_Mode_Solver.ModeSet``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Immutable, sequence-like modes returned by either FEM solver.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSet(modes: 'Sequence[Mode]', *, frequency: 'float', k0: 'float | None' = None, dimension: 'int | None' = None, backend: 'str' = 'fem', metadata: 'Mapping[str, Any] | None' = None) -> 'None'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``modes``
     - Required
     - ``Sequence[Mode]``
     - Mode or mode collection. Integer selectors use the owning API indexing convention; supplied mode sets must match frequency, ky, and transverse geometry.
   * - ``frequency``
     - Required
     - ``float``
     - Ordinary frequency in Hz; must be finite and positive. Keyword-only.
   * - ``k0``
     - Optional
     - ``float | None``
     - Vacuum wavenumber in rad/m, used for coordinate scaling and material/PML evaluation. Keyword-only. Default: ``None``.
   * - ``dimension``
     - Optional
     - ``int | None``
     - Spatial dimension of the mesh/geometry. Electrostatics supports 1 or 2; periodic vector fields support 3. Keyword-only. Default: ``None``.
   * - ``backend``
     - Optional
     - ``str``
     - Eigensolver backend selector. Accepted choices are given by the signature Literal or the method behavior notes; auto selects by system size. Keyword-only. Default: ``'fem'``.
   * - ``metadata``
     - Optional
     - ``Mapping[str, Any] | None``
     - Structured metadata/diagnostic container associated with the mesh or result. solve_info/metadata retain applied adaptive controls and stopping history. Keyword-only. Default: ``None``.

Returns: ``ModeSet``.

``FEM_Mode_Solver.ModeSet.__getitem__``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Select an item using Python square-bracket indexing; integer indices are zero based.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSet.__getitem__(index: 'int | slice') -> 'Mode | tuple[Mode, ...]'

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

Returns: ``Mode | tuple[Mode, ...]``.

``FEM_Mode_Solver.ModeSet.__iter__``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Iterate over stored items in their existing order.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSet.__iter__() -> 'Iterator[Mode]'

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

Returns: ``Iterator[Mode]``.

``FEM_Mode_Solver.ModeSet.__len__``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Return the number of stored items through Python len().

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSet.__len__() -> 'int'

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

``FEM_Mode_Solver.ModeSet.attenuation_constant``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's attenuation constant value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSet.attenuation_constant

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

``FEM_Mode_Solver.ModeSet.beta``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's beta value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSet.beta

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

``FEM_Mode_Solver.ModeSet.by_polarization``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``by_polarization`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSet.by_polarization(polarization: 'str') -> "'ModeSet'"

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

Returns: ``ModeSet``.

``FEM_Mode_Solver.ModeSet.components``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's components value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSet.components

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

``FEM_Mode_Solver.ModeSet.count``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

S.count(value) -> integer -- return number of occurrences of value

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSet.count(value)

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

``FEM_Mode_Solver.ModeSet.index``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

S.index(value, [start, [stop]]) -> integer -- return first index of value. Raises ValueError if the value is not present.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSet.index(value, start=0, stop=None)

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

``FEM_Mode_Solver.ModeSet.mode``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Return a mode by user-facing one-based number.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSet.mode(number: 'int') -> 'Mode'

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

Returns: ``Mode``.

``FEM_Mode_Solver.ModeSet.neff``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's neff value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSet.neff

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

``FEM_Mode_Solver.ModeSet.propagation_constant``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's propagation constant value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSet.propagation_constant

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

FEM_Mode_Solver.ModeSolver1D
----------------------------

``FEM_Mode_Solver.ModeSolver1D``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FEM-native mode solver for an x-stratified cross-section.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSolver1D(frequency: 'float', x_range: 'float | Sequence[float]', num_modes: 'int', neff_guess: 'complex | None' = None, *, background_epsilon: 'MaterialInput' = 1.0, background_mu: 'MaterialInput' = 1.0) -> 'None'

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
   * - ``num_modes``
     - Required
     - ``int``
     - Number of modes or candidate eigenpairs requested; a positive integer. Candidate pools may include roots later rejected by validation.
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

Returns: ``ModeSolver1D``.

Geometry is recorded continuously by the ``add_*`` methods. Call ``discretize`` only after object placement, then call ``solve``.

``FEM_Mode_Solver.ModeSolver1D.add_UPML``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Backward-compatible spelling of ``add_pml``.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSolver1D.add_UPML(pml_width: 'float', n: 'int' = 3, sigma_max: 'float' = 5.0, direction: 'str' = 'all') -> 'PMLSpec'

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
     - Propagation filter forward/backward/all for mode solves; for PML placement, the selected transverse side(s), such as x-, x+, or x. Default: ``'all'``.

Returns: ``PMLSpec``.

``FEM_Mode_Solver.ModeSolver1D.add_impedance_surface``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Record a scalar impedance object.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSolver1D.add_impedance_surface(Zs: 'complex | None' = None, *, preset: 'str | None' = None, x_range: 'Sequence[float]', name: 'str | None' = None) -> 'BoundaryRegion'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``Zs``
     - Optional
     - ``complex | None``
     - Surface impedance in ohms. Supply an explicit passive complex value or a supported metal through the alternative material input. Default: ``None``.
   * - ``preset``
     - Optional
     - ``str | None``
     - Visualization component preset or mapping of named complex field-component arrays. Keyword-only. Default: ``None``.
   * - ``x_range``
     - Required
     - ``Sequence[float]``
     - Physical axis bounds in metres, with upper > lower. Range-based solver constructors also accept a positive extent for (0, extent). Keyword-only.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Returns: ``BoundaryRegion``.

Geometry and meshing support the object now; assembly intentionally raises a capability error until the orientation-dependent 1D Robin form is validated for both polarizations.

``FEM_Mode_Solver.ModeSolver1D.add_layer``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Place a material interval and return its geometry handle.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSolver1D.add_layer(epsilon: 'MaterialInput', mu: 'MaterialInput', x_range: 'Sequence[float]', *, name: 'str | None' = None) -> 'Region'

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
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Returns: ``Region``.

Later regions take precedence in overlap areas. Material interfaces are inserted into the mesh exactly during ``discretize``.

``FEM_Mode_Solver.ModeSolver1D.add_pec``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add an opaque PEC interval, or set both outer walls to PEC.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSolver1D.add_pec(x_range: 'Sequence[float] | None' = None, components: 'object | None' = None, *, name: 'str | None' = None) -> 'BoundaryRegion | None'

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
   * - ``components``
     - Optional
     - ``object | None``
     - Field component selection, using Cartesian electric/magnetic names such as Ex or Hy. The visualization API also supports its documented aggregate quantities. Default: ``None``.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Returns: ``BoundaryRegion | None``.

``FEM_Mode_Solver.ModeSolver1D.add_pmc``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add an opaque PMC interval, or set both outer walls to PMC.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSolver1D.add_pmc(x_range: 'Sequence[float] | None' = None, components: 'object | None' = None, *, name: 'str | None' = None) -> 'BoundaryRegion | None'

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
   * - ``components``
     - Optional
     - ``object | None``
     - Field component selection, using Cartesian electric/magnetic names such as Ex or Hy. The visualization API also supports its documented aggregate quantities. Default: ``None``.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Returns: ``BoundaryRegion | None``.

``FEM_Mode_Solver.ModeSolver1D.add_pml``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Place a physical transformation-optics PML at selected x ends.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSolver1D.add_pml(pml_width: 'float', n: 'int' = 3, sigma_max: 'float' = 5.0, direction: 'str' = 'all') -> 'PMLSpec'

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
     - Propagation filter forward/backward/all for mode solves; for PML placement, the selected transverse side(s), such as x-, x+, or x. Default: ``'all'``.

Returns: ``PMLSpec``.

``FEM_Mode_Solver.ModeSolver1D.discretize``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create an interface-conforming first-order line mesh.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSolver1D.discretize(max_element_size: 'float | None' = None, *, resolution: 'int | None' = None, wavelength_elements: 'int' = 4, material_aware: 'bool' = True, element_order: 'int' = 1, quadrature_order: 'int' = 4) -> 'FEMMesh1D'

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
     - Physical element-edge length in metres. A maximum target is an upper sizing request; material, boundary, and wavelength constraints may produce smaller cells. Default: ``None``.
   * - ``resolution``
     - Optional
     - ``int | None``
     - Initial mesh-resolution request. 1D uses an interval count; 2D accepts axis counts. Physical maximum size and wavelength constraints may increase the generated count. Keyword-only. Default: ``None``.
   * - ``wavelength_elements``
     - Optional
     - ``int``
     - Minimum requested elements per shortest local material wavelength; integer at least four. Public adaptive workflows default to four. Keyword-only. Default: ``4``.
   * - ``material_aware``
     - Optional
     - ``bool``
     - Enable element-size reduction in high-index/high-permittivity material regions while retaining the global maximum-edge cap. Keyword-only. Default: ``True``.
   * - ``element_order``
     - Optional
     - ``int``
     - Finite-element polynomial-order selection. Standalone 2D modes and WaveFEM scattering accept 1 (N1/P1) or 2 (N2/P2); other backends retain their fixed compatible spaces. Keyword-only. Default: ``1``.
   * - ``quadrature_order``
     - Optional
     - ``int``
     - Finite-element integration order. Higher-order mixed elements require at least fourth-order quadrature; PML and material variation can require more. Keyword-only. Default: ``4``.

Returns: ``FEMMesh1D``.

By default, element density follows the local material wavenumber: intervals with a larger conservative material-index estimate receive smaller elements. ``wavelength_elements`` additionally sets the minimum number of elements per shortest local material wavelength. Set ``material_aware=False`` to use a spatially uniform target size.

``FEM_Mode_Solver.ModeSolver1D.discretized``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Whether a current, geometry-matching mesh is available.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSolver1D.discretized

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

``FEM_Mode_Solver.ModeSolver1D.mesh_data``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Return the common FEM mesh wrapper used by ``discretize()``.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSolver1D.mesh_data

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

Returns: ``FEMMesh1D``.

``FEM_Mode_Solver.ModeSolver1D.native_mesh``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Return the underlying scikit-fem line mesh.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSolver1D.native_mesh

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

Returns: ``Python object described by this operation``.

``FEM_Mode_Solver.ModeSolver1D.refine``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Remesh the current geometry with ``factor`` times the density.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSolver1D.refine(factor: 'float' = 2.0) -> 'FEMMesh1D'

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

Returns: ``FEMMesh1D``.

Refinement scales every active size control, preserves material-aware grading and exact interfaces, and invalidates any previously solved modes. Repeated calls refine relative to the most recent mesh.

``FEM_Mode_Solver.ModeSolver1D.remove``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Remove a previously returned geometry handle.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSolver1D.remove(handle: 'Region | BoundaryRegion | PMLSpec') -> 'None'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``handle``
     - Required
     - ``Region | BoundaryRegion | PMLSpec``
     - Previously returned region/boundary/PML handle to remove, or an index/key for a container operation as indicated by the method.

Returns: ``None``.

``FEM_Mode_Solver.ModeSolver1D.result``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Return the latest solved modes, or raise before the first solve.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSolver1D.result

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

Returns: ``ModeSet``.

``FEM_Mode_Solver.ModeSolver1D.set_outer_boundary``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set both transverse truncation walls to ``'pec'`` or ``'pmc'``.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSolver1D.set_outer_boundary(kind: 'str') -> 'None'

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

``FEM_Mode_Solver.ModeSolver1D.solve``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Solve from a coarse mesh and refine until the discretization residual meets adaptive_tolerance or max_refinements mesh updates are used. Pass max_refinements=0 for one solve on the initial mesh. Remaining options are passed to the algebraic mode solve.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSolver1D.solve(neff_guess: 'complex | None' = None, num_modes: 'int | None' = None, *, tol: 'float' = 1e-10, residual_tolerance: 'float' = 1e-07, dense_limit: 'int' = 450, max_refinements: 'int' = 2, adaptive_tolerance: 'float' = 0.05) -> 'ModeSet'

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
   * - ``tol``
     - Optional
     - ``float``
     - Positive numerical tolerance. Linear/QEP residual tolerances validate the algebraic solve; they do not set the adaptive mesh threshold. Keyword-only. Default: ``1e-10``.
   * - ``residual_tolerance``
     - Optional
     - ``float``
     - Positive numerical tolerance. Linear/QEP residual tolerances validate the algebraic solve; they do not set the adaptive mesh threshold. Keyword-only. Default: ``1e-07``.
   * - ``dense_limit``
     - Optional
     - ``int``
     - Matrix-size cutoff for dense eigensolving; larger systems use a sparse backend. This is a dimension limit, not a mesh-error threshold. Keyword-only. Default: ``450``.
   * - ``max_refinements``
     - Optional
     - ``int``
     - Maximum mesh updates after the initial solve; nonnegative integer. Zero keeps the initial mesh. Solver default is 2; examples explicitly use 0. Keyword-only. Default: ``2``.
   * - ``adaptive_tolerance``
     - Optional
     - ``float``
     - Positive finite threshold for the normalized discretization estimator. Stop when residual <= threshold. Independent of algebraic tolerances; exhausting the budget is not convergence. Keyword-only. Default: ``0.05``.

Returns: ``ModeSet``.

``FEM_Mode_Solver.ModeSolver1D.visualize``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plot one solved mode using the shared visualization implementation.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSolver1D.visualize(mode: 'int' = 1, *, component: 'str | None' = None, components: 'Sequence[str] | str | None' = None, quantity: 'str' = 'real', mesh: 'bool' = False, mesh_overlay: 'bool | None' = None, material: 'bool | None' = None, field: 'bool' = True, normalize: 'bool' = False, cmap: 'str | None' = None, axes: 'Any | None' = None, title: 'str | None' = None, show: 'bool' = True, legacy_component_flags: 'Any') -> 'object'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``mode``
     - Optional
     - ``int``
     - Mode or mode collection. Integer selectors use the owning API indexing convention; supplied mode sets must match frequency, ky, and transverse geometry. Default: ``1``.
   * - ``component``
     - Optional
     - ``str | None``
     - Field component selection, using Cartesian electric/magnetic names such as Ex or Hy. The visualization API also supports its documented aggregate quantities. Keyword-only. Default: ``None``.
   * - ``components``
     - Optional
     - ``Sequence[str] | str | None``
     - Field component selection, using Cartesian electric/magnetic names such as Ex or Hy. The visualization API also supports its documented aggregate quantities. Keyword-only. Default: ``None``.
   * - ``quantity``
     - Optional
     - ``str``
     - Complex-data display selection, such as real, imag, abs, or phase; accepted values are given by the owning plotting API. Keyword-only. Default: ``'real'``.
   * - ``mesh``
     - Optional
     - ``bool``
     - Conforming FEM mesh object. Physical meshes carry SI coordinates; computational meshes may be scaled by the reference length for assembly. Keyword-only. Default: ``False``.
   * - ``mesh_overlay``
     - Optional
     - ``bool | None``
     - Enable display of the figure, mesh overlay, or colorbar respectively. show=False returns plotting objects without opening a window. Keyword-only. Default: ``None``.
   * - ``material``
     - Optional
     - ``bool | None``
     - Material object defining relative electric and magnetic response. Exterior fills points outside placed material regions. Keyword-only. Default: ``None``.
   * - ``field``
     - Optional
     - ``bool``
     - Complex electromagnetic field samples or FE field objects. Cartesian components follow x,y,z order; left/right denote the selected trace side. Keyword-only. Default: ``True``.
   * - ``normalize``
     - Optional
     - ``bool``
     - Modal/field normalization convention or flag. Unit-longitudinal-power normalization applies only to modes with usable real power. Keyword-only. Default: ``False``.
   * - ``cmap``
     - Optional
     - ``str | None``
     - Matplotlib colormap name or object used for scalar field rendering. Keyword-only. Default: ``None``.
   * - ``axes``
     - Optional
     - ``Any | None``
     - Existing Matplotlib axes for embedding a plot; None creates suitable axes when accepted. Keyword-only. Default: ``None``.
   * - ``title``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.
   * - ``show``
     - Optional
     - ``bool``
     - Enable display of the figure, mesh overlay, or colorbar respectively. show=False returns plotting objects without opening a window. Keyword-only. Default: ``True``.
   * - ``legacy_component_flags``
     - Required
     - ``Any``
     - Compatibility-only input from older APIs; it does not add subpixel FEM averaging. Unsupported legacy component selections are rejected. Keyword-only.

Returns: ``Python object described by this operation``.

``FEM_Mode_Solver.ModeSolver1D.visualize_with_gui``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open all solved modes in the interactive GUI.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSolver1D.visualize_with_gui() -> 'object'

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

Returns: ``Python object described by this operation``.

FEM_Mode_Solver.ModeSolver2D
----------------------------

``FEM_Mode_Solver.ModeSolver2D``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Full-vector 2D FEM waveguide mode solver.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSolver2D(frequency: 'float', x_range: 'float | Sequence[float]', y_range: 'float | Sequence[float]', num_modes: 'int' = 4, neff_guess: 'complex | None' = None, *, guess: 'complex | None' = None, background_epsilon: 'MaterialInput' = 1.0, background_mu: 'MaterialInput' = 1.0, boundary: 'str' = 'pec') -> 'None'

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
   * - ``y_range``
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
   * - ``guess``
     - Optional
     - ``complex | None``
     - Dimensionless effective index beta/k0. A guess selects roots near that complex value; None uses the solver estimate. Keyword-only. Default: ``None``.
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

Returns: ``ModeSolver2D``.

Parameters are physical SI values. A scalar ``x_range`` or ``y_range`` denotes ``(0, extent)``; a pair denotes explicit lower and upper bounds. Geometry can be placed in any order before an explicit call to ``discretize``.

``FEM_Mode_Solver.ModeSolver2D.add_UPML``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Backward-compatible spelling of ``add_pml``.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSolver2D.add_UPML(pml_width: 'float', n: 'int' = 3, sigma_max: 'float' = 5.0, direction: 'str' = 'all') -> 'PMLSpec'

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
     - Propagation filter forward/backward/all for mode solves; for PML placement, the selected transverse side(s), such as x-, x+, or x. Default: ``'all'``.

Returns: ``PMLSpec``.

``FEM_Mode_Solver.ModeSolver2D.add_circle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Place a circular or annular material region.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSolver2D.add_circle(epsilon: 'MaterialInput', mu: 'MaterialInput', center: 'Sequence[float]', r1: 'float', r2: 'float | None' = None, *, subpixels: 'int | None' = None, name: 'str | None' = None) -> 'Region'

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
   * - ``r1``
     - Required
     - ``float``
     - Outer circle radius in metres; finite and positive.
   * - ``r2``
     - Optional
     - ``float | None``
     - Optional inner circle radius in metres, creating an annulus; must be smaller than r1. Default: ``None``.
   * - ``subpixels``
     - Optional
     - ``int | None``
     - Compatibility-only input from older APIs; it does not add subpixel FEM averaging. Unsupported legacy component selections are rejected. Keyword-only. Default: ``None``.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Returns: ``Region``.

``FEM_Mode_Solver.ModeSolver2D.add_impedance_surface``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add an opaque conductor whose exposed facets obey scalar SIBC.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSolver2D.add_impedance_surface(Zs: 'complex | None' = None, *, preset: 'str | None' = None, x_range: 'Sequence[float]', y_range: 'Sequence[float]', name: 'str | None' = None) -> 'BoundaryRegion'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``Zs``
     - Optional
     - ``complex | None``
     - Surface impedance in ohms. Supply an explicit passive complex value or a supported metal through the alternative material input. Default: ``None``.
   * - ``preset``
     - Optional
     - ``str | None``
     - Visualization component preset or mapping of named complex field-component arrays. Keyword-only. Default: ``None``.
   * - ``x_range``
     - Required
     - ``Sequence[float]``
     - Physical axis bounds in metres, with upper > lower. Range-based solver constructors also accept a positive extent for (0, extent). Keyword-only.
   * - ``y_range``
     - Required
     - ``Sequence[float]``
     - Physical axis bounds in metres, with upper > lower. Range-based solver constructors also accept a positive extent for (0, extent). Keyword-only.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Returns: ``BoundaryRegion``.

Supply exactly one of ``Zs`` (ohms) or a good-conductor metal ``preset`` such as ``"Cu"``. Presets are evaluated at this solver's frequency using the package's ``exp(+j*omega*t)`` convention.

``FEM_Mode_Solver.ModeSolver2D.add_mesh_refinement``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Place a local mesh-size control without changing the physics.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSolver2D.add_mesh_refinement(shape: 'Shape2D', max_element_size: 'float', *, transition_width: 'float' = 0.0, name: 'str | None' = None) -> 'MeshRefinement'

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
   * - ``transition_width``
     - Optional
     - ``float``
     - Physical distance in metres over which the local mesh-size target transitions back to the surrounding target; None selects the mesher default when permitted. Keyword-only. Default: ``0.0``.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Returns: ``MeshRefinement``.

``shape`` may be a ``Rectangle``, ``Circle``, or ``Polygon``. The shape becomes an exact OCC partition during ``discretize``; it does not create a material or boundary object.

``FEM_Mode_Solver.ModeSolver2D.add_pec``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add an internal PEC object, or select PEC for the outer wall.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSolver2D.add_pec(x_range: 'Sequence[float] | None' = None, y_range: 'Sequence[float] | None' = None, components: 'Sequence[str] | str | None' = None, *, name: 'str | None' = None) -> 'BoundaryRegion | None'

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
   * - ``y_range``
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

``FEM_Mode_Solver.ModeSolver2D.add_pmc``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add an internal PMC object, or select PMC for the outer wall.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSolver2D.add_pmc(x_range: 'Sequence[float] | None' = None, y_range: 'Sequence[float] | None' = None, components: 'Sequence[str] | str | None' = None, *, name: 'str | None' = None) -> 'BoundaryRegion | None'

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
   * - ``y_range``
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

``FEM_Mode_Solver.ModeSolver2D.add_pml``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add a physical-width uniaxial PML to selected exterior side(s).

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSolver2D.add_pml(pml_width: 'float', n: 'int' = 3, sigma_max: 'float' = 5.0, direction: 'str' = 'all') -> 'PMLSpec'

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
     - Propagation filter forward/backward/all for mode solves; for PML placement, the selected transverse side(s), such as x-, x+, or x. Default: ``'all'``.

Returns: ``PMLSpec``.

``FEM_Mode_Solver.ModeSolver2D.add_polygon``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Place an arbitrary simple polygon material region.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSolver2D.add_polygon(epsilon: 'MaterialInput', mu: 'MaterialInput', points: 'Sequence[Sequence[float]]', *, name: 'str | None' = None) -> 'Region'

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

``FEM_Mode_Solver.ModeSolver2D.add_rectangle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Place a conformingly meshed rectangular material region.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSolver2D.add_rectangle(epsilon: 'MaterialInput', mu: 'MaterialInput', x_range: 'Sequence[float]', y_range: 'Sequence[float]', *, subpixels: 'int | None' = None, name: 'str | None' = None) -> 'Region'

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
   * - ``y_range``
     - Required
     - ``Sequence[float]``
     - Physical axis bounds in metres, with upper > lower. Range-based solver constructors also accept a positive extent for (0, extent).
   * - ``subpixels``
     - Optional
     - ``int | None``
     - Compatibility-only input from older APIs; it does not add subpixel FEM averaging. Unsupported legacy component selections are rejected. Keyword-only. Default: ``None``.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Returns: ``Region``.

``FEM_Mode_Solver.ModeSolver2D.add_triangle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Place a triangular material region.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSolver2D.add_triangle(epsilon: 'MaterialInput', mu: 'MaterialInput', p1: 'Sequence[float]', p2: 'Sequence[float]', p3: 'Sequence[float]', *, subpixels: 'int | None' = None, name: 'str | None' = None) -> 'Region'

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
   * - ``subpixels``
     - Optional
     - ``int | None``
     - Compatibility-only input from older APIs; it does not add subpixel FEM averaging. Unsupported legacy component selections are rejected. Keyword-only. Default: ``None``.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Returns: ``Region``.

``FEM_Mode_Solver.ModeSolver2D.discretize``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mesh the scene and assemble its analytic quadratic pencil.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSolver2D.discretize(*, max_element_size: 'float | None' = None, resolution: 'tuple[int, int] | None' = None, wavelength_elements: 'int' = 4, material_aware: 'bool' = True, interface_refinement: 'float | None' = 0.7, interface_refinement_width: 'float | None' = None, boundary_refinement: 'float | None' = 0.5, boundary_refinement_width: 'float | None' = None, element_order: 'int' = 1, quadrature_order: 'int' = 4) -> 'FEMMesh2D'

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
   * - ``material_aware``
     - Optional
     - ``bool``
     - Enable element-size reduction in high-index/high-permittivity material regions while retaining the global maximum-edge cap. Keyword-only. Default: ``True``.
   * - ``interface_refinement``
     - Optional
     - ``float | None``
     - Optional size multiplier in (0, 1] near conductor/exterior boundaries or material interfaces. None disables this sizing field. Keyword-only. Default: ``0.7``.
   * - ``interface_refinement_width``
     - Optional
     - ``float | None``
     - Physical distance in metres over which the local mesh-size target transitions back to the surrounding target; None selects the mesher default when permitted. Keyword-only. Default: ``None``.
   * - ``boundary_refinement``
     - Optional
     - ``float | None``
     - Optional size multiplier in (0, 1] near conductor/exterior boundaries or material interfaces. None disables this sizing field. Keyword-only. Default: ``0.5``.
   * - ``boundary_refinement_width``
     - Optional
     - ``float | None``
     - Physical distance in metres over which the local mesh-size target transitions back to the surrounding target; None selects the mesher default when permitted. Keyword-only. Default: ``None``.
   * - ``element_order``
     - Optional
     - ``int``
     - Finite-element polynomial-order selection. Standalone 2D modes and WaveFEM scattering accept 1 (N1/P1) or 2 (N2/P2); other backends retain their fixed compatible spaces. Keyword-only. Default: ``1``.
   * - ``quadrature_order``
     - Optional
     - ``int``
     - Finite-element integration order. Higher-order mixed elements require at least fourth-order quadrature; PML and material variation can require more. Keyword-only. Default: ``4``.

Returns: ``FEMMesh2D``.

Material-aware sizing is enabled by default: higher local propagation wavenumber produces smaller elements while ``max_element_size`` stays a global characteristic-size target. ``interface_refinement`` and ``boundary_refinement`` are optional size multipliers in ``(0, 1]``; ``None`` disables the corresponding distance field. PEC, PMC, and impedance walls all participate in boundary sizing.

``FEM_Mode_Solver.ModeSolver2D.discretized``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's discretized value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSolver2D.discretized

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

``FEM_Mode_Solver.ModeSolver2D.mesh``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Return the common FEM mesh wrapper used by ``discretize()``.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSolver2D.mesh

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

Returns: ``FEMMesh2D``.

``FEM_Mode_Solver.ModeSolver2D.mesh_data``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's mesh data value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSolver2D.mesh_data

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

Returns: ``FEMMesh2D``.

``FEM_Mode_Solver.ModeSolver2D.native_mesh``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Return the underlying scikit-fem triangular mesh.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSolver2D.native_mesh

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

``FEM_Mode_Solver.ModeSolver2D.refine``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Remesh with ``factor`` times the density and rebuild the system.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSolver2D.refine(factor: 'float' = 2.0) -> 'FEMMesh2D'

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

Returns: ``FEMMesh2D``.

All previous discretization options are retained. Global, wavelength, interface, boundary, and explicit local size targets are scaled consistently; any previous modal solution is invalidated.

``FEM_Mode_Solver.ModeSolver2D.remove``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Remove a placed object and invalidate any existing mesh.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSolver2D.remove(handle: 'Region | BoundaryRegion | MeshRefinement | PMLSpec') -> 'None'

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

``FEM_Mode_Solver.ModeSolver2D.result``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's result value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSolver2D.result

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

Returns: ``ModeSet``.

``FEM_Mode_Solver.ModeSolver2D.set_outer_boundary``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set outer boundary; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSolver2D.set_outer_boundary(kind: 'str') -> 'None'

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

``FEM_Mode_Solver.ModeSolver2D.solution``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ModeSet hook consumed by the backend-neutral visualizers.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSolver2D.solution

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

Returns: ``ModeSet | None``.

``FEM_Mode_Solver.ModeSolver2D.solve``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Solve from a coarse mesh and refine until the discretization residual meets adaptive_tolerance or max_refinements mesh updates are used. Pass max_refinements=0 for one solve on the initial mesh. Remaining options are passed to the algebraic mode solve.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSolver2D.solve(neff_guess: 'complex | None' = None, num_modes: 'int | None' = None, *, direction: 'Direction' = 'forward', eigensolver_tolerance: 'float' = 1e-10, residual_tolerance: 'float' = 1e-08, divergence_tolerance: 'float' = 1e-07, propagation_ratio_tolerance: 'float' = 0.001, dense_linearization_limit: 'int' = 700, max_refinements: 'int' = 2, adaptive_tolerance: 'float' = 0.05) -> 'ModeSet'

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
     - Maximum accepted weak Gauss-law defect. Periodic 3D uses a squared normalized defect; other modal backends use their documented divergence residual. Keyword-only. Default: ``1e-07``.
   * - ``propagation_ratio_tolerance``
     - Optional
     - ``float``
     - Positive relative real/imaginary propagation criterion used to classify propagating and evanescent roots. Keyword-only. Default: ``0.001``.
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

Returns: ``ModeSet``.

``FEM_Mode_Solver.ModeSolver2D.system``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's system value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSolver2D.system

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

Returns: ``ModeFEMSystem2D``.

``FEM_Mode_Solver.ModeSolver2D.visualize``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plot sampled FEM fields using the common static visualizer.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSolver2D.visualize(mode: 'int | Mode' = 1, *, component: 'str | None' = None, components: 'Sequence[str] | str | None' = None, quantity: 'str' = 'real', mesh: 'bool' = False, mesh_overlay: 'bool | None' = None, material: 'bool | None' = None, field: 'bool' = True, normalize: 'bool' = False, cmap: 'str | None' = None, axes: 'Any | None' = None, title: 'str | None' = None, show: 'bool' = True, legacy_component_flags: 'Any') -> 'Any'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``mode``
     - Optional
     - ``int | Mode``
     - Mode or mode collection. Integer selectors use the owning API indexing convention; supplied mode sets must match frequency, ky, and transverse geometry. Default: ``1``.
   * - ``component``
     - Optional
     - ``str | None``
     - Field component selection, using Cartesian electric/magnetic names such as Ex or Hy. The visualization API also supports its documented aggregate quantities. Keyword-only. Default: ``None``.
   * - ``components``
     - Optional
     - ``Sequence[str] | str | None``
     - Field component selection, using Cartesian electric/magnetic names such as Ex or Hy. The visualization API also supports its documented aggregate quantities. Keyword-only. Default: ``None``.
   * - ``quantity``
     - Optional
     - ``str``
     - Complex-data display selection, such as real, imag, abs, or phase; accepted values are given by the owning plotting API. Keyword-only. Default: ``'real'``.
   * - ``mesh``
     - Optional
     - ``bool``
     - Conforming FEM mesh object. Physical meshes carry SI coordinates; computational meshes may be scaled by the reference length for assembly. Keyword-only. Default: ``False``.
   * - ``mesh_overlay``
     - Optional
     - ``bool | None``
     - Enable display of the figure, mesh overlay, or colorbar respectively. show=False returns plotting objects without opening a window. Keyword-only. Default: ``None``.
   * - ``material``
     - Optional
     - ``bool | None``
     - Material object defining relative electric and magnetic response. Exterior fills points outside placed material regions. Keyword-only. Default: ``None``.
   * - ``field``
     - Optional
     - ``bool``
     - Complex electromagnetic field samples or FE field objects. Cartesian components follow x,y,z order; left/right denote the selected trace side. Keyword-only. Default: ``True``.
   * - ``normalize``
     - Optional
     - ``bool``
     - Modal/field normalization convention or flag. Unit-longitudinal-power normalization applies only to modes with usable real power. Keyword-only. Default: ``False``.
   * - ``cmap``
     - Optional
     - ``str | None``
     - Matplotlib colormap name or object used for scalar field rendering. Keyword-only. Default: ``None``.
   * - ``axes``
     - Optional
     - ``Any | None``
     - Existing Matplotlib axes for embedding a plot; None creates suitable axes when accepted. Keyword-only. Default: ``None``.
   * - ``title``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.
   * - ``show``
     - Optional
     - ``bool``
     - Enable display of the figure, mesh overlay, or colorbar respectively. show=False returns plotting objects without opening a window. Keyword-only. Default: ``True``.
   * - ``legacy_component_flags``
     - Required
     - ``Any``
     - Compatibility-only input from older APIs; it does not add subpixel FEM averaging. Unsupported legacy component selections are rejected. Keyword-only.

Returns: ``Any``.

``FEM_Mode_Solver.ModeSolver2D.visualize_with_gui``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open all solved modes in the interactive field viewer.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeSolver2D.visualize_with_gui() -> 'Any'

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

FEM_Mode_Solver.ModeViewer
--------------------------

``FEM_Mode_Solver.ModeViewer``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Interactive Matplotlib controller returned by ``visualize_with_gui``.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeViewer(source: 'Mode | ModeSet | SupportsModeVisualization | Any', *, mode: 'int' = 1, component: 'str | None' = None, quantity: 'str' = 'real', mesh: 'bool' = False, material: 'bool | None' = None, field: 'bool' = True, normalize: 'bool' = False, cmap: 'str | None' = None) -> 'None'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``source``
     - Required
     - ``Mode | ModeSet | SupportsModeVisualization | Any``
     - Equivalent-current source, mode-set/result object, or plotting data source accepted by the owning function; see its concrete type annotation.
   * - ``mode``
     - Optional
     - ``int``
     - Mode or mode collection. Integer selectors use the owning API indexing convention; supplied mode sets must match frequency, ky, and transverse geometry. Keyword-only. Default: ``1``.
   * - ``component``
     - Optional
     - ``str | None``
     - Field component selection, using Cartesian electric/magnetic names such as Ex or Hy. The visualization API also supports its documented aggregate quantities. Keyword-only. Default: ``None``.
   * - ``quantity``
     - Optional
     - ``str``
     - Complex-data display selection, such as real, imag, abs, or phase; accepted values are given by the owning plotting API. Keyword-only. Default: ``'real'``.
   * - ``mesh``
     - Optional
     - ``bool``
     - Conforming FEM mesh object. Physical meshes carry SI coordinates; computational meshes may be scaled by the reference length for assembly. Keyword-only. Default: ``False``.
   * - ``material``
     - Optional
     - ``bool | None``
     - Material object defining relative electric and magnetic response. Exterior fills points outside placed material regions. Keyword-only. Default: ``None``.
   * - ``field``
     - Optional
     - ``bool``
     - Complex electromagnetic field samples or FE field objects. Cartesian components follow x,y,z order; left/right denote the selected trace side. Keyword-only. Default: ``True``.
   * - ``normalize``
     - Optional
     - ``bool``
     - Modal/field normalization convention or flag. Unit-longitudinal-power normalization applies only to modes with usable real power. Keyword-only. Default: ``False``.
   * - ``cmap``
     - Optional
     - ``str | None``
     - Matplotlib colormap name or object used for scalar field rendering. Keyword-only. Default: ``None``.

Returns: ``ModeViewer``.

``FEM_Mode_Solver.ModeViewer.close``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``close`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeViewer.close() -> 'None'

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

``FEM_Mode_Solver.ModeViewer.mode``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's mode value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeViewer.mode

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

Returns: ``Mode``.

``FEM_Mode_Solver.ModeViewer.show``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``show`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.ModeViewer.show(*, block: 'bool | None' = None) -> 'None'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``block``
     - Optional
     - ``bool | None``
     - HDF5 case selector or mode/block slice bounds, using zero-based indexing and a stop-exclusive interval. Keyword-only. Default: ``None``.

Returns: ``None``.

FEM_Mode_Solver.NotDiscretizedError
-----------------------------------

``FEM_Mode_Solver.NotDiscretizedError``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A solve or plot was requested before discretization.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.NotDiscretizedError(*args: 'object')

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

FEM_Mode_Solver.PMLSpec
-----------------------

``FEM_Mode_Solver.PMLSpec``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Construct a ``PMLSpec`` record. Its public data fields use the same names and types as the constructor inputs below. Solvers normally construct mesh/system/result records for you.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.PMLSpec(thickness: 'float', order: 'int' = 3, sigma_max: 'float' = 5.0, direction: 'str' = 'all') -> None

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
     - Propagation filter forward/backward/all for mode solves; for PML placement, the selected transverse side(s), such as x-, x+, or x. Default: ``'all'``.

Returns: ``PMLSpec``.

``FEM_Mode_Solver.PMLSpec.stretch``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``stretch`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.PMLSpec.stretch(depth: 'ArrayLike') -> 'NDArray[np.complex128]'

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

FEM_Mode_Solver.Polygon
-----------------------

``FEM_Mode_Solver.Polygon``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define a closed polygon from ordered physical vertices. Use this immutable primitive in geometry/material placement APIs.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.Polygon(points: 'tuple[tuple[float, float], ...]') -> None

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

``FEM_Mode_Solver.Polygon.bounds``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's bounds value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.Polygon.bounds

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

``FEM_Mode_Solver.Polygon.contains``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``contains`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.Polygon.contains(x: 'ArrayLike', y: 'ArrayLike') -> 'NDArray[np.bool_]'

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

FEM_Mode_Solver.Rectangle
-------------------------

``FEM_Mode_Solver.Rectangle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define an axis-aligned physical rectangle. Use this immutable primitive in geometry/material placement APIs.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.Rectangle(x: 'tuple[float, float]', y: 'tuple[float, float]') -> None

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

``FEM_Mode_Solver.Rectangle.bounds``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's bounds value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.Rectangle.bounds

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

``FEM_Mode_Solver.Rectangle.contains``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``contains`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.Rectangle.contains(x: 'ArrayLike', y: 'ArrayLike') -> 'NDArray[np.bool_]'

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

FEM_Mode_Solver.Region
----------------------

``FEM_Mode_Solver.Region``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Construct a ``Region`` record. Its public data fields use the same names and types as the constructor inputs below. Solvers normally construct mesh/system/result records for you.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.Region(id: 'int', name: 'str', shape: 'Interval | Shape2D', material: 'Material') -> None

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
     - ``Interval | Shape2D``
     - Geometry primitive or region selector identifying the physical support. Use the class/union in the signature; electrostatics also accepts named exterior boundaries.
   * - ``material``
     - Required
     - ``Material``
     - Material object defining relative electric and magnetic response. Exterior fills points outside placed material regions.

Returns: ``Region``.

FEM_Mode_Solver.SampledFields
-----------------------------

``FEM_Mode_Solver.SampledFields``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Immutable samples of one modal field on common 1D or 2D coordinates.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.SampledFields(coordinates: 'ArrayLike | Sequence[ArrayLike]', values: 'Mapping[str, ArrayLike] | ArrayLike', *, dimension: 'int | None' = None, mesh_points: 'ArrayLike | None' = None, mesh_cells: 'ArrayLike | None' = None, material: 'ArrayLike | None' = None, metadata: 'Mapping[str, Any] | None' = None) -> 'None'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``coordinates``
     - Required
     - ``ArrayLike | Sequence[ArrayLike]``
     - Coordinates used by the object or evaluation operation. Mesh geometry uses physical metres; low-level FE operators use their basis coordinate scale. See the array-shape conventions.
   * - ``values``
     - Required
     - ``Mapping[str, ArrayLike] | ArrayLike``
     - Scalar or array to validate, transform, interpolate, or store. It must satisfy the owning operation and the expected type in this table.
   * - ``dimension``
     - Optional
     - ``int | None``
     - Spatial dimension of the mesh/geometry. Electrostatics supports 1 or 2; periodic vector fields support 3. Keyword-only. Default: ``None``.
   * - ``mesh_points``
     - Optional
     - ``ArrayLike | None``
     - Mesh-node coordinates. Native scikit-fem arrays are dimension by node; standalone result coordinates are commonly node by dimension. Keyword-only. Default: ``None``.
   * - ``mesh_cells``
     - Optional
     - ``ArrayLike | None``
     - Integer simplex connectivity. scikit-fem uses vertices-per-cell by cell; standalone exported geometry commonly uses cell by vertices-per-cell. Keyword-only. Default: ``None``.
   * - ``material``
     - Optional
     - ``ArrayLike | None``
     - Material object defining relative electric and magnetic response. Exterior fills points outside placed material regions. Keyword-only. Default: ``None``.
   * - ``metadata``
     - Optional
     - ``Mapping[str, Any] | None``
     - Structured metadata/diagnostic container associated with the mesh or result. solve_info/metadata retain applied adaptive controls and stopping history. Keyword-only. Default: ``None``.

Returns: ``SampledFields``.

``coordinates`` may be an ``(N, 1)``/``(N, 2)`` point array or a tuple of coordinate axes. ``values`` may be a component mapping, or one array whose final axis follows ``metadata['component_order']``. All public arrays are defensive copies with writes disabled.

``FEM_Mode_Solver.SampledFields.component``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Return a sampled component using case-insensitive Maxwell names.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.SampledFields.component(name: 'str') -> 'NumericArray'

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

Returns: ``NumericArray``.

``FEM_Mode_Solver.SampledFields.components``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's components value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.SampledFields.components

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

``FEM_Mode_Solver.SampledFields.quantity``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Return a real, imaginary, magnitude, or phase view for plotting.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.SampledFields.quantity(component: 'str', quantity: 'str' = 'real') -> 'NumericArray'

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

Returns: ``NumericArray``.

``FEM_Mode_Solver.SampledFields.vector_magnitude``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Return ``|E|`` or ``|H|`` from all available Cartesian components.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.SampledFields.vector_magnitude(field: 'str') -> 'FloatArray'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``field``
     - Required
     - ``str``
     - Complex electromagnetic field samples or FE field objects. Cartesian components follow x,y,z order; left/right denote the selected trace side.

Returns: ``FloatArray``.

``FEM_Mode_Solver.SampledFields.x``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's x value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.SampledFields.x

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

``FEM_Mode_Solver.SampledFields.y``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's y value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.SampledFields.y

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

FEM_Mode_Solver.SolverError
---------------------------

``FEM_Mode_Solver.SolverError``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The polynomial eigenproblem could not produce valid modes.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.SolverError(*args: 'object')

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

FEM_Mode_Solver.StaleDiscretizationError
----------------------------------------

``FEM_Mode_Solver.StaleDiscretizationError``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The continuous model changed after it was discretized.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.StaleDiscretizationError(*args: 'object')

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

FEM_Mode_Solver.canonical_metal_name
------------------------------------

``FEM_Mode_Solver.canonical_metal_name``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``canonical_metal_name`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.canonical_metal_name(value: 'str') -> 'str'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``value``
     - Required
     - ``str``
     - Scalar or array to validate, transform, interpolate, or store. It must satisfy the owning operation and the expected type in this table.

Returns: ``str``.

FEM_Mode_Solver.good_conductor_surface_impedance
------------------------------------------------

``FEM_Mode_Solver.good_conductor_surface_impedance``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Return ``(1+i)*sqrt(pi*f*mu0*mu_r*rho)`` for ``exp(+iwt)``.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.good_conductor_surface_impedance(metal: 'str', frequency: 'float', *, relative_permeability: 'float' = 1.0) -> 'complex'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``metal``
     - Required
     - ``str``
     - Supported metal name used to look up bulk resistivity for the good-conductor surface impedance.
   * - ``frequency``
     - Required
     - ``float``
     - Ordinary frequency in Hz; must be finite and positive.
   * - ``relative_permeability``
     - Optional
     - ``float``
     - Relative permeability, scalar or Cartesian diagonal where supported. WaveFEM scattering requires equal actual and background permeability. Keyword-only. Default: ``1.0``.

Returns: ``complex``.

FEM_Mode_Solver.validate_surface_impedance
------------------------------------------

``FEM_Mode_Solver.validate_surface_impedance``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``validate_surface_impedance`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.validate_surface_impedance(value: 'complex | float') -> 'complex'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``value``
     - Required
     - ``complex | float``
     - Scalar or array to validate, transform, interpolate, or store. It must satisfy the owning operation and the expected type in this table.

Returns: ``complex``.

FEM_Mode_Solver.visualize
-------------------------

``FEM_Mode_Solver.visualize``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plot one mode using common component and quantity controls.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.visualize(source: 'Mode | ModeSet | SupportsModeVisualization | Any', mode: 'int | Mode' = 1, component: 'str | None' = None, *, components: 'Sequence[str] | str | None' = None, quantity: 'str' = 'real', mesh: 'bool' = False, mesh_overlay: 'bool | None' = None, material: 'bool | None' = None, field: 'bool' = True, normalize: 'bool' = False, cmap: 'str | None' = None, axes: 'Any | None' = None, title: 'str | None' = None, show: 'bool' = True, **legacy_component_flags: 'Any') -> 'tuple[Any, NDArray[Any]]'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``source``
     - Required
     - ``Mode | ModeSet | SupportsModeVisualization | Any``
     - Equivalent-current source, mode-set/result object, or plotting data source accepted by the owning function; see its concrete type annotation.
   * - ``mode``
     - Optional
     - ``int | Mode``
     - Mode or mode collection. Integer selectors use the owning API indexing convention; supplied mode sets must match frequency, ky, and transverse geometry. Default: ``1``.
   * - ``component``
     - Optional
     - ``str | None``
     - Field component selection, using Cartesian electric/magnetic names such as Ex or Hy. The visualization API also supports its documented aggregate quantities. Default: ``None``.
   * - ``components``
     - Optional
     - ``Sequence[str] | str | None``
     - Field component selection, using Cartesian electric/magnetic names such as Ex or Hy. The visualization API also supports its documented aggregate quantities. Keyword-only. Default: ``None``.
   * - ``quantity``
     - Optional
     - ``str``
     - Complex-data display selection, such as real, imag, abs, or phase; accepted values are given by the owning plotting API. Keyword-only. Default: ``'real'``.
   * - ``mesh``
     - Optional
     - ``bool``
     - Conforming FEM mesh object. Physical meshes carry SI coordinates; computational meshes may be scaled by the reference length for assembly. Keyword-only. Default: ``False``.
   * - ``mesh_overlay``
     - Optional
     - ``bool | None``
     - Enable display of the figure, mesh overlay, or colorbar respectively. show=False returns plotting objects without opening a window. Keyword-only. Default: ``None``.
   * - ``material``
     - Optional
     - ``bool | None``
     - Material object defining relative electric and magnetic response. Exterior fills points outside placed material regions. Keyword-only. Default: ``None``.
   * - ``field``
     - Optional
     - ``bool``
     - Complex electromagnetic field samples or FE field objects. Cartesian components follow x,y,z order; left/right denote the selected trace side. Keyword-only. Default: ``True``.
   * - ``normalize``
     - Optional
     - ``bool``
     - Modal/field normalization convention or flag. Unit-longitudinal-power normalization applies only to modes with usable real power. Keyword-only. Default: ``False``.
   * - ``cmap``
     - Optional
     - ``str | None``
     - Matplotlib colormap name or object used for scalar field rendering. Keyword-only. Default: ``None``.
   * - ``axes``
     - Optional
     - ``Any | None``
     - Existing Matplotlib axes for embedding a plot; None creates suitable axes when accepted. Keyword-only. Default: ``None``.
   * - ``title``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.
   * - ``show``
     - Optional
     - ``bool``
     - Enable display of the figure, mesh overlay, or colorbar respectively. show=False returns plotting objects without opening a window. Keyword-only. Default: ``True``.
   * - ``**legacy_component_flags``
     - Optional
     - ``Any``
     - Compatibility-only input from older APIs; it does not add subpixel FEM averaging. Unsupported legacy component selections are rejected.

Returns: ``tuple[Any, NDArray[Any]]``.

``mode`` is one-based for compatibility with the original mode solvers. A single ``component`` or several ``components`` may be requested. The legacy boolean flags ``ex=True`` through ``hz=True``, plus ``eabs`` and ``habs``, remain accepted by thin solver wrappers.

FEM_Mode_Solver.visualize_with_gui
----------------------------------

``FEM_Mode_Solver.visualize_with_gui``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create an interactive mode/component/quantity viewer.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.visualize_with_gui(source: 'Mode | ModeSet | SupportsModeVisualization | Any', *, mode: 'int' = 1, component: 'str | None' = None, quantity: 'str' = 'real', mesh: 'bool' = False, mesh_overlay: 'bool | None' = None, material: 'bool | None' = None, field: 'bool' = True, normalize: 'bool' = False, cmap: 'str | None' = None, show: 'bool' = True, block: 'bool | None' = None) -> 'ModeViewer'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``source``
     - Required
     - ``Mode | ModeSet | SupportsModeVisualization | Any``
     - Equivalent-current source, mode-set/result object, or plotting data source accepted by the owning function; see its concrete type annotation.
   * - ``mode``
     - Optional
     - ``int``
     - Mode or mode collection. Integer selectors use the owning API indexing convention; supplied mode sets must match frequency, ky, and transverse geometry. Keyword-only. Default: ``1``.
   * - ``component``
     - Optional
     - ``str | None``
     - Field component selection, using Cartesian electric/magnetic names such as Ex or Hy. The visualization API also supports its documented aggregate quantities. Keyword-only. Default: ``None``.
   * - ``quantity``
     - Optional
     - ``str``
     - Complex-data display selection, such as real, imag, abs, or phase; accepted values are given by the owning plotting API. Keyword-only. Default: ``'real'``.
   * - ``mesh``
     - Optional
     - ``bool``
     - Conforming FEM mesh object. Physical meshes carry SI coordinates; computational meshes may be scaled by the reference length for assembly. Keyword-only. Default: ``False``.
   * - ``mesh_overlay``
     - Optional
     - ``bool | None``
     - Enable display of the figure, mesh overlay, or colorbar respectively. show=False returns plotting objects without opening a window. Keyword-only. Default: ``None``.
   * - ``material``
     - Optional
     - ``bool | None``
     - Material object defining relative electric and magnetic response. Exterior fills points outside placed material regions. Keyword-only. Default: ``None``.
   * - ``field``
     - Optional
     - ``bool``
     - Complex electromagnetic field samples or FE field objects. Cartesian components follow x,y,z order; left/right denote the selected trace side. Keyword-only. Default: ``True``.
   * - ``normalize``
     - Optional
     - ``bool``
     - Modal/field normalization convention or flag. Unit-longitudinal-power normalization applies only to modes with usable real power. Keyword-only. Default: ``False``.
   * - ``cmap``
     - Optional
     - ``str | None``
     - Matplotlib colormap name or object used for scalar field rendering. Keyword-only. Default: ``None``.
   * - ``show``
     - Optional
     - ``bool``
     - Enable display of the figure, mesh overlay, or colorbar respectively. show=False returns plotting objects without opening a window. Keyword-only. Default: ``True``.
   * - ``block``
     - Optional
     - ``bool | None``
     - HDF5 case selector or mode/block slice bounds, using zero-based indexing and a stop-exclusive interval. Keyword-only. Default: ``None``.

Returns: ``ModeViewer``.

The GUI uses Matplotlib widgets, so it works with any interactive Matplotlib backend and does not require solver-specific Tk code. The returned controller must be kept alive while the window is open.

FEM_Mode_Solver.assembly.ModeFEMSystem2D
----------------------------------------

``FEM_Mode_Solver.assembly.ModeFEMSystem2D``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reduced quadratic pencil plus data needed to reconstruct FEM fields.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.assembly.ModeFEMSystem2D(basis: 'Basis', physical_mesh: 'MeshTri', computational_mesh: 'MeshTri', A0: 'csr_matrix', A1: 'csr_matrix', A2: 'csr_matrix', free_dofs: 'IntArray', full_size: 'int', transverse_indices: 'IntArray', longitudinal_indices: 'IntArray', gauss_transverse: 'csr_matrix', gauss_longitudinal: 'csr_matrix', gauss_test_dofs: 'IntArray', frequency: 'float', k0: 'float', boundary: 'str', material_at: 'MaterialEvaluator', quadrature_order: 'int', impedance_boundaries: 'tuple[tuple[IntArray, complex], ...]') -> None

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
   * - ``physical_mesh``
     - Required
     - ``MeshTri``
     - Conforming FEM mesh object. Physical meshes carry SI coordinates; computational meshes may be scaled by the reference length for assembly.
   * - ``computational_mesh``
     - Required
     - ``MeshTri``
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
   * - ``free_dofs``
     - Required
     - ``IntArray``
     - Integer degree-of-freedom indices selecting constrained/free unknowns or admissible scalar test functions.
   * - ``full_size``
     - Required
     - ``int``
     - Total number of nodes or degrees of freedom in the relevant full space; a nonnegative/positive integer as required by the constructor.
   * - ``transverse_indices``
     - Required
     - ``IntArray``
     - Indices/slices selecting transverse, longitudinal, or Cartesian components from the full mixed coefficient vector.
   * - ``longitudinal_indices``
     - Required
     - ``IntArray``
     - Indices/slices selecting transverse, longitudinal, or Cartesian components from the full mixed coefficient vector.
   * - ``gauss_transverse``
     - Required
     - ``csr_matrix``
     - Discrete weak divergence/Gauss operator used to validate modal charge consistency in the associated scalar test space.
   * - ``gauss_longitudinal``
     - Required
     - ``csr_matrix``
     - Discrete weak divergence/Gauss operator used to validate modal charge consistency in the associated scalar test space.
   * - ``gauss_test_dofs``
     - Required
     - ``IntArray``
     - Integer degree-of-freedom indices selecting constrained/free unknowns or admissible scalar test functions.
   * - ``frequency``
     - Required
     - ``float``
     - Ordinary frequency in Hz; must be finite and positive.
   * - ``k0``
     - Required
     - ``float``
     - Vacuum wavenumber in rad/m, used for coordinate scaling and material/PML evaluation.
   * - ``boundary``
     - Required
     - ``str``
     - Outer transverse boundary condition. Modal solvers support PEC/PMC as documented; integrated WaveFEM supports PEC or transverse PML terminated by PEC.
   * - ``material_at``
     - Required
     - ``MaterialEvaluator``
     - Vectorized material/field callback evaluated at the coordinates supplied by the calling API; return the scalar, diagonal array, or field shape specified by that operation.
   * - ``quadrature_order``
     - Required
     - ``int``
     - Finite-element integration order. Higher-order mixed elements require at least fourth-order quadrature; PML and material variation can require more.
   * - ``impedance_boundaries``
     - Required
     - ``tuple[tuple[IntArray, complex], ...]``
     - Ordered layer/region/boundary specifications. Later overlapping material regions take precedence where the geometry API permits overlap.

Returns: ``ModeFEMSystem2D``.

``FEM_Mode_Solver.assembly.ModeFEMSystem2D.divergence_residual``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``divergence_residual`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.assembly.ModeFEMSystem2D.divergence_residual(full_vector: 'ArrayLike', neff: 'complex') -> 'float'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``full_vector``
     - Required
     - ``ArrayLike``
     - Complex finite-element coefficient vector or coefficient values, ordered exactly as the associated basis/system. Full vectors include constrained/periodic copies.
   * - ``neff``
     - Required
     - ``complex``
     - Dimensionless effective index beta/k0. A guess selects roots near that complex value; None uses the solver estimate.

Returns: ``float``.

``FEM_Mode_Solver.assembly.ModeFEMSystem2D.expand``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``expand`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.assembly.ModeFEMSystem2D.expand(vector: 'ArrayLike') -> 'ComplexArray'

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

``FEM_Mode_Solver.assembly.ModeFEMSystem2D.ndofs``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's ndofs value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.assembly.ModeFEMSystem2D.ndofs

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

``FEM_Mode_Solver.assembly.ModeFEMSystem2D.polynomial``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``polynomial`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.assembly.ModeFEMSystem2D.polynomial(neff: 'complex') -> 'csr_matrix'

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

``FEM_Mode_Solver.assembly.ModeFEMSystem2D.relative_hermiticity_errors``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``relative_hermiticity_errors`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.assembly.ModeFEMSystem2D.relative_hermiticity_errors() -> 'tuple[float, float, float]'

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

``FEM_Mode_Solver.assembly.ModeFEMSystem2D.relative_residual``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``relative_residual`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.assembly.ModeFEMSystem2D.relative_residual(vector: 'ArrayLike', neff: 'complex') -> 'float'

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

``FEM_Mode_Solver.assembly.ModeFEMSystem2D.split_full``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``split_full`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.assembly.ModeFEMSystem2D.split_full(vector: 'ArrayLike') -> 'tuple[ComplexArray, ComplexArray]'

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

Returns: ``tuple[ComplexArray, ComplexArray]``.

FEM_Mode_Solver.assembly.assemble_mode_system_2d
------------------------------------------------

``FEM_Mode_Solver.assembly.assemble_mode_system_2d``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Assemble the dimensionless full-vector propagation-constant pencil.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.assembly.assemble_mode_system_2d(mesh: 'MeshTri', *, frequency: 'float', k0: 'float', material_at: 'MaterialEvaluator', boundary: 'str' = 'pec', quadrature_order: 'int' = 4, element_order: 'int' = 1, pec_facets: 'ArrayLike | None' = None, impedance_boundaries: 'Sequence[ImpedanceBoundaryInput] | None' = None) -> 'ModeFEMSystem2D'

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
   * - ``boundary``
     - Optional
     - ``str``
     - Outer transverse boundary condition. Modal solvers support PEC/PMC as documented; integrated WaveFEM supports PEC or transverse PML terminated by PEC. Keyword-only. Default: ``'pec'``.
   * - ``quadrature_order``
     - Optional
     - ``int``
     - Finite-element integration order. Higher-order mixed elements require at least fourth-order quadrature; PML and material variation can require more. Keyword-only. Default: ``4``.
   * - ``element_order``
     - Optional
     - ``int``
     - Finite-element polynomial-order selection. Standalone 2D modes and WaveFEM scattering accept 1 (N1/P1) or 2 (N2/P2); other backends retain their fixed compatible spaces. Keyword-only. Default: ``1``.
   * - ``pec_facets``
     - Optional
     - ``ArrayLike | None``
     - Facet indices in the corresponding mesh. Actual PEC remains constrained; released facets form apertures; inserted facets prescribe scattered tangential fields. Keyword-only. Default: ``None``.
   * - ``impedance_boundaries``
     - Optional
     - ``Sequence[ImpedanceBoundaryInput] | None``
     - Ordered layer/region/boundary specifications. Later overlapping material regions take precedence where the geometry API permits overlap. Keyword-only. Default: ``None``.

Returns: ``ModeFEMSystem2D``.

``impedance_boundaries`` is a sequence of ``(facet_indices, Zs)`` pairs, where each impedance is in ohms. Supplied facets must be boundary facets, must not overlap each other, and must not also be constrained as PEC. When ``boundary='pec'`` and ``pec_facets`` is omitted, impedance facets automatically replace the default PEC condition on those exterior facets.

FEM_Mode_Solver.assembly.evaluate_material
------------------------------------------

``FEM_Mode_Solver.assembly.evaluate_material``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluate and validate relative diagonal ``(epsilon, mu)`` fields.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.assembly.evaluate_material(material_at: 'MaterialEvaluator', x: 'NDArray[np.floating]', y: 'NDArray[np.floating]') -> 'tuple[ComplexArray, ComplexArray]'

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
   * - ``y``
     - Required
     - ``NDArray[np.floating]``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature.

Returns: ``tuple[ComplexArray, ComplexArray]``.

FEM_Mode_Solver.assembly.linearized_pencil
------------------------------------------

``FEM_Mode_Solver.assembly.linearized_pencil``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Return a first companion linearization of ``system``.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.assembly.linearized_pencil(system: 'ModeFEMSystem2D') -> 'tuple[csc_matrix, csc_matrix]'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``system``
     - Required
     - ``ModeFEMSystem2D``
     - Assembled FEM system containing compatible bases, sparse operators, constraints, and material/reference-scale metadata.

Returns: ``tuple[csc_matrix, csc_matrix]``.

FEM_Mode_Solver.assembly.solve_qep_candidates
---------------------------------------------

``FEM_Mode_Solver.assembly.solve_qep_candidates``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Return finite candidate roots/vectors nearest ``target``.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.assembly.solve_qep_candidates(system: 'ModeFEMSystem2D', *, target: 'complex', candidate_count: 'int', tolerance: 'float', dense_linearization_limit: 'int' = 700) -> 'tuple[ComplexArray, ComplexArray, str]'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``system``
     - Required
     - ``ModeFEMSystem2D``
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
     - Required
     - ``float``
     - Positive numerical tolerance. Linear/QEP residual tolerances validate the algebraic solve; they do not set the adaptive mesh threshold. Keyword-only.
   * - ``dense_linearization_limit``
     - Optional
     - ``int``
     - Matrix-size cutoff for dense eigensolving; larger systems use a sparse backend. This is a dimension limit, not a mesh-error threshold. Keyword-only. Default: ``700``.

Returns: ``tuple[ComplexArray, ComplexArray, str]``.

FEM_Mode_Solver.geometry.GeometryModel1D
----------------------------------------

``FEM_Mode_Solver.geometry.GeometryModel1D``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Construct a ``GeometryModel1D`` record. Its public data fields use the same names and types as the constructor inputs below. Solvers normally construct mesh/system/result records for you.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.geometry.GeometryModel1D(x_span: 'float | Sequence[float]', background: 'Material') -> 'None'

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
   * - ``background``
     - Required
     - ``Material``
     - In placement methods, True adds a z-invariant background region/sheet; False adds an actual-device perturbation. In material/geometry constructors, this is the exterior Material object.

Returns: ``GeometryModel1D``.

``FEM_Mode_Solver.geometry.GeometryModel1D.add_boundary``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add boundary; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.geometry.GeometryModel1D.add_boundary(interval: 'Interval', kind: 'str', *, impedance: 'complex | None' = None, name: 'str | None' = None) -> 'BoundaryRegion'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``interval``
     - Required
     - ``Interval``
     - Geometry primitive or region selector identifying the physical support. Use the class/union in the signature; electrostatics also accepts named exterior boundaries.
   * - ``kind``
     - Required
     - ``str``
     - Object/boundary/scene kind. Use the permitted Literal values or the documented selector for the owning class.
   * - ``impedance``
     - Optional
     - ``complex | None``
     - Surface impedance in ohms. Supply an explicit passive complex value or a supported metal through the alternative material input. Keyword-only. Default: ``None``.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Returns: ``BoundaryRegion``.

``FEM_Mode_Solver.geometry.GeometryModel1D.add_change_listener``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Notify an owning solver whenever the continuous scene changes.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.geometry.GeometryModel1D.add_change_listener(callback: 'Callable[[], None]') -> 'None'

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

``FEM_Mode_Solver.geometry.GeometryModel1D.add_pml``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add pml; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.geometry.GeometryModel1D.add_pml(spec: 'PMLSpec') -> 'PMLSpec'

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

``FEM_Mode_Solver.geometry.GeometryModel1D.add_region``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add region; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.geometry.GeometryModel1D.add_region(interval: 'Interval', material: 'Material', *, name: 'str | None' = None) -> 'Region'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``interval``
     - Required
     - ``Interval``
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

``FEM_Mode_Solver.geometry.GeometryModel1D.material_at``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``material_at`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.geometry.GeometryModel1D.material_at(x: 'ArrayLike') -> 'tuple[NDArray[np.complex128], NDArray[np.complex128]]'

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

Returns: ``tuple[NDArray[np.complex128], NDArray[np.complex128]]``.

``FEM_Mode_Solver.geometry.GeometryModel1D.remove``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Remove ; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.geometry.GeometryModel1D.remove(item: 'Region | BoundaryRegion | MeshRefinement') -> 'None'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``item``
     - Required
     - ``Region | BoundaryRegion | MeshRefinement``
     - Previously returned region/boundary/PML handle to remove, or an index/key for a container operation as indicated by the method.

Returns: ``None``.

``FEM_Mode_Solver.geometry.GeometryModel1D.set_outer_boundary``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set outer boundary; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.geometry.GeometryModel1D.set_outer_boundary(kind: 'str') -> 'None'

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

``FEM_Mode_Solver.geometry.GeometryModel1D.transformed_material_at``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``transformed_material_at`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.geometry.GeometryModel1D.transformed_material_at(x: 'ArrayLike') -> 'tuple[NDArray[np.complex128], NDArray[np.complex128]]'

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

Returns: ``tuple[NDArray[np.complex128], NDArray[np.complex128]]``.

FEM_Mode_Solver.geometry.GeometryModel2D
----------------------------------------

``FEM_Mode_Solver.geometry.GeometryModel2D``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Construct a ``GeometryModel2D`` record. Its public data fields use the same names and types as the constructor inputs below. Solvers normally construct mesh/system/result records for you.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.geometry.GeometryModel2D(x_span: 'float | Sequence[float]', y_span: 'float | Sequence[float]', background: 'Material') -> 'None'

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
   * - ``background``
     - Required
     - ``Material``
     - In placement methods, True adds a z-invariant background region/sheet; False adds an actual-device perturbation. In material/geometry constructors, this is the exterior Material object.

Returns: ``GeometryModel2D``.

``FEM_Mode_Solver.geometry.GeometryModel2D.add_boundary``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add boundary; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.geometry.GeometryModel2D.add_boundary(shape: 'Shape2D', kind: 'str', *, impedance: 'complex | None' = None, name: 'str | None' = None) -> 'BoundaryRegion'

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
   * - ``impedance``
     - Optional
     - ``complex | None``
     - Surface impedance in ohms. Supply an explicit passive complex value or a supported metal through the alternative material input. Keyword-only. Default: ``None``.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Returns: ``BoundaryRegion``.

``FEM_Mode_Solver.geometry.GeometryModel2D.add_change_listener``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Notify an owning solver whenever the continuous scene changes.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.geometry.GeometryModel2D.add_change_listener(callback: 'Callable[[], None]') -> 'None'

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

``FEM_Mode_Solver.geometry.GeometryModel2D.add_mesh_refinement``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add a non-physical local mesh-size region to the continuous scene.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.geometry.GeometryModel2D.add_mesh_refinement(shape: 'Shape2D', max_element_size: 'float', *, transition_width: 'float' = 0.0, name: 'str | None' = None) -> 'MeshRefinement'

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
   * - ``transition_width``
     - Optional
     - ``float``
     - Physical distance in metres over which the local mesh-size target transitions back to the surrounding target; None selects the mesher default when permitted. Keyword-only. Default: ``0.0``.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Returns: ``MeshRefinement``.

``FEM_Mode_Solver.geometry.GeometryModel2D.add_pml``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add pml; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.geometry.GeometryModel2D.add_pml(spec: 'PMLSpec') -> 'PMLSpec'

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

``FEM_Mode_Solver.geometry.GeometryModel2D.add_region``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add region; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.geometry.GeometryModel2D.add_region(shape: 'Shape2D', material: 'Material', *, name: 'str | None' = None) -> 'Region'

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

``FEM_Mode_Solver.geometry.GeometryModel2D.material_at``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``material_at`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.geometry.GeometryModel2D.material_at(x: 'ArrayLike', y: 'ArrayLike') -> 'tuple[NDArray[np.complex128], NDArray[np.complex128]]'

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

Returns: ``tuple[NDArray[np.complex128], NDArray[np.complex128]]``.

``FEM_Mode_Solver.geometry.GeometryModel2D.pml_interfaces``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``pml_interfaces`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.geometry.GeometryModel2D.pml_interfaces() -> 'tuple[tuple[float, ...], tuple[float, ...]]'

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

Returns: ``tuple[tuple[float, ...], tuple[float, ...]]``.

``FEM_Mode_Solver.geometry.GeometryModel2D.remove``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Remove ; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.geometry.GeometryModel2D.remove(item: 'Region | BoundaryRegion | MeshRefinement') -> 'None'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``item``
     - Required
     - ``Region | BoundaryRegion | MeshRefinement``
     - Previously returned region/boundary/PML handle to remove, or an index/key for a container operation as indicated by the method.

Returns: ``None``.

``FEM_Mode_Solver.geometry.GeometryModel2D.set_outer_boundary``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set outer boundary; inputs and selection controls are listed below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.geometry.GeometryModel2D.set_outer_boundary(kind: 'str') -> 'None'

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

``FEM_Mode_Solver.geometry.GeometryModel2D.transformed_material_at``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``transformed_material_at`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.geometry.GeometryModel2D.transformed_material_at(x: 'ArrayLike', y: 'ArrayLike') -> 'tuple[NDArray[np.complex128], NDArray[np.complex128]]'

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

Returns: ``tuple[NDArray[np.complex128], NDArray[np.complex128]]``.

FEM_Mode_Solver.geometry.material
---------------------------------

``FEM_Mode_Solver.geometry.material``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Small convenience used by the public placement methods.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.geometry.material(epsilon: 'MaterialInput', mu: 'MaterialInput') -> 'Material'

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

Returns: ``Material``.

FEM_Mode_Solver.geometry.physical_span
--------------------------------------

``FEM_Mode_Solver.geometry.physical_span``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Return a finite increasing physical span.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.geometry.physical_span(value: 'float | Sequence[float]', name: 'str') -> 'tuple[float, float]'

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

A positive scalar denotes a domain from zero to that value, matching the older mode-solver constructors. Geometry primitives should use explicit ``(minimum, maximum)`` pairs.

FEM_Mode_Solver.materials.diagonal_values
-----------------------------------------

``FEM_Mode_Solver.materials.diagonal_values``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Normalize a scalar or three diagonal entries in physical x/y/z order.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.materials.diagonal_values(value: 'MaterialInput', name: 'str') -> 'tuple[complex, complex, complex]'

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

FEM_Mode_Solver.meshing.discretize_1d
-------------------------------------

``FEM_Mode_Solver.meshing.discretize_1d``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create an interface-conforming, material-aware line mesh.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.meshing.discretize_1d(geometry: 'GeometryModel1D', *, resolution: 'int | None' = None, max_element_size: 'float | None' = None, element_order: 'int' = 1, vacuum_wavenumber: 'float | None' = None, wavelength_elements: 'int' = 10, material_aware: 'bool' = True) -> 'FEMMesh1D'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``geometry``
     - Required
     - ``GeometryModel1D``
     - Geometry model containing physical bounds, material regions, conductor boundaries, PMLs, and sizing requests.
   * - ``resolution``
     - Optional
     - ``int | None``
     - Initial mesh-resolution request. 1D uses an interval count; 2D accepts axis counts. Physical maximum size and wavelength constraints may increase the generated count. Keyword-only. Default: ``None``.
   * - ``max_element_size``
     - Optional
     - ``float | None``
     - Physical element-edge length in metres. A maximum target is an upper sizing request; material, boundary, and wavelength constraints may produce smaller cells. Keyword-only. Default: ``None``.
   * - ``element_order``
     - Optional
     - ``int``
     - Finite-element polynomial-order selection. Standalone 2D modes and WaveFEM scattering accept 1 (N1/P1) or 2 (N2/P2); other backends retain their fixed compatible spaces. Keyword-only. Default: ``1``.
   * - ``vacuum_wavenumber``
     - Optional
     - ``float | None``
     - Vacuum wavenumber in rad/m, used for coordinate scaling and material/PML evaluation. Keyword-only. Default: ``None``.
   * - ``wavelength_elements``
     - Optional
     - ``int``
     - Minimum requested elements per shortest local material wavelength; integer at least four. Public adaptive workflows default to four. Keyword-only. Default: ``10``.
   * - ``material_aware``
     - Optional
     - ``bool``
     - Enable element-size reduction in high-index/high-permittivity material regions while retaining the global maximum-edge cap. Keyword-only. Default: ``True``.

Returns: ``FEMMesh1D``.

``resolution`` and ``max_element_size`` define the target in the lowest-index interval. In material-aware mode that size is reduced in proportion to the local index estimate ``sqrt(max(abs(epsilon)) * max(abs(mu)))``, so a high-Dk interval receives more elements without moving its exact material interfaces. When ``vacuum_wavenumber`` is supplied, the local size is also limited to one material wavelength divided by ``wavelength_elements``.

Set ``material_aware=False`` for a uniform target size. The shortest wavelength is still respected, but its target is then applied throughout the domain instead of only in the high-wavenumber intervals.

FEM_Mode_Solver.meshing.discretize_2d
-------------------------------------

``FEM_Mode_Solver.meshing.discretize_2d``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate a conforming, optionally material-aware Gmsh mesh.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.meshing.discretize_2d(geometry: 'GeometryModel2D', *, max_element_size: 'float | None' = None, resolution: 'tuple[int, int] | None' = None, element_order: 'int' = 1, material_aware: 'bool' = True, vacuum_wavenumber: 'float | None' = None, wavelength_elements: 'int' = 10, interface_refinement: 'float | None' = None, interface_refinement_width: 'float | None' = None, boundary_refinement: 'float | None' = 0.5, boundary_refinement_width: 'float | None' = None, _refinement_scale: 'float' = 1.0) -> 'FEMMesh2D'

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
     - Optional
     - ``float | None``
     - Physical element-edge length in metres. A maximum target is an upper sizing request; material, boundary, and wavelength constraints may produce smaller cells. Keyword-only. Default: ``None``.
   * - ``resolution``
     - Optional
     - ``tuple[int, int] | None``
     - Initial mesh-resolution request. 1D uses an interval count; 2D accepts axis counts. Physical maximum size and wavelength constraints may increase the generated count. Keyword-only. Default: ``None``.
   * - ``element_order``
     - Optional
     - ``int``
     - Finite-element polynomial-order selection. Standalone 2D modes and WaveFEM scattering accept 1 (N1/P1) or 2 (N2/P2); other backends retain their fixed compatible spaces. Keyword-only. Default: ``1``.
   * - ``material_aware``
     - Optional
     - ``bool``
     - Enable element-size reduction in high-index/high-permittivity material regions while retaining the global maximum-edge cap. Keyword-only. Default: ``True``.
   * - ``vacuum_wavenumber``
     - Optional
     - ``float | None``
     - Vacuum wavenumber in rad/m, used for coordinate scaling and material/PML evaluation. Keyword-only. Default: ``None``.
   * - ``wavelength_elements``
     - Optional
     - ``int``
     - Minimum requested elements per shortest local material wavelength; integer at least four. Public adaptive workflows default to four. Keyword-only. Default: ``10``.
   * - ``interface_refinement``
     - Optional
     - ``float | None``
     - Optional size multiplier in (0, 1] near conductor/exterior boundaries or material interfaces. None disables this sizing field. Keyword-only. Default: ``None``.
   * - ``interface_refinement_width``
     - Optional
     - ``float | None``
     - Physical distance in metres over which the local mesh-size target transitions back to the surrounding target; None selects the mesher default when permitted. Keyword-only. Default: ``None``.
   * - ``boundary_refinement``
     - Optional
     - ``float | None``
     - Optional size multiplier in (0, 1] near conductor/exterior boundaries or material interfaces. None disables this sizing field. Keyword-only. Default: ``0.5``.
   * - ``boundary_refinement_width``
     - Optional
     - ``float | None``
     - Physical distance in metres over which the local mesh-size target transitions back to the surrounding target; None selects the mesher default when permitted. Keyword-only. Default: ``None``.
   * - ``_refinement_scale``
     - Optional
     - ``float``
     - Mesh-density multiplier. Public refine requires a finite value greater than one; the internal refinement scale accumulates requested sizing changes. Keyword-only. Default: ``1.0``.

Returns: ``FEMMesh2D``.

``max_element_size`` is a global Gmsh characteristic-size target. With ``material_aware=True`` each material receives a smaller local target in proportion to its propagation-wavenumber proxy ``sqrt(max(abs(epsilon_i)) * max(abs(mu_i)))``. An optional ``interface_refinement`` in ``(0, 1]`` further scales the edge target near material jumps. Refinement controls stored on ``geometry`` are applied without changing material or boundary provenance.

FEM_Mode_Solver.visualization.SupportsModeVisualization
-------------------------------------------------------

``FEM_Mode_Solver.visualization.SupportsModeVisualization``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Structural typing protocol for compatible field/visualization objects. Implement the declared attributes and methods; this protocol is not instantiated directly.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.visualization.SupportsModeVisualization()

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

Returns: ``SupportsModeVisualization``.

Solvers normally satisfy this protocol by storing the ``ModeSet`` returned by ``solve()`` in a ``solution`` property.

``FEM_Mode_Solver.visualization.SupportsModeVisualization.solution``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's solution value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   FEM_Mode_Solver.visualization.SupportsModeVisualization.solution

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

Returns: ``ModeSet | None``.

Export aliases and constants
----------------------------

Aliases below have exactly the same input tables and return contracts as their targets. Constants/type aliases are values, not calls, and take no input arguments.

.. list-table:: Exports
   :header-rows: 1

   * - Name
     - Value or target
   * - ``FEM_Mode_Solver.boundaries.canonical_metal_name``
     - ``FEM_Mode_Solver.canonical_metal_name``
   * - ``FEM_Mode_Solver.boundaries.good_conductor_surface_impedance``
     - ``FEM_Mode_Solver.good_conductor_surface_impedance``
   * - ``FEM_Mode_Solver.boundaries.validate_surface_impedance``
     - ``FEM_Mode_Solver.validate_surface_impedance``
   * - ``FEM_Mode_Solver.exceptions.BackendCapabilityError``
     - ``FEM_Mode_Solver.BackendCapabilityError``
   * - ``FEM_Mode_Solver.exceptions.ConfigurationError``
     - ``FEM_Mode_Solver.ConfigurationError``
   * - ``FEM_Mode_Solver.exceptions.FEMModeSolverError``
     - ``FEM_Mode_Solver.FEMModeSolverError``
   * - ``FEM_Mode_Solver.exceptions.GeometryError``
     - ``FEM_Mode_Solver.GeometryError``
   * - ``FEM_Mode_Solver.exceptions.MeshError``
     - ``FEM_Mode_Solver.MeshError``
   * - ``FEM_Mode_Solver.exceptions.NotDiscretizedError``
     - ``FEM_Mode_Solver.NotDiscretizedError``
   * - ``FEM_Mode_Solver.exceptions.SolverError``
     - ``FEM_Mode_Solver.SolverError``
   * - ``FEM_Mode_Solver.exceptions.StaleDiscretizationError``
     - ``FEM_Mode_Solver.StaleDiscretizationError``
   * - ``FEM_Mode_Solver.geometry.BoundaryRegion``
     - ``FEM_Mode_Solver.BoundaryRegion``
   * - ``FEM_Mode_Solver.geometry.Circle``
     - ``FEM_Mode_Solver.Circle``
   * - ``FEM_Mode_Solver.geometry.Interval``
     - ``FEM_Mode_Solver.Interval``
   * - ``FEM_Mode_Solver.geometry.MeshRefinement``
     - ``FEM_Mode_Solver.MeshRefinement``
   * - ``FEM_Mode_Solver.geometry.PMLSpec``
     - ``FEM_Mode_Solver.PMLSpec``
   * - ``FEM_Mode_Solver.geometry.Polygon``
     - ``FEM_Mode_Solver.Polygon``
   * - ``FEM_Mode_Solver.geometry.Rectangle``
     - ``FEM_Mode_Solver.Rectangle``
   * - ``FEM_Mode_Solver.geometry.Region``
     - ``FEM_Mode_Solver.Region``
   * - ``FEM_Mode_Solver.materials.Material``
     - ``FEM_Mode_Solver.Material``
   * - ``FEM_Mode_Solver.meshing.FEMMesh1D``
     - ``FEM_Mode_Solver.FEMMesh1D``
   * - ``FEM_Mode_Solver.meshing.FEMMesh2D``
     - ``FEM_Mode_Solver.FEMMesh2D``
   * - ``FEM_Mode_Solver.meshing.MeshInfo``
     - ``FEM_Mode_Solver.MeshInfo``
   * - ``FEM_Mode_Solver.results.Mode``
     - ``FEM_Mode_Solver.Mode``
   * - ``FEM_Mode_Solver.results.ModeSet``
     - ``FEM_Mode_Solver.ModeSet``
   * - ``FEM_Mode_Solver.results.SampledFields``
     - ``FEM_Mode_Solver.SampledFields``
   * - ``FEM_Mode_Solver.solver_1d.ModeSolver1D``
     - ``FEM_Mode_Solver.ModeSolver1D``
   * - ``FEM_Mode_Solver.solver_2d.ModeSolver2D``
     - ``FEM_Mode_Solver.ModeSolver2D``
   * - ``FEM_Mode_Solver.visualization.ModeViewer``
     - ``FEM_Mode_Solver.ModeViewer``
   * - ``FEM_Mode_Solver.visualization.visualize``
     - ``FEM_Mode_Solver.visualize``
   * - ``FEM_Mode_Solver.visualization.visualize_with_gui``
     - ``FEM_Mode_Solver.visualize_with_gui``
   * - ``FEM_Mode_Solver.METAL_RESISTIVITIES_OHM_M``
     - ``mappingproxy({'aluminium': 2.65e-08, 'copper': 1.676e-08, 'gold': 2.192e-08, 'molybdenum': 5.34e-08, 'palladium': 1.054e-07, 'silver': 1.586e-08, 'tungsten': 5.28e-08, 'zinc': 5.964e-08})``
   * - ``FEM_Mode_Solver.assembly.ImpedanceBoundaryInput``
     - ``Type alias; see the array/material conventions and owning module annotation.``
   * - ``FEM_Mode_Solver.assembly.MaterialEvaluator``
     - ``Type alias; see the array/material conventions and owning module annotation.``
   * - ``FEM_Mode_Solver.boundaries.METAL_RESISTIVITIES_OHM_M``
     - ``mappingproxy({'aluminium': 2.65e-08, 'copper': 1.676e-08, 'gold': 2.192e-08, 'molybdenum': 5.34e-08, 'palladium': 1.054e-07, 'silver': 1.586e-08, 'tungsten': 5.28e-08, 'zinc': 5.964e-08})``
   * - ``FEM_Mode_Solver.constants.C_0``
     - ``299792458.0``
   * - ``FEM_Mode_Solver.constants.EPSILON_0``
     - ``8.8541878188e-12``
   * - ``FEM_Mode_Solver.constants.ETA_0``
     - ``376.7303134118051``
   * - ``FEM_Mode_Solver.constants.MU_0``
     - ``1.25663706127e-06``
   * - ``FEM_Mode_Solver.materials.MaterialInput``
     - ``float | complex | numpy.number | tuple[float | complex | numpy.number, float | complex | numpy.number, float | complex | numpy.number] | list[float | complex | numpy.number] | numpy.ndarray``
