fem_waveguide_modes user API
============================

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

Materials support scalar or diagonal epsilon and mu. Unsupported off-diagonal
forms raise an explicit configuration error. Periodic archives store periodic
field envelopes; the Bloch phase is recorded separately from these fields.

Supported exports
-----------------

``ModeSolver1D``, ``ModeSolver2D``, ``Mode``, ``ModeSet``, ``SampledFields``, ``BackendCapabilityError``, ``ConfigurationError``, ``FEMModeSolverError``, ``GeometryError``, ``MeshError``, ``SolverError``, ``NoResultError``, ``PersistenceError``, ``load_result``.

Solver construction and operations
----------------------------------

``ModeSolver1D``
~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver1D(*, frequency: 'float', x_range: 'float | Sequence[float]', background_material: 'materials.Material' = Material(name='vacuum', epsilon=(1+0j), mu=(1+0j))) -> 'None'

FEM-native mode solver for an x-stratified cross-section.

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
   * - ``x_range``
     - ``float | Sequence[float]``
     - Required
     - ``—``
     - Physical x extent or increasing bounds, in metres.
   * - ``background_material``
     - ``materials.Material``
     - Optional
     - ``Material(name='vacuum', epsilon=(1+0j), mu=(1+0j))``
     - Predefined bulk Material assigned to unfilled space.

Returns: a configured ``ModeSolver1D``.

``ModeSolver1D.add_geometry``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver1D.add_geometry(*, shape, material, name=None, clip=False)

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

``ModeSolver1D.add_layer``
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver1D.add_layer(*, x_range, material, name=None, clip=False)

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

``ModeSolver1D.set_material``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver1D.set_material(*, geometry, material)

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

``ModeSolver1D.set_shape``
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver1D.set_shape(*, geometry, shape)

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

``ModeSolver1D.set_boundary``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver1D.set_boundary(*, material)

Set the exterior PEC/PMC wall; internal SIBC uses geometry assignment.

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

``ModeSolver1D.add_pml``
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver1D.add_pml(*, thickness: 'float', order: 'int' = 3, sigma_max: 'float' = 5.0, direction: 'str' = 'all') -> 'PMLSpec'

Place a physical transformation-optics PML at selected x ends.

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
   * - ``order``
     - ``int``
     - Optional
     - ``3``
     - Polynomial order of the PML profile.
   * - ``sigma_max``
     - ``float``
     - Optional
     - ``5.0``
     - Backend PML-strength magnitude; the outgoing stretch has negative imaginary sign.
   * - ``direction``
     - ``str``
     - Optional
     - ``'all'``
     - Propagation direction for solve; selected coordinate direction for PML.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``ModeSolver1D.remove``
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver1D.remove(*, geometry)

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

``ModeSolver1D.refine``
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver1D.refine(factor: 'float' = 2.0) -> 'FEMMesh1D'

Remesh the current geometry with ``factor`` times the density.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``factor``
     - ``float``
     - Optional
     - ``2.0``
     - Factor control for this operation.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``ModeSolver1D.mesh``
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver1D.mesh(*, max_element_size: 'float | None' = None, resolution: 'int | None' = None, wavelength_elements: 'int' = 4, material_aware: 'bool' = True, element_order: 'int' = 1, quadrature_order: 'int' = 4)

Build the initial mesh; solve() may subsequently refine it.

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
   * - ``resolution``
     - ``int | None``
     - Optional
     - ``None``
     - Initial node counts; use instead of a maximum element size.
   * - ``wavelength_elements``
     - ``int``
     - Optional
     - ``4``
     - Minimum number of initial elements per local wavelength.
   * - ``material_aware``
     - ``bool``
     - Optional
     - ``True``
     - Use material-dependent initial mesh sizing.
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

Returns: the initial mesh stored in ``mesh_data``.

``ModeSolver1D.solve``
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver1D.solve(*, neff_guess: 'complex | None' = None, num_modes: 'int' = 4, eigensolver_tolerance: 'float' = 1e-10, residual_tolerance: 'float' = 1e-07, dense_limit: 'int' = 450, max_refinements: 'int' = 2, adaptive_tolerance: 'float' = 0.05)

Solve and return modes with separate algebraic and adaptive controls.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``neff_guess``
     - ``complex | None``
     - Optional
     - ``None``
     - Dimensionless complex effective-index search target.
   * - ``num_modes``
     - ``int``
     - Optional
     - ``4``
     - Number of modes requested; positive integer.
   * - ``eigensolver_tolerance``
     - ``float``
     - Optional
     - ``1e-10``
     - Algebraic eigensolver convergence tolerance.
   * - ``residual_tolerance``
     - ``float``
     - Optional
     - ``1e-07``
     - Maximum accepted eigenproblem residual, separate from adaptation.
   * - ``dense_limit``
     - ``int``
     - Optional
     - ``450``
     - Dense limit control for this operation.
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

``ModeSolver1D.show``
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver1D.show(*, block: 'bool' = True)

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

``ModeSolver2D``
~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver2D(*, frequency: 'float', x_range: 'float | Sequence[float]', y_range: 'float | Sequence[float]', background_material: 'materials.Material' = Material(name='vacuum', epsilon=(1+0j), mu=(1+0j)), boundary: 'materials.IdealBoundary' = IdealBoundary(name='PEC', kind='pec')) -> 'None'

Full-vector 2D FEM waveguide mode solver.

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
   * - ``x_range``
     - ``float | Sequence[float]``
     - Required
     - ``—``
     - Physical x extent or increasing bounds, in metres.
   * - ``y_range``
     - ``float | Sequence[float]``
     - Required
     - ``—``
     - Physical y extent or increasing bounds, in metres.
   * - ``background_material``
     - ``materials.Material``
     - Optional
     - ``Material(name='vacuum', epsilon=(1+0j), mu=(1+0j))``
     - Predefined bulk Material assigned to unfilled space.
   * - ``boundary``
     - ``materials.IdealBoundary``
     - Optional
     - ``IdealBoundary(name='PEC', kind='pec')``
     - Predefined PEC or PMC exterior-boundary material.

Returns: a configured ``ModeSolver2D``.

``ModeSolver2D.add_geometry``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver2D.add_geometry(*, shape, material, name=None, clip=False)

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

``ModeSolver2D.add_rectangle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver2D.add_rectangle(*, x_range, y_range, material, name=None, clip=False)

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

``ModeSolver2D.add_circle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver2D.add_circle(*, center, radius, material, name=None, clip=False)

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

``ModeSolver2D.add_polygon``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver2D.add_polygon(*, points, material, name=None, clip=False)

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

``ModeSolver2D.set_material``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver2D.set_material(*, geometry, material)

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

``ModeSolver2D.set_shape``
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver2D.set_shape(*, geometry, shape)

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

``ModeSolver2D.set_boundary``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver2D.set_boundary(*, material)

Set the exterior PEC/PMC wall; internal SIBC uses geometry assignment.

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

``ModeSolver2D.add_mesh_refinement``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver2D.add_mesh_refinement(*, shape: 'Shape2D', max_element_size: 'float', transition_width: 'float' = 0.0, name: 'str | None' = None) -> 'MeshRefinement'

Place a local mesh-size control without changing the physics.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``shape``
     - ``Shape2D``
     - Required
     - ``—``
     - A predefined cem_common.shapes object in metres.
   * - ``max_element_size``
     - ``float``
     - Required
     - ``—``
     - Maximum initial element edge length in metres.
   * - ``transition_width``
     - ``float``
     - Optional
     - ``0.0``
     - Transition width control for this operation.
   * - ``name``
     - ``str | None``
     - Optional
     - ``None``
     - Optional name used for later identification and diagnostics.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``ModeSolver2D.add_pml``
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver2D.add_pml(*, thickness: 'float', order: 'int' = 3, sigma_max: 'float' = 5.0, direction: 'str' = 'all') -> 'PMLSpec'

Add a physical-width uniaxial PML to selected exterior side(s).

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
   * - ``order``
     - ``int``
     - Optional
     - ``3``
     - Polynomial order of the PML profile.
   * - ``sigma_max``
     - ``float``
     - Optional
     - ``5.0``
     - Backend PML-strength magnitude; the outgoing stretch has negative imaginary sign.
   * - ``direction``
     - ``str``
     - Optional
     - ``'all'``
     - Propagation direction for solve; selected coordinate direction for PML.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``ModeSolver2D.remove``
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver2D.remove(*, geometry)

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

``ModeSolver2D.refine``
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver2D.refine(factor: 'float' = 2.0) -> 'FEMMesh2D'

Remesh with ``factor`` times the density and rebuild the system.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``factor``
     - ``float``
     - Optional
     - ``2.0``
     - Factor control for this operation.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``ModeSolver2D.mesh``
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver2D.mesh(*, max_element_size: 'float | None' = None, resolution: 'tuple[int, int] | None' = None, wavelength_elements: 'int' = 4, material_aware: 'bool' = True, interface_refinement: 'float | None' = 0.7, interface_refinement_width: 'float | None' = None, boundary_refinement: 'float | None' = 0.5, boundary_refinement_width: 'float | None' = None, element_order: 'int' = 1, quadrature_order: 'int' = 4)

Build the initial mesh; solve() may subsequently refine it.

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
   * - ``resolution``
     - ``tuple[int, int] | None``
     - Optional
     - ``None``
     - Initial node counts; use instead of a maximum element size.
   * - ``wavelength_elements``
     - ``int``
     - Optional
     - ``4``
     - Minimum number of initial elements per local wavelength.
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
   * - ``interface_refinement_width``
     - ``float | None``
     - Optional
     - ``None``
     - Interface refinement width control for this operation.
   * - ``boundary_refinement``
     - ``float | None``
     - Optional
     - ``0.5``
     - Boundary refinement control for this operation.
   * - ``boundary_refinement_width``
     - ``float | None``
     - Optional
     - ``None``
     - Boundary refinement width control for this operation.
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

Returns: the initial mesh stored in ``mesh_data``.

``ModeSolver2D.solve``
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver2D.solve(*, neff_guess: 'complex | None' = None, num_modes: 'int' = 4, direction: 'Direction' = 'forward', eigensolver_tolerance: 'float' = 1e-10, residual_tolerance: 'float' = 1e-08, divergence_tolerance: 'float' = 1e-07, propagation_ratio_tolerance: 'float' = 0.001, dense_linearization_limit: 'int' = 700, max_refinements: 'int' = 2, adaptive_tolerance: 'float' = 0.05)

Solve and return modes with separate algebraic and adaptive controls.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``neff_guess``
     - ``complex | None``
     - Optional
     - ``None``
     - Dimensionless complex effective-index search target.
   * - ``num_modes``
     - ``int``
     - Optional
     - ``4``
     - Number of modes requested; positive integer.
   * - ``direction``
     - ``Direction``
     - Optional
     - ``'forward'``
     - Propagation direction for solve; selected coordinate direction for PML.
   * - ``eigensolver_tolerance``
     - ``float``
     - Optional
     - ``1e-10``
     - Algebraic eigensolver convergence tolerance.
   * - ``residual_tolerance``
     - ``float``
     - Optional
     - ``1e-08``
     - Maximum accepted eigenproblem residual, separate from adaptation.
   * - ``divergence_tolerance``
     - ``float``
     - Optional
     - ``1e-07``
     - Maximum accepted discrete Gauss-law residual.
   * - ``propagation_ratio_tolerance``
     - ``float``
     - Optional
     - ``0.001``
     - Propagation ratio tolerance control for this operation.
   * - ``dense_linearization_limit``
     - ``int``
     - Optional
     - ``700``
     - Dense linearization limit control for this operation.
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

``ModeSolver2D.show``
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver2D.show(*, block: 'bool' = True)

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

``ModeSet.plot``
~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSet.plot(*, component: 'str | None' = None, quantity: 'str' = 'real', mode: 'int' = 0)

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
     - Zero-based mode index.

Returns: a Matplotlib Figure.

``ModeSet.show``
~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSet.show(*, block: 'bool' = True)

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

``ModeSet.save``
~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSet.save(path)

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

``ModeSet.mode``
~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSet.mode(number: 'int') -> 'Mode'

Return a mode by its zero-based index.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``number``
     - ``int``
     - Required
     - ``—``
     - Zero-based mode index.

Returns: the selected mode.

``ModeSet.by_polarization``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSet.by_polarization(polarization: 'str') -> "'ModeSet'"

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``polarization``
     - ``str``
     - Required
     - ``—``
     - Polarization control for this operation.

Returns: the selected data or diagnostic report.

Result data and diagnostics
---------------------------

``mesh_data.coordinates`` stores physical nodes in metres; ``elements`` stores
zero-based connectivity. ``axes`` identifies physical coordinate order.
``mesh_data.metadata['context']`` records material and boundary configuration.
The result is an inspection snapshot; editing it cannot restart a solver.

Each selected mode exposes ``neff`` (dimensionless), ``beta`` (rad/m),
``fields`` (sampled E/H in V/m and A/m), ``coefficients`` (FEM expansion),
and ``residual`` (algebraic eigenproblem residual). Field component names
include Ex, Ey, Ez, Hx, Hy, and Hz. Coordinate and cell ownership arrays
preserve field locations; they must not be interpreted as interchangeable
nodal or cell values. ``solve_info`` records adaptive history separately.

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
