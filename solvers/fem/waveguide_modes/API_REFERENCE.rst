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

``ModeSolver1D``, ``ModeSolver2D``, ``Mode``, ``ModeSet``, ``SampledFields``, ``Material``, ``Interval``, ``Rectangle``, ``Circle``, ``Polygon``, ``good_conductor_surface_impedance``, ``BackendCapabilityError``, ``ConfigurationError``, ``FEMModeSolverError``, ``GeometryError``, ``MeshError``, ``SolverError``, ``load_result``, ``NoResultError``, ``PersistenceError``.

Solver construction and operations
----------------------------------

``ModeSolver1D``
~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver1D(*, frequency: 'float', x_range: 'float | Sequence[float]', background_epsilon: 'MaterialInput' = 1.0, background_mu: 'MaterialInput' = 1.0) -> 'None'

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
   * - ``background_epsilon``
     - ``MaterialInput``
     - Optional
     - ``1.0``
     - Relative permittivity of the unfilled domain.
   * - ``background_mu``
     - ``MaterialInput``
     - Optional
     - ``1.0``
     - Relative permeability of the unfilled domain.

Returns: a configured ``ModeSolver1D``.

``ModeSolver1D.add_layer``
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver1D.add_layer(*, epsilon: 'MaterialInput', mu: 'MaterialInput', x_range: 'Sequence[float]', name: 'str | None' = None) -> 'Region'

Place a material interval and return its geometry handle.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``epsilon``
     - ``MaterialInput``
     - Required
     - ``—``
     - Relative permittivity; supported scalar/tensor forms are described below.
   * - ``mu``
     - ``MaterialInput``
     - Required
     - ``—``
     - Relative permeability; supported scalar/diagonal forms are described below.
   * - ``x_range``
     - ``Sequence[float]``
     - Required
     - ``—``
     - Physical x extent or increasing bounds, in metres.
   * - ``name``
     - ``str | None``
     - Optional
     - ``None``
     - Optional name used for later identification and diagnostics.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``ModeSolver1D.add_pec``
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver1D.add_pec(*, x_range: 'Sequence[float] | None' = None, components: 'object | None' = None, name: 'str | None' = None) -> 'BoundaryRegion | None'

Add an opaque PEC interval, or set both outer walls to PEC.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``x_range``
     - ``Sequence[float] | None``
     - Optional
     - ``None``
     - Physical x extent or increasing bounds, in metres.
   * - ``components``
     - ``object | None``
     - Optional
     - ``None``
     - Components control for this operation.
   * - ``name``
     - ``str | None``
     - Optional
     - ``None``
     - Optional name used for later identification and diagnostics.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``ModeSolver1D.add_pmc``
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver1D.add_pmc(*, x_range: 'Sequence[float] | None' = None, components: 'object | None' = None, name: 'str | None' = None) -> 'BoundaryRegion | None'

Add an opaque PMC interval, or set both outer walls to PMC.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``x_range``
     - ``Sequence[float] | None``
     - Optional
     - ``None``
     - Physical x extent or increasing bounds, in metres.
   * - ``components``
     - ``object | None``
     - Optional
     - ``None``
     - Components control for this operation.
   * - ``name``
     - ``str | None``
     - Optional
     - ``None``
     - Optional name used for later identification and diagnostics.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``ModeSolver1D.add_impedance_surface``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver1D.add_impedance_surface(*, Zs: 'complex | None' = None, preset: 'str | None' = None, x_range: 'Sequence[float]', name: 'str | None' = None) -> 'BoundaryRegion'

Record a scalar impedance object.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``Zs``
     - ``complex | None``
     - Optional
     - ``None``
     - Surface impedance in ohms; alternatively select a metal preset.
   * - ``preset``
     - ``str | None``
     - Optional
     - ``None``
     - Metal name for the good-conductor impedance model.
   * - ``x_range``
     - ``Sequence[float]``
     - Required
     - ``—``
     - Physical x extent or increasing bounds, in metres.
   * - ``name``
     - ``str | None``
     - Optional
     - ``None``
     - Optional name used for later identification and diagnostics.

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

``ModeSolver1D.set_outer_boundary``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver1D.set_outer_boundary(*, kind: 'str') -> 'None'

Set both transverse truncation walls to ``'pec'`` or ``'pmc'``.

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
     - Required
     - ``—``
     - Kind control for this operation.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``ModeSolver1D.remove``
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver1D.remove(handle: 'Region | BoundaryRegion | PMLSpec') -> 'None'

Remove a previously returned geometry handle.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``handle``
     - ``Region | BoundaryRegion | PMLSpec``
     - Required
     - ``—``
     - Handle control for this operation.

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

    ModeSolver2D(*, frequency: 'float', x_range: 'float | Sequence[float]', y_range: 'float | Sequence[float]', background_epsilon: 'MaterialInput' = 1.0, background_mu: 'MaterialInput' = 1.0, boundary: 'str' = 'pec') -> 'None'

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
   * - ``background_epsilon``
     - ``MaterialInput``
     - Optional
     - ``1.0``
     - Relative permittivity of the unfilled domain.
   * - ``background_mu``
     - ``MaterialInput``
     - Optional
     - ``1.0``
     - Relative permeability of the unfilled domain.
   * - ``boundary``
     - ``str``
     - Optional
     - ``'pec'``
     - Exterior boundary condition.

Returns: a configured ``ModeSolver2D``.

``ModeSolver2D.add_rectangle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver2D.add_rectangle(*, epsilon: 'MaterialInput', mu: 'MaterialInput', x_range: 'Sequence[float]', y_range: 'Sequence[float]', name: 'str | None' = None) -> 'Region'

Place a conformingly meshed rectangular material region.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``epsilon``
     - ``MaterialInput``
     - Required
     - ``—``
     - Relative permittivity; supported scalar/tensor forms are described below.
   * - ``mu``
     - ``MaterialInput``
     - Required
     - ``—``
     - Relative permeability; supported scalar/diagonal forms are described below.
   * - ``x_range``
     - ``Sequence[float]``
     - Required
     - ``—``
     - Physical x extent or increasing bounds, in metres.
   * - ``y_range``
     - ``Sequence[float]``
     - Required
     - ``—``
     - Physical y extent or increasing bounds, in metres.
   * - ``name``
     - ``str | None``
     - Optional
     - ``None``
     - Optional name used for later identification and diagnostics.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``ModeSolver2D.add_circle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver2D.add_circle(*, epsilon: 'MaterialInput', mu: 'MaterialInput', center: 'Sequence[float]', r1: 'float', r2: 'float | None' = None, name: 'str | None' = None) -> 'Region'

Place a circular or annular material region.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``epsilon``
     - ``MaterialInput``
     - Required
     - ``—``
     - Relative permittivity; supported scalar/tensor forms are described below.
   * - ``mu``
     - ``MaterialInput``
     - Required
     - ``—``
     - Relative permeability; supported scalar/diagonal forms are described below.
   * - ``center``
     - ``Sequence[float]``
     - Required
     - ``—``
     - Physical centre coordinates in metres.
   * - ``r1``
     - ``float``
     - Required
     - ``—``
     - R1 control for this operation.
   * - ``r2``
     - ``float | None``
     - Optional
     - ``None``
     - R2 control for this operation.
   * - ``name``
     - ``str | None``
     - Optional
     - ``None``
     - Optional name used for later identification and diagnostics.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``ModeSolver2D.add_polygon``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver2D.add_polygon(*, epsilon: 'MaterialInput', mu: 'MaterialInput', points: 'Sequence[Sequence[float]]', name: 'str | None' = None) -> 'Region'

Place an arbitrary simple polygon material region.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``epsilon``
     - ``MaterialInput``
     - Required
     - ``—``
     - Relative permittivity; supported scalar/tensor forms are described below.
   * - ``mu``
     - ``MaterialInput``
     - Required
     - ``—``
     - Relative permeability; supported scalar/diagonal forms are described below.
   * - ``points``
     - ``Sequence[Sequence[float]]``
     - Required
     - ``—``
     - Ordered polygon vertex coordinates in metres.
   * - ``name``
     - ``str | None``
     - Optional
     - ``None``
     - Optional name used for later identification and diagnostics.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``ModeSolver2D.add_triangle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver2D.add_triangle(*, epsilon: 'MaterialInput', mu: 'MaterialInput', p1: 'Sequence[float]', p2: 'Sequence[float]', p3: 'Sequence[float]', name: 'str | None' = None) -> 'Region'

Place a triangular material region.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``epsilon``
     - ``MaterialInput``
     - Required
     - ``—``
     - Relative permittivity; supported scalar/tensor forms are described below.
   * - ``mu``
     - ``MaterialInput``
     - Required
     - ``—``
     - Relative permeability; supported scalar/diagonal forms are described below.
   * - ``p1``
     - ``Sequence[float]``
     - Required
     - ``—``
     - P1 control for this operation.
   * - ``p2``
     - ``Sequence[float]``
     - Required
     - ``—``
     - P2 control for this operation.
   * - ``p3``
     - ``Sequence[float]``
     - Required
     - ``—``
     - P3 control for this operation.
   * - ``name``
     - ``str | None``
     - Optional
     - ``None``
     - Optional name used for later identification and diagnostics.

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
     - Shape control for this operation.
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

``ModeSolver2D.add_pec``
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver2D.add_pec(*, x_range: 'Sequence[float] | None' = None, y_range: 'Sequence[float] | None' = None, components: 'Sequence[str] | str | None' = None, name: 'str | None' = None) -> 'BoundaryRegion | None'

Add an internal PEC object, or select PEC for the outer wall.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``x_range``
     - ``Sequence[float] | None``
     - Optional
     - ``None``
     - Physical x extent or increasing bounds, in metres.
   * - ``y_range``
     - ``Sequence[float] | None``
     - Optional
     - ``None``
     - Physical y extent or increasing bounds, in metres.
   * - ``components``
     - ``Sequence[str] | str | None``
     - Optional
     - ``None``
     - Components control for this operation.
   * - ``name``
     - ``str | None``
     - Optional
     - ``None``
     - Optional name used for later identification and diagnostics.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``ModeSolver2D.add_pmc``
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver2D.add_pmc(*, x_range: 'Sequence[float] | None' = None, y_range: 'Sequence[float] | None' = None, components: 'Sequence[str] | str | None' = None, name: 'str | None' = None) -> 'BoundaryRegion | None'

Add an internal PMC object, or select PMC for the outer wall.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``x_range``
     - ``Sequence[float] | None``
     - Optional
     - ``None``
     - Physical x extent or increasing bounds, in metres.
   * - ``y_range``
     - ``Sequence[float] | None``
     - Optional
     - ``None``
     - Physical y extent or increasing bounds, in metres.
   * - ``components``
     - ``Sequence[str] | str | None``
     - Optional
     - ``None``
     - Components control for this operation.
   * - ``name``
     - ``str | None``
     - Optional
     - ``None``
     - Optional name used for later identification and diagnostics.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``ModeSolver2D.add_impedance_surface``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver2D.add_impedance_surface(*, Zs: 'complex | None' = None, preset: 'str | None' = None, x_range: 'Sequence[float]', y_range: 'Sequence[float]', name: 'str | None' = None) -> 'BoundaryRegion'

Add an opaque conductor whose exposed facets obey scalar SIBC.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``Zs``
     - ``complex | None``
     - Optional
     - ``None``
     - Surface impedance in ohms; alternatively select a metal preset.
   * - ``preset``
     - ``str | None``
     - Optional
     - ``None``
     - Metal name for the good-conductor impedance model.
   * - ``x_range``
     - ``Sequence[float]``
     - Required
     - ``—``
     - Physical x extent or increasing bounds, in metres.
   * - ``y_range``
     - ``Sequence[float]``
     - Required
     - ``—``
     - Physical y extent or increasing bounds, in metres.
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

``ModeSolver2D.set_outer_boundary``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver2D.set_outer_boundary(*, kind: 'str') -> 'None'

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
     - Required
     - ``—``
     - Kind control for this operation.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``ModeSolver2D.remove``
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver2D.remove(handle: 'Region | BoundaryRegion | MeshRefinement | PMLSpec') -> 'None'

Remove a placed object and invalidate any existing mesh.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``handle``
     - ``Region | BoundaryRegion | MeshRefinement | PMLSpec``
     - Required
     - ``—``
     - Handle control for this operation.

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

``Material``
~~~~~~~~~~~~

.. code-block:: python

    Material(epsilon: 'MaterialInput' = 1.0, mu: 'MaterialInput' = 1.0) -> 'None'

Relative diagonal permittivity and permeability.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``epsilon``
     - ``MaterialInput``
     - Optional
     - ``1.0``
     - Relative permittivity; supported scalar/tensor forms are described below.
   * - ``mu``
     - ``MaterialInput``
     - Optional
     - ``1.0``
     - Relative permeability; supported scalar/diagonal forms are described below.

Returns: an immutable geometry/material value for solver configuration.

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

    Circle(center: 'tuple[float, float]', radius: 'float', inner_radius: 'float | None' = None) -> None

Circle(center: 'tuple[float, float]', radius: 'float', inner_radius: 'float | None' = None)

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
   * - ``inner_radius``
     - ``float | None``
     - Optional
     - ``None``
     - Inner radius control for this operation.

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

``good_conductor_surface_impedance``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    good_conductor_surface_impedance(metal: 'str', frequency: 'float', *, relative_permeability: 'float' = 1.0) -> 'complex'

Return ``(1+i)*sqrt(pi*f*mu0*mu_r*rho)`` for ``exp(+iwt)``.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``metal``
     - ``str``
     - Required
     - ``—``
     - Metal control for this operation.
   * - ``frequency``
     - ``float``
     - Required
     - ``—``
     - Operating frequency in hertz; finite and positive.
   * - ``relative_permeability``
     - ``float``
     - Optional
     - ``1.0``
     - Relative permeability control for this operation.

Returns: the passive surface impedance in ohms.

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
