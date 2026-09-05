fem_periodic_modes user API
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

The time convention is ``exp(+i*omega*t)`` with guided propagation
``exp(-i*beta*z)``. Passive materials have nonpositive imaginary constitutive
values; passive forward attenuation is ``-Im(beta)``.

Materials support scalar or diagonal epsilon and mu. Unsupported off-diagonal
forms raise an explicit configuration error. Periodic archives store periodic
field envelopes; the Bloch phase is recorded separately from these fields.

Supported exports
-----------------

``PeriodicModeSolver2D``, ``PeriodicModeSolver3D``, ``PeriodicMode``, ``PeriodicModeSet``, ``PeriodicSampledFields``, ``PeriodicSweepResult``, ``Material``, ``Rectangle``, ``Circle``, ``Polygon``, ``Box``, ``Sphere``, ``Cylinder``, ``BackendCapabilityError``, ``ConfigurationError``, ``FEMPeriodicSolverError``, ``GeometryError``, ``MeshError``, ``PersistenceError``, ``SolverError``, ``load_result``, ``NoResultError``.

Solver construction and operations
----------------------------------

``PeriodicModeSolver2D``
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver2D(*, frequency: 'float', x_range: 'float | Sequence[float]', z_range: 'float | Sequence[float]', polarization: 'str' = 'both', background_epsilon: 'MaterialInput' = 1.0, background_mu: 'MaterialInput' = 1.0, boundary: 'str' = 'pec') -> 'None'

Solve complex Floquet propagation constants in one ``x-z`` unit cell.

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
   * - ``z_range``
     - ``float | Sequence[float]``
     - Required
     - ``—``
     - Physical z extent or increasing bounds, in metres.
   * - ``polarization``
     - ``str``
     - Optional
     - ``'both'``
     - Polarization control for this operation.
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

Returns: a configured ``PeriodicModeSolver2D``.

``PeriodicModeSolver2D.add_rectangle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver2D.add_rectangle(*, epsilon: 'MaterialInput', mu: 'MaterialInput', x_range: 'Sequence[float]', z_range: 'Sequence[float]', name: 'str | None' = None) -> 'Region'

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
   * - ``z_range``
     - ``Sequence[float]``
     - Required
     - ``—``
     - Physical z extent or increasing bounds, in metres.
   * - ``name``
     - ``str | None``
     - Optional
     - ``None``
     - Optional name used for later identification and diagnostics.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``PeriodicModeSolver2D.add_circle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver2D.add_circle(*, epsilon: 'MaterialInput', mu: 'MaterialInput', center: 'Sequence[float]', radius: 'float', inner_radius: 'float | None' = None, name: 'str | None' = None) -> 'Region'

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
   * - ``name``
     - ``str | None``
     - Optional
     - ``None``
     - Optional name used for later identification and diagnostics.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``PeriodicModeSolver2D.add_polygon``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver2D.add_polygon(*, epsilon: 'MaterialInput', mu: 'MaterialInput', points: 'Sequence[Sequence[float]]', name: 'str | None' = None) -> 'Region'

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

``PeriodicModeSolver2D.add_triangle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver2D.add_triangle(*, epsilon: 'MaterialInput', mu: 'MaterialInput', p1: 'Sequence[float]', p2: 'Sequence[float]', p3: 'Sequence[float]', name: 'str | None' = None) -> 'Region'

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

``PeriodicModeSolver2D.add_mesh_refinement``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver2D.add_mesh_refinement(*, shape: 'Shape2D', max_element_size: 'float', name: 'str | None' = None) -> 'MeshRefinement'

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
   * - ``name``
     - ``str | None``
     - Optional
     - ``None``
     - Optional name used for later identification and diagnostics.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``PeriodicModeSolver2D.add_pec``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver2D.add_pec(*, x_range: 'Sequence[float] | None' = None, z_range: 'Sequence[float] | None' = None, components: 'Sequence[str] | str | None' = None, name: 'str | None' = None) -> 'BoundaryRegion | None'

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
   * - ``z_range``
     - ``Sequence[float] | None``
     - Optional
     - ``None``
     - Physical z extent or increasing bounds, in metres.
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

``PeriodicModeSolver2D.add_pmc``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver2D.add_pmc(*, x_range: 'Sequence[float] | None' = None, z_range: 'Sequence[float] | None' = None, components: 'Sequence[str] | str | None' = None, name: 'str | None' = None) -> 'BoundaryRegion | None'

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
   * - ``z_range``
     - ``Sequence[float] | None``
     - Optional
     - ``None``
     - Physical z extent or increasing bounds, in metres.
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

``PeriodicModeSolver2D.add_pml``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver2D.add_pml(*, thickness: 'float', order: 'int' = 3, sigma_max: 'float' = 5.0, direction: 'str' = 'x') -> 'PMLSpec'

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
     - ``'x'``
     - Propagation direction for solve; selected coordinate direction for PML.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``PeriodicModeSolver2D.set_outer_boundary``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver2D.set_outer_boundary(*, kind: 'str') -> 'None'

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

``PeriodicModeSolver2D.remove``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver2D.remove(handle: 'Region | BoundaryRegion | MeshRefinement | PMLSpec') -> 'None'

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

``PeriodicModeSolver2D.refine``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver2D.refine(factor: 'float' = 2.0) -> 'FEMPeriodicMesh2D'

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

``PeriodicModeSolver2D.solve``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver2D.solve(*, neff_guess: 'complex | None' = None, num_modes: 'int' = 4, direction: 'Direction' = 'forward', eigensolver: 'str' = 'auto', arnoldi_backend: 'str' = 'auto', eigensolver_tolerance: 'float' = 1e-10, residual_tolerance: 'float' = 1e-08, propagation_ratio_tolerance: 'float' = 0.001, max_pml_fraction: 'float | None' = 0.5, dense_linearization_limit: 'int' = 700, max_refinements: 'int' = 2, adaptive_tolerance: 'float' = 0.05)

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
   * - ``eigensolver``
     - ``str``
     - Optional
     - ``'auto'``
     - Eigensolver control for this operation.
   * - ``arnoldi_backend``
     - ``str``
     - Optional
     - ``'auto'``
     - Arnoldi backend control for this operation.
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
   * - ``propagation_ratio_tolerance``
     - ``float``
     - Optional
     - ``0.001``
     - Propagation ratio tolerance control for this operation.
   * - ``max_pml_fraction``
     - ``float | None``
     - Optional
     - ``0.5``
     - Max pml fraction control for this operation.
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

``PeriodicModeSolver2D.mesh``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver2D.mesh(*, max_element_size: 'float | None' = None, resolution: 'tuple[int, int] | None' = None, wavelength_elements: 'int' = 4, element_order: 'int' = 1, quadrature_order: 'int' = 4)

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

``PeriodicModeSolver2D.show``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver2D.show(*, block: 'bool' = True)

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

``PeriodicModeSolver3D``
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver3D(*, frequency: 'float', x_range: 'float | tuple[float, float]', y_range: 'float | tuple[float, float]', z_range: 'float | tuple[float, float]', background_epsilon: 'MaterialInput' = 1.0, background_mu: 'MaterialInput' = 1.0, boundary: 'str' = 'pec') -> 'None'

Solve complex fixed-frequency Bloch propagation in a tetrahedral cell.

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
     - ``float | tuple[float, float]``
     - Required
     - ``—``
     - Physical x extent or increasing bounds, in metres.
   * - ``y_range``
     - ``float | tuple[float, float]``
     - Required
     - ``—``
     - Physical y extent or increasing bounds, in metres.
   * - ``z_range``
     - ``float | tuple[float, float]``
     - Required
     - ``—``
     - Physical z extent or increasing bounds, in metres.
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

Returns: a configured ``PeriodicModeSolver3D``.

``PeriodicModeSolver3D.add_box``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver3D.add_box(*, epsilon: 'MaterialInput', mu: 'MaterialInput', x_range: 'tuple[float, float]', y_range: 'tuple[float, float]', z_range: 'tuple[float, float]', name: 'str | None' = None) -> 'object'

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
     - ``tuple[float, float]``
     - Required
     - ``—``
     - Physical x extent or increasing bounds, in metres.
   * - ``y_range``
     - ``tuple[float, float]``
     - Required
     - ``—``
     - Physical y extent or increasing bounds, in metres.
   * - ``z_range``
     - ``tuple[float, float]``
     - Required
     - ``—``
     - Physical z extent or increasing bounds, in metres.
   * - ``name``
     - ``str | None``
     - Optional
     - ``None``
     - Optional name used for later identification and diagnostics.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``PeriodicModeSolver3D.add_cylinder``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver3D.add_cylinder(*, epsilon: 'MaterialInput', mu: 'MaterialInput', center: 'tuple[float, float]', radius: 'float', z_range: 'tuple[float, float]', name: 'str | None' = None) -> 'object'

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
     - ``tuple[float, float]``
     - Required
     - ``—``
     - Physical centre coordinates in metres.
   * - ``radius``
     - ``float``
     - Required
     - ``—``
     - Positive radius in metres.
   * - ``z_range``
     - ``tuple[float, float]``
     - Required
     - ``—``
     - Physical z extent or increasing bounds, in metres.
   * - ``name``
     - ``str | None``
     - Optional
     - ``None``
     - Optional name used for later identification and diagnostics.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``PeriodicModeSolver3D.add_sphere``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver3D.add_sphere(*, epsilon: 'MaterialInput', mu: 'MaterialInput', center: 'tuple[float, float, float]', radius: 'float', name: 'str | None' = None) -> 'object'

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
     - ``tuple[float, float, float]``
     - Required
     - ``—``
     - Physical centre coordinates in metres.
   * - ``radius``
     - ``float``
     - Required
     - ``—``
     - Positive radius in metres.
   * - ``name``
     - ``str | None``
     - Optional
     - ``None``
     - Optional name used for later identification and diagnostics.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``PeriodicModeSolver3D.add_pec``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver3D.add_pec(*, shape: 'Shape3D', name: 'str | None' = None) -> 'object'

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``shape``
     - ``Shape3D``
     - Required
     - ``—``
     - Shape control for this operation.
   * - ``name``
     - ``str | None``
     - Optional
     - ``None``
     - Optional name used for later identification and diagnostics.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``PeriodicModeSolver3D.add_pmc``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver3D.add_pmc(*, shape: 'Shape3D', name: 'str | None' = None) -> 'object'

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``shape``
     - ``Shape3D``
     - Required
     - ``—``
     - Shape control for this operation.
   * - ``name``
     - ``str | None``
     - Optional
     - ``None``
     - Optional name used for later identification and diagnostics.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``PeriodicModeSolver3D.add_pml``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver3D.add_pml(*, thickness: 'float', order: 'int' = 3, sigma_max: 'float' = 5.0, direction: 'str' = 'all') -> 'object'

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

``PeriodicModeSolver3D.add_mesh_refinement``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver3D.add_mesh_refinement(*, shape: 'Shape3D', max_element_size: 'float', name: 'str | None' = None) -> 'object'

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``shape``
     - ``Shape3D``
     - Required
     - ``—``
     - Shape control for this operation.
   * - ``max_element_size``
     - ``float``
     - Required
     - ``—``
     - Maximum initial element edge length in metres.
   * - ``name``
     - ``str | None``
     - Optional
     - ``None``
     - Optional name used for later identification and diagnostics.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``PeriodicModeSolver3D.remove``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver3D.remove(handle: 'object') -> 'None'

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``handle``
     - ``object``
     - Required
     - ``—``
     - Handle control for this operation.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``PeriodicModeSolver3D.set_outer_boundary``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver3D.set_outer_boundary(*, kind: 'str') -> 'None'

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

``PeriodicModeSolver3D.refine``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver3D.refine(factor: 'float' = 2.0) -> 'PeriodicMesh3D'

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

``PeriodicModeSolver3D.solve``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver3D.solve(*, neff_guess: 'complex | None' = None, num_modes: 'int' = 4, direction: 'Direction' = 'forward', eigensolver: 'str' = 'auto', arnoldi_backend: 'str' = 'auto', eigensolver_tolerance: 'float' = 1e-10, residual_tolerance: 'float' = 1e-08, divergence_tolerance: 'float' = 1e-06, propagation_ratio_tolerance: 'float' = 0.001, max_pml_fraction: 'float | None' = 0.5, dense_linearization_limit: 'int' = 700, ncv: 'int | None' = None, max_restarts: 'int' = 12, random_seed: 'int' = 0, max_refinements: 'int' = 2, adaptive_tolerance: 'float' = 0.05)

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
   * - ``eigensolver``
     - ``str``
     - Optional
     - ``'auto'``
     - Eigensolver control for this operation.
   * - ``arnoldi_backend``
     - ``str``
     - Optional
     - ``'auto'``
     - Arnoldi backend control for this operation.
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
     - ``1e-06``
     - Maximum accepted discrete Gauss-law residual.
   * - ``propagation_ratio_tolerance``
     - ``float``
     - Optional
     - ``0.001``
     - Propagation ratio tolerance control for this operation.
   * - ``max_pml_fraction``
     - ``float | None``
     - Optional
     - ``0.5``
     - Max pml fraction control for this operation.
   * - ``dense_linearization_limit``
     - ``int``
     - Optional
     - ``700``
     - Dense linearization limit control for this operation.
   * - ``ncv``
     - ``int | None``
     - Optional
     - ``None``
     - Ncv control for this operation.
   * - ``max_restarts``
     - ``int``
     - Optional
     - ``12``
     - Max restarts control for this operation.
   * - ``random_seed``
     - ``int``
     - Optional
     - ``0``
     - Random seed control for this operation.
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

``PeriodicModeSolver3D.mesh``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver3D.mesh(*, max_element_size: 'float | None' = None, wavelength_elements: 'int' = 4, material_aware: 'bool' = True, quadrature_order: 'int' = 3)

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
   * - ``quadrature_order``
     - ``int``
     - Optional
     - ``3``
     - Element integration order.

Returns: the initial mesh stored in ``mesh_data``.

``PeriodicModeSolver3D.show``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver3D.show(*, block: 'bool' = True)

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

``PeriodicModeSet.plot``
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSet.plot(*, component: 'str | None' = None, quantity: 'str' = 'real', mode: 'int' = 0)

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

``PeriodicModeSet.show``
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSet.show(*, block: 'bool' = True)

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

Returns: the written Path.

``PeriodicModeSet.mode``
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSet.mode(number: 'int') -> 'PeriodicMode'

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

``PeriodicModeSet.by_polarization``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSet.by_polarization(polarization: 'str') -> 'tuple[PeriodicMode, ...]'

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

``PeriodicSweepResult.from_results``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicSweepResult.from_results(results)

Combine solved periodic mode sets into a frequency or parameter sweep.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``results``
     - ``Sequence[PeriodicModeSet]``
     - Required
     - ``—``
     - Nonempty sequence of completed periodic mode sets, in sweep order.

Returns: the selected data or diagnostic report.

``PeriodicSweepResult.plot``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicSweepResult.plot(*, case=0, component='Ey', quantity='real', mode=0)

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``case``
     - ``int``
     - Optional
     - ``0``
     - Zero-based sweep case index.
   * - ``component``
     - ``str | None``
     - Optional
     - ``'Ey'``
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

``PeriodicSweepResult.show``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicSweepResult.show(*, block=True)

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

``PeriodicSweepResult.save``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicSweepResult.save(path)

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

Periodic fields are Bloch envelopes. ``period`` is in m; each mode also
provides ``gamma``, ``bloch_multiplier``, folded propagation quantities,
and Gauss-law/PML filtering diagnostics. Combine solved cases with
``PeriodicSweepResult.from_results(results)`` and call ``save(path)``.
Loaded multi-case archives index cases lazily.

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

``Rectangle``
~~~~~~~~~~~~~

.. code-block:: python

    Rectangle(x: 'tuple[float, float]', z: 'tuple[float, float]') -> None

Rectangle(x: 'tuple[float, float]', z: 'tuple[float, float]')

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
   * - ``z``
     - ``tuple[float, float]``
     - Required
     - ``—``
     - Z control for this operation.

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

``Box``
~~~~~~~

.. code-block:: python

    Box(x: 'tuple[float, float]', y: 'tuple[float, float]', z: 'tuple[float, float]') -> None

Box(x: 'tuple[float, float]', y: 'tuple[float, float]', z: 'tuple[float, float]')

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
   * - ``z``
     - ``tuple[float, float]``
     - Required
     - ``—``
     - Z control for this operation.

Returns: an immutable geometry/material value for solver configuration.

``Sphere``
~~~~~~~~~~

.. code-block:: python

    Sphere(center: 'tuple[float, float, float]', radius: 'float') -> None

Sphere(center: 'tuple[float, float, float]', radius: 'float')

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``center``
     - ``tuple[float, float, float]``
     - Required
     - ``—``
     - Physical centre coordinates in metres.
   * - ``radius``
     - ``float``
     - Required
     - ``—``
     - Positive radius in metres.

Returns: an immutable geometry/material value for solver configuration.

``Cylinder``
~~~~~~~~~~~~

.. code-block:: python

    Cylinder(center: 'tuple[float, float]', radius: 'float', z: 'tuple[float, float]') -> None

Cylinder(center: 'tuple[float, float]', radius: 'float', z: 'tuple[float, float]')

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
   * - ``z``
     - ``tuple[float, float]``
     - Required
     - ``—``
     - Z control for this operation.

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
