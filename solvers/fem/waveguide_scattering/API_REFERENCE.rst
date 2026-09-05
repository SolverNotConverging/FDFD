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

``WaveguideScatteringSolver2D``, ``ScatteringResult``, ``FrequencySweepResult``, ``IncidentMode``, ``Mode``, ``ModeSet``, ``Diagnostic``, ``DiagnosticReport``, ``ConfigurationError``, ``MaterialError``, ``MeshError``, ``ModeProjectionError``, ``ModeSolverError``, ``SolverError``, ``ViewerError``, ``load_result``, ``NoResultError``, ``PersistenceError``.

Solver construction and operations
----------------------------------

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
