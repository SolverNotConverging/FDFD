WaveFEM API reference
=====================

This document describes the complete supported WaveFEM API in version ``0.0.1``. It covers the convenience names exported from ``import wavefem as wf`` and the lower-level research interfaces available from individual ``wavefem.*`` modules.

The high-level API is the recommended interface for scattering simulations. The lower-level interfaces expose meshes, mixed finite-element systems, equivalent sources, monitor traces, and modal projectors for validation and research workflows.

Conventions used by every API
-----------------------------

Coordinates, fields, and phasors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* The computational plane is ``(x, z)``.

* The invariant/Fourier direction is ``y``.

* Field components are always ordered ``(x, y, z)``.

* Time dependence is ``exp(-i*omega*t)``.

* Fourier and guided-wave dependence is ``exp(+i*ky*y + i*beta*z)``.

* Therefore ``partial_y`` is replaced by ``i*ky``.

* The magnetic field follows ``curl(E) = i*omega*mu*H``.

Units
~~~~~

.. list-table::
   :header-rows: 1

   * - Quantity
     - Public unit
   * - ``x``, ``z``, spans, radii, PML thickness, reference planes
     - metre
   * - ordinary frequency
     - hertz
   * - ``omega``
     - radian per second
   * - ``k0``, ``ky``, ``beta``
     - radian per metre
   * - relative permittivity and permeability
     - dimensionless
   * - ``neff = beta/k0``
     - dimensionless
   * - modal and port power
     - watt per metre of invariant ``y`` length
   * - S-parameters and power ratios
     - dimensionless

Exactly one of ``frequency``, ``omega``, or ``wavelength`` must be supplied to each frequency-selecting constructor. Ordinary ``frequency`` in hertz is the preferred public input. ``omega`` and ``wavelength`` are retained as compatibility inputs, and they resolve to the same immutable ``Frequency`` object.

Array shapes
~~~~~~~~~~~~

* A field evaluated at ``N`` points has shape ``(3, N)``.

* A field evaluated on an arbitrary NumPy-shaped coordinate array has shape ``(3, *coordinate_shape)``.

* ``Mode.E_x`` is cellwise with ``Ncell`` entries.

* ``Mode.E_y`` and ``Mode.E_z`` are nodal with ``Nnode = Ncell + 1`` entries.

* Result coordinates have shape ``(2, N)`` in ``(x, z)`` order.

Recommended high-level call order
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   sim = wf.Scattering2D(...)
   sim.add_rectangle(...)      # define background guide and perturbations
   sim.add_pml(...)
   sim.set_monitors(...)       # optional; otherwise chosen automatically
   sim.mesh(...)
   modes = sim.solve_modes(max_refinements=0, ...)
   sim.set_incident_mode(modes[0])
   result = sim.run(max_refinements=0, h5_path="wavefem_result.h5")

Several methods deliberately invalidate later stages:

.. list-table::
   :header-rows: 1

   * - Call
     - State invalidated
   * - ``add_rectangle``, ``add_circle``, ``add_polygon``, ``add_pml``
     - mesh, modes, incident mode
   * - ``set_monitors``
     - mesh
   * - ``mesh``
     - modes and incident mode
   * - ``solve_modes`` or ``set_modes``
     - incident mode

Consequently, define geometry and PMLs before meshing, and always select the incident mode after the final mesh and mode solve.

Top-level API index
-------------------

The following names are available directly from ``wavefem``:

.. list-table::
   :header-rows: 1

   * - Area
     - Names
   * - Scattering
     - ``Scattering2D``, ``SolverOptions``, ``ScatteringResult``, ``FrequencySweepResult``
   * - Modes
     - ``CrossSection``, ``ModeSolver``, ``ModeSet``, ``Mode``, ``IncidentMode``
   * - Materials and PML
     - ``Material``, ``PML``, ``PMLLayout``
   * - Visualization scene
     - ``Scene2D``, ``SceneLine``
   * - Frequency
     - ``Frequency``, ``resolve_frequency``
   * - HDF5
     - ``H5FileData``, ``H5ResultData``, ``H5ModeData``, ``load_h5``, ``save_result_h5``, ``save_sweep_h5``
   * - Native viewer
     - ``find_viewer_executable``, ``launch_viewer``
   * - Diagnostics
     - ``Diagnostic``, ``DiagnosticReport``
   * - Constants
     - ``C0``, ``EPSILON_0``, ``MU_0``, ``ETA_0``
   * - Exceptions
     - ``WaveFEMError`` and its specialized subclasses

High-level scattering API
-------------------------

``Scattering2D``
~~~~~~~~~~~~~~~~

.. code-block:: python

   wf.Scattering2D(
       *,
       frequency: float | None = None,
       omega: float | None = None,
       wavelength: float | None = None,
       angle: float | None = None,
       ky: float | None = None,
       x_span: Sequence[float],
       z_span: Sequence[float],
       background_eps: complex | float = 1.0,
       background_mu: complex | float = 1.0,
       transverse_boundary: Literal["pec", "pmc"] | None = None,
       solver_options: wf.SolverOptions | None = None,
   )

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``frequency``
     - Conditional: exactly one frequency input
     - ``float | None``
     - Ordinary frequency in Hz; must be finite and positive. Keyword-only. Default: ``None``.
   * - ``omega``
     - Conditional: exactly one frequency input
     - ``float | None``
     - Angular frequency in rad/s; must be finite and positive. Keyword-only. Default: ``None``.
   * - ``wavelength``
     - Conditional: exactly one frequency input
     - ``float | None``
     - Vacuum wavelength in metres; must be finite and positive. Keyword-only. Default: ``None``.
   * - ``angle``
     - Optional
     - ``float | None``
     - Propagation angle in degrees. The integrated lead solver resolves ky from the selected forward modal family. Keyword-only. Default: ``None``.
   * - ``ky``
     - Optional
     - ``float | None``
     - Real Fourier wavenumber along invariant y, in rad/m; zero gives propagation in the computational plane. Keyword-only. Default: ``None``.
   * - ``x_span``
     - Required
     - ``Sequence[float]``
     - Physical axis bounds in metres, with upper > lower. Range-based solver constructors also accept a positive extent for (0, extent). Keyword-only.
   * - ``z_span``
     - Required
     - ``Sequence[float]``
     - Physical axis bounds in metres, with upper > lower. Range-based solver constructors also accept a positive extent for (0, extent). Keyword-only.
   * - ``background_eps``
     - Optional
     - ``complex | float``
     - Relative permittivity. Scalar or Cartesian diagonal material inputs are supported where the signature permits; callbacks return scalar/broadcast-compatible values at physical coordinates. Keyword-only. Default: ``1.0``.
   * - ``background_mu``
     - Optional
     - ``complex | float``
     - Relative permeability, scalar or Cartesian diagonal where supported. WaveFEM scattering requires equal actual and background permeability. Keyword-only. Default: ``1.0``.
   * - ``transverse_boundary``
     - Optional
     - ``Literal['pec', 'pmc'] | None``
     - Outer transverse boundary condition. Modal solvers support PEC/PMC as documented; integrated WaveFEM supports PEC or transverse PML terminated by PEC. Keyword-only. Default: ``None``.
   * - ``solver_options``
     - Optional
     - ``SolverOptions | None``
     - Configuration object or keyword mapping forwarded to mode solving, meshing, or scattering. Set max_refinements=0 in SolverOptions and mode_options for fixed-mesh sweep examples. Keyword-only. Default: ``None``.

Creates a full-vector scattered-field simulation. The domain is the supplied rectangular ``x-z`` interval. ``background_eps`` and ``background_mu`` describe the exterior material of both the actual device and the unperturbed guide.

Parameters:

* ``frequency``: preferred ordinary frequency in hertz. It must be positive, finite, and real.

* ``omega``: compatibility alternative in radians per second.

* ``wavelength``: compatibility alternative vacuum wavelength in metres. Exactly one of the three spectral arguments must be supplied.

* ``angle``: propagation angle in degrees, measured from +z toward +y and restricted to ``(-90, 90)``. ``None`` defaults to normal propagation. For a nonzero angle, ``solve_modes`` first finds the normal-incidence modal families; ``set_incident_mode`` then resolves the selected family with ``ky = k0 * neff_total * sin(angle)``.

* ``ky``: compatibility input for a directly prescribed finite real invariant-direction wavenumber. It is mutually exclusive with ``angle``; new integrated-solver code should use ``angle``.

* ``x_span``, ``z_span``: two finite, strictly increasing coordinates.

* ``background_eps``, ``background_mu``: scalar relative constitutive values. They may be complex with nonnegative imaginary parts for passive loss.

* ``transverse_boundary``: ``"pec"`` for a closed transverse guide, or ``None`` when an x-directed PML will be added. ``"pmc"`` is reserved but currently raises ``NotImplementedError`` in ``Scattering2D``.

* ``solver_options``: optional ``SolverOptions`` instance.

Important public attributes include ``frequency``, ``angle``, derived ``ky``, ``geometry``, ``pml``, ``mesh_data``, ``modes``, ``incident``, ``left_monitor``, and ``right_monitor``. For a nonzero angle, ``ky`` remains zero until an incident family is selected. The latter four object attributes are ``None`` until the corresponding workflow stage completes.

The integrated API accepts passive physical materials, requires lossless uniform leads for modal power projection, supports compact material loss, and supports three perturbation types: volume permittivity contrast, finite slots released from z-invariant zero-thickness background PEC sheets, and finite actual-only constant-x PEC plates.

``Scattering2D.from_material_function``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   Scattering2D.from_material_function(
       *,
       frequency: float | None = None,
       omega: float | None = None,
       wavelength: float | None = None,
       angle: float | None = None,
       ky: float | None = None,
       domain: tuple[Sequence[float], Sequence[float]],
       eps_r: Callable,
       eps_background: Callable,
       transverse_boundary: Literal["pec", "pmc"] | None = None,
       solver_options: wf.SolverOptions | None = None,
   ) -> Scattering2D

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``frequency``
     - Conditional: exactly one frequency input
     - ``float | None``
     - Ordinary frequency in Hz; must be finite and positive. Keyword-only. Default: ``None``.
   * - ``omega``
     - Conditional: exactly one frequency input
     - ``float | None``
     - Angular frequency in rad/s; must be finite and positive. Keyword-only. Default: ``None``.
   * - ``wavelength``
     - Conditional: exactly one frequency input
     - ``float | None``
     - Vacuum wavelength in metres; must be finite and positive. Keyword-only. Default: ``None``.
   * - ``angle``
     - Optional
     - ``float | None``
     - Propagation angle in degrees. The integrated lead solver resolves ky from the selected forward modal family. Keyword-only. Default: ``None``.
   * - ``ky``
     - Optional
     - ``float | None``
     - Real Fourier wavenumber along invariant y, in rad/m; zero gives propagation in the computational plane. Keyword-only. Default: ``None``.
   * - ``domain``
     - Required
     - ``tuple[Sequence[float], Sequence[float]]``
     - Physical domain bounds. Electrostatics: one interval in 1D or a pair of intervals in 2D. WaveFEM callbacks: (x_span, z_span). Keyword-only.
   * - ``eps_r``
     - Required
     - ``MaterialFunction``
     - Relative permittivity. Scalar or Cartesian diagonal material inputs are supported where the signature permits; callbacks return scalar/broadcast-compatible values at physical coordinates. Keyword-only.
   * - ``eps_background``
     - Required
     - ``MaterialFunction``
     - Relative-permittivity callback for the lossless z-invariant unperturbed lead; actual contrast must be compact and bracketed by the monitors. Keyword-only.
   * - ``transverse_boundary``
     - Optional
     - ``Literal['pec', 'pmc'] | None``
     - Outer transverse boundary condition. Modal solvers support PEC/PMC as documented; integrated WaveFEM supports PEC or transverse PML terminated by PEC. Keyword-only. Default: ``None``.
   * - ``solver_options``
     - Optional
     - ``SolverOptions | None``
     - Configuration object or keyword mapping forwarded to mode solving, meshing, or scattering. Set max_refinements=0 in SolverOptions and mode_options for fixed-mesh sweep examples. Keyword-only. Default: ``None``.

Creates a device from actual and unperturbed relative-permittivity callbacks. ``domain`` is ``(x_span, z_span)``. Callbacks receive physical SI coordinates and should normally accept ``callback(x, z)``. A one-argument ``callback(x)`` is also accepted for a z-invariant profile. Scalar or broadcast-compatible array outputs are allowed.

``eps_r`` defines the actual device. ``eps_background`` must define the z-invariant unperturbed lead used by the equivalent source. Callback devices use ``mu_r = 1`` and cannot be mixed with geometry primitives.

Automatic cross-section inference is intentionally disabled for callbacks. Create a compatible ``CrossSection``, solve it with ``ModeSolver``, then call ``set_modes`` after meshing:

Callback-defined simulations currently support ``angle=0`` or an explicitly prescribed compatibility ``ky``. Resolving a nonzero angle requires the integrated geometry-backed lead so WaveFEM can re-solve the selected modal family.

.. code-block:: python

   import numpy as np

   sim = wf.Scattering2D.from_material_function(
       frequency=193.414489e12,
       domain=((0.0, 1.0e-6), (-3.0e-6, 3.0e-6)),
       eps_r=lambda x, z: np.where(np.abs(z) <= 0.3e-6, 1.002, 1.0)
       + 0.0 * np.asarray(x),
       eps_background=lambda x: np.ones_like(x, dtype=np.complex128),
       transverse_boundary="pec",
   )
   sim.add_pml(z=0.8e-6)
   sim.set_monitors(left=-1.0e-6, right=1.0e-6)
   sim.mesh(wavelength_elements=8)

   cross_section = wf.CrossSection(
       (0.0, 1.0e-6),
       background=wf.Material(eps_r=1.0),
       boundary="pec",
   )
   modes = wf.ModeSolver(
       cross_section, frequency=193.414489e12
   ).solve(max_refinements=0, num_modes=1, neff_guess=1.0)
   sim.set_modes(modes)
   sim.set_incident_mode(0)
   result = sim.run(max_refinements=0, h5_path="callback_result.h5")

Caller-validated callback requirements in version ``0.0.1``:

* ``eps_background`` is lossless and independent of z.

* The supplied ``ModeSet`` represents that exact background and contains positive-z roots (``forward`` or ``right-decaying``), not backward roots.

* ``eps_r - eps_background`` has compact support outside every PML.

* Explicit monitor lines bracket the complete contrast and lie in uniform sections where ``eps_r == eps_background``.

``set_modes`` checks frequency, ``ky``, transverse span, and the open-guide light-line filter, but it does not yet prove these callback-specific physical invariants. Violating them invalidates the scattered-field equation or the incoming/outgoing projection ordering.

``x_span`` and ``z_span``
^^^^^^^^^^^^^^^^^^^^^^^^^

Read-only properties returning the validated domain bounds as ``tuple[float, float]`` in metres.

``add_rectangle``
^^^^^^^^^^^^^^^^^

.. code-block:: python

   sim.add_rectangle(
       *,
       x: Sequence[float],
       z: tuple[float, float] | Literal["all"],
       eps: complex | float,
       mu: complex | float = 1.0,
       background: bool = False,
       name: str | None = None,
   ) -> wavefem.geometry.Region

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``x``
     - Required
     - ``Sequence[float]``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature. Keyword-only.
   * - ``z``
     - Required
     - ``tuple[float, float] | Literal['all']``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature. Keyword-only.
   * - ``eps``
     - Required
     - ``complex | float``
     - Relative permittivity. Scalar or Cartesian diagonal material inputs are supported where the signature permits; callbacks return scalar/broadcast-compatible values at physical coordinates. Keyword-only.
   * - ``mu``
     - Optional
     - ``complex | float``
     - Relative permeability, scalar or Cartesian diagonal where supported. WaveFEM scattering requires equal actual and background permeability. Keyword-only. Default: ``1.0``.
   * - ``background``
     - Optional
     - ``bool``
     - In placement methods, True adds a z-invariant background region/sheet; False adds an actual-device perturbation. In material/geometry constructors, this is the exterior Material object. Keyword-only. Default: ``False``.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Adds an axis-aligned material rectangle and returns its ``Region``.

* Set ``background=True`` only for a z-invariant unperturbed-guide layer; such a rectangle must use ``z="all"``.

* With ``background=False``, the rectangle changes only the actual device and contributes to the compact scattered-field source.

* ``name`` must be unique. An automatic name is generated when omitted.

* Later regions override earlier material assignments where they overlap.

* The rectangle must lie entirely inside the domain.

Adding a rectangle invalidates the mesh, mode set, and incident selection.

``add_circle``
^^^^^^^^^^^^^^

.. code-block:: python

   sim.add_circle(
       *,
       center: Sequence[float],
       radius: float,
       eps: complex | float,
       mu: complex | float = 1.0,
       name: str | None = None,
   ) -> wavefem.geometry.Region

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``center``
     - Required
     - ``Sequence[float]``
     - Centre coordinates in metres, ordered as the package computational axes: x-y, x-z, or x-y-z. Keyword-only.
   * - ``radius``
     - Required
     - ``float``
     - Positive radius in metres; inner_radius describes the hollow inner boundary where supported. Keyword-only.
   * - ``eps``
     - Required
     - ``complex | float``
     - Relative permittivity. Scalar or Cartesian diagonal material inputs are supported where the signature permits; callbacks return scalar/broadcast-compatible values at physical coordinates. Keyword-only.
   * - ``mu``
     - Optional
     - ``complex | float``
     - Relative permeability, scalar or Cartesian diagonal where supported. WaveFEM scattering requires equal actual and background permeability. Keyword-only. Default: ``1.0``.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Adds a finite circular perturbation to the actual material. ``center`` is ``(x, z)`` in metres and ``radius`` must be positive. A circle cannot define a background guide because it is not z-invariant.

``add_polygon``
^^^^^^^^^^^^^^^

.. code-block:: python

   sim.add_polygon(
       *,
       points: Sequence[Sequence[float]],
       eps: complex | float,
       mu: complex | float = 1.0,
       name: str | None = None,
   ) -> wavefem.geometry.Region

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``points``
     - Required
     - ``Sequence[Sequence[float]]``
     - Coordinates used by the object or evaluation operation. Mesh geometry uses physical metres; low-level FE operators use their basis coordinate scale. See the array-shape conventions. Keyword-only.
   * - ``eps``
     - Required
     - ``complex | float``
     - Relative permittivity. Scalar or Cartesian diagonal material inputs are supported where the signature permits; callbacks return scalar/broadcast-compatible values at physical coordinates. Keyword-only.
   * - ``mu``
     - Optional
     - ``complex | float``
     - Relative permeability, scalar or Cartesian diagonal where supported. WaveFEM scattering requires equal actual and background permeability. Keyword-only. Default: ``1.0``.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Adds a finite polygonal perturbation. ``points`` contains at least three ordered ``(x, z)`` vertices in metres. The polygon must lie inside the domain. Self-intersection is not repaired or inferred; provide a simple polygon.

``add_pec``
^^^^^^^^^^^

.. code-block:: python

   sim.add_pec(
       *,
       x: float,
       z: Sequence[float] | Literal["all"] = "all",
       background: bool = True,
       name: str | None = None,
   ) -> wavefem.geometry.PECSheet

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``x``
     - Required
     - ``float``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature. Keyword-only.
   * - ``z``
     - Optional
     - ``Sequence[float] | Literal['all']``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature. Keyword-only. Default: ``'all'``.
   * - ``background``
     - Optional
     - ``bool``
     - In placement methods, True adds a z-invariant background region/sheet; False adds an actual-device perturbation. In material/geometry constructors, this is the exterior Material object. Keyword-only. Default: ``True``.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Adds an ideal zero-thickness PEC sheet at constant x and returns its immutable ``PECSheet`` record. Coordinates are in metres. The sheet must lie strictly inside ``x_span``; numerical outer PEC walls already exist and are not geometry objects. With ``background=True``, the sheet must use ``z="all"``; it is present in both the modal background and actual device before slots are cut. With ``background=False``, ``z=(z_min,z_max)`` must be a finite compact span strictly inside the non-PML interior. Such a plate is absent from the lead mode and is inserted only into the actual device.

The mesh conforms to the sheet. Its Nedelec tangential edge DOFs and nodal ``E_y`` DOFs are constrained, while normal ``E_x`` remains free and may jump across the two faces. A background sheet's coordinate is also added to the one-dimensional wave-port cross-section. On an actual-only plate, the scattered trace is prescribed as ``E_sc,t=-E_inc,t``, which enforces ``E_total,t=0``. Adding a sheet invalidates existing mesh, modes, and incident state.

``add_slot``
^^^^^^^^^^^^

.. code-block:: python

   sim.add_slot(
       *,
       pec: wavefem.geometry.PECSheet | str,
       z: Sequence[float],
       name: str | None = None,
   ) -> wavefem.geometry.PECSlot

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``pec``
     - Required
     - ``PECSheet | str``
     - PEC sheet/plane handle(s) that define zero tangential electric field. Slots must refer to a background sheet. Keyword-only.
   * - ``z``
     - Required
     - ``Sequence[float]``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature. Keyword-only.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Cuts a finite opening from the actual-device copy of a background PEC sheet. ``pec`` is either the exact ``PECSheet`` returned by this simulation's ``add_pec`` or its unique name. ``z=(z_min, z_max)`` is a finite, strictly increasing span in metres that must lie strictly inside the sheet; slots on one sheet may not overlap or touch. The background profile deliberately retains the complete sheet, so its lead modes remain z-invariant.

Meshing inserts both slot endpoints as conforming partitions. Facets inside the opening are omitted from the actual PEC constraint set and recorded as released-background facets. During ``solve``, their natural two-sided magnetic reaction is assembled from the incident mode. This boundary source remains nonzero even when actual and background permittivities are identical everywhere. Adding a slot invalidates existing mesh, modes, and incident state.

``add_pml``
^^^^^^^^^^^

.. code-block:: python

   sim.add_pml(
       *,
       x: float | wf.PML | None = None,
       z: float | wf.PML | None = None,
       order: int = 3,
       target_reflection: float = 1e-8,
   ) -> None

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``x``
     - Optional
     - ``float | PML | None``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature. Keyword-only. Default: ``None``.
   * - ``z``
     - Optional
     - ``float | PML | None``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature. Keyword-only. Default: ``None``.
   * - ``order``
     - Optional
     - ``int``
     - Polynomial grading exponent for PML stretching; must satisfy the positive-integer constraints of the PML API. Keyword-only. Default: ``3``.
   * - ``target_reflection``
     - Optional
     - ``float``
     - Target PML amplitude reflection used to derive its attenuation profile; strictly between zero and one. Keyword-only. Default: ``1e-08``.

Configures symmetric PMLs at both ends of either selected axis.

* A numeric ``x`` or ``z`` is interpreted as PML thickness in metres.

* A ``PML`` instance supplies its own thickness, order, and target.

* An omitted axis preserves its current PML, so separate ``add_pml(x=...)`` and ``add_pml(z=...)`` calls accumulate.

* ``order`` and ``target_reflection`` apply only to numeric thickness inputs.

* Two PMLs on an axis must leave a non-PML interior.

A z-PML is mandatory for ``solve()``. An x-PML is required for open transverse structures. PMLs are transformation-optics layers terminated by the outer PEC truncation.

``set_monitors``
^^^^^^^^^^^^^^^^

.. code-block:: python

   sim.set_monitors(*, left: float, right: float) -> None

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``left``
     - Required
     - ``float``
     - Lead/trace side or physical monitor coordinate, according to the method signature. Integrated scattering currently launches from the left lead. Keyword-only.
   * - ``right``
     - Required
     - ``float``
     - Lead/trace side or physical monitor coordinate, according to the method signature. Integrated scattering currently launches from the left lead. Keyword-only.

Sets physical z coordinates for the two modal monitor lines. The coordinates must satisfy ``left < right``, lie inside the non-PML interior, surround every geometry-defined finite perturbation, and cross uniform lead material. The lines are inserted as mesh-conforming partitions during ``mesh()``.

If omitted, WaveFEM chooses monitors between a geometry-defined perturbation and the z-PMLs. Callback devices should always set them explicitly because WaveFEM cannot infer the callback contrast bounds. Call this method before meshing and before selecting an incident reference plane.

``mesh``
^^^^^^^^

.. code-block:: python

   sim.mesh(
       *,
       max_element_size: float | None = None,
       wavelength_elements: int = 4,
       refine_interfaces: bool = True,
       dielectric_refinement_factor: float = 0.5,
       pec_refinement_factor: float = 0.5,
       pec_refinement_distance: float | None = None,
   ) -> wavefem.mesh.Mesh2D

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
   * - ``refine_interfaces``
     - Optional
     - ``bool``
     - Enable the corresponding geometry-based local mesh-size field. Disabling sizing does not remove conforming material or PEC interfaces. Keyword-only. Default: ``True``.
   * - ``dielectric_refinement_factor``
     - Optional
     - ``float``
     - Local edge-size multiplier in (0, 1], applied at material regions or actual PEC curves respectively. Keyword-only. Default: ``0.5``.
   * - ``pec_refinement_factor``
     - Optional
     - ``float``
     - Local edge-size multiplier in (0, 1], applied at material regions or actual PEC curves respectively. Keyword-only. Default: ``0.5``.
   * - ``pec_refinement_distance``
     - Optional
     - ``float | None``
     - Physical distance in metres over which the local mesh-size target transitions back to the surrounding target; None selects the mesher default when permitted. Keyword-only. Default: ``None``.

Generates a conforming first-order triangular Gmsh mesh.

The derived conservative base edge target is ``vacuum_wavelength / (wavelength_elements * maximum_material_index)``. For geometry-defined materials with ``refine_interfaces=True``, each material region additionally receives a wavelength-scaled local target. A region of index ``n`` uses the base target multiplied by ``min(1, dielectric_refinement_factor * exterior_index / n)``. ``dielectric_refinement_factor`` must be in ``(0, 1]``; its default halves the target before applying the local wavelength ratio. This preserves the previous global resolution cap while making explicitly modeled material regions finer than the exterior. Material callbacks retain global sizing because their samples cannot be mapped to reliable Gmsh surfaces. Callback-based maximum-index estimation uses a finite sampling grid, so narrow features require an explicit ``max_element_size``.

When ``max_element_size`` is supplied, WaveFEM uses the smaller of that value and the derived base value; local dielectric and PEC targets may be smaller. ``wavelength_elements`` must be an integer of at least four.

Material boundaries, monitor lines, PML interfaces, internal PEC sheets, and every PEC-slot endpoint are always mesh-conforming, independent of sizing fields. With ``refine_interfaces=True``, dielectric local-wavelength fields and an actual-PEC distance field are both enabled. ``pec_refinement_factor`` is the PEC edge target as a fraction in ``(0, 1]`` of the smallest local material target. ``pec_refinement_distance`` is the physical distance over which that target transitions back to the base size; ``None`` selects three times the smallest material target. Setting ``refine_interfaces=False`` disables both local fields but does not remove conforming interfaces or PEC facets.

The returned ``Mesh2D`` identifies background, actual, released, and inserted PEC facet sets so low-level workflows do not need geometric reclassification.

The returned ``Mesh2D.info.requested_maximum_edge`` reveals the selected target. Warnings are issued when a PML spans fewer than three requested edge lengths or a monitor is fewer than two requested edge lengths from a perturbation. Meshing clears previously solved modes and incident selection.

``solve_modes``
^^^^^^^^^^^^^^^

.. code-block:: python

   sim.solve_modes(
       *,
       side: Literal["left", "right"] = "left",
       num_modes: int = 4,
       neff_guess: complex | None = None,
       num_elements: int | None = None,
       max_refinements: int | None = None,
       adaptive_tolerance: float | None = None,
   ) -> wf.ModeSet

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``side``
     - Optional
     - ``Literal['left', 'right']``
     - Lead/trace side or physical monitor coordinate, according to the method signature. Integrated scattering currently launches from the left lead. Keyword-only. Default: ``'left'``.
   * - ``num_modes``
     - Optional
     - ``int``
     - Number of modes or candidate eigenpairs requested; a positive integer. Candidate pools may include roots later rejected by validation. Keyword-only. Default: ``4``.
   * - ``neff_guess``
     - Optional
     - ``complex | None``
     - Dimensionless effective index beta/k0. A guess selects roots near that complex value; None uses the solver estimate. Keyword-only. Default: ``None``.
   * - ``num_elements``
     - Optional
     - ``int | None``
     - Initial number of 1D intervals; interfaces can add mesh nodes. Adaptive passes may increase this count. Keyword-only. Default: ``None``.
   * - ``max_refinements``
     - Optional
     - ``int | None``
     - Maximum mesh updates after the initial solve; nonnegative integer. Zero keeps the initial mesh. Solver default is 2; examples explicitly use 0. None inherits the value from this simulation's SolverOptions. Keyword-only. Default: ``None``.
   * - ``adaptive_tolerance``
     - Optional
     - ``float | None``
     - Positive finite threshold for the normalized discretization estimator. Stop when residual <= threshold. Independent of algebraic tolerances; exhausting the budget is not convergence. None inherits the value from this simulation's SolverOptions. Keyword-only. Default: ``None``.

Builds the z-invariant background ``CrossSection`` and solves forward mode families near ``neff_guess``.

* ``side`` validates the requested lead name. The current device architecture uses the same unperturbed cross-section on both sides.

* ``num_modes`` is the number of validated modes required.

* ``num_elements`` controls the initial 1D cross-section mesh. When omitted, it is derived from the 2D target size with a minimum of 16 elements.

* ``max_refinements`` and ``adaptive_tolerance`` override ``SolverOptions`` for the lead solve, including any subsequent angle resolution.

* Geometry-backed background layers must be z-invariant rectangles.

* Geometry-backed background PEC sheets become internal PEC points in the one-dimensional port problem. At each point ``E_y`` and ``E_z`` are zero while the two one-sided normal ``E_x`` traces remain unknown.

* Lead materials must be lossless.

* Open guides require an x-PML. With one present, the integrated workflow filters PML/radiation candidates and retains bound modes above the exterior light line.

The method stores and returns the resulting ``ModeSet`` and clears any previous incident selection. With a nonzero ``angle``, this first set is the normal-incidence family catalogue used to choose the incident family. Callback devices must use ``set_modes`` instead.

``set_modes``
^^^^^^^^^^^^^

.. code-block:: python

   sim.set_modes(modes: wf.ModeSet) -> wf.ModeSet

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``modes``
     - Required
     - ``ModeSet``
     - Mode or mode collection. Integer selectors use the owning API indexing convention; supplied mode sets must match frequency, ky, and transverse geometry.

Binds an externally solved, nonempty mode set. Every mode is checked for matching ``omega``, ``ky``, and transverse span. When an x-PML is configured, the same bound-mode light-line filter used by ``solve_modes`` is applied. Discarded radiation candidates produce a ``RuntimeWarning``.

The caller must currently supply positive-z modal family members (``forward`` or ``right-decaying``), a lossless z-invariant background, and modes that solve that exact background. Version ``0.0.1`` does not canonicalize backward roots or fully validate callback lead material. Use this method for callback-defined devices or custom cross-section studies. It clears any previous incident selection.

``set_incident_mode``
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   sim.set_incident_mode(
       mode: int | wf.Mode,
       *,
       side: Literal["left", "right"] = "left",
       reference_plane: float | None = None,
       amplitude: complex = 1.0,
   ) -> wf.IncidentMode

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``mode``
     - Required
     - ``int | Mode``
     - Mode or mode collection. Integer selectors use the owning API indexing convention; supplied mode sets must match frequency, ky, and transverse geometry.
   * - ``side``
     - Optional
     - ``Literal['left', 'right']``
     - Lead/trace side or physical monitor coordinate, according to the method signature. Integrated scattering currently launches from the left lead. Keyword-only. Default: ``'left'``.
   * - ``reference_plane``
     - Optional
     - ``float | None``
     - Physical longitudinal reference coordinate(s) in metres at which incident or scattering amplitudes are defined. Keyword-only. Default: ``None``.
   * - ``amplitude``
     - Optional
     - ``complex``
     - Complex modal amplitude(s); squared magnitude is power for a unit-power propagating mode. Integrated incidence requires a nonzero finite launch amplitude. Keyword-only. Default: ``1.0``.

Selects a propagating, unit-power mode from the simulation's current ``ModeSet``. For an integrated simulation with nonzero ``angle``, this call re-solves the selected normal-incidence family at the requested angle, updates ``sim.ky`` and ``sim.modes`` to the actual oblique solution, and launches the matching resolved mode.

* ``mode`` may be a zero-based index or the exact ``Mode`` object contained in the current set. External and stale mode objects are rejected.

* ``side="left"`` launches toward positive z. ``side="right"`` constructs the correct negative-z field, but the integrated ``solve()`` path currently supports left incidence only.

* ``reference_plane`` is the physical z coordinate where ``amplitude`` is defined. It defaults to the launch-side monitor.

* For a unit-power propagating mode, incident power is ``abs(amplitude)**2`` W/m. Zero and numerically tiny amplitudes are rejected.

The returned ``IncidentMode`` is also stored as ``sim.incident``.

``solve``
^^^^^^^^^

.. code-block:: python

   sim.solve(
       *,
       h5_path: str | os.PathLike[str] | None = None,
       max_refinements: int | None = None,
       adaptive_tolerance: float | None = None,
   ) -> wf.ScatteringResult

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``h5_path``
     - Optional
     - ``str | PathLike[str] | None``
     - Filesystem destination/source for HDF5 persistence or viewer launch. A directory is accepted only by viewer/directory-inspection APIs; None follows the method-specific default. Keyword-only. Default: ``None``.
   * - ``max_refinements``
     - Optional
     - ``int | None``
     - Maximum mesh updates after the initial solve; nonnegative integer. Zero keeps the initial mesh. Solver default is 2; examples explicitly use 0. None inherits the value from this simulation's SolverOptions. Keyword-only. Default: ``None``.
   * - ``adaptive_tolerance``
     - Optional
     - ``float | None``
     - Positive finite threshold for the normalized discretization estimator. Stop when residual <= threshold. Independent of algebraic tolerances; exhausting the budget is not convergence. None inherits the value from this simulation's SolverOptions. Keyword-only. Default: ``None``.

Assembles the mixed Maxwell system, forms the volume permittivity-contrast source and any released-PEC aperture source, prescribes ``-E_inc,t`` on finite inserted PEC plates, solves the outgoing scattered field, reconstructs total E and H, projects both lead monitors onto forward/backward modes, and computes power terms.

Preconditions:

* Geometry and PMLs have been configured. A missing mesh is generated automatically.

* If modes are missing, one forward lead mode is solved automatically. Callback devices must supply modes through ``set_modes()``.

* If no incident mode is selected, mode zero is selected automatically; it must be propagating.

* A z-directed PML is configured.

* Actual and background permeability are identical.

The returned fields are sampled in the non-PML control volume between the two z monitors. Port S-parameters are normalized by the prescribed incident amplitude. Radiation is measured through transverse control surfaces when an x-PML exists; for a closed transverse guide ``radiated_power`` is exactly zero.

When ``h5_path`` is a path, the complete result and its lead modes are written to a schema-versioned HDF5 file after the solve succeeds. The returned frozen result is copied with its absolute ``h5_path`` field set to the written path. With the default ``None``, no file is created and ``result.h5_path`` remains ``None``. The destination's parent directory must already exist. Persistence requires ``h5py``; a missing or unloadable HDF5 runtime raises ``ConfigurationError`` after the numerical solve, and the call does not return a result whose persistence failed.

``run``
^^^^^^^

.. code-block:: python

   sim.run(
       *,
       h5_path: str | os.PathLike[str] = "wavefem_result.h5",
       max_refinements: int | None = None,
       adaptive_tolerance: float | None = None,
   ) -> wf.ScatteringResult

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``h5_path``
     - Optional
     - ``str | PathLike[str]``
     - Filesystem destination/source for HDF5 persistence or viewer launch. A directory is accepted only by viewer/directory-inspection APIs; None follows the method-specific default. Keyword-only. Default: ``'wavefem_result.h5'``.
   * - ``max_refinements``
     - Optional
     - ``int | None``
     - Maximum mesh updates after the initial solve; nonnegative integer. Zero keeps the initial mesh. Solver default is 2; examples explicitly use 0. None inherits the value from this simulation's SolverOptions. Keyword-only. Default: ``None``.
   * - ``adaptive_tolerance``
     - Optional
     - ``float | None``
     - Positive finite threshold for the normalized discretization estimator. Stop when residual <= threshold. Independent of algebraic tolerances; exhausting the budget is not convergence. None inherits the value from this simulation's SolverOptions. Keyword-only. Default: ``None``.

The persistence-first terminal operation. ``run`` has the same numerical preconditions and behavior as ``solve``, but a path is mandatory and defaults to ``wavefem_result.h5``. It delegates to ``solve(h5_path=h5_path)``, returns the same ``ScatteringResult`` type, and guarantees that a successful call has an associated HDF5 file recorded in ``result.h5_path``.

Use ``run`` for normal application workflows, ``solve(h5_path=some_path)`` when the destination is conditional, and bare ``solve()`` only when an explicitly in-memory result is desired.

``sweep_frequencies``
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   sim.sweep_frequencies(
       frequencies_hz: Sequence[float],
       *,
       h5_path: str | os.PathLike[str] | None = "wavefem_sweep.h5",
       mesh_options: Mapping[str, object] | None = None,
       mode_options: Mapping[str, object] | None = None,
       incident_mode: int = 0,
       amplitude: complex = 1.0,
       reference_plane: float | None = None,
       mode_factory: Callable[[float], wf.ModeSet] | None = None,
   ) -> wf.FrequencySweepResult

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``frequencies_hz``
     - Required
     - ``Sequence[float]``
     - One-dimensional ordinary-frequency sequence in Hz, in the order of the associated results. Sweep solvers require positive, strictly increasing values.
   * - ``h5_path``
     - Optional
     - ``str | PathLike[str] | None``
     - Filesystem destination/source for HDF5 persistence or viewer launch. A directory is accepted only by viewer/directory-inspection APIs; None follows the method-specific default. Keyword-only. Default: ``'wavefem_sweep.h5'``.
   * - ``mesh_options``
     - Optional
     - ``Mapping[str, object] | None``
     - Configuration object or keyword mapping forwarded to mode solving, meshing, or scattering. Set max_refinements=0 in SolverOptions and mode_options for fixed-mesh sweep examples. Keyword-only. Default: ``None``.
   * - ``mode_options``
     - Optional
     - ``Mapping[str, object] | None``
     - Configuration object or keyword mapping forwarded to mode solving, meshing, or scattering. Set max_refinements=0 in SolverOptions and mode_options for fixed-mesh sweep examples. Keyword-only. Default: ``None``.
   * - ``incident_mode``
     - Optional
     - ``int``
     - Mode, case, array, or mapping selector. Python indexing is zero based; explicit mode(number) and standalone visualization use their documented one-based numbers. Keyword-only. Default: ``0``.
   * - ``amplitude``
     - Optional
     - ``complex``
     - Complex modal amplitude(s); squared magnitude is power for a unit-power propagating mode. Integrated incidence requires a nonzero finite launch amplitude. Keyword-only. Default: ``1.0``.
   * - ``reference_plane``
     - Optional
     - ``float | None``
     - Physical longitudinal reference coordinate(s) in metres at which incident or scattering amplitudes are defined. Keyword-only. Default: ``None``.
   * - ``mode_factory``
     - Optional
     - ``Callable[[float], ModeSet] | None``
     - Callable frequency_hz -> ModeSet supplying compatible lead modes for each callback-defined frequency-sweep point. Keyword-only. Default: ``None``.

Runs independent scattering simulations at a nonempty, strictly increasing sequence of positive ordinary frequencies in hertz. The source ``sim`` acts as a physical-configuration template and is not mutated: material regions, PEC sheets and slots, material callbacks, PMLs, monitors, transverse boundary, propagation angle (or compatibility ``ky``), and solver options are copied into a new simulation for every frequency. Each point is then meshed, given a fresh compatible mode set, launched, and solved. Angle-based sweeps recompute ``ky`` from the selected incident family at every frequency.

Parameters:

* ``frequencies_hz``: finite positive 1D values in hertz. Duplicate, descending, Boolean, complex, empty, or multidimensional inputs raise ``ConfigurationError``.

* ``h5_path``: sweep-file destination. The default writes ``wavefem_sweep.h5``; ``None`` explicitly disables persistence. A successful write is recorded in ``sweep.h5_path`` as an absolute ``Path``.

* ``mesh_options``: keyword mapping forwarded to ``mesh()`` at every point, for example ``{"wavelength_elements": 10, "max_element_size": 80e-9}``.

* ``mode_options``: keyword mapping forwarded to ``solve_modes()``, for example ``{"num_modes": 4, "neff_guess": 2.4, "num_elements": 120}``. The integrated sweep currently requires its effective ``side`` to be ``"left"``. If ``num_modes`` is omitted it defaults to at least ``incident_mode + 1``.

* ``incident_mode``: zero-based modal-family index launched at every point.

* ``amplitude``: complex incident modal amplitude at the reference plane. For unit-power modes, its incident power is ``abs(amplitude)**2`` W/m.

* ``reference_plane``: common physical z reference in metres. ``None`` uses each point's automatically selected left monitor.

* ``mode_factory``: callback used only for material-function devices. It is called as ``mode_factory(frequency_hz)`` and must return a compatible positive-z ``ModeSet`` for that exact point. Geometry-backed devices normally omit it and let ``solve_modes()`` construct their modes.

The returned ``FrequencySweepResult.results[i]`` is a complete ``ScatteringResult``, not a summary-only record. If any point fails, the method raises the original solver/configuration exception and does not return a partial sweep. HDF5 writing happens only after all points succeed.

Mode roots are solved and ordered independently at each frequency; version ``0.0.1`` does not yet perform cross-frequency field-overlap branch tracking. Near a modal crossing or cutoff, a fixed integer ``incident_mode`` can therefore refer to a different physical branch. Use close frequency spacing, a physically informed ``neff_guess``, and inspect the saved modal E/H profiles in the viewer before interpreting a multimode curve as one continuous branch.

Example:

.. code-block:: python

   frequencies_hz = np.linspace(190.0e12, 196.0e12, 13)
   from dataclasses import replace
   sim.solver_options = replace(sim.solver_options, max_refinements=0)
   sweep = sim.sweep_frequencies(
       frequencies_hz,
       h5_path="frequency_sweep.h5",
       mesh_options={"wavelength_elements": 10},
       mode_options={"max_refinements": 0, "num_modes": 2, "neff_guess": 2.4},
   )
   print(sweep.S11, sweep.S21)

``SolverOptions``
~~~~~~~~~~~~~~~~~

.. code-block:: python

   wf.SolverOptions(
       linear_solver: Literal["direct"] = "direct",
       tolerance: float = 1e-10,
       quadrature_order: int = 4,
       projection_condition_limit: float = 1e12,
       element_order: int = 1,
       max_refinements: int = 2,
       adaptive_tolerance: float = 0.05,
   )

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``linear_solver``
     - Optional
     - ``Literal['direct']``
     - Sparse field solver selector. WaveFEM currently accepts only direct factorization. Default: ``'direct'``.
   * - ``tolerance``
     - Optional
     - ``float``
     - Positive numerical tolerance. Linear/QEP residual tolerances validate the algebraic solve; they do not set the adaptive mesh threshold. Default: ``1e-10``.
   * - ``quadrature_order``
     - Optional
     - ``int``
     - Finite-element integration order. Higher-order mixed elements require at least fourth-order quadrature; PML and material variation can require more. Default: ``4``.
   * - ``projection_condition_limit``
     - Optional
     - ``float``
     - Maximum allowed modal Gram-system condition number; must exceed one. Default: ``1000000000000.0``.
   * - ``element_order``
     - Optional
     - ``int``
     - Finite-element polynomial-order selection. Standalone 2D modes and WaveFEM scattering accept 1 (N1/P1) or 2 (N2/P2); other backends retain their fixed compatible spaces. Default: ``1``.
   * - ``max_refinements``
     - Optional
     - ``int``
     - Maximum mesh updates after the initial solve; nonnegative integer. Zero keeps the initial mesh. Solver default is 2; examples explicitly use 0. Default: ``2``.
   * - ``adaptive_tolerance``
     - Optional
     - ``float``
     - Positive finite threshold for the normalized discretization estimator. Stop when residual <= threshold. Independent of algebraic tolerances; exhausting the budget is not convergence. Default: ``0.05``.

* ``linear_solver``: only ``"direct"`` is implemented.

* ``tolerance``: maximum accepted relative residual for the sparse field solve.

* ``quadrature_order``: mixed FEM and monitor quadrature order; must be at least two.
  Raised to at least four when ``element_order=2``.

* ``element_order``: ``1`` selects the Nedelec N1/scalar P1 pair; ``2`` selects
  the compatible N2/P2 pair on affine triangles.  Applies to scattering fields;
  lead-mode discretization is controlled separately.

* ``projection_condition_limit``: maximum accepted condition number of the normalized modal Gram system; must exceed one.

Instances are frozen and validated at construction.

``max_refinements`` is a nonnegative integer limiting mesh updates after the
initial solve. ``adaptive_tolerance`` must be finite and positive; it stops
refinement when the normalized normal-D/tangential-H interface residual is
at or below the threshold. Zero refinements keeps the initial mesh. Both lead
and scattering solves inherit these defaults, including frequency sweeps.
Lead modes bisect marked intervals; scattering regenerates its conforming mesh
at 1.5 times the previous density, retaining element order and geometry sizing.
These are discretization estimators, independent of algebraic residuals and
not certified error bounds. ``solve_info`` contains ``adaptive_history``,
``adaptive_residual``, ``adaptive_converged``, and the applied controls. A spent
budget returns the last solution with ``adaptive_converged=False``. Only the
final adapted result is written to HDF5.

Scattering results
------------------

``ScatteringResult``
~~~~~~~~~~~~~~~~~~~~

``ScatteringResult`` is a frozen, self-contained result object. Users normally receive it from ``Scattering2D.solve()`` rather than constructing it manually. Its stored arrays and metadata are sufficient for post-processing without rerunning the FEM solve.

.. code-block:: python

   wf.ScatteringResult(
       coordinates,
       E_incident,
       E_scattered,
       H_incident,
       H_scattered,
       s_parameters,
       reflected_power: float,
       transmitted_power: float,
       radiated_power: float,
       absorbed_power: float,
       incident_power: float,
       ndofs: int,
       solve_info={},
       mesh_info={},
       projection_condition_numbers={},
       reference_planes={},
       port_betas={},
       frequency_hz: float | None = None,
       ky: float | None = None,
       modes: tuple[wf.Mode, ...] = (),
       h5_path: pathlib.Path | None = None,
       scene: wf.Scene2D | None = None,
   )

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``coordinates``
     - Required
     - ``RealArray``
     - Coordinates used by the object or evaluation operation. Mesh geometry uses physical metres; low-level FE operators use their basis coordinate scale. See the array-shape conventions.
   * - ``E_incident``
     - Required
     - ``ComplexArray``
     - Complex sampled incident/scattered/total field array, shaped (3, number of sample points), in V/m for E or A/m for H.
   * - ``E_scattered``
     - Required
     - ``ComplexArray``
     - Complex sampled incident/scattered/total field array, shaped (3, number of sample points), in V/m for E or A/m for H.
   * - ``H_incident``
     - Required
     - ``ComplexArray``
     - Complex sampled incident/scattered/total field array, shaped (3, number of sample points), in V/m for E or A/m for H.
   * - ``H_scattered``
     - Required
     - ``ComplexArray``
     - Complex sampled incident/scattered/total field array, shaped (3, number of sample points), in V/m for E or A/m for H.
   * - ``s_parameters``
     - Required
     - ``Mapping[PortKey, complex]``
     - Mapping from port/mode channel keys to complex scattering amplitudes, normalized to the incident mode amplitude.
   * - ``reflected_power``
     - Required
     - ``float``
     - Modal or electromagnetic power in W/m of invariant length for 2.5D fields. Complex power retains reactive flux; power ratios are reported separately.
   * - ``transmitted_power``
     - Required
     - ``float``
     - Modal or electromagnetic power in W/m of invariant length for 2.5D fields. Complex power retains reactive flux; power ratios are reported separately.
   * - ``radiated_power``
     - Required
     - ``float``
     - Modal or electromagnetic power in W/m of invariant length for 2.5D fields. Complex power retains reactive flux; power ratios are reported separately.
   * - ``absorbed_power``
     - Required
     - ``float``
     - Modal or electromagnetic power in W/m of invariant length for 2.5D fields. Complex power retains reactive flux; power ratios are reported separately.
   * - ``incident_power``
     - Required
     - ``float``
     - Modal or electromagnetic power in W/m of invariant length for 2.5D fields. Complex power retains reactive flux; power ratios are reported separately.
   * - ``ndofs``
     - Required
     - ``int``
     - Total number of nodes or degrees of freedom in the relevant full space; a nonnegative/positive integer as required by the constructor.
   * - ``solve_info``
     - Optional
     - ``Mapping[str, Any]``
     - Structured metadata/diagnostic container associated with the mesh or result. solve_info/metadata retain applied adaptive controls and stopping history. Default: ``fresh default container``.
   * - ``mesh_info``
     - Optional
     - ``Mapping[str, Any]``
     - Structured metadata/diagnostic container associated with the mesh or result. solve_info/metadata retain applied adaptive controls and stopping history. Default: ``fresh default container``.
   * - ``projection_condition_numbers``
     - Optional
     - ``Mapping[str, float]``
     - Measured condition number(s) of the modal projection system. Default: ``fresh default container``.
   * - ``reference_planes``
     - Optional
     - ``Mapping[str, float]``
     - Physical longitudinal reference coordinate(s) in metres at which incident or scattering amplitudes are defined. Default: ``fresh default container``.
   * - ``port_betas``
     - Optional
     - ``Mapping[BetaKey, complex]``
     - Complex longitudinal propagation constant(s), in rad/m. Default: ``fresh default container``.
   * - ``frequency_hz``
     - Optional
     - ``float | None``
     - Ordinary frequency in Hz; must be finite and positive. Default: ``None``.
   * - ``ky``
     - Optional
     - ``float | None``
     - Real Fourier wavenumber along invariant y, in rad/m; zero gives propagation in the computational plane. Default: ``None``.
   * - ``modes``
     - Optional
     - ``tuple[Mode, ...]``
     - Mode or mode collection. Integer selectors use the owning API indexing convention; supplied mode sets must match frequency, ky, and transverse geometry. Default: ``fresh default container``.
   * - ``h5_path``
     - Optional
     - ``Path | None``
     - Filesystem destination/source for HDF5 persistence or viewer launch. A directory is accepted only by viewer/directory-inspection APIs; None follows the method-specific default. Default: ``None``.
   * - ``scene``
     - Optional
     - ``Scene2D | None``
     - Portable geometry overlay: physical points/triangles and named boundary, monitor, or PML lines for visualization. Default: ``None``.

The displayed empty mappings denote dataclass factories, not shared mutable defaults. Construction validates all array shapes, finiteness, nonnegative powers, S keys, beta directions, and metadata mappings. Invalid manually constructed results raise ``ValueError``.

Stored fields
^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Field
     - Meaning
   * - ``coordinates``
     - Real array ``(2, N)`` containing physical ``x,z`` samples
   * - ``E_incident``, ``E_scattered``
     - Complex arrays ``(3, N)``
   * - ``H_incident``, ``H_scattered``
     - Complex arrays ``(3, N)``
   * - ``s_parameters``
     - Read-only mapping ``(side, out_mode, in_mode) -> complex``
   * - ``reflected_power``
     - Total nonnegative reflected modal power, W/m
   * - ``transmitted_power``
     - Total nonnegative transmitted modal power, W/m
   * - ``radiated_power``
     - Outward transverse radiation power, W/m
   * - ``absorbed_power``
     - Integrated passive material absorption, W/m
   * - ``incident_power``
     - Prescribed incident power, W/m
   * - ``ndofs``
     - Number of mixed FEM degrees of freedom
   * - ``solve_info``
     - Read-only numerical metadata and raw diagnostics
   * - ``mesh_info``
     - Read-only mesh metadata
   * - ``projection_condition_numbers``
     - Read-only monitor-to-condition mapping
   * - ``reference_planes``
     - Current left/right S-parameter reference planes, m
   * - ``port_betas``
     - ``(side, mode) -> +z beta`` mapping used for de-embedding
   * - ``frequency_hz``
     - Ordinary solve frequency in hertz, or ``None`` when unknown
   * - ``ky``
     - Prescribed invariant-direction wavenumber in rad/m, or ``None`` when unknown
   * - ``modes``
     - Tuple of lead modes sampled into HDF5 when the result is persisted
   * - ``h5_path``
     - Absolute persisted-file path associated by an integrated run, otherwise ``None``
   * - ``scene``
     - Optional full-domain ``Scene2D`` material mesh and visualization overlays

The coordinates are flattened FEM quadrature samples inside the non-PML control volume, not mesh nodes or an arbitrary evaluation grid. Duplicate locations may occur where quadrature points belong to adjacent elements.

``solve_info`` from the integrated solver includes the direct-solve residual, length scale, active-source fraction, projection residuals, incoming amplitude mismatch, independent energy residual, unclamped raw powers, and port-Gram normalization errors. These metadata keys are diagnostic rather than a separately versioned stable schema.

Current integrated metadata keys include:

.. list-table::
   :header-rows: 1

   * - Key
     - Meaning
   * - ``method``, ``relative_residual``
     - Linear-solver method and free-DOF residual
   * - ``length_scale``
     - Metres represented by one computational unit
   * - ``source_active_fraction``
     - Fraction of quadrature points with nonzero contrast source
   * - ``released_pec_facet_count``
     - Number of background PEC facets released as finite slots
   * - ``inserted_pec_facet_count``
     - Number of actual-only PEC facets inserted as finite plates
   * - ``prescribed_pec_dof_count``
     - Number of PEC trace DOFs with nonzero scattered-field data
   * - ``left_projection_residual``, ``right_projection_residual``
     - Weighted E/H reconstruction errors
   * - ``projected_incoming_amplitude``, ``prescribed_incoming_amplitude``
     - Independent projection check against the launched amplitude
   * - ``incoming_projection_relative_error``
     - Relative mismatch of those two amplitudes
   * - ``independent_energy_residual``
     - Closed-control-surface Poynting/absorption residual
   * - ``raw_radiated_power``, ``raw_absorbed_power``
     - Unclamped flux-derived powers, W/m
   * - ``raw_reflected_modal_power``, ``raw_transmitted_modal_power``
     - Unclamped Gram-derived port powers, W/m
   * - ``forward_port_gram_diagonal_error``, ``backward_port_gram_diagonal_error``
     - Sampled deviation from unit signed modal power

``mesh_info`` currently contains ``nodes``, ``elements``, ``minimum_edge``, ``maximum_edge``, and ``requested_maximum_edge``. The condition-number mapping uses ``"left"`` and ``"right"`` keys.

In an integrated result, both initial ``reference_planes`` entries equal the incident launch reference plane, not the two physical monitor coordinates. Modal traces are evaluated against that common phase plane, which makes a uniform guide's initial transmission close to ``1+0j``.

The dataclass is frozen, and its mappings are copied into read-only proxies. The sampled field arrays are not made recursively immutable. Arrays owned by ``scene``, when present, are defensive read-only copies.

``E_total`` and ``H_total``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Read-only computed properties:

.. code-block:: python

   result.E_total == result.E_incident + result.E_scattered
   result.H_total == result.H_incident + result.H_scattered

Both have shape ``(3, N)``.

``S``, ``S11``, and ``S21``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   result.S(
       side: str,
       *,
       out_mode: int = 0,
       in_mode: int = 0,
   ) -> complex

``side`` is ``"left"`` for reflected output or ``"right"`` for transmitted output. Mode indices are zero-based. A missing combination raises ``KeyError``.

``result.S11`` is shorthand for ``S("left", out_mode=0, in_mode=0)``. ``result.S21`` is shorthand for ``S("right", out_mode=0, in_mode=0)``. When a higher-order input was launched, use ``S(..., in_mode=index)`` rather than these two mode-zero shorthands.

Power properties
^^^^^^^^^^^^^^^^

.. code-block:: python

   result.reflection
   result.transmission
   result.power_balance
   result.power_balance_error

* ``reflection = reflected_power / incident_power``.

* ``transmission = transmitted_power / incident_power``.

* ``power_balance = (R_power + T_power + radiation + absorption) / input``.

* ``power_balance_error = abs(1 - power_balance)``.

For multimode ports, reflected and transmitted powers come from the full propagating-mode power Gram, not a naive sum of ``abs(S)**2``.

``field``
^^^^^^^^^

.. code-block:: python

   result.field(
       component: str = "E",
       *,
       quantity: Literal[
           "complex", "abs", "real", "imag", "phase", "norm"
       ] = "complex",
       part: Literal["total", "incident", "scattered"] = "total",
   ) -> numpy.ndarray

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``component``
     - Optional
     - ``str``
     - Field component selection, using Cartesian electric/magnetic names such as Ex or Hy. The visualization API also supports its documented aggregate quantities. Default: ``'E'``.
   * - ``quantity``
     - Optional
     - ``Literal['complex', 'abs', 'real', 'imag', 'phase', 'norm']``
     - Complex-data display selection, such as real, imag, abs, or phase; accepted values are given by the owning plotting API. Keyword-only. Default: ``'complex'``.
   * - ``part``
     - Optional
     - ``Literal['total', 'incident', 'scattered']``
     - Complex-data display selection, such as real, imag, abs, or phase; accepted values are given by the owning plotting API. Keyword-only. Default: ``'total'``.

``component`` may be ``"E"``, ``"H"``, ``"Ex"``, ``"Ey"``, ``"Ez"``, ``"Hx"``, ``"Hy"``, or ``"Hz"``.

* A Cartesian component returns one value per stored coordinate.

* Bare ``"E"`` or ``"H"`` returns the Euclidean vector magnitude and treats ``quantity="complex"`` as ``"norm"``.

* ``"abs"``/``"norm"`` return magnitude, ``"real"`` and ``"imag"`` return parts, and ``"phase"`` returns radians from ``numpy.angle``.

* ``part`` selects the total, analytic incident, or FEM scattered field.

``plot_field``
^^^^^^^^^^^^^^

.. code-block:: python

   result.plot_field(
       component: str = "E",
       *,
       quantity: Literal["abs", "real", "imag", "phase", "norm"] = "abs",
       part: Literal["total", "incident", "scattered"] = "total",
       ax=None,
       cmap=None,
       levels: int = 50,
       colorbar: bool = True,
   )

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``component``
     - Optional
     - ``str``
     - Field component selection, using Cartesian electric/magnetic names such as Ex or Hy. The visualization API also supports its documented aggregate quantities. Default: ``'E'``.
   * - ``quantity``
     - Optional
     - ``Literal['abs', 'real', 'imag', 'phase', 'norm']``
     - Complex-data display selection, such as real, imag, abs, or phase; accepted values are given by the owning plotting API. Keyword-only. Default: ``'abs'``.
   * - ``part``
     - Optional
     - ``Literal['total', 'incident', 'scattered']``
     - Complex-data display selection, such as real, imag, abs, or phase; accepted values are given by the owning plotting API. Keyword-only. Default: ``'total'``.
   * - ``ax``
     - Optional
     - ``Any | None``
     - Existing Matplotlib axes for embedding a plot; None creates suitable axes when accepted. Keyword-only. Default: ``None``.
   * - ``cmap``
     - Optional
     - ``Any | None``
     - Matplotlib colormap name or object used for scalar field rendering. Keyword-only. Default: ``None``.
   * - ``levels``
     - Optional
     - ``int``
     - Contour-level count or explicit contour levels for scalar field plots. Keyword-only. Default: ``50``.
   * - ``colorbar``
     - Optional
     - ``bool``
     - Enable display of the figure, mesh overlay, or colorbar respectively. show=False returns plotting objects without opening a window. Keyword-only. Default: ``True``.

Plots a scalar field and returns the Matplotlib axes without calling ``matplotlib.pyplot.show``. It uses triangular filled contours for a 2D point cloud and falls back to a scatter plot for collinear or unsuitable samples. Duplicate coordinates are averaged only for visualization. Default colormaps are ``"twilight"`` for phase, ``"RdBu_r"`` for real/imaginary parts, and ``"viridis"`` otherwise. The display convention is ``z`` on the horizontal axis and ``x`` on the vertical axis; stored coordinates remain ordered ``(x, z)``.

``visualize`` and ``visualize_with_gui``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   result.visualize(component="Ey", quantity="real", show=True)
   process = result.visualize_with_gui()

``visualize`` calls ``plot_field`` and displays the Matplotlib window by default. Pass ``show=False`` to embed the returned axes without calling ``matplotlib.pyplot.show()``. The separate, zero-argument ``visualize_with_gui()`` method reuses an existing ``result.h5_path``; an in-memory result is first saved as ``wavefem_result.h5``. It opens the complete result, including all stored modes, and returns the native viewer ``subprocess.Popen`` handle.

``save_h5``
^^^^^^^^^^^

.. code-block:: python

   result.save_h5(path: str | os.PathLike[str]) -> pathlib.Path

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``path``
     - Required
     - ``str | PathLike[str]``
     - Filesystem destination/source for HDF5 persistence or viewer launch. A directory is accepted only by viewer/directory-inspection APIs; None follows the method-specific default.

Persists the result with ``save_result_h5``, including sampled incident, scattered, and total E/H fields; indexed S-parameters; all five power terms; solve, mesh, projection, reference-plane, and beta metadata; and every mode in ``result.modes``. When ``result.scene`` is present, it also persists the complete material mesh and boundary/port/PML overlays. The return value is the resolved absolute destination path.

``ScatteringResult`` is frozen, so calling ``save_h5`` does not change ``result.h5_path``. By contrast, ``Scattering2D.run`` and ``Scattering2D.solve(h5_path=...)`` return a copied result whose ``h5_path`` records the destination. Existing files are replaced atomically only after a complete temporary file has been written successfully. The parent directory must already exist.

``check``
^^^^^^^^^

.. code-block:: python

   result.check(
       *,
       power_balance_tolerance: float = 1e-3,
       projection_condition_warning: float = 1e10,
       projection_residual_warning: float = 1e-3,
       incoming_projection_warning: float = 1e-3,
       port_gram_diagonal_warning: float = 1e-2,
       s_parameter_power_tolerance: float = 1e-6,
   ) -> wf.DiagnosticReport

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``power_balance_tolerance``
     - Optional
     - ``float``
     - Positive diagnostic tolerance for energy accounting or agreement between field-derived and S-parameter-derived powers. Keyword-only. Default: ``0.001``.
   * - ``projection_condition_warning``
     - Optional
     - ``float``
     - Threshold at which the corresponding projection/normalization diagnostic is reported as a warning. Keyword-only. Default: ``10000000000.0``.
   * - ``projection_residual_warning``
     - Optional
     - ``float``
     - Threshold at which the corresponding projection/normalization diagnostic is reported as a warning. Keyword-only. Default: ``0.001``.
   * - ``incoming_projection_warning``
     - Optional
     - ``float``
     - Threshold at which the corresponding projection/normalization diagnostic is reported as a warning. Keyword-only. Default: ``0.001``.
   * - ``port_gram_diagonal_warning``
     - Optional
     - ``float``
     - Threshold at which the corresponding projection/normalization diagnostic is reported as a warning. Keyword-only. Default: ``0.01``.
   * - ``s_parameter_power_tolerance``
     - Optional
     - ``float``
     - Positive diagnostic tolerance for energy accounting or agreement between field-derived and S-parameter-derived powers. Keyword-only. Default: ``1e-06``.

Returns structured diagnostics and never prints. It checks:

* reported and independently integrated power balance;

* negative unclamped raw powers;

* modal-projection conditioning and residual;

* disagreement between prescribed and projected input amplitude;

* unit-power port-Gram diagonal errors;

* single-mode consistency between ``abs(S)**2`` and reported power.

Possible diagnostic codes include ``ok``, ``poor_power_balance``, ``poor_independent_energy_balance``, ``negative_raw_power``, ``ill_conditioned_projection``, ``poor_projection_residual``, ``incoming_projection_mismatch``, ``port_gram_normalization_error``, ``s_parameter_power_mismatch``, and corresponding ``invalid_*`` or overflow errors when stored metadata is malformed.

Warnings do not make ``DiagnosticReport.ok`` false; only diagnostics with severity ``"error"`` do.

``deembed``
^^^^^^^^^^^

.. code-block:: python

   result.deembed(*, left: float, right: float) -> wf.ScatteringResult

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``left``
     - Required
     - ``float``
     - Lead/trace side or physical monitor coordinate, according to the method signature. Integrated scattering currently launches from the left lead. Keyword-only.
   * - ``right``
     - Required
     - ``float``
     - Lead/trace side or physical monitor coordinate, according to the method signature. Integrated scattering currently launches from the left lead. Keyword-only.

Returns a new result with S-parameters shifted to new physical reference planes. Fields, powers, and the original object are unchanged. Because an existing associated HDF5 file still contains the old reference planes and amplitudes, the returned result clears ``h5_path``; call ``save_h5`` explicitly to persist the de-embedded result.

For stored positive-z roots and ``exp(+i*beta*z)``, WaveFEM applies

.. code-block:: text

   left output:  exp(i * (beta_in + beta_out) * (old_left - new_left))
   right output: exp(i * beta_in * (old_left - new_left)
                     - i * beta_out * (old_right - new_right))

Both original reference planes and every required port beta must exist. New planes are not required to lie inside the simulated domain or satisfy ``left < right``; physically meaningful plane selection is the caller's responsibility.

``FrequencySweepResult``
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   wf.FrequencySweepResult(
       frequencies_hz: numpy.ndarray,
       results: tuple[wf.ScatteringResult, ...],
       h5_path: pathlib.Path | None = None,
   )

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``frequencies_hz``
     - Required
     - ``FloatArray``
     - One-dimensional ordinary-frequency sequence in Hz, in the order of the associated results. Sweep solvers require positive, strictly increasing values.
   * - ``results``
     - Required
     - ``tuple[Any, ...]``
     - Solved result object(s) to persist, visualize, de-embed, or collect into a frequency sweep.
   * - ``h5_path``
     - Optional
     - ``Path | None``
     - Filesystem destination/source for HDF5 persistence or viewer launch. A directory is accepted only by viewer/directory-inspection APIs; None follows the method-specific default. Default: ``None``.

A frozen ordered collection returned by ``Scattering2D.sweep_frequencies``. ``frequencies_hz`` is copied to a read-only ``float64`` array and must be nonempty, finite, positive, one-dimensional, and strictly increasing. ``results`` is converted to a tuple with exactly one entry per frequency. If a result exposes ``frequency_hz``, it must agree with the corresponding grid value to relative tolerance ``1e-12``. ``h5_path`` identifies an already-written sweep and is normalized to ``Path``; it may be ``None`` for an in-memory sweep.

``FrequencySweepResult.S``
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   sweep.S(
       side: str,
       *,
       out_mode: int = 0,
       in_mode: int = 0,
   ) -> numpy.ndarray

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``side``
     - Required
     - ``str``
     - Lead/trace side or physical monitor coordinate, according to the method signature. Integrated scattering currently launches from the left lead.
   * - ``out_mode``
     - Optional
     - ``int``
     - Mode, case, array, or mapping selector. Python indexing is zero based; explicit mode(number) and standalone visualization use their documented one-based numbers. Keyword-only. Default: ``0``.
   * - ``in_mode``
     - Optional
     - ``int``
     - Mode, case, array, or mapping selector. Python indexing is zero based; explicit mode(number) and standalone visualization use their documented one-based numbers. Keyword-only. Default: ``0``.

Calls the indexed ``ScatteringResult.S`` accessor for every result and returns a complex array in the same order as ``frequencies_hz``. ``side="left"`` selects reflected output and ``side="right"`` selects transmitted output. Missing mode combinations propagate the individual result's ``KeyError``.

Sweep array properties
^^^^^^^^^^^^^^^^^^^^^^

All properties allocate one numeric array ordered like ``frequencies_hz``:

* ``S11``: complex fundamental reflected amplitude, equivalent to ``S("left", out_mode=0, in_mode=0)``.

* ``S21``: complex fundamental transmitted amplitude, equivalent to ``S("right", out_mode=0, in_mode=0)``.

* ``reflection``: total reflected-power ratio from every point.

* ``transmission``: total transmitted-power ratio from every point.

* ``power_balance_error``: dimensionless power-balance error from every point.

* ``incident_power``: launched modal power in W/m at every point.

* ``radiated_power``: outward transverse radiation power in W/m at every point.

* ``absorbed_power``: integrated passive material absorption in W/m at every point.

* ``power_balance``: accounted output-power fraction ``(reflected + transmitted + radiated + absorbed) / incident`` at every point.

Use the indexed ``S`` method for higher-order or converted modes. The scalar power arrays include every propagating output mode represented by each point's power Gram; they are not necessarily ``abs(S11)**2`` or ``abs(S21)**2`` in a multimode lead.

``FrequencySweepResult.save_h5``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   sweep.save_h5(path: str | os.PathLike[str]) -> pathlib.Path

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``path``
     - Required
     - ``str | PathLike[str]``
     - Filesystem destination/source for HDF5 persistence or viewer launch. A directory is accepted only by viewer/directory-inspection APIs; None follows the method-specific default.

Writes all frequencies and complete per-point results with ``save_sweep_h5``. Modes are taken from each result's ``modes`` tuple. The absolute written path is returned, but the frozen sweep's ``h5_path`` is not mutated; integrated ``sweep_frequencies(h5_path=...)`` returns a copied object with that field set.

``Diagnostic``
~~~~~~~~~~~~~~

.. code-block:: python

   wf.Diagnostic(
       severity: Literal["info", "warning", "error"],
       code: str,
       message: str,
   )

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``severity``
     - Required
     - ``Literal['info', 'warning', 'error']``
     - Diagnostic code, human-readable explanation, or severity label used in a result diagnostic report.
   * - ``code``
     - Required
     - ``str``
     - Diagnostic code, human-readable explanation, or severity label used in a result diagnostic report.
   * - ``message``
     - Required
     - ``str``
     - Diagnostic code, human-readable explanation, or severity label used in a result diagnostic report.

A single machine-readable diagnostic. ``code`` is suitable for programmatic filtering; ``message`` is human-readable. Direct construction relies on callers respecting the annotations.

``DiagnosticReport``
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   wf.DiagnosticReport(diagnostics: tuple[wf.Diagnostic, ...])

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``diagnostics``
     - Required
     - ``tuple[Diagnostic, ...]``
     - Structured metadata/diagnostic container associated with the mesh or result. solve_info/metadata retain applied adaptive controls and stopping history.

* ``diagnostics`` preserves diagnostic order.

* ``ok`` is true when no item has severity ``"error"``.

* ``warnings`` returns only items whose severity is ``"warning"``.

Visualization scene API
-----------------------

Scene records are solver-neutral data persisted with a result for accurate post-processing. They use physical ``(x, z)`` storage order and SI metres. A viewer may transpose that presentation to put ``z`` horizontally, but it must not transpose the stored arrays. ``SceneKind`` is the module-level type alias ``Literal["pec", "pmc", "wave_port", "pml"]``.

``SceneLine``
~~~~~~~~~~~~~

.. code-block:: python

   wf.SceneLine(
       kind: str,
       endpoints: ArrayLike,
       label: str = "",
   )

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``kind``
     - Required
     - ``SceneKind | str``
     - Object/boundary/scene kind. Use the permitted Literal values or the documented selector for the owning class.
   * - ``endpoints``
     - Required
     - ``FloatArray``
     - Two physical endpoint coordinates, in computational-plane order, describing a scene line or boundary segment.
   * - ``label``
     - Optional
     - ``str``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Default: ``''``.

A frozen line-overlay record:

* ``kind`` is case-normalized to one of ``"pec"``, ``"pmc"``, ``"wave_port"``, or ``"pml"``. Other strings raise ``ValueError``.

* ``endpoints`` has shape ``(2, 2)``. Each row is one endpoint and each row is ordered ``(x, z)`` in metres. Values must be finite, real, and distinct.

* ``label`` is optional human-readable text stored in HDF5 and available to inspection tools.

Construction makes an owned, read-only ``float64`` copy of ``endpoints``. Malformed shapes, complex/non-finite coordinates, zero-length segments, and non-text kinds or labels raise ``ValueError``.

``Scene2D``
~~~~~~~~~~~

.. code-block:: python

   wf.Scene2D(
       points: ArrayLike,
       triangles: ArrayLike,
       eps_r: ArrayLike,
       x_span: tuple[float, float],
       z_span: tuple[float, float],
       lines: tuple[wf.SceneLine, ...] = (),
   )

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``points``
     - Required
     - ``FloatArray``
     - Coordinates used by the object or evaluation operation. Mesh geometry uses physical metres; low-level FE operators use their basis coordinate scale. See the array-shape conventions.
   * - ``triangles``
     - Required
     - ``IntArray``
     - Integer simplex connectivity. scikit-fem uses vertices-per-cell by cell; standalone exported geometry commonly uses cell by vertices-per-cell.
   * - ``eps_r``
     - Required
     - ``ComplexArray``
     - Relative permittivity. Scalar or Cartesian diagonal material inputs are supported where the signature permits; callbacks return scalar/broadcast-compatible values at physical coordinates.
   * - ``x_span``
     - Required
     - ``tuple[float, float]``
     - Physical axis bounds in metres, with upper > lower. Range-based solver constructors also accept a positive extent for (0, extent).
   * - ``z_span``
     - Required
     - ``tuple[float, float]``
     - Physical axis bounds in metres, with upper > lower. Range-based solver constructors also accept a positive extent for (0, extent).
   * - ``lines``
     - Optional
     - ``tuple[SceneLine, ...]``
     - Portable geometry overlay: physical points/triangles and named boundary, monitor, or PML lines for visualization. Default: ``fresh default container``.

A frozen full-domain material mesh and its overlay segments:

.. list-table::
   :header-rows: 1

   * - Field
     - Shape and meaning
   * - ``points``
     - ``(2, N)`` real mesh vertices in stored ``(x, z)`` order, metres
   * - ``triangles``
     - ``(3, M)`` integer vertex connectivity, one triangle per column
   * - ``eps_r``
     - ``(M,)`` complex physical relative permittivity at element centroids
   * - ``x_span``
     - Strictly increasing full-domain x limits in metres
   * - ``z_span``
     - Strictly increasing full-domain z limits in metres
   * - ``lines``
     - Tuple of ``SceneLine`` boundary, port, and PML overlays

``eps_r`` is the actual, untransformed physical material; the complex PML stretch is deliberately not folded into it. This lets a viewer shade dielectrics independently of the PML interface. The integrated solver writes four outer ``"pec"`` segments because its complete numerical outer boundary is homogeneous PEC, two ``"wave_port"`` segments at modal projection monitors, and every enabled internal PML interface as ``"pml"``. The ``"pmc"`` kind is supported for future/custom scene producers; the current high-level scattering solve rejects a PMC transverse boundary instead of fabricating one. Internal actual PEC sheets are written as additional ``"pec"`` segments. A finite slot therefore appears as collinear yellow ground segments separated by the actual opening, rather than as an unbroken background-sheet line.

Construction defensively copies ``points``, ``triangles``, and ``eps_r`` and marks them read-only. It validates finite values, exact integer connectivity, in-range and distinct indices, nondegenerate triangle area, one material value per triangle, strictly increasing spans, vertices and lines inside the domain, and that every line is a ``SceneLine``. Violations raise ``ValueError``.

HDF5 persistence API
--------------------

WaveFEM HDF5 files use ``SCHEMA_NAME`` with value ``"wavefem"`` and integer ``SCHEMA_VERSION`` with value ``1``. These constants are exported by ``wavefem.hdf5`` so external inspection tools can identify the format without duplicating magic values. Complex arrays use native HDF5 complex storage, numeric datasets are gzip-compressed, and arbitrary metadata is encoded as validated JSON.

Single and sweep files contain sampled fields and observables, not executable FEM objects. ``load_h5`` therefore works without reconstructing a geometry, mesh, sparse matrix, or solver backend. Writing and loading require a working ``h5py`` installation.

Each result may additionally contain an additive ``scene`` group with ``points``, ``triangles``, ``eps_r``, ``x_span``, ``z_span``, and ``lines/{kind,endpoints,label}`` datasets. The scene subgroup has format ``"wavefem-scene"``, version ``1``, and coordinate order ``"x,z"``. This extension does not change the root schema version: files created before scene support remain valid, and their loaded ``H5ResultData.scene`` is ``None``.

``save_result_h5``
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   wf.save_result_h5(
       result: object,
       path: str | os.PathLike[str],
       *,
       modes: Iterable[object] = (),
   ) -> pathlib.Path

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``result``
     - Required
     - ``object``
     - Solved result object(s) to persist, visualize, de-embed, or collect into a frequency sweep.
   * - ``path``
     - Required
     - ``os.PathLike[str] | str``
     - Filesystem destination/source for HDF5 persistence or viewer launch. A directory is accepted only by viewer/directory-inspection APIs; None follows the method-specific default.
   * - ``modes``
     - Optional
     - ``Iterable[object]``
     - Mode or mode collection. Integer selectors use the owning API indexing convention; supplied mode sets must match frequency, ky, and transverse geometry. Keyword-only. Default: ``()``.

Writes one duck-typed scattering result using schema version 1 and returns the resolved absolute path. The object must expose the same field arrays, S-parameter mapping, five power values, and result metadata as a ``ScatteringResult``. Optional ``frequency_hz``, ``ky``, ``modes``, and ``scene`` attributes are used when available. An optional scene may be a ``Scene2D`` or a duck-typed equivalent exposing the same fields and line attributes; it is fully normalized and validated before writing. Explicit ``modes`` takes precedence; the convenience ``ScatteringResult.save_h5`` passes ``result.modes`` here.

If explicit frequency metadata is unavailable, the writer may recover the frequency from compatible legacy ``solve_info["length_scale"]`` metadata or stored modes. Unknown single-result frequency is represented explicitly and loads as ``None``; it is never guessed from field samples.

The destination directory must already exist. The writer creates a temporary file in that directory, flushes it, and atomically replaces the destination, so an existing valid result is not replaced by a partially written file. Invalid arrays, keys, non-finite values, inconsistent total fields, invalid metadata, path errors, and unavailable HDF5 support raise ``ConfigurationError``.

``save_sweep_h5``
~~~~~~~~~~~~~~~~~

.. code-block:: python

   wf.save_sweep_h5(
       frequencies_hz: ArrayLike,
       results: Sequence[object],
       path: str | os.PathLike[str],
       *,
       modes_per_result: Sequence[Iterable[object]] | None = None,
   ) -> pathlib.Path

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``frequencies_hz``
     - Required
     - ``ArrayLike``
     - One-dimensional ordinary-frequency sequence in Hz, in the order of the associated results. Sweep solvers require positive, strictly increasing values.
   * - ``results``
     - Required
     - ``Sequence[object]``
     - Solved result object(s) to persist, visualize, de-embed, or collect into a frequency sweep.
   * - ``path``
     - Required
     - ``os.PathLike[str] | str``
     - Filesystem destination/source for HDF5 persistence or viewer launch. A directory is accepted only by viewer/directory-inspection APIs; None follows the method-specific default.
   * - ``modes_per_result``
     - Optional
     - ``Sequence[Iterable[object]] | None``
     - Mode or mode collection. Integer selectors use the owning API indexing convention; supplied mode sets must match frequency, ky, and transverse geometry. Keyword-only. Default: ``None``.

Writes one nonempty ordered result sequence and returns the absolute path. ``frequencies_hz`` must be a real positive 1D array with exactly one entry per result. ``Scattering2D.sweep_frequencies`` additionally requires strict increasing order before calling this lower-level writer.

When ``modes_per_result`` is ``None``, each result's optional ``modes`` attribute is used. Otherwise it must contain exactly one iterable per result and those explicit mode groups are stored. The supplied sweep frequency is forced into each prepared record and checked against any frequency metadata already on the result. Validation and atomic-replacement behavior match ``save_result_h5``.

``load_h5``
~~~~~~~~~~~

.. code-block:: python

   wf.load_h5(path: str | os.PathLike[str]) -> wf.H5FileData

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``path``
     - Required
     - ``os.PathLike[str] | str``
     - Filesystem destination/source for HDF5 persistence or viewer launch. A directory is accepted only by viewer/directory-inspection APIs; None follows the method-specific default.

Opens, fully reads, and validates one WaveFEM file, then closes it and returns portable in-memory records. Validation covers format name, supported schema version, single/sweep kind, result count, frequencies, field shapes and finiteness, the identities ``E_total = E_incident + E_scattered`` and ``H_total = H_incident + H_scattered``, S-parameter keys, nonnegative powers, mode shapes, metadata types, and every optional scene mesh/span/overlay. A missing file, corrupt HDF5 container, foreign format, unsupported schema version, or inconsistent dataset raises ``ValueError``. An unloadable ``h5py`` runtime raises ``ConfigurationError``.

The returned arrays are detached from the file, copied, and marked read-only. Mappings are read-only proxies. No live ``h5py.File`` handle remains.

``H5FileData``
~~~~~~~~~~~~~~

.. code-block:: python

   wf.H5FileData(
       path: pathlib.Path,
       kind: Literal["single", "sweep"],
       frequencies_hz: numpy.ndarray,
       results: tuple[wf.H5ResultData, ...],
   )

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
   * - ``kind``
     - Required
     - ``Literal['single', 'sweep']``
     - Object/boundary/scene kind. Use the permitted Literal values or the documented selector for the owning class.
   * - ``frequencies_hz``
     - Required
     - ``FloatArray``
     - One-dimensional ordinary-frequency sequence in Hz, in the order of the associated results. Sweep solvers require positive, strictly increasing values.
   * - ``results``
     - Required
     - ``tuple[H5ResultData, ...]``
     - Solved result object(s) to persist, visualize, de-embed, or collect into a frequency sweep.

The top-level frozen loader record:

* ``path``: resolved source file.

* ``kind``: ``"single"`` for one run or ``"sweep"`` for an ordered sweep.

* ``frequencies_hz``: read-only ``float64`` values. An unknown single-run frequency uses the schema's NaN sentinel at this top level; the associated ``H5ResultData.frequency_hz`` is ``None``.

* ``results``: one complete portable record per stored point.

``H5ResultData``
~~~~~~~~~~~~~~~~

.. code-block:: python

   wf.H5ResultData(
       frequency_hz,
       ky,
       coordinates,
       E_incident,
       E_scattered,
       E_total,
       H_incident,
       H_scattered,
       H_total,
       s_parameters,
       powers,
       modes,
       metadata,
       scene: wf.Scene2D | None = None,
   )

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``frequency_hz``
     - Required
     - ``float | None``
     - Ordinary frequency in Hz; must be finite and positive.
   * - ``ky``
     - Required
     - ``float | None``
     - Real Fourier wavenumber along invariant y, in rad/m; zero gives propagation in the computational plane.
   * - ``coordinates``
     - Required
     - ``FloatArray``
     - Coordinates used by the object or evaluation operation. Mesh geometry uses physical metres; low-level FE operators use their basis coordinate scale. See the array-shape conventions.
   * - ``E_incident``
     - Required
     - ``ComplexArray``
     - Complex sampled incident/scattered/total field array, shaped (3, number of sample points), in V/m for E or A/m for H.
   * - ``E_scattered``
     - Required
     - ``ComplexArray``
     - Complex sampled incident/scattered/total field array, shaped (3, number of sample points), in V/m for E or A/m for H.
   * - ``E_total``
     - Required
     - ``ComplexArray``
     - Complex sampled incident/scattered/total field array, shaped (3, number of sample points), in V/m for E or A/m for H.
   * - ``H_incident``
     - Required
     - ``ComplexArray``
     - Complex sampled incident/scattered/total field array, shaped (3, number of sample points), in V/m for E or A/m for H.
   * - ``H_scattered``
     - Required
     - ``ComplexArray``
     - Complex sampled incident/scattered/total field array, shaped (3, number of sample points), in V/m for E or A/m for H.
   * - ``H_total``
     - Required
     - ``ComplexArray``
     - Complex sampled incident/scattered/total field array, shaped (3, number of sample points), in V/m for E or A/m for H.
   * - ``s_parameters``
     - Required
     - ``Mapping[PortKey, complex]``
     - Mapping from port/mode channel keys to complex scattering amplitudes, normalized to the incident mode amplitude.
   * - ``powers``
     - Required
     - ``Mapping[str, float]``
     - Modal or electromagnetic power in W/m of invariant length for 2.5D fields. Complex power retains reactive flux; power ratios are reported separately.
   * - ``modes``
     - Required
     - ``tuple[H5ModeData, ...]``
     - Mode or mode collection. Integer selectors use the owning API indexing convention; supplied mode sets must match frequency, ky, and transverse geometry.
   * - ``metadata``
     - Required
     - ``Mapping[str, Any]``
     - Structured metadata/diagnostic container associated with the mesh or result. solve_info/metadata retain applied adaptive controls and stopping history.
   * - ``scene``
     - Optional
     - ``Scene2D | None``
     - Portable geometry overlay: physical points/triangles and named boundary, monitor, or PML lines for visualization. Default: ``None``.

The frozen portable representation of one result. ``frequency_hz`` is positive hertz or ``None``; ``ky`` is rad/m or ``None``; ``coordinates`` has shape ``(2, N)``; and every E/H field has shape ``(3, N)`` in ``(x,y,z)`` component order. ``s_parameters`` maps ``(side, out_mode, in_mode)`` to complex amplitude. ``powers`` maps ``reflected_power``, ``transmitted_power``, ``radiated_power``, ``absorbed_power``, and ``incident_power`` to W/m. ``modes`` is a tuple of ``H5ModeData``; ``metadata`` contains the serializable result metadata collected from the original object. ``scene`` is the validated full-domain material and overlay record, or ``None`` for an older/schema-v1 file without the optional group. Scene arrays and line endpoints are detached read-only copies.

Unlike ``ScatteringResult``, this record intentionally has no solver, de-embedding, or FEM interpolation methods. Plot it through the viewer/helper APIs or inspect its arrays directly.

``H5ModeData``
~~~~~~~~~~~~~~

.. code-block:: python

   wf.H5ModeData(
       x,
       E,
       H,
       metadata,
       raw_components,
   )

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``x``
     - Required
     - ``FloatArray``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature.
   * - ``E``
     - Required
     - ``ComplexArray``
     - Complex electromagnetic field samples or FE field objects. Cartesian components follow x,y,z order; left/right denote the selected trace side.
   * - ``H``
     - Required
     - ``ComplexArray``
     - Complex electromagnetic field samples or FE field objects. Cartesian components follow x,y,z order; left/right denote the selected trace side.
   * - ``metadata``
     - Required
     - ``Mapping[str, Any]``
     - Structured metadata/diagnostic container associated with the mesh or result. solve_info/metadata retain applied adaptive controls and stopping history.
   * - ``raw_components``
     - Required
     - ``Mapping[str, NDArray[Any]]``
     - Visualization component preset or mapping of named complex field-component arrays.

One portable sampled lead mode. ``x`` is a read-only transverse grid in metres; ``E`` and ``H`` have shape ``(3, n)`` in Cartesian order. ``metadata`` contains modal scalars and labels such as ``beta``, ``neff``, ``power``, ``complex_power``, ``ky``, ``omega``, ``direction``, ``classification``, ``normalization``, and residuals. ``raw_components`` preserves available mixed-representation arrays such as ``x_nodes``, cellwise ``E_x``, nodal ``E_y/E_z``, magnetic samples, and endpoint ``H_x`` traces for research inspection.

Separate HDF5 viewer project
----------------------------

The GUI is intentionally not part of the ``wavefem`` Python distribution. The sibling `WaveFEMViewer project <../WaveFEMViewer/README.rst>`_ owns its native C++20/Qt source, ``wavefem-viewer`` executable, lazy HDF5 reader, cached QPainter renderer, deployment scripts, and user documentation. It does not import or depend on Python, NumPy, h5py, Matplotlib, or ``wavefem``, and can therefore inspect result files on a machine without the FEM solver installed.

Its README documents cross-platform CMake builds, installation, direct-path launch, file-picker workflow, tab controls, supported schema data, and the headless inspection/benchmark utility. Python exposes ``wf.find_viewer_executable()`` and ``wf.launch_viewer(path=None)``. Discovery checks ``WAVEFEM_VIEWER_EXECUTABLE``, standalone ``WaveFEMViewer/build*`` trees, repository-root ``build*/WaveFEMViewer`` trees, ``PATH``, and the default Windows installation, in that order. A directory target populates the native viewer's HDF5 selector; ``None`` targets the current directory.

``FrequencySweepResult.visualize()`` draws S11/S21 with Matplotlib. ``FrequencySweepResult.visualize_with_gui()`` reuses an associated archive or saves ``wavefem_sweep.h5``, then opens every sweep point and stored mode. The ``wavefem-inspect-h5`` command is headless by default; ``--gui`` launches the native viewer, and ``--gui`` without a path opens the current directory. For every 2D vector/material plot it displays ``z`` horizontally and ``x`` vertically while leaving file storage in ``(x, z)`` order. Dielectric material is grey, PEC is yellow, PMC is blue, wave ports are red, and PML interfaces are green dashed lines.

Standalone mode API
-------------------

``CrossSection``
~~~~~~~~~~~~~~~~

.. code-block:: python

   wf.CrossSection(
       x_span: tuple[float, float],
       background: wf.Material = wf.Material(),
       boundary: Literal["pec", "pmc"] | None = None,
       layers: list[wavefem.modes.Layer] = [],
       pml: wf.PML | None = None,
       pec_boundaries: list[wavefem.modes.PECBoundary] = [],
   )

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``x_span``
     - Required
     - ``tuple[float, float]``
     - Physical axis bounds in metres, with upper > lower. Range-based solver constructors also accept a positive extent for (0, extent).
   * - ``background``
     - Optional
     - ``Material``
     - In placement methods, True adds a z-invariant background region/sheet; False adds an actual-device perturbation. In material/geometry constructors, this is the exterior Material object. Default: ``fresh default container``.
   * - ``boundary``
     - Optional
     - ``BoundaryKind | None``
     - Outer transverse boundary condition. Modal solvers support PEC/PMC as documented; integrated WaveFEM supports PEC or transverse PML terminated by PEC. None is allowed while recording geometry, but assemble/solve requires an explicit truncation boundary. Default: ``None``.
   * - ``layers``
     - Optional
     - ``list[Layer]``
     - Ordered layer/region/boundary specifications. Later overlapping material regions take precedence where the geometry API permits overlap. Default: ``fresh default container``.
   * - ``pml``
     - Optional
     - ``PML | None``
     - PML specification/layout describing the absorbing strips and their grading profile. Default: ``None``.
   * - ``pec_boundaries``
     - Optional
     - ``list[PECBoundary]``
     - PEC sheet/plane handle(s) that define zero tangential electric field. Slots must refer to a background sheet. Default: ``fresh default container``.

Represents a z-uniform one-dimensional material profile.

* ``x_span`` is the finite transverse interval in metres.

* ``background`` fills the interval outside explicit layers.

* ``boundary`` must be explicitly ``"pec"`` or ``"pmc"`` before mode assembly. ``None`` prevents accidental use of a closed box as an open guide.

* ``layers`` contains non-overlapping material intervals. Prefer ``add_layer``; directly pre-populating this list bypasses its overlap/name checks.

* ``pml`` places equal transverse PMLs inside both ends. A mode PML requires a PEC outer wall and must leave a physical interior.

* ``pec_boundaries`` contains zero-thickness internal PEC points. Prefer ``add_pec``; directly supplied entries must be ``PECBoundary`` instances and are revalidated during construction.

``add_layer``
^^^^^^^^^^^^^

.. code-block:: python

   cross_section.add_layer(
       *,
       x: Sequence[float],
       material: wf.Material,
       name: str | None = None,
   ) -> wavefem.modes.Layer

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``x``
     - Required
     - ``Sequence[float]``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature. Keyword-only.
   * - ``material``
     - Required
     - ``Material``
     - Material object defining relative electric and magnetic response. Exterior fills points outside placed material regions. Keyword-only.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Adds a non-overlapping, mesh-conforming material interval. The interval must be inside ``x_span`` and its name must be unique.

``add_pec``
^^^^^^^^^^^

.. code-block:: python

   cross_section.add_pec(
       *,
       x: float,
       name: str | None = None,
   ) -> wavefem.modes.PECBoundary

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``x``
     - Required
     - ``float``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature. Keyword-only.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Adds a named, zero-thickness PEC sheet invariant in z. ``x`` is a finite physical coordinate strictly inside ``x_span``; duplicate coordinates and duplicate PEC names raise ``ConfigurationError``. The returned immutable ``PECBoundary(x, name)`` record is retained in the sorted ``cross_section.pec_boundaries`` list.

The coordinate is guaranteed to be a mode-mesh node. Assembly constrains the nodal tangential components ``E_y`` and ``E_z`` at that node for either PEC or PMC outer truncation. Every cellwise ``E_x`` DOF remains free, preserving independent normal traces and surface charge on the two sheet faces. The sheet node is also omitted from bulk divergence-test rows, because the surface charge makes a volume Gauss-law residual inappropriate there.

``interfaces``
^^^^^^^^^^^^^^

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

Returns sorted outer boundaries and all material, internal-PEC, and PML interfaces in metres. The mode mesh always conforms to these coordinates.

``material_at``
^^^^^^^^^^^^^^^

.. code-block:: python

   cross_section.material_at(x) -> tuple[eps_r_array, mu_r_array]

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

Evaluates scalar physical material values at arbitrary x coordinates. Layer endpoints are included in their layer masks.

``diagonal_material_at``
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   cross_section.diagonal_material_at(
       x,
       *,
       k_reference: float,
   ) -> tuple[eps_diagonal, mu_diagonal]

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
   * - ``k_reference``
     - Required
     - ``float``
     - Vacuum wavenumber in rad/m, used for coordinate scaling and material/PML evaluation. Keyword-only.

Returns arrays with leading component order ``(x, y, z)``. With no PML, all three diagonal entries equal the scalar material. In a transverse PML the transformation-optics factor is ``(1/sx, sx, sx)``.

``ModeSolver``
~~~~~~~~~~~~~~

.. code-block:: python

   wf.ModeSolver(
       cross_section: wf.CrossSection,
       *,
       frequency: float | None = None,
       omega: float | None = None,
       wavelength: float | None = None,
       ky: float = 0.0,
       num_elements: int = 24,
       quadrature_order: int = 4,
       dense_linearization_limit: int = 420,
   )

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``cross_section``
     - Required
     - ``CrossSection``
     - One-dimensional background guide with physical x bounds, layers, truncation, and optional PEC/PML specifications.
   * - ``frequency``
     - Conditional: exactly one frequency input
     - ``float | None``
     - Ordinary frequency in Hz; must be finite and positive. Keyword-only. Default: ``None``.
   * - ``omega``
     - Conditional: exactly one frequency input
     - ``float | None``
     - Angular frequency in rad/s; must be finite and positive. Keyword-only. Default: ``None``.
   * - ``wavelength``
     - Conditional: exactly one frequency input
     - ``float | None``
     - Vacuum wavelength in metres; must be finite and positive. Keyword-only. Default: ``None``.
   * - ``ky``
     - Optional
     - ``float``
     - Real Fourier wavenumber along invariant y, in rad/m; zero gives propagation in the computational plane. Keyword-only. Default: ``0.0``.
   * - ``num_elements``
     - Optional
     - ``int``
     - Initial number of 1D intervals; interfaces can add mesh nodes. Adaptive passes may increase this count. Keyword-only. Default: ``24``.
   * - ``quadrature_order``
     - Optional
     - ``int``
     - Finite-element integration order. Higher-order mixed elements require at least fourth-order quadrature; PML and material variation can require more. Keyword-only. Default: ``4``.
   * - ``dense_linearization_limit``
     - Optional
     - ``int``
     - Matrix-size cutoff for dense eigensolving; larger systems use a sparse backend. This is a dimension limit, not a mesh-error threshold. Keyword-only. Default: ``420``.

Solves the full-vector fixed-frequency, fixed-``ky`` quadratic eigenproblem for ``neff = beta/k0``.

* ``cross_section`` supplies material layers and transverse truncation.

* ``frequency`` is the preferred ordinary-frequency input in hertz. ``omega`` (rad/s) and ``wavelength`` (metres) remain mutually exclusive compatibility alternatives; exactly one spectral argument is required.

* ``ky`` must currently be finite and real.

* ``num_elements`` controls the target one-dimensional mesh resolution and must be at least two. Added material/PML interfaces can change the exact element count.

* ``quadrature_order`` must be at least two.

* ``dense_linearization_limit`` selects dense generalized-QZ below the given linearized matrix size; larger problems use sparse shift-invert.

The electric representation is cellwise ``E_x`` plus nodal ``E_y,E_z``, matching the trace of the 2D mixed Nedelec-H1 space.

``assemble``
^^^^^^^^^^^^

.. code-block:: python

   solver.assemble() -> wavefem.modes.ModeFEMSystem

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

Assembles and returns the dimensionless quadratic pencil ``A0 + neff*A1 + neff**2*A2`` and diagnostic operators. An explicit cross-section boundary is required.

``solve``
^^^^^^^^^

.. code-block:: python

   solver.solve(
       *,
       num_modes: int = 4,
       neff_guess: complex | None = None,
       direction: Literal["forward", "backward", "all"] = "forward",
       eigensolver_tolerance: float = 1e-10,
       residual_tolerance: float = 1e-8,
       divergence_tolerance: float = 1e-7,
       propagation_ratio_tolerance: float = 1e-3,
       max_refinements: int = 2,
       adaptive_tolerance: float = 0.05,
   ) -> wf.ModeSet

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``num_modes``
     - Optional
     - ``int``
     - Number of validated forward/backward modes to return; positive integer.
   * - ``neff_guess``
     - Optional
     - ``complex | None``
     - Dimensionless effective index beta/k0. A guess selects roots near that complex value; None uses the solver estimate. Keyword-only. Default: ``None``.
   * - ``direction``
     - Optional
     - ``RequestedDirection``
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
   * - ``max_refinements``
     - Optional
     - ``int``
     - Maximum mesh updates after the initial solve; nonnegative integer. Zero keeps the initial mesh. Solver default is 2; examples explicitly use 0. Keyword-only. Default: ``2``.
   * - ``adaptive_tolerance``
     - Optional
     - ``float``
     - Positive finite threshold for the normalized discretization estimator. Stop when residual <= threshold. Independent of algebraic tolerances; exhausting the budget is not convergence. Keyword-only. Default: ``0.05``.

Finds roots nearest ``neff_guess``. When no guess is supplied, a value is derived from the largest local index after accounting for ``ky/k0``. Mode ordering follows proximity to that guess and is not a permanent physical mode identifier.

Candidate roots must pass:

* the quadratic-pencil relative residual;

* a weak ``div(eps_r E)=0`` residual;

* the requested direction classification;

* duplicate-mode rejection.

A mode is classified as propagating when the real fraction of its complex power exceeds ``propagation_ratio_tolerance``. Propagating modes are normalized to unit absolute real power. Other roots receive an energy-like normalization and a decay direction. If fewer than ``num_modes`` pass all checks, ``ModeSolverError`` explains the rejection counts.

Standalone x-PML solves may also return discretized radiation/PML candidates; the integrated ``Scattering2D`` path applies an additional bound-mode filter.

``ModeSet``
~~~~~~~~~~~

.. code-block:: python

   wf.ModeSet(
       modes: tuple[wf.Mode, ...],
       system: wavefem.modes.ModeFEMSystem,
       solve_info: dict[str, object],
   )

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``modes``
     - Required
     - ``tuple[Mode, ...]``
     - Mode or mode collection. Integer selectors use the owning API indexing convention; supplied mode sets must match frequency, ky, and transverse geometry.
   * - ``system``
     - Required
     - ``ModeFEMSystem``
     - Assembled FEM system containing compatible bases, sparse operators, constraints, and material/reference-scale metadata.
   * - ``solve_info``
     - Required
     - ``dict[str, object]``
     - Structured metadata/diagnostic container associated with the mesh or result. solve_info/metadata retain applied adaptive controls and stopping history.

An immutable ``Sequence[Mode]``. It supports ``len(modes)``, iteration, integer indexing, and slicing. A slice returns a plain tuple. ``system`` exposes the assembled eigenproblem and ``solve_info`` records method, candidate count, guess, direction, and residual tolerances. Immutability is shallow: ``solve_info`` itself remains a mutable dictionary.

``Mode``
~~~~~~~~

``Mode`` is an immutable normalized modal family member. It is normally created by ``ModeSolver``; direct construction performs no additional post-init array validation.

.. code-block:: python

   wf.Mode(
       beta: complex,
       neff: complex,
       E_x,
       E_y,
       E_z,
       H_x,
       H_y,
       H_z,
       x_nodes,
       power: float,
       complex_power: complex,
       ky: float,
       omega: float,
       direction,
       classification,
       normalization,
       residual: float,
       divergence_residual: float,
       H_x_left=None,
       H_x_right=None,
   )

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``beta``
     - Required
     - ``complex``
     - Complex longitudinal propagation constant(s), in rad/m.
   * - ``neff``
     - Required
     - ``complex``
     - Dimensionless effective index beta/k0. A guess selects roots near that complex value; None uses the solver estimate.
   * - ``E_x``
     - Required
     - ``ComplexArray``
     - Complex electromagnetic field samples or FE field objects. Cartesian components follow x,y,z order; left/right denote the selected trace side.
   * - ``E_y``
     - Required
     - ``ComplexArray``
     - Complex electromagnetic field samples or FE field objects. Cartesian components follow x,y,z order; left/right denote the selected trace side.
   * - ``E_z``
     - Required
     - ``ComplexArray``
     - Complex electromagnetic field samples or FE field objects. Cartesian components follow x,y,z order; left/right denote the selected trace side.
   * - ``H_x``
     - Required
     - ``ComplexArray``
     - Complex electromagnetic field samples or FE field objects. Cartesian components follow x,y,z order; left/right denote the selected trace side.
   * - ``H_y``
     - Required
     - ``ComplexArray``
     - Complex electromagnetic field samples or FE field objects. Cartesian components follow x,y,z order; left/right denote the selected trace side.
   * - ``H_z``
     - Required
     - ``ComplexArray``
     - Complex electromagnetic field samples or FE field objects. Cartesian components follow x,y,z order; left/right denote the selected trace side.
   * - ``x_nodes``
     - Required
     - ``FloatArray``
     - Strictly increasing physical 1D mesh-node coordinates, in metres.
   * - ``power``
     - Required
     - ``float``
     - Modal or electromagnetic power in W/m of invariant length for 2.5D fields. Complex power retains reactive flux; power ratios are reported separately.
   * - ``complex_power``
     - Required
     - ``complex``
     - Modal or electromagnetic power in W/m of invariant length for 2.5D fields. Complex power retains reactive flux; power ratios are reported separately.
   * - ``ky``
     - Required
     - ``float``
     - Real Fourier wavenumber along invariant y, in rad/m; zero gives propagation in the computational plane.
   * - ``omega``
     - Required
     - ``float``
     - Angular frequency in rad/s; must be finite and positive.
   * - ``direction``
     - Required
     - ``ModeDirection``
     - Propagation filter forward/backward/all for mode solves; for PML placement, the selected transverse side(s), such as x-, x+, or x.
   * - ``classification``
     - Required
     - ``ModeClassification``
     - Mode-family label: TE/TM/hybrid or propagating/evanescent as appropriate to the owning result or solver.
   * - ``normalization``
     - Required
     - ``Literal['unit-power', 'energy-like']``
     - Modal/field normalization convention or flag. Unit-longitudinal-power normalization applies only to modes with usable real power.
   * - ``residual``
     - Required
     - ``float``
     - Stored numerical-validation diagnostic. Algebraic and Gauss residuals measure discrete equation defects; pml_fraction measures energy in absorbing layers.
   * - ``divergence_residual``
     - Required
     - ``float``
     - Stored numerical-validation diagnostic. Algebraic and Gauss residuals measure discrete equation defects; pml_fraction measures energy in absorbing layers.
   * - ``H_x_left``
     - Optional
     - ``ComplexArray | None``
     - Complex electromagnetic field samples or FE field objects. Cartesian components follow x,y,z order; left/right denote the selected trace side. Default: ``None``.
   * - ``H_x_right``
     - Optional
     - ``ComplexArray | None``
     - Complex electromagnetic field samples or FE field objects. Cartesian components follow x,y,z order; left/right denote the selected trace side. Default: ``None``.

.. list-table::
   :header-rows: 1

   * - Field
     - Meaning
   * - ``beta``
     - Complex propagation constant, rad/m
   * - ``neff``
     - ``beta/k0``
   * - ``E_x``
     - Cellwise electric coefficients, ``(Ncell,)``
   * - ``E_y``, ``E_z``
     - Nodal electric coefficients, ``(Nnode,)``
   * - ``H_x``, ``H_y``, ``H_z``
     - Cell-centred magnetic samples, ``(Ncell,)``
   * - ``H_x_left``, ``H_x_right``
     - Optional per-cell endpoint traces for accurate interpolation
   * - ``x_nodes``
     - Strictly increasing physical mesh nodes, m
   * - ``power``
     - Signed real longitudinal power after normalization, W/m
   * - ``complex_power``
     - Complex longitudinal Poynting flux
   * - ``ky``, ``omega``
     - Spectral parameters used to create the mode
   * - ``direction``
     - ``forward``, ``backward``, ``right-decaying``, ``left-decaying``, or ``indeterminate``
   * - ``classification``
     - ``propagating`` or ``evanescent``
   * - ``normalization``
     - ``unit-power`` or ``energy-like``
   * - ``residual``
     - Relative quadratic-pencil residual
   * - ``divergence_residual``
     - Normalized weak Gauss-law residual

Properties:

* ``x``: cell-centre x coordinates.

* ``E``: cell-centred ``(Ex,Ey,Ez)`` array.

* ``H``: cell-centred ``(Hx,Hy,Hz)`` array.

* ``is_propagating``: true when ``classification == "propagating"``.

``sample_E`` and ``sample_H``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   mode.sample_E(x) -> complex ndarray
   mode.sample_H(x) -> complex ndarray

Evaluate transverse traces with output shape ``(3, *np.shape(x))``. Coordinates must be finite, real, and within ``x_nodes``. ``E_x`` and ``H_y,H_z`` retain their cellwise representation; nodal electric components are linearly interpolated. ``H_x`` uses per-cell linear endpoint interpolation when endpoint data are available.

At an internal boundary, a cellwise value uses the cell immediately to the right; the final outer endpoint uses the last cell.

``fields``
^^^^^^^^^^

.. code-block:: python

   mode.fields(
       x,
       z,
       reference_plane: float = 0.0,
   ) -> tuple[E, H]

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
   * - ``z``
     - Required
     - ``ArrayLike``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature.
   * - ``reference_plane``
     - Optional
     - ``float``
     - Physical longitudinal reference coordinate(s) in metres at which incident or scattering amplitudes are defined. Default: ``0.0``.

Broadcasts x and z and multiplies both traces by ``exp(i*beta*(z-reference_plane))``.

``phase_factor``
^^^^^^^^^^^^^^^^

.. code-block:: python

   mode.phase_factor(z, *, reference_plane: float = 0.0)

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``z``
     - Required
     - ``ArrayLike``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature.
   * - ``reference_plane``
     - Optional
     - ``float``
     - Physical longitudinal reference coordinate(s) in metres at which incident or scattering amplitudes are defined. Keyword-only. Default: ``0.0``.

Returns only the complex propagation factor.

``counterpropagating`` and ``backward``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``counterpropagating()`` returns the exact z-reflected family member: ``beta -> -beta``, ``E -> (Ex,Ey,-Ez)``, and ``H -> (-Hx,-Hy,Hz)``. It is a spatial reflection, not complex conjugation, and reverses signed power.

``backward()`` returns the object unchanged if it already propagates/decays toward negative z; otherwise it calls ``counterpropagating()``.

Incident-field API
------------------

``IncidentMode``
~~~~~~~~~~~~~~~~

.. code-block:: python

   wf.IncidentMode(
       mode: wf.Mode,
       side: Literal["left", "right"] = "left",
       reference_plane: float = 0.0,
       amplitude: complex = 1.0 + 0.0j,
   )

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``mode``
     - Required
     - ``Mode``
     - Mode or mode collection. Integer selectors use the owning API indexing convention; supplied mode sets must match frequency, ky, and transverse geometry.
   * - ``side``
     - Optional
     - ``IncidentSide``
     - Lead/trace side or physical monitor coordinate, according to the method signature. Integrated scattering currently launches from the left lead. Default: ``'left'``.
   * - ``reference_plane``
     - Optional
     - ``float``
     - Physical longitudinal reference coordinate(s) in metres at which incident or scattering amplitudes are defined. Default: ``0.0``.
   * - ``amplitude``
     - Optional
     - ``complex``
     - Complex modal amplitude(s); squared magnitude is power for a unit-power propagating mode. Integrated incidence requires a nonzero finite launch amplitude. Default: ``(1+0j)``.

``IncidentSide`` is the type alias ``Literal["left", "right"]`` in ``wavefem.incident``.

Wraps a lead mode as an analytic incident field. If the supplied mode points away from the requested launch direction, its exact counterpropagating member is selected automatically. Indeterminate modes are rejected.

Properties:

* ``direction``: classification of the actually launched mode.

* ``beta``: propagation constant including its launch-direction sign.

* ``signed_power``: ``abs(amplitude)**2 * mode.power``.

Methods:

.. code-block:: python

   incident.fields(x, z) -> tuple[E, H]
   incident.E(x, z) -> E
   incident.H(x, z) -> H
   incident(x, z) -> E

All methods accept broadcast-compatible coordinates in metres. The callable form aliases ``E`` so an ``IncidentMode`` can be passed directly to equivalent source assembly.

Standalone ``IncidentMode`` permits energy-normalized evanescent fields for research use; ``Scattering2D.set_incident_mode`` deliberately restricts the integrated power workflow to propagating unit-power modes.

Frequency API
-------------

``Frequency``
~~~~~~~~~~~~~

.. code-block:: python

   wf.Frequency(omega: float)

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``omega``
     - Required
     - ``float``
     - Required positive angular frequency in rad/s; use Frequency.from_frequency for Hz.

An immutable canonical spectral point. Direct construction interprets the argument as positive angular frequency in radians per second.

Named constructors:

.. code-block:: python

   wf.Frequency.from_wavelength(wavelength_m)
   wf.Frequency.from_frequency(frequency_hz)
   wf.Frequency.from_omega(omega_rad_per_s)

Derived read-only properties:

* ``angular_frequency`` and ``omega``: rad/s.

* ``frequency``: Hz.

* ``wavelength``: vacuum wavelength in metres.

* ``k0``: vacuum angular wavenumber in rad/m.

All inputs must be finite, real, and strictly positive.

``resolve_frequency``
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   wf.resolve_frequency(
       *,
       frequency=None,
       omega=None,
       wavelength=None,
   ) -> wf.Frequency

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``frequency``
     - Conditional: exactly one frequency input
     - ``RealInput | None``
     - Ordinary frequency in Hz; must be finite and positive. Keyword-only. Default: ``None``.
   * - ``omega``
     - Conditional: exactly one frequency input
     - ``RealInput | None``
     - Angular frequency in rad/s; must be finite and positive. Keyword-only. Default: ``None``.
   * - ``wavelength``
     - Conditional: exactly one frequency input
     - ``RealInput | None``
     - Vacuum wavelength in metres; must be finite and positive. Keyword-only. Default: ``None``.

Requires exactly one non-``None`` specification. Supplying none or more than one raises ``ConfigurationError`` even when multiple values are numerically consistent.

Material API
------------

``Material``
~~~~~~~~~~~~

.. code-block:: python

   wf.Material(
       eps_r: complex = 1.0 + 0.0j,
       mu_r: complex = 1.0 + 0.0j,
   )

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``eps_r``
     - Optional
     - ``complex``
     - Relative permittivity. Scalar or Cartesian diagonal material inputs are supported where the signature permits; callbacks return scalar/broadcast-compatible values at physical coordinates. Default: ``(1+0j)``.
   * - ``mu_r``
     - Optional
     - ``complex``
     - Relative permeability, scalar or Cartesian diagonal where supported. WaveFEM scattering requires equal actual and background permeability. Default: ``(1+0j)``.

An immutable isotropic material using relative constitutive scalars. Values must be finite. Arrays, sequences, mappings, and tensor-like inputs raise ``NotImplementedError`` rather than being silently reduced to a scalar.

For ``exp(-i*omega*t)``, passive loss normally has nonnegative imaginary permittivity/permeability.

* ``is_lossless`` is true only when both imaginary parts are exactly zero.

* ``is_passive`` is true when both imaginary parts are nonnegative.

The low-level material object can represent active values for research. ``Scattering2D`` rejects active materials because its integrated power accounting is passive-only.

Zero or negative real constitutive values are not rejected by ``Material`` itself. Whether such a model is physically and numerically appropriate is a solver-level responsibility.

PML API
-------

``PML``
~~~~~~~

.. code-block:: python

   wf.PML(
       thickness: float,
       order: int = 3,
       target_reflection: float = 1e-8,
   )

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
   * - ``target_reflection``
     - Optional
     - ``float``
     - Target PML amplitude reflection used to derive its attenuation profile; strictly between zero and one. Default: ``1e-08``.

An immutable polynomial complex-stretch specification.

* ``thickness`` is positive and measured in metres.

* ``order`` is a positive polynomial order.

* ``target_reflection`` is a nominal amplitude target strictly between zero and one, not a guaranteed achieved reflection for a finite discretization.

``maximum_imaginary_stretch``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   pml.maximum_imaginary_stretch(k_reference: float) -> float

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``k_reference``
     - Required
     - ``float``
     - Vacuum wavenumber in rad/m, used for coordinate scaling and material/PML evaluation.

Computes the profile peak that gives the nominal two-pass reflection target for a positive reference wavenumber in rad/m.

``stretch``
^^^^^^^^^^^

.. code-block:: python

   pml.stretch(depth, k_reference) -> complex ndarray

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
   * - ``k_reference``
     - Required
     - ``float``
     - Vacuum wavenumber in rad/m, used for coordinate scaling and material/PML evaluation.

Evaluates ``s = 1 + i*alpha_max*(clip(depth,0,thickness)/thickness)**order``. Depth and reference wavenumber use SI units.

``PMLLayout``
~~~~~~~~~~~~~

.. code-block:: python

   wf.PMLLayout(x: wf.PML | None = None, z: wf.PML | None = None)

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``x``
     - Optional
     - ``PML | None``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature. Default: ``None``.
   * - ``z``
     - Optional
     - ``PML | None``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature. Default: ``None``.

Groups independent symmetric x- and z-directed PML specifications.

* ``validate_domain(x_span, z_span)`` checks that two PMLs leave an interior.

* ``stretching(x, z, *, x_span, z_span, k_reference)`` returns broadcast ``(sx, sz)`` arrays.

* ``transform_isotropic(eps_r, mu_r, sx, sz)`` returns diagonal tensors in ``(x,y,z)`` order. The transformation factors are ``(sz/sx, sx*sz, sx/sz)``.

* ``interfaces(x_span, z_span)`` returns the internal x and z PML-interface coordinates that a conforming mesh should include.

Constants
---------

The following compact constants are exported at the top level and are sourced from ``scipy.constants``:

.. list-table::
   :header-rows: 1

   * - Name
     - Meaning
     - Unit
   * - ``C0``
     - vacuum speed of light
     - m/s
   * - ``EPSILON_0``
     - vacuum permittivity
     - F/m
   * - ``MU_0``
     - vacuum permeability
     - H/m
   * - ``ETA_0``
     - ``sqrt(MU_0/EPSILON_0)`` vacuum impedance
     - ohm

``wavefem.constants`` also provides descriptive aliases: ``SPEED_OF_LIGHT_M_PER_S``, ``VACUUM_PERMITTIVITY_F_PER_M``, ``VACUUM_PERMEABILITY_H_PER_M``, and ``VACUUM_IMPEDANCE_OHM``.

Exception hierarchy
-------------------

.. code-block:: text

   WaveFEMError
   ├── ConfigurationError
   │   └── MaterialError
   ├── MeshError
   ├── ModeSolverError
   ├── ModeProjectionError
   ├── SolverError
   └── ViewerError

* ``WaveFEMError``: base class for actionable package errors.

* ``ConfigurationError``: incomplete or inconsistent simulation input.

* ``MaterialError``: invalid material scalar or representation.

* ``MeshError``: Gmsh generation, import, or region-tagging failure.

* ``ModeSolverError``: requested validated eigenmodes could not be produced.

* ``ModeProjectionError``: monitor fields could not be reliably decomposed.

* ``SolverError``: FEM linear/eigenvalue solution failed.

* ``ViewerError``: the standalone native viewer could not be found or launched.

Standard ``ValueError`` and ``TypeError`` are used by some lower-level numerical helpers when their raw array contracts are violated. ``NotImplementedError`` marks an explicitly unsupported physical path such as integrated PMC truncation or right-incident scattering.

Advanced geometry and mesh API
------------------------------

These names are imported from ``wavefem.geometry`` and ``wavefem.mesh``. They are useful when inspecting or assembling a custom low-level workflow.

Geometry shapes
~~~~~~~~~~~~~~~

.. code-block:: python

   Rectangle(x: tuple[float, float], z: tuple[float, float])
   Circle(center: tuple[float, float], radius: float)
   Polygon(points: tuple[tuple[float, float], ...])

Each frozen shape implements ``contains(x, z) -> bool_array`` with NumPy broadcasting. Prefer ``GeometryModel.add_*`` for validation before meshing.

``Region``
~~~~~~~~~~

.. code-block:: python

   Region(
       name: str,
       shape: Rectangle | Circle | Polygon,
       material: Material,
       background: bool,
       physical_tag: int,
   )

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
   * - ``shape``
     - Required
     - ``Shape``
     - Geometry primitive or region selector identifying the physical support. Use the class/union in the signature; electrostatics also accepts named exterior boundaries.
   * - ``material``
     - Required
     - ``Material``
     - Material object defining relative electric and magnetic response. Exterior fills points outside placed material regions.
   * - ``background``
     - Required
     - ``bool``
     - In placement methods, True adds a z-invariant background region/sheet; False adds an actual-device perturbation. In material/geometry constructors, this is the exterior Material object.
   * - ``physical_tag``
     - Required
     - ``int``
     - Stable Gmsh physical material tag(s) linking mesh cells to material regions.

Associates a material and stable physical tag with a shape. ``background=True`` means the region is present in both the actual and unperturbed profiles. ``contains(x,z)`` delegates to its shape.

``PECSheet``, ``PECSlot``, and ``PECSegment``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   PECSheet(name: str, x: float, z: tuple[float, float], background: bool)
   PECSlot(name: str, sheet_name: str, z: tuple[float, float])
   PECSegment(name: str, x: float, z: tuple[float, float])

These are frozen topology records; use ``GeometryModel.add_pec`` and ``GeometryModel.add_slot`` instead of constructing them manually.

* ``PECSheet`` describes one ideal constant-x sheet. ``background=True`` means the sheet belongs to the straight guide and, before subtraction, the actual device.

* ``PECSlot`` names a compact z interval removed only from the actual profile of ``sheet_name``.

* ``PECSegment`` is a derived closed line segment returned by ``GeometryModel.pec_segments``. A slotted actual sheet becomes multiple segments, whereas its background form remains one complete segment.

``GeometryModel``
~~~~~~~~~~~~~~~~~

.. code-block:: python

   GeometryModel(
       x_span: tuple[float, float],
       z_span: tuple[float, float],
       exterior: Material,
       regions: list[Region] = [],
       pec_sheets: list[PECSheet] = [],
       pec_slots: list[PECSlot] = [],
   )

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``x_span``
     - Required
     - ``tuple[float, float]``
     - Physical axis bounds in metres, with upper > lower. Range-based solver constructors also accept a positive extent for (0, extent).
   * - ``z_span``
     - Required
     - ``tuple[float, float]``
     - Physical axis bounds in metres, with upper > lower. Range-based solver constructors also accept a positive extent for (0, extent).
   * - ``exterior``
     - Required
     - ``Material``
     - Material object defining relative electric and magnetic response. Exterior fills points outside placed material regions.
   * - ``regions``
     - Optional
     - ``list[Region]``
     - Ordered layer/region/boundary specifications. Later overlapping material regions take precedence where the geometry API permits overlap. Default: ``fresh default container``.
   * - ``pec_sheets``
     - Optional
     - ``list[PECSheet]``
     - PEC sheet/plane handle(s) that define zero tangential electric field. Slots must refer to a background sheet. Default: ``fresh default container``.
   * - ``pec_slots``
     - Optional
     - ``list[PECSlot]``
     - Finite aperture specification(s) releasing part of a background PEC sheet in the actual device. Default: ``fresh default container``.

Maintains the actual/background distinction and stable insertion-order tags.

* ``add_rectangle(..., material, background=False, name=None)`` supports ``z="all"`` and requires it for a background layer.

* ``add_circle(..., material, background=False, name=None)`` rejects background circles.

* ``add_polygon(..., material, background=False, name=None)`` rejects background polygons.

* ``add_pec(x, z="all", background=False, name=None) -> PECSheet`` adds an ideal constant-x sheet. Its coordinate must lie strictly inside ``x_span``; coincident overlapping sheets and duplicate geometry names are rejected. A background sheet must be z-invariant and therefore requires ``z="all"``. An actual-only sheet may instead have a finite z span. The integrated ``Scattering2D.add_pec`` accepts that finite form when it lies strictly inside the non-PML interior.

* ``add_slot(pec, z, name=None) -> PECSlot`` subtracts a compact interval from the actual copy of a background sheet. ``pec`` is a sheet owned by this model or its name. The span must lie strictly inside the sheet, and slots on the same sheet cannot overlap or touch.

* ``slots_in(pec) -> tuple[PECSlot, ...]`` resolves a sheet and returns its slots sorted by increasing z.

* ``pec_segments(profile="actual"|"background") -> tuple[PECSegment, ...]`` derives non-overlapping sheet segments. The background profile contains only complete background sheets. The actual profile includes every sheet with its finite slots subtracted. An unknown profile raises ``ValueError``.

* ``background_regions`` and ``perturbations`` return immutable tuples.

* ``material_at(x,z,profile="actual"|"background")`` returns scalar ``(eps_r,mu_r)`` arrays. The actual profile applies all background regions first and then all perturbations; insertion order applies within each group, so a perturbation overrides a background layer on overlap.

* ``region_tag_at(x,z)`` returns actual-material physical tags.

* ``physical_names`` maps tag 1 to ``"exterior"`` and subsequent stable tags to region names.

``MeshInfo``
~~~~~~~~~~~~

.. code-block:: python

   MeshInfo(
       nodes: int,
       elements: int,
       minimum_edge: float,
       maximum_edge: float,
       requested_maximum_edge: float,
   )

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

All edge lengths are physical metres.

``Mesh2D``
~~~~~~~~~~

.. code-block:: python

   Mesh2D(
       mesh: skfem.MeshTri,
       element_tags: ndarray,
       physical_names: dict[int, str],
       info: MeshInfo,
       background_pec_facets: ndarray = empty,
       actual_pec_facets: ndarray = empty,
       released_pec_facets: ndarray = empty,
       pec_slot_facets: dict[str, ndarray] = {},
       inserted_pec_facets: ndarray = empty,
   )

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
   * - ``info``
     - Required
     - ``MeshInfo``
     - Structured metadata/diagnostic container associated with the mesh or result. solve_info/metadata retain applied adaptive controls and stopping history.
   * - ``background_pec_facets``
     - Optional
     - ``NDArray[np.int32]``
     - Facet indices in the corresponding mesh. Actual PEC remains constrained; released facets form apertures; inserted facets prescribe scattered tangential fields. Default: ``fresh default container``.
   * - ``actual_pec_facets``
     - Optional
     - ``NDArray[np.int32]``
     - Facet indices in the corresponding mesh. Actual PEC remains constrained; released facets form apertures; inserted facets prescribe scattered tangential fields. Default: ``fresh default container``.
   * - ``released_pec_facets``
     - Optional
     - ``NDArray[np.int32]``
     - Facet indices in the corresponding mesh. Actual PEC remains constrained; released facets form apertures; inserted facets prescribe scattered tangential fields. Default: ``fresh default container``.
   * - ``pec_slot_facets``
     - Optional
     - ``dict[str, NDArray[np.int32]]``
     - Mapping from slot names to the facet indices released on those apertures. Default: ``fresh default container``.
   * - ``inserted_pec_facets``
     - Optional
     - ``NDArray[np.int32]``
     - Facet indices in the corresponding mesh. Actual PEC remains constrained; released facets form apertures; inserted facets prescribe scattered tangential fields. Default: ``fresh default container``.

``elements_in(region: str | int)`` returns zero-based triangle indices matching a physical name or tag and raises ``MeshError`` for an unknown name.

PEC arrays contain sorted global ``MeshTri`` facet indices:

* ``background_pec_facets`` is the union of complete z-invariant background sheets;

* ``actual_pec_facets`` is the constraint set after finite slots are removed and finite actual-only plates are inserted;

* ``released_pec_facets`` is exactly ``background - actual`` and is the aperture source support;

* ``inserted_pec_facets`` is exactly ``actual - background`` and carries the nonhomogeneous scattered-field PEC trace;

* ``pec_slot_facets`` maps every slot name to its released subset.

``pec_facets(profile)`` returns the actual or background array. ``facets_in_slot(name)`` returns one named slot array and raises ``MeshError`` for an unknown name. The arrays are topology metadata; material triangle tags do not encode a zero-thickness PEC.

``generate_mesh``
~~~~~~~~~~~~~~~~~

.. code-block:: python

   generate_mesh(
       geometry: GeometryModel,
       *,
       max_element_size: float,
       x_partitions: tuple[float, ...] = (),
       z_partitions: tuple[float, ...] = (),
       refine_dielectrics: bool = True,
       dielectric_refinement_factor: float = 0.5,
       refine_pec: bool = True,
       pec_refinement_factor: float = 0.5,
       pec_refinement_distance: float | None = None,
   ) -> Mesh2D

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
   * - ``x_partitions``
     - Optional
     - ``tuple[float, ...]``
     - Interior physical coordinates at which the Gmsh mesh must contain conforming partition lines. Keyword-only. Default: ``()``.
   * - ``z_partitions``
     - Optional
     - ``tuple[float, ...]``
     - Interior physical coordinates at which the Gmsh mesh must contain conforming partition lines. Keyword-only. Default: ``()``.
   * - ``refine_dielectrics``
     - Optional
     - ``bool``
     - Enable the corresponding geometry-based local mesh-size field. Disabling sizing does not remove conforming material or PEC interfaces. Keyword-only. Default: ``True``.
   * - ``dielectric_refinement_factor``
     - Optional
     - ``float``
     - Local edge-size multiplier in (0, 1], applied at material regions or actual PEC curves respectively. Keyword-only. Default: ``0.5``.
   * - ``refine_pec``
     - Optional
     - ``bool``
     - Enable the corresponding geometry-based local mesh-size field. Disabling sizing does not remove conforming material or PEC interfaces. Keyword-only. Default: ``True``.
   * - ``pec_refinement_factor``
     - Optional
     - ``float``
     - Local edge-size multiplier in (0, 1], applied at material regions or actual PEC curves respectively. Keyword-only. Default: ``0.5``.
   * - ``pec_refinement_distance``
     - Optional
     - ``float | None``
     - Physical distance in metres over which the local mesh-size target transitions back to the surrounding target; None selects the mesher default when permitted. Keyword-only. Default: ``None``.

Creates a first-order conforming triangular mesh using Gmsh OCC fragments. Partition coordinates strictly inside the domain become conforming grid lines. All PEC x coordinates and all sheet/slot z endpoints are inserted automatically in addition to explicit partitions.

``max_element_size`` is the exterior/base target. With ``refine_dielectrics=True``, each physical material surface receives a restricted size field proportional to its local wavelength; material index is evaluated as ``sqrt(abs(eps_r * mu_r))``. ``dielectric_refinement_factor`` must be in ``(0, 1]`` and multiplies every non-exterior material-region target before the wavelength ratio. With ``refine_pec=True``, the actual PEC curves receive a distance-threshold field. ``pec_refinement_factor`` must be in ``(0, 1]`` and multiplies the smallest material target at the PEC. ``pec_refinement_distance`` must be positive when supplied and controls the physical transition distance; its default is three smallest-material target lengths. The two fields are combined by pointwise minimum, so PEC refinement also applies inside a dielectric. Disabling a field changes element sizes, not topological conformance.

After conversion to ``MeshTri``, constant-x facets are classified against actual and background segments; failure of the released-facet union to equal the named-slot union raises ``MeshError``. Material tags are evaluated from the actual geometry at triangle centroids. Gmsh failures are wrapped in ``MeshError``. On Windows, invoke Python through ``conda run`` so Gmsh DLLs are discoverable.

Advanced material tensors
-------------------------

The public physical material remains scalar. The following ``wavefem.materials`` types carry diagonal tensors produced internally by PML transformations.

``DiagonalTensor``
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   DiagonalTensor(xx: complex, yy: complex, zz: complex)

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``xx``
     - Required
     - ``complex``
     - Diagonal tensor entries in Cartesian x, y, z order, in the relative-material units of this object.
   * - ``yy``
     - Required
     - ``complex``
     - Diagonal tensor entries in Cartesian x, y, z order, in the relative-material units of this object.
   * - ``zz``
     - Required
     - ``complex``
     - Diagonal tensor entries in Cartesian x, y, z order, in the relative-material units of this object.

* ``isotropic(value)`` creates three equal entries.

* ``is_isotropic`` checks exact equality.

* ``as_array()`` returns a new complex ``(3,)`` array in ``(xx,yy,zz)`` order.

``DiagonalMaterial``
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   DiagonalMaterial(eps_r: DiagonalTensor, mu_r: DiagonalTensor)

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``eps_r``
     - Required
     - ``DiagonalTensor``
     - Relative permittivity. Scalar or Cartesian diagonal material inputs are supported where the signature permits; callbacks return scalar/broadcast-compatible values at physical coordinates.
   * - ``mu_r``
     - Required
     - ``DiagonalTensor``
     - Relative permeability, scalar or Cartesian diagonal where supported. WaveFEM scattering requires equal actual and background permeability.

``from_isotropic(material)`` expands a scalar ``Material``.

``as_diagonal_material``
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   as_diagonal_material(
       material: Material | DiagonalMaterial,
   ) -> DiagonalMaterial

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``material``
     - Required
     - ``Material | DiagonalMaterial``
     - Material object defining relative electric and magnetic response. Exterior fills points outside placed material regions.

Accepts a ``Material`` or existing ``DiagonalMaterial`` and returns the explicit diagonal representation used by FEM assembly and PML transformations. Other types raise ``MaterialError``.

These types do not add general off-diagonal anisotropy.

Advanced operator and FEM API
-----------------------------

``modified_curl``
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from wavefem.operators import modified_curl

   modified_curl(tangential, invariant, ky) -> ndarray

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``tangential``
     - Required
     - ``TangentialHcurlField``
     - Tangential electric-trace values on the selected boundary/PEC facets.
   * - ``invariant``
     - Required
     - ``InvariantH1Field``
     - Invariant/Fourier-axis designation for the reduced-dimensional Maxwell operator.
   * - ``ky``
     - Required
     - ``complex | float``
     - Real Fourier wavenumber along invariant y, in rad/m; zero gives propagation in the computational plane.

``tangential`` represents ``(Ex,Ez)`` and supplies the scikit-fem 2D curl ``partial_x Ez - partial_z Ex``. ``invariant`` represents ``Ey`` and supplies ``grad``. The result is

.. code-block:: text

   (i*ky*Ez - partial_z*Ey,
    partial_z*Ex - partial_x*Ez,
    partial_x*Ey - i*ky*Ex)

in physical component order. ``TangentialHcurlField`` and ``InvariantH1Field`` are structural typing protocols for these inputs.

``electric_field_vector``
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   electric_field_vector(tangential, invariant) -> ndarray

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``tangential``
     - Required
     - ``TangentialHcurlField``
     - Tangential electric-trace values on the selected boundary/PEC facets.
   * - ``invariant``
     - Required
     - ``NDArray[np.generic]``
     - Invariant/Fourier-axis designation for the reduced-dimensional Maxwell operator.

Combines ``(Ex,Ez)`` and ``Ey`` as ``(Ex,Ey,Ez)``.

``MaxwellParameters``
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   MaxwellParameters(
       k0: float,
       ky: complex = 0.0,
       eps_r=1.0,
       mu_r=1.0,
   )

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``k0``
     - Required
     - ``float``
     - Vacuum wavenumber in rad/m, used for coordinate scaling and material/PML evaluation.
   * - ``ky``
     - Optional
     - ``complex``
     - Real Fourier wavenumber along invariant y, in rad/m; zero gives propagation in the computational plane. Default: ``0.0``.
   * - ``eps_r``
     - Optional
     - ``ConstitutiveCoefficient``
     - Relative permittivity. Scalar or Cartesian diagonal material inputs are supported where the signature permits; callbacks return scalar/broadcast-compatible values at physical coordinates. Default: ``1.0``.
   * - ``mu_r``
     - Optional
     - ``ConstitutiveCoefficient``
     - Relative permeability, scalar or Cartesian diagonal where supported. WaveFEM scattering requires equal actual and background permeability. Default: ``1.0``.

``k0`` is positive. ``ky`` is currently real. Each constitutive coefficient may be a scalar, a three-entry diagonal, a quadrature-compatible array, or a ``coefficient(x,z)`` callback. Components use ``(x,y,z)`` order.

``MaxwellParameters.from_material(k0=..., material=..., ky=...)`` expands a scalar ``Material`` into explicit diagonal arrays.

``MixedFEMSystem``
~~~~~~~~~~~~~~~~~~

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
   * - ``matrix``
     - Required
     - ``csr_matrix``
     - Sparse/dense operator matrix. QEP coefficients represent A0 + neff*A1 + neff**2*A2 in the system coefficient ordering.
   * - ``parameters``
     - Required
     - ``MaxwellParameters``
     - Validated physical/operator parameter object for the low-level Maxwell formulation.
   * - ``physical_mesh``
     - Required
     - ``MeshTri``
     - Conforming FEM mesh object. Physical meshes carry SI coordinates; computational meshes may be scaled by the reference length for assembly.
   * - ``length_scale``
     - Optional
     - ``float``
     - Physical metres per computational coordinate unit; WaveFEM normally uses 1/k0. Default: ``1.0``.
   * - ``internal_pec_facets``
     - Optional
     - ``NDArray[np.int64]``
     - Facet indices in the corresponding mesh. Actual PEC remains constrained; released facets form apertures; inserted facets prescribe scattered tangential fields. Default: ``fresh default container``.
   * - ``quadrature_order``
     - Optional
     - ``int``
     - Finite-element integration order. Higher-order mixed elements require at least fourth-order quadrature; PML and material variation can require more. Default: ``4``.

Stores:

* ``basis``: composite N1/P1 or N2/P2 basis;

* ``matrix``: sparse complex Maxwell matrix;

* ``parameters``: physical ``MaxwellParameters``;

* ``physical_mesh``: original SI ``MeshTri``;

* ``length_scale``: metres per computational coordinate unit;

* ``quadrature_order``: integration order retained for volume and facet forms;

* ``internal_pec_facets``: sorted unique interior-facet indices on which the actual electric tangential trace is constrained.

Properties ``ndofs``, ``pec_dofs``, ``dimensionless_k0``, and ``dimensionless_ky`` expose assembly sizes/scales. ``pec_dofs`` is the union of the complete outer-boundary DOF set and every Nedelec/scalar trace DOF on ``internal_pec_facets``; normal electric traces are not part of the facet constraint. ``physical_coordinates()`` returns quadrature coordinates in metres.

``MixedFieldSolution``
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   MixedFieldSolution(
       basis,
       coefficients: complex ndarray,
       solve_info: Mapping[str, object] | None = None,
   )

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
   * - ``coefficients``
     - Required
     - ``NDArray[np.complex128]``
     - Complex finite-element coefficient vector or coefficient values, ordered exactly as the associated basis/system. Full vectors include constrained/periodic copies.
   * - ``solve_info``
     - Optional
     - ``Mapping[str, object] | None``
     - Structured metadata/diagnostic container associated with the mesh or result. solve_info/metadata retain applied adaptive controls and stopping history. Default: ``None``.

Validates a finite coefficient vector of length ``basis.N`` and stores metadata as a read-only mapping.

* ``split_coefficients()`` safely returns Nedelec and H1 coefficient blocks using scikit-fem's topology-aware split.

* ``interpolate()`` returns quadrature fields ``(E_t,E_y)``.

Assembly helpers
~~~~~~~~~~~~~~~~

``create_mixed_basis``
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   create_mixed_basis(mesh: skfem.MeshTri, *, intorder: int = 4,
                      element_order: int = 1) -> skfem.Basis

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
   * - ``intorder``
     - Optional
     - ``int``
     - Finite-element integration order. Higher-order mixed elements require at least fourth-order quadrature; PML and material variation can require more. Keyword-only. Default: ``4``.
   * - ``element_order``
     - Optional
     - ``int``
     - Finite-element polynomial-order selection. Standalone 2D modes and WaveFEM scattering accept 1 (N1/P1) or 2 (N2/P2); other backends retain their fixed compatible spaces. Keyword-only. Default: ``1``.

Creates a compatible N1/P1 (order 1) or N2/P2 (order 2) composite basis.
Integration uses at least ``2 * element_order``.  Other element orders raise
``ValueError``.

``evaluate_diagonal_coefficient``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   evaluate_diagonal_coefficient(
       coefficient,
       x,
       z,
       *,
       name: str = "coefficient",
   ) -> complex ndarray

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``coefficient``
     - Required
     - ``ConstitutiveCoefficient``
     - Complex finite-element coefficient vector or coefficient values, ordered exactly as the associated basis/system. Full vectors include constrained/periodic copies.
   * - ``x``
     - Required
     - ``NDArray[np.floating]``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature.
   * - ``z``
     - Required
     - ``NDArray[np.floating]``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature.
   * - ``name``
     - Optional
     - ``str``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``'coefficient'``.

Normalizes scalar/diagonal data to shape ``(3, nelements, nquadrature)``. Accepted callback results include a scalar, ``x.shape``, ``(3,)``, ``(3,*x.shape)``, or ``(*x.shape,3)``.

``assemble_maxwell_matrix``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   assemble_maxwell_matrix(
       basis: skfem.Basis,
       parameters: MaxwellParameters,
   ) -> scipy.sparse.csr_matrix

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
   * - ``parameters``
     - Required
     - ``MaxwellParameters``
     - Validated physical/operator parameter object for the low-level Maxwell formulation.

Assembles the complex sesquilinear curl-curl minus material-mass matrix. The basis must be the expected Nedelec-H1 composite and ``mu_r`` must be nonzero at every quadrature point.

``assemble_mixed_system``
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   assemble_mixed_system(
       mesh: skfem.MeshTri,
       parameters: MaxwellParameters,
       *,
       intorder: int = 4,
       element_order: int = 1,
       length_scale: float = 1.0,
       internal_pec_facets: ArrayLike = (),
   ) -> MixedFEMSystem

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
   * - ``parameters``
     - Required
     - ``MaxwellParameters``
     - Validated physical/operator parameter object for the low-level Maxwell formulation.
   * - ``intorder``
     - Optional
     - ``int``
     - Finite-element integration order. Higher-order mixed elements require at least fourth-order quadrature; PML and material variation can require more. Keyword-only. Default: ``4``.
   * - ``element_order``
     - Optional
     - ``int``
     - Finite-element polynomial-order selection. Standalone 2D modes and WaveFEM scattering accept 1 (N1/P1) or 2 (N2/P2); other backends retain their fixed compatible spaces. Keyword-only. Default: ``1``.
   * - ``length_scale``
     - Optional
     - ``float``
     - Physical metres per computational coordinate unit; WaveFEM normally uses 1/k0. Keyword-only. Default: ``1.0``.
   * - ``internal_pec_facets``
     - Optional
     - ``ArrayLike``
     - Facet indices in the corresponding mesh. Actual PEC remains constrained; released facets form apertures; inserted facets prescribe scattered tangential fields. Keyword-only. Default: ``()``.

Creates the basis and matrix. ``length_scale`` is the number of physical metres per computational coordinate unit. Material callbacks still receive physical metres. The high-level solver uses ``length_scale = 1/k0`` for conditioning. The requested ``intorder`` is retained so boundary loads and essential-trace projections use a true one-dimensional facet rule of the same order rather than reusing the triangular volume rule. ``internal_pec_facets`` must be a one-dimensional integer array of valid interior facets. Indices are topology-preserving under mesh rescaling and are stored on the returned system; boundary facets, noninteger arrays, and out-of-range indices raise ``ValueError``.

``assemble_load_vector``
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   assemble_load_vector(basis, source) -> complex ndarray

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
   * - ``source``
     - Required
     - ``VectorSource``
     - Equivalent-current source, mode-set/result object, or plotting data source accepted by the owning function; see its concrete type annotation.

Assembles ``integral(conj(V) dot source)``. The source is an array or callback with three physical components.

``solve_homogeneous_pec``
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   solve_homogeneous_pec(
       system: MixedFEMSystem,
       load,
       *,
       residual_tolerance: float = 1e-7,
   ) -> MixedFieldSolution

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``system``
     - Required
     - ``MixedFEMSystem``
     - Assembled FEM system containing compatible bases, sparse operators, constraints, and material/reference-scale metadata.
   * - ``load``
     - Required
     - ``NDArray[np.generic]``
     - Complex assembled load vector or full prescribed-DOF vector. Values must match the system dimension and boundary convention.
   * - ``residual_tolerance``
     - Optional
     - ``float``
     - Positive numerical tolerance. Linear/QEP residual tolerances validate the algebraic solve; they do not set the adaptive mesh threshold. Keyword-only. Default: ``1e-07``.

Condenses all outer PEC DOFs and every registered actual internal-PEC DOF, performs a SciPy direct solve, and validates the relative residual on free DOFs. Singular/resonant systems and non-finite solutions raise ``SolverError``.

``solve_prescribed_pec``
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   solve_prescribed_pec(
       system: MixedFEMSystem,
       load,
       *,
       boundary_values: ArrayLike,
       residual_tolerance: float = 1e-7,
   ) -> MixedFieldSolution

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``system``
     - Required
     - ``MixedFEMSystem``
     - Assembled FEM system containing compatible bases, sparse operators, constraints, and material/reference-scale metadata.
   * - ``load``
     - Required
     - ``NDArray[np.generic]``
     - Complex assembled load vector or full prescribed-DOF vector. Values must match the system dimension and boundary convention.
   * - ``boundary_values``
     - Required
     - ``ArrayLike``
     - Complex assembled load vector or full prescribed-DOF vector. Values must match the system dimension and boundary convention. Keyword-only.
   * - ``residual_tolerance``
     - Optional
     - ``float``
     - Positive numerical tolerance. Linear/QEP residual tolerances validate the algebraic solve; they do not set the adaptive mesh threshold. Keyword-only. Default: ``1e-07``.

Performs the same condensed direct solve while imposing a full mixed-space coefficient vector on the PEC DOF set. Entries outside ``system.pec_dofs`` must be zero. The residual normalization includes the effective load induced by the prescribed trace, so a boundary-only scattered-field solve remains a relative rather than absolute check. ``solve_homogeneous_pec`` is the zero-data special case.

``relative_hermiticity_error``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   relative_hermiticity_error(matrix) -> float

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

Returns ``norm(A-A.H)/norm(A)``, with the unnormalized numerator used for a zero matrix.

Advanced equivalent-source API
------------------------------

``IncidentField`` is the ``wavefem.sources`` callback type ``Callable[[x_array, z_array], object]``. Depending on the argument name it supplies electric or magnetic incident fields, always as three components in ``(x,y,z)`` order. A result may have shape ``(3,)``, ``(3,*x.shape)``, or ``(*x.shape,3)`` and must be finite.

``EquivalentSource``
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   EquivalentSource(
       load: complex ndarray,
       active_quadrature_fraction: float,
       maximum_delta_eps: float,
       released_pec_facet_count: int = 0,
       inserted_pec_facet_count: int = 0,
   )

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``load``
     - Required
     - ``NDArray[np.complex128]``
     - Complex assembled load vector or full prescribed-DOF vector. Values must match the system dimension and boundary convention.
   * - ``active_quadrature_fraction``
     - Required
     - ``float``
     - Source-support diagnostic: active quadrature fraction, largest permittivity contrast, or count of released/inserted PEC facets respectively.
   * - ``maximum_delta_eps``
     - Required
     - ``float``
     - Source-support diagnostic: active quadrature fraction, largest permittivity contrast, or count of released/inserted PEC facets respectively.
   * - ``released_pec_facet_count``
     - Optional
     - ``int``
     - Source-support diagnostic: active quadrature fraction, largest permittivity contrast, or count of released/inserted PEC facets respectively. Default: ``0``.
   * - ``inserted_pec_facet_count``
     - Optional
     - ``int``
     - Source-support diagnostic: active quadrature fraction, largest permittivity contrast, or count of released/inserted PEC facets respectively. Default: ``0``.

Stores the assembled RHS, fraction of quadrature points where material contrast is nonzero, maximum absolute permittivity contrast, and released and inserted PEC facet counts. ``is_zero`` is false when either the assembled load is nonzero or inserted PEC data are present. A boundary-only PEC perturbation can therefore have zero contrast fraction and zero ``maximum_delta_eps`` while ``is_zero`` is false.

``assemble_inserted_pec_boundary_values``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   assemble_inserted_pec_boundary_values(
       system: MixedFEMSystem,
       *,
       inserted_pec_facets: ArrayLike,
       incident: IncidentField,
   ) -> complex ndarray

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``system``
     - Required
     - ``MixedFEMSystem``
     - Assembled FEM system containing compatible bases, sparse operators, constraints, and material/reference-scale metadata.
   * - ``inserted_pec_facets``
     - Required
     - ``ArrayLike``
     - Facet indices in the corresponding mesh. Actual PEC remains constrained; released facets form apertures; inserted facets prescribe scattered tangential fields. Keyword-only.
   * - ``incident``
     - Required
     - ``IncidentField``
     - Incident-mode object or incident magnetic-field evaluator used to build contrast/aperture loads at physical coordinates. Keyword-only.

Projects ``-E_inc,t`` separately on the inserted-facet traces of the Nedelec and continuous-scalar component spaces using one-dimensional facet quadrature, maps the component coefficients through scikit-fem's topology-aware composite ordering, and returns a full vector that is zero away from the inserted facets. The result depends only on tangential incident data at the plate, not on its normal component or off-facet field. Every inserted facet must be an interior member of ``system.internal_pec_facets``.

``assemble_released_pec_source``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   assemble_released_pec_source(
       system: MixedFEMSystem,
       *,
       released_pec_facets: ArrayLike,
       incident_magnetic: IncidentField,
   ) -> complex ndarray

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``system``
     - Required
     - ``MixedFEMSystem``
     - Assembled FEM system containing compatible bases, sparse operators, constraints, and material/reference-scale metadata.
   * - ``released_pec_facets``
     - Required
     - ``ArrayLike``
     - Facet indices in the corresponding mesh. Actual PEC remains constrained; released facets form apertures; inserted facets prescribe scattered tangential fields. Keyword-only.
   * - ``incident_magnetic``
     - Required
     - ``IncidentField``
     - Incident-mode object or incident magnetic-field evaluator used to build contrast/aperture loads at physical coordinates. Keyword-only.

Assembles the scattered-field aperture load for facets that are PEC in the background guide but open in the actual device. For each facet it creates both ``InteriorFacetBasis`` traces with a one-dimensional edge rule, orients an outward normal for each adjacent element, and samples ``incident_magnetic`` an infinitesimal distance inside that element. This preserves a discontinuous one-sided magnetic trace at the background sheet; evaluating exactly at the coordinate would incorrectly select the same mode cell on both sides.

On the dimensionless mesh the released-boundary weak load is

.. code-block:: text

   -i (k0 * length_scale) ETA_0
       * sum_s integral_Gamma conj(V) dot (H_inc,s x n_s) ds.

The sum explicitly includes the two adjacent element sides; there is no additional image-equivalence multiplier. A continuous magnetic field gives cancelling opposite-normal contributions. Facets must be unique valid interior indices and disjoint from ``system.internal_pec_facets``; otherwise ``ValueError`` is raised. An empty array returns an exact all-zero vector.

``ScatteredFieldSolution``
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   ScatteredFieldSolution(
       field: MixedFieldSolution,
       source: EquivalentSource,
   )

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``field``
     - Required
     - ``MixedFieldSolution``
     - Complex electromagnetic field samples or FE field objects. Cartesian components follow x,y,z order; left/right denote the selected trace side.
   * - ``source``
     - Required
     - ``EquivalentSource``
     - Equivalent-current source, mode-set/result object, or plotting data source accepted by the owning function; see its concrete type annotation.

Pairs the solved scattered field with the exact source diagnostics that produced it.

``assemble_equivalent_source``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   assemble_equivalent_source(
       system: MixedFEMSystem,
       *,
       eps_background,
       mu_background=1.0,
       incident: IncidentField,
       released_pec_facets: ArrayLike = (),
       inserted_pec_facets: ArrayLike = (),
       incident_magnetic: IncidentField | None = None,
   ) -> EquivalentSource

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``system``
     - Required
     - ``MixedFEMSystem``
     - Assembled FEM system containing compatible bases, sparse operators, constraints, and material/reference-scale metadata.
   * - ``eps_background``
     - Required
     - ``ConstitutiveCoefficient``
     - Relative-permittivity callback for the lossless z-invariant unperturbed lead; actual contrast must be compact and bracketed by the monitors. Keyword-only.
   * - ``mu_background``
     - Optional
     - ``ConstitutiveCoefficient``
     - Relative permeability, scalar or Cartesian diagonal where supported. WaveFEM scattering requires equal actual and background permeability. Keyword-only. Default: ``1.0``.
   * - ``incident``
     - Required
     - ``IncidentField``
     - Incident-mode object or incident magnetic-field evaluator used to build contrast/aperture loads at physical coordinates. Keyword-only.
   * - ``released_pec_facets``
     - Optional
     - ``ArrayLike``
     - Facet indices in the corresponding mesh. Actual PEC remains constrained; released facets form apertures; inserted facets prescribe scattered tangential fields. Keyword-only. Default: ``()``.
   * - ``inserted_pec_facets``
     - Optional
     - ``ArrayLike``
     - Facet indices in the corresponding mesh. Actual PEC remains constrained; released facets form apertures; inserted facets prescribe scattered tangential fields. Keyword-only. Default: ``()``.
   * - ``incident_magnetic``
     - Optional
     - ``IncidentField | None``
     - Incident-mode object or incident magnetic-field evaluator used to build contrast/aperture loads at physical coordinates. Keyword-only. Default: ``None``.

Assembles ``k0**2 * (eps_actual-eps_background) * E_inc`` using quadrature values. ``incident(x,z)`` returns three components. Actual and background permeability must agree to numerical tolerance; otherwise ``ConfigurationError`` is raised. When ``released_pec_facets`` is nonempty, the function also adds ``assemble_released_pec_source``; ``incident_magnetic`` then becomes mandatory. ``inserted_pec_facets`` is validated and counted separately; its essential data are assembled by ``assemble_inserted_pec_boundary_values`` during the solve.

``solve_scattered_pec``
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   solve_scattered_pec(
       system: MixedFEMSystem,
       *,
       eps_background,
       mu_background=1.0,
       incident: IncidentField,
       released_pec_facets: ArrayLike = (),
       inserted_pec_facets: ArrayLike = (),
       incident_magnetic: IncidentField | None = None,
       residual_tolerance: float = 1e-7,
   ) -> ScatteredFieldSolution

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``system``
     - Required
     - ``MixedFEMSystem``
     - Assembled FEM system containing compatible bases, sparse operators, constraints, and material/reference-scale metadata.
   * - ``eps_background``
     - Required
     - ``ConstitutiveCoefficient``
     - Relative-permittivity callback for the lossless z-invariant unperturbed lead; actual contrast must be compact and bracketed by the monitors. Keyword-only.
   * - ``mu_background``
     - Optional
     - ``ConstitutiveCoefficient``
     - Relative permeability, scalar or Cartesian diagonal where supported. WaveFEM scattering requires equal actual and background permeability. Keyword-only. Default: ``1.0``.
   * - ``incident``
     - Required
     - ``IncidentField``
     - Incident-mode object or incident magnetic-field evaluator used to build contrast/aperture loads at physical coordinates. Keyword-only.
   * - ``released_pec_facets``
     - Optional
     - ``ArrayLike``
     - Facet indices in the corresponding mesh. Actual PEC remains constrained; released facets form apertures; inserted facets prescribe scattered tangential fields. Keyword-only. Default: ``()``.
   * - ``inserted_pec_facets``
     - Optional
     - ``ArrayLike``
     - Facet indices in the corresponding mesh. Actual PEC remains constrained; released facets form apertures; inserted facets prescribe scattered tangential fields. Keyword-only. Default: ``()``.
   * - ``incident_magnetic``
     - Optional
     - ``IncidentField | None``
     - Incident-mode object or incident magnetic-field evaluator used to build contrast/aperture loads at physical coordinates. Keyword-only. Default: ``None``.
   * - ``residual_tolerance``
     - Optional
     - ``float``
     - Positive numerical tolerance. Linear/QEP residual tolerances validate the algebraic solve; they do not set the adaptive mesh threshold. Keyword-only. Default: ``1e-07``.

Forms the combined volume/aperture equivalent source, assembles the inserted PEC trace, and calls the prescribed actual-PEC field solver. ``released_pec_facets`` must not be in the system's actual constraint set; ``inserted_pec_facets`` must be in it. In the high-level workflow the physical outgoing condition is supplied by constitutive PML tensors inside the outer PEC boundary.

Advanced monitor API
--------------------

``MonitorSamples``
~~~~~~~~~~~~~~~~~~

Fields on a sorted ``z=constant`` line:

.. code-block:: python

   MonitorSamples(x, weights, E, H, z)

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``x``
     - Required
     - ``RealArray``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature.
   * - ``weights``
     - Required
     - ``RealArray``
     - Quadrature or line-integration weights corresponding one-to-one with the sampled field coordinates.
   * - ``E``
     - Required
     - ``ComplexArray``
     - Complex electromagnetic field samples or FE field objects. Cartesian components follow x,y,z order; left/right denote the selected trace side.
   * - ``H``
     - Required
     - ``ComplexArray``
     - Complex electromagnetic field samples or FE field objects. Cartesian components follow x,y,z order; left/right denote the selected trace side.
   * - ``z``
     - Required
     - ``float``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature.

``x`` and positive integration ``weights`` have shape ``(N,)``; E and H have shape ``(3,N)``; ``z`` is the physical line coordinate.

``HorizontalMonitorSamples``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fields on a sorted ``x=constant`` line:

.. code-block:: python

   HorizontalMonitorSamples(z, weights, E, H, x)

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``z``
     - Required
     - ``RealArray``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature.
   * - ``weights``
     - Required
     - ``RealArray``
     - Quadrature or line-integration weights corresponding one-to-one with the sampled field coordinates.
   * - ``E``
     - Required
     - ``ComplexArray``
     - Complex electromagnetic field samples or FE field objects. Cartesian components follow x,y,z order; left/right denote the selected trace side.
   * - ``H``
     - Required
     - ``ComplexArray``
     - Complex electromagnetic field samples or FE field objects. Cartesian components follow x,y,z order; left/right denote the selected trace side.
   * - ``x``
     - Required
     - ``float``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature.

``z`` and weights have shape ``(N,)``; E and H have shape ``(3,N)``.

``sample_vertical_monitor``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   sample_vertical_monitor(
       basis,
       coefficients,
       *,
       z: float,
       ky: complex = 0.0,
       omega: float,
       mu_r=1.0,
       length_scale: float = 1.0,
       intorder: int = 4,
       tolerance: float | None = None,
   ) -> MonitorSamples

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
   * - ``coefficients``
     - Required
     - ``ArrayLike``
     - Complex finite-element coefficient vector or coefficient values, ordered exactly as the associated basis/system. Full vectors include constrained/periodic copies.
   * - ``z``
     - Required
     - ``float``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature. Keyword-only.
   * - ``ky``
     - Optional
     - ``complex``
     - Real Fourier wavenumber along invariant y, in rad/m; zero gives propagation in the computational plane. Keyword-only. Default: ``0.0``.
   * - ``omega``
     - Required
     - ``float``
     - Angular frequency in rad/s; must be finite and positive. Keyword-only.
   * - ``mu_r``
     - Optional
     - ``ConstitutiveCoefficient``
     - Relative permeability, scalar or Cartesian diagonal where supported. WaveFEM scattering requires equal actual and background permeability. Keyword-only. Default: ``1.0``.
   * - ``length_scale``
     - Optional
     - ``float``
     - Physical metres per computational coordinate unit; WaveFEM normally uses 1/k0. Keyword-only. Default: ``1.0``.
   * - ``intorder``
     - Optional
     - ``int``
     - Finite-element integration order. Higher-order mixed elements require at least fourth-order quadrature; PML and material variation can require more. Keyword-only. Default: ``4``.
   * - ``tolerance``
     - Optional
     - ``float | None``
     - Positive numerical tolerance. Linear/QEP residual tolerances validate the algebraic solve; they do not set the adaptive mesh threshold. Keyword-only. Default: ``None``.

Samples a mesh-conforming physical ``z=constant`` interior-facet line. Traces from both adjacent elements are averaged, H is reconstructed from ``curl(E)/(i*omega*mu)``, and samples/weights are sorted by physical x. The requested line must coincide with interior mesh facets.

``sample_horizontal_monitor``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   sample_horizontal_monitor(
       basis,
       coefficients,
       *,
       x: float,
       ky: complex = 0.0,
       omega: float,
       mu_r=1.0,
       length_scale: float = 1.0,
       intorder: int = 4,
       tolerance: float | None = None,
   ) -> HorizontalMonitorSamples

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
   * - ``coefficients``
     - Required
     - ``ArrayLike``
     - Complex finite-element coefficient vector or coefficient values, ordered exactly as the associated basis/system. Full vectors include constrained/periodic copies.
   * - ``x``
     - Required
     - ``float``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature. Keyword-only.
   * - ``ky``
     - Optional
     - ``complex``
     - Real Fourier wavenumber along invariant y, in rad/m; zero gives propagation in the computational plane. Keyword-only. Default: ``0.0``.
   * - ``omega``
     - Required
     - ``float``
     - Angular frequency in rad/s; must be finite and positive. Keyword-only.
   * - ``mu_r``
     - Optional
     - ``ConstitutiveCoefficient``
     - Relative permeability, scalar or Cartesian diagonal where supported. WaveFEM scattering requires equal actual and background permeability. Keyword-only. Default: ``1.0``.
   * - ``length_scale``
     - Optional
     - ``float``
     - Physical metres per computational coordinate unit; WaveFEM normally uses 1/k0. Keyword-only. Default: ``1.0``.
   * - ``intorder``
     - Optional
     - ``int``
     - Finite-element integration order. Higher-order mixed elements require at least fourth-order quadrature; PML and material variation can require more. Keyword-only. Default: ``4``.
   * - ``tolerance``
     - Optional
     - ``float | None``
     - Positive numerical tolerance. Linear/QEP residual tolerances validate the algebraic solve; they do not set the adaptive mesh threshold. Keyword-only. Default: ``None``.

The corresponding physical ``x=constant`` sampler, sorted by z. It is used for transverse Poynting-flux/radiation accounting.

For both samplers, ``basis`` coordinates are interpreted as physical coordinates divided by ``length_scale``. Returned coordinates and weights are in SI units.

Advanced modal-projection API
-----------------------------

``ModalTrace``
~~~~~~~~~~~~~~

.. code-block:: python

   ModalTrace(E: complex ndarray, H: complex ndarray, label: str = "mode")

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``E``
     - Required
     - ``ComplexArray``
     - Complex electromagnetic field samples or FE field objects. Cartesian components follow x,y,z order; left/right denote the selected trace side.
   * - ``H``
     - Required
     - ``ComplexArray``
     - Complex electromagnetic field samples or FE field objects. Cartesian components follow x,y,z order; left/right denote the selected trace side.
   * - ``label``
     - Optional
     - ``str``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Default: ``'mode'``.

Stores one candidate mode on a monitor quadrature. E and H must be finite, equal-shaped ``(3,N)`` arrays.

``ProjectionResult``
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   ProjectionResult(
       amplitudes,
       gram_matrix,
       condition_number,
       relative_residual,
       labels,
   )

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``amplitudes``
     - Required
     - ``ComplexArray``
     - Complex modal amplitude(s); squared magnitude is power for a unit-power propagating mode. Integrated incidence requires a nonzero finite launch amplitude.
   * - ``gram_matrix``
     - Required
     - ``ComplexArray``
     - Complex modal power-Gram matrix used to solve for amplitudes in a possibly non-orthogonal mode basis.
   * - ``condition_number``
     - Required
     - ``float``
     - Measured condition number(s) of the modal projection system.
   * - ``relative_residual``
     - Required
     - ``float``
     - Stored numerical-validation diagnostic. Algebraic and Gauss residuals measure discrete equation defects; pml_fraction measures energy in absorbing layers.
   * - ``labels``
     - Required
     - ``tuple[str, ...]``
     - Human-readable labels corresponding to the plotted modes, components, or sweep curves.

* ``amplitudes`` follows the input trace order.

* ``gram_matrix`` is the electromagnetic power Gram.

* ``condition_number`` is computed after trace-norm scaling.

* ``relative_residual`` is the weighted E/H reconstruction residual.

* ``labels`` are copied from the traces.

``ElectromagneticProjector``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   ElectromagneticProjector(
       traces: Iterable[ModalTrace],
       weights,
       *,
       impedance: float | None = None,
       condition_limit: float = 1e12,
   )

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``traces``
     - Required
     - ``Iterable[ModalTrace]``
     - Sampled modal/field traces on a monitor or interface, in the ordering expected by the projector.
   * - ``weights``
     - Required
     - ``ArrayLike``
     - Quadrature or line-integration weights corresponding one-to-one with the sampled field coordinates.
   * - ``impedance``
     - Optional
     - ``float | None``
     - Surface impedance in ohms. Supply an explicit passive complex value or a supported metal through the alternative material input. Keyword-only. Default: ``None``.
   * - ``condition_limit``
     - Optional
     - ``float``
     - Maximum allowed modal Gram-system condition number; must exceed one. Keyword-only. Default: ``1000000000000.0``.

Constructs a forward/backward modal basis on one shared monitor quadrature. Weights must be finite and positive. ``impedance`` balances E and H in the reported reconstruction residual; amplitudes themselves come from the power Gram. ``condition_limit`` rejects near-singular decompositions.

``project(E,H) -> ProjectionResult`` solves the dense Gram system. Target fields must use the same ``(3,N)`` quadrature as the traces.

``modal_power_from_gram``
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   modal_power_from_gram(
       amplitudes,
       gram_matrix,
       *,
       indices=None,
       normalize_diagonal: bool = False,
   ) -> float

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``amplitudes``
     - Required
     - ``ArrayLike``
     - Complex modal amplitude(s); squared magnitude is power for a unit-power propagating mode. Integrated incidence requires a nonzero finite launch amplitude.
   * - ``gram_matrix``
     - Required
     - ``ArrayLike``
     - Complex modal power-Gram matrix used to solve for amplitudes in a possibly non-orthogonal mode basis.
   * - ``indices``
     - Optional
     - ``ArrayLike | None``
     - Mode, case, array, or mapping selector. Python indexing is zero based; explicit mode(number) and standalone visualization use their documented one-based numbers. Keyword-only. Default: ``None``.
   * - ``normalize_diagonal``
     - Optional
     - ``bool``
     - Scale the modal Gram diagonal before projection to improve conditioning across differently normalized modes. Keyword-only. Default: ``False``.

Returns signed real modal flux from ``Re(a.T @ G @ conj(a))``. ``indices`` can restrict the calculation to propagating families. ``normalize_diagonal=True`` removes small sampled unit-power diagonal errors by a congruence scaling and rejects a zero-power selected trace.

Advanced mode-system records
----------------------------

``Layer``
~~~~~~~~~

.. code-block:: python

   Layer(x: tuple[float, float], material: Material, name: str)

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
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature.
   * - ``material``
     - Required
     - ``Material``
     - Material object defining relative electric and magnetic response. Exterior fills points outside placed material regions.
   * - ``name``
     - Required
     - ``str``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported.

The immutable interval record returned by ``CrossSection.add_layer``.

``PECBoundary``
~~~~~~~~~~~~~~~

.. code-block:: python

   PECBoundary(x: float, name: str)

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``x``
     - Required
     - ``float``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature.
   * - ``name``
     - Required
     - ``str``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported.

The immutable internal-sheet record returned by ``CrossSection.add_pec``. Constructing one directly is supported only as an entry in the ``CrossSection.pec_boundaries`` initializer; ``CrossSection.__post_init__`` revalidates its coordinate and name through the same public addition path.

``ModeFEMSystem``
~~~~~~~~~~~~~~~~~

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``x_nodes``
     - Required
     - ``FloatArray``
     - Strictly increasing physical 1D mesh-node coordinates, in metres.
   * - ``xi_nodes``
     - Required
     - ``FloatArray``
     - Dimensionless 1D mesh-node coordinates, scaled by the reference length.
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
   * - ``ex_slice``
     - Required
     - ``slice``
     - Indices/slices selecting transverse, longitudinal, or Cartesian components from the full mixed coefficient vector.
   * - ``ey_slice``
     - Required
     - ``slice``
     - Indices/slices selecting transverse, longitudinal, or Cartesian components from the full mixed coefficient vector.
   * - ``ez_slice``
     - Required
     - ``slice``
     - Indices/slices selecting transverse, longitudinal, or Cartesian components from the full mixed coefficient vector.
   * - ``divergence_x``
     - Required
     - ``csr_matrix``
     - Discrete weak divergence/Gauss operator used to validate modal charge consistency in the associated scalar test space.
   * - ``epsilon_mass``
     - Required
     - ``csr_matrix``
     - Relative-permittivity mass operator, with longitudinal weighting for the z-specific variant.
   * - ``epsilon_mass_z``
     - Required
     - ``csr_matrix``
     - Relative-permittivity mass operator, with longitudinal weighting for the z-specific variant.
   * - ``divergence_test_dofs``
     - Required
     - ``IntArray``
     - Integer degree-of-freedom indices selecting constrained/free unknowns or admissible scalar test functions.
   * - ``frequency``
     - Required
     - ``Frequency``
     - Ordinary frequency in Hz; must be finite and positive.
   * - ``ky``
     - Required
     - ``float``
     - Real Fourier wavenumber along invariant y, in rad/m; zero gives propagation in the computational plane.
   * - ``eta``
     - Required
     - ``float``
     - Dimensionless Fourier wavenumber ky/k0 in the normalized Maxwell/modal equations.
   * - ``boundary``
     - Required
     - ``BoundaryKind``
     - Outer transverse boundary condition. Modal solvers support PEC/PMC as documented; integrated WaveFEM supports PEC or transverse PML terminated by PEC.

Stores the 1D mode mesh, dimensionless nodes, sparse ``A0/A1/A2`` pencil, free-DOF mapping, component slices, divergence operators, frequency, ``ky``, ``eta=ky/k0``, and boundary kind. Internal PEC nodes are represented in the free-DOF mapping and excluded from ``divergence_test_dofs``.

* ``ndofs``: reduced unconstrained electric DOFs.

* ``elements``: number of 1D cells.

* ``polynomial(neff)``: evaluates ``A0+neff*A1+neff**2*A2``.

* ``expand(vector)``: inserts constrained DOFs into the full component vector.

* ``relative_hermiticity_errors()``: one relative error for each pencil matrix.

* ``divergence_residual(full_vector, neff)``: normalized weak Gauss-law residual.

Package metadata
----------------

``wavefem.__version__`` is ``"0.0.1"`` for this implementation.

The top-level ``wavefem.__all__`` is treated as the stable convenience API. The advanced module APIs above expose numerical internals for research and may evolve faster than the top-level workflow.

Integrated-solver limitations summary
-------------------------------------

* Physical public materials are scalar and isotropic; internal PML tensors are diagonal.

* The scattered-field workflow supports volume permittivity contrast, finite slots released from z-invariant background PEC sheets, and finite constant-x actual-only PEC plates. Permeability contrast, arbitrary curved PEC insertions, and finite-conductivity sheets are not implemented.

* ``Scattering2D`` supports passive reciprocal problems, compact loss, one incident mode, and left incidence.

* Uniform leads must be lossless for the integrated projection/power path.

* Scattering supports compatible N1/P1 and N2/P2 elements on affine triangles
  and uses a sparse direct solver.  Per-cell hp mixing is not implemented.

* A z-PML is mandatory. Open transverse structures additionally require an x-PML; integrated PMC truncation is not implemented.

* Callback devices require explicit modes and the caller-validated physical invariants listed under ``from_material_function``.

* High-level imported meshes and arbitrary-point result evaluation are not yet exposed; lower-level mesh/FEM interfaces remain available.

* Every new structure requires mesh, monitor, mode-mesh, and PML convergence checks. A nominal PML reflection target is not a validation result.

Additional public API contracts
-------------------------------

The following exported records, methods, properties, and research helpers complement the behavior guide above. Signatures show library defaults; examples use max_refinements=0.

``wavefem.ConfigurationError``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The requested simulation configuration is incomplete or inconsistent.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.ConfigurationError(*args: 'object')

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

``wavefem.DiagnosticReport.ok``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's ok value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.DiagnosticReport.ok

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

``wavefem.DiagnosticReport.warnings``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's warnings value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.DiagnosticReport.warnings

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

Returns: ``tuple[Diagnostic, ...]``.

``wavefem.Frequency.angular_frequency``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Angular frequency in radians per second.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.Frequency.angular_frequency

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

``wavefem.Frequency.frequency``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ordinary frequency in hertz.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.Frequency.frequency

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

``wavefem.Frequency.from_frequency``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a spectral point from ordinary frequency in hertz.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.Frequency.from_frequency(frequency: 'RealInput') -> "'Frequency'"

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``frequency``
     - Required
     - ``RealInput``
     - Ordinary frequency in Hz; must be finite and positive.

Returns: ``Frequency``.

``wavefem.Frequency.from_omega``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a spectral point from angular frequency in radians/second.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.Frequency.from_omega(omega: 'RealInput') -> "'Frequency'"

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``omega``
     - Required
     - ``RealInput``
     - Angular frequency in rad/s; must be finite and positive.

Returns: ``Frequency``.

``wavefem.Frequency.from_wavelength``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a spectral point from vacuum wavelength in metres.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.Frequency.from_wavelength(wavelength: 'RealInput') -> "'Frequency'"

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``wavelength``
     - Required
     - ``RealInput``
     - Vacuum wavelength in metres; must be finite and positive.

Returns: ``Frequency``.

``wavefem.Frequency.k0``
~~~~~~~~~~~~~~~~~~~~~~~~

Vacuum angular wavenumber in radians per metre.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.Frequency.k0

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

``wavefem.Frequency.wavelength``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Vacuum wavelength in metres.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.Frequency.wavelength

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

``wavefem.FrequencySweepResult.S11``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fundamental reflected modal amplitude at every frequency.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.FrequencySweepResult.S11

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

``wavefem.FrequencySweepResult.S21``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fundamental transmitted modal amplitude at every frequency.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.FrequencySweepResult.S21

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

``wavefem.FrequencySweepResult.absorbed_power``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Material-absorbed power at every frequency.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.FrequencySweepResult.absorbed_power

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

``wavefem.FrequencySweepResult.incident_power``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Incident modal power at every frequency.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.FrequencySweepResult.incident_power

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

``wavefem.FrequencySweepResult.power_balance``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Accounted output-power fraction at every frequency.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.FrequencySweepResult.power_balance

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

``wavefem.FrequencySweepResult.power_balance_error``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dimensionless power-balance error at every frequency.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.FrequencySweepResult.power_balance_error

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

``wavefem.FrequencySweepResult.radiated_power``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Outward radiated power at every frequency.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.FrequencySweepResult.radiated_power

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

``wavefem.FrequencySweepResult.reflection``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Total reflected-power ratio at every frequency.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.FrequencySweepResult.reflection

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

``wavefem.FrequencySweepResult.transmission``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Total transmitted-power ratio at every frequency.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.FrequencySweepResult.transmission

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

``wavefem.FrequencySweepResult.visualize``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plot the fundamental S11/S21 response with Matplotlib.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.FrequencySweepResult.visualize(*, ax: 'Any | None' = None, show: 'bool' = True) -> 'Any'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``ax``
     - Optional
     - ``Any | None``
     - Existing Matplotlib axes for embedding a plot; None creates suitable axes when accepted. Keyword-only. Default: ``None``.
   * - ``show``
     - Optional
     - ``bool``
     - Enable display of the figure, mesh overlay, or colorbar respectively. show=False returns plotting objects without opening a window. Keyword-only. Default: ``True``.

Returns: ``Any``.

``wavefem.FrequencySweepResult.visualize_with_gui``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open every sweep point and its stored modes in the native viewer.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.FrequencySweepResult.visualize_with_gui() -> 'Any'

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

``wavefem.IncidentMode.E``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluate the incident electric field in ``(x, y, z)`` order.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.IncidentMode.E(x: 'ArrayLike', z: 'ArrayLike') -> 'ComplexArray'

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
   * - ``z``
     - Required
     - ``ArrayLike``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature.

Returns: ``ComplexArray``.

``wavefem.IncidentMode.H``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluate the incident magnetic field in ``(x, y, z)`` order.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.IncidentMode.H(x: 'ArrayLike', z: 'ArrayLike') -> 'ComplexArray'

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
   * - ``z``
     - Required
     - ``ArrayLike``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature.

Returns: ``ComplexArray``.

``wavefem.IncidentMode.beta``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Propagation constant, including the launch-direction sign.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.IncidentMode.beta

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

``wavefem.IncidentMode.direction``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Direction classification of the actually launched mode.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.IncidentMode.direction

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

Returns: ``str``.

``wavefem.IncidentMode.fields``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluate amplitude-scaled ``(E, H)`` anywhere in the straight lead.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.IncidentMode.fields(x: 'ArrayLike', z: 'ArrayLike') -> 'tuple[ComplexArray, ComplexArray]'

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
   * - ``z``
     - Required
     - ``ArrayLike``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature.

Returns: ``tuple[ComplexArray, ComplexArray]``.

``wavefem.IncidentMode.signed_power``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Signed longitudinal real power after amplitude scaling.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.IncidentMode.signed_power

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

``wavefem.Material.is_lossless``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Whether both relative constitutive scalars have zero loss part.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.Material.is_lossless

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

``wavefem.Material.is_passive``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Whether both scalars obey the passive sign for ``exp(-i*omega*t)``.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.Material.is_passive

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

``wavefem.MaterialError``
~~~~~~~~~~~~~~~~~~~~~~~~~

A material value or constitutive representation is invalid.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.MaterialError(*args: 'object')

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

Returns: ``MaterialError``.

``wavefem.MeshError``
~~~~~~~~~~~~~~~~~~~~~

Mesh generation, import, or physical-region tagging failed.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.MeshError(*args: 'object')

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

``wavefem.Mode.E``
~~~~~~~~~~~~~~~~~~

Cell-centred electric field in physical ``(x, y, z)`` order.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.Mode.E

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

``wavefem.Mode.H``
~~~~~~~~~~~~~~~~~~

Cell-centred magnetic field in physical ``(x, y, z)`` order.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.Mode.H

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

``wavefem.Mode.backward``
~~~~~~~~~~~~~~~~~~~~~~~~~

Return this modal family member propagating or decaying toward -z.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.Mode.backward() -> 'Mode'

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

``wavefem.Mode.counterpropagating``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Return the exact z-mirrored mode at the same ``omega`` and ``ky``.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.Mode.counterpropagating() -> 'Mode'

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

For scalar z-invariant media, reflecting ``z -> -z`` changes

``beta -> -beta``, ``E -> (E_x, E_y, -E_z)``, and ``H -> (-H_x, -H_y, H_z)``.

This is a spatial reflection, not complex conjugation. It therefore remains correct for loss under the ``exp(-i*omega*t)`` convention and reverses both the signed real power and complex longitudinal flux.

``wavefem.Mode.is_propagating``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's is propagating value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.Mode.is_propagating

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

``wavefem.Mode.sample_E``
~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluate the transverse FEM electric trace at physical ``x``.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.Mode.sample_E(x: 'ArrayLike') -> 'ComplexArray'

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

Returns: ``ComplexArray``.

The result has shape ``(3, *np.shape(x))`` in physical ``(x, y, z)`` component order. ``E_x`` retains its piecewise-constant Nedelec trace, while ``E_y`` and ``E_z`` are interpolated linearly from their nodal coefficients. At an internal element boundary the P0 component uses the element immediately to the right; the final endpoint uses the last element.

``wavefem.Mode.sample_H``
~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluate the reconstructed magnetic trace at physical ``x``.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.Mode.sample_H(x: 'ArrayLike') -> 'ComplexArray'

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

Returns: ``ComplexArray``.

``H_x`` is piecewise linear within each cell (and can jump across a material interface), while ``H_y,H_z`` are cellwise. Older manually constructed Mode objects without endpoint data retain the midpoint-P0 fallback. The output shape and component order match ``sample_E``.

``wavefem.Mode.x``
~~~~~~~~~~~~~~~~~~

Cell-centre physical x coordinates.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.Mode.x

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

``wavefem.ModeProjectionError``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fields could not be projected reliably onto the requested lead modes.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.ModeProjectionError(*args: 'object')

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

Returns: ``ModeProjectionError``.

``wavefem.ModeSet.__getitem__``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Select an item using Python square-bracket indexing; integer indices are zero based.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.ModeSet.__getitem__(index: 'int | slice') -> 'Mode | tuple[Mode, ...]'

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

``wavefem.ModeSet.__iter__``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Iterate over stored items in their existing order.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.ModeSet.__iter__() -> 'Iterator[Mode]'

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

``wavefem.ModeSet.__len__``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Return the number of stored items through Python len().

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.ModeSet.__len__() -> 'int'

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

``wavefem.ModeSet.count``
~~~~~~~~~~~~~~~~~~~~~~~~~

S.count(value) -> integer -- return number of occurrences of value

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.ModeSet.count(value)

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

``wavefem.ModeSet.index``
~~~~~~~~~~~~~~~~~~~~~~~~~

S.index(value, [start, [stop]]) -> integer -- return first index of value. Raises ValueError if the value is not present.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.ModeSet.index(value, start=0, stop=None)

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

``wavefem.ModeSolverError``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The guided-mode eigenproblem could not produce valid requested modes.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.ModeSolverError(*args: 'object')

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

Returns: ``ModeSolverError``.

``wavefem.PMLLayout.interfaces``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Return internal x/z coordinates that should be mesh-conforming.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.PMLLayout.interfaces(x_span: 'Sequence[float]', z_span: 'Sequence[float]') -> 'tuple[tuple[float, ...], tuple[float, ...]]'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``x_span``
     - Required
     - ``Sequence[float]``
     - Physical axis bounds in metres, with upper > lower. Range-based solver constructors also accept a positive extent for (0, extent).
   * - ``z_span``
     - Required
     - ``Sequence[float]``
     - Physical axis bounds in metres, with upper > lower. Range-based solver constructors also accept a positive extent for (0, extent).

Returns: ``tuple[tuple[float, ...], tuple[float, ...]]``.

``wavefem.PMLLayout.stretching``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Return diagonal stretch components ``(sx, sz)``.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.PMLLayout.stretching(x: 'ArrayLike', z: 'ArrayLike', *, x_span: 'Sequence[float]', z_span: 'Sequence[float]', k_reference: 'float') -> 'tuple[ComplexArray, ComplexArray]'

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
   * - ``x_span``
     - Required
     - ``Sequence[float]``
     - Physical axis bounds in metres, with upper > lower. Range-based solver constructors also accept a positive extent for (0, extent). Keyword-only.
   * - ``z_span``
     - Required
     - ``Sequence[float]``
     - Physical axis bounds in metres, with upper > lower. Range-based solver constructors also accept a positive extent for (0, extent). Keyword-only.
   * - ``k_reference``
     - Required
     - ``float``
     - Vacuum wavenumber in rad/m, used for coordinate scaling and material/PML evaluation. Keyword-only.

Returns: ``tuple[ComplexArray, ComplexArray]``.

``wavefem.PMLLayout.transform_isotropic``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Transform scalar material into diagonal ``(x, y, z)`` tensors.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.PMLLayout.transform_isotropic(eps_r: 'ArrayLike', mu_r: 'ArrayLike', sx: 'ArrayLike', sz: 'ArrayLike') -> 'tuple[ComplexArray, ComplexArray]'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``eps_r``
     - Required
     - ``ArrayLike``
     - Relative permittivity. Scalar or Cartesian diagonal material inputs are supported where the signature permits; callbacks return scalar/broadcast-compatible values at physical coordinates.
   * - ``mu_r``
     - Required
     - ``ArrayLike``
     - Relative permeability, scalar or Cartesian diagonal where supported. WaveFEM scattering requires equal actual and background permeability.
   * - ``sx``
     - Required
     - ``ArrayLike``
     - Complex coordinate-stretch factor along x or z, broadcast to the evaluation grid.
   * - ``sz``
     - Required
     - ``ArrayLike``
     - Complex coordinate-stretch factor along x or z, broadcast to the evaluation grid.

Returns: ``tuple[ComplexArray, ComplexArray]``.

``wavefem.PMLLayout.validate_domain``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``validate_domain`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.PMLLayout.validate_domain(x_span: 'Sequence[float]', z_span: 'Sequence[float]') -> 'None'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``x_span``
     - Required
     - ``Sequence[float]``
     - Physical axis bounds in metres, with upper > lower. Range-based solver constructors also accept a positive extent for (0, extent).
   * - ``z_span``
     - Required
     - ``Sequence[float]``
     - Physical axis bounds in metres, with upper > lower. Range-based solver constructors also accept a positive extent for (0, extent).

Returns: ``None``.

``wavefem.Scattering2D.x_span``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's x span value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.Scattering2D.x_span

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

``wavefem.Scattering2D.z_span``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's z span value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.Scattering2D.z_span

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

``wavefem.ScatteringResult.E_total``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's E total value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.ScatteringResult.E_total

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

``wavefem.ScatteringResult.H_total``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's H total value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.ScatteringResult.H_total

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

``wavefem.ScatteringResult.S``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Return an indexed outgoing modal amplitude.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.ScatteringResult.S(side: 'str', *, out_mode: 'int' = 0, in_mode: 'int' = 0) -> 'complex'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``side``
     - Required
     - ``str``
     - Lead/trace side or physical monitor coordinate, according to the method signature. Integrated scattering currently launches from the left lead.
   * - ``out_mode``
     - Optional
     - ``int``
     - Mode, case, array, or mapping selector. Python indexing is zero based; explicit mode(number) and standalone visualization use their documented one-based numbers. Keyword-only. Default: ``0``.
   * - ``in_mode``
     - Optional
     - ``int``
     - Mode, case, array, or mapping selector. Python indexing is zero based; explicit mode(number) and standalone visualization use their documented one-based numbers. Keyword-only. Default: ``0``.

Returns: ``complex``.

``wavefem.ScatteringResult.S11``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's S11 value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.ScatteringResult.S11

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

``wavefem.ScatteringResult.S21``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's S21 value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.ScatteringResult.S21

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

``wavefem.ScatteringResult.power_balance``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's power balance value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.ScatteringResult.power_balance

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

``wavefem.ScatteringResult.power_balance_error``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's power balance error value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.ScatteringResult.power_balance_error

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

``wavefem.ScatteringResult.reflection``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's reflection value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.ScatteringResult.reflection

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

``wavefem.ScatteringResult.transmission``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's transmission value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.ScatteringResult.transmission

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

``wavefem.ScatteringResult.visualize``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create and show a Matplotlib field figure.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.ScatteringResult.visualize(component: 'str' = 'E', *, quantity: "Literal['abs', 'real', 'imag', 'phase', 'norm']" = 'abs', part: "Literal['total', 'incident', 'scattered']" = 'total', ax: 'Any | None' = None, cmap: 'Any | None' = None, levels: 'int' = 50, colorbar: 'bool' = True, show: 'bool' = True) -> 'Any'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``component``
     - Optional
     - ``str``
     - Field component selection, using Cartesian electric/magnetic names such as Ex or Hy. The visualization API also supports its documented aggregate quantities. Default: ``'E'``.
   * - ``quantity``
     - Optional
     - ``Literal['abs', 'real', 'imag', 'phase', 'norm']``
     - Complex-data display selection, such as real, imag, abs, or phase; accepted values are given by the owning plotting API. Keyword-only. Default: ``'abs'``.
   * - ``part``
     - Optional
     - ``Literal['total', 'incident', 'scattered']``
     - Complex-data display selection, such as real, imag, abs, or phase; accepted values are given by the owning plotting API. Keyword-only. Default: ``'total'``.
   * - ``ax``
     - Optional
     - ``Any | None``
     - Existing Matplotlib axes for embedding a plot; None creates suitable axes when accepted. Keyword-only. Default: ``None``.
   * - ``cmap``
     - Optional
     - ``Any | None``
     - Matplotlib colormap name or object used for scalar field rendering. Keyword-only. Default: ``None``.
   * - ``levels``
     - Optional
     - ``int``
     - Contour-level count or explicit contour levels for scalar field plots. Keyword-only. Default: ``50``.
   * - ``colorbar``
     - Optional
     - ``bool``
     - Enable display of the figure, mesh overlay, or colorbar respectively. show=False returns plotting objects without opening a window. Keyword-only. Default: ``True``.
   * - ``show``
     - Optional
     - ``bool``
     - Enable display of the figure, mesh overlay, or colorbar respectively. show=False returns plotting objects without opening a window. Keyword-only. Default: ``True``.

Returns: ``Any``.

Pass ``show=False`` when embedding the returned axes. Use the zero-argument ``visualize_with_gui`` method for the native viewer.

``wavefem.ScatteringResult.visualize_with_gui``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open the complete result and all stored modes in the native viewer.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.ScatteringResult.visualize_with_gui() -> 'Any'

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

``wavefem.SolverError``
~~~~~~~~~~~~~~~~~~~~~~~

A finite-element linear or eigenvalue solve failed.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.SolverError(*args: 'object')

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

``wavefem.ViewerError``
~~~~~~~~~~~~~~~~~~~~~~~

The standalone native viewer could not be found or launched.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.ViewerError(*args: 'object')

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

Returns: ``ViewerError``.

``wavefem.WaveFEMError``
~~~~~~~~~~~~~~~~~~~~~~~~

Base class for actionable WaveFEM errors.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.WaveFEMError(*args: 'object')

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

Returns: ``WaveFEMError``.

``wavefem.find_viewer_executable``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Find the native GUI in an override, build tree, ``PATH``, or install.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.find_viewer_executable() -> 'Path'

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

Returns: ``Path``.

``WAVEFEM_VIEWER_EXECUTABLE`` has highest priority. Source checkouts are searched for both standalone ``WaveFEMViewer/build*`` trees and root CMake ``build*/WaveFEMViewer`` trees, including multi-config subdirectories, before potentially older installed copies.

``wavefem.launch_viewer``
~~~~~~~~~~~~~~~~~~~~~~~~~

Launch the native viewer for a result file or results directory.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.launch_viewer(path: 'str | PathLike[str] | None' = None) -> 'subprocess.Popen[bytes]'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``path``
     - Optional
     - ``str | PathLike[str] | None``
     - Filesystem destination/source for HDF5 persistence or viewer launch. A directory is accepted only by viewer/directory-inspection APIs; None follows the method-specific default. Default: ``None``.

Returns: ``subprocess.Popen[bytes]``.

With ``path=None`` the current directory is opened. Passing a directory lets the native viewer populate its in-window selector with every readable ``.h5`` and ``.hdf5`` file there.

``wavefem.fem.MaxwellParameters.from_material``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Construct parameters from ``wavefem.materials`` material data.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.fem.MaxwellParameters.from_material(*, k0: 'float', material: 'Material', ky: 'complex' = 0.0) -> 'MaxwellParameters'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``k0``
     - Required
     - ``float``
     - Vacuum wavenumber in rad/m, used for coordinate scaling and material/PML evaluation. Keyword-only.
   * - ``material``
     - Required
     - ``Material``
     - Material object defining relative electric and magnetic response. Exterior fills points outside placed material regions. Keyword-only.
   * - ``ky``
     - Optional
     - ``complex``
     - Real Fourier wavenumber along invariant y, in rad/m; zero gives propagation in the computational plane. Keyword-only. Default: ``0.0``.

Returns: ``MaxwellParameters``.

``wavefem.fem.MixedFEMSystem.dimensionless_k0``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Vacuum wavenumber used on the scaled computational mesh.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.fem.MixedFEMSystem.dimensionless_k0

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

``wavefem.fem.MixedFEMSystem.dimensionless_ky``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prescribed y-wavenumber used on the scaled computational mesh.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.fem.MixedFEMSystem.dimensionless_ky

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

``wavefem.fem.MixedFEMSystem.ndofs``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Number of mixed finite-element degrees of freedom.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.fem.MixedFEMSystem.ndofs

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

``wavefem.fem.MixedFEMSystem.pec_dofs``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Outer and internal DOFs imposing zero tangential electric field.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.fem.MixedFEMSystem.pec_dofs

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

Returns: ``NDArray[np.integer]``.

``wavefem.fem.MixedFEMSystem.physical_coordinates``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Return cell-basis quadrature coordinates in metres.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.fem.MixedFEMSystem.physical_coordinates() -> 'NDArray[np.float64]'

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

``wavefem.fem.MixedFieldSolution.interpolate``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Interpolate and return the quadrature fields ``(E_t, E_y)``.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.fem.MixedFieldSolution.interpolate() -> 'tuple[object, object]'

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

Returns: ``tuple[object, object]``.

``wavefem.fem.MixedFieldSolution.split_coefficients``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Return ``(E_t coefficients, E_y coefficients)`` safely.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.fem.MixedFieldSolution.split_coefficients() -> 'tuple[NDArray[np.complex128], NDArray[np.complex128]]'

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

Returns: ``tuple[NDArray[np.complex128], NDArray[np.complex128]]``.

Composite-space DOFs are grouped by topological type and must not be split by assuming contiguous element blocks.

``wavefem.geometry.Rectangle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Axis-aligned material rectangle in the x-z solve plane.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.geometry.Rectangle(x: 'tuple[float, float]', z: 'tuple[float, float]') -> None

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

``wavefem.geometry.Rectangle.contains``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``contains`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.geometry.Rectangle.contains(x: 'ArrayLike', z: 'ArrayLike') -> 'NDArray[np.bool_]'

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

``wavefem.geometry.Circle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Circular material region in the x-z solve plane.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.geometry.Circle(center: 'tuple[float, float]', radius: 'float') -> None

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

``wavefem.geometry.Circle.contains``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``contains`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.geometry.Circle.contains(x: 'ArrayLike', z: 'ArrayLike') -> 'NDArray[np.bool_]'

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

``wavefem.geometry.Polygon``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simple polygon represented by ordered ``(x, z)`` vertices.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.geometry.Polygon(points: 'tuple[tuple[float, float], ...]') -> None

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

``wavefem.geometry.Polygon.contains``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``contains`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.geometry.Polygon.contains(x: 'ArrayLike', z: 'ArrayLike') -> 'NDArray[np.bool_]'

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

``wavefem.geometry.Region.contains``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``contains`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.geometry.Region.contains(x: 'ArrayLike', z: 'ArrayLike') -> 'NDArray[np.bool_]'

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

``wavefem.geometry.PECSheet``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An ideal zero-thickness PEC sheet parallel to the z axis.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.geometry.PECSheet(name: 'str', x: 'float', z: 'tuple[float, float]', background: 'bool') -> None

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
   * - ``x``
     - Required
     - ``float``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature.
   * - ``z``
     - Required
     - ``tuple[float, float]``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature.
   * - ``background``
     - Required
     - ``bool``
     - In placement methods, True adds a z-invariant background region/sheet; False adds an actual-device perturbation. In material/geometry constructors, this is the exterior Material object.

Returns: ``PECSheet``.

A background sheet belongs to the unperturbed guide and, before slots are cut, to the actual device as well. Background sheets must span the whole solve domain in z so that the lead eigenproblem remains invariant.

``wavefem.geometry.PECSlot``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A finite actual-device opening cut from one background PEC sheet.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.geometry.PECSlot(name: 'str', sheet_name: 'str', z: 'tuple[float, float]') -> None

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
   * - ``sheet_name``
     - Required
     - ``str``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported.
   * - ``z``
     - Required
     - ``tuple[float, float]``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature.

Returns: ``PECSlot``.

``wavefem.geometry.PECSegment``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One closed PEC line segment in an actual or background profile.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.geometry.PECSegment(name: 'str', x: 'float', z: 'tuple[float, float]') -> None

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
   * - ``x``
     - Required
     - ``float``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature.
   * - ``z``
     - Required
     - ``tuple[float, float]``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature.

Returns: ``PECSegment``.

``wavefem.geometry.GeometryModel.add_circle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add a circle; compact circles cannot define a straight guide.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.geometry.GeometryModel.add_circle(*, center: 'Sequence[float]', radius: 'float', material: 'Material | complex | float', background: 'bool' = False, name: 'str | None' = None) -> 'Region'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``center``
     - Required
     - ``Sequence[float]``
     - Centre coordinates in metres, ordered as the package computational axes: x-y, x-z, or x-y-z. Keyword-only.
   * - ``radius``
     - Required
     - ``float``
     - Positive radius in metres; inner_radius describes the hollow inner boundary where supported. Keyword-only.
   * - ``material``
     - Required
     - ``Material | complex | float``
     - Material object defining relative electric and magnetic response. Exterior fills points outside placed material regions. Keyword-only.
   * - ``background``
     - Optional
     - ``bool``
     - In placement methods, True adds a z-invariant background region/sheet; False adds an actual-device perturbation. In material/geometry constructors, this is the exterior Material object. Keyword-only. Default: ``False``.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Returns: ``Region``.

``wavefem.geometry.GeometryModel.add_pec``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add an ideal, mesh-conforming, constant-x PEC sheet.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.geometry.GeometryModel.add_pec(*, x: 'float', z: 'ZExtent' = 'all', background: 'bool' = False, name: 'str | None' = None) -> 'PECSheet'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``x``
     - Required
     - ``float``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature. Keyword-only.
   * - ``z``
     - Optional
     - ``ZExtent``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature. Keyword-only. Default: ``'all'``.
   * - ``background``
     - Optional
     - ``bool``
     - In placement methods, True adds a z-invariant background region/sheet; False adds an actual-device perturbation. In material/geometry constructors, this is the exterior Material object. Keyword-only. Default: ``False``.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Returns: ``PECSheet``.

``background=True`` makes the sheet part of both the unperturbed guide and the actual device. Such a sheet must use ``z="all"``. Finite openings are then introduced with ``add_slot``. With ``background=False``, a finite z span describes an actual-only plate.

``wavefem.geometry.GeometryModel.add_polygon``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add a simple polygon; compact polygons modify the actual device.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.geometry.GeometryModel.add_polygon(*, points: 'Iterable[Sequence[float]]', material: 'Material | complex | float', background: 'bool' = False, name: 'str | None' = None) -> 'Region'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``points``
     - Required
     - ``Iterable[Sequence[float]]``
     - Coordinates used by the object or evaluation operation. Mesh geometry uses physical metres; low-level FE operators use their basis coordinate scale. See the array-shape conventions. Keyword-only.
   * - ``material``
     - Required
     - ``Material | complex | float``
     - Material object defining relative electric and magnetic response. Exterior fills points outside placed material regions. Keyword-only.
   * - ``background``
     - Optional
     - ``bool``
     - In placement methods, True adds a z-invariant background region/sheet; False adds an actual-device perturbation. In material/geometry constructors, this is the exterior Material object. Keyword-only. Default: ``False``.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Returns: ``Region``.

``wavefem.geometry.GeometryModel.add_rectangle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add an axis-aligned rectangle.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.geometry.GeometryModel.add_rectangle(*, x: 'Sequence[float]', z: 'ZExtent', material: 'Material | complex | float', background: 'bool' = False, name: 'str | None' = None) -> 'Region'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``x``
     - Required
     - ``Sequence[float]``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature. Keyword-only.
   * - ``z``
     - Required
     - ``ZExtent``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature. Keyword-only.
   * - ``material``
     - Required
     - ``Material | complex | float``
     - Material object defining relative electric and magnetic response. Exterior fills points outside placed material regions. Keyword-only.
   * - ``background``
     - Optional
     - ``bool``
     - In placement methods, True adds a z-invariant background region/sheet; False adds an actual-device perturbation. In material/geometry constructors, this is the exterior Material object. Keyword-only. Default: ``False``.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Returns: ``Region``.

Background-guide regions must use ``z="all"`` so the material is invariant along the nominal propagation direction.

``wavefem.geometry.GeometryModel.add_slot``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cut a compact actual-only slot from a z-invariant background PEC.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.geometry.GeometryModel.add_slot(*, pec: 'PECSheet | str', z: 'Sequence[float]', name: 'str | None' = None) -> 'PECSlot'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``pec``
     - Required
     - ``PECSheet | str``
     - PEC sheet/plane handle(s) that define zero tangential electric field. Slots must refer to a background sheet. Keyword-only.
   * - ``z``
     - Required
     - ``Sequence[float]``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature. Keyword-only.
   * - ``name``
     - Optional
     - ``str | None``
     - Human-readable name used for geometry selection, physical tags, diagnostics, or plot labels. None selects an automatic label where supported. Keyword-only. Default: ``None``.

Returns: ``PECSlot``.

``wavefem.geometry.GeometryModel.background_regions``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's background regions value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.geometry.GeometryModel.background_regions

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

Returns: ``tuple[Region, ...]``.

``wavefem.geometry.GeometryModel.material_at``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluate scalar ``(eps_r, mu_r)`` arrays at arbitrary points.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.geometry.GeometryModel.material_at(x: 'ArrayLike', z: 'ArrayLike', *, profile: "Literal['actual', 'background']") -> 'tuple[NDArray[np.complex128], NDArray[np.complex128]]'

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
   * - ``profile``
     - Required
     - ``Literal['actual', 'background']``
     - Material profile selector, usually actual or background for the scattered-field formulation. Keyword-only.

Returns: ``tuple[NDArray[np.complex128], NDArray[np.complex128]]``.

``wavefem.geometry.GeometryModel.pec_segments``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Return non-overlapping PEC segments for one material profile.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.geometry.GeometryModel.pec_segments(*, profile: 'Profile') -> 'tuple[PECSegment, ...]'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``profile``
     - Required
     - ``Profile``
     - Material profile selector, usually actual or background for the scattered-field formulation. Keyword-only.

Returns: ``tuple[PECSegment, ...]``.

Background sheets are returned whole in the background profile. In the actual profile each finite slot is subtracted, producing the PEC segments on either side of the opening.

``wavefem.geometry.GeometryModel.perturbations``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's perturbations value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.geometry.GeometryModel.perturbations

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

Returns: ``tuple[Region, ...]``.

``wavefem.geometry.GeometryModel.physical_names``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's physical names value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.geometry.GeometryModel.physical_names

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

Returns: ``dict[int, str]``.

``wavefem.geometry.GeometryModel.region_tag_at``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Return stable actual-material physical tags at the points.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.geometry.GeometryModel.region_tag_at(x: 'ArrayLike', z: 'ArrayLike') -> 'NDArray[np.int32]'

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
   * - ``z``
     - Required
     - ``ArrayLike``
     - Physical coordinate(s) in metres. Point-evaluation methods accept broadcast-compatible arrays; placement methods accept the axis interval or selector shown in the signature.

Returns: ``NDArray[np.int32]``.

``wavefem.geometry.GeometryModel.slots_in``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Return the slots cut from ``pec``, sorted by increasing z.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.geometry.GeometryModel.slots_in(pec: 'PECSheet | str') -> 'tuple[PECSlot, ...]'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``pec``
     - Required
     - ``PECSheet | str``
     - PEC sheet/plane handle(s) that define zero tangential electric field. Slots must refer to a background sheet.

Returns: ``tuple[PECSlot, ...]``.

``wavefem.materials.DiagonalMaterial.from_isotropic``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create the internal representation of one public material.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.materials.DiagonalMaterial.from_isotropic(material: 'Material') -> "'DiagonalMaterial'"

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``material``
     - Required
     - ``Material``
     - Material object defining relative electric and magnetic response. Exterior fills points outside placed material regions.

Returns: ``DiagonalMaterial``.

``wavefem.materials.DiagonalTensor.as_array``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Return a new complex array in physical ``(xx, yy, zz)`` order.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.materials.DiagonalTensor.as_array() -> 'ComplexArray'

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

``wavefem.materials.DiagonalTensor.is_isotropic``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Whether all three Cartesian diagonal entries are equal.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.materials.DiagonalTensor.is_isotropic

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

``wavefem.materials.DiagonalTensor.isotropic``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Expand one scalar without changing its value or component order.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.materials.DiagonalTensor.isotropic(value: 'ScalarMaterialInput') -> "'DiagonalTensor'"

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``value``
     - Required
     - ``ScalarMaterialInput``
     - Scalar or array to validate, transform, interpolate, or store. It must satisfy the owning operation and the expected type in this table.

Returns: ``DiagonalTensor``.

``wavefem.mesh.Mesh2D.elements_in``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the ``elements_in`` operation on the supplied data; the returned value follows the type contract below.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.mesh.Mesh2D.elements_in(region: 'str | int') -> 'NDArray[np.int64]'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``region``
     - Required
     - ``str | int``
     - Geometry primitive or region selector identifying the physical support. Use the class/union in the signature; electrostatics also accepts named exterior boundaries.

Returns: ``NDArray[np.int64]``.

``wavefem.mesh.Mesh2D.facets_in_slot``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Return global facet indices released by one named PEC slot.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.mesh.Mesh2D.facets_in_slot(slot: 'str') -> 'NDArray[np.int32]'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``slot``
     - Required
     - ``str``
     - Finite aperture specification(s) releasing part of a background PEC sheet in the actual device.

Returns: ``NDArray[np.int32]``.

``wavefem.mesh.Mesh2D.pec_facets``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Return global ``MeshTri`` facet indices for one PEC profile.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.mesh.Mesh2D.pec_facets(profile: "Literal['actual', 'background']") -> 'NDArray[np.int32]'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``profile``
     - Required
     - ``Literal['actual', 'background']``
     - Material profile selector, usually actual or background for the scattered-field formulation.

Returns: ``NDArray[np.int32]``.

``wavefem.modes.ModeFEMSystem.divergence_residual``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Return the normalized weak residual of ``div(eps_r E) = 0``.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.modes.ModeFEMSystem.divergence_residual(full_vector: 'ArrayLike', neff: 'complex') -> 'float'

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

Test functions vanish at the two outer endpoints, so conductor surface charge does not contaminate this bulk Gauss-law diagnostic.

``wavefem.modes.ModeFEMSystem.elements``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's elements value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.modes.ModeFEMSystem.elements

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

``wavefem.modes.ModeFEMSystem.expand``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Expand a reduced PEC/PMC vector into physical component blocks.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.modes.ModeFEMSystem.expand(vector: 'ArrayLike') -> 'ComplexArray'

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

``wavefem.modes.ModeFEMSystem.ndofs``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Number of unconstrained electric-field degrees of freedom.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.modes.ModeFEMSystem.ndofs

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

``wavefem.modes.ModeFEMSystem.polynomial``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Return ``A0 + neff*A1 + neff**2*A2``.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.modes.ModeFEMSystem.polynomial(neff: 'complex') -> 'csr_matrix'

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

``wavefem.modes.ModeFEMSystem.relative_hermiticity_errors``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Return coefficient Hermiticity errors for lossless validation.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.modes.ModeFEMSystem.relative_hermiticity_errors() -> 'tuple[float, float, float]'

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

``wavefem.operators.TangentialHcurlField``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Structural typing protocol for compatible field/visualization objects. Implement the declared attributes and methods; this protocol is not instantiated directly.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.operators.TangentialHcurlField()

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

Returns: ``TangentialHcurlField``.

``wavefem.operators.TangentialHcurlField.__getitem__``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Select an item using Python square-bracket indexing; integer indices are zero based.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.operators.TangentialHcurlField.__getitem__(key: 'SupportsIndex') -> 'NDArray[np.generic]'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``key``
     - Required
     - ``SupportsIndex``
     - Mode, case, array, or mapping selector. Python indexing is zero based; explicit mode(number) and standalone visualization use their documented one-based numbers.

Returns: ``NDArray[np.generic]``.

``wavefem.operators.InvariantH1Field``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Structural typing protocol for compatible field/visualization objects. Implement the declared attributes and methods; this protocol is not instantiated directly.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.operators.InvariantH1Field()

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

Returns: ``InvariantH1Field``.

``wavefem.projection.ElectromagneticProjector.project``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Solve the small dense electromagnetic Gram system for amplitudes.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.projection.ElectromagneticProjector.project(E: 'ArrayLike', H: 'ArrayLike') -> 'ProjectionResult'

.. list-table:: Input arguments
   :header-rows: 1
   :widths: 18 17 25 40

   * - Argument
     - Required / optional
     - Expected type
     - Explanation
   * - ``E``
     - Required
     - ``ArrayLike``
     - Complex electromagnetic field samples or FE field objects. Cartesian components follow x,y,z order; left/right denote the selected trace side.
   * - ``H``
     - Required
     - ``ArrayLike``
     - Complex electromagnetic field samples or FE field objects. Cartesian components follow x,y,z order; left/right denote the selected trace side.

Returns: ``ProjectionResult``.

``wavefem.sources.EquivalentSource.is_zero``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the object's is zero value; this is an attribute access, not a function call.

Signature (defaults are library defaults):

.. code-block:: text

   wavefem.sources.EquivalentSource.is_zero

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

Export aliases and constants
----------------------------

Aliases below have exactly the same input tables and return contracts as their targets. Constants/type aliases are values, not calls, and take no input arguments.

.. list-table:: Exports
   :header-rows: 1

   * - Name
     - Value or target
   * - ``wavefem.exceptions.ConfigurationError``
     - ``wavefem.ConfigurationError``
   * - ``wavefem.exceptions.MaterialError``
     - ``wavefem.MaterialError``
   * - ``wavefem.exceptions.MeshError``
     - ``wavefem.MeshError``
   * - ``wavefem.exceptions.ModeProjectionError``
     - ``wavefem.ModeProjectionError``
   * - ``wavefem.exceptions.ModeSolverError``
     - ``wavefem.ModeSolverError``
   * - ``wavefem.exceptions.SolverError``
     - ``wavefem.SolverError``
   * - ``wavefem.exceptions.ViewerError``
     - ``wavefem.ViewerError``
   * - ``wavefem.exceptions.WaveFEMError``
     - ``wavefem.WaveFEMError``
   * - ``wavefem.frequency.Frequency``
     - ``wavefem.Frequency``
   * - ``wavefem.frequency.resolve_frequency``
     - ``wavefem.resolve_frequency``
   * - ``wavefem.hdf5.H5FileData``
     - ``wavefem.H5FileData``
   * - ``wavefem.hdf5.H5ModeData``
     - ``wavefem.H5ModeData``
   * - ``wavefem.hdf5.H5ResultData``
     - ``wavefem.H5ResultData``
   * - ``wavefem.hdf5.load_h5``
     - ``wavefem.load_h5``
   * - ``wavefem.hdf5.save_result_h5``
     - ``wavefem.save_result_h5``
   * - ``wavefem.hdf5.save_sweep_h5``
     - ``wavefem.save_sweep_h5``
   * - ``wavefem.incident.IncidentMode``
     - ``wavefem.IncidentMode``
   * - ``wavefem.materials.Material``
     - ``wavefem.Material``
   * - ``wavefem.modes.CrossSection``
     - ``wavefem.CrossSection``
   * - ``wavefem.modes.Mode``
     - ``wavefem.Mode``
   * - ``wavefem.modes.ModeSet``
     - ``wavefem.ModeSet``
   * - ``wavefem.modes.ModeSolver``
     - ``wavefem.ModeSolver``
   * - ``wavefem.pml.PML``
     - ``wavefem.PML``
   * - ``wavefem.pml.PMLLayout``
     - ``wavefem.PMLLayout``
   * - ``wavefem.results.Diagnostic``
     - ``wavefem.Diagnostic``
   * - ``wavefem.results.DiagnosticReport``
     - ``wavefem.DiagnosticReport``
   * - ``wavefem.results.ScatteringResult``
     - ``wavefem.ScatteringResult``
   * - ``wavefem.scattering.Scattering2D``
     - ``wavefem.Scattering2D``
   * - ``wavefem.scattering.SolverOptions``
     - ``wavefem.SolverOptions``
   * - ``wavefem.scene.Scene2D``
     - ``wavefem.Scene2D``
   * - ``wavefem.scene.SceneLine``
     - ``wavefem.SceneLine``
   * - ``wavefem.sweep.FrequencySweepResult``
     - ``wavefem.FrequencySweepResult``
   * - ``wavefem.viewer.find_viewer_executable``
     - ``wavefem.find_viewer_executable``
   * - ``wavefem.viewer.launch_viewer``
     - ``wavefem.launch_viewer``
   * - ``wavefem.C0``
     - ``299792458.0``
   * - ``wavefem.EPSILON_0``
     - ``8.8541878188e-12``
   * - ``wavefem.ETA_0``
     - ``376.7303134118051``
   * - ``wavefem.MU_0``
     - ``1.25663706127e-06``
   * - ``wavefem.SCHEMA_NAME``
     - ``'wavefem'``
   * - ``wavefem.SCHEMA_VERSION``
     - ``1``
   * - ``wavefem.constants.C0``
     - ``299792458.0``
   * - ``wavefem.constants.EPSILON_0``
     - ``8.8541878188e-12``
   * - ``wavefem.constants.ETA_0``
     - ``376.7303134118051``
   * - ``wavefem.constants.MU_0``
     - ``1.25663706127e-06``
   * - ``wavefem.constants.SPEED_OF_LIGHT_M_PER_S``
     - ``299792458.0``
   * - ``wavefem.constants.VACUUM_IMPEDANCE_OHM``
     - ``376.7303134118051``
   * - ``wavefem.constants.VACUUM_PERMEABILITY_H_PER_M``
     - ``1.25663706127e-06``
   * - ``wavefem.constants.VACUUM_PERMITTIVITY_F_PER_M``
     - ``8.8541878188e-12``
   * - ``wavefem.hdf5.SCHEMA_NAME``
     - ``'wavefem'``
   * - ``wavefem.hdf5.SCHEMA_VERSION``
     - ``1``
   * - ``wavefem.incident.IncidentSide``
     - ``typing.Literal['left', 'right']``
   * - ``wavefem.scene.SceneKind``
     - ``typing.Literal['pec', 'pmc', 'wave_port', 'pml']``
   * - ``wavefem.sources.IncidentField``
     - ``typing.Callable[[numpy.ndarray[tuple[typing.Any, ...], numpy.dtype[numpy.floating]], numpy.ndarray[tuple[typing.Any, ...], numpy.dtype[numpy.floating]]], object]``
