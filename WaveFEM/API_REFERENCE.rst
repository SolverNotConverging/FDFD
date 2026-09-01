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
   modes = sim.solve_modes(...)
   sim.set_incident_mode(modes[0])
   result = sim.run(h5_path="wavefem_result.h5")

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
   ).solve(num_modes=1, neff_guess=1.0)
   sim.set_modes(modes)
   sim.set_incident_mode(0)
   result = sim.run(h5_path="callback_result.h5")

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

Sets physical z coordinates for the two modal monitor lines. The coordinates must satisfy ``left < right``, lie inside the non-PML interior, surround every geometry-defined finite perturbation, and cross uniform lead material. The lines are inserted as mesh-conforming partitions during ``mesh()``.

If omitted, WaveFEM chooses monitors between a geometry-defined perturbation and the z-PMLs. Callback devices should always set them explicitly because WaveFEM cannot infer the callback contrast bounds. Call this method before meshing and before selecting an incident reference plane.

``mesh``
^^^^^^^^

.. code-block:: python

   sim.mesh(
       *,
       max_element_size: float | None = None,
       wavelength_elements: int = 10,
       refine_interfaces: bool = True,
       dielectric_refinement_factor: float = 0.5,
       pec_refinement_factor: float = 0.5,
       pec_refinement_distance: float | None = None,
   ) -> wavefem.mesh.Mesh2D

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
   ) -> wf.ModeSet

Builds the z-invariant background ``CrossSection`` and solves forward mode families near ``neff_guess``.

* ``side`` validates the requested lead name. The current device architecture uses the same unperturbed cross-section on both sides.

* ``num_modes`` is the number of validated modes required.

* ``num_elements`` controls the 1D cross-section mesh. When omitted, it is derived from the 2D target size with a minimum of 40 elements.

* Geometry-backed background layers must be z-invariant rectangles.

* Geometry-backed background PEC sheets become internal PEC points in the one-dimensional port problem. At each point ``E_y`` and ``E_z`` are zero while the two one-sided normal ``E_x`` traces remain unknown.

* Lead materials must be lossless.

* Open guides require an x-PML. With one present, the integrated workflow filters PML/radiation candidates and retains bound modes above the exterior light line.

The method stores and returns the resulting ``ModeSet`` and clears any previous incident selection. With a nonzero ``angle``, this first set is the normal-incidence family catalogue used to choose the incident family. Callback devices must use ``set_modes`` instead.

``set_modes``
^^^^^^^^^^^^^

.. code-block:: python

   sim.set_modes(modes: wf.ModeSet) -> wf.ModeSet

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
   ) -> wf.ScatteringResult

Assembles the mixed Maxwell system, forms the volume permittivity-contrast source and any released-PEC aperture source, prescribes ``-E_inc,t`` on finite inserted PEC plates, solves the outgoing scattered field, reconstructs total E and H, projects both lead monitors onto forward/backward modes, and computes power terms.

Preconditions:

* ``mesh()`` has completed.

* ``solve_modes()`` or ``set_modes()`` has completed.

* ``set_incident_mode()`` has selected a left-incident propagating mode.

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
   ) -> wf.ScatteringResult

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
   sweep = sim.sweep_frequencies(
       frequencies_hz,
       h5_path="frequency_sweep.h5",
       mesh_options={"wavelength_elements": 10},
       mode_options={"num_modes": 2, "neff_guess": 2.4},
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
   )

* ``linear_solver``: only ``"direct"`` is implemented.

* ``tolerance``: maximum accepted relative residual for the sparse field solve.

* ``quadrature_order``: mixed FEM and monitor quadrature order; must be at least two.

* ``projection_condition_limit``: maximum accepted condition number of the normalized modal Gram system; must exceed one.

Instances are frozen and validated at construction.

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

Writes all frequencies and complete per-point results with ``save_sweep_h5``. Modes are taken from each result's ``modes`` tuple. The absolute written path is returned, but the frozen sweep's ``h5_path`` is not mutated; integrated ``sweep_frequencies(h5_path=...)`` returns a copied object with that field set.

``Diagnostic``
~~~~~~~~~~~~~~

.. code-block:: python

   wf.Diagnostic(
       severity: Literal["info", "warning", "error"],
       code: str,
       message: str,
   )

A single machine-readable diagnostic. ``code`` is suitable for programmatic filtering; ``message`` is human-readable. Direct construction relies on callers respecting the annotations.

``DiagnosticReport``
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   wf.DiagnosticReport(diagnostics: tuple[wf.Diagnostic, ...])

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

Writes one nonempty ordered result sequence and returns the absolute path. ``frequencies_hz`` must be a real positive 1D array with exactly one entry per result. ``Scattering2D.sweep_frequencies`` additionally requires strict increasing order before calling this lower-level writer.

When ``modes_per_result`` is ``None``, each result's optional ``modes`` attribute is used. Otherwise it must contain exactly one iterable per result and those explicit mode groups are stored. The supplied sweep frequency is forced into each prepared record and checked against any frequency metadata already on the result. Validation and atomic-replacement behavior match ``save_result_h5``.

``load_h5``
~~~~~~~~~~~

.. code-block:: python

   wf.load_h5(path: str | os.PathLike[str]) -> wf.H5FileData

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

One portable sampled lead mode. ``x`` is a read-only transverse grid in metres; ``E`` and ``H`` have shape ``(3, n)`` in Cartesian order. ``metadata`` contains modal scalars and labels such as ``beta``, ``neff``, ``power``, ``complex_power``, ``ky``, ``omega``, ``direction``, ``classification``, ``normalization``, and residuals. ``raw_components`` preserves available mixed-representation arrays such as ``x_nodes``, cellwise ``E_x``, nodal ``E_y/E_z``, magnetic samples, and endpoint ``H_x`` traces for research inspection.

Separate HDF5 viewer project
----------------------------

The GUI is intentionally not part of the ``wavefem`` Python distribution. The sibling `WaveFEMViewer project <../WaveFEMViewer/README.rst>`_ owns its native C++20/Qt source, ``wavefem-viewer`` executable, lazy HDF5 reader, cached QPainter renderer, deployment scripts, and user documentation. It does not import or depend on Python, NumPy, h5py, Matplotlib, or ``wavefem``, and can therefore inspect result files on a machine without the FEM solver installed.

Its README documents cross-platform CMake builds, installation, direct-path launch, file-picker workflow, tab controls, supported schema data, and the headless inspection/benchmark utility. Python exposes ``wf.find_viewer_executable()`` and ``wf.launch_viewer(path=None)``. Discovery checks ``WAVEFEM_VIEWER_EXECUTABLE``, standalone ``WaveFEMViewer/build*`` trees, repository-root ``build*/WaveFEMViewer`` trees, ``PATH``, and the default Windows installation, in that order. A directory target populates the native viewer's HDF5 selector; ``None`` targets the current directory.

``FrequencySweepResult.visualize()`` draws S11/S21 with Matplotlib. ``FrequencySweepResult.visualize_with_gui()`` reuses an associated archive or saves ``wavefem_sweep.h5``, then opens every sweep point and stored mode. The ``wavefem-inspect-h5`` command is headless by default; ``--gui`` launches the native viewer, and ``--gui`` without a path opens the current directory. The example inspector launches that directory GUI when run with no arguments. For every 2D vector/material plot it displays ``z`` horizontally and ``x`` vertically while leaving file storage in ``(x, z)`` order. Dielectric material is grey, PEC is yellow, PMC is blue, wave ports are red, and PML interfaces are green dashed lines.

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

Adds a non-overlapping, mesh-conforming material interval. The interval must be inside ``x_span`` and its name must be unique.

``add_pec``
^^^^^^^^^^^

.. code-block:: python

   cross_section.add_pec(
       *,
       x: float,
       name: str | None = None,
   ) -> wavefem.modes.PECBoundary

Adds a named, zero-thickness PEC sheet invariant in z. ``x`` is a finite physical coordinate strictly inside ``x_span``; duplicate coordinates and duplicate PEC names raise ``ConfigurationError``. The returned immutable ``PECBoundary(x, name)`` record is retained in the sorted ``cross_section.pec_boundaries`` list.

The coordinate is guaranteed to be a mode-mesh node. Assembly constrains the nodal tangential components ``E_y`` and ``E_z`` at that node for either PEC or PMC outer truncation. Every cellwise ``E_x`` DOF remains free, preserving independent normal traces and surface charge on the two sheet faces. The sheet node is also omitted from bulk divergence-test rows, because the surface charge makes a volume Gauss-law residual inappropriate there.

``interfaces``
^^^^^^^^^^^^^^

Returns sorted outer boundaries and all material, internal-PEC, and PML interfaces in metres. The mode mesh always conforms to these coordinates.

``material_at``
^^^^^^^^^^^^^^^

.. code-block:: python

   cross_section.material_at(x) -> tuple[eps_r_array, mu_r_array]

Evaluates scalar physical material values at arbitrary x coordinates. Layer endpoints are included in their layer masks.

``diagonal_material_at``
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   cross_section.diagonal_material_at(
       x,
       *,
       k_reference: float,
   ) -> tuple[eps_diagonal, mu_diagonal]

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
       num_elements: int = 160,
       quadrature_order: int = 4,
       dense_linearization_limit: int = 420,
   )

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
   ) -> wf.ModeSet

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

Broadcasts x and z and multiplies both traces by ``exp(i*beta*(z-reference_plane))``.

``phase_factor``
^^^^^^^^^^^^^^^^

.. code-block:: python

   mode.phase_factor(z, *, reference_plane: float = 0.0)

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

An immutable polynomial complex-stretch specification.

* ``thickness`` is positive and measured in metres.

* ``order`` is a positive polynomial order.

* ``target_reflection`` is a nominal amplitude target strictly between zero and one, not a guaranteed achieved reflection for a finite discretization.

``maximum_imaginary_stretch``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   pml.maximum_imaginary_stretch(k_reference: float) -> float

Computes the profile peak that gives the nominal two-pass reflection target for a positive reference wavenumber in rad/m.

``stretch``
^^^^^^^^^^^

.. code-block:: python

   pml.stretch(depth, k_reference) -> complex ndarray

Evaluates ``s = 1 + i*alpha_max*(clip(depth,0,thickness)/thickness)**order``. Depth and reference wavenumber use SI units.

``PMLLayout``
~~~~~~~~~~~~~

.. code-block:: python

   wf.PMLLayout(x: wf.PML | None = None, z: wf.PML | None = None)

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

* ``isotropic(value)`` creates three equal entries.

* ``is_isotropic`` checks exact equality.

* ``as_array()`` returns a new complex ``(3,)`` array in ``(xx,yy,zz)`` order.

``DiagonalMaterial``
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   DiagonalMaterial(eps_r: DiagonalTensor, mu_r: DiagonalTensor)

``from_isotropic(material)`` expands a scalar ``Material``.

``as_diagonal_material``
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   as_diagonal_material(
       material: Material | DiagonalMaterial,
   ) -> DiagonalMaterial

Accepts a ``Material`` or existing ``DiagonalMaterial`` and returns the explicit diagonal representation used by FEM assembly and PML transformations. Other types raise ``MaterialError``.

These types do not add general off-diagonal anisotropy.

Advanced operator and FEM API
-----------------------------

``modified_curl``
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from wavefem.operators import modified_curl

   modified_curl(tangential, invariant, ky) -> ndarray

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

``k0`` is positive. ``ky`` is currently real. Each constitutive coefficient may be a scalar, a three-entry diagonal, a quadrature-compatible array, or a ``coefficient(x,z)`` callback. Components use ``(x,y,z)`` order.

``MaxwellParameters.from_material(k0=..., material=..., ky=...)`` expands a scalar ``Material`` into explicit diagonal arrays.

``MixedFEMSystem``
~~~~~~~~~~~~~~~~~~

Stores:

* ``basis``: composite ``ElementTriN1 * ElementTriP1`` basis;

* ``matrix``: sparse complex Maxwell matrix;

* ``parameters``: physical ``MaxwellParameters``;

* ``physical_mesh``: original SI ``MeshTri``;

* ``length_scale``: metres per computational coordinate unit;

* ``quadrature_order``: integration order retained for volume and facet forms;

* ``internal_pec_facets``: sorted unique interior-facet indices on which the actual electric tangential trace is constrained.

Properties ``ndofs``, ``pec_dofs``, ``dimensionless_k0``, and ``dimensionless_ky`` expose assembly sizes/scales. ``pec_dofs`` is the union of the complete outer-boundary DOF set and every Nedelec/P1 trace DOF on ``internal_pec_facets``; normal electric traces are not part of the facet constraint. ``physical_coordinates()`` returns quadrature coordinates in metres.

``MixedFieldSolution``
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   MixedFieldSolution(
       basis,
       coefficients: complex ndarray,
       solve_info: Mapping[str, object] | None = None,
   )

Validates a finite coefficient vector of length ``basis.N`` and stores metadata as a read-only mapping.

* ``split_coefficients()`` safely returns Nedelec and H1 coefficient blocks using scikit-fem's topology-aware split.

* ``interpolate()`` returns quadrature fields ``(E_t,E_y)``.

Assembly helpers
~~~~~~~~~~~~~~~~

``create_mixed_basis``
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   create_mixed_basis(mesh: skfem.MeshTri, *, intorder: int = 4) -> skfem.Basis

Creates the conforming first-order composite basis.

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

Normalizes scalar/diagonal data to shape ``(3, nelements, nquadrature)``. Accepted callback results include a scalar, ``x.shape``, ``(3,)``, ``(3,*x.shape)``, or ``(*x.shape,3)``.

``assemble_maxwell_matrix``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   assemble_maxwell_matrix(
       basis: skfem.Basis,
       parameters: MaxwellParameters,
   ) -> scipy.sparse.csr_matrix

Assembles the complex sesquilinear curl-curl minus material-mass matrix. The basis must be the expected Nedelec-H1 composite and ``mu_r`` must be nonzero at every quadrature point.

``assemble_mixed_system``
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   assemble_mixed_system(
       mesh: skfem.MeshTri,
       parameters: MaxwellParameters,
       *,
       intorder: int = 4,
       length_scale: float = 1.0,
       internal_pec_facets: ArrayLike = (),
   ) -> MixedFEMSystem

Creates the basis and matrix. ``length_scale`` is the number of physical metres per computational coordinate unit. Material callbacks still receive physical metres. The high-level solver uses ``length_scale = 1/k0`` for conditioning. The requested ``intorder`` is retained so boundary loads and essential-trace projections use a true one-dimensional facet rule of the same order rather than reusing the triangular volume rule. ``internal_pec_facets`` must be a one-dimensional integer array of valid interior facets. Indices are topology-preserving under mesh rescaling and are stored on the returned system; boundary facets, noninteger arrays, and out-of-range indices raise ``ValueError``.

``assemble_load_vector``
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   assemble_load_vector(basis, source) -> complex ndarray

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

Performs the same condensed direct solve while imposing a full mixed-space coefficient vector on the PEC DOF set. Entries outside ``system.pec_dofs`` must be zero. The residual normalization includes the effective load induced by the prescribed trace, so a boundary-only scattered-field solve remains a relative rather than absolute check. ``solve_homogeneous_pec`` is the zero-data special case.

``relative_hermiticity_error``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   relative_hermiticity_error(matrix) -> float

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

Projects ``-E_inc,t`` separately on the inserted-facet traces of the Nedelec and continuous-P1 component spaces using one-dimensional facet quadrature, maps the component coefficients through scikit-fem's topology-aware composite ordering, and returns a full vector that is zero away from the inserted facets. The result depends only on tangential incident data at the plate, not on its normal component or off-facet field. Every inserted facet must be an interior member of ``system.internal_pec_facets``.

``assemble_released_pec_source``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   assemble_released_pec_source(
       system: MixedFEMSystem,
       *,
       released_pec_facets: ArrayLike,
       incident_magnetic: IncidentField,
   ) -> complex ndarray

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

Forms the combined volume/aperture equivalent source, assembles the inserted PEC trace, and calls the prescribed actual-PEC field solver. ``released_pec_facets`` must not be in the system's actual constraint set; ``inserted_pec_facets`` must be in it. In the high-level workflow the physical outgoing condition is supplied by constitutive PML tensors inside the outer PEC boundary.

Advanced monitor API
--------------------

``MonitorSamples``
~~~~~~~~~~~~~~~~~~

Fields on a sorted ``z=constant`` line:

.. code-block:: python

   MonitorSamples(x, weights, E, H, z)

``x`` and positive integration ``weights`` have shape ``(N,)``; E and H have shape ``(3,N)``; ``z`` is the physical line coordinate.

``HorizontalMonitorSamples``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fields on a sorted ``x=constant`` line:

.. code-block:: python

   HorizontalMonitorSamples(z, weights, E, H, x)

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

The corresponding physical ``x=constant`` sampler, sorted by z. It is used for transverse Poynting-flux/radiation accounting.

For both samplers, ``basis`` coordinates are interpreted as physical coordinates divided by ``length_scale``. Returned coordinates and weights are in SI units.

Advanced modal-projection API
-----------------------------

``ModalTrace``
~~~~~~~~~~~~~~

.. code-block:: python

   ModalTrace(E: complex ndarray, H: complex ndarray, label: str = "mode")

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

Returns signed real modal flux from ``Re(a.T @ G @ conj(a))``. ``indices`` can restrict the calculation to propagating families. ``normalize_diagonal=True`` removes small sampled unit-power diagonal errors by a congruence scaling and rejects a zero-power selected trace.

Advanced mode-system records
----------------------------

``Layer``
~~~~~~~~~

.. code-block:: python

   Layer(x: tuple[float, float], material: Material, name: str)

The immutable interval record returned by ``CrossSection.add_layer``.

``PECBoundary``
~~~~~~~~~~~~~~~

.. code-block:: python

   PECBoundary(x: float, name: str)

The immutable internal-sheet record returned by ``CrossSection.add_pec``. Constructing one directly is supported only as an entry in the ``CrossSection.pec_boundaries`` initializer; ``CrossSection.__post_init__`` revalidates its coordinate and name through the same public addition path.

``ModeFEMSystem``
~~~~~~~~~~~~~~~~~

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

* The scattering mesh uses first-order triangular Nedelec/P1 elements and a sparse direct solver.

* A z-PML is mandatory. Open transverse structures additionally require an x-PML; integrated PMC truncation is not implemented.

* Callback devices require explicit modes and the caller-validated physical invariants listed under ``from_material_function``.

* High-level imported meshes and arbitrary-point result evaluation are not yet exposed; lower-level mesh/FEM interfaces remain available.

* Every new structure requires mesh, monitor, mode-mesh, and PML convergence checks. A nominal PML reflection target is not a validation result.
