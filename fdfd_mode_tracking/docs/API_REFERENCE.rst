fdfd_mode_tracking API
==========================

This package is included in the single FDFD distribution. The
`guide <guide.rst>`_ defines conventions, verification requirements and limitations.

Material-first sweep solvers
----------------------------

``ModeTracker1D``
~~~~~~~~~~~~~~~~~

.. code-block:: python

   ModeTracker1D(*, frequencies, x_range,
                 background_material=materials.vacuum, port=PortSpec())

``frequencies`` is a unique positive sequence in hertz; ``x_range`` is a metre
extent or increasing pair. ``background_material`` is a bulk material and
``port`` controls confinement and direction.

``ModeTracker1D.add_geometry`` accepts keyword-only ``shape``, ``material``,
``name`` and ``clip``. ``ModeTracker1D.add_layer`` accepts ``x_range``,
``material``, ``name`` and ``clip``. ``ModeTracker1D.set_material`` accepts
``geometry`` and ``material``; ``ModeTracker1D.set_shape`` accepts ``geometry``
and ``shape``; ``ModeTracker1D.remove`` accepts ``geometry``.

``ModeTracker1D.mesh`` accepts keyword-only ``resolution``, ``max_element_size``
and ``subpixels``. It defines the fixed base Yee grid used at every requested
frequency and returns ``GridData``.

``ModeTracker1D.solve`` accepts keyword-only ``num_modes``, ``neff_guess``,
``polarization``, ``eigensolver_tolerance``,
``reference_frequency``, ``tracking_config`` and ``progress``. It returns a
``TrackedSweep``. ``ModeTracker1D.show`` accepts keyword-only ``component``,
``quantity`` and ``block`` and opens the completed sweep viewer.

``num_modes`` controls the returned count at each frequency; all usable modes
are tracked, including branches first returned later. ``neff_guess=None`` uses
the highest material square-root index (by magnitude), including background and
geometry. Diagonal media use all principal epsilon/mu pair products. An explicit
``neff_guess`` overrides this default. Internal candidate expansion is disabled
for the material-first API, even if ``tracking_config.max_candidates`` is larger.

``ModeTracker2D``
~~~~~~~~~~~~~~~~~

.. code-block:: python

   ModeTracker2D(*, frequencies, x_range, y_range,
                 background_material=materials.vacuum, port=PortSpec())

Constructor parameters have the same meaning as the 1D tracker, with ``y_range``
adding the second metre extent. ``ModeTracker2D.add_geometry`` accepts ``shape``,
``material``, ``name`` and ``clip``. Convenience methods are
``ModeTracker2D.add_rectangle`` with ``x_range``, ``y_range``, ``material``,
``name`` and ``clip``; ``ModeTracker2D.add_circle`` with ``center``, ``radius``,
``material``, ``name`` and ``clip``; and ``ModeTracker2D.add_polygon`` with
``points``, ``material``, ``name`` and ``clip``.

``ModeTracker2D.set_material`` accepts ``geometry`` and ``material``;
``ModeTracker2D.set_shape`` accepts ``geometry`` and ``shape``;
``ModeTracker2D.remove`` accepts ``geometry``. ``ModeTracker2D.mesh`` accepts
``resolution``, ``max_element_size`` and ``subpixels``. ``ModeTracker2D.solve``
accepts ``num_modes``, ``neff_guess``, ``polarization``,
``eigensolver_tolerance``,
``reference_frequency``, ``tracking_config`` and ``progress``.
``ModeTracker2D.show`` accepts ``component``, ``quantity`` and ``block``.

Configuration
-------------

``PortSpec`` has keyword-only parameters ``boundary='open'`` (``open`` or
``enclosed``), ``direction=1`` (+1/-1 along z), ``reference_plane=0.0`` (metres)
and ``name='port'``. These parameter names are ``boundary``, ``direction``,
``reference_plane`` and ``name``.

``VerificationSpec`` is passed to the scene factory with ``mesh_factor`` (1 or 2)
and ``padding_fraction`` (0, .25 or .5). It describes a verification request;
the factory must preserve the physical structure while applying it.

``TrackingConfig`` exposes these keyword-only controls:

* Candidate search: ``num_candidates`` (6), ``max_candidates`` (12),
  ``neff_guess`` (None), ``polarization`` (``both``; also TE/TM in 1D),
  ``eigensolver_tolerance`` (1e-10).
* Numerical and cutoff checks: ``residual_tolerance`` (1e-8),
  ``cutoff_neff`` (1e-6).
* Assignment: ``overlap_min`` (.8), ``assignment_margin`` (.02),
  ``unmatched_cost`` (.65), ``cluster_gap`` (1e-5).
* Verification: ``verification_beta_tolerance`` (1e-3),
  ``verification_overlap`` (.999), ``edge_fraction_max`` (1e-3).
* Work limits: ``max_depth`` (8), ``max_solves`` (200),
  ``min_relative_step`` (1e-5).

Tracking
--------

``track_modes`` has the signature:

.. code-block:: python

   track_modes(make_solver, frequencies, *, port=None, config=None,
               seed_modes=(0,), reference_frequency=None, progress=True)

``make_solver(frequency_hz, verification_spec)`` returns a newly configured,
meshed waveguide solver. ``frequencies`` are positive, finite, unique hertz.
``seed_modes`` are candidate indices at ``reference_frequency`` (the lowest
frequency by default). Proven bound cutoff candidates are valid identity seeds
and continue into higher-frequency samples. Exact degenerate clusters expand the
seed set. Matching uses complex field overlaps, eigenvalue prediction and
one-to-one assignment with unmatched states; no model or training is required.

Passing ``seed_modes=None`` enables automatic tracking of the complete solved
set, births and reacquisition. With this choice, a missing ``neff_guess`` is
computed from each factory scene's materials and candidate expansion is disabled.
Explicit seed tuples retain the targeted factory workflow.

``progress=True`` displays a tqdm frequency counter for both APIs. Each distinct
frequency advances it once after its candidate solve and verification attempts.
Adaptive frequencies grow the total. The status shows frequency and primary,
mesh-refinement or padding stage. Cache hits and candidate-expansion retries do
not count a frequency twice. Completed evaluations can still contain invalid or
unverified modes; this is a work counter, not an eligibility counter. Failed
primary solves leave the counter incomplete. ``progress=False`` hides it, and
the bar closes on normal completion, errors and interruption.

The return type ``TrackedSweep`` records ``port``, ``config``,
``requested_frequencies``, ``samples``, ``events``, ``unresolved_intervals`` and
``metadata``. Constructed records use keyword-only arguments. Each sample has
``frequency``, ``candidates``, ``candidate_indices``, ``phases``, ``overlaps`` and
``clusters``. Negative candidate indices indicate absent/unmatched tracks.
Automatic sweeps keep equally sized global track arrays at every frequency,
including -1 entries before a branch is first observed. Metadata records
``automatic_tracking`` and ``num_tracks``; ``branch_discovered`` events identify
later track origins. Solver metadata records the actual ``neff_guess`` used.

Persistence and export
----------------------

``TrackedSweep.save`` accepts ``path`` and writes a versioned HDF5 archive
atomically. ``load_sweep`` accepts ``path`` and returns its data without invoking the solver.
New archives use schema 1.1. Schema 1.0 sweeps remain readable; their retired
scoring-weight field is ignored and historical events are preserved as data.

``TrackedSweep.export`` accepts ``track=0``, keyword-only ``frequencies=None``
and ``amplitude=1.0``. These parameter names are ``track``, ``frequencies`` and
``amplitude``. The default selects the originally requested frequencies.
It returns a tuple of ``PortMode`` records with ``frequency``, ``beta``,
``fields``, ``field_coordinates``, ``complex_power`` and ``metadata``.
Profiles are discrete samples, not an interpolated spectrum. Invalid samples,
unmatched branches, uncertain identity and individual cluster exports raise
``ValueError``. An unresolved cutoff interval is retained in export metadata.

``TrackedSweep.plot`` accepts keyword-only ``component='E'`` and
``quantity='magnitude'`` and returns the interactive Matplotlib Figure without
showing it. ``TrackedSweep.show`` accepts ``component``, ``quantity`` and
``block=True``, opens the same GUI, and returns the Figure. Components are
``E``, ``H``, ``Ex``, ``Ey``, ``Ez``, ``Hx``, ``Hy`` or ``Hz``; quantities are
``magnitude``, ``real``, ``imag`` or ``phase``.

Dispersion segments are solid when both endpoints are injection eligible and
dashed for confidently tracked but ineligible branches. The latter retain
their x markers and cannot be exported. Missing/numerically invalid samples
and unresolved identity intervals break connections; reliable exact-cutoff
identities retain spectral-only connections. Loading an existing archive uses
these display rules without recomputing the sweep.

``export_subspace`` accepts ``sweep``, ``frequency``, ``cluster_index`` and
``coefficients`` and returns an
explicit eigenmode superposition as a tuple of ``PortMode`` terms. Coefficients
refer to the sample's stored smooth orthonormal basis. Every term retains its
own propagation constant; an arbitrary near-degenerate mixture is not exported
as a single mode.

Archive errors raise ``fdfd_common.errors.PersistenceError``. Invalid factory
contracts and configuration raise ``ValueError``. Solver errors are propagated
except for explicitly handled convergence failures and work-budget exhaustion,
which produce unresolved events after a valid reference seed exists.
