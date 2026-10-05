Tracked port mode examples
==========================

Install the required Python packages from the `root README <../../README.md>`_.
The scripts run directly from top to bottom and import shared materials from
the solver package. Installing FDFD itself is optional. The ``tracked_*`` examples and
the degeneracy/crossing examples open an interactive Matplotlib viewer.

* ``1d_parallel_plate_cutoff.py`` tracks a physical PEC guide above and below
  cutoff, saves a complete HDF5 sweep, and exports a discrete port-profile table.
* ``1d_tracked_parallel_plate.py`` uses the material-first ``ModeTracker1D`` API,
  crosses cutoff, saves the sweep, and opens the interactive all-mode viewer.
* ``2d_tracked_dielectric_waveguide.py`` uses ``ModeTracker2D`` for an open
  dielectric guide and displays all returned vector modes at each frequency.
* ``2d_tracked_coplanar_waveguide.py`` sweeps an open, finite-board CPW
  from 6 to 30 GHz. A 1.2 mm centre strip and two 0.6 mm slots sit on a 1.2 mm,
  epsilon-r = 4 substrate, 12 mm wide. There is no backing ground or housing.
  Vacuum extends 4 mm beyond the board/metal bounding box on every side,
  giving a 20 by 9.4 mm domain with a 100 by 47 grid. Six candidates per
  frequency are tracked. No PML is used: confinement requires mesh refinement
  and two exterior-domain enlargements with the same physical geometry.
  Edit ``frequencies``, ``air_padding``, and ``cell_size`` near the top of the
  script to change the sweep or refine the grid. Inspect the saved candidate
  evidence when assessing confinement.
* ``2d_degenerate_square_waveguide.py`` follows the two-dimensional
  TE10/TE01 eigenspace of a square PEC guide. The GUI marks both members as a
  degenerate subspace instead of assigning physical meaning to an arbitrary
  eigensolver basis rotation.
* ``1d_anisotropic_pec_mode_crossing.py`` uses a diagonal anisotropic dielectric to
  create a true TE1/TM1 crossing. The distinct polarizations remain separate
  tracked branches as their propagation constants exchange order.

To run without opening a viewer, omit the ``sweep.show(...)`` line.

The interactive viewer shows tracked dispersion curves and every candidate
returned by each primary/adaptive solve. Set just ``num_modes`` to control the
count at every frequency; all usable candidates are automatically tracked.
The default index guess comes from the materials. New branches receive track
IDs as they appear. Confidently tracked but injection-ineligible modes have
dashed connections and keep their red x markers. Connections are solid only
when both endpoints are eligible. Absent/numerically invalid samples and
unresolved identity intervals break lines; reliable exact-cutoff identities
retain spectral connections without becoming exportable.
Click either dispersion panel or move
the solved-frequency slider to update the field grid. The component and quantity
selectors update all mode panels together. Non-bound and spurious modes carry a
red ``x``; degenerate modes use diamonds and cutoff-bracket modes use stars. A
proven bound exact-cutoff candidate keeps the star without an ``x`` so the branch
remains visible, although its singular exact-cutoff profile cannot be exported.

Generated artifacts live under ``fdfd_mode_tracking/outputs/examples/``.
See the `tracking guide <../docs/guide.rst>`_.
