Tracked port mode examples
==========================

Install the checkout into the active environment before running these examples.
None of these examples requires an FDTD engine. The ``tracked_*`` examples and
the degeneracy/crossing examples open an interactive Matplotlib viewer.

* ``parallel_plate_cutoff.py`` tracks a physical PEC guide above and below
  cutoff, saves a complete HDF5 sweep, and exports a discrete port-profile table.
* ``tracked_parallel_plate_1d.py`` uses the material-first ``ModeTracker1D`` API,
  crosses cutoff, saves the sweep, and opens the interactive all-mode viewer.
* ``tracked_dielectric_waveguide_2d.py`` uses ``ModeTracker2D`` for an open
  dielectric guide and displays all returned vector modes at each frequency.
* ``tracked_coplanar_waveguide_2d.py`` sweeps an open, finite-board CPW
  from 6 to 30 GHz. A 1.2 mm centre strip and two 0.6 mm slots sit on a 1.2 mm,
  epsilon-r = 4 substrate, 12 mm wide. There is no backing ground or housing.
  Vacuum extends 8 mm beyond the complete board/metal bounding box in each of
  the four directions (+/-x and +/-y), giving a 28 by 17.4 mm domain. The
  0.1 mm mesh (280 by 174 cells) resolves each slot with six cells and the
  0.2 mm metal thickness with two. Six candidates per frequency are tracked.
  No PML is used: confinement requires mesh refinement AND two exterior-domain
  enlargements with the same physical geometry. These checks reach 560 by
  348 cells and make this example substantially more expensive than before.
  Padding is not a guarantee of confinement, especially for weakly bound
  branches. Failed checks remain visible; inspect saved candidate evidence
  before interpreting an ``x`` as radiation rather than mesh uncertainty.
  Run normally for the frequency-selectable field viewer, or call
  ``main(show=False)`` headlessly. ``air_padding`` and ``cell_size`` can be
  changed on ``build_tracker`` or ``main`` for further convergence studies;
  ``frequencies`` can restrict a costly run to selected samples.
* ``degenerate_square_waveguide_2d.py`` follows the two-dimensional
  TE10/TE01 eigenspace of a square PEC guide. The GUI marks both members as a
  degenerate subspace instead of assigning physical meaning to an arbitrary
  eigensolver basis rotation.
* ``anisotropic_mode_crossing_1d.py`` uses a diagonal anisotropic dielectric to
  create a true TE1/TM1 crossing. The distinct polarizations remain separate
  tracked branches as their propagation constants exchange order.

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

Generated artifacts live under ``outputs/fdfd_mode_tracking/examples/``.
See the `tracking guide <../docs/guide.rst>`_.
