FDFD Mode Tracking Development
==============================

Hybrid conventional/neural mode tracking for frequency-domain port profiles,
including bound evanescent modes. The package is included in the FDFD distribution.

The port eigenproblem uses no PML. Tracking must preserve mode identity, complex
phase, and degenerate subspaces, and assess injection suitability separately.

* `Implementation plan <PLAN.md>`_: repository integration, algorithms, data
  contracts, milestones, and the proposed package layout.
* `Validation plan <VALIDATION.md>`_: numerical fixtures, no-PML artifact checks,
  neural evaluation, and FDTD acceptance tests.
* `Existing waveguide solver <../waveguide_modes/README.rst>`_
* `Implemented API and guide <../../../doc/solvers/fdfd/mode_tracking/guide.rst>`_
* `Mathematical reference <../../../doc/solvers/fdfd/mode_tracking/mathematics.rst>`_
* `Runnable examples <../../../examples/fdfd/mode_tracking/README.rst>`_

The conventional tracker includes measured residuals, common field normalization,
subspace transport, confinement verification, adaptive cutoff brackets and HDF5
persistence. An optional MLP workflow is provided with an analytical demonstrator
dataset. Exact-cutoff reconstruction and FDTD execution are not included.
