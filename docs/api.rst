API reference
=============

Everything users need is importable from :mod:`fdfd`. Modules whose names
start with an underscore are internal.

Solvers
-------

.. autoclass:: fdfd.ModeSolver1D
   :inherited-members:
   :exclude-members: cell_tensors

.. autoclass:: fdfd.ModeSolver2D
   :inherited-members:

.. autoclass:: fdfd.PeriodicModeSolver2D
   :inherited-members:

.. autoclass:: fdfd.PeriodicModeSolver3D
   :inherited-members:

.. autoclass:: fdfd.BandStructureSolver2D
   :inherited-members:

.. autofunction:: fdfd.bloch_path

.. autoclass:: fdfd.ScatteringSolver2D
   :inherited-members:

.. autoclass:: fdfd.ModeTracker1D
   :inherited-members:

.. autoclass:: fdfd.ModeTracker2D
   :inherited-members:

Results
-------

.. autoclass:: fdfd.SolveInfo

.. autoclass:: fdfd.Result

.. autoclass:: fdfd.FieldResult

.. autoclass:: fdfd.ModeResult

.. autoclass:: fdfd.ScatteringResult

.. autoclass:: fdfd.BandResult

.. autoclass:: fdfd.TrackedSweep

.. autofunction:: fdfd.load

.. autoclass:: fdfd.core.grid.GridData

Mode tracking
-------------

.. autoclass:: fdfd.PortSpec

.. autoclass:: fdfd.TrackingConfig

.. autofunction:: fdfd.tracking.track_modes

.. autofunction:: fdfd.tracking.export_subspace

.. autoclass:: fdfd.tracking.PortMode

.. autoclass:: fdfd.tracking.VerificationSpec

Materials
---------

.. autoclass:: fdfd.Material

.. autoclass:: fdfd.GoodConductor

.. autoclass:: fdfd.SurfaceImpedance

.. automodule:: fdfd.core.materials
   :members: PEC, PMC, vacuum, air, copper, aluminium, gold, silver, molybdenum, palladium, tungsten, zinc, RESISTIVITIES
   :no-index:

Shapes
------

.. automodule:: fdfd.core.shapes
   :members:
   :exclude-members: Shape

Utilities
---------

.. autofunction:: fdfd.plot_dispersion

.. autofunction:: fdfd.suggest_resolution

.. autofunction:: fdfd.suggest_pml_strength

.. autofunction:: fdfd.core.eigs.solve

Errors
------

.. automodule:: fdfd.core.errors
   :members:
