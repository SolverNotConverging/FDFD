Capabilities
============

Unsupported assignments raise :class:`~fdfd.BackendCapabilityError` when
geometry is added, not at solve time.

.. list-table::
   :header-rows: 1
   :stub-columns: 1

   * -
     - ModeSolver1D/2D
     - PeriodicModeSolver2D/3D
     - BandStructureSolver2D
     - ScatteringSolver2D
     - ModeTracker1D/2D
   * - Bulk material
     - diagonal
     - diagonal
     - scalar only
     - diagonal
     - diagonal
   * - Lossy material
     - yes
     - yes
     - yes
     - yes
     - yes
   * - PEC objects
     - yes
     - yes
     - no
     - yes
     - yes
   * - PMC objects
     - yes
     - yes
     - no
     - no
     - yes
   * - Surface impedance / metals (SIBC)
     - yes
     - no
     - no
     - no
     - yes
   * - PML
     - any side
     - x, y sides (z is periodic)
     - no (fully periodic)
     - any side
     - no (verified by padding)
   * - Outer boundary without PML
     - PMC (Yee closure)
     - 2D: PEC or PMC; 3D: PMC
     - periodic
     - PEC
     - PMC (Yee closure)
   * - Polarization
     - 1D: TE/TM/both; 2D: vector
     - 2D: TE/TM/both; 3D: vector
     - TE/TM/both
     - TE/TM
     - 1D: TE/TM/both; 2D: vector
   * - ``device="cuda"``
     - yes
     - yes
     - yes
     - yes
     - yes
   * - ``method="refined"``
     - yes
     - yes
     - yes
     - (direct solve)
     - arpack only
   * - Frequency-dispersive materials
     - one frequency per solver
     - one frequency per solver
     - no (frequency is the eigenvalue)
     - one frequency per solver
     - materials fixed over the sweep

Off-diagonal (fully anisotropic) tensors are not supported by any solver.
