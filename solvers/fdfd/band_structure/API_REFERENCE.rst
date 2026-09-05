fdfd_band_structure user API
============================

Version 1.0.0. These FDFD implementations retain their existing numerical
workflow. The uniform mesh/solve/show contract applies to the FEM families.
Phasors use exp(+i omega t); passive relative materials have nonpositive
imaginary values.

Configuration and units
-----------------------

Constructor extents and frequencies use metres and hertz. Nx/Ny/Nz are
Yee cell counts. Geometry range helpers distinguish integer grid-index bounds
from floating-point physical positions in metres; slices select grid indices.
Band-structure shapes use physical coordinates. Materials are relative
diagonal values. Mode normalization, field locations, and existing selectors
are preserved by this release.

``BandStructureSolver2D``
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    BandStructureSolver2D(a: 'float', Nx: 'int', Ny: 'int | None' = None, *, b: 'float | None' = None, background_er: 'float' = 1.0, background_ur: 'float' = 1.0, boundary_conditions: 'tuple[int, int]' = (1, 1)) -> 'None'

Finite-difference frequency-domain band diagram solver.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``a``
     - ``float``
     - Required
     - ``—``
     - Unit-cell x period in metres.
   * - ``Nx``
     - ``int``
     - Required
     - ``—``
     - Number of Yee cells along x; positive integer.
   * - ``Ny``
     - ``int | None``
     - Optional
     - ``None``
     - Number of Yee cells along y; positive integer.
   * - ``b``
     - ``float | None``
     - Optional
     - ``None``
     - Unit-cell y period in metres; None uses a.
   * - ``background_er``
     - ``float``
     - Optional
     - ``1.0``
     - Background er control for this operation.
   * - ``background_ur``
     - ``float``
     - Optional
     - ``1.0``
     - Background ur control for this operation.
   * - ``boundary_conditions``
     - ``tuple[int, int]``
     - Optional
     - ``(1, 1)``
     - Boundary conditions control for this operation.

Returns: a configured solver.

``BandStructureSolver2D.add_object``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    BandStructureSolver2D.add_object(mask: 'MaskLike', *, er: 'complex | float | np.ndarray | None' = None, ur: 'complex | float | np.ndarray | None' = None) -> 'None'

Insert an object into the unit cell.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``mask``
     - ``MaskLike``
     - Required
     - ``—``
     - Mask control for this operation.
   * - ``er``
     - ``complex | float | np.ndarray | None``
     - Optional
     - ``None``
     - Er control for this operation.
   * - ``ur``
     - ``complex | float | np.ndarray | None``
     - Optional
     - ``None``
     - Ur control for this operation.

Returns: the documented data or None when storing state on the solver.

``BandStructureSolver2D.add_circular_inclusion``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    BandStructureSolver2D.add_circular_inclusion(radius: 'float', *, center: 'tuple[float, float]' = (0.0, 0.0), er: 'complex | float | None' = None, ur: 'complex | float | None' = None) -> 'None'

Convenience wrapper that inserts a circular inclusion.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``radius``
     - ``float``
     - Required
     - ``—``
     - Positive radius in metres.
   * - ``center``
     - ``tuple[float, float]``
     - Optional
     - ``(0.0, 0.0)``
     - Physical centre coordinates in metres.
   * - ``er``
     - ``complex | float | None``
     - Optional
     - ``None``
     - Er control for this operation.
   * - ``ur``
     - ``complex | float | None``
     - Optional
     - ``None``
     - Ur control for this operation.

Returns: the documented data or None when storing state on the solver.

``BandStructureSolver2D.default_rectangular_lattice_path``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    BandStructureSolver2D.default_rectangular_lattice_path() -> 'tuple[list[np.ndarray], list[str]]'

Return the Γ–X–M–Y–Γ path for a rectangular lattice.

Returns: the documented data or None when storing state on the solver.

``BandStructureSolver2D.generate_bloch_path``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    BandStructureSolver2D.generate_bloch_path(symmetry_points: 'Sequence[Sequence[float]]', total_points: 'int') -> 'tuple[np.ndarray, list[int]]'

Sample a polyline connecting the supplied symmetry points.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``symmetry_points``
     - ``Sequence[Sequence[float]]``
     - Required
     - ``—``
     - Symmetry points control for this operation.
   * - ``total_points``
     - ``int``
     - Required
     - ``—``
     - Total points control for this operation.

Returns: the documented data or None when storing state on the solver.

``BandStructureSolver2D.compute_band_structure``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    BandStructureSolver2D.compute_band_structure(beta_path: 'np.ndarray', *, num_bands: 'int', polarisations: 'Iterable[str]' = ('TE', 'TM'), eig_sigma: 'float' = 0.0) -> 'BandStructureResult'

Solve for the requested polarisations along ``beta_path``.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``beta_path``
     - ``np.ndarray``
     - Required
     - ``—``
     - Bloch vectors with shape (2, samples), in radians per metre.
   * - ``num_bands``
     - ``int``
     - Required
     - ``—``
     - Positive number of frequency bands to compute.
   * - ``polarisations``
     - ``Iterable[str]``
     - Optional
     - ``('TE', 'TM')``
     - Requested TE/TM polarization names.
   * - ``eig_sigma``
     - ``float``
     - Optional
     - ``0.0``
     - Frequency eigenproblem shift.

Returns: a BandStructureResult with frequencies and eigenvalues by polarization.

``BandStructureSolver2D.set_tick_labels``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    BandStructureSolver2D.set_tick_labels(labels: 'Sequence[str]', positions: 'Sequence[int]') -> 'None'

Attach labels to the symmetry points in the Brillouin zone path.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``labels``
     - ``Sequence[str]``
     - Required
     - ``—``
     - Labels control for this operation.
   * - ``positions``
     - ``Sequence[int]``
     - Required
     - ``—``
     - Positions control for this operation.

Returns: the documented data or None when storing state on the solver.

``BandStructureSolver2D.plot_band_diagram``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    BandStructureSolver2D.plot_band_diagram(result: 'BandStructureResult', *, wnmax: 'float | None' = None, path_artist_kwargs: 'dict[str, Any] | None' = None) -> 'tuple[Figure, tuple[Axes, Axes, Axes]]'

Create the unit-cell, Bloch-path and band-diagram figure.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``result``
     - ``BandStructureResult``
     - Required
     - ``—``
     - Result control for this operation.
   * - ``wnmax``
     - ``float | None``
     - Optional
     - ``None``
     - Wnmax control for this operation.
   * - ``path_artist_kwargs``
     - ``dict[str, Any] | None``
     - Optional
     - ``None``
     - Path artist kwargs control for this operation.

Returns: the documented data or None when storing state on the solver.

Results and examples
--------------------

``compute_band_structure`` returns frequency arrays in Hz and eigenvalues,
indexed by TE/TM polarization. Use ``plot_band_diagram`` to display them.

Invalid dimensions, materials, and solver controls raise ValueError or
NotImplementedError. Numerical backend failures remain visible.

Run the examples with the installed package; no repository path changes
are required. See `README.rst <README.rst>`_ and the ``examples/`` directory.
Assembly routines, matrix builders, and Arnoldi kernels are implementation
details and are excluded from this reference.
