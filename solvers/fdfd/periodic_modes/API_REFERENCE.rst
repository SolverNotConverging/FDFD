fdfd_periodic_modes user API
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

``PeriodicModeSolver2D``
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver2D(polarization, freq, x_range, z_range, Nx, Nz, num_modes, mode_filter=True, guess=0, tol=0, ncv=None)

2D Bloch-periodic TE/TM mode solver on a periodic Yee grid.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``polarization``
     - ``array-like or scalar``
     - Required
     - ``—``
     - Polarization control for this operation.
   * - ``freq``
     - ``array-like or scalar``
     - Required
     - ``—``
     - Freq control for this operation.
   * - ``x_range``
     - ``float / m``
     - Required
     - ``—``
     - Physical x extent in metres.
   * - ``z_range``
     - ``float / m``
     - Required
     - ``—``
     - Physical z extent in metres.
   * - ``Nx``
     - ``int``
     - Required
     - ``—``
     - Number of Yee cells along x; positive integer.
   * - ``Nz``
     - ``int``
     - Required
     - ``—``
     - Number of Yee cells along z; positive integer.
   * - ``num_modes``
     - ``int``
     - Required
     - ``—``
     - Number of modes requested; positive integer.
   * - ``mode_filter``
     - ``bool``
     - Optional
     - ``True``
     - Mode filter control for this operation.
   * - ``guess``
     - ``int``
     - Optional
     - ``0``
     - Spectral shift for the existing FDFD eigenproblem; see the solver example.
   * - ``tol``
     - ``int``
     - Optional
     - ``0``
     - Algebraic eigenproblem tolerance; zero retains the existing backend convention.
   * - ``ncv``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - Arnoldi subspace size; None selects the backend default.

Returns: a configured solver.

``PeriodicModeSolver2D.add_rectangle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver2D.add_rectangle(epsilon, mu, x_range, z_range, *, subpixels=8)

Add a subpixel-smoothed rectangular material region on the cell grid.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``epsilon``
     - ``float | complex | array-like / relative``
     - Required
     - ``—``
     - Relative permittivity; supported scalar/tensor forms are described below.
   * - ``mu``
     - ``float | complex | array-like / relative``
     - Required
     - ``—``
     - Relative permeability; supported scalar/diagonal forms are described below.
   * - ``x_range``
     - ``tuple[float, float] | tuple[int, int] | slice``
     - Required
     - ``—``
     - Range along x: floating-point positions in metres, integer grid indices, or an index slice.
   * - ``z_range``
     - ``tuple[float, float] | tuple[int, int] | slice``
     - Required
     - ``—``
     - Range along z: floating-point positions in metres, integer grid indices, or an index slice.
   * - ``subpixels``
     - ``int``
     - Optional
     - ``8``
     - Number of subcell samples used to average region material values.

Returns: the documented data or None when storing state on the solver.

``PeriodicModeSolver2D.add_pec``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver2D.add_pec(x_range, z_range, components=None)

Add an exact PEC region by constraining staggered field DOFs.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``x_range``
     - ``tuple[float, float] | tuple[int, int] | slice``
     - Required
     - ``—``
     - Range along x: floating-point positions in metres, integer grid indices, or an index slice.
   * - ``z_range``
     - ``tuple[float, float] | tuple[int, int] | slice``
     - Required
     - ``—``
     - Range along z: floating-point positions in metres, integer grid indices, or an index slice.
   * - ``components``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - Components control for this operation.

Returns: the documented data or None when storing state on the solver.

``PeriodicModeSolver2D.add_pmc``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver2D.add_pmc(x_range, z_range, components=None)

Add an exact PMC region by constraining staggered field DOFs.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``x_range``
     - ``tuple[float, float] | tuple[int, int] | slice``
     - Required
     - ``—``
     - Range along x: floating-point positions in metres, integer grid indices, or an index slice.
   * - ``z_range``
     - ``tuple[float, float] | tuple[int, int] | slice``
     - Required
     - ``—``
     - Range along z: floating-point positions in metres, integer grid indices, or an index slice.
   * - ``components``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - Components control for this operation.

Returns: the documented data or None when storing state on the solver.

``PeriodicModeSolver2D.add_pml``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver2D.add_pml(pml_width=30, n=3, sigma_max=5.0, direction='all')

Add an x-directed uniaxial PML for ``exp(+j*omega*t)`` phasors.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``pml_width``
     - ``int``
     - Optional
     - ``30``
     - PML thickness in Yee cells.
   * - ``n``
     - ``int``
     - Optional
     - ``3``
     - Polynomial PML order.
   * - ``sigma_max``
     - ``float``
     - Optional
     - ``5.0``
     - Backend PML-strength magnitude; the outgoing stretch has negative imaginary sign.
   * - ``direction``
     - ``str``
     - Optional
     - ``'all'``
     - Propagation direction for solve; selected coordinate direction for PML.

Returns: the documented data or None when storing state on the solver.

``PeriodicModeSolver2D.solve``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver2D.solve(guess=None, tol=None, ncv=None, method='eigs', max_restarts=12, random_seed=0, kernel_backend='auto')

Solve periodic modes with SciPy or refined shift-invert Arnoldi.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``guess``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - Spectral shift for the existing FDFD eigenproblem; see the solver example.
   * - ``tol``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - Algebraic eigenproblem tolerance; zero retains the existing backend convention.
   * - ``ncv``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - Arnoldi subspace size; None selects the backend default.
   * - ``method``
     - ``str``
     - Optional
     - ``'eigs'``
     - Method control for this operation.
   * - ``max_restarts``
     - ``int``
     - Optional
     - ``12``
     - Maximum refined Arnoldi restarts.
   * - ``random_seed``
     - ``int``
     - Optional
     - ``0``
     - Deterministic initial-vector seed.
   * - ``kernel_backend``
     - ``str``
     - Optional
     - ``'auto'``
     - Refined-kernel backend: auto, numpy, or cython.

Returns: the documented data or None when storing state on the solver.

``PeriodicModeSolver2D.visualize_with_gui``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver2D.visualize_with_gui()

Returns: the documented data or None when storing state on the solver.

``PeriodicModeSolver3D``
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver3D(Nx, Ny, Nz, x_range, y_range, z_range, freq, num_modes, sigma_guess=None, tol=0, ncv=None)

Full-vector Bloch-periodic solver using ``exp(+j*omega*t)`` phasors.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``Nx``
     - ``int``
     - Required
     - ``—``
     - Number of Yee cells along x; positive integer.
   * - ``Ny``
     - ``int``
     - Required
     - ``—``
     - Number of Yee cells along y; positive integer.
   * - ``Nz``
     - ``int``
     - Required
     - ``—``
     - Number of Yee cells along z; positive integer.
   * - ``x_range``
     - ``float / m``
     - Required
     - ``—``
     - Physical x extent in metres.
   * - ``y_range``
     - ``float / m``
     - Required
     - ``—``
     - Physical y extent in metres.
   * - ``z_range``
     - ``float / m``
     - Required
     - ``—``
     - Physical z extent in metres.
   * - ``freq``
     - ``array-like or scalar``
     - Required
     - ``—``
     - Freq control for this operation.
   * - ``num_modes``
     - ``int``
     - Required
     - ``—``
     - Number of modes requested; positive integer.
   * - ``sigma_guess``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - Sigma guess control for this operation.
   * - ``tol``
     - ``int``
     - Optional
     - ``0``
     - Algebraic eigenproblem tolerance; zero retains the existing backend convention.
   * - ``ncv``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - Arnoldi subspace size; None selects the backend default.

Returns: a configured solver.

``PeriodicModeSolver3D.add_block``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver3D.add_block(er, mr, x_range, y_range, z_range, *, subpixels=8)

Add a subpixel-smoothed rectangular block on the cell material grid.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``er``
     - ``array-like or scalar``
     - Required
     - ``—``
     - Er control for this operation.
   * - ``mr``
     - ``array-like or scalar``
     - Required
     - ``—``
     - Mr control for this operation.
   * - ``x_range``
     - ``tuple[float, float] | tuple[int, int] | slice``
     - Required
     - ``—``
     - Range along x: floating-point positions in metres, integer grid indices, or an index slice.
   * - ``y_range``
     - ``tuple[float, float] | tuple[int, int] | slice``
     - Required
     - ``—``
     - Range along y: floating-point positions in metres, integer grid indices, or an index slice.
   * - ``z_range``
     - ``tuple[float, float] | tuple[int, int] | slice``
     - Required
     - ``—``
     - Range along z: floating-point positions in metres, integer grid indices, or an index slice.
   * - ``subpixels``
     - ``int``
     - Optional
     - ``8``
     - Number of subcell samples used to average region material values.

Returns: the documented data or None when storing state on the solver.

``PeriodicModeSolver3D.add_pec``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver3D.add_pec(x_range, y_range, z_range, components=None)

Add an exact PEC region by constraining staggered field DOFs.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``x_range``
     - ``tuple[float, float] | tuple[int, int] | slice``
     - Required
     - ``—``
     - Range along x: floating-point positions in metres, integer grid indices, or an index slice.
   * - ``y_range``
     - ``tuple[float, float] | tuple[int, int] | slice``
     - Required
     - ``—``
     - Range along y: floating-point positions in metres, integer grid indices, or an index slice.
   * - ``z_range``
     - ``tuple[float, float] | tuple[int, int] | slice``
     - Required
     - ``—``
     - Range along z: floating-point positions in metres, integer grid indices, or an index slice.
   * - ``components``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - Components control for this operation.

Returns: the documented data or None when storing state on the solver.

``PeriodicModeSolver3D.add_pmc``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver3D.add_pmc(x_range, y_range, z_range, components=None)

Add an exact PMC region by constraining staggered field DOFs.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``x_range``
     - ``tuple[float, float] | tuple[int, int] | slice``
     - Required
     - ``—``
     - Range along x: floating-point positions in metres, integer grid indices, or an index slice.
   * - ``y_range``
     - ``tuple[float, float] | tuple[int, int] | slice``
     - Required
     - ``—``
     - Range along y: floating-point positions in metres, integer grid indices, or an index slice.
   * - ``z_range``
     - ``tuple[float, float] | tuple[int, int] | slice``
     - Required
     - ``—``
     - Range along z: floating-point positions in metres, integer grid indices, or an index slice.
   * - ``components``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - Components control for this operation.

Returns: the documented data or None when storing state on the solver.

``PeriodicModeSolver3D.add_UPML``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver3D.add_UPML(sides=('-x', '+x', '-y', '+y'), width=10, max_loss=5, n=3)

Add a transverse UPML for the ``exp(+j*omega*t)`` convention.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``sides``
     - ``tuple``
     - Optional
     - ``('-x', '+x', '-y', '+y')``
     - Sides control for this operation.
   * - ``width``
     - ``int``
     - Optional
     - ``10``
     - PML thickness in Yee cells.
   * - ``max_loss``
     - ``int``
     - Optional
     - ``5``
     - Maximum PML stretch loss magnitude.
   * - ``n``
     - ``int``
     - Optional
     - ``3``
     - Polynomial PML order.

Returns: the documented data or None when storing state on the solver.

``PeriodicModeSolver3D.solve``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver3D.solve(method='refined', sigma_guess=None, tol=None, ncv=None, max_restarts=12, random_seed=0, kernel_backend='auto')

Solve periodic modes.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``method``
     - ``str``
     - Optional
     - ``'refined'``
     - Method control for this operation.
   * - ``sigma_guess``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - Sigma guess control for this operation.
   * - ``tol``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - Algebraic eigenproblem tolerance; zero retains the existing backend convention.
   * - ``ncv``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - Arnoldi subspace size; None selects the backend default.
   * - ``max_restarts``
     - ``int``
     - Optional
     - ``12``
     - Maximum refined Arnoldi restarts.
   * - ``random_seed``
     - ``int``
     - Optional
     - ``0``
     - Deterministic initial-vector seed.
   * - ``kernel_backend``
     - ``str``
     - Optional
     - ``'auto'``
     - Refined-kernel backend: auto, numpy, or cython.

Returns: the documented data or None when storing state on the solver.

``PeriodicModeSolver3D.plot``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver3D.plot(mode=0, x=None, y=None, z=None, *, save=None, show=True)

Plot ``|Ex|``, ``|Ey|``, ``|Hx|``, ``|Hy|`` for a 2D slice of the 3D field.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``mode``
     - ``int``
     - Optional
     - ``0``
     - Visualization mode index; waveguide visualize uses 1-based selection, 3D periodic plot uses 0-based selection.
   * - ``x``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - X control for this operation.
   * - ``y``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - Y control for this operation.
   * - ``z``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - Z control for this operation.
   * - ``save``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - Save control for this operation.
   * - ``show``
     - ``bool``
     - Optional
     - ``True``
     - Show control for this operation.

Returns: the documented data or None when storing state on the solver.

``PeriodicModeSolver3D.plot_field_plane``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver3D.plot_field_plane(axis, index, mode_index=0, field='Ex')

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``axis``
     - ``array-like or scalar``
     - Required
     - ``—``
     - Axis control for this operation.
   * - ``index``
     - ``array-like or scalar``
     - Required
     - ``—``
     - Index control for this operation.
   * - ``mode_index``
     - ``int``
     - Optional
     - ``0``
     - Zero-based stored periodic mode index.
   * - ``field``
     - ``str``
     - Optional
     - ``'Ex'``
     - Field control for this operation.

Returns: the documented data or None when storing state on the solver.

``PeriodicModeSolver3D.visualize_with_gui``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver3D.visualize_with_gui()

Returns: the documented data or None when storing state on the solver.

``PeriodicModeSolver3D.save_results``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver3D.save_results(path, include_eigenvectors=False, compressed=True)

Save all calculated results + inputs to a single NPZ file.

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
   * - ``include_eigenvectors``
     - ``bool``
     - Optional
     - ``False``
     - Include large raw eigenvector arrays in the NPZ archive.
   * - ``compressed``
     - ``bool``
     - Optional
     - ``True``
     - Compress NPZ datasets.

Returns: the documented data or None when storing state on the solver.

``PeriodicModeSolver3D.load_results``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    PeriodicModeSolver3D.load_results(path)

Recreate a solver instance from a saved NPZ (from save_results). Returns a fully populated solver ready for plotting.

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

Returns: a stored periodic solver result for inspection.

Results and examples
--------------------

Solves store effective indices and field arrays on the solver. ``neff`` is
dimensionless; attenuation follows -Im(neff), multiplied by the free-space
wavenumber for inverse metres. The 1D waveguide implementation separates
TE and TM arrays. Consult the bundled example for the corresponding viewer.

Invalid dimensions, materials, and solver controls raise ValueError or
NotImplementedError. Numerical backend failures remain visible.

Run the examples with the installed package; no repository path changes
are required. See `README.rst <README.rst>`_ and the ``examples/`` directory.
Assembly routines, matrix builders, and Arnoldi kernels are implementation
details and are excluded from this reference.
