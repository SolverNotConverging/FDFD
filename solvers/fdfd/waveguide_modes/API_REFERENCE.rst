fdfd_waveguide_modes user API
=============================

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

``ModeSolver1D``
~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver1D(frequency, x_range, Nx, num_modes, guess=None)

1D Yee-grid mode solver using ``exp(+j*omega*t - j*beta*z)``.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``frequency``
     - ``array-like or scalar``
     - Required
     - ``—``
     - Operating frequency in hertz; finite and positive.
   * - ``x_range``
     - ``float / m``
     - Required
     - ``—``
     - Physical x extent in metres.
   * - ``Nx``
     - ``int``
     - Required
     - ``—``
     - Number of Yee cells along x; positive integer.
   * - ``num_modes``
     - ``int``
     - Required
     - ``—``
     - Number of modes requested; positive integer.
   * - ``guess``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - Spectral shift for the existing FDFD eigenproblem; see the solver example.

Returns: a configured solver.

``ModeSolver1D.add_layer``
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver1D.add_layer(epsilon, mu, x_range, *, subpixels=100)

Add a subpixel-smoothed isotropic or diagonal-anisotropic material layer.

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
   * - ``subpixels``
     - ``int``
     - Optional
     - ``100``
     - Number of subcell samples used to average region material values.

Returns: the documented data or None when storing state on the solver.

``ModeSolver1D.add_pec``
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver1D.add_pec(x_range, components=None)

Add a PEC cell region and expand it onto surrounding electric components.

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
   * - ``components``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - Components control for this operation.

Returns: the documented data or None when storing state on the solver.

``ModeSolver1D.add_pmc``
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver1D.add_pmc(x_range, components=None)

Add a PMC cell region and expand it onto surrounding magnetic components.

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
   * - ``components``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - Components control for this operation.

Returns: the documented data or None when storing state on the solver.

``ModeSolver1D.add_pml``
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver1D.add_pml(pml_width=50, n=3, sigma_max=25, direction='all')

Add a uniaxial PML for the ``exp(+j*omega*t)`` convention.

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
     - ``50``
     - PML thickness in Yee cells.
   * - ``n``
     - ``int``
     - Optional
     - ``3``
     - Polynomial PML order.
   * - ``sigma_max``
     - ``int``
     - Optional
     - ``25``
     - Backend PML-strength magnitude; the outgoing stretch has negative imaginary sign.
   * - ``direction``
     - ``str``
     - Optional
     - ``'all'``
     - Propagation direction for solve; selected coordinate direction for PML.

Returns: the documented data or None when storing state on the solver.

``ModeSolver1D.add_impedance_surface``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver1D.add_impedance_surface(Zs: complex | None = None, *, preset: str | None = None, x_range)

Mark opaque cells whose exposed interfaces obey a scalar SIBC.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``Zs``
     - ``complex | None``
     - Optional
     - ``None``
     - Surface impedance in ohms; alternatively select a metal preset.
   * - ``preset``
     - ``str | None``
     - Optional
     - ``None``
     - Metal name for the good-conductor impedance model.
   * - ``x_range``
     - ``tuple[float, float] | tuple[int, int] | slice``
     - Required
     - ``—``
     - Range along x: floating-point positions in metres, integer grid indices, or an index slice.

Returns: the documented data or None when storing state on the solver.

``ModeSolver1D.solve``
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver1D.solve(sigma=None)

Solve TE and TM slab modes and recover staggered field components.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``sigma``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - Optional spectral shift overriding the stored eigenproblem target.

Returns: the documented data or None when storing state on the solver.

``ModeSolver1D.visualize``
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver1D.visualize(mode=1, **kwargs)

Visualize selected field components for a given one-based mode index.

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
     - ``1``
     - Visualization mode index; waveguide visualize uses 1-based selection, 3D periodic plot uses 0-based selection.
   * - ``kwargs``
     - ``array-like or scalar``
     - Required
     - ``—``
     - Kwargs control for this operation.

Returns: the documented data or None when storing state on the solver.

``ModeSolver1D.visualize_with_gui``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver1D.visualize_with_gui()

Launch an interactive Tk GUI to inspect mode profiles.

Returns: the documented data or None when storing state on the solver.

``ModeSolver2D``
~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver2D(frequency, x_range, y_range, Nx, Ny, num_modes, guess=None)

2D Yee-grid mode solver using ``exp(+j*omega*t - j*beta*z)``.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``frequency``
     - ``array-like or scalar``
     - Required
     - ``—``
     - Operating frequency in hertz; finite and positive.
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
   * - ``num_modes``
     - ``int``
     - Required
     - ``—``
     - Number of modes requested; positive integer.
   * - ``guess``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - Spectral shift for the existing FDFD eigenproblem; see the solver example.

Returns: a configured solver.

``ModeSolver2D.add_rectangle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver2D.add_rectangle(epsilon, mu, x_range, y_range, *, subpixels=8)

Add a rectangular isotropic or diagonal-anisotropic material region on the cell grid.

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
   * - ``y_range``
     - ``tuple[float, float] | tuple[int, int] | slice``
     - Required
     - ``—``
     - Range along y: floating-point positions in metres, integer grid indices, or an index slice.
   * - ``subpixels``
     - ``int``
     - Optional
     - ``8``
     - Number of subcell samples used to average region material values.

Returns: the documented data or None when storing state on the solver.

``ModeSolver2D.add_circle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver2D.add_circle(epsilon, mu, center, r1, r2=None, *, subpixels=8)

Add a subpixel-smoothed circular or annular material region.

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
   * - ``center``
     - ``tuple[float, ...] / m``
     - Required
     - ``—``
     - Physical centre coordinates in metres.
   * - ``r1``
     - ``array-like or scalar``
     - Required
     - ``—``
     - R1 control for this operation.
   * - ``r2``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - R2 control for this operation.
   * - ``subpixels``
     - ``int``
     - Optional
     - ``8``
     - Number of subcell samples used to average region material values.

Returns: the documented data or None when storing state on the solver.

``ModeSolver2D.add_triangle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver2D.add_triangle(epsilon, mu, p1, p2, p3, *, subpixels=8)

Add a subpixel-smoothed triangular material region.

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
   * - ``p1``
     - ``array-like or scalar``
     - Required
     - ``—``
     - P1 control for this operation.
   * - ``p2``
     - ``array-like or scalar``
     - Required
     - ``—``
     - P2 control for this operation.
   * - ``p3``
     - ``array-like or scalar``
     - Required
     - ``—``
     - P3 control for this operation.
   * - ``subpixels``
     - ``int``
     - Optional
     - ``8``
     - Number of subcell samples used to average region material values.

Returns: the documented data or None when storing state on the solver.

``ModeSolver2D.add_pec``
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver2D.add_pec(x_range, y_range, components=None)

Add a PEC cell region and expand it onto surrounding Yee electric components.

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
   * - ``components``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - Components control for this operation.

Returns: the documented data or None when storing state on the solver.

``ModeSolver2D.add_pmc``
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver2D.add_pmc(x_range, y_range, components=None)

Add a PMC cell region and expand it onto surrounding Yee magnetic components.

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
   * - ``components``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - Components control for this operation.

Returns: the documented data or None when storing state on the solver.

``ModeSolver2D.add_pml``
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver2D.add_pml(pml_width=50, n=3, sigma_max=5, direction='all')

Add a uniaxial PML for the ``exp(+j*omega*t)`` convention.

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
     - ``50``
     - PML thickness in Yee cells.
   * - ``n``
     - ``int``
     - Optional
     - ``3``
     - Polynomial PML order.
   * - ``sigma_max``
     - ``int``
     - Optional
     - ``5``
     - Backend PML-strength magnitude; the outgoing stretch has negative imaginary sign.
   * - ``direction``
     - ``str``
     - Optional
     - ``'all'``
     - Propagation direction for solve; selected coordinate direction for PML.

Returns: the documented data or None when storing state on the solver.

``ModeSolver2D.add_impedance_surface``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver2D.add_impedance_surface(Zs: complex | None = None, *, preset: str | None = None, x_range, y_range)

Mark an opaque cell region whose exposed faces obey a scalar SIBC.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``Zs``
     - ``complex | None``
     - Optional
     - ``None``
     - Surface impedance in ohms; alternatively select a metal preset.
   * - ``preset``
     - ``str | None``
     - Optional
     - ``None``
     - Metal name for the good-conductor impedance model.
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

Returns: the documented data or None when storing state on the solver.

``ModeSolver2D.solve``
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver2D.solve(sigma=None)

Solve for transverse modes and recover all six staggered field components.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``sigma``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - Optional spectral shift overriding the stored eigenproblem target.

Returns: the documented data or None when storing state on the solver.

``ModeSolver2D.visualize``
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver2D.visualize(mode=1, **kwargs)

Visualize selected field components for a given one-based mode index.

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
     - ``1``
     - Visualization mode index; waveguide visualize uses 1-based selection, 3D periodic plot uses 0-based selection.
   * - ``kwargs``
     - ``array-like or scalar``
     - Required
     - ``—``
     - Kwargs control for this operation.

Returns: the documented data or None when storing state on the solver.

``ModeSolver2D.visualize_with_gui``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ModeSolver2D.visualize_with_gui()

Visualize field components with a dropdown menu for mode selection.

Returns: the documented data or None when storing state on the solver.

``good_conductor_surface_impedance``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    good_conductor_surface_impedance(metal: str, frequency_hz: numbers.Real, *, relative_permeability: numbers.Real = 1.0) -> complex

Return the good-conductor surface impedance at one frequency.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``metal``
     - ``str``
     - Required
     - ``—``
     - Metal control for this operation.
   * - ``frequency_hz``
     - ``numbers.Real``
     - Required
     - ``—``
     - Frequency hz control for this operation.
   * - ``relative_permeability``
     - ``numbers.Real``
     - Optional
     - ``1.0``
     - Relative permeability control for this operation.

Returns: surface impedance in ohms.

``METAL_RESISTIVITIES_OHM_M`` contains the supported metal presets.

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
