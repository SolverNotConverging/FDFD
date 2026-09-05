fdfd_scattering user API
========================

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

``ScatteringSolver2D``
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSolver2D(frequency, x_range, y_range, Nx, Ny)

2-D frequency–domain FDFD solver (TEz / TMz) on a Yee grid. Geometry is defined on the cell centres,   E/H - derivatives on staggered edges via the helper `yeeder2d`.

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

Returns: a configured solver.

``ScatteringSolver2D.add_object``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSolver2D.add_object(er_tensor, mr_tensor, region_mask)

region_mask : boolean Ny×Nx array – cells where the object lives er_tensor   : scalar or len-3 list/array  (ε_xx, ε_yy, ε_zz) mr_tensor   : same for μ

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``er_tensor``
     - ``array-like or scalar``
     - Required
     - ``—``
     - Relative diagonal electric constitutive tensor.
   * - ``mr_tensor``
     - ``array-like or scalar``
     - Required
     - ``—``
     - Relative diagonal magnetic constitutive tensor.
   * - ``region_mask``
     - ``array-like or scalar``
     - Required
     - ``—``
     - Boolean array identifying cell centres in the object.

Returns: the documented data or None when storing state on the solver.

``ScatteringSolver2D.add_source``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSolver2D.add_source(src_type: str = 'plane_wave', angle_deg: float = 0.0, polarization: str = 'TE', location: tuple | None = None, amplitude: float | complex = 1.0)

Populate self.source (flattened Ny×Nx) with the incident field.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``src_type``
     - ``str``
     - Optional
     - ``'plane_wave'``
     - Src type control for this operation.
   * - ``angle_deg``
     - ``float``
     - Optional
     - ``0.0``
     - Angle deg control for this operation.
   * - ``polarization``
     - ``str``
     - Optional
     - ``'TE'``
     - Polarization control for this operation.
   * - ``location``
     - ``tuple | None``
     - Optional
     - ``None``
     - Location control for this operation.
   * - ``amplitude``
     - ``float | complex``
     - Optional
     - ``1.0``
     - Complex incident-mode amplitude.

Returns: the documented data or None when storing state on the solver.

``ScatteringSolver2D.add_UPML``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSolver2D.add_UPML(pml_width=20, n=3, sigma_max=5.0, direction='both')

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
     - ``20``
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
     - ``'both'``
     - Propagation direction for solve; selected coordinate direction for PML.

Returns: the documented data or None when storing state on the solver.

``ScatteringSolver2D.add_mask``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSolver2D.add_mask(value: int | float | numpy.ndarray | scipy.sparse._matrix.spmatrix = 30)

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``value``
     - ``int | float | numpy.ndarray | scipy.sparse._matrix.spmatrix``
     - Optional
     - ``30``
     - Value control for this operation.

Returns: the documented data or None when storing state on the solver.

``ScatteringSolver2D.solve_total_field_TE``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSolver2D.solve_total_field_TE(reuse_factorisation: bool = True)

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``reuse_factorisation``
     - ``bool``
     - Optional
     - ``True``
     - Reuse factorisation control for this operation.

Returns: the documented data or None when storing state on the solver.

``ScatteringSolver2D.solve_total_field_TM``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSolver2D.solve_total_field_TM(reuse_factorisation: bool = True)

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``reuse_factorisation``
     - ``bool``
     - Optional
     - ``True``
     - Reuse factorisation control for this operation.

Returns: the documented data or None when storing state on the solver.

``ScatteringSolver2D.TE_Visualization``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSolver2D.TE_Visualization()

4-panel figure: 1 – ``|ε_r|``   (structure) 2 – Re(incident Ez) 3 – Q mask 4 – Re(total Ez)

Returns: the documented data or None when storing state on the solver.

``ScatteringSolver2D.TM_Visualization``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSolver2D.TM_Visualization()

4-panel figure analogous to TE_Visualization, but for TM (Hz).

Returns: the documented data or None when storing state on the solver.

Results and examples
--------------------

TE/TM solves retain sampled fields on the solver. Geometry and sources
are configured before solving; field arrays follow the Yee-grid locations.

Invalid dimensions, materials, and solver controls raise ValueError or
NotImplementedError. Numerical backend failures remain visible.

Run the examples with the installed package; no repository path changes
are required. See `README.rst <README.rst>`_ and the ``examples/`` directory.
Assembly routines, matrix builders, and Arnoldi kernels are implementation
details and are excluded from this reference.
