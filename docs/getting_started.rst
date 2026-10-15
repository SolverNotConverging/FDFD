Getting started
===============

Installation
------------

From a checkout, in a Python 3.10+ environment:

.. code-block:: console

   pip install -e .

or, with `uv <https://docs.astral.sh/uv/>`_, ``uv sync`` (which also installs
the test and lint tools). GPU acceleration and the compiled Arnoldi kernels are
optional; see :doc:`acceleration`. ``python -m fdfd info`` reports what is
available.

The workflow
------------

Every solver follows the same five steps.

1. **Construct** the solver with its frequency (except band structures) and
   domain. All arguments are keyword-only, lengths are in metres and ranges are
   ``(min, max)`` pairs.

2. **Add geometry.** Define :class:`~fdfd.Material` objects once and assign
   them to shapes. Every solver has :meth:`add_geometry` for any
   :mod:`fdfd.shapes` shape, plus convenience methods such as
   ``add_rectangle``. Later objects win where shapes overlap. Each call returns
   a handle for :meth:`set_material`, :meth:`set_shape` or :meth:`remove`.

   .. code-block:: python

      import fdfd
      from fdfd import materials, shapes

      substrate = fdfd.Material(name="substrate", epsilon=3.55 - 0.01j)
      solver = fdfd.ModeSolver2D(frequency=50e9, x_range=(0, 12e-3), y_range=(0, 10e-3))
      solver.add_rectangle(x_range=(2e-3, 10e-3), y_range=(4e-3, 5e-3), material=substrate)
      solver.add_rectangle(x_range=(5e-3, 7e-3), y_range=(5e-3, 5.1e-3), material=materials.copper)

3. **Add a PML** where the domain should be open (``"all"``, an axis such as
   ``"x"``, or a side such as ``"x+"``):

   .. code-block:: python

      solver.add_pml(thickness=1e-3, sides="y+")

4. **Mesh** with a cell count or a maximum cell size. ``subpixels`` (default 8)
   controls how finely material interfaces are averaged:

   .. code-block:: python

      solver.mesh(resolution=(120, 100))
      # or: solver.mesh(max_element_size=0.1e-3)

   :func:`fdfd.suggest_resolution` gives a starting cell count from the
   shortest wavelength in the problem.

5. **Solve**, then inspect, save and reload the result:

   .. code-block:: python

      result = solver.solve(num_modes=4, device="auto")
      print(result.neff, result.info.converged)
      result.show()                    # interactive viewer
      result.plot(mode=2, component="Ey", quantity="magnitude").savefig("ey.png")
      result.save("modes.h5")
      same = fdfd.load("modes.h5")     # any result type

Solve arguments
---------------

The same names mean the same thing everywhere:

``num_modes``
   Number of eigensolutions returned (modal solvers).
``guess``
   Where to search: an effective index for mode solvers, a frequency in hertz
   for band structures. ``None`` picks a sensible default (fundamental modes,
   or the lowest bands).
``polarization``
   ``"TE"``, ``"TM"`` or ``"both"`` for scalar 1D/2D problems. Full-vector
   solvers have no polarization.
``tol``
   Eigensolver tolerance; 0 means machine precision.
``device``
   ``"cpu"`` (SciPy), ``"cuda"`` (cuDSS on the GPU) or ``"auto"`` (CUDA when
   available, otherwise CPU).
``method``
   ``"arpack"`` or ``"refined"`` Arnoldi (modal solvers).

Results
-------

All results derive from :class:`~fdfd.Result` and share ``family``, ``grid``,
``info`` (a :class:`~fdfd.SolveInfo`), ``metadata``, ``plot()``, ``show()``
and ``save()``. :class:`~fdfd.ModeResult` adds ``neff``, ``beta`` and
``attenuation_constant``; field results store each component on its own Yee
lattice with explicit ``field_coordinates``.

Examples
--------

The ``examples/`` folder has runnable scripts for every solver, grouped by
family. Each writes its results to an ``outputs/`` folder next to it, and
``examples/postprocessing/inspect_result.py`` reopens any saved result.
