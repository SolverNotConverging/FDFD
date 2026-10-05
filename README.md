# FDFD

Electromagnetic solvers. Each root solver folder contains its own `src/`, `docs/`, and `examples/`.

| Solver | Import | Documentation | Examples |
|---|---|---|---|
| band structure | `fdfd_band_structure` | [Guide](fdfd_band_structure/docs/guide.rst) | [Examples](fdfd_band_structure/examples/README.rst) |
| mode tracking | `fdfd_mode_tracking` | [Guide](fdfd_mode_tracking/docs/guide.rst) | [Examples](fdfd_mode_tracking/examples/README.rst) |
| periodic modes | `fdfd_periodic_modes` | [Guide](fdfd_periodic_modes/docs/guide.rst) | [Examples](fdfd_periodic_modes/examples/README.rst) |
| scattering | `fdfd_scattering` | [Guide](fdfd_scattering/docs/guide.rst) | [Examples](fdfd_scattering/examples/README.rst) |
| waveguide modes | `fdfd_waveguide_modes` | [Guide](fdfd_waveguide_modes/docs/guide.rst) | [Examples](fdfd_waveguide_modes/examples/README.rst) |

## Run from the checkout

Install the Python dependencies, then run an example. No solver package installation is needed.

```sh
python -m pip install -r requirements.txt
python fdfd_waveguide_modes/examples/parallel_plate_waveguide_1d.py
```

You can also run examples as modules from the root:

```sh
python -m fdfd_waveguide_modes.examples.parallel_plate_waveguide_1d
```

Shared materials and geometry live in `cem_common/`; `periodic_eigensolver/` provides
the NumPy eigensolver and an optional Cython kernel. To compile the kernel in place:

```sh
python -m pip install "Cython>=3,<4" "setuptools>=77,<83"
python setup_cython.py build_ext --inplace
```

Without the compiled kernel, the default backend uses NumPy. A C compiler is needed only for this optional build.

## Checks

```sh
python -m pip install -r requirements-dev.txt
python -m pytest
python scripts/check_documentation.py
python scripts/qualify_examples.py --import-only
```

`uv sync` can also create a dependency environment without installing the solvers.
Generated results are saved under ignored `outputs/`. Source is [MIT licensed](LICENSE).
