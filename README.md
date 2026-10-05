# FDFD

Electromagnetic solvers using the finite difference frequency domain method.

Shared materials, shapes, persistence, and periodic eigensolver libraries are included.

| Solver | Python package | Documentation |
|---|---|---|
| band structure | `fdfd_band_structure` | [Guide](doc/solvers/fdfd/band_structure/guide.rst) |
| mode tracking | `fdfd_mode_tracking` | [Guide](doc/solvers/fdfd/mode_tracking/guide.rst) |
| periodic modes | `fdfd_periodic_modes` | [Guide](doc/solvers/fdfd/periodic_modes/guide.rst) |
| scattering | `fdfd_scattering` | [Guide](doc/solvers/fdfd/scattering/guide.rst) |
| waveguide modes | `fdfd_waveguide_modes` | [Guide](doc/solvers/fdfd/waveguide_modes/guide.rst) |

## Install from source

Python 3.11–3.13 is supported; `.python-version` selects Python 3.12.
Install uv, clone this repository, and build from the repository root:

```sh
git clone https://github.com/SolverNotConverging/FDFD.git FDFD
cd FDFD
uv sync
uv run python -m fdfd info
uv run python examples/fdfd/waveguide_modes/rectangular_waveguide_2d.py
```


A C compiler is required for the periodic eigensolver Cython extension.

## Examples and checks

See [examples](examples/README.rst), [documentation](doc/README.rst), and
[benchmarks](benchmarks/README.md). Electromagnetic solvers use `exp(+i*omega*t)`;
passive permittivity has nonpositive imaginary part.

```sh
uv run python -m pytest
uv run python scripts/check_documentation.py
```

Original source is available under the [MIT license](LICENSE).
