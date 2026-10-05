# FDFD

Electromagnetic solvers for band structure, mode tracking, periodic modes, scattering, and waveguide modes. Each solver folder contains its own `src/`, `docs/`, and `examples/`. Shared materials and geometry live in `fdfd_common/`.

Use Python 3.11–3.13. There are two ways to use the solvers.

## 1. Clone and run examples

Clone the repository, install the required Python packages, and run an example:

```sh
git clone https://github.com/SolverNotConverging/FDFD.git
cd FDFD
python -m pip install -r requirements.txt
python fdfd_waveguide_modes/examples/parallel_plate_waveguide_1d.py
```

Examples import the solver source directly, so you do not need to install FDFD itself. You can also run examples as modules from the repository folder:

```sh
python -m fdfd_waveguide_modes.examples.parallel_plate_waveguide_1d
```

## 2. Install and import from any folder

From the cloned repository folder, install FDFD and its required Python packages:

```sh
python -m pip install .
```

You can then import the solvers in your own scripts or notebooks from any folder, using the same Python environment:

```python
from fdfd_common import Material, materials, shapes
from fdfd_waveguide_modes import ModeSolver1D, ModeSolver2D
```

You can also install directly from GitHub:

```sh
python -m pip install "git+https://github.com/SolverNotConverging/FDFD.git"
```

## Optional Cython kernel

Both methods use NumPy by default and work without a compiler. `periodic_eigensolver/` also provides an optional Cython kernel that accelerates part of the periodic eigensolver. It requires a C compiler; the solvers do not require Qt or any other native build.

For examples using the cloned source, build the kernel in the repository folder:

```sh
python -m pip install "Cython>=3,<4" "setuptools>=77,<83"
python setup_cython.py build_ext --inplace
```

The solver automatically uses the compiled kernel when available, and otherwise uses NumPy. To include the kernel in an installed FDFD package, use these PowerShell commands from the repository folder:

```powershell
$env:FDFD_BUILD_CYTHON = "1"
python -m pip install --force-reinstall .
Remove-Item Env:FDFD_BUILD_CYTHON
```

On Linux or macOS, use `FDFD_BUILD_CYTHON=1 python -m pip install --force-reinstall .`.

## Solver guides and examples

| Solver | Import | Documentation | Examples |
|---|---|---|---|
| band structure | `fdfd_band_structure` | [Guide](fdfd_band_structure/docs/guide.rst) | [Examples](fdfd_band_structure/examples/README.rst) |
| mode tracking | `fdfd_mode_tracking` | [Guide](fdfd_mode_tracking/docs/guide.rst) | [Examples](fdfd_mode_tracking/examples/README.rst) |
| periodic modes | `fdfd_periodic_modes` | [Guide](fdfd_periodic_modes/docs/guide.rst) | [Examples](fdfd_periodic_modes/examples/README.rst) |
| scattering | `fdfd_scattering` | [Guide](fdfd_scattering/docs/guide.rst) | [Examples](fdfd_scattering/examples/README.rst) |
| waveguide modes | `fdfd_waveguide_modes` | [Guide](fdfd_waveguide_modes/docs/guide.rst) | [Examples](fdfd_waveguide_modes/examples/README.rst) |

Generated results are saved under `outputs/`. Source is [MIT licensed](LICENSE).
