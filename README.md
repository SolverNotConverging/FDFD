# FDFD

This repository provides finite-difference frequency-domain (FDFD) solvers for electromagnetic problems, including waveguide modes, periodic eigenmodes, photonic band structures, scattering, and mode tracking. Each solver family includes its source code, documentation, and runnable examples.

You can run examples directly from the downloaded source after installing the required Python packages, or optionally install FDFD to use the solvers from any folder in the same environment.

## Choose a solver

| Solver | What it solves | Import | Documentation | Examples |
|---|---|---|---|---|
| band structure | Finds eigenfrequencies versus Bloch wavevector in 2D periodic structures, for photonic bands and band gaps. | `fdfd_band_structure` | [Guide](fdfd_band_structure/docs/guide.rst) | [Examples](fdfd_band_structure/examples/README.rst) |
| mode tracking | Follows the same waveguide modes across a frequency sweep, including crossings, degeneracies, and cutoff; provides bound-mode fields for ports. | `fdfd_mode_tracking` | [Guide](fdfd_mode_tracking/docs/guide.rst) | [Examples](fdfd_mode_tracking/examples/README.rst) |
| periodic modes | Finds Bloch propagation constants and fields at a chosen frequency in 2D or 3D structures that repeat along the propagation direction, such as loaded guides and leaky-wave antennas. | `fdfd_periodic_modes` | [Guide](fdfd_periodic_modes/docs/guide.rst) | [Examples](fdfd_periodic_modes/examples/README.rst) |
| scattering | Computes total and scattered fields for 2D TE or TM illumination problems, such as a plane wave incident on a dielectric cylinder. | `fdfd_scattering` | [Guide](fdfd_scattering/docs/guide.rst) | [Examples](fdfd_scattering/examples/README.rst) |
| waveguide modes | Finds propagation constants, effective indices, and mode fields at a chosen frequency for layered 1D guides or 2D waveguide cross sections. | `fdfd_waveguide_modes` | [Guide](fdfd_waveguide_modes/docs/guide.rst) | [Examples](fdfd_waveguide_modes/examples/README.rst) |

Each solver folder contains its own `src/`, `docs/`, and `examples/`. Shared materials and geometry live in `fdfd_common/`.

## Download, install required packages, and run

Download the repository using **Code → Download ZIP** on [GitHub](https://github.com/SolverNotConverging/FDFD), extract it, and open a terminal in the extracted folder. You can also download it with Git:

```sh
git clone https://github.com/SolverNotConverging/FDFD.git
cd FDFD
```

Install the required Python packages and run an example:

```sh
python -m pip install -r requirements.txt
python fdfd_waveguide_modes/examples/1d_parallel_plate_waveguide.py
```

The examples import the solver source directly. **Installing FDFD itself is not required.** Choose other examples from the table above.

## Optional installation: import from any folder

Installing FDFD mainly lets you import and use the solvers from any folder in the same Python environment, including your own scripts and notebooks. The installation also installs the required Python packages. You can install the release wheel directly, or install from the downloaded source.

The [release wheel](https://github.com/SolverNotConverging/FDFD/releases/latest)
works on Windows, macOS, and Linux. It is pure Python (`py3-none-any`), includes
the NumPy fallback, and contains no Cython kernel. Install it directly without
building FDFD:

```sh
python -m pip install https://github.com/SolverNotConverging/FDFD/releases/download/v1.1.0/fdfd-1.1.0-py3-none-any.whl
```

In an activated environment, you can use `uv pip install` with the same wheel URL.
In a conda environment, use that environment's `python -m pip install` command.
The methods below install from the downloaded source instead.

### pip

With your Python environment active:

```sh
python -m pip install .
```

### uv

With your environment active, [uv](https://docs.astral.sh/uv/pip/packages/) can install FDFD:

```sh
uv pip install .
```

To create an environment first, run `uv venv`. Activate it with `.\.venv\Scripts\Activate.ps1` in PowerShell, or `source .venv/bin/activate` on Linux or macOS, then run the installation command above. Keep that environment active when using FDFD from another folder.

### Conda environment

Create and activate a [Conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html#using-pip-in-an-environment), then install FDFD with its pip:

```sh
conda create -n fdfd python pip
conda activate fdfd
python -m pip install .
```

After installation, you can use imports such as these from any folder in that environment:

```python
from fdfd_waveguide_modes import ModeSolver1D, ModeSolver2D, Material, materials, shapes
```

## Optional Cython build

The Cython kernel is **completely optional**. It accelerates the periodic eigenmodes calculation by speeding up parts of `periodic_eigensolver`. If it is not built, the solver automatically uses the NumPy fallback. Normal use and installation require no compiler; building this optional kernel requires a C compiler.

For examples using the downloaded source, build the kernel in the repository folder:

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

Generated results are saved in the `outputs/` folder inside each solver family, such as `fdfd_waveguide_modes/outputs/`. Source is [MIT licensed](LICENSE).
