# FDFD

Finite-Difference Frequency-Domain solvers for computational electromagnetics.

The repository is organised by problem type. Each solver folder contains the solver implementation, example scripts, and a solver-specific ``README.rst`` with API and workflow notes.

## Time-Harmonic Convention

Electromagnetic solvers in this repository use the phasor convention

$$
\mathbf{F}(\mathbf{r},t)=\Re\{\mathbf{F}(\mathbf{r})e^{+j\omega t}\}.
$$

This is paired with the forward and inverse Fourier transforms

$$
\widetilde{F}(\omega)=\int_{-\infty}^{\infty}F(t)e^{-j\omega t}\,dt,
\qquad
F(t)=\frac{1}{2\pi}\int_{-\infty}^{\infty}\widetilde{F}(\omega)e^{+j\omega t}\,d\omega.
$$

In other words: the forward transform carries ``-j*omega*t`` and the
frequency-domain wave/phasor carries ``+j*omega*t``. This is also the sign
pair used by the usual forward/inverse FFT convention.

Forward and outgoing waves therefore use ``exp(-j k·r)`` (and guided modes use ``exp(-j beta z)``). Under this convention:

- passive bulk materials have $\mathrm{Im}(\epsilon_r)\le 0$ and $\mathrm{Im}(\mu_r)\le 0$;
- passive forward guided modes have $\mathrm{Im}(n_\mathrm{eff})\le 0$, with field-amplitude attenuation $\alpha=-k_0\mathrm{Im}(n_\mathrm{eff})\ge 0$ (and power decaying as $e^{-2\alpha z}$);
- positive conductivity enters bulk permittivity as $-j\sigma/(\omega\epsilon_0)$ and the PML stretch as $1-j\sigma/(\omega\epsilon_0)$;
- passive scalar surface impedance still has $\mathrm{Re}(Z_s)\ge 0$; the good-conductor preset is $Z_s=(1+j)R_s$;
- decaying complex-frequency eigenmodes have $\mathrm{Im}(\omega)\ge 0$;
- outgoing two-dimensional Green functions use $H_0^{(2)}$.

The electrostatic solver has no time dependence, so this convention does not affect it.

## Solver Map

| Folder | Solver | Use case | Documentation |
|---|---|---|---|
| `FEM_Mode_Solver/` | `ModeSolver1D`, `ModeSolver2D` | Standalone conforming-FEM modes with adaptive meshing and 2D SIBC conductors | [`README.rst`](FEM_Mode_Solver/README.rst) |
| `Mode_Solver_1D/` | `ModeSolver1D` | TE/TM modes of 1D slab waveguides | [`README.rst`](Mode_Solver_1D/README.rst) |
| `Mode_Solver_2D/` | `ModeSolver2D` | Full-vector modes of 2D waveguide cross-sections | [`README.rst`](Mode_Solver_2D/README.rst) |
| `Periodic_Solver_2D/` | `PeriodicModeSolver2D` | 2D Bloch-periodic TE/TM unit-cell modes | [`README.rst`](Periodic_Solver_2D/README.rst) |
| `Periodic_Solver_3D/` | `PeriodicModeSolver3D` | 3D Bloch-periodic full-vector modes | [`README.rst`](Periodic_Solver_3D/README.rst) |
| `Band_Diagram_Solver/` | `BandDiagramSolver2D` | 2D photonic-crystal band diagrams | [`README.rst`](Band_Diagram_Solver/README.rst) |
| `Scattering/` | `FDFD2DScatteringSolver` | 2D TEz/TMz scattering problems | [`README.rst`](Scattering/README.rst) |
| `Electrostatic_Solver/` | `ElectrostaticSolver` | 1D/2D electrostatic potential problems | [`README.rst`](Electrostatic_Solver/README.rst) |

Utility scripts at the repository root include `Mesh_points_calculation.py` and `PML_sigma_calculation.py`.

## Requirements

The solvers are plain Python modules. A typical environment needs:

```bash
pip install numpy scipy matplotlib
```

Some visualizers use Tk through Matplotlib. If GUI windows do not open, install the Tk package for your Python distribution.

The standalone FEM mode package additionally needs scikit-fem and Gmsh; its
isolated installation and environment are documented in
[`FEM_Mode_Solver/README.rst`](FEM_Mode_Solver/README.rst).

## Basic Workflow

1. Pick the solver folder that matches the physics.
2. Read that folder's `README.rst` for the API and expected outputs.
3. Run or modify the example script in the same folder.
4. Keep generated data in the folder's `example_outputs/` directory.

## Examples

```bash
cd Mode_Solver_1D
python example_grounded_isotropic_slab.py
```

```bash
cd Mode_Solver_2D
python example_ridge_dielectric_waveguide.py
```

```bash
python -m FEM_Mode_Solver.examples.slab_1d
python -m FEM_Mode_Solver.examples.ridge_2d
python -m FEM_Mode_Solver.examples.microstrip_sibc
```

```bash
cd Band_Diagram_Solver
python example_square_lattice.py
```

## Notes

- Length units are SI metres unless an example explicitly normalises geometry.
- Material values are relative tensors unless stated otherwise.
- Large grids produce large sparse systems. Start with coarse examples before increasing resolution.
- Several solvers use shift-invert eigensolves. If convergence is poor, adjust the mode count, grid size, or eigenvalue guess.

## Reference

R. Rumpf, *Electromagnetic and Photonic Simulation for the Beginner: Finite-Difference Frequency-Domain in MATLAB*. Artech House, 2022.
