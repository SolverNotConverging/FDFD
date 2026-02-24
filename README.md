# FDFD_CEM

Beginner-friendly **Finite-Difference Frequency-Domain (FDFD)** solvers for computational electromagnetics.
Every solver follows the same pattern: create a Yee grid; assign material distributions; apply boundary conditions if
radiation is occurring; and solve linear or eigenvalue problems for the desired field components.
The repository is organised by application area, allowing you to jump directly to the relevant solver.

## 📁 Repository map

### Overview (what each solver is for)

- `Mode_Solver_1D/`: 1‑D slab waveguide eigen-modes (TE/TM) with anisotropy, impedance sheets, and UPML.
- `Mode_Solver_2D/`: 2‑D waveguide cross-section eigen-modes for structures uniform along propagation.
- `Periodic_Solver_2D/`: 2‑D Bloch-periodic waveguides (leaky-wave antennas, periodic lines) with TE/TM solvers.
- `Periodic_Solver_3D/`: 3‑D Bloch-periodic eigen-modes with full-vector fields.
- `Band_Diagram_Solver/`: 2‑D photonic crystal band diagrams (TE/TM bands over Bloch paths).
- `Scattering/`: 2‑D TEz/TMz scattering with total-field/scattered-field masking.
- `Electrostatic_Solver/`: 1‑D/2‑D electrostatic solvers bundled for convenience.

### Solver groups

| Group                    | Folder                  | Main entry point                                                              | Typical problems                                       |
|--------------------------|-------------------------|-------------------------------------------------------------------------------|--------------------------------------------------------|
| Waveguide & cavity modes | `Mode_Solver_1D/`       | [`Mode_Solver_1D.py`](Mode_Solver_1D/Mode_Solver_1D.py)                       | Slab waveguides, impedance sheets, anisotropic layers. |
| Waveguide & cavity modes | `Mode_Solver_2D/`       | [`Mode_Solver_2D.py`](Mode_Solver_2D/Mode_Solver_2D.py)                       | Ridge/rectangular waveguides, cross-section modes.     |
| Periodic structures      | `Periodic_Solver_2D/`   | [`Periodic_Mode_Solver_2D.py`](Periodic_Solver_2D/Periodic_Mode_Solver_2D.py) | 2‑D periodic waveguides, leaky-wave antennas (TM/TE).  |
| Periodic structures      | `Periodic_Solver_3D/`   | [`Periodic_Solver_3D.py`](Periodic_Solver_3D/Periodic_Solver_3D.py)           | 3‑D periodic unit cells, Bloch modes.                  |
| Band diagrams            | `Band_Diagram_Solver/`  | [`Band_Diagram_Solver.py`](Band_Diagram_Solver/Band_Diagram_Solver.py)        | Photonic crystal TE/TM band diagrams.                  |
| Scattering               | `Scattering/`           | [`Scattering_Solver_2D.py`](Scattering/Scattering_Solver_2D.py)               | 2‑D TEz/TMz scattering (plane wave or point source).   |
| Electrostatic            | `Electrostatic_Solver/` | [`Electrostatic_Solver.py`](Electrostatic_Solver/Electrostatic_Solver.py)     | Static field problems (1‑D/2‑D).                       |

### Examples and outputs

Example scripts live next to each solver, and each solver has an `example_outputs/` folder for CSV/NPZ outputs plus
plotting helpers.

- `Mode_Solver_1D/example_anisotropic_slab.py` and `Mode_Solver_1D/example_isotropic_slab.py`
- `Mode_Solver_2D/example_ridge_dielectric_waveguide.py` and
  `Mode_Solver_2D/example_rectangular_dielectric_waveguide.py`
- `Mode_Solver_1D/Modal_1D_Dispersion.py` and `Mode_Solver_2D/Modal_2D_Dispersion.py`
- `Periodic_Solver_2D/Periodic_2D_Dispersion.py` and `Periodic_Solver_2D/example_surface_wave_leaky_wave_antenna.py`
- `Periodic_Solver_3D/Periodic_3D_Dispersion.py` and `Periodic_Solver_3D/example_image_guide_leaky_wave_antenna.py`
- `Scattering/example_scattering_by_cylinder.py`
- `Band_Diagram_Solver/example_square_lattice.py` and `Band_Diagram_Solver/example_rectangular_unitcell.py`

Output data is written to `example_outputs/` inside each solver directory. Modal 1‑D/2‑D and periodic 2‑D dispersion
scripts save CSV files (with matching plotting helpers in the same folder); periodic 3‑D saves NPZ datasets for
full-field storage.

Personal field-visualisation and frequency-sweep scripts live in `personal_use/` and are gitignored. Run them from
inside that folder; they import the solvers from the main directories.

## 🧭 Detailed workflows

The following sections explain the end-to-end process for the core mode
solvers. Each workflow mirrors the implementation in the corresponding
Python module so you know exactly which API calls to use.

### 1‑D waveguide modes (`Mode_Solver_1D`)

1. **Instantiate the solver** – create [`ModeSolver1D`](Mode_Solver_1D/Mode_Solver_1D.py)
   with the operating frequency, spatial span and grid resolution. The
   constructor normalises the derivative matrices using the free-space
   wavenumber and prepares diagonal material tensors.【F:Mode_Solver_1D/Mode_Solver_1D.py†L9-L61】
2. **Populate materials** – call `add_object()` to assign permittivity and
   permeability to slices of the slab. Scalars or length-3 tuples (xx,
   yy, zz) allow isotropic or diagonal-anisotropic regions. Surface
   impedance sheets can be inserted with `add_impedance_surface()` and
   the balanced update ensures TE/TM loadings remain matched.【F:Mode_Solver_1D/Mode_Solver_1D.py†L64-L162】
3. **Solve the eigen-problem for modes** – `solve()` builds sparse diagonal
   matrices for ε/µ, assembles the TE/TM operators and calls
   `scipy.sparse.linalg.eigs`. Propagation constants are the square root
   of each eigenvalue (γ = α + jβ) and field components are back-solved
   on the Yee grid.【F:Mode_Solver_1D/Mode_Solver_1D.py†L198-L246】
4. **Inspect modal fields** – use `visualize_with_gui()` for an interactive
   plot of Ey/Hx/Hz (TE) and Hy/Ex/Ez (TM) along the waveguide together
   with α/β readouts.【F:Mode_Solver_1D/Mode_Solver_1D.py†L248-L352】

Example: `Mode_Solver_1D/example_isotropic_slab.py`, `Mode_Solver_1D/example_anisotropic_slab.py`, and
`Mode_Solver_1D/Modal_1D_Dispersion.py`.

### 2‑D waveguide modes (`Mode_Solver_2D`)

1. **Instantiate the solver** – construct [`ModeSolver2D`](Mode_Solver_2D/Mode_Solver_2D.py)
   with frequency, cross-section sizes and grid counts. The class
   pre-computes Yee-derivative matrices normalised by k₀ and initialises
   2‑D ε/µ tensors.【F:Mode_Solver_2D/Mode_Solver_2D.py†L13-L41】
2. **Populate materials** – `add_object()` writes isotropic or diagonal-anisotropic rectangles into the permittivity and
   permeability
   maps. Optional helpers add impedance sheets (`add_impedance_surface()`) aligned with x or y walls or add UPML
   regions (`add_UPML()`)
   at simulation boundaries.【F:Mode_Solver_2D/Mode_Solver_2D.py†L43-L185】
3. **Solve the eigen-problem for modes** – `solve()` block-assembles the P and Q matrices,
   forms Ω = P·Q and computes the requested number of eigenmodes using a
   shift-invert strategy. Electric and magnetic field components are
   reconstructed by applying the derivative operators and inverse
   material tensors.【F:Mode_Solver_2D/Mode_Solver_2D.py†L187-L230】
4. **Inspect modal fields** – `visualize()` or `visualize_with_gui()` reshape
   the eigenvectors into 2‑D maps, normalise magnitudes and overlay the
   material profile for context.【F:Mode_Solver_2D/Mode_Solver_2D.py†L232-L362】

Example: `Mode_Solver_2D/example_ridge_dielectric_waveguide.py`,
`Mode_Solver_2D/example_rectangular_dielectric_waveguide.py`, and `Mode_Solver_2D/Modal_2D_Dispersion.py`.

### 2‑D periodic structures (`Periodic_Solver_2D`)

1. **Instantiate the solver** – instantiate either
   [`PeriodicTMModeSolver`](Periodic_Solver_2D/Periodic_Mode_Solver_2D.py) to compute the
   TM field triplet (Hy, Ex, Ez) or [`PeriodicTEModeSolver`](Periodic_Solver_2D/Periodic_Mode_Solver_2D.py)
   for the complementary TE components (Ey, Hx, Hz). Both constructors
   share the same signature (frequency, domain sizes, grid resolution)
   and build Bloch-periodic derivative operators along *z*.【F:Periodic_Solver_2D/Periodic_Mode_Solver_2D.py†L10-L73】【F:
   Periodic_Solver_2D/Periodic_Mode_Solver_2D.py†L204-L287】
2. **Populate materials** – `add_object()` populates regions (slices
   along *x* and *z*) with scalar or anisotropic permittivity/permeability.
   Optional `add_UPML()` stretches the coordinates to absorb radiation
   at the transverse boundaries.【F:Periodic_Solver_2D/Periodic_Mode_Solver_2D.py†L75-L129】【F:
   Periodic_Solver_2D/Periodic_Mode_Solver_2D.py†L238-L270】
3. **Solve the eigen-problem for modes** – `solve()` assembles the generalised
   eigen-system A·v = λ·B·v with shift-invert around the supplied guess
   for the complex propagation constant. The resulting eigenvalues are
   normalised by k₀ to yield γ/k₀, whose imaginary part is β and real
   part is −α.【F:Periodic_Solver_2D/Periodic_Mode_Solver_2D.py†L131-L167】【F:
   Periodic_Solver_2D/Periodic_Mode_Solver_2D.py†L272-L306】
4. **Inspect modal fields** – `visualize_with_gui()` reshapes the eigenvectors to
   display the available field components for the chosen polarisation: |Hy|/|Ex|/|Ez| for TM or |Ey|/|Hx|/|Hz| for TE,
   overlaid on the permittivity map and annotated with the complex propagation constants.【F:
   Periodic_Solver_2D/Periodic_Mode_Solver_2D.py†L169-L231】【F:Periodic_Solver_2D/Periodic_Mode_Solver_2D.py†L308-L380】

Example: `Periodic_Solver_2D/example_surface_wave_leaky_wave_antenna.py` and
`Periodic_Solver_2D/Periodic_2D_Dispersion.py`.

### 3‑D periodic structures (`Periodic_Solver_3D`)

1. **Initialise the solver** – create [`PeriodicModeSolver3D`](Periodic_Solver_3D/Periodic_Solver_3D.py)
   with grid dimensions, physical spans and frequency. The constructor
   builds Kronecker-product derivative matrices with periodicity along z
   and allocates 3‑D ε/µ arrays.【F:Periodic_Solver_3D/Periodic_Solver_3D.py†L9-L63】
2. **Populate materials** – `add_object()` writes scalar or anisotropic
   permittivity/permeability tensors into cuboidal regions of the unit
   cell.  `add_UPML()` optionally wraps selected faces with polynomial
   UPML stretches.【F:Periodic_Solver_3D/Periodic_Solver_3D.py†L65-L123】
3. **Solve the eigen-problem for modes** – `solve()` constructs the full-vector
   generalised eigen-problem (A, B) for the four tangential field
   components, applies shift-invert and divides the eigenvalues by k₀ to
   obtain the complex propagation constants γ/k₀.【F:Periodic_Solver_3D/Periodic_Solver_3D.py†L125-L178】
4. **Inspect modal fields** – `store_fields()` reshapes the eigenvectors
   into volumetric Ex/Ey/Hx/Hy arrays that can be sliced with
   `plot_field_plane()` for visual analysis.【F:Periodic_Solver_3D/Periodic_Solver_3D.py†L180-L216】

Example: `Periodic_Solver_3D/example_image_guide_leaky_wave_antenna.py` and
`Periodic_Solver_3D/Periodic_3D_Dispersion.py`.

### 2‑D scattering (`Scattering`)

1. **Instantiate the solver** – create [`FDFD2DScatteringSolver`](Scattering/Scattering_Solver_2D.py) with frequency,
   domain size, and grid resolution. The class prepares the Yee-grid operators and coordinate grids.【F:
   Scattering/Scattering_Solver_2D.py†L9-L79】
2. **Define materials** – call `add_object()` with a boolean mask to assign ε/µ for scatterers embedded in the
   background medium.【F:Scattering/Scattering_Solver_2D.py†L81-L109】
3. **Add excitation** – use `add_source()` for plane waves or point sources, selecting TE/TM polarization and source
   parameters.【F:Scattering/Scattering_Solver_2D.py†L111-L153】
4. **Absorbing boundaries and TF/SF mask** – apply `add_UPML()` and `add_mask()` to create the
   total-field/scattered-field region.【F:Scattering/Scattering_Solver_2D.py†L155-L199】
5. **Solve and visualise** – run `solve_total_field_TE()` or `solve_total_field_TM()` and plot quick diagnostics with
   `TE_Visualization()` or `TM_Visualization()`.【F:Scattering/Scattering_Solver_2D.py†L201-L268】

Example: `Scattering/example_scattering_by_cylinder.py`.

### Photonic band diagrams (`Band_Diagram_Solver`)

[`BandDiagramSolver2D`](Band_Diagram_Solver/Band_Diagram_Solver.py) is a fully
fledged class replacing the previous script-style implementation. The
workflow mirrors the other solvers:

1. **Instantiate the solver** with the rectangular lattice periods (`a`
   along x and optional `b` along y) and Yee grid size. The constructor
   creates a 2×-refined helper grid that matches Rumpf's subpixel averaging
   strategy.【F:Band_Diagram_Solver/Band_Diagram_Solver.py†L57-L118】
2. **Populate materials** using `add_object()` or convenience helpers such as
   `add_circular_inclusion()`; masks can be arrays or callables of the
   helper grid coordinates.【F:Band_Diagram_Solver/Band_Diagram_Solver.py†L120-L167】
3. **Define the Bloch path** with `default_rectangular_lattice_path()`
   (Γ–X–M–Y–Γ for rectangular cells) and `generate_bloch_path()`, then
   optionally set tick labels for the symmetry points.【F:Band_Diagram_Solver/Band_Diagram_Solver.py†L169-L270】
4. **Compute the bands** using `compute_band_structure()`, which extracts
   the Yee-grid material tensors, builds derivative operators for each
   Bloch vector and solves the TE/TM sparse eigen-problems. Eigenvalues
   are sorted and normalised to `a/λ`. Results are returned as a
   `BandStructureResult` dataclass for easy post-processing.【F:Band_Diagram_Solver/Band_Diagram_Solver.py†L272-L361】

5. **Plot the diagram** with `plot_band_diagram()`, which renders the unit
   cell, overlays the sampled Bloch path directly in reciprocal space,
   charts the TE/TM bands, saves the figure to `band_diagram.png`, and
   displays it. You can tweak the path styling via the optional
   ``path_artist_kwargs`` argument.【F:Band_Diagram_Solver/Band_Diagram_Solver.py†L363-L459】

Examples: `Band_Diagram_Solver/example_square_lattice.py` and `Band_Diagram_Solver/example_rectangular_unitcell.py`.

---

### Reference

R. Rumpf, *Electromagnetic and Photonic Simulation for the Beginner:
Finite-Difference Frequency-Domain in MATLAB*. Artech House, 2022.
