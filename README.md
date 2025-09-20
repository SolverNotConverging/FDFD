# FDFD_CEM

Beginner-friendly **Finite-Difference Frequency-Domain (FDFD)** solvers for
computational electromagnetics.  Every solver follows the same pattern:
create a Yee grid, assign material distributions, apply boundary
conditions and solve sparse eigenvalue problems for the desired field
components.  The repository is organised by application area so you can
jump directly to the solver that matches your problem.

## 📁 Repository map

### Waveguide & cavity mode solvers

| Folder | Description |
| --- | --- |
| `Mode_Solver_1D/` | 1‑D slab waveguide eigen-mode solver with anisotropic materials, impedance sheets and uniaxial PML.  Main entry point: [`FDFD_1D_Mode_Solver.py`](Mode_Solver_1D/FDFD_1D_Mode_Solver.py). |
| `Mode_Solver_2D/` | 2‑D cross-section mode solver for structures that are uniform along the propagation axis.  Supports anisotropy, impedance sheets and UPML.  Main entry point: [`FDFD_Mode_Solver.py`](Mode_Solver_2D/FDFD_Mode_Solver.py). |
| `Periodic_2D/` | FDFD solver for 2‑D periodic waveguides (e.g. leaky-wave antennas).  Periodicity is enforced along *z*; materials may vary along *x* and *z*. |
| `Periodic_3D/` | 3‑D periodic mode solver with Bloch-periodic boundary conditions along *z* and full-vector fields. |

### Photonic crystal analysis

| Folder | Description |
| --- | --- |
| `Band_Diagram_Solver/` | Contains the new class-based band diagram engine [`2D_Band_Diagram.py`](Band_Diagram_Solver/2D_Band_Diagram.py).  Users can compose a unit cell by adding objects, sweep Bloch wave vectors and visualise TE/TM photonic bands. |

### Other solvers and utilities

| Folder / File | Purpose |
| --- | --- |
| `Scattering/` | 2‑D TE/TM scattering solver formulated with the QAAQ matrix approach. |
| `Electrostatic_Solver/` | Electrostatic field solvers in 1‑D and 2‑D (not FDFD-based but bundled for convenience). |
| `Mesh_points_calculation.py` | Generates spatial mesh points for arbitrary simulation domains. |
| `PML_sigma_calculation.py` | Utility for deriving polynomial conductivity profiles used in UPML implementations. |

---

## 🧭 Detailed workflows

The following sections explain the end-to-end process for the core mode
solvers.  Each workflow mirrors the implementation in the corresponding
Python module so you know exactly which API calls to use.

### 1‑D waveguide modes (`Mode_Solver_1D`)

1. **Instantiate the solver** – create [`FDFDModeSolver`](Mode_Solver_1D/FDFD_1D_Mode_Solver.py)
   with the operating frequency, spatial span and grid resolution.  The
   constructor normalises the derivative matrices using the free-space
   wavenumber and prepares diagonal material tensors.【F:Mode_Solver_1D/FDFD_1D_Mode_Solver.py†L9-L61】
2. **Define materials** – call `add_object()` to assign permittivity and
   permeability to slices of the slab.  Scalars or length-3 tuples (xx,
   yy, zz) allow isotropic or diagonal-anisotropic regions.  Surface
   impedance sheets can be inserted with `add_impedance_surface()` and
   the balanced update ensures TE/TM loadings remain matched.【F:Mode_Solver_1D/FDFD_1D_Mode_Solver.py†L64-L162】
3. **Add absorbing boundaries** – `add_UPML()` wraps the domain with
   uniaxial PML by stretching ε and µ according to the requested
   conductivity taper.【F:Mode_Solver_1D/FDFD_1D_Mode_Solver.py†L164-L196】
4. **Solve the eigen-problem** – `solve()` builds sparse diagonal
   matrices for ε/µ, assembles the TE/TM operators and calls
   `scipy.sparse.linalg.eigs`.  Propagation constants are the square root
   of each eigenvalue (γ = α + jβ) and field components are back-solved
   on the Yee grid.【F:Mode_Solver_1D/FDFD_1D_Mode_Solver.py†L198-L246】
5. **Inspect the results** – use `visualize_with_gui()` for an interactive
   plot of Ey/Hx/Hz (TE) and Hy/Ex/Ez (TM) along the waveguide together
   with α/β readouts.【F:Mode_Solver_1D/FDFD_1D_Mode_Solver.py†L248-L352】

### 2‑D waveguide modes (`Mode_Solver_2D`)

1. **Instantiate the solver** – construct [`FDFDModeSolver`](Mode_Solver_2D/FDFD_Mode_Solver.py)
   with frequency, cross-section sizes and grid counts.  The class
   pre-computes Yee-derivative matrices normalised by k₀ and initialises
   2‑D ε/µ tensors.【F:Mode_Solver_2D/FDFD_Mode_Solver.py†L13-L41】
2. **Populate the cross-section** – `add_object()` writes isotropic or
   diagonal-anisotropic rectangles into the permittivity and permeability
   maps.  Optional helpers add UPML regions (`add_UPML()`) or impedance
   sheets (`add_impedance_surface()`) aligned with x or y walls.【F:Mode_Solver_2D/FDFD_Mode_Solver.py†L43-L185】
3. **Solve for modes** – `solve()` block-assembles the P and Q matrices,
   forms Ω = P·Q and computes the requested number of eigenmodes using a
   shift-invert strategy.  Electric and magnetic field components are
   reconstructed by applying the derivative operators and inverse
   material tensors.【F:Mode_Solver_2D/FDFD_Mode_Solver.py†L187-L230】
4. **Visualise fields** – `visualize()` or `visualize_with_gui()` reshape
   the eigenvectors into 2‑D maps, normalise magnitudes and overlay the
   material profile for context.【F:Mode_Solver_2D/FDFD_Mode_Solver.py†L232-L362】

### 2‑D periodic structures (`Periodic_2D`)


1. **Select the polarisation solver** – instantiate either
   [`TM_Mode_Solver`](Periodic_2D/Periodic_Mode_Solver.py) to compute the
   TM field triplet (Hy, Ex, Ez) or [`TE_Mode_Solver`](Periodic_2D/Periodic_Mode_Solver.py)
   for the complementary TE components (Ey, Hx, Hz).  Both constructors
   share the same signature (frequency, domain sizes, grid resolution)
   and build Bloch-periodic derivative operators along *z*.【F:Periodic_2D/Periodic_Mode_Solver.py†L10-L73】【F:Periodic_2D/Periodic_Mode_Solver.py†L204-L287】
   
2. **Define the unit cell** – `add_object()` populates regions (slices
   along *x* and *z*) with scalar or anisotropic permittivity/permeability.
   Optional `add_UPML()` stretches the coordinates to absorb radiation
   at the transverse boundaries.【F:Periodic_2D/Periodic_Mode_Solver.py†L75-L129】【F:Periodic_2D/Periodic_Mode_Solver.py†L238-L270】

3. **Solve the Bloch eigen-problem** – `solve()` assembles the generalised
   eigen-system A·v = λ·B·v with shift-invert around the supplied guess
   for the complex propagation constant.  The resulting eigenvalues are
   normalised by k₀ to yield γ/k₀, whose imaginary part is β and real
   part is −α.【F:Periodic_2D/Periodic_Mode_Solver.py†L131-L167】【F:Periodic_2D/Periodic_Mode_Solver.py†L272-L306】
   
4. **Post-process** – `visualize_with_gui()` reshapes the eigenvectors to
   display the available field components for the chosen polarisation:
   |Hy|/|Ex|/|Ez| for TM or |Ey|/|Hx|/|Hz| for TE, overlaid on the
   permittivity map and annotated with the complex propagation constants.【F:Periodic_2D/Periodic_Mode_Solver.py†L169-L231】【F:Periodic_2D/Periodic_Mode_Solver.py†L308-L380】

### 3‑D periodic structures (`Periodic_3D`)

1. **Initialise the solver** – create [`Periodic_3D_Mode_Solver`](Periodic_3D/Periodic_Mode_Solver_3D.py)
   with grid dimensions, physical spans and frequency.  The constructor
   builds Kronecker-product derivative matrices with periodicity along z
   and allocates 3‑D ε/µ arrays.【F:Periodic_3D/Periodic_Mode_Solver_3D.py†L9-L63】
2. **Populate materials** – `add_object()` writes scalar or anisotropic
   permittivity/permeability tensors into cuboidal regions of the unit
   cell.  `add_UPML()` optionally wraps selected faces with polynomial
   UPML stretches.【F:Periodic_3D/Periodic_Mode_Solver_3D.py†L65-L123】
3. **Solve for Bloch modes** – `solve()` constructs the full-vector
   generalised eigen-problem (A, B) for the four tangential field
   components, applies shift-invert and divides the eigenvalues by k₀ to
   obtain the complex propagation constants γ/k₀.【F:Periodic_3D/Periodic_Mode_Solver_3D.py†L125-L178】
4. **Inspect modal fields** – `store_fields()` reshapes the eigenvectors
   into volumetric Ex/Ey/Hx/Hy arrays that can be sliced with
   `plot_field_plane()` for visual analysis.【F:Periodic_3D/Periodic_Mode_Solver_3D.py†L180-L216】

---

## 📊 Photonic band diagrams (`Band_Diagram_Solver`)

[`BandDiagramSolver2D`](Band_Diagram_Solver/2D_Band_Diagram.py) is a
fully fledged class replacing the previous script-style implementation.
The workflow mirrors the other solvers:

1. **Instantiate the solver** with the lattice constant and Yee grid size.
   The constructor creates a 2×-refined helper grid that matches Rumpf's
   subpixel averaging strategy.【F:Band_Diagram_Solver/2D_Band_Diagram.py†L38-L88】
2. **Add geometry** using `add_object()` or convenience helpers such as
   `add_circular_inclusion()`; masks can be arrays or callables of the
   helper grid coordinates.【F:Band_Diagram_Solver/2D_Band_Diagram.py†L90-L134】
3. **Define the Bloch path** with `default_high_symmetry_path()` (Γ–X–M–Γ
   for square lattices) and `generate_bloch_path()`, then optionally set
   tick labels for the symmetry points.【F:Band_Diagram_Solver/2D_Band_Diagram.py†L142-L204】
4. **Compute the bands** using `compute_band_structure()`, which extracts
   the Yee-grid material tensors, builds derivative operators for each
   Bloch vector and solves the TE/TM sparse eigen-problems.  Eigenvalues
   are sorted and normalised to `a/λ`.  Results are returned as a
   `BandStructureResult` dataclass for easy post-processing.【F:Band_Diagram_Solver/2D_Band_Diagram.py†L206-L293】
5. **Plot the diagram** with `plot_band_diagram()` to reproduce the
   familiar unit-cell plus band plot layout.  The helper accepts an
   optional illustration image for the bottom-left panel.【F:Band_Diagram_Solver/2D_Band_Diagram.py†L311-L358】


A runnable example script,
[`example_square_lattice.py`](Band_Diagram_Solver/example_square_lattice.py),
mirrors the dielectric-rod unit-cell calculation and demonstrates how to
instantiate the solver, sweep the Γ–X–M–Γ path and plot the TE/TM bands.【F:Band_Diagram_Solver/example_square_lattice.py†L1-L41】

---

### Reference

R. Rumpf, *Electromagnetic and Photonic Simulation for the Beginner:
Finite-Difference Frequency-Domain in MATLAB*.  Artech House, 2022.
