# FDFD\_CEM

This repository contains a beginner-friendly suite of **Finite-Difference Frequency-Domain (FDFD)** solvers for computational electromagnetics. The solvers are modularly organized by application area—covering mode analysis, band diagram computation, scattering, and electrostatics.

## 📁 Repository Structure

* `Band_Diagram_Solver`: FDFD-based solver for computing band diagrams of 2D photonic crystals.
  *(Preliminary and not yet class-structured)*

* `Electrostatic_Solver`: Solves electrostatic field distributions in 1D and 2D.
  *(Note: strictly speaking, this is not FDFD-based)*

* `Mode_Solver_1D`: FDFD mode solver for 1D slab waveguides.
  *Supports anisotropic materials and impedance surfaces.*

* `Mode_Solver_2D`: FDFD mode solver for 2D waveguides (structures homogeneous along the propagation direction).
  *Supports anisotropic materials, impedance surfaces, and PML boundaries.*

* `Mode_Solver_Periodic`: FDFD solver for periodic waveguide structures, including leaky-wave antennas.
  *Supports anisotropic materials and PML.*

* `Scattering`: 2D FDFD TM/TE solver for electromagnetic scattering problems using the QAAQ formulation.
  *Supports anisotropic materials and PML.*

* `Mesh_points_calculation.py`: Utility for generating spatial mesh grid points for the simulation domain.

* `PML_sigma_calculation.py`: Utility for computing sigma (conductivity) profiles in perfectly matched layers (PML).

---

## 🧠 Modal Analysis Workflow

For modal analysis:

1. **Create the mesh** using your preferred geometry and resolution.
2. **Add objects** using `solver.add_object()`.
3. **Apply absorbing boundaries** using `solver.add_absorbing_boundary()` or `solver.add_UPML()`.
4. **Solve modes** using `solver.solve()`
5. **Visualize modal fields** along with their propagation constants (α and β) using: `solver.visualize_with_gui()`


### Reference
R. Rumpf, Electromagnetic and Photonic Simulation for the Beginner:
Finite-Difference Frequency-Domain in MATLAB. Artech House, 2022.


