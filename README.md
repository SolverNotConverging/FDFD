# FDFD_CEM

This repository contains a beginner-level **Finite-Difference Frequency-Domain (FDFD)** based solver suite for computational electromagnetics. The solvers are organized by application area, such as mode solving, band diagram calculation, scattering, and electrostatics.

## 📁 Repository Structure

- `Band_Diagram_Solver`: 2D photonic crystal band diagram solver.
- `Electrostatic_Solver`: Solves 1D and 2D electrostatic problems using FDFD.
- `Mode_Solver_1D`: FDFD solver for guided mode analysis in 1D structures.
- `Mode_Solver_2D`: FDFD mode solver for 2D waveguide configurations.
- `Mode_Solver_Periodic`: FDFD periodic mode solver for photonic crystals and periodic waveguides.
- `Scattering`: 2D scattering problem solver for dielectric objects.
- `Mesh_points_calculation.py`: Utility to compute mesh grid points.
- `PML_sigma_calculation.py`: Utility to compute PML conductivity profiles.

---

## 🚀 Getting Started

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/FDFD_CEM.git
   cd FDFD_CEM
   ```

2. Install required dependencies (NumPy and Matplotlib):
   ```bash
   pip install numpy matplotlib
   ```

---

## 🧠 How to Use Each Solver

### 1. **Band Diagram Solver**
**Path**: `Band_Diagram_Solver/2D_Band_Diagram.py`  
Use this to compute band structures for 2D periodic media.

```bash
python Band_Diagram_Solver/2D_Band_Diagram.py
```

Make sure the lattice and material configurations are set in the script.

---

### 2. **Electrostatic Solver**
**Path**: `Electrostatic_Solver/`  
Use `1D_Example.py` or `2D_example.py` to solve electrostatic potential distributions.

```bash
python Electrostatic_Solver/1D_Example.py
python Electrostatic_Solver/2D_example.py
```

Modify boundary conditions and permittivity distributions as needed.

---

### 3. **1D Mode Solver**
**Path**: `Mode_Solver_1D/FDFD_1D_Mode_Solver.py`  
This solver computes guided modes in 1D waveguides.

```bash
python Mode_Solver_1D/1D_Dispersion.py
python Mode_Solver_1D/1D_Field_Visualization.py
```

---

### 4. **2D Mode Solver**
**Path**: `Mode_Solver_2D/FDFD_Mode_Solver.py`  
Run `2D_Dispersion.py` and `2D_Field_Visualization.py` to compute and visualize modes.

```bash
python Mode_Solver_2D/2D_Dispersion.py
python Mode_Solver_2D/2D_Field_Visualization.py
```

---

### 5. **Periodic Mode Solver**
**Path**: `Mode_Solver_Periodic/Periodic_Mode_Solver.py`  
For periodic boundary condition problems and frequency sweeps:

```bash
python Mode_Solver_Periodic/2D_Periodic_Freq_Sweep.py
python Mode_Solver_Periodic/2D_Periodic_Field_Visualization.py
```

---

### 6. **Scattering Solver**
**Path**: `Scattering/Scattering_Solver_2D.py`  
Use for computing field distributions from incident waves on scatterers.

```bash
python Scattering/Scattering_Example.py
```

---

## 📌 Notes
- The derivative operators are implemented using `yee_derivative.py`.
- Make sure your Python version is 3.6+.
- Figures and visualizations are saved using `matplotlib`.
