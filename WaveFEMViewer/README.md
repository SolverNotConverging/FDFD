# WaveFEM Viewer

WaveFEM Viewer is a standalone desktop application for inspecting HDF5 files
written by the WaveFEM electromagnetic solver.  Its source code and package
metadata live entirely in this `WaveFEMViewer` directory.  It does **not**
import or require the `wavefem` Python package, so it can be installed on a
visualization workstation without installing the FEM solver stack.

The viewer supports:

- indexed modal S-parameters over a frequency sweep;
- sampled modal electric-field and magnetic-field components;
- incident, scattered, and total 2D electric and magnetic vector fields;
- the saved dielectric material mesh and saved boundary/port/PML overlays.

## Requirements

- Python 3.10 or newer;
- NumPy, h5py, and Matplotlib (installed automatically by pip);
- Tk/Tkinter, supplied by the Python or Conda installation rather than pip.

The project is designed to be used in the repository's
`RF_Engineering_env` Conda environment.  On Windows, open Anaconda Prompt or
PowerShell and check that Tk is available:

```powershell
conda activate RF_Engineering_env
python -c "import tkinter; print(tkinter.TkVersion)"
```

If that import fails, install Tk in the same environment:

```powershell
conda install -n RF_Engineering_env tk
```

## Install

From the repository root, install an editable development copy:

```powershell
conda activate RF_Engineering_env
python -m pip install -e .\WaveFEMViewer
```

Editable installation is convenient while working on the viewer because
source changes take effect without reinstalling.  For a normal, non-editable
installation use:

```powershell
conda activate RF_Engineering_env
python -m pip install .\WaveFEMViewer
```

To install the test dependency as well:

```powershell
python -m pip install -e ".\WaveFEMViewer[test]"
```

Confirm the command-line entry point without opening a desktop window:

```powershell
wavefem-viewer --help
python -m wavefem_viewer --help
```

## Use

Launch the empty viewer and choose **Open HDF5…**:

```powershell
conda activate RF_Engineering_env
wavefem-viewer
```

Or open a result immediately:

```powershell
wavefem-viewer C:\path\to\simulation_results.h5
```

The equivalent module command is:

```powershell
python -m wavefem_viewer C:\path\to\simulation_results.h5
```

The frequency selector at the top chooses the current result in a sweep.  It
updates all five views:

1. **S-parameters** shows a numeric table for the selected frequency and a
   sweep plot.  The plot selector switches between dB magnitude, linear
   magnitude, phase in degrees, real part, and imaginary part.
2. **Modal E** selects a saved port mode, Cartesian component or norm, and
   absolute/real/imaginary representation of the modal electric field.
3. **Modal H** provides the same controls for the modal magnetic field.
4. **2D Vector E** selects total, incident, or scattered electric field and
   its real or imaginary vector part.
5. **2D Vector H** provides the same controls for the magnetic field.

Each plot includes Matplotlib's navigation toolbar for zooming, panning,
resetting the view, and saving an image.

## Plot coordinates and scene colors

WaveFEM stores physical coordinates in `(x, z)` order.  In every 2D vector
plot the propagation coordinate **z is the horizontal axis** and the
transverse coordinate **x is the vertical axis**.  Therefore the horizontal
quiver component is `E_z` or `H_z`, while the vertical component is `E_x` or
`H_x`.

When a result includes its optional `scene` data, the viewer draws:

- dielectric triangles with a grey-only map of `Re(eps_r)`;
- PEC boundaries as solid yellow lines;
- PMC boundaries as solid blue lines;
- wave-port planes as solid red lines;
- PML interfaces as dashed green lines.

The saved full-domain `z_span` and `x_span` set the vector-plot limits.  An
older schema-v1 file without a `scene` group is still supported; only the
field arrows are shown and Matplotlib determines the limits from those
samples.

## Supported HDF5 format

`load_h5()` accepts WaveFEM schema version 1 files with `single` or `sweep`
kind.  It independently validates frequencies, fields, S-parameters, powers,
modes, and metadata.  If present, every result's `scene` group is expected to
contain:

- `points`: real array `(2, N)` in `(x, z)` row order;
- `triangles`: integer array `(3, M)` indexing `points`;
- `eps_r`: real or complex array `(M,)`, one value per triangle;
- `x_span` and `z_span`: increasing real arrays `(2,)`;
- `lines/kind`: `pec`, `pmc`, `wave_port`, or `pml`;
- `lines/endpoints`: real array `(L, 2, 2)` in endpoint, then `(x, z)`, order;
- `lines/label`: one UTF-8 label per line.

Invalid or incomplete data produce an explanatory error dialog.  The reader
opens files read-only and copies all numerical arrays into non-writeable NumPy
arrays, so visualization cannot alter the saved simulation.

## Python API

The package exposes a small API for notebooks and custom applications without
requiring the Tk GUI:

```python
from wavefem_viewer import load_h5, plot_vector_field_2d

saved = load_h5("simulation_results.h5")
result = saved.results[0]
quiver = plot_vector_field_2d(
    axes,
    result.coordinates,
    result.E_total,
    field_name="E",
    quantity="real",
    scene=result.scene,
)
```

- `load_h5(path)` validates and returns a `FileData` record.  `kind` is
  `"single"` or `"sweep"`, `frequencies_hz` is the ordered ordinary-frequency
  array, and `results` contains one `ResultData` per frequency.
- `s_parameter_rows(result_or_mapping)` normalizes and sorts indexed modal
  S-parameters for a table.
- `plot_s_parameters(ax, frequencies_hz, results, quantity=..., keys=None)`
  plots a sweep and returns the created line artists.  Missing keys form NaN
  gaps rather than being silently interpreted as zero.
- `plot_modal_field(ax, x, field, field_name=..., component=...,
  quantity=...)` plots one modal component or vector norm and returns its line
  artist.
- `plot_scene(ax, scene)` draws the material and line overlays, applies the
  full-domain limits, and returns a `SceneArtists` record.
- `plot_vector_field_2d(ax, coordinates, field, field_name=...,
  quantity=..., max_arrows=900, scene=None)` averages duplicate FEM samples,
  subsamples dense fields, applies the z-horizontal/x-vertical convention,
  and returns the quiver artist.  `plot_vector_field` is an alias.
- `H5ViewerApp(root)` embeds the complete viewer in a caller-owned Tk root.
  `load_path(path, show_error=True)` loads a file programmatically and reports
  success with a boolean.

Importing `wavefem_viewer` does not start Tk or select a Matplotlib GUI
backend.  Tk and TkAgg are loaded only when `H5ViewerApp` is instantiated or
the command-line application is launched.

## Uninstall

Remove only the standalone viewer package from the active environment:

```powershell
conda activate RF_Engineering_env
python -m pip uninstall wavefem-viewer
```

Confirm the removal when pip asks.  Uninstalling the viewer does not uninstall
WaveFEM and does not delete any `.h5` result files.  Dependencies shared with
other packages (NumPy, h5py, and Matplotlib) are intentionally left installed.

If an editable installation still appears after moving this source directory,
run the same uninstall command again in the environment where it was
originally installed, then verify with:

```powershell
python -m pip show wavefem-viewer
```

No output from `pip show` means the package is no longer installed.

## Run the tests

From this directory:

```powershell
conda activate RF_Engineering_env
python -m pytest
```

The plotting tests use Matplotlib's non-interactive Agg backend and do not
open a window.
